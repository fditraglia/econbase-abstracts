"""
Incremental refresh of econ_em_embeddings.parquet.

Safe to re-run at will. A run with nothing new costs one arXiv query and
writes nothing. A run with new or revised papers embeds only those and
appends them, rather than rebuilding the whole file.

    uv run refresh_embeddings.py --dry-run    # report what would change
    uv run refresh_embeddings.py              # do it

Requires Ollama running with the frozen model pulled:

    ollama cp qwen3-embedding:latest qwen3-embedding-econbase:v1

Why a frozen copy rather than `qwen3-embedding`: `latest` is a moving tag.
If it ever repoints, papers embedded afterwards land in a different vector
space from the existing ones, cosine similarities across the boundary stop
meaning anything, and nothing visibly fails. A local copy cannot be moved
by an upstream push, and the digest check below catches anything else.

R analogy: this is the equivalent of an incremental `rbind()` onto a saved
.rds, with a checksum file beside it recording what produced it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta, timezone

import polars as pl
from pylatexenc.latex2text import LatexNodes2Text

import build_similarity_table

# --- Configuration -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "econ_em_embeddings.parquet")
PAPERS_FILE = os.path.join(BASE_DIR, "econ_em_papers.parquet")
MANIFEST_FILE = os.path.join(BASE_DIR, "embeddings_manifest.json")
CACHE_FILE = os.path.join(BASE_DIR, ".refresh_cache.parquet")

CATEGORY = "econ.EM"
MODEL_NAME = "qwen3-embedding-econbase:v1"   # frozen copy; see the docstring
BATCH_SIZE = 32
OLLAMA_EMBED = "http://localhost:11434/api/embed"
OLLAMA_TAGS = "http://localhost:11434/api/tags"

# How far back to re-check on a run. arXiv is queried newest-updated-first and
# we stop once we are safely past what we already hold; the margin absorbs
# clock skew and papers whose metadata settles a little after posting.
WATERMARK_MARGIN = timedelta(days=2)
# Upper bound on how many results a single run will walk through, so a bad
# watermark cannot turn into an unbounded sweep of the archive.
MAX_SCAN = 4000

converter = LatexNodes2Text()


def clean_latex(text: str) -> str:
    """Convert LaTeX markup to plain Unicode, falling back to the raw string."""
    try:
        return converter.latex_to_text(text)
    except Exception:
        return text


def base_of(arxiv_id: str) -> str:
    """'2405.06779v3' -> '2405.06779'."""
    return re.sub(r"v\d+$", "", arxiv_id)


def version_of(arxiv_id: str) -> int:
    m = re.search(r"v(\d+)$", arxiv_id)
    return int(m.group(1)) if m else 0


def text_for(title: str, abstract: str) -> str:
    """Exactly the string that gets embedded. Keep in step with embed_abstracts.py."""
    return f"{title}\n{abstract}"


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# --- Model identity ----------------------------------------------------------
def model_digest(name: str) -> str:
    """The digest Ollama reports for a model, or '' if it does not have it."""
    with urllib.request.urlopen(OLLAMA_TAGS, timeout=30) as r:
        for m in json.loads(r.read())["models"]:
            if m["name"] == name:
                return m["digest"]
    return ""


def check_model(manifest: dict, allow_change: bool) -> str:
    digest = model_digest(MODEL_NAME)
    if not digest:
        raise SystemExit(
            f"Ollama does not have {MODEL_NAME}. Create the frozen copy with:\n"
            f"    ollama cp qwen3-embedding:latest {MODEL_NAME}"
        )
    recorded = manifest.get("model_digest")
    if recorded and recorded != digest and not allow_change:
        raise SystemExit(
            f"Embedding model changed.\n"
            f"  manifest: {recorded}\n"
            f"  ollama:   {digest}\n"
            "Vectors from two models are not comparable, so appending would silently\n"
            "corrupt the file. Re-embed the whole corpus, or pass --allow-model-change\n"
            "if you know the digest moved for a benign reason."
        )
    return digest


# --- Existing state ----------------------------------------------------------
def load_existing() -> pl.DataFrame:
    """Read the embeddings file, adding the bookkeeping columns if absent."""
    if not os.path.exists(EMBEDDINGS_FILE):
        return pl.DataFrame()
    df = pl.read_parquet(EMBEDDINGS_FILE)
    if "base_id" not in df.columns:
        df = df.with_columns(
            pl.col("arxiv_id").str.replace(r"v\d+$", "").alias("base_id"),
            pl.col("arxiv_id").str.extract(r"v(\d+)$", 1)
              .cast(pl.Int32).fill_null(0).alias("version"),
        )
    if "text_sha" not in df.columns:
        df = df.with_columns(
            (pl.col("title") + "\n" + pl.col("abstract"))
            .map_elements(sha, return_dtype=pl.String).alias("text_sha")
        )
    return df


def load_manifest() -> dict:
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE) as f:
            return json.load(f)
    return {}


# --- arXiv ------------------------------------------------------------------
def fetch_since(watermark: datetime, limit: int | None) -> list:
    """Papers whose *last update* is at or after the watermark, newest first.

    Sorting by last-updated rather than submission date is what makes revisions
    visible: a paper revised today keeps its original submission date, so a
    submission-date query would never return it again.
    """
    import arxiv

    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
    search = arxiv.Search(
        query=f"cat:{CATEGORY}",
        max_results=MAX_SCAN,
        sort_by=arxiv.SortCriterion.LastUpdatedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    out = []
    for res in client.results(search):
        if res.updated < watermark:
            break
        out.append(res)
        if limit and len(out) >= limit:
            break
        if len(out) >= MAX_SCAN:
            print(f"  stopped at the {MAX_SCAN}-result scan cap", flush=True)
            break
    return out


def row_from(res) -> dict:
    short = res.entry_id.split("/")[-1]
    title, abstract = clean_latex(res.title), clean_latex(res.summary)
    return {
        "arxiv_id": short,
        "base_id": base_of(short),
        "version": version_of(short),
        "authors": ", ".join(a.name for a in res.authors),
        "title_raw": res.title,
        "abstract_raw": res.summary,
        "title": title,
        "abstract": abstract,
        "primary_category": res.primary_category,
        "categories": ", ".join(res.categories),
        "is_primary_econ_em": res.primary_category == CATEGORY,
        "published": res.published.date(),
        "url": res.entry_id,
        "text_sha": sha(text_for(title, abstract)),
    }


# --- Embedding ---------------------------------------------------------------
def embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    t0 = time.time()
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        payload = json.dumps({"model": MODEL_NAME, "input": batch}).encode()
        req = urllib.request.Request(
            OLLAMA_EMBED, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            vectors.extend(json.loads(resp.read())["embeddings"])
        done = i + len(batch)
        rate = done / max(time.time() - t0, 1e-9)
        print(f"  embedded {done}/{len(texts)} ({rate:.1f}/s)", flush=True)
    return vectors


# --- Publishing --------------------------------------------------------------
# The vectors are not in git: the app reads the small similarity table instead, and
# versioning a 184 MB file on every refresh would exhaust the Git LFS allowance in
# about four runs. The bucket keeps one current copy, with a pointer file beside it
# so a consumer can tell what it has without downloading the whole thing.
BUCKET = "gs://econbase-embeddings"


def gcloud_path() -> str | None:
    found = shutil.which("gcloud")
    if found:
        return found
    standalone = Path.home() / "google-cloud-sdk" / "bin" / "gcloud"
    return str(standalone) if standalone.exists() else None


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def publish() -> None:
    """Upload the vectors and a pointer describing them.

    Exits non-zero if this fails. The local files are already written, so re-running
    is cheap, and a refresh that quietly stopped short of publishing would leave the
    bucket describing a corpus nobody else can see.
    """
    gcloud = gcloud_path()
    if gcloud is None:
        raise SystemExit("gcloud not found; install it or pass --no-publish")

    manifest = json.load(open(MANIFEST_FILE))
    manifest.update({
        "object": os.path.basename(EMBEDDINGS_FILE),
        "sha256": sha256_of(EMBEDDINGS_FILE),
        "bytes": os.path.getsize(EMBEDDINGS_FILE),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    })
    pointer = os.path.join(BASE_DIR, ".current.json")
    with open(pointer, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\npublishing to {BUCKET} ...", flush=True)
    for src, dest in ((EMBEDDINGS_FILE, f"{BUCKET}/{os.path.basename(EMBEDDINGS_FILE)}"),
                      (pointer, f"{BUCKET}/current.json")):
        r = subprocess.run([gcloud, "storage", "cp", src, dest],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"upload of {os.path.basename(src)} failed:\n{r.stderr}")
    os.remove(pointer)
    print(f"published {manifest['n_papers']} papers, "
          f"sha256 {manifest['sha256'][:16]}", flush=True)


# --- Atomic write ------------------------------------------------------------
def write_atomic(df: pl.DataFrame, path: str) -> None:
    """Write beside the target, then rename. A crash leaves the old file intact."""
    tmp = path + ".tmp"
    df.write_parquet(tmp)
    os.replace(tmp, path)


# --- Embedding cache ---------------------------------------------------------
# Keyed on the hash of the embedded text, so it stays valid across runs and is
# invalidated automatically when an abstract changes. Tagged with the model
# digest, since vectors from a different model must never be reused.
def load_cache(digest: str) -> pl.DataFrame | None:
    if not os.path.exists(CACHE_FILE):
        return None
    df = pl.read_parquet(CACHE_FILE)
    if df.height and df["model_digest"][0] != digest:
        print("  cache was built with a different model; ignoring it", flush=True)
        return None
    return df


def save_cache(df: pl.DataFrame, digest: str) -> None:
    keep = df.select(["text_sha", "embedding"]).with_columns(
        pl.lit(digest).alias("model_digest"))
    old = load_cache(digest)
    if old is not None and old.height:
        keep = pl.concat([old.select(keep.columns), keep],
                         how="vertical").unique(subset=["text_sha"], keep="last")
    write_atomic(keep, CACHE_FILE)
    print(f"  cached {keep.height} embeddings", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change and exit")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many arXiv results (for testing)")
    ap.add_argument("--no-table", action="store_true",
                    help="skip rebuilding similarity_top100.parquet")
    ap.add_argument("--no-publish", action="store_true",
                    help="skip uploading the vectors to the bucket")
    ap.add_argument("--allow-model-change", action="store_true",
                    help="proceed even if the model digest differs from the manifest")
    args = ap.parse_args()

    manifest = load_manifest()
    digest = check_model(manifest, args.allow_model_change)
    print(f"model {MODEL_NAME} @ {digest[:16]}", flush=True)

    existing = load_existing()
    if existing.height:
        by_base = dict(zip(existing["base_id"].to_list(),
                           zip(existing["version"].to_list(),
                               existing["text_sha"].to_list())))
        print(f"holding {existing.height} papers", flush=True)
    else:
        by_base = {}
        print("no existing embeddings; this will be a full build", flush=True)

    # Where to resume from. The manifest's watermark is authoritative; on a first
    # run fall back to the newest publication date already held.
    if manifest.get("max_updated_seen"):
        watermark = datetime.fromisoformat(manifest["max_updated_seen"])
    elif existing.height:
        watermark = datetime.combine(existing["published"].max(),
                                     datetime.min.time(), tzinfo=timezone.utc)
    else:
        watermark = datetime(1991, 8, 14, tzinfo=timezone.utc)
    scan_from = watermark          # what this run claims to have covered up to
    watermark -= WATERMARK_MARGIN  # re-check a little before it, for clock skew
    print(f"scanning arXiv for anything updated since {watermark.date()}", flush=True)

    results = fetch_since(watermark, args.limit)
    print(f"  {len(results)} results returned", flush=True)
    if not results:
        print("nothing to do.")
        return

    rows, new_ids, revised_ids = [], [], []
    for res in results:
        row = row_from(res)
        prev = by_base.get(row["base_id"])
        if prev is None:
            rows.append(row)
            new_ids.append(row["base_id"])
        else:
            prev_version, prev_sha = prev
            # Re-embed only when the paper was actually revised AND the text moved.
            # A version bump that leaves title and abstract alone changes nothing
            # we embed, so it is not worth the compute.
            if row["version"] > prev_version and row["text_sha"] != prev_sha:
                rows.append(row)
                revised_ids.append(row["base_id"])

    max_updated = max(r.updated for r in results)
    print(f"\nnew papers:     {len(new_ids)}")
    print(f"revised papers: {len(revised_ids)}")
    if revised_ids[:5]:
        print("  e.g. " + ", ".join(revised_ids[:5]))

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return
    if not rows:
        # Still advance the watermark: these results were checked and needed no work.
        manifest.update({"max_updated_seen": max_updated.isoformat(),
                         "checked_at": datetime.now(timezone.utc).isoformat()})
        with open(MANIFEST_FILE, "w") as f:
            json.dump(manifest, f, indent=2)
        print("\nnothing to embed; watermark advanced.")
        return

    fresh = pl.DataFrame(rows)

    # Embedding is the only slow step, so its output is cached before the merge.
    # An interrupted or failed merge then costs seconds to retry rather than
    # re-running the model over everything.
    # An empty embedding column has to be typed. An untyped null column cannot be
    # concatenated with real vectors later, which is exactly how this failed twice.
    EMPTY_VEC = pl.lit(None, dtype=pl.List(pl.Float64))
    cached = load_cache(digest)
    if cached is not None and cached.height:
        reuse = cached.join(fresh.select("text_sha"), on="text_sha", how="semi")
        if reuse.height:
            print(f"reusing {reuse.height} embeddings from the cache", flush=True)
        fresh = fresh.join(reuse.select(["text_sha", "embedding"]),
                           on="text_sha", how="left")
    else:
        fresh = fresh.with_columns(EMPTY_VEC.alias("embedding"))

    todo = fresh.filter(pl.col("embedding").is_null())
    if todo.height:
        print(f"\nembedding {todo.height} papers...", flush=True)
        vectors = embed([text_for(t, a) for t, a in
                         zip(todo["title"].to_list(), todo["abstract"].to_list())])
        done = todo.drop("embedding").with_columns(pl.Series("embedding", vectors))
        # Cache before merging, not after: the merge is the step that has failed,
        # and the embedding is the step that is expensive.
        save_cache(done, digest)
        reused = fresh.filter(pl.col("embedding").is_not_null())
        fresh = (pl.concat([reused.select(done.columns).cast(done.schema), done],
                           how="vertical") if reused.height else done)

    if existing.height:
        # Drop the superseded rows, then stack. Align dtypes as well as column
        # order: a Python int arrives as Int64 while the stored column is Int32,
        # and polars refuses to concatenate across that.
        kept = existing.join(fresh.select("base_id"), on="base_id", how="anti")
        combined = pl.concat(
            [kept, fresh.select(kept.columns).cast(kept.schema)], how="vertical")
    else:
        combined = fresh
    combined = combined.sort("published", descending=True)

    write_atomic(combined, EMBEDDINGS_FILE)
    write_atomic(combined.drop("embedding"), PAPERS_FILE)
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)   # the work is safely in the parquet now

    # A --limit run has seen only the newest slice of what changed, so it must not
    # claim to have checked everything up to the newest result. Write back the point
    # the scan STARTED from instead. Writing null here is not safe either: the
    # fallback is derived from the data, which this run has just moved forward, so a
    # later run would skip everything in between. That happened once; hence the note.
    watermark_out = scan_from.isoformat() if args.limit else max_updated.isoformat()
    if args.limit:
        print(f"  --limit: watermark held at {scan_from.date()}", flush=True)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_papers": combined.height,
        "max_published": str(combined["published"].max()),
        "max_updated_seen": watermark_out,
        "model": MODEL_NAME,
        "model_digest": digest,
        # Read off the data rather than the batch just embedded, which is empty when
        # everything came from the cache.
        "embedding_dim": len(combined["embedding"][0]),
        "last_run_new": len(new_ids),
        "last_run_revised": len(revised_ids),
    }
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nwrote {combined.height} papers to {os.path.basename(EMBEDDINGS_FILE)}")
    print(f"newest publication date: {manifest['max_published']}")

    # New vectors mean the served table is out of date. Rebuilding here rather than
    # leaving it to be remembered: a refresh without it leaves the app recommending
    # from the old corpus with nothing to show that it is doing so.
    if not args.no_table:
        print("\nrebuilding the similarity table...", flush=True)
        build_similarity_table.main()

    if not args.no_publish:
        publish()


if __name__ == "__main__":
    main()
