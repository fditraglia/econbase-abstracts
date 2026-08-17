# /// script
# dependencies = ["polars", "numpy", "scipy"]
# ///
"""Embedding similarity against Martin's citation measure.

Two ways of finding related work, compared on the same papers and the same people:

  * cosine similarity between embedded title+abstract text (this repository)
  * accumulated citation weight from citations.db (econ-corpus)

Their ranked lists come from the stored related_papers / related_papers_by_author
tables, which are what the site actually serves. Scoring an *arbitrary* pair, which
the stored tables cannot do because they keep only the top 10, uses a rebuild of the
measure from its primitives: references_ joined to intensity, exactly as
econ-corpus/citations/related.py defines it.

reproduce_check() reports how closely that rebuild matches the stored tables. At the
paper level it agrees on 96% of lists, the remainder being ties. At the author level
it agrees on ~70%, because the aggregation needs a paper-to-author map we do not hold
locally: 357 of the 3,962 papers touching a citation edge have no row in the serve
database's author_papers, which covers only authors with profile pages. That is why
their lists are read rather than rebuilt.

Usage:
    uv run compare_citation_embedding.py                # everything
    uv run compare_citation_embedding.py --limit 200    # quick pass while developing
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
EMB = HERE.parent / "econ_em_embeddings.parquet"
CIT = Path.home() / "corpus" / "arxiv" / "citations" / "citations.db"
SERVE = sorted(glob.glob(str(Path.home() / "corpus/arxiv/serve/corpus-*.db")))
OUT_JSON = HERE / "comparison_results.json"
SEED = 20260817


def base_of(x: str) -> str:
    return re.sub(r"v\d+$", "", x)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_embeddings():
    df = pl.read_parquet(EMB, columns=["base_id", "title", "authors", "published", "embedding"])
    ids = df["base_id"].to_list()
    assert all(i == base_of(i) for i in ids), "identifiers must be version-stripped"
    assert len(set(ids)) == len(ids), "duplicate papers in the embedding file"
    X = np.array(df["embedding"].to_list(), dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    years = np.array([d.year for d in df["published"].to_list()], dtype=np.int32)
    print(f"embeddings: {X.shape[0]} papers x {X.shape[1]} dims, "
          f"norms in [{np.linalg.norm(X, axis=1).min():.4f}, {np.linalg.norm(X, axis=1).max():.4f}]")
    return ids, {p: i for i, p in enumerate(ids)}, X, years, df


def load_citation_edges(db: sqlite3.Connection) -> dict[tuple[str, str], float]:
    """Directed citing -> cited edges, weighted by summed composite_index.

    This is the definition in econ-corpus/citations/related.py: every reference that
    resolves to a corpus paper, weighted by that reference's composite intensity, summed
    when one paper cites another under more than one bibliography key.
    """
    rows = db.execute("""
        SELECT r.arxiv_id, r.in_corpus_match, i.composite_index
        FROM references_ r
        JOIN intensity i ON i.arxiv_id = r.arxiv_id AND i.cite_key = r.cite_key
        WHERE r.in_corpus_match IS NOT NULL
    """).fetchall()
    edges: dict[tuple[str, str], float] = defaultdict(float)
    for citing, cited, w in rows:
        if citing != cited:
            edges[(citing, cited)] += (w or 0.0)
    print(f"citation edges: {len(edges)} directed pairs from {len(rows)} resolved references")
    return dict(edges)


def load_author_papers(serve_db: sqlite3.Connection) -> dict[str, set[str]]:
    own = defaultdict(set)
    for a, p in serve_db.execute("SELECT author_id, arxiv_id FROM author_papers"):
        own[a].add(p)
    print(f"author -> own papers: {len(own)} authors, "
          f"{sum(len(v) for v in own.values())} rows")
    return own


# ---------------------------------------------------------------------------
# Rebuild check: do we reproduce Martin's stored lists?
# ---------------------------------------------------------------------------
def top_n(d: dict[str, float], n: int) -> list[str]:
    return [k for k, _ in sorted(d.items(), key=lambda kv: -kv[1])[:n]]


def reproduce_check(db, edges, own):
    """Rebuild the stored related_papers / related_papers_by_author tables and compare.

    A mismatch means our reading of the measure is wrong, and every number below would
    be measuring something other than what the site serves.
    """
    out_lists, in_lists = defaultdict(dict), defaultdict(dict)
    for (a, b), w in edges.items():
        out_lists[a][b] = w
        in_lists[b][a] = w

    stored = defaultdict(dict)
    for a, d, r, w in db.execute(
            "SELECT arxiv_id, direction, related_arxiv_id, rank FROM related_papers"):
        stored[(a, d)][r] = w
    agree = total = 0
    for (a, d), items in stored.items():
        mine = top_n(out_lists[a] if d == "out" else in_lists[a], len(items))
        total += 1
        agree += set(mine) == set(items.keys())
    print(f"paper-level rebuild reproduces {agree}/{total} stored lists "
          f"({100*agree/total:.1f}%)")

    # author level
    a_out, a_in = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(float))
    paper_authors = defaultdict(list)
    for aid, papers in own.items():
        for p in papers:
            paper_authors[p].append(aid)
    for (citing, cited), w in edges.items():
        for aid in paper_authors.get(citing, ()):
            if cited not in own[aid]:
                a_out[aid][cited] += w
        for aid in paper_authors.get(cited, ()):
            if citing not in own[aid]:
                a_in[aid][citing] += w

    stored_a = defaultdict(dict)
    for a, d, r in db.execute(
            "SELECT author_id, direction, related_arxiv_id FROM related_papers_by_author"):
        stored_a[(a, d)][r] = 1
    agree = total = 0
    for (a, d), items in stored_a.items():
        mine = top_n(a_out[a] if d == "out" else a_in[a], len(items))
        total += 1
        agree += set(mine) == set(items.keys())
    print(f"author-level rebuild reproduces {agree}/{total} stored lists "
          f"({100*agree/total:.1f}%)")
    return a_out, a_in, out_lists, in_lists


# ---------------------------------------------------------------------------
# The comparison itself
# ---------------------------------------------------------------------------
def compare_lists(target_scores: dict[str, float], sims: np.ndarray, idx_of, ids,
                  years, self_idx: int | None, exclude: set[str], rng, w_lookup):
    """One unit of comparison: their ranked list against ours, at matched depth.

    target_scores maps candidate paper -> citation weight (their measure).
    sims is our cosine score against every paper. Returns a dict of statistics, or
    None if the unit is unusable.
    """
    theirs = [p for p in top_n(target_scores, len(target_scores)) if p in idx_of]
    n = len(theirs)
    if n == 0:
        return None

    mask = np.ones(len(ids), dtype=bool)
    if self_idx is not None:
        mask[self_idx] = False
    for p in exclude:
        j = idx_of.get(p)
        if j is not None:
            mask[j] = False

    order = np.argsort(-np.where(mask, sims, -np.inf))
    ours = [ids[j] for j in order[:n]]

    their_i = [idx_of[p] for p in theirs]
    our_i = [idx_of[p] for p in ours]

    # Baselines of the same size: uniform, and matched on publication year so the
    # citation measure is not rewarded merely for pointing at older papers.
    pool = np.flatnonzero(mask)
    unif_i = rng.choice(pool, size=n, replace=False)
    year_i = []
    for j in their_i:
        cand = pool[years[pool] == years[j]]
        year_i.append(int(rng.choice(cand)) if len(cand) else int(rng.choice(pool)))

    def mean_cos(ii):
        return float(np.mean(sims[ii])) if len(ii) else float("nan")

    def mean_w(paper_ids):
        return float(np.mean([w_lookup(p) for p in paper_ids])) if paper_ids else float("nan")

    return {
        "n": n,
        "overlap": len(set(theirs) & set(ours)) / n,
        "cos_theirs": mean_cos(their_i),
        "cos_ours": mean_cos(our_i),
        "cos_year_matched": mean_cos(year_i),
        "cos_uniform": mean_cos(unif_i),
        "w_theirs": mean_w(theirs),
        "w_ours": mean_w(ours),
        "w_uniform": mean_w([ids[j] for j in unif_i]),
    }


def summarize(rows: list[dict], label: str) -> dict:
    if not rows:
        return {"label": label, "units": 0}
    keys = [k for k in rows[0] if k != "n"]
    out = {"label": label, "units": len(rows),
           "median_depth": float(np.median([r["n"] for r in rows]))}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        vals = vals[~np.isnan(vals)]
        out[k] = float(vals.mean()) if len(vals) else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of units per part (development only)")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    if not CIT.exists():
        raise SystemExit(f"citations database not found at {CIT}\n"
                         "Pull it with: gcloud storage rsync -r "
                         "gs://econbase-arxiv-corpus/citations/ ~/corpus/arxiv/citations/")
    if not SERVE:
        raise SystemExit("serve database not found under ~/corpus/arxiv/serve/")

    ids, idx_of, X, years, df = load_embeddings()
    db = sqlite3.connect(CIT)
    serve = sqlite3.connect(SERVE[-1])
    edges = load_citation_edges(db)
    own = load_author_papers(serve)
    a_out, a_in, p_out, p_in = reproduce_check(db, edges, own)

    # symmetric citation weight, for scoring arbitrary pairs
    sym = defaultdict(float)
    for (a, b), w in edges.items():
        sym[(a, b)] += w
        sym[(b, a)] += w

    results = {"corpus": {"papers_embedded": len(ids),
                          "citation_edges": len(edges),
                          "seed": SEED}}

    # Their served lists, read rather than rebuilt.
    stored_paper = defaultdict(lambda: defaultdict(float))
    for a, d, r, w in db.execute(
            "SELECT arxiv_id, direction, related_arxiv_id, weight FROM related_papers"):
        stored_paper[(a, d)][r] += w
    stored_author = defaultdict(lambda: defaultdict(float))
    for a, d, r, w in db.execute("SELECT author_id, direction, related_arxiv_id, weight "
                                 "FROM related_papers_by_author"):
        stored_author[(a, d)][r] += w

    def pooled(store, key):
        merged = defaultdict(float)
        for d in ("out", "in"):
            for k, v in store.get((key, d), {}).items():
                merged[k] += v
        return {k: v for k, v in merged.items() if k in idx_of}

    # ---- Part 1a: paper level ------------------------------------------------
    paper_units = sorted({k for k, _ in stored_paper} & set(idx_of))
    if args.limit:
        paper_units = list(rng.permutation(paper_units)[:args.limit])
    print(f"\npaper level: {len(paper_units)} papers with a served citation list")

    by_direction = {d: [] for d in ("pooled", "out", "in")}
    for p in paper_units:
        i = idx_of[p]
        sims = X @ X[i]          # one product per paper, reused by all three directions
        for direction in by_direction:
            target = (pooled(stored_paper, p) if direction == "pooled"
                      else {k: v for k, v in stored_paper.get((p, direction), {}).items()
                            if k in idx_of})
            if not target:
                continue
            r = compare_lists(target, sims, idx_of, ids, years, i, set(), rng,
                              lambda q, p=p: sym.get((p, q), 0.0))
            if r:
                by_direction[direction].append(r)
    for direction, rows in by_direction.items():
        results[f"paper_{direction}"] = summarize(rows, f"paper level, {direction}")
        print("  " + json.dumps(results[f"paper_{direction}"]))
    # Tidy rows for the report: one line per paper, so the write-up can show the
    # distribution rather than only its mean.
    pl.DataFrame([dict(r, direction=d) for d, rows in by_direction.items() for r in rows]) \
      .write_csv(HERE / "comparison_paper_level.csv")

    # ---- Part 1b: author level ----------------------------------------------
    authors = sorted({k for k, _ in stored_author})
    authors = [a for a in authors if len(own[a] & set(idx_of)) >= 1]
    if args.limit:
        authors = list(rng.permutation(authors)[:args.limit])
    print(f"\nauthor level: {len(authors)} authors with a served list and >=1 embedded paper")

    # Only two rules are worth reporting. Summing cosines over an author's papers and
    # averaging them differ by a constant for a given author, so they rank candidates
    # identically; only the maximum is a genuinely different rule.
    by_rule = {(rule, m): [] for rule in ("max", "mean") for m in (1, 3)}
    for a in authors:
        mine_papers = sorted(own[a] & set(idx_of))
        target = pooled(stored_author, a)
        if not target:
            continue
        # One matrix product per author, reused by both aggregation rules.
        S = X[[idx_of[p] for p in mine_papers]] @ X.T
        for rule, sims in (("max", S.max(axis=0)), ("mean", S.mean(axis=0))):
            r = compare_lists(target, sims, idx_of, ids, years, None,
                              set(mine_papers), rng,
                              lambda q, a=a: a_out[a].get(q, 0.0) + a_in[a].get(q, 0.0))
            if not r:
                continue
            for m in (1, 3):
                if len(mine_papers) >= m:
                    by_rule[(rule, m)].append(r)

    for (rule, m), rows in by_rule.items():
        key = f"author_{rule}_min{m}"
        results[key] = summarize(rows, f"author level, {rule}, >={m} own papers")
        print("  " + json.dumps(results[key]))
    pl.DataFrame([dict(r, rule=rule, min_own=m)
                  for (rule, m), rows in by_rule.items() for r in rows]) \
      .write_csv(HERE / "comparison_author_level.csv")

    # ---- Part 1c: pair-level statistics -------------------------------------
    linked = [(a, b) for (a, b) in edges if a in idx_of and b in idx_of]
    cos_linked = np.array([float(X[idx_of[a]] @ X[idx_of[b]]) for a, b in linked])
    w_linked = np.array([edges[(a, b)] for a, b in linked])
    rho = spearmanr(cos_linked, w_linked)
    n_neg = min(200_000, len(ids) * 40)
    ai = rng.integers(0, len(ids), n_neg)
    bi = rng.integers(0, len(ids), n_neg)
    keep = ai != bi
    ai, bi = ai[keep], bi[keep]
    linked_set = set(linked)
    keep = np.array([(ids[i], ids[j]) not in linked_set for i, j in zip(ai, bi)])
    ai, bi = ai[keep], bi[keep]
    cos_unlinked = np.einsum("ij,ij->i", X[ai], X[bi])

    def auc_of(pos, neg):
        """Area under the ROC curve, via the rank-sum identity."""
        allv = np.concatenate([pos, neg])
        ranks = allv.argsort().argsort().astype(np.float64) + 1
        n1, n0 = len(pos), len(neg)
        return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    # A citation link can only run from a later paper to an earlier one, so linked pairs
    # are not a random draw in time. The matched negatives hold the pair of publication
    # years fixed, which asks whether cosine still separates them once age cannot.
    by_year = defaultdict(list)
    for j, y in enumerate(years):
        by_year[int(y)].append(j)
    matched = []
    for a, b in linked:
        ya, yb = int(years[idx_of[a]]), int(years[idx_of[b]])
        ca, cb = by_year.get(ya), by_year.get(yb)
        if not ca or not cb:
            continue
        for _ in range(5):
            i2, j2 = int(rng.choice(ca)), int(rng.choice(cb))
            if i2 != j2 and (ids[i2], ids[j2]) not in linked_set:
                matched.append((i2, j2))
                break
    mi = np.array([p[0] for p in matched])
    mj = np.array([p[1] for p in matched])
    cos_matched = np.einsum("ij,ij->i", X[mi], X[mj])

    results["pairs"] = {
        "linked_pairs": len(cos_linked),
        "unlinked_sampled": int(len(cos_unlinked)),
        "year_matched_sampled": int(len(cos_matched)),
        "spearman_cos_vs_weight_on_linked": float(rho.statistic),
        "spearman_p": float(rho.pvalue),
        "mean_cos_linked": float(cos_linked.mean()),
        "mean_cos_unlinked": float(cos_unlinked.mean()),
        "mean_cos_year_matched": float(cos_matched.mean()),
        "auc_cos_predicts_link": auc_of(cos_linked, cos_unlinked),
        "auc_cos_predicts_link_year_matched": auc_of(cos_linked, cos_matched),
    }
    print("\npairs: " + json.dumps(results["pairs"], indent=2))

    step = max(1, len(cos_linked) // 6000)
    pl.DataFrame({"cosine": np.concatenate([cos_linked[::step], cos_unlinked[::step * 13],
                                            cos_matched[::step]]),
                  "group": (["citation-linked"] * len(cos_linked[::step])
                            + ["unlinked"] * len(cos_unlinked[::step * 13])
                            + ["unlinked, year-matched"] * len(cos_matched[::step]))}) \
      .write_csv(HERE / "comparison_pairs.csv")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
