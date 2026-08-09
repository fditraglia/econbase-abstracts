# /// script
# dependencies = ["polars", "numpy"]
# ///
"""Embed the introductions of papers that also have an abstract embedding.

Writes experiments/intro_embeddings.parquet. Resumable: re-running skips
papers already embedded, so the job can be interrupted freely.
"""
import json, time, urllib.request
from pathlib import Path
import polars as pl

HERE = Path(__file__).resolve().parent
OUT = HERE / "intro_embeddings.parquet"
URL = "http://localhost:11434/api/embed"
MODEL = "qwen3-embedding"


def embed(text: str) -> list[float]:
    req = urllib.request.Request(
        URL,
        data=json.dumps({"model": MODEL, "input": [text]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["embeddings"][0]


emb = pl.read_parquet(HERE.parent / "econ_em_embeddings.parquet",
                      columns=["arxiv_id", "title", "published"])
emb = emb.with_columns(pl.col("arxiv_id").str.replace(r"v\d+$", "").alias("base_id"))

intr = pl.read_parquet("/Users/francisditraglia/pangram-arxiv/data/intros.parquet")
intr = intr.filter((pl.col("method") == "intro-section")
                   & (pl.col("words_intro") > 300)
                   & (pl.col("words_intro") < 4000))

pool = emb.join(intr.select(["base_id", "intro_text", "words_intro"]), on="base_id", how="inner")

done = set()
if OUT.exists():
    done = set(pl.read_parquet(OUT, columns=["base_id"])["base_id"].to_list())
todo = pool.filter(~pl.col("base_id").is_in(list(done))) if done else pool
print(f"pool={pool.height} done={len(done)} todo={todo.height}", flush=True)

rows, t0 = [], time.time()
for i, r in enumerate(todo.iter_rows(named=True), 1):
    # Title carries strong topical signal, as in the abstract arm.
    rows.append({"base_id": r["base_id"],
                 "intro_embedding": embed(f"{r['title']}\n{r['intro_text']}")})
    if i % 25 == 0 or i == todo.height:
        el = time.time() - t0
        print(f"{i}/{todo.height}  {el/i:.2f}s/paper  eta {(todo.height-i)*el/i/60:.1f} min", flush=True)
        new = pl.DataFrame(rows)
        if OUT.exists():
            new = pl.concat([pl.read_parquet(OUT), new])
        new.write_parquet(OUT)
        rows = []
print("done ->", OUT, flush=True)
