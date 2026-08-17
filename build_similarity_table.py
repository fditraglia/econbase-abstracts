"""
Precompute the cosine-similarity table the web app serves from.

The app only ever ranks papers against a paper already in the corpus, so it never
needs the embeddings at request time -- only the answers. This turns a 184 MB file
of 4096-dimensional vectors into a ~3 MB table of each paper's closest neighbors,
which is what gets deployed.

The depth is not a judgement call. app.py asks for max(n_show * 2, 50) neighbors and
the interface caps n_show at 50, so 100 covers every request it can make and the
output is identical rather than approximate. If a future feature asks for more, or
filters candidates before ranking, this has to be rebuilt deeper.

    uv run build_similarity_table.py

R analogy: precomputing a distance matrix once and storing the top-k per row, rather
than recomputing dist() on every call.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import polars as pl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS = os.path.join(BASE_DIR, "econ_em_embeddings.parquet")
OUT = os.path.join(BASE_DIR, "similarity_top100.parquet")
MANIFEST = os.path.join(BASE_DIR, "embeddings_manifest.json")
TOP_K = 100
BLOCK = 512          # rows of the similarity matrix held at once


def main() -> None:
    df = pl.read_parquet(EMBEDDINGS, columns=["arxiv_id", "embedding"])
    ids = df["arxiv_id"].to_list()
    X = np.array(df["embedding"].to_list(), dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    n = len(ids)
    print(f"{n} papers x {X.shape[1]} dims; taking the top {TOP_K} of each")

    neighbor_ids: list[list[str]] = []
    cosines: list[list[float]] = []
    for start in range(0, n, BLOCK):
        S = X[start:start + BLOCK] @ X.T
        for r in range(S.shape[0]):
            S[r, start + r] = -1.0          # a paper is not its own neighbor
            top = np.argpartition(-S[r], TOP_K)[:TOP_K]
            top = top[np.argsort(-S[r][top])]
            neighbor_ids.append([ids[t] for t in top])
            cosines.append([float(S[r][t]) for t in top])
        print(f"  {min(start + BLOCK, n)}/{n}", flush=True)

    table = pl.DataFrame({
        "arxiv_id": ids,
        "neighbor_ids": neighbor_ids,
        "cosines": cosines,
    }).with_columns(pl.col("cosines").cast(pl.List(pl.Float32)))

    tmp = OUT + ".tmp"
    table.write_parquet(tmp, compression="zstd")
    os.replace(tmp, OUT)
    size = os.path.getsize(OUT) / 1e6
    print(f"wrote {OUT} ({size:.1f} MB)")

    # Record what this was built from, so a stale table is detectable.
    if os.path.exists(MANIFEST):
        m = json.load(open(MANIFEST))
        m["similarity_table"] = {
            "rows": n,
            "top_k": TOP_K,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "from_n_papers": m.get("n_papers"),
            "model_digest": m.get("model_digest"),
        }
        json.dump(m, open(MANIFEST, "w"), indent=2)
        print("recorded the build in embeddings_manifest.json")


if __name__ == "__main__":
    main()
