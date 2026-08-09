# /// script
# dependencies = ["polars", "numpy"]
# ///
"""Export tidy CSVs the Quarto report reads. Keeps the R side free of .npy."""
from pathlib import Path
import numpy as np, polars as pl

HERE = Path(__file__).resolve().parent
pool = pl.read_parquet(HERE / "pool.parquet")["base_id"].to_list()
cos = np.load(HERE / "cos.npy"); jac = np.load(HERE / "jac.npy")
intro = pl.read_parquet(HERE / "intro_embeddings.parquet")
have = {r["base_id"]: r["intro_embedding"] for r in intro.iter_rows(named=True)}
keep = [i for i, p in enumerate(pool) if p in have]
A = cos[np.ix_(keep, keep)]; J = jac[np.ix_(keep, keep)]
Y = np.array([have[pool[i]] for i in keep], dtype=np.float32)
Y /= np.linalg.norm(Y, axis=1, keepdims=True)
B = Y @ Y.T
for M in (A, B, J):
    np.fill_diagonal(M, np.nan)
n = len(keep)

iu = np.triu_indices(n, 1)
pl.DataFrame({
    "abstract_cos": A[iu].astype(np.float32),
    "intro_cos": B[iu].astype(np.float32),
    "coupling": J[iu].astype(np.float32),
}).write_csv(HERE / "report_pairs.csv")

# per-paper top-k overlap with the coupling rater, both arms
rows = []
for k in (5, 10, 20):
    for i in range(n):
        if not np.nansum(J[i]) > 0:
            continue
        cj = set(np.argsort(-np.nan_to_num(J[i], nan=-9))[:k])
        for arm, M in (("abstract", A), ("intro", B)):
            ck = set(np.argsort(-np.nan_to_num(M[i], nan=-9))[:k])
            rows.append({"k": k, "arm": arm, "overlap": len(ck & cj) / k})
pl.DataFrame(rows).write_csv(HERE / "report_overlap.csv")

# how much the two arms agree with each other, by k
rows = []
for k in (1, 3, 5, 10, 20, 50):
    v = [len(set(np.argsort(-np.nan_to_num(A[i], nan=-9))[:k]) &
             set(np.argsort(-np.nan_to_num(B[i], nan=-9))[:k])) / k for i in range(n)]
    rows.append({"k": k, "share": float(np.mean(v))})
pl.DataFrame(rows).write_csv(HERE / "report_armagree.csv")
print("wrote report_pairs.csv, report_overlap.csv, report_armagree.csv")
