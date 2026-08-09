# /// script
# dependencies = ["polars", "numpy", "scipy"]
# ///
"""Compare the abstract and introduction representations.

Two questions. How much do the arms differ from each other? And does either
agree better with bibliographic coupling, the one independent measure available?

Writes results.json and prints the same numbers.
"""
import json
from pathlib import Path
import numpy as np, polars as pl
from scipy.stats import spearmanr

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
    np.fill_diagonal(M, -9)
n = len(keep)

iu = np.triu_indices(n, 1)
nz = J[iu] > 0
topk = lambda M, i, k: set(np.argsort(-M[i])[:k])
coupled = [i for i in range(n) if J[i].max() > 0]

R = {
    "papers": n,
    "pairs": int(len(iu[0])),
    "pairs_sharing_a_reference": int(nz.sum()),
    "arms_agree_top1": int(sum(np.argmax(A[i]) == np.argmax(B[i]) for i in range(n))),
    "mean_topk_overlap_between_arms": {
        str(k): round(float(np.mean([len(topk(A, i, k) & topk(B, i, k)) for i in range(n)])), 2)
        for k in (5, 10, 20)},
    "spearman_between_arms": round(float(spearmanr(A[iu], B[iu]).statistic), 3),
    "mean_cosine": {"abstract": round(float(A[iu].mean()), 3),
                    "intro": round(float(B[iu].mean()), 3)},
    "spearman_with_coupling_on_coupled_pairs": {
        "abstract": round(float(spearmanr(A[iu][nz], J[iu][nz]).statistic), 3),
        "intro": round(float(spearmanr(B[iu][nz], J[iu][nz]).statistic), 3)},
    "top10_overlap_with_coupling": {
        "abstract": round(float(np.mean([len(topk(A, i, 10) & topk(J, i, 10)) for i in coupled])), 2),
        "intro": round(float(np.mean([len(topk(B, i, 10) & topk(J, i, 10)) for i in coupled])), 2)},
}
(HERE / "results.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
