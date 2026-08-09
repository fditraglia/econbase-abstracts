# /// script
# dependencies = ["polars", "numpy"]
# ///
"""Build disagreement triplets: query A, one candidate from each arm.

For each query paper we take the top neighbor under arm 1 and under arm 2.
A triplet is kept only when the two arms nominate different papers AND each
arm ranks its own nominee clearly above the rival's, so the pair genuinely
discriminates between the representations. Display order is randomised and
recorded, so position bias can be measured rather than assumed.
"""
import argparse, json, random
from pathlib import Path
import numpy as np, polars as pl

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--seed", type=int, default=20260809)
ap.add_argument("--gap", type=int, default=5, help="min rank gap for a nominee to count as preferred")
a = ap.parse_args()

pool = pl.read_parquet(HERE / "pool.parquet")["base_id"].to_list()
cos = np.load(HERE / "cos.npy")                      # abstract arm
intro = pl.read_parquet(HERE / "intro_embeddings.parquet")
have = {r["base_id"]: r["intro_embedding"] for r in intro.iter_rows(named=True)}

keep = [i for i, p in enumerate(pool) if p in have]
ids = [pool[i] for i in keep]
A = cos[np.ix_(keep, keep)]
Y = np.array([have[p] for p in ids], dtype=np.float32)
Y /= np.linalg.norm(Y, axis=1, keepdims=True)
B = Y @ Y.T
np.fill_diagonal(A, -9); np.fill_diagonal(B, -9)
print(f"papers with both arms: {len(ids)}")

meta = pl.read_parquet(HERE.parent / "econ_em_embeddings.parquet",
                       columns=["arxiv_id", "title", "abstract", "authors", "published"])
meta = meta.with_columns(pl.col("arxiv_id").str.replace(r"v\d+$", "").alias("base_id"))
M = {r["base_id"]: r for r in meta.iter_rows(named=True)}

rank_a = np.argsort(-A, axis=1); rank_b = np.argsort(-B, axis=1)
pos_a = np.argsort(rank_a, axis=1); pos_b = np.argsort(rank_b, axis=1)

rng = random.Random(a.seed)
trip = []
for i in range(len(ids)):
    ca, cb = rank_a[i, 0], rank_b[i, 0]
    if ca == cb:
        continue
    # each arm must clearly prefer its own nominee over the rival's
    if pos_b[i, ca] - pos_b[i, cb] < a.gap:  continue
    if pos_a[i, cb] - pos_a[i, ca] < a.gap:  continue
    flip = rng.random() < 0.5
    left, right = (cb, ca) if flip else (ca, cb)
    trip.append({
        "qid": ids[i], "left": ids[left], "right": ids[right],
        "left_arm":  "intro" if flip else "abstract",
        "right_arm": "abstract" if flip else "intro",
        "rank_abstract_of_left": int(pos_a[i, left]), "rank_intro_of_left": int(pos_b[i, left]),
        "rank_abstract_of_right": int(pos_a[i, right]), "rank_intro_of_right": int(pos_b[i, right]),
    })
rng.shuffle(trip)
trip = trip[:a.n]
print(f"discriminating triplets: {len(trip)}")

def card(p):
    r = M[p]
    return {"id": p, "title": r["title"], "abstract": r["abstract"],
            "authors": r["authors"], "year": str(r["published"])[:4]}

out = [{**t, "q": card(t["qid"]), "L": card(t["left"]), "R": card(t["right"])} for t in trip]
(HERE / "triplets.json").write_text(json.dumps(out, indent=1))
print("wrote", HERE / "triplets.json")
