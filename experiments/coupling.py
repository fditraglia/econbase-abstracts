# /// script
# dependencies = ["polars", "numpy", "scipy"]
# ///
"""Bibliographic coupling over the pilot pool, and its agreement with abstract cosine.

Coupling asks whether two papers cite the same works. Reference identity is a
(first-author surname, year, normalized title prefix) key, which never requires
resolving a reference to a corpus paper.
"""
import re, sqlite3, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, polars as pl
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
# Martin's citation-intensity database, in the layout econ-corpus/docs/00-cold-start.md
# prescribes: gcloud storage rsync -r gs://econbase-arxiv-corpus/ ~/corpus/arxiv/
DB = Path.home() / "corpus" / "arxiv" / "citation_intensity" / "citation_intensity.db"
if not DB.exists():
    raise SystemExit(
        f"citation-intensity database not found at {DB}\n"
        "Fetch it with:  gcloud storage rsync -r "
        "gs://econbase-arxiv-corpus/citation_intensity/ ~/corpus/arxiv/citation_intensity/"
    )

emb = pl.read_parquet(HERE.parent / "econ_em_embeddings.parquet")
emb = emb.with_columns(pl.col("arxiv_id").str.replace(r"v\d+$", "").alias("base_id"))
intr = pl.read_parquet("/Users/francisditraglia/pangram-arxiv/data/intros.parquet")
intr = intr.filter((pl.col("method") == "intro-section")
                   & (pl.col("words_intro") > 300) & (pl.col("words_intro") < 4000))
pool = emb.join(intr.select(["base_id"]), on="base_id", how="inner")
ids = pool["base_id"].to_list()
print(f"pool: {len(ids)} papers")

def refkey(a, y, t):
    sn = re.split(r"[,;&]| and ", (a or "").strip())[0].split()
    sn = re.sub(r"[^a-z]", "", sn[-1].lower()) if sn else ""
    if not sn or not y:
        return None
    return (sn, str(y)[:4], re.sub(r"[^a-z0-9]", "", (t or "").lower())[:40])

db = sqlite3.connect(DB)
refs = defaultdict(set)
for p, a, y, t in db.execute("select paper_id, authors, year, title from references_"):
    k = refkey(a, y, t)
    if k:
        refs[p].add(k)

have = [i for i in ids if len(refs.get(i, ())) >= 5]
print(f"pool papers with >=5 keyed references: {len(have)}")

idx = {p: i for i, p in enumerate(have)}
sets = [refs[p] for p in have]
n = len(have)
coup = np.zeros((n, n))
inv = defaultdict(list)
for i, s in enumerate(sets):
    for k in s:
        inv[k].append(i)
for k, ps in inv.items():
    if len(ps) > 200:
        continue
    for a in range(len(ps)):
        for b in range(a + 1, len(ps)):
            coup[ps[a], ps[b]] += 1
            coup[ps[b], ps[a]] += 1
sizes = np.array([len(s) for s in sets], float)
jac = coup / (sizes[:, None] + sizes[None, :] - coup)
np.fill_diagonal(jac, 0.0)

sub = pool.filter(pl.col("base_id").is_in(have))
order = {p: i for i, p in enumerate(sub["base_id"].to_list())}
perm = [order[p] for p in have]
X = np.array(sub["embedding"].to_list(), dtype=np.float32)[perm]
X = X / np.linalg.norm(X, axis=1, keepdims=True)
cos = X @ X.T
np.fill_diagonal(cos, 0.0)

iu = np.triu_indices(n, 1)
c, j = cos[iu], jac[iu]
nz = j > 0
print(f"\npairs: {len(c)}   with any shared reference: {nz.sum()} ({100*nz.mean():.1f}%)")
print(f"Spearman(cosine, coupling), all pairs      : {spearmanr(c, j).statistic:.3f}")
print(f"Spearman(cosine, coupling), coupled pairs  : {spearmanr(c[nz], j[nz]).statistic:.3f}")
print(f"mean cosine | coupled   : {c[nz].mean():.3f}")
print(f"mean cosine | uncoupled : {c[~nz].mean():.3f}")

# top-10 overlap per paper between the two measures
ov = []
for i in range(n):
    a = set(np.argsort(-cos[i])[:10]); b = set(np.argsort(-jac[i])[:10])
    if jac[i].max() > 0:
        ov.append(len(a & b))
print(f"\nmean overlap of top-10 neighbors (cosine vs coupling): {np.mean(ov):.2f} of 10")
np.save(HERE / "cos.npy", cos); np.save(HERE / "jac.npy", jac)
pl.DataFrame({"base_id": have}).write_parquet(HERE / "pool.parquet")
