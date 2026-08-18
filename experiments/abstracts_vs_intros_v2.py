# /// script
# dependencies = ["polars", "numpy", "scipy"]
# ///
"""Abstracts versus introductions, judged against corrected reference data.

Re-runs the 9 August experiment. That version built its bibliographic coupling measure
from citation_intensity.db, which the corpus has since replaced with a rebuilt reference
extraction. Everything here is rebuilt from citations.db's references_ table instead;
nothing else about the design changes, so the earlier numbers and these are directly
comparable.

Coupling rather than direct citation links, for the same reason as last time: only 203
citation links fall inside the 606-paper pool, far too few to rank anything, whereas
8.3% of pairs in the pool share at least one reference.

Usage:
    uv run abstracts_vs_intros_v2.py
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
CIT = Path.home() / "corpus" / "arxiv" / "citations" / "citations.db"
MIN_REFS = 5          # as in the 9 August run
TOP_K = 10
OUT = HERE / "abstracts_vs_intros_v2.json"


def ref_key(authors_raw: str, year: str, title: str):
    """(first-author surname, year, normalized title prefix).

    Identity by content, so a reference never has to be resolved to a corpus paper --
    which matters because 93% of references point outside the corpus.
    """
    surname = re.split(r"[,;&]| and ", (authors_raw or "").strip())[0].split()
    surname = re.sub(r"[^a-z]", "", surname[-1].lower()) if surname else ""
    if not surname or not year:
        return None
    return (surname, str(year)[:4], re.sub(r"[^a-z0-9]", "", (title or "").lower())[:40])


def unit_rows(M: np.ndarray) -> np.ndarray:
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def main() -> None:
    if not CIT.exists():
        raise SystemExit(f"citations database not found at {CIT}")

    abstracts = pl.read_parquet(HERE.parent / "econ_em_embeddings.parquet",
                                columns=["base_id", "title", "embedding"])
    intros = pl.read_parquet(HERE / "intro_embeddings.parquet")
    intros = intros.unique(subset=["base_id"], keep="first")
    pool_df = abstracts.join(intros, on="base_id", how="inner")
    print(f"papers with both an abstract and an introduction embedding: {pool_df.height}")

    db = sqlite3.connect(CIT)
    pool_ids = set(pool_df["base_id"].to_list())
    refs: dict[str, set] = defaultdict(set)
    for p, a, y, t in db.execute(
            "SELECT arxiv_id, authors_raw, year, title_raw FROM references_"):
        if p in pool_ids:
            k = ref_key(a, y, t)
            if k:
                refs[p].add(k)

    keep = [p for p in pool_df["base_id"].to_list() if len(refs.get(p, ())) >= MIN_REFS]
    print(f"of those, with at least {MIN_REFS} keyed references: {len(keep)}")
    pool_df = pool_df.filter(pl.col("base_id").is_in(keep)).sort("base_id")
    ids = pool_df["base_id"].to_list()
    n = len(ids)

    A = unit_rows(np.array(pool_df["embedding"].to_list(), dtype=np.float32))
    I = unit_rows(np.array(pool_df["intro_embedding"].to_list(), dtype=np.float32))
    assert A.shape == I.shape, "the two arms must cover the same papers in the same order"

    # --- coupling: Jaccard overlap of reference sets -------------------------
    sets = [refs[p] for p in ids]
    inv = defaultdict(list)
    for i, s in enumerate(sets):
        for k in s:
            inv[k].append(i)
    coup = np.zeros((n, n), dtype=np.float32)
    for k, members in inv.items():
        if len(members) > 200:      # a reference cited by hundreds carries no signal
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                coup[members[x], members[y]] += 1
                coup[members[y], members[x]] += 1
    sizes = np.array([len(s) for s in sets], dtype=np.float32)
    jac = coup / (sizes[:, None] + sizes[None, :] - coup)
    np.fill_diagonal(jac, 0.0)

    cosA, cosI = A @ A.T, I @ I.T
    np.fill_diagonal(cosA, 0.0)
    np.fill_diagonal(cosI, 0.0)

    iu = np.triu_indices(n, 1)
    j = jac[iu]
    coupled = j > 0
    print(f"pairs: {len(j):,}   sharing a reference: {coupled.sum():,} "
          f"({100*coupled.mean():.1f}%)")

    def per_paper_overlap(C: np.ndarray) -> np.ndarray:
        """Overlap with the coupling top-10, one value per paper, for paired testing."""
        return np.array([
            len(set(np.argsort(-C[i])[:TOP_K]) & set(np.argsort(-jac[i])[:TOP_K]))
            for i in range(n) if jac[i].max() > 0
        ], dtype=float)

    def arm_stats(C: np.ndarray, label: str) -> dict:
        c = C[iu]
        top_overlap = per_paper_overlap(C)
        return {
            "arm": label,
            "spearman_all_pairs": float(spearmanr(c, j).statistic),
            "spearman_coupled_pairs": float(spearmanr(c[coupled], j[coupled]).statistic),
            "mean_cos_coupled": float(c[coupled].mean()),
            "mean_cos_uncoupled": float(c[~coupled].mean()),
            "top10_overlap_with_coupling": float(np.mean(top_overlap)),
        }

    out = {"pool": n, "pairs": int(len(j)), "coupled_pairs": int(coupled.sum()),
           "abstract": arm_stats(cosA, "abstract"),
           "intro": arm_stats(cosI, "introduction")}

    # --- agreement between the two arms --------------------------------------
    same_top1 = sum(int(np.argmax(cosA[i]) == np.argmax(cosI[i])) for i in range(n))
    ov = [len(set(np.argsort(-cosA[i])[:TOP_K]) & set(np.argsort(-cosI[i])[:TOP_K]))
          for i in range(n)]
    out["arms"] = {
        "same_nearest_paper": same_top1 / n,
        "mean_top10_overlap": float(np.mean(ov)),
        "spearman_between_arms": float(spearmanr(cosA[iu], cosI[iu]).statistic),
    }

    # Paired comparison of the two arms on the same papers. The question the experiment
    # exists to answer is whether introductions beat abstracts, and the two per-paper
    # overlap series are paired, so the difference is what carries the evidence.
    oa, oi = per_paper_overlap(cosA), per_paper_overlap(cosI)
    d = oi - oa
    se = d.std(ddof=1) / np.sqrt(len(d))
    w = wilcoxon(oi, oa, zero_method="wilcox") if np.any(d != 0) else None
    out["paired"] = {
        "papers": int(len(d)),
        "mean_difference_intro_minus_abstract": float(d.mean()),
        "std_error": float(se),
        "ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
        "wilcoxon_p": float(w.pvalue) if w is not None else None,
        "intro_better": int((d > 0).sum()),
        "abstract_better": int((d < 0).sum()),
        "tied": int((d == 0).sum()),
    }

    print(json.dumps(out, indent=2))
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
