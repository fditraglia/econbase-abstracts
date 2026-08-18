# Comparing ways of finding related work

Two questions, both judged against the citation data in `citations.db`.

**Does embedding more of a paper find better related work?** Introductions come out
ahead of abstracts by 0.11 of one neighbor in ten (95% interval 0.01 to 0.20,
Wilcoxon p = 0.025) — a detectable advantage, too small to justify six to seven
times the embedding cost. Keep abstracts. Run `abstracts_vs_intros_v2.py`; the
figures with plots and tables are in `embedding-experiment.qmd`.

**How much does embedding similarity differ from the citation measure?** They share
about a fifth of what they name, at the paper level and the author level alike,
while each ranks the other's picks far above chance. Run
`compare_citation_embedding.py`; it writes the JSON and CSVs the write-up reads.

The write-up itself is in the private coordination repo, at
`econbase-sqare-docs/reference/citation-vs-embedding.qmd`, because it discusses
another team's pipeline. It renders against the data produced here.

## Reproducing

Requires Ollama with the frozen embedding model (`qwen3-embedding-econbase:v1`) and
the corpus in the layout `econ-corpus/docs/00-cold-start.md` prescribes:

```bash
for d in metadata citations serve; do
  gcloud storage rsync -r gs://econbase-arxiv-corpus/$d/ ~/corpus/arxiv/$d/
done
```

```bash
uv run compare_citation_embedding.py   # -> comparison_results.json + three CSVs
uv run abstracts_vs_intros_v2.py       # -> abstracts_vs_intros_v2.json
quarto render ~/econbase-sqare-docs/reference/citation-vs-embedding.qmd
```

Both read only local files, take a few minutes, and reproduce exactly.
`compare_citation_embedding.py --limit N` runs a random subsample while developing.

Introduction embeddings come from `embed_intros.py` (~35 min, resumable, writes
`intro_embeddings.parquet`). It currently covers 606 papers.

## Superseded, kept for reference

`coupling.py`, `compare_arms.py` and `export_for_report.py` were the 9 August
analysis. They read `citation_intensity.db`, which the corpus has since replaced,
so their figures predate the replacement. Do not re-run them;
`abstracts_vs_intros_v2.py` is the current version of the same comparison.

`make_triplets.py` and `judge.py` build and serve the human triplet task. That
elicitation was abandoned as unworkable — see the report — and they are kept
because the list-comparison design that should replace it reuses most of them.

Large intermediates (`*.npy`, `intro_embeddings.parquet`, `report_pairs.csv`) are
gitignored and regenerate from the scripts above.
