# Abstract versus introduction embeddings

Does embedding more of a paper find better related work? Findings are in
**`embedding-experiment.qmd`** (render for the report with figures and tables);
the short version is that introductions return different neighbors from
abstracts, not better ones, at six to seven times the compute.

## Reproducing

Requires Ollama with `qwen3-embedding` pulled, and the citation-intensity
database at `~/Downloads/citation_intensity_citation_intensity.db`
(from `gs://econbase-arxiv-corpus/citation_intensity/`).

```bash
uv run embed_intros.py        # ~35 min, resumable — writes intro_embeddings.parquet
uv run coupling.py            # bibliographic coupling + agreement with abstract cosine
uv run compare_arms.py        # the head-to-head -> results.json
uv run export_for_report.py   # tidy CSVs the report reads
quarto render embedding-experiment.qmd
```

`make_triplets.py` and `judge.py` build and serve the human triplet task. That
elicitation was abandoned as unworkable — see the report — and they are kept
because the list-comparison design that should replace it reuses most of them.

Large intermediates (`*.npy`, `intro_embeddings.parquet`, `report_pairs.csv`)
are gitignored and regenerate from the scripts above.
