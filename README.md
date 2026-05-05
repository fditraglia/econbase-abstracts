---
title: Referee Recommender
emoji: 😻
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
license: mit
short_description: Use Qwen embeddings to recommend econometrics referees
---

# Referee Recommender

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/fditraglia/referee-recommender)

Suggests potential referees for an econometrics paper. Enter an arXiv ID from category `econ.EM` and get back a ranked list of authors whose recent work is closest to yours in embedding space.

## How it works

1. **Fetch** all `econ.EM` papers (primary + cross-listed) from arXiv.
2. **Embed** each title + abstract with [Qwen3-Embedding-8B](https://ollama.com/library/qwen3-embedding) (4096-dim) via a local Ollama server.
3. **Recommend**: cosine-similarity search against the query paper, then aggregate hits by author to produce the referee ranking.

## Run locally

The embeddings parquet ships with the repo via Git LFS, so you can run the app without Ollama or any rebuild step.

**Prerequisites:** Python ≥ 3.12, [uv](https://docs.astral.sh/uv/), and [Git LFS](https://git-lfs.com/).

```bash
git lfs install                 # one-time per machine
git clone <this repo>
cd econbase-abstracts
uv sync                         # creates .venv and installs deps
uv run python app.py            # serves at http://localhost:5000
```

Open http://localhost:5000, paste an arXiv ID from `econ.EM` (e.g. `2401.12345`), and you'll get a ranked list of suggested referees.

The first request takes a moment while the 164 MB parquet loads into memory; afterwards it's snappy. If you see `Loaded N papers, embedding matrix shape: (N, 4096)` in the terminal, the app is ready.

## Rebuild embeddings from scratch

Only needed if you want fresher arXiv data or a different embedding model.

**Additional prerequisite:** [Ollama](https://ollama.com/) running locally.

```bash
ollama pull qwen3-embedding
uv run fetch_abstracts.py       # arXiv → econ_em_papers.parquet
uv run embed_abstracts.py       # papers → econ_em_embeddings.parquet (~30 min on M4 Max)
```

## Data

`econ_em_embeddings.parquet` columns:

`arxiv_id`, `authors`, `title_raw`, `abstract_raw`, `title`, `abstract`, `primary_category`, `categories`, `is_primary_econ_em`, `published`, `url`, `embedding` (list of 4096 floats)

Both raw and LaTeX-cleaned text are kept so cleaning errors can be audited. `is_primary_econ_em` distinguishes primary from cross-listed papers.

## Deployment

The app is deployed to Hugging Face Spaces (Docker SDK, free CPU tier). The `Dockerfile` runs gunicorn on port 7860 (HF requirement) using the slimmer `requirements.txt` (no embedding-time dependencies).

```bash
git push hf master:main
```

The `hf` remote points to `git@hf.co:spaces/fditraglia/referee-recommender`.

## Notes

- We benchmarked Ollama against `mlx-embeddings`; Ollama (llama.cpp + Metal) was ~1.5× faster on an M4 Max.
- Qwen3-Embedding documents are embedded as plain text (no prefix). Only search-time queries get the `Instruct: ...\nQuery: ...` wrapper.
