# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Embed and cluster econometrics paper abstracts from arXiv (category econ.EM) using Qwen3-Embedding-8B via Ollama, then serve a referee-recommender web app via Flask. The owner is an R user learning Python — code comments include R analogies for key concepts.

## Commands

```bash
uv run fetch_abstracts.py    # Pull all econ.EM papers from arXiv → econ_em_papers.parquet
uv run embed_abstracts.py    # Embed papers via Ollama → econ_em_embeddings.parquet (~30 min)
uv run python app.py         # Run Flask app locally at http://localhost:5000
```

Embedding requires Ollama running locally: `ollama pull qwen3-embedding`

## Deployment (Render)

The Flask app is deployed on Render. It auto-deploys on push to GitHub.

- **Build command**: `pip install -r requirements.txt`
- **Start command**: `gunicorn app:app`
- **Parquet files**: Tracked via Git LFS (too large for regular Git). Run `git lfs install` after cloning.

## Pipeline and data flow

1. **fetch_abstracts.py** → `econ_em_papers.parquet`: Fetches all econ.EM papers (primary + cross-listed), cleans LaTeX to Unicode, stores both raw and cleaned text
2. **embed_abstracts.py** → `econ_em_embeddings.parquet`: Reads papers parquet, embeds title+abstract via Ollama's HTTP API, appends 4096-dim embedding column
3. **app.py**: Flask web app — loads embeddings parquet, serves referee recommendations via cosine similarity + author aggregation

## Key design decisions

- **Ollama over MLX**: We benchmarked both; Ollama (llama.cpp + Metal) is ~1.5x faster than mlx-embeddings on M4 Max. MLX also had a `batch_encode_plus` incompatibility with transformers 5.x.
- **Qwen3-Embedding document format**: Documents get plain text (no prefix). Only queries for later search get the `Instruct: ...\nQuery: ...` wrapper.
- **Raw + cleaned text**: Both `title_raw`/`abstract_raw` and `title`/`abstract` (LaTeX→Unicode) are stored so cleaning errors can be audited.
- **Primary vs cross-listed**: `is_primary_econ_em` boolean and `primary_category` column let you filter either way.

## Parquet schema (econ_em_embeddings.parquet)

`arxiv_id`, `authors`, `title_raw`, `abstract_raw`, `title`, `abstract`, `primary_category`, `categories`, `is_primary_econ_em`, `published`, `url`, `embedding` (list of 4096 floats)

## Joplin notes (MCP)

Project notes live in the **SQARE-EconBase** notebook under Research Projects. Key notes:
- "ArXiv Econometrics Embeddings: Project Reference" — canonical project plan
- "Python & Toolchain Learning Log" — running log of Python/toolchain concepts learned (update as we go)
