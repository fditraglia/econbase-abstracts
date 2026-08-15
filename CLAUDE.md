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

## Deployment (Hugging Face Spaces)

The Flask app is deployed at https://huggingface.co/spaces/fditraglia/referee-recommender (Docker SDK, free CPU tier). Deploy with:

```bash
git push hf master:main    # HF Spaces (SSH remote)
```

The `hf` remote points to `git@hf.co:spaces/fditraglia/referee-recommender`. The Dockerfile runs gunicorn on port 7860 (HF requirement). Parquet files are tracked via Git LFS.

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

## Corpus data (the `econ-corpus` bucket)

The team's arXiv corpus is the upstream source for paper metadata and for anything read beyond the abstract. It lives in Google Cloud Storage, **not** on Google Drive or Dropbox, and is rebuilt daily at 03:00 UTC by a cron job on the IONOS server.

The Google Cloud CLI is installed at `~/google-cloud-sdk` (Google's standalone installer, not the Homebrew cask, which has given trouble). Pull the three directories that matter — under 1 GB:

```bash
for d in metadata citations serve; do
  gcloud storage rsync -r gs://econbase-arxiv-corpus/$d/ ~/corpus/arxiv/$d/
done
```

| Path | What it holds |
|---|---|
| `metadata/arxiv_econ_papers.db` | one row per econ.EM paper, including cross-listed; the source for the embedding refresh |
| `citations/citations.db` | main text, parsed references, weighted citation intensity (`composite_index`), `related_papers`, `related_authors_sym` |
| `serve/main-text/<id>.txt.gz` | main text per paper, title through conclusion, appendix dropped |

**Skip two directories.** `source_cache/` is 8.3 GB of raw LaTeX and is unnecessary now that main text is extracted upstream. `citation_intensity/` is superseded by `citations/citations.db` — an audit found seven classes of extraction bug in it, and the bucket README asks that nothing new be built against it. `experiments/coupling.py` still reads it; see `experiments/README.md`.

Method and schema for the citation data are documented in `econ-corpus/docs/06-citations-pipeline.md`. Read it before trusting a number from `citations.db`.

## Joplin notes (MCP)

Project notes live in the **SQARE-EconBase** notebook under Research Projects. Key notes:
- "ArXiv Econometrics Embeddings: Project Reference" — canonical project plan
- "Python & Toolchain Learning Log" — running log of Python/toolchain concepts learned (update as we go)

## Project context

This repository is one component of the **EconBase / SQARE** project. For the roadmap, how the repositories relate, and current priorities, see the private coordination repo **`fditraglia/econbase-sqare-docs`**.
