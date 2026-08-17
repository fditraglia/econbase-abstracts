# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Embed and cluster econometrics paper abstracts from arXiv (category econ.EM) using Qwen3-Embedding-8B via Ollama, then serve a referee-recommender web app via Flask. The owner is an R user learning Python — code comments include R analogies for key concepts.

## Commands

```bash
uv run refresh_embeddings.py --dry-run   # what has changed on arXiv since the last run
uv run refresh_embeddings.py             # embed what is new, rebuild the table, publish
git add -A && git commit && git push hf master:main   # deploy the result
uv run python app.py                     # Flask app at http://localhost:5000
```

One refresh does three things: embeds papers that are new or revised, rebuilds `similarity_top100.parquet`, and uploads the vectors to the bucket. `--no-table` and `--no-publish` skip the last two. Skipping the rebuild by hand is the trap it exists to avoid — the app would serve recommendations from the old corpus with nothing to show it.

It is safe to re-run at any time; a run with nothing new costs one arXiv query. `fetch_abstracts.py` and `embed_abstracts.py` remain for a cold build from nothing.

Embedding needs Ollama running with the frozen model: `ollama cp qwen3-embedding:latest qwen3-embedding-econbase:v1`. The frozen copy matters because `latest` is a moving tag — if it repoints, new vectors land in a different space from the old ones, cosine similarities across the boundary become meaningless, and nothing visibly breaks. The refresh checks the model digest against `embeddings_manifest.json` and refuses to run if it has moved.

## Deployment (Hugging Face Spaces)

The Flask app is deployed at https://huggingface.co/spaces/fditraglia/referee-recommender (Docker SDK, free CPU tier). Deploy with:

```bash
git push hf master:main    # HF Spaces (SSH remote)
```

The `hf` remote points to `git@hf.co:spaces/fditraglia/referee-recommender`. The Dockerfile runs gunicorn on port 7860 (HF requirement). Parquet files are tracked via Git LFS.

The Space mirrors whatever tree is pushed, so removing a file from the repository removes it from the Space at the next deploy — no separate deletion step. It also means the Space can lag: it serves the last tree pushed to it, not the last commit made here. Check with `git ls-remote hf` when it matters.

## Pipeline and data flow

1. **refresh_embeddings.py** → `econ_em_papers.parquet` + `econ_em_embeddings.parquet`: asks arXiv what has changed since the last run, sorted by *last updated* rather than submission date so revisions are visible, cleans LaTeX to Unicode, and embeds only papers that are new or whose abstract text actually moved. Keys on the version-stripped id, writes through a temporary file, and records what it did in `embeddings_manifest.json`.
2. **build_similarity_table.py** → `similarity_top100.parquet`: each paper's 100 nearest neighbors with their cosine similarities, about 4 MB.
3. **app.py**: Flask web app — loads the papers parquet and the similarity table, serves referee recommendations via author aggregation.

**The app never reads the 4096-dimensional vectors.** It only ranks papers against a paper already in the corpus, so it needs the answers rather than the raw material. The interface can ask for at most 100 neighbors, so a top-100 table reproduces its output exactly rather than approximately — verified over 300 papers, identical referee lists and ordering, differences only at the 1e-6 level from float32 summation order.

`econ_em_embeddings.parquet` is therefore gitignored. It is needed for analysis and for rebuilding the table, and its canonical copy is `gs://econbase-embeddings/econ_em_embeddings.parquet`, with `current.json` beside it recording the paper count, date, model digest and SHA-256. Fetch it with:

```bash
gcloud storage cp gs://econbase-embeddings/econ_em_embeddings.parquet .
```

## Key design decisions

- **Ollama over MLX**: We benchmarked both; Ollama (llama.cpp + Metal) is ~1.5x faster than mlx-embeddings on M4 Max. MLX also had a `batch_encode_plus` incompatibility with transformers 5.x.
- **Qwen3-Embedding document format**: Documents get plain text (no prefix). Only queries for later search get the `Instruct: ...\nQuery: ...` wrapper.
- **Raw + cleaned text**: Both `title_raw`/`abstract_raw` and `title`/`abstract` (LaTeX→Unicode) are stored so cleaning errors can be audited.
- **Primary vs cross-listed**: `is_primary_econ_em` boolean and `primary_category` column let you filter either way.

## Parquet schemas

`econ_em_papers.parquet` (deployed, ~4 MB): `arxiv_id`, `base_id`, `version`, `authors`, `title_raw`, `abstract_raw`, `title`, `abstract`, `primary_category`, `categories`, `is_primary_econ_em`, `published`, `url`, `text_sha`

`econ_em_embeddings.parquet` (bucket only): the same columns plus `embedding`, a list of 4096 floats.

`similarity_top100.parquet` (deployed, ~4 MB): `arxiv_id`, `neighbor_ids` (list of 100 strings), `cosines` (list of 100 float32), ordered by descending similarity and excluding the paper itself.

`base_id` is the identifier with the version suffix stripped. A revision changes `arxiv_id` from `v3` to `v4`, so anything joining across refreshes must key on `base_id`; the app accepts either form and redirects to the current one.

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

**Skip two directories.** `source_cache/` is 8.3 GB of raw LaTeX and is unnecessary now that main text is extracted upstream. `citation_intensity/` has been superseded by `citations/citations.db`; the bucket README asks that nothing new be built against it. `experiments/coupling.py` still reads it, so its figures predate the replacement — see `experiments/README.md`.

Method and schema for the citation data are documented in the `econ-corpus` repository, which also owns any question about how a measure is defined. Read that before trusting a number from `citations.db`.

## This repository is public

The GitHub repository is public, and the whole working tree is also pushed to a public Hugging Face Space, `CLAUDE.md` included. Everything written here is published, so keep notes operational — which data to use, which is superseded, what needs re-running — and keep assessments of other people's code and other repositories' internal history in the private coordination repo instead.

## Joplin notes (MCP)

Project notes live in the **SQARE-EconBase** notebook under Research Projects. Key notes:
- "ArXiv Econometrics Embeddings: Project Reference" — canonical project plan
- "Python & Toolchain Learning Log" — running log of Python/toolchain concepts learned (update as we go)

## Project context

This repository is one component of the **EconBase / SQARE** project. For the roadmap, how the repositories relate, and current priorities, see the private coordination repo **`fditraglia/econbase-sqare-docs`**.
