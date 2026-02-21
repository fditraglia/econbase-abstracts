"""
Step 1: Fetch econometrics paper metadata from arXiv.

Queries the econ.EM (Econometrics) category and saves titles, abstracts,
and metadata to a Parquet file for later embedding.

Usage:
    uv run fetch_abstracts.py
"""

import arxiv
import polars as pl
from pylatexenc.latex2text import LatexNodes2Text

# --- Configuration -----------------------------------------------------------
CATEGORY = "econ.EM"
MAX_RESULTS = 10000  # upper bound; arXiv will return fewer
OUTPUT_FILE = "econ_em_papers.parquet"

# --- 1. Query arXiv ---------------------------------------------------------
# arxiv.Search builds an API query. "cat:econ.EM" means "primary or
# cross-listed in the Econometrics category." We sort by submission date
# so the most recent papers come first.
#
# R analogy: this is like httr2::request() |> req_perform() — it builds
# the request, then the Client executes it lazily (one page at a time).

print(f"Querying arXiv for {CATEGORY} papers (all years)...")

search = arxiv.Search(
    query=f"cat:{CATEGORY}",
    max_results=MAX_RESULTS,
    sort_by=arxiv.SortCriterion.SubmittedDate,
)

# arxiv.Client().results() returns a *generator* — it doesn't download
# everything at once. It fetches pages of results lazily, respecting
# arXiv's rate limits (3 seconds between requests) automatically.
#
# R analogy: generators don't really exist in R. The closest thing is
# reading a file line-by-line with readLines(n=1) in a loop, rather than
# slurping the whole file into memory.

client = arxiv.Client()

# Collect all results into a list. The econ.EM category started in
# September 2017, so this is ~7 years of papers (~3,500-5,000 total).
papers = list(client.results(search))

print(f"Found {len(papers)} papers.")

# --- 2. Clean LaTeX ---------------------------------------------------------
# arXiv abstracts often contain raw LaTeX like \beta or \frac{x}{y}.
# These tokenise into meaningless fragments for an embedding model.
# We convert them to Unicode (β, x/y) using pylatexenc.
#
# R analogy: think of this like a custom stringr::str_replace_all() pass,
# but instead of regex, it actually parses LaTeX syntax.

converter = LatexNodes2Text()


def clean_latex(text: str) -> str:
    """Convert LaTeX markup to plain Unicode text."""
    try:
        return converter.latex_to_text(text)
    except Exception:
        # If parsing fails (rare), fall back to the raw string
        return text


# --- 3. Build a DataFrame ---------------------------------------------------
# Polars is a fast DataFrame library (like data.table or dplyr in R).
# We extract the fields we need from each paper result.
#
# Key Python concept: *list comprehensions*. The expression
#   [paper.title for paper in papers]
# is equivalent to R's
#   sapply(papers, function(p) p$title)

df = pl.DataFrame({
    "arxiv_id": [paper.entry_id.split("/")[-1] for paper in papers],
    "authors": [
        ", ".join(a.name for a in paper.authors) for paper in papers
    ],
    "title_raw": [paper.title for paper in papers],
    "abstract_raw": [paper.summary for paper in papers],
    "title": [clean_latex(paper.title) for paper in papers],
    "abstract": [clean_latex(paper.summary) for paper in papers],
    "primary_category": [paper.primary_category for paper in papers],
    "categories": [", ".join(paper.categories) for paper in papers],
    "is_primary_econ_em": [
        paper.primary_category == CATEGORY for paper in papers
    ],
    "published": [paper.published.date() for paper in papers],
    "url": [paper.entry_id for paper in papers],
})

# Show a summary (like R's str() or glimpse())
print(f"\nDataFrame shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Primary econ.EM: {df['is_primary_econ_em'].sum()}")
print(f"Cross-listed:    {(~df['is_primary_econ_em']).sum()}")
print(f"Date range:      {df['published'].min()} to {df['published'].max()}")
print()
print(df.head(3))

# --- 4. Save to Parquet -----------------------------------------------------
# Parquet is a compressed columnar file format — much smaller and faster
# than CSV, and it preserves column types (no guessing on read).
# R analogy: arrow::write_parquet()

df.write_parquet(OUTPUT_FILE)
print(f"\nSaved to {OUTPUT_FILE}")
