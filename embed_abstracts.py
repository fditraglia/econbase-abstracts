"""
Step 2: Embed paper titles and abstracts using Qwen3-Embedding-8B.

Reads the cleaned paper metadata from Parquet, runs each paper's
title + abstract through the embedding model via Ollama, and saves
the result.

Requires Ollama running locally with the model already pulled:
    ollama pull qwen3-embedding

Usage:
    uv run embed_abstracts.py
"""

import json
import time
import urllib.request

import numpy as np
import polars as pl

# --- Configuration -----------------------------------------------------------
INPUT_FILE = "econ_em_papers.parquet"
OUTPUT_FILE = "econ_em_embeddings.parquet"
MODEL_NAME = "qwen3-embedding"
BATCH_SIZE = 32

# Ollama runs as a local HTTP server on port 11434. We send it JSON
# requests and get JSON responses — no Python SDK needed.
# R analogy: like calling httr2::request("http://localhost:11434/...")
OLLAMA_URL = "http://localhost:11434/api/embed"

# --- 1. Load the paper data --------------------------------------------------
print(f"Loading papers from {INPUT_FILE}...", flush=True)
df = pl.read_parquet(INPUT_FILE)
print(f"Loaded {df.shape[0]} papers.", flush=True)

# --- 2. Prepare text for embedding -------------------------------------------
# We concatenate title + abstract for each paper. Ablation studies show
# this outperforms abstract-only because the title carries a strong
# topical signal.
#
# Qwen3-Embedding distinguishes "query" vs "document" roles:
#   - Documents (what we're embedding now): plain text, no prefix
#   - Queries (for later search): get an "Instruct: ...\nQuery: ..." prefix
#
# So here we just pass the raw text.

texts = [
    f"{title}\n{abstract}"
    for title, abstract in zip(df["title"].to_list(), df["abstract"].to_list())
]

print(f"Prepared {len(texts)} texts for embedding.", flush=True)

# --- 3. Embed in batches via Ollama ------------------------------------------
# Ollama's /api/embed endpoint accepts a list of strings in the "input"
# field and returns a list of embedding vectors. We send batches of 32.
#
# Under the hood, Ollama uses llama.cpp with a Metal backend to run the
# model on the GPU — same hardware as MLX, but a more optimised runtime.
# Benchmarking showed ~0.36s/paper vs MLX's ~0.54s/paper on M4 Max.

print(f"\nEmbedding {len(texts)} papers in batches of {BATCH_SIZE}...", flush=True)
all_embeddings = []

n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
t_start = time.time()

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1

    # Build and send the HTTP request to Ollama
    payload = json.dumps({"model": MODEL_NAME, "input": batch}).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())

    all_embeddings.extend(result["embeddings"])

    elapsed = time.time() - t_start
    rate = (i + len(batch)) / elapsed
    eta = (len(texts) - i - len(batch)) / rate if rate > 0 else 0
    print(
        f"  Batch {batch_num}/{n_batches} — "
        f"{i + len(batch)}/{len(texts)} papers — "
        f"{rate:.1f} papers/s — ETA {eta:.0f}s",
        flush=True,
    )

elapsed_total = time.time() - t_start
print(f"\nEmbedded {len(texts)} papers in {elapsed_total:.1f}s", flush=True)

# --- 4. Save to Parquet ------------------------------------------------------
# Convert to numpy matrix for a quick shape check, then store as
# list-of-lists in Parquet (each row's embedding is a list of floats).
embeddings_matrix = np.array(all_embeddings)
print(f"Embedding matrix shape: {embeddings_matrix.shape}", flush=True)

df = df.with_columns(
    pl.Series("embedding", embeddings_matrix.tolist())
)

df.write_parquet(OUTPUT_FILE)
print(f"Saved to {OUTPUT_FILE}", flush=True)
