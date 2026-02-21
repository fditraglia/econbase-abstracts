"""
Referee Recommender — Flask web app.

Enter an arXiv ID, get recommended referees based on embedding similarity
of econ.EM papers.

Usage:
    uv run python app.py
    # Then open http://localhost:5000

R analogy: this is like a single-file Shiny app — UI + server logic in one
place — but using Flask (Python's lightweight web framework) instead.
"""

from collections import defaultdict
from flask import Flask, render_template_string, request, redirect, url_for
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load data on startup
# ---------------------------------------------------------------------------
# Read the parquet file with pandas (pyarrow backend handles the nested
# embedding lists). We convert embeddings to a pre-normalised float32
# numpy matrix so cosine similarity is just a dot product.
#
# R analogy: like loading an .rds file in global.R for a Shiny app — it
# runs once when the app starts, not on every request.

print("Loading embeddings...", flush=True)
df = pd.read_parquet("econ_em_embeddings.parquet")

# Build the embedding matrix: each row is a paper's 4096-dim vector
embeddings = np.array(df["embedding"].tolist(), dtype=np.float32)

# L2-normalise so dot product == cosine similarity
# R analogy: like sweep(mat, 1, sqrt(rowSums(mat^2)), "/")
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1  # guard against zero vectors
embeddings = embeddings / norms

# Lookup dict: arxiv_id string → row index in the matrix
id_to_idx = {aid: i for i, aid in enumerate(df["arxiv_id"])}

N_PAPERS = len(df)
print(f"Loaded {N_PAPERS} papers, embedding matrix shape: {embeddings.shape}",
      flush=True)


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------

def find_similar(arxiv_id, top_k=50):
    """Return indices and cosine similarities of the top-k most similar papers."""
    idx = id_to_idx[arxiv_id]
    query_vec = embeddings[idx]  # already normalised

    # Single matrix-vector dot product → all cosine similarities (< 1 ms)
    sims = embeddings @ query_vec
    # Zero out the query paper itself
    sims[idx] = -1

    # argpartition is O(n) vs O(n log n) for full sort — faster for large n
    # R analogy: no direct equivalent; sort(x, partial=k) is similar
    top_indices = np.argpartition(sims, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
    top_sims = sims[top_indices]

    return top_indices, top_sims


def aggregate_referees(arxiv_id, agg_rule="mean-top-3", top_k_papers=50):
    """
    Aggregate author scores from the top-k most similar papers.

    Each similar paper contributes its cosine similarity score to all of its
    authors. Then we aggregate per-author using the chosen rule and remove
    the query paper's own authors.

    Returns a list of dicts sorted by score descending:
        [{"author": str, "score": float, "n_papers": int,
          "papers": [{"title": str, "arxiv_id": str, "sim": float}, ...]}, ...]
    """
    top_indices, top_sims = find_similar(arxiv_id, top_k=top_k_papers)

    # Collect per-author similarity scores and paper info
    author_scores = defaultdict(list)
    author_papers = defaultdict(list)

    for idx, sim in zip(top_indices, top_sims):
        row = df.iloc[idx]
        # Authors are stored as comma-separated string
        authors = [a.strip() for a in row["authors"].split(",")]
        for author in authors:
            author_scores[author].append(float(sim))
            author_papers[author].append({
                "title": row["title"],
                "arxiv_id": row["arxiv_id"],
                "sim": float(sim),
            })

    # Remove query paper's own authors
    query_row = df.iloc[id_to_idx[arxiv_id]]
    query_authors = {a.strip() for a in query_row["authors"].split(",")}

    results = []
    for author, scores in author_scores.items():
        if author in query_authors:
            continue

        scores_sorted = sorted(scores, reverse=True)

        if agg_rule == "max":
            score = scores_sorted[0]
        elif agg_rule == "mean-top-3":
            top3 = scores_sorted[:3]
            score = sum(top3) / len(top3)
        elif agg_rule == "sum":
            score = sum(scores_sorted)
        else:
            score = scores_sorted[0]

        results.append({
            "author": author,
            "score": score,
            "n_papers": len(scores),
            "papers": sorted(author_papers[author],
                             key=lambda p: p["sim"], reverse=True),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        arxiv_id = request.form.get("arxiv_id", "").strip()
        if arxiv_id:
            return redirect(url_for("recommend", arxiv_id=arxiv_id))
    return render_template_string(HOME_TEMPLATE, n_papers=N_PAPERS)


@app.route("/recommend/<path:arxiv_id>")
def recommend(arxiv_id):
    arxiv_id = arxiv_id.strip()
    if arxiv_id not in id_to_idx:
        return render_template_string(ERROR_TEMPLATE, arxiv_id=arxiv_id,
                                      n_papers=N_PAPERS)

    # Read controls from query params (defaults: mean-top-3, 10 results)
    agg_rule = request.args.get("agg", "mean-top-3")
    if agg_rule not in ("max", "mean-top-3", "sum"):
        agg_rule = "mean-top-3"
    n_show = request.args.get("n", "10")
    try:
        n_show = int(n_show)
    except ValueError:
        n_show = 10
    if n_show not in (10, 20, 50):
        n_show = 10

    # Get query paper info
    query_row = df.iloc[id_to_idx[arxiv_id]]

    # Get referees
    referees = aggregate_referees(arxiv_id, agg_rule=agg_rule, top_k_papers=50)

    # Get similar papers for display
    top_indices, top_sims = find_similar(arxiv_id, top_k=n_show)
    similar_papers = []
    for idx, sim in zip(top_indices, top_sims):
        row = df.iloc[idx]
        similar_papers.append({
            "title": row["title"],
            "authors": row["authors"],
            "published": str(row["published"]),
            "arxiv_id": row["arxiv_id"],
            "url": row["url"],
            "abstract": row["abstract"],
            "sim": float(sim),
        })

    return render_template_string(
        RECOMMEND_TEMPLATE,
        paper=query_row,
        referees=referees[:n_show],
        similar_papers=similar_papers,
        agg_rule=agg_rule,
        n_show=n_show,
        arxiv_id=arxiv_id,
        n_papers=N_PAPERS,
    )


# ---------------------------------------------------------------------------
# Inline templates
# ---------------------------------------------------------------------------

STYLE = """
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: 960px;
        margin: 0 auto;
        padding: 20px;
        background: #fafafa;
    }
    h1 { font-size: 1.6em; margin-bottom: 4px; }
    h2 { font-size: 1.3em; margin: 24px 0 12px; border-bottom: 2px solid #2563eb; padding-bottom: 4px; }
    h3 { font-size: 1.1em; margin: 16px 0 8px; }
    a { color: #2563eb; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .subtitle { color: #666; font-size: 0.9em; margin-bottom: 20px; }
    .card {
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .card p { margin: 6px 0; }
    .meta { color: #666; font-size: 0.85em; }
    .abstract { font-size: 0.92em; margin-top: 8px; }
    form.search-form {
        display: flex;
        gap: 8px;
        margin: 20px 0;
    }
    form.search-form input[type="text"] {
        flex: 1;
        padding: 10px 14px;
        border: 1px solid #ccc;
        border-radius: 6px;
        font-size: 1em;
    }
    form.search-form button {
        padding: 10px 20px;
        background: #2563eb;
        color: #fff;
        border: none;
        border-radius: 6px;
        font-size: 1em;
        cursor: pointer;
    }
    form.search-form button:hover { background: #1d4ed8; }
    .controls {
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 16px;
        display: flex;
        gap: 24px;
        align-items: center;
        flex-wrap: wrap;
    }
    .controls label { margin-right: 8px; font-size: 0.9em; }
    .controls select, .controls input[type="radio"] { cursor: pointer; }
    .controls .control-group { display: flex; align-items: center; gap: 6px; }
    .controls button {
        padding: 6px 16px;
        background: #2563eb;
        color: #fff;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.9em;
    }
    .controls button:hover { background: #1d4ed8; }
    table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 16px;
        background: #fff;
    }
    th, td {
        text-align: left;
        padding: 8px 10px;
        border-bottom: 1px solid #eee;
        font-size: 0.9em;
    }
    th { background: #f5f5f5; font-weight: 600; }
    tr:hover { background: #f9f9f9; }
    .paper-list .paper-item {
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .paper-list .paper-item .title { font-weight: 600; }
    .paper-list .paper-item .sim-score {
        display: inline-block;
        background: #e0edff;
        color: #1d4ed8;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: 600;
    }
    .expand-toggle {
        background: none;
        border: none;
        color: #2563eb;
        cursor: pointer;
        font-size: 0.85em;
        padding: 0;
    }
    .expand-toggle:hover { text-decoration: underline; }
    .paper-links { font-size: 0.85em; color: #666; margin-top: 4px; }
    .paper-links a { margin-right: 8px; }
    .error { color: #dc2626; font-weight: 600; }
    .home-link { display: inline-block; margin-bottom: 16px; font-size: 0.9em; }
</style>
"""

HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Referee Recommender</title>
    """ + STYLE + """
</head>
<body>
    <h1>Referee Recommender</h1>
    <p class="subtitle">{{ n_papers | number_format }} econ.EM papers from arXiv through February 19, 2026</p>
    <form class="search-form" method="post">
        <input type="text" name="arxiv_id" placeholder="Enter an arXiv ID (e.g. 2301.12345)" autofocus>
        <button type="submit">Find referees</button>
    </form>
</body>
</html>
"""

ERROR_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Not Found — Referee Recommender</title>
    """ + STYLE + """
</head>
<body>
    <a class="home-link" href="/">&larr; Back to search</a>
    <h1>Referee Recommender</h1>
    <p class="subtitle">{{ n_papers | number_format }} econ.EM papers from arXiv through February 19, 2026</p>
    <div class="card">
        <p class="error">arXiv ID "{{ arxiv_id }}" not found in the database.</p>
        <p>Make sure the ID matches a paper in the econ.EM category (e.g. 2301.12345).</p>
    </div>
</body>
</html>
"""

RECOMMEND_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{{ paper.title }} — Referee Recommender</title>
    """ + STYLE + """
</head>
<body>
    <a class="home-link" href="/">&larr; Back to search</a>
    <h1>Referee Recommender</h1>
    <p class="subtitle">{{ n_papers | number_format }} econ.EM papers from arXiv through February 19, 2026</p>

    <!-- Query paper card -->
    <h2>Query Paper</h2>
    <div class="card">
        <h3>{{ paper.title }}</h3>
        <p class="meta">{{ paper.authors }}</p>
        <p class="meta">Published: {{ paper.published }} &middot;
           <a href="{{ paper.url }}" target="_blank">arXiv:{{ arxiv_id }}</a></p>
        <p class="abstract">{{ paper.abstract }}</p>
    </div>

    <!-- Controls -->
    <form class="controls" method="get" action="/recommend/{{ arxiv_id }}">
        <div class="control-group">
            <label><strong>Aggregation:</strong></label>
            <label><input type="radio" name="agg" value="max" {{ "checked" if agg_rule == "max" }}> Max</label>
            <label><input type="radio" name="agg" value="mean-top-3" {{ "checked" if agg_rule == "mean-top-3" }}> Mean top-3</label>
            <label><input type="radio" name="agg" value="sum" {{ "checked" if agg_rule == "sum" }}> Sum</label>
        </div>
        <div class="control-group">
            <label for="n"><strong>Show:</strong></label>
            <select name="n" id="n">
                <option value="10" {{ "selected" if n_show == 10 }}>10</option>
                <option value="20" {{ "selected" if n_show == 20 }}>20</option>
                <option value="50" {{ "selected" if n_show == 50 }}>50</option>
            </select>
        </div>
        <button type="submit">Update</button>
    </form>

    <!-- Recommended referees -->
    <h2>Recommended Referees</h2>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Author</th>
                <th>Score</th>
                <th>Papers in top 50</th>
                <th>Justifying papers</th>
            </tr>
        </thead>
        <tbody>
            {% for ref in referees %}
            <tr>
                <td>{{ loop.index }}</td>
                <td>{{ ref.author }}</td>
                <td>{{ "%.4f" | format(ref.score) }}</td>
                <td>{{ ref.n_papers }}</td>
                <td class="paper-links">
                    {% for p in ref.papers[:5] %}
                    <a href="https://arxiv.org/abs/{{ p.arxiv_id }}" target="_blank"
                       title="{{ p.title }} ({{ "%.4f" | format(p.sim) }})">
                       {{ p.arxiv_id }}</a>
                    {% endfor %}
                    {% if ref.papers | length > 5 %}
                    <span class="meta">+{{ ref.papers | length - 5 }} more</span>
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>

    <!-- Similar papers -->
    <h2>Most Similar Papers</h2>
    <div class="paper-list">
        {% for p in similar_papers %}
        <div class="paper-item">
            <span class="sim-score">{{ "%.4f" | format(p.sim) }}</span>
            <a class="title" href="{{ p.url }}" target="_blank">{{ p.title }}</a>
            <p class="meta">{{ p.authors }}</p>
            <p class="meta">Published: {{ p.published }}</p>
            <details>
                <summary class="expand-toggle">Show abstract</summary>
                <p class="abstract">{{ p.abstract }}</p>
            </details>
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Custom Jinja2 filter
# ---------------------------------------------------------------------------

@app.template_filter("number_format")
def number_format(value):
    """Format an integer with comma separators (e.g. 5099 → '5,099')."""
    return f"{int(value):,}"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
