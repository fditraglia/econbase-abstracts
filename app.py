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

import bisect
from collections import defaultdict
import os
import re
from flask import Flask, jsonify, render_template_string, request, redirect, url_for
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load data on startup
# ---------------------------------------------------------------------------
# The app only ever ranks papers against a paper already in the corpus, so it
# needs each paper's closest neighbors and not the 4096-dimensional vectors they
# were computed from. Those live in gs://econbase-embeddings and are used for
# analysis; build_similarity_table.py turns them into the small table read here.
# Storing the top 100 makes this exact rather than approximate: the interface can
# ask for at most max(50 * 2, 50) = 100 neighbors.
#
# R analogy: like loading an .rds file in global.R for a Shiny app — it
# runs once when the app starts, not on every request.

print("Loading papers...", flush=True)
df = pd.read_parquet(os.path.join(BASE_DIR, "econ_em_papers.parquet"))

# Lookup dict: arxiv_id string → row index
id_to_idx = {aid: i for i, aid in enumerate(df["arxiv_id"])}

# A revision changes a paper's identifier (2405.06779v3 becomes v4), so a bookmarked
# link would break on every refresh. Accept the version-stripped form as well.
base_to_idx = {}
for aid, i in id_to_idx.items():
    base_to_idx.setdefault(re.sub(r"v\d+$", "", aid), i)

N_PAPERS = len(df)

print("Loading the similarity table...", flush=True)
_sim = pd.read_parquet(os.path.join(BASE_DIR, "similarity_top100.parquet"))
_sim_order = {aid: i for i, aid in enumerate(_sim["arxiv_id"])}
NEIGHBOR_IDX = np.array(
    [[id_to_idx[n] for n in row] for row in _sim["neighbor_ids"]], dtype=np.int32)
NEIGHBOR_SIM = np.array(_sim["cosines"].tolist(), dtype=np.float32)
TABLE_DEPTH = NEIGHBOR_IDX.shape[1]
del _sim
print(f"Loaded {N_PAPERS} papers, top-{TABLE_DEPTH} neighbors each", flush=True)

# Pre-sorted IDs for O(log n) prefix matching in autocomplete
sorted_ids = sorted(df["arxiv_id"].tolist())

# Co-author graph: author → set of all co-authors across the dataset
# R analogy: like building an adjacency list from an edge list
coauthor_graph = defaultdict(set)
for _, row in df.iterrows():
    authors = [a.strip() for a in row["authors"].split(",")]
    for i, a in enumerate(authors):
        for b in authors[i + 1:]:
            coauthor_graph[a].add(b)
            coauthor_graph[b].add(a)
print(f"Built co-author graph: {len(coauthor_graph)} authors", flush=True)

# The corpus date is read off the data, never written down. A hardcoded date is
# right until the first refresh and wrong silently ever after -- this one said
# 19 February 2026 for the six months the corpus was already past it.
CORPUS_THROUGH = pd.to_datetime(df["published"]).max().strftime("%B %-d, %Y")
app.jinja_env.globals["corpus_through"] = CORPUS_THROUGH
print(f"Corpus runs through {CORPUS_THROUGH}", flush=True)


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------

def resolve(query):
    """Map a user-supplied identifier to the one we hold, or None.

    Accepts '2405.06779v4', '2405.06779', or a stale version such as
    '2405.06779v3', all of which name the same paper.
    """
    q = (query or "").strip()
    if q in id_to_idx:
        return q
    i = base_to_idx.get(re.sub(r"v\d+$", "", q))
    return None if i is None else df["arxiv_id"].iloc[i]


def find_similar(arxiv_id, top_k=50):
    """Return indices and cosine similarities of the top-k most similar papers.

    A lookup rather than a computation. The table is already sorted by descending
    similarity and excludes the paper itself, so this is a slice.
    """
    row = _sim_order[arxiv_id]
    if top_k > TABLE_DEPTH:
        # Silently returning a short list would look like a corpus with few close
        # papers rather than a table that needs rebuilding deeper.
        print(f"WARNING: asked for {top_k} neighbors but the table holds "
              f"{TABLE_DEPTH}; rebuild with a larger TOP_K", flush=True)
        top_k = TABLE_DEPTH
    return NEIGHBOR_IDX[row, :top_k], NEIGHBOR_SIM[row, :top_k]


def aggregate_referees(arxiv_id, agg_rule="mean-top-3", top_k_papers=50,
                       exclude_coauthors=True):
    """
    Aggregate author scores from the top-k most similar papers.

    Each similar paper contributes its cosine similarity score to all of its
    authors. Then we aggregate per-author using the chosen rule and remove
    the query paper's own authors (and optionally their co-authors).

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

    # Build exclusion set: query authors + optionally all their co-authors
    query_row = df.iloc[id_to_idx[arxiv_id]]
    query_authors = {a.strip() for a in query_row["authors"].split(",")}
    if exclude_coauthors:
        excluded = set(query_authors)
        for a in query_authors:
            excluded |= coauthor_graph.get(a, set())
    else:
        excluded = query_authors

    results = []
    for author, scores in author_scores.items():
        if author in excluded:
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
        query = request.form.get("arxiv_id", "").strip()
        if query:
            # If it names a paper we hold, go straight to recommendations
            canonical = resolve(query)
            if canonical:
                return redirect(url_for("recommend", arxiv_id=canonical))
            # Otherwise, search
            return redirect(url_for("search", q=query))
    return render_template_string(HOME_TEMPLATE, n_papers=N_PAPERS)


@app.route("/recommend/<path:arxiv_id>")
def recommend(arxiv_id):
    canonical = resolve(arxiv_id)
    if canonical is None:
        return render_template_string(ERROR_TEMPLATE, arxiv_id=arxiv_id.strip(),
                                      n_papers=N_PAPERS)
    if canonical != arxiv_id.strip():
        # A link made before the paper was revised. Send it to the current version
        # rather than failing, and keep any options the user had set.
        return redirect(url_for("recommend", arxiv_id=canonical, **request.args))
    arxiv_id = canonical

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

    # Co-author exclusion: default on (first load has no query params)
    # HTML checkboxes don't send a value when unchecked, so we check
    # whether *any* query params were sent to distinguish first load
    # from an explicit "unchecked" submission.
    if request.args:
        excl_coauth = request.args.get("excl_coauth") is not None
    else:
        excl_coauth = True  # default on first load

    # Get query paper info
    query_row = df.iloc[id_to_idx[arxiv_id]]

    # Get referees
    referees = aggregate_referees(arxiv_id, agg_rule=agg_rule,
                                  top_k_papers=50,
                                  exclude_coauthors=excl_coauth)

    # Get similar papers for display — fetch extra so we can filter
    top_indices, top_sims = find_similar(arxiv_id, top_k=max(n_show * 2, 50))

    # Build excluded author set for filtering similar papers
    if excl_coauth:
        query_authors = {a.strip() for a in query_row["authors"].split(",")}
        excluded_authors = set(query_authors)
        for a in query_authors:
            excluded_authors |= coauthor_graph.get(a, set())
    else:
        excluded_authors = None

    similar_papers = []
    for idx, sim in zip(top_indices, top_sims):
        if len(similar_papers) >= n_show:
            break
        row = df.iloc[idx]
        # When co-author exclusion is on, skip papers where every author
        # is in the excluded set
        if excluded_authors is not None:
            paper_authors = {a.strip() for a in row["authors"].split(",")}
            if paper_authors <= excluded_authors:
                continue
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
        excl_coauth=excl_coauth,
    )


@app.route("/api/autocomplete")
def autocomplete():
    """Return up to 10 matching papers as JSON for the autocomplete dropdown."""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])

    results = []
    if q[0].isdigit():
        # arXiv ID prefix match using bisect for O(log n) lookup
        lo = bisect.bisect_left(sorted_ids, q)
        for i in range(lo, min(lo + 10, len(sorted_ids))):
            if not sorted_ids[i].startswith(q):
                break
            aid = sorted_ids[i]
            row = df.iloc[id_to_idx[aid]]
            results.append({
                "arxiv_id": aid,
                "title": row["title"],
                "authors": row["authors"],
            })
    else:
        # Case-insensitive substring match on title and authors
        q_lower = q.lower()
        for _, row in df.iterrows():
            if (q_lower in row["title"].lower()
                    or q_lower in row["authors"].lower()):
                results.append({
                    "arxiv_id": row["arxiv_id"],
                    "title": row["title"],
                    "authors": row["authors"],
                })
                if len(results) >= 10:
                    break

    return jsonify(results)


@app.route("/search")
def search():
    """Search results page — show all papers matching the query."""
    q = request.args.get("q", "").strip()
    if not q:
        return redirect(url_for("home"))

    results = []
    if q[0].isdigit():
        lo = bisect.bisect_left(sorted_ids, q)
        for i in range(lo, len(sorted_ids)):
            if not sorted_ids[i].startswith(q):
                break
            aid = sorted_ids[i]
            row = df.iloc[id_to_idx[aid]]
            results.append({
                "arxiv_id": aid,
                "title": row["title"],
                "authors": row["authors"],
                "published": str(row["published"]),
            })
    else:
        q_lower = q.lower()
        for _, row in df.iterrows():
            if (q_lower in row["title"].lower()
                    or q_lower in row["authors"].lower()):
                results.append({
                    "arxiv_id": row["arxiv_id"],
                    "title": row["title"],
                    "authors": row["authors"],
                    "published": str(row["published"]),
                })

    return render_template_string(SEARCH_TEMPLATE, q=q, results=results,
                                  n_papers=N_PAPERS)


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
        width: 100%;
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
    .search-wrap { position: relative; flex: 1; }
    .autocomplete-list {
        position: absolute;
        top: 100%;
        left: 0;
        right: 0;
        background: #fff;
        border: 1px solid #ccc;
        border-top: none;
        border-radius: 0 0 6px 6px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        z-index: 100;
        max-height: 400px;
        overflow-y: auto;
    }
    .autocomplete-item {
        padding: 8px 14px;
        cursor: pointer;
        border-bottom: 1px solid #eee;
    }
    .autocomplete-item:last-child { border-bottom: none; }
    .autocomplete-item:hover, .autocomplete-item.active {
        background: #e0edff;
    }
    .autocomplete-item .ac-title { font-weight: 600; font-size: 0.9em; }
    .autocomplete-item .ac-meta { font-size: 0.8em; color: #666; }
    .no-results { padding: 20px; text-align: center; color: #666; }
</style>
"""

AUTOCOMPLETE_JS = """
<script>
(function() {
    const input = document.querySelector('.search-form input[type="text"]');
    const wrap = input.parentElement;
    let dropdown = null;
    let activeIdx = -1;
    let debounceTimer = null;
    let items = [];

    function createDropdown() {
        if (dropdown) return dropdown;
        dropdown = document.createElement('div');
        dropdown.className = 'autocomplete-list';
        wrap.appendChild(dropdown);
        return dropdown;
    }

    function closeDropdown() {
        if (dropdown) { dropdown.remove(); dropdown = null; }
        activeIdx = -1;
        items = [];
    }

    function renderItems(data) {
        if (!data.length) { closeDropdown(); return; }
        const dd = createDropdown();
        dd.innerHTML = data.map((d, i) =>
            `<div class="autocomplete-item" data-idx="${i}" data-id="${d.arxiv_id}">
                <div class="ac-title">${escHtml(d.title)}</div>
                <div class="ac-meta">${escHtml(d.authors)} &middot; ${d.arxiv_id}</div>
            </div>`
        ).join('');
        items = dd.querySelectorAll('.autocomplete-item');
        activeIdx = -1;
        items.forEach(item => {
            item.addEventListener('mousedown', e => {
                e.preventDefault();
                window.location.href = '/recommend/' + item.dataset.id;
            });
        });
    }

    function setActive(idx) {
        items.forEach(el => el.classList.remove('active'));
        activeIdx = idx;
        if (idx >= 0 && idx < items.length) {
            items[idx].classList.add('active');
            items[idx].scrollIntoView({block: 'nearest'});
        }
    }

    function escHtml(s) {
        const d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }

    input.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        const q = input.value.trim();
        if (!q) { closeDropdown(); return; }
        debounceTimer = setTimeout(() => {
            fetch('/api/autocomplete?q=' + encodeURIComponent(q))
                .then(r => r.json())
                .then(renderItems)
                .catch(() => closeDropdown());
        }, 150);
    });

    input.addEventListener('keydown', e => {
        if (!dropdown) return;
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            setActive(Math.min(activeIdx + 1, items.length - 1));
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            setActive(Math.max(activeIdx - 1, -1));
        } else if (e.key === 'Enter' && activeIdx >= 0) {
            e.preventDefault();
            window.location.href = '/recommend/' + items[activeIdx].dataset.id;
        } else if (e.key === 'Escape') {
            closeDropdown();
        }
    });

    document.addEventListener('click', e => {
        if (!wrap.contains(e.target)) closeDropdown();
    });
})();
</script>
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
    <p class="subtitle">{{ n_papers | number_format }} econ.EM papers from arXiv through {{ corpus_through }}</p>
    <form class="search-form" method="post">
        <div class="search-wrap">
            <input type="text" name="arxiv_id"
                   placeholder="Search by arXiv ID, title, or author name"
                   autofocus autocomplete="off">
        </div>
        <button type="submit">Find referees</button>
    </form>
    """ + AUTOCOMPLETE_JS + """
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
    <p class="subtitle">{{ n_papers | number_format }} econ.EM papers from arXiv through {{ corpus_through }}</p>
    <div class="card">
        <p class="error">arXiv ID "{{ arxiv_id }}" not found in the database.</p>
        <p>Make sure the ID matches a paper in the econ.EM category (e.g. 2301.12345).</p>
    </div>
</body>
</html>
"""

SEARCH_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Search: {{ q }} — Referee Recommender</title>
    """ + STYLE + """
</head>
<body>
    <a class="home-link" href="/">&larr; Back to search</a>
    <h1>Referee Recommender</h1>
    <p class="subtitle">{{ n_papers | number_format }} econ.EM papers from arXiv through {{ corpus_through }}</p>

    <form class="search-form" method="post" action="/">
        <div class="search-wrap">
            <input type="text" name="arxiv_id" value="{{ q }}"
                   placeholder="Search by arXiv ID, title, or author name"
                   autofocus autocomplete="off">
        </div>
        <button type="submit">Find referees</button>
    </form>
    """ + AUTOCOMPLETE_JS + """

    <h2>Search results for "{{ q }}"</h2>
    {% if results %}
    <div class="paper-list">
        {% for p in results %}
        <div class="paper-item">
            <a class="title" href="/recommend/{{ p.arxiv_id }}">{{ p.title }}</a>
            <p class="meta">{{ p.authors }}</p>
            <p class="meta">Published: {{ p.published }} &middot; arXiv:{{ p.arxiv_id }}</p>
        </div>
        {% endfor %}
    </div>
    {% else %}
    <div class="no-results">
        <p>No papers found matching "{{ q }}".</p>
        <p>Try searching by arXiv ID (e.g. 2301.12345), paper title, or author name.</p>
    </div>
    {% endif %}
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
    <p class="subtitle">{{ n_papers | number_format }} econ.EM papers from arXiv through {{ corpus_through }}</p>

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
        <div class="control-group">
            <label>
                <input type="checkbox" name="excl_coauth" value="1"
                       {{ "checked" if excl_coauth }}>
                <strong>Exclude co-authors</strong>
            </label>
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
    <h2>Most Similar Papers{{ " (excluding co-authored)" if excl_coauth }}</h2>
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
