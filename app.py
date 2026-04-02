"""
Netflix Recommendation System — Streamlit UI
Run: streamlit run app.py
Requirements: pip install streamlit joblib scikit-learn pandas numpy
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import re

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark cinematic Netflix-inspired aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0a0a !important;
    color: #e8e8e8;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111111 0%, #0a0a0a 100%) !important;
    border-right: 1px solid #1e1e1e;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}
.hero-logo {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5rem;
    letter-spacing: 0.08em;
    line-height: 1;
    background: linear-gradient(135deg, #E50914 0%, #ff4444 50%, #E50914 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none;
    margin: 0;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #888;
    margin-top: 0.3rem;
}
.hero-divider {
    width: 60px;
    height: 2px;
    background: #E50914;
    margin: 1.5rem auto;
}

/* ── Search box ── */
.stTextInput > div > div > input {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px !important;
    color: #f0f0f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s;
}
.stTextInput > div > div > input:focus {
    border-color: #E50914 !important;
    box-shadow: 0 0 0 2px rgba(229,9,20,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #E50914 !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background: #ff1a1a !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(229,9,20,0.4) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px !important;
    color: #f0f0f0 !important;
}
.stSelectbox > div > div:focus-within {
    border-color: #E50914 !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
    background: #E50914 !important;
    color: white !important;
}
.stSlider [data-baseweb="slider"] > div > div > div {
    background: #E50914 !important;
}

/* ── Card ── */
.rec-card {
    background: #141414;
    border: 1px solid #222;
    border-radius: 6px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.rec-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: #E50914;
    border-radius: 3px 0 0 3px;
}
.rec-card:hover {
    border-color: #333;
    background: #1a1a1a;
    transform: translateX(2px);
}
.card-rank {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.5rem;
    color: #222;
    position: absolute;
    right: 1.2rem;
    top: 0.6rem;
    line-height: 1;
}
.card-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #f0f0f0;
    margin-bottom: 0.3rem;
    padding-right: 3rem;
}
.card-meta {
    font-size: 0.8rem;
    color: #666;
    font-weight: 300;
    letter-spacing: 0.02em;
    margin-bottom: 0.5rem;
}
.card-genre {
    font-size: 0.75rem;
    color: #999;
    margin-bottom: 0.5rem;
}
.card-desc {
    font-size: 0.82rem;
    color: #777;
    font-weight: 300;
    line-height: 1.5;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.score-bar-wrap {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.6rem;
}
.score-bar-bg {
    flex: 1;
    height: 3px;
    background: #2a2a2a;
    border-radius: 2px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #E50914, #ff6b6b);
    border-radius: 2px;
    transition: width 0.5s ease;
}
.score-label {
    font-size: 0.72rem;
    color: #E50914;
    font-weight: 600;
    white-space: nowrap;
}

/* ── Badge ── */
.badge {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 3px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-right: 0.3rem;
}
.badge-movie   { background: rgba(229,9,20,0.15); color: #E50914; border: 1px solid rgba(229,9,20,0.3); }
.badge-tv      { background: rgba(30,100,255,0.15); color: #5b9cff; border: 1px solid rgba(30,100,255,0.3); }
.badge-rating  { background: rgba(255,255,255,0.06); color: #aaa; border: 1px solid #2a2a2a; }
.badge-year    { background: rgba(255,255,255,0.04); color: #888; border: 1px solid #222; }

/* ── Stats row ── */
.stat-box {
    background: #141414;
    border: 1px solid #1e1e1e;
    border-radius: 6px;
    padding: 1rem;
    text-align: center;
}
.stat-num {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: #E50914;
    line-height: 1;
}
.stat-label {
    font-size: 0.72rem;
    color: #555;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* ── Input title card ── */
.input-card {
    background: linear-gradient(135deg, #1a0505 0%, #141414 100%);
    border: 1px solid #2e1010;
    border-radius: 6px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.5rem;
}
.input-title-text {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: #f0f0f0;
    letter-spacing: 0.05em;
    margin-bottom: 0.2rem;
}
.input-type-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    background: rgba(229,9,20,0.2);
    border: 1px solid rgba(229,9,20,0.4);
    border-radius: 3px;
    color: #E50914;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-right: 0.5rem;
}

/* ── Section header ── */
.section-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    letter-spacing: 0.1em;
    color: #f0f0f0;
    margin-bottom: 0.3rem;
}
.section-sub {
    font-size: 0.78rem;
    color: #555;
    letter-spacing: 0.05em;
    margin-bottom: 1.2rem;
}

/* ── Autocomplete suggestions ── */
.suggestion-item {
    background: #181818;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 0.5rem 0.8rem;
    margin-bottom: 0.3rem;
    cursor: pointer;
    font-size: 0.88rem;
    color: #ccc;
    transition: all 0.15s;
}
.suggestion-item:hover {
    background: #222;
    border-color: #E50914;
    color: #f0f0f0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #111; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #E50914; }

/* ── Sidebar labels ── */
.stSidebar label, .stSidebar .stSelectbox label,
.stSidebar .stSlider label {
    color: #888 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}
.sidebar-section {
    border-top: 1px solid #1e1e1e;
    padding-top: 1rem;
    margin-top: 1rem;
}
.sidebar-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1rem;
    letter-spacing: 0.15em;
    color: #555;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* ── No results ── */
.no-result {
    text-align: center;
    padding: 3rem 1rem;
    color: #444;
}
.no-result-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.no-result-text { font-size: 1.1rem; color: #555; }
.no-result-hint { font-size: 0.82rem; color: #3a3a3a; margin-top: 0.5rem; }

/* Hide streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD PKL ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load all PKL artifacts. Returns None values if files not found."""
    base = "models"
    artifacts = {}

    # Try to load cosine similarity matrix
    cos_path = os.path.join(base, "cosine_similarity.pkl")
    if os.path.exists(cos_path):
        artifacts["cosine_sim"] = joblib.load(cos_path)
    else:
        artifacts["cosine_sim"] = None

    # Title index map
    idx_path = os.path.join(base, "title_indices.pkl")
    if os.path.exists(idx_path):
        artifacts["indices"] = joblib.load(idx_path)
    else:
        artifacts["indices"] = None

    # Cleaned DataFrame
    df_path = os.path.join(base, "netflix_df.pkl")
    if os.path.exists(df_path):
        artifacts["df"] = joblib.load(df_path)
    else:
        # Fallback: try loading the cleaned CSV
        csv_path = "netflix_cleaned.csv"
        if os.path.exists(csv_path):
            artifacts["df"] = pd.read_csv(csv_path)
        else:
            artifacts["df"] = None

    # TF-IDF vectorizer (optional, for new query support)
    tfidf_path = os.path.join(base, "tfidf_vectorizer.pkl")
    if os.path.exists(tfidf_path):
        artifacts["tfidf"] = joblib.load(tfidf_path)
    else:
        artifacts["tfidf"] = None

    return artifacts


def recommend(title: str, n: int, filter_type: str, min_year: int,
              max_year: int, cosine_sim, indices, df) -> pd.DataFrame | None:
    """
    Core recommendation function.
    Parameters:
        title       — input Netflix title (case-insensitive)
        n           — number of results
        filter_type — 'All', 'Movie', or 'TV Show'
        min_year    — minimum release_year filter
        max_year    — maximum release_year filter
        cosine_sim  — preloaded similarity matrix
        indices     — title→index Series
        df          — cleaned Netflix DataFrame
    """
    t = title.strip().lower()

    if t not in indices.index:
        return None

    idx = indices[t]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx]

    top_50_idx = [s[0] for s in sim_scores[:100]]
    scores_map = {s[0]: round(float(s[1]), 4) for s in sim_scores[:100]}

    recs = df.iloc[top_50_idx].copy()
    recs["similarity"] = recs.index.map(scores_map)

    # Filters
    if filter_type != "All":
        recs = recs[recs["type"] == filter_type]

    if "release_year" in recs.columns:
        recs = recs[
            (recs["release_year"] >= min_year) &
            (recs["release_year"] <= max_year)
        ]

    return recs.head(n).reset_index(drop=True)


def fuzzy_suggest(query: str, indices, max_results: int = 6) -> list:
    """Return title suggestions matching the query string."""
    q = query.strip().lower()
    if len(q) < 2:
        return []
    return [t for t in indices.index if q in t.lower()][:max_results]


def get_badges(row) -> str:
    """Generate HTML badge string for a result row."""
    type_class = "badge-movie" if str(row.get("type", "")) == "Movie" else "badge-tv"
    type_label = str(row.get("type", "N/A"))
    rating = str(row.get("rating", ""))
    year = str(int(row["release_year"])) if pd.notna(row.get("release_year")) else ""
    badges = f'<span class="badge {type_class}">{type_label}</span>'
    if rating and rating != "nan":
        badges += f'<span class="badge badge-rating">{rating}</span>'
    if year:
        badges += f'<span class="badge badge-year">{year}</span>'
    return badges


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    arts = load_artifacts()
    cosine_sim = arts["cosine_sim"]
    indices    = arts["indices"]
    df         = arts["df"]

    ready = all(v is not None for v in [cosine_sim, indices, df])

    # ── HERO ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-logo">NETFLIXREC</div>
        <div class="hero-sub">Content-Based Recommendation Engine</div>
        <div class="hero-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── DATASET NOT LOADED WARNING ────────────────────────────────────────────
    if not ready:
        st.error("⚠️  PKL files not found. Make sure the `models/` folder with all `.pkl` files is in the same directory as `app.py`.")
        st.markdown("""
        **Expected files:**
        ```
        models/
          cosine_similarity.pkl
          title_indices.pkl
          netflix_df.pkl
          tfidf_vectorizer.pkl
        ```
        Run the Jupyter notebook first to generate these files.
        """)
        return

    # ── STATS BAR ─────────────────────────────────────────────────────────────
    total     = len(df)
    n_movies  = int((df["type"] == "Movie").sum()) if "type" in df.columns else 0
    n_tv      = int((df["type"] == "TV Show").sum()) if "type" in df.columns else 0
    n_genres  = df["primary_genre"].nunique() if "primary_genre" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    for col, num, label in zip(
        [c1, c2, c3, c4],
        [total, n_movies, n_tv, n_genres],
        ["TOTAL TITLES", "MOVIES", "TV SHOWS", "GENRES"]
    ):
        col.markdown(f"""
        <div class="stat-box">
            <div class="stat-num">{num:,}</div>
            <div class="stat-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem 0 0.5rem;">
            <div style="font-family:'Bebas Neue',sans-serif; font-size:1.6rem;
                        letter-spacing:0.1em; color:#E50914;">FILTERS</div>
            <div style="font-size:0.72rem; color:#444; letter-spacing:0.1em;
                        text-transform:uppercase; margin-top:0.1rem;">Refine your results</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">Content Type</div>', unsafe_allow_html=True)
        filter_type = st.selectbox(
            "Content Type",
            options=["All", "Movie", "TV Show"],
            index=0,
            label_visibility="collapsed"
        )

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">Number of Results</div>', unsafe_allow_html=True)
        n_results = st.slider("Results", min_value=5, max_value=20, value=10, step=1,
                              label_visibility="collapsed")

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">Release Year Range</div>', unsafe_allow_html=True)
        year_min = int(df["release_year"].min()) if "release_year" in df.columns else 1950
        year_max = int(df["release_year"].max()) if "release_year" in df.columns else 2021
        year_range = st.slider(
            "Year Range", min_value=year_min, max_value=year_max,
            value=(year_min, year_max), step=1,
            label_visibility="collapsed"
        )

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">About</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.78rem; color:#555; line-height:1.6;">
            Recommendations powered by<br>
            <span style="color:#E50914;">TF-IDF + Cosine Similarity</span><br><br>
            Input any Netflix title to find<br>
            the most similar content in the<br>
            catalog based on description,<br>
            genre, cast, and director.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── SEARCH AREA ───────────────────────────────────────────────────────────
    search_col, btn_col = st.columns([4, 1])

    with search_col:
        query = st.text_input(
            "Search",
            placeholder="Search a Netflix title... e.g. Breaking Bad, Inception, Dark",
            label_visibility="collapsed",
            key="search_input"
        )

    with btn_col:
        search_clicked = st.button("🔍  Find", use_container_width=True)

    # ── AUTOCOMPLETE SUGGESTIONS ──────────────────────────────────────────────
    if query and len(query) >= 2 and not search_clicked:
        suggestions = fuzzy_suggest(query, indices, max_results=5)
        if suggestions:
            st.markdown(
                f'<div style="font-size:0.72rem; color:#555; '
                f'letter-spacing:0.08em; text-transform:uppercase; '
                f'margin-bottom:0.5rem;">Suggestions</div>',
                unsafe_allow_html=True
            )
            sug_cols = st.columns(min(len(suggestions), 5))
            for i, sug in enumerate(suggestions):
                with sug_cols[i % len(sug_cols)]:
                    if st.button(sug.title(), key=f"sug_{i}"):
                        query = sug
                        search_clicked = True

    # ── RESULTS ───────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    if search_clicked and query:
        with st.spinner(""):
            results = recommend(
                title=query,
                n=n_results,
                filter_type=filter_type,
                min_year=year_range[0],
                max_year=year_range[1],
                cosine_sim=cosine_sim,
                indices=indices,
                df=df
            )

        if results is None:
            # Not found
            suggestions = fuzzy_suggest(query, indices, max_results=4)
            st.markdown(f"""
            <div class="no-result">
                <div class="no-result-icon">🎬</div>
                <div class="no-result-text">"{query}" not found in catalog</div>
                <div class="no-result-hint">Check the spelling or try a suggestion below</div>
            </div>
            """, unsafe_allow_html=True)
            if suggestions:
                st.markdown(
                    '<div style="text-align:center; color:#555; '
                    'font-size:0.78rem; letter-spacing:0.1em; '
                    'text-transform:uppercase; margin-bottom:0.5rem;">'
                    'Did you mean?</div>',
                    unsafe_allow_html=True
                )
                sc = st.columns(min(len(suggestions), 4))
                for i, sug in enumerate(suggestions):
                    with sc[i % len(sc)]:
                        if st.button(sug.title(), key=f"fallback_{i}"):
                            st.session_state["search_input"] = sug
                            st.rerun()

        elif len(results) == 0:
            st.markdown("""
            <div class="no-result">
                <div class="no-result-icon">🔎</div>
                <div class="no-result-text">No results match your filters</div>
                <div class="no-result-hint">Try loosening the year range or content type filter</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # ── INPUT TITLE INFO ──────────────────────────────────────────────
            t_lower = query.strip().lower()
            if t_lower in indices.index:
                src_idx = indices[t_lower]
                if isinstance(src_idx, pd.Series):
                    src_idx = src_idx.iloc[0]
                src_row = df.iloc[src_idx]
                src_desc = str(src_row.get("description", ""))[:160] + "..."
                src_genre = str(src_row.get("listed_in", "")).split(",")[0].strip()
                src_type  = str(src_row.get("type", ""))
                src_year  = str(int(src_row["release_year"])) if pd.notna(src_row.get("release_year")) else ""

                type_badge_class = "input-type-badge"
                st.markdown(f"""
                <div class="input-card">
                    <div style="font-size:0.68rem; color:#555; letter-spacing:0.15em;
                                text-transform:uppercase; margin-bottom:0.5rem;">
                        Because you searched for
                    </div>
                    <div class="input-title-text">{src_row.get('title','').upper()}</div>
                    <div style="margin:0.4rem 0 0.7rem;">
                        <span class="{type_badge_class}">{src_type}</span>
                        <span style="font-size:0.78rem; color:#555;">{src_genre} · {src_year}</span>
                    </div>
                    <div style="font-size:0.82rem; color:#666; font-weight:300;
                                line-height:1.5; font-style:italic;">
                        {src_desc}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── RESULTS HEADER ────────────────────────────────────────────────
            st.markdown(f"""
            <div class="section-header">RECOMMENDATIONS</div>
            <div class="section-sub">{len(results)} titles · sorted by similarity score</div>
            """, unsafe_allow_html=True)

            # ── RESULT CARDS ──────────────────────────────────────────────────
            for i, row in results.iterrows():
                title_str   = str(row.get("title", "Unknown"))
                genre_str   = str(row.get("listed_in", ""))[:80]
                desc_str    = str(row.get("description", ""))[:200]
                score       = float(row.get("similarity", 0))
                score_pct   = round(score * 100, 1)
                score_bar   = round(score * 100, 1)
                badges_html = get_badges(row)

                st.markdown(f"""
                <div class="rec-card">
                    <div class="card-rank">#{i+1:02d}</div>
                    <div class="card-title">{title_str}</div>
                    <div class="card-meta">{badges_html}</div>
                    <div class="card-genre">📂 {genre_str}</div>
                    <div class="card-desc">{desc_str}</div>
                    <div class="score-bar-wrap">
                        <div class="score-bar-bg">
                            <div class="score-bar-fill" style="width:{score_bar}%"></div>
                        </div>
                        <div class="score-label">{score_pct}% match</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    elif not query:
        # ── DEFAULT STATE — Popular Titles ───────────────────────────────────
        st.markdown("""
        <div class="section-header">POPULAR SEARCHES</div>
        <div class="section-sub">Try one of these to get started</div>
        """, unsafe_allow_html=True)

        popular = []
        if df is not None:
            # Show a sample from different genres
            for gtype in ["Movie", "TV Show"]:
                sample = df[df["type"] == gtype].head(4)["title"].tolist() if "type" in df.columns else []
                popular.extend(sample)
        popular = popular[:8]

        if popular:
            rows = [popular[:4], popular[4:8]]
            for row_titles in rows:
                cols = st.columns(4)
                for j, t in enumerate(row_titles):
                    with cols[j]:
                        if st.button(t, key=f"pop_{t}"):
                            st.session_state["search_input"] = t
                            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
