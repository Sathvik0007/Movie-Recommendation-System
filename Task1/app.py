"""
CineMatch — Movie Recommendation System (Streamlit UI)
"""

import streamlit as st
import pandas as pd
from data.download_data import download_dataset
from recommender import (
    load_data, ContentBasedEngine, CollaborativeEngine, SentimentEngine
)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

# ── Download dataset if needed ───────────────────────────────────────────────
with st.spinner("Preparing dataset (one-time download)..."):
    download_dataset()

# ── Load data & build engines (cached) ───────────────────────────────────────
@st.cache_resource
def init_engines():
    movies, ratings, tags = load_data()
    cb = ContentBasedEngine(movies)
    cf = CollaborativeEngine(ratings, movies)
    se = SentimentEngine(tags, movies, ratings)
    avg_ratings = ratings.groupby("movieId")["rating"].mean()
    return movies, ratings, tags, cb, cf, se, avg_ratings

movies, ratings, tags, cb_engine, cf_engine, sent_engine, avg_ratings = init_engines()

# ── Session state for user ratings ───────────────────────────────────────────
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🎬 CineMatch")
st.sidebar.markdown("**Movie Recommendation System**")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Mode",
    ["🆕 New User (rate movies)", "👤 Existing User (from dataset)"],
)

existing_user_id = None
if "Existing" in mode:
    user_ids = sorted(ratings["userId"].unique())
    existing_user_id = st.sidebar.selectbox("Select User ID", user_ids)
    # Load that user's ratings
    user_df = ratings[ratings["userId"] == existing_user_id]
    st.session_state.user_ratings = dict(
        zip(user_df["movieId"], user_df["rating"])
    )

st.sidebar.markdown("---")
st.sidebar.metric("Movies Rated", len(st.session_state.user_ratings))

if st.sidebar.button("🗑️ Clear My Ratings"):
    st.session_state.user_ratings = {}
    st.rerun()

# ── Helper ───────────────────────────────────────────────────────────────────
def display_movies(df, show_score=False, allow_rating=False):
    """Render a movie table with optional rating widgets."""
    if df.empty:
        st.info("No movies to display.")
        return
    cols = st.columns(min(5, len(df)))
    for i, (_, row) in enumerate(df.head(10).iterrows()):
        with cols[i % 5]:
            avg = avg_ratings.get(row["movieId"], 0)
            sentiment = sent_engine.get_sentiment(row["movieId"])
            sent_label = "🟢" if sentiment > 0.2 else ("🔴" if sentiment < -0.2 else "🟡")

            st.markdown(f"**{row['title']}**")
            st.caption(f"🎭 {row['genres']}")
            st.caption(f"⭐ {avg:.1f}/5 &nbsp; {sent_label} Sent: {sentiment:+.2f}")

            if show_score and "score" in row and pd.notna(row.get("score")):
                st.caption(f"📊 Match: {row['score']:.2f}")

            if allow_rating:
                key = f"rate_{row['movieId']}"
                current = st.session_state.user_ratings.get(row["movieId"], 0)
                r = st.slider(
                    "Rate", 1, 5, value=int(current) if current else 3,
                    key=key, label_visibility="collapsed"
                )
                if st.button("✓", key=f"btn_{row['movieId']}"):
                    st.session_state.user_ratings[row["movieId"]] = r
                    st.rerun()

# ── Main Tabs ────────────────────────────────────────────────────────────────
tab_browse, tab_recs, tab_rated = st.tabs(
    ["🍿 Browse & Rate", "✨ Recommendations", "⭐ My Ratings"]
)

# ── TAB 1: Browse & Rate ────────────────────────────────────────────────────
with tab_browse:
    st.header("Browse Movies")
    st.caption("Search and rate movies to get personalized recommendations.")

    all_genres = sorted(
        set(g for genres in movies["genres"] for g in genres.split("|") if g != "(no genres listed)")
    )
    selected_genre = st.selectbox("Filter by Genre", ["All"] + all_genres)

    search = st.text_input("🔍 Search by title")

    filtered = movies.copy()
    if selected_genre != "All":
        filtered = filtered[filtered["genres"].str.contains(selected_genre, case=False)]
    if search:
        filtered = filtered[filtered["title"].str.contains(search, case=False, na=False)]

    st.markdown(f"Showing **{min(10, len(filtered))}** of **{len(filtered)}** movies")
    display_movies(filtered.head(10), allow_rating=True)

# ── TAB 2: Recommendations ──────────────────────────────────────────────────
with tab_recs:
    st.header("Recommendations For You")
    ur = st.session_state.user_ratings

    if len(ur) < 3:
        st.warning("⚠️ Rate at least **3 movies** to unlock recommendations.")
    else:
        # Content-Based
        st.subheader("🎯 Content-Based Filtering")
        st.caption("Movies similar in genre to your top-rated films (TF-IDF + Cosine Similarity).")
        cb_recs = cb_engine.recommend(ur, n=10)
        display_movies(cb_recs, show_score=True)

        st.markdown("---")

        # Collaborative
        st.subheader("👥 Collaborative Filtering")
        st.caption("Users with similar taste also enjoyed these (User-Based Cosine Similarity).")
        cf_recs = cf_engine.recommend(ur, n=10)
        display_movies(cf_recs, show_score=True)

        st.markdown("---")

        # Sentiment
        st.subheader("💬 Sentiment-Aware Picks")
        st.caption("Top-rated movies re-ranked by audience sentiment analysis (VADER on tags).")
        se_recs = sent_engine.recommend(ur, n=10)
        display_movies(se_recs, show_score=True)

# ── TAB 3: My Ratings ───────────────────────────────────────────────────────
with tab_rated:
    st.header("My Ratings")
    ur = st.session_state.user_ratings
    if not ur:
        st.info("You haven't rated any movies yet. Go to **Browse & Rate** to start!")
    else:
        rated_df = movies[movies["movieId"].isin(ur.keys())].copy()
        rated_df["my_rating"] = rated_df["movieId"].map(ur)
        rated_df = rated_df.sort_values("my_rating", ascending=False)
        st.dataframe(
            rated_df[["title", "genres", "my_rating"]].rename(
                columns={"title": "Title", "genres": "Genres", "my_rating": "My Rating"}
            ),
            use_container_width=True,
            hide_index=True,
        )
