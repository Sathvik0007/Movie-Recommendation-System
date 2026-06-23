"""
Movie Recommendation Engines
─────────────────────────────
1. Content-Based Filtering   (TF-IDF on genres + user tags)
2. Collaborative Filtering   (User-based cosine similarity)
3. Sentiment-Aware Ranking   (VADER on user tags)
4. Hybrid Engine             (Weighted combination of all three)
5. Evaluation Metrics        (RMSE, Precision@K, Recall@K)
6. Cold-Start Handling       (Popularity-based fallback)
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from math import sqrt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ml-latest-small")


def load_data():
    """Load movies, ratings, and tags DataFrames."""
    movies = pd.read_csv(os.path.join(DATA_PATH, "movies.csv"))
    ratings = pd.read_csv(os.path.join(DATA_PATH, "ratings.csv"))
    tags = pd.read_csv(os.path.join(DATA_PATH, "tags.csv"))

    # Aggregate user-generated tags per movie as pseudo-overview text
    tag_text = tags.groupby("movieId")["tag"].apply(
        lambda x: " ".join(x.dropna().astype(str))
    ).rename("tag_text")
    movies = movies.merge(tag_text, on="movieId", how="left")
    movies["tag_text"] = movies["tag_text"].fillna("")

    # Build rich content field: genres (pipe→space) + tag text
    movies["content"] = (
        movies["genres"].str.replace("|", " ", regex=False)
        + " " + movies["tag_text"]
    )

    return movies, ratings, tags


# ---------------------------------------------------------------------------
# Cold-Start Handler
# ---------------------------------------------------------------------------

class ColdStartHandler:
    """Popularity-based fallback for new users with few or no ratings."""

    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        self.movies = movies
        # Weighted rating: (count / (count + m)) * avg + (m / (count + m)) * C
        stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
        C = stats["mean"].mean()           # global mean rating
        m = stats["count"].quantile(0.70)  # min votes threshold
        stats["weighted"] = (
            (stats["count"] / (stats["count"] + m)) * stats["mean"]
            + (m / (stats["count"] + m)) * C
        )
        self.popularity = stats.sort_values("weighted", ascending=False)

    def recommend(self, exclude_ids: set = None, n: int = 10):
        """Return top-n popular movies, excluding already-rated ones."""
        pool = self.popularity
        if exclude_ids:
            pool = pool[~pool.index.isin(exclude_ids)]
        top_ids = pool.head(n).index
        result = self.movies[self.movies["movieId"].isin(top_ids)].copy()
        result["score"] = result["movieId"].map(pool["weighted"])
        result["explanation"] = "🔥 Popular among all users (cold-start fallback)"
        return result.sort_values("score", ascending=False)


# ---------------------------------------------------------------------------
# 1. Content-Based Filtering (Enhanced)
# ---------------------------------------------------------------------------

class ContentBasedEngine:
    """
    Recommends movies similar to the user's liked films.
    Uses TF-IDF on genres + user-generated tags (as proxy for overview/keywords).
    """

    def __init__(self, movies: pd.DataFrame):
        self.movies = movies
        self.tfidf = TfidfVectorizer(
            stop_words="english", max_features=5000, ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf.fit_transform(movies["content"])

    def recommend(self, user_ratings: dict, n: int = 10):
        """
        Build a user profile from TF-IDF vectors of liked movies (rating >= 4),
        then find most similar unseen movies.
        Returns DataFrame with score and explanation columns.
        """
        liked = [mid for mid, r in user_ratings.items() if r >= 4]
        if not liked:
            liked = list(user_ratings.keys())[:5]
        if not liked:
            return pd.DataFrame()

        # Collect indices of liked movies
        indices = []
        for mid in liked:
            idx = self.movies.index[self.movies["movieId"] == mid]
            if len(idx):
                indices.append(idx[0])
        if not indices:
            return pd.DataFrame()

        # User profile = weighted-average TF-IDF vector of liked movies
        profile = self.tfidf_matrix[indices].mean(axis=0)
        profile = np.asarray(profile)
        sim_scores = cosine_similarity(profile, self.tfidf_matrix).flatten()

        # Exclude already-rated movies
        for mid in user_ratings:
            idx = self.movies.index[self.movies["movieId"] == mid]
            if len(idx):
                sim_scores[idx[0]] = -1

        top_idx = sim_scores.argsort()[::-1][:n]
        result = self.movies.iloc[top_idx].copy()
        result["score"] = sim_scores[top_idx]

        # Build explanations: top matching genres from liked movies
        liked_genres = set()
        for mid in liked[:5]:
            row = self.movies[self.movies["movieId"] == mid]
            if not row.empty:
                liked_genres.update(row.iloc[0]["genres"].split("|"))
        liked_genres.discard("(no genres listed)")
        genre_str = ", ".join(sorted(liked_genres)[:4])
        result["explanation"] = result.apply(
            lambda r: f"🎯 Similar content profile to your liked genres: {genre_str}", axis=1
        )
        return result


# ---------------------------------------------------------------------------
# 2. Collaborative Filtering (User-Based)
# ---------------------------------------------------------------------------

class CollaborativeEngine:
    """User-based collaborative filtering using cosine similarity."""

    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        self.movies = movies
        self.ratings = ratings
        self.user_item = ratings.pivot_table(
            index="userId", columns="movieId", values="rating"
        ).fillna(0)

    def recommend(self, user_ratings: dict, n: int = 10, k_users: int = 20):
        """
        Find top-k similar users, recommend their highly-rated unseen movies.
        Returns DataFrame with score, confidence, and explanation.
        """
        if not user_ratings:
            return pd.DataFrame()

        # Build active user vector
        active_vec = pd.Series(0.0, index=self.user_item.columns)
        for mid, r in user_ratings.items():
            if mid in active_vec.index:
                active_vec[mid] = r

        active_arr = active_vec.values.reshape(1, -1)
        sims = cosine_similarity(active_arr, self.user_item.values).flatten()

        # Top-k similar users
        top_k_idx = sims.argsort()[::-1][:k_users]
        top_k_sims = sims[top_k_idx]

        # Weighted average prediction
        weighted_sum = np.zeros(len(self.user_item.columns))
        sim_sum = 0.0
        contributing_users = 0
        for ui, s in zip(top_k_idx, top_k_sims):
            if s <= 0:
                continue
            weighted_sum += s * self.user_item.iloc[ui].values
            sim_sum += s
            contributing_users += 1

        if sim_sum == 0:
            return pd.DataFrame()

        predicted = weighted_sum / sim_sum

        # Zero out already-rated
        for mid in user_ratings:
            if mid in self.user_item.columns:
                predicted[self.user_item.columns.get_loc(mid)] = -1

        top_movie_idx = predicted.argsort()[::-1][:n]
        rec_ids = self.user_item.columns[top_movie_idx]
        result = self.movies[self.movies["movieId"].isin(rec_ids)].copy()

        score_map = dict(zip(rec_ids, predicted[top_movie_idx]))
        result["score"] = result["movieId"].map(score_map)
        # Confidence = max similarity among contributing users (0-1)
        result["confidence"] = round(float(top_k_sims[0]), 3) if len(top_k_sims) else 0
        result["explanation"] = (
            f"👥 {contributing_users} similar users (max similarity "
            f"{top_k_sims[0]:.2f}) also rated this highly"
        )
        return result.sort_values("score", ascending=False)


# ---------------------------------------------------------------------------
# 3. Sentiment-Aware Ranking
# ---------------------------------------------------------------------------

class SentimentEngine:
    """Re-ranks movies using VADER sentiment on user-generated tags."""

    def __init__(self, tags: pd.DataFrame, movies: pd.DataFrame, ratings: pd.DataFrame):
        self.movies = movies
        self.sia = SentimentIntensityAnalyzer()

        if tags.empty or tags["tag"].dropna().empty:
            self.sentiment_scores = pd.Series(dtype=float)
        else:
            tag_text = tags.groupby("movieId")["tag"].apply(
                lambda x: " ".join(x.dropna().astype(str))
            )
            self.sentiment_scores = tag_text.apply(
                lambda t: self.sia.polarity_scores(t)["compound"]
            )

        self.avg_ratings = ratings.groupby("movieId")["rating"].mean()

    def get_sentiment(self, movie_id: int) -> float:
        return self.sentiment_scores.get(movie_id, 0.0)

    def recommend(self, user_ratings: dict, n: int = 10):
        rated_set = set(user_ratings.keys())
        combined = pd.DataFrame({"avg_rating": self.avg_ratings})
        combined["sentiment"] = combined.index.map(
            lambda mid: self.sentiment_scores.get(mid, 0.0)
        )
        combined["combined_score"] = (
            combined["avg_rating"] / 5.0 * 0.6
            + (combined["sentiment"] + 1) / 2.0 * 0.4
        )
        combined = combined[~combined.index.isin(rated_set)]
        top = combined.nlargest(n, "combined_score")

        result = self.movies[self.movies["movieId"].isin(top.index)].copy()
        result["score"] = result["movieId"].map(top["combined_score"])
        result["sentiment"] = result["movieId"].map(
            lambda mid: self.sentiment_scores.get(mid, 0.0)
        )
        sent_label = lambda s: "positive" if s > 0.2 else ("negative" if s < -0.2 else "neutral")
        result["explanation"] = result["sentiment"].apply(
            lambda s: f"💬 Audience sentiment: {sent_label(s)} ({s:+.2f}), boosted by high avg rating"
        )
        return result.sort_values("score", ascending=False)


# ---------------------------------------------------------------------------
# 4. Hybrid Engine (Weighted Combination)
# ---------------------------------------------------------------------------

class HybridEngine:
    """
    Combines content-based, collaborative, and sentiment scores
    using configurable weights into a single ranked list.
    """

    def __init__(self, cb: ContentBasedEngine, cf: CollaborativeEngine,
                 se: SentimentEngine, cold: ColdStartHandler):
        self.cb = cb
        self.cf = cf
        self.se = se
        self.cold = cold

    def recommend(self, user_ratings: dict, n: int = 10,
                  w_cb: float = 0.4, w_cf: float = 0.4, w_se: float = 0.2):
        """
        Weighted hybrid: score = w_cb * CB_score + w_cf * CF_score + w_se * SE_score.
        Falls back to cold-start if < 3 ratings.
        """
        # Cold-start fallback
        if len(user_ratings) < 3:
            return self.cold.recommend(exclude_ids=set(user_ratings.keys()), n=n)

        # Get individual recommendations (more than n to ensure overlap)
        pool_n = n * 3
        cb_df = self.cb.recommend(user_ratings, n=pool_n)
        cf_df = self.cf.recommend(user_ratings, n=pool_n)
        se_df = self.se.recommend(user_ratings, n=pool_n)

        # Normalize scores to 0–1 within each engine
        def normalize(df):
            if df.empty or "score" not in df.columns:
                return {}
            s = df.set_index("movieId")["score"]
            mn, mx = s.min(), s.max()
            if mx - mn == 0:
                return dict(zip(s.index, [0.5] * len(s)))
            return ((s - mn) / (mx - mn)).to_dict()

        cb_scores = normalize(cb_df)
        cf_scores = normalize(cf_df)
        se_scores = normalize(se_df)

        # Merge all candidate movie IDs
        all_ids = set(cb_scores) | set(cf_scores) | set(se_scores)
        all_ids -= set(user_ratings.keys())

        rows = []
        for mid in all_ids:
            cb_s = cb_scores.get(mid, 0)
            cf_s = cf_scores.get(mid, 0)
            se_s = se_scores.get(mid, 0)
            hybrid = w_cb * cb_s + w_cf * cf_s + w_se * se_s

            # Build explanation showing contribution of each engine
            parts = []
            if cb_s > 0:
                parts.append(f"Content: {cb_s:.2f}")
            if cf_s > 0:
                parts.append(f"Collab: {cf_s:.2f}")
            if se_s > 0:
                parts.append(f"Sentiment: {se_s:.2f}")
            explanation = "⚡ Hybrid — " + " · ".join(parts) if parts else "⚡ Hybrid"

            rows.append({
                "movieId": mid, "hybrid_score": hybrid,
                "cb_score": cb_s, "cf_score": cf_s, "se_score": se_s,
                "explanation": explanation,
            })

        if not rows:
            return pd.DataFrame()

        hybrid_df = pd.DataFrame(rows).sort_values("hybrid_score", ascending=False).head(n)
        result = hybrid_df.merge(
            self.cb.movies[["movieId", "title", "genres"]], on="movieId", how="left"
        )
        result["score"] = result["hybrid_score"]
        return result


# ---------------------------------------------------------------------------
# 5. Evaluation Metrics
# ---------------------------------------------------------------------------

class Evaluator:
    """Computes RMSE, Precision@K, and Recall@K via train/test split."""

    def __init__(self, ratings: pd.DataFrame):
        self.ratings = ratings

    def evaluate(self, cf_engine_class, movies, k: int = 10,
                 test_size: float = 0.2, threshold: float = 4.0):
        """
        Split ratings into train/test, build CF engine on train,
        then compute metrics on test.
        Returns dict with RMSE, Precision@K, Recall@K.
        """
        train, test = train_test_split(
            self.ratings, test_size=test_size, random_state=42
        )

        # Build CF on training data
        cf = cf_engine_class(train, movies)

        # Group test by user
        test_by_user = test.groupby("userId")

        rmse_errors = []
        precisions = []
        recalls = []

        # Sample users for speed (evaluating all 610 users is slow in Streamlit)
        sampled_users = test_by_user.groups.keys()
        user_sample = list(sampled_users)[:50]

        for uid in user_sample:
            user_test = test_by_user.get_group(uid)
            user_train = train[train["userId"] == uid]
            if user_train.empty:
                continue

            train_ratings = dict(zip(user_train["movieId"], user_train["rating"]))
            recs = cf.recommend(train_ratings, n=k)

            if recs.empty:
                continue

            rec_ids = set(recs["movieId"].tolist())

            # Relevant = test movies rated >= threshold
            relevant = set(user_test[user_test["rating"] >= threshold]["movieId"])

            # Precision@K and Recall@K
            hits = rec_ids & relevant
            precisions.append(len(hits) / k if k > 0 else 0)
            recalls.append(len(hits) / len(relevant) if relevant else 0)

            # RMSE: for movies that appear in both recs and test
            for _, row in user_test.iterrows():
                mid = row["movieId"]
                if mid in rec_ids and "score" in recs.columns:
                    pred_row = recs[recs["movieId"] == mid]
                    if not pred_row.empty:
                        pred = pred_row["score"].iloc[0]
                        # Clamp prediction to 0-5 range
                        pred = max(0, min(5, pred))
                        rmse_errors.append((row["rating"] - pred) ** 2)

        rmse = sqrt(np.mean(rmse_errors)) if rmse_errors else float("nan")
        avg_precision = np.mean(precisions) if precisions else 0.0
        avg_recall = np.mean(recalls) if recalls else 0.0

        return {
            "RMSE": round(rmse, 4),
            "Precision@K": round(avg_precision, 4),
            "Recall@K": round(avg_recall, 4),
            "K": k,
            "Users Evaluated": len(user_sample),
            "Test Size": len(test),
        }
