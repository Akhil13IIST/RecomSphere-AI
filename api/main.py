
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, load_npz
from fastapi import FastAPI, HTTPException, Query

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

app = FastAPI(
    title="RecomSphere AI",
    description="Hybrid Personalized Movie Recommendation Engine",
    version="1.0.0",
)

print("Loading RecomSphere models...")

# Load datasets.
ratings = pd.read_parquet(DATA_DIR / "ratings.parquet")
movies = pd.read_parquet(DATA_DIR / "movies_enriched.parquet")

# Load model mappings.
mappings = joblib.load(MODEL_DIR / "item_cf_mappings.joblib")

user_to_idx = mappings["user_to_idx"]
movie_to_idx = mappings["movie_to_idx"]

user_ids = mappings["user_ids"]
movie_ids = mappings["movie_ids"]

# Load item-item similarity.
cf_similarity = load_npz(
    MODEL_DIR / "item_similarity.npz"
)

# Load content vectors.
content_vectors = load_npz(
    MODEL_DIR / "content_movie_vectors.npz"
)

# Reconstruct the same per-user chronological training split.
ratings = ratings.sort_values(
    ["user_id", "timestamp"]
).reset_index(drop=True)

train_parts = []

for _, group in ratings.groupby("user_id", sort=False):
    split_index = int(len(group) * 0.80)
    train_parts.append(group.iloc[:split_index])

train = pd.concat(train_parts, ignore_index=True)

# Build positive user-item interaction matrix.
positive = train[train["rating"] >= 4]

rows = positive["user_id"].map(user_to_idx).to_numpy()
cols = positive["movie_id"].map(movie_to_idx).to_numpy()

user_item = csr_matrix(
    (
        np.ones(len(positive), dtype=np.float32),
        (rows, cols),
    ),
    shape=(len(user_ids), len(movie_ids)),
)

train_seen = (
    train.groupby("user_id")["movie_id"]
    .apply(set)
    .to_dict()
)

movie_lookup = movies.set_index("movie_id").to_dict("index")

print("RecomSphere models loaded successfully.")


@app.get("/")
def home():
    return {
        "application": "RecomSphere AI",
        "status": "running",
        "models": [
            "Collaborative Filtering",
            "Content-Based Filtering",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "users": len(user_ids),
        "movies": len(movie_ids),
    }


@app.get("/recommend/{user_id}")
def recommend(
    user_id: int,
    limit: int = Query(default=10, ge=1, le=50),
    method: str = Query(default="collaborative"),
):
    if user_id not in user_to_idx:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    if method not in {"collaborative", "content", "hybrid"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid recommendation method",
        )

    user_idx = user_to_idx[user_id]
    profile = user_item[user_idx].toarray().ravel()

    liked_indices = np.flatnonzero(profile > 0)

    if not len(liked_indices):
        raise HTTPException(
            status_code=404,
            detail="No positive rating history for this user",
        )

    cf_scores = np.asarray(
        cf_similarity @ profile
    ).ravel()

    if method in {"content", "hybrid"}:
        content_scores = np.asarray(
            content_vectors @ (
                content_vectors[liked_indices].mean(axis=0).T
            )
        ).ravel()
    else:
        content_scores = np.zeros(len(movie_ids))

    seen = train_seen.get(user_id, set())

    eligible = np.ones(len(movie_ids), dtype=bool)

    for movie_id in seen:
        if movie_id in movie_to_idx:
            eligible[movie_to_idx[movie_id]] = False

    def normalize(values):
        values = np.asarray(values, dtype=float)
        minimum = values[eligible].min()
        maximum = values[eligible].max()

        if maximum - minimum < 1e-12:
            return np.zeros_like(values)

        return (values - minimum) / (maximum - minimum)

    if method == "collaborative":
        scores = cf_scores
    elif method == "content":
        scores = content_scores
    else:
        scores = (
            0.70 * normalize(cf_scores)
            + 0.30 * normalize(content_scores)
        )

    scores[~eligible] = -np.inf

    top_indices = np.argsort(scores)[::-1][:limit]

    results = []

    for index in top_indices:
        movie_id = int(movie_ids[index])
        movie = movie_lookup[movie_id]

        results.append({
            "movie_id": movie_id,
            "title": movie["title"],
            "genres": movie["genres"],
            "score": round(float(scores[index]), 5),
        })

    return {
        "user_id": user_id,
        "method": method,
        "recommendations": results,
    }


@app.get("/similar/{movie_id}")
def similar_movies(
    movie_id: int,
    limit: int = Query(default=10, ge=1, le=50),
):
    if movie_id not in movie_to_idx:
        raise HTTPException(
            status_code=404,
            detail="Movie not found",
        )

    index = movie_to_idx[movie_id]

    similarities = (
        content_vectors @ content_vectors[index].T
    ).toarray().ravel()

    similarities[index] = -np.inf

    top_indices = np.argsort(similarities)[::-1][:limit]

    results = []

    for i in top_indices:
        mid = int(movie_ids[i])
        movie = movie_lookup[mid]

        results.append({
            "movie_id": mid,
            "title": movie["title"],
            "genres": movie["genres"],
            "similarity": round(float(similarities[i]), 5),
        })

    return {
        "source_movie": movie_lookup[movie_id]["title"],
        "similar_movies": results,
    }


@app.get("/movies/search")
def search_movies(
    q: str = Query(min_length=1, max_length=100),
    limit: int = Query(default=20, ge=1, le=100),
):
    matches = movies[
        movies["title"].str.contains(
            q,
            case=False,
            na=False,
            regex=False,
        )
    ].head(limit)

    return {
        "results": matches[
            ["movie_id", "title", "genres"]
        ].to_dict(orient="records")
    }
