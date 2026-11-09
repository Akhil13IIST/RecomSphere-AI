
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 10
TRAIN_RATIO = 0.80
RELEVANCE_THRESHOLD = 4

print("Loading movie metadata and ratings...")

movies = pd.read_parquet(DATA_DIR / "movies_enriched.parquet")
ratings = pd.read_parquet(DATA_DIR / "ratings.parquet")

movies = movies.sort_values("movie_id").reset_index(drop=True)

movie_ids = movies["movie_id"].to_numpy()
movie_to_idx = {movie_id: i for i, movie_id in enumerate(movie_ids)}

# Build content representations.
print("Building TF-IDF vectors...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    stop_words="english",
)

movie_vectors = vectorizer.fit_transform(
    movies["content"].fillna("")
)

print("Computing content similarity...")

similarity = cosine_similarity(
    movie_vectors,
    dense_output=True,
)

# Same per-user chronological split as collaborative filtering.
ratings = ratings.sort_values(
    ["user_id", "timestamp"]
).reset_index(drop=True)

train_parts = []
test_parts = []

for _, group in ratings.groupby("user_id", sort=False):
    split_index = int(len(group) * TRAIN_RATIO)
    train_parts.append(group.iloc[:split_index])
    test_parts.append(group.iloc[split_index:])

train = pd.concat(train_parts, ignore_index=True)
test = pd.concat(test_parts, ignore_index=True)

train_seen = (
    train.groupby("user_id")["movie_id"]
    .apply(set)
    .to_dict()
)

test_relevant = (
    test[test["rating"] >= RELEVANCE_THRESHOLD]
    .groupby("user_id")["movie_id"]
    .apply(set)
    .to_dict()
)

# Use liked movies to build a user content profile.
def recommend(user_id, k=TOP_K):
    history = train[
        (train["user_id"] == user_id) &
        (train["rating"] >= RELEVANCE_THRESHOLD)
    ]

    if history.empty:
        return []

    indices = [
        movie_to_idx[movie_id]
        for movie_id in history["movie_id"]
        if movie_id in movie_to_idx
    ]

    if not indices:
        return []

    # Average similarity to positively rated movies.
    scores = similarity[indices].mean(axis=0)

    seen = train_seen.get(user_id, set())

    seen_indices = [
        movie_to_idx[movie_id]
        for movie_id in seen
        if movie_id in movie_to_idx
    ]

    scores[seen_indices] = -np.inf

    top_indices = np.argsort(scores)[::-1][:k]

    return [
        movie_ids[i]
        for i in top_indices
        if np.isfinite(scores[i])
    ]

# Evaluate recommendations.
def evaluate():
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    hit_scores = []
    ap_scores = []

    for user_id, relevant in test_relevant.items():
        if not relevant:
            continue

        recommendations = recommend(user_id)

        if not recommendations:
            continue

        hits = [
            int(movie_id in relevant)
            for movie_id in recommendations
        ]

        hit_count = sum(hits)

        precision_scores.append(hit_count / TOP_K)
        recall_scores.append(hit_count / len(relevant))
        hit_scores.append(float(hit_count > 0))

        dcg = sum(
            hit / np.log2(rank + 2)
            for rank, hit in enumerate(hits)
        )

        ideal_count = min(len(relevant), TOP_K)

        idcg = sum(
            1 / np.log2(rank + 2)
            for rank in range(ideal_count)
        )

        ndcg_scores.append(dcg / idcg if idcg else 0)

        cumulative_hits = 0
        precision_sum = 0

        for rank, hit in enumerate(hits, start=1):
            if hit:
                cumulative_hits += 1
                precision_sum += cumulative_hits / rank

        ap_scores.append(
            precision_sum / min(len(relevant), TOP_K)
        )

    return {
        "precision_at_10": float(np.mean(precision_scores)),
        "recall_at_10": float(np.mean(recall_scores)),
        "ndcg_at_10": float(np.mean(ndcg_scores)),
        "hit_rate_at_10": float(np.mean(hit_scores)),
        "map_at_10": float(np.mean(ap_scores)),
        "evaluated_users": len(precision_scores),
    }

print("\nEvaluating content-based recommendations...")

metrics = evaluate()

print("\nCONTENT-BASED RESULTS")
print("=" * 45)

for key, value in metrics.items():
    print(f"{key}: {value}")

# Save model.
joblib.dump(
    vectorizer,
    MODEL_DIR / "content_vectorizer.joblib",
)

joblib.dump(
    movie_to_idx,
    MODEL_DIR / "content_movie_mapping.joblib",
)

np.save(
    MODEL_DIR / "content_movie_ids.npy",
    movie_ids,
)

from scipy.sparse import save_npz

save_npz(
    MODEL_DIR / "content_movie_vectors.npz",
    movie_vectors,
)

with open(REPORT_DIR / "content_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nContent-based model saved successfully.")
