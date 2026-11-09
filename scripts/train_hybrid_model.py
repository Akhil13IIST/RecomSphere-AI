
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TOP_K = 10
TRAIN_RATIO = 0.80
RELEVANCE_THRESHOLD = 4

CF_WEIGHT = 0.70
CONTENT_WEIGHT = 0.30

print("Loading datasets and trained models...")

ratings = pd.read_parquet(DATA_DIR / "ratings.parquet")

mappings = joblib.load(MODEL_DIR / "item_cf_mappings.joblib")

user_to_idx = mappings["user_to_idx"]
movie_to_idx = mappings["movie_to_idx"]
movie_ids = mappings["movie_ids"]

cf_similarity = load_npz(
    MODEL_DIR / "item_similarity.npz"
)

content_vectorizer = joblib.load(
    MODEL_DIR / "content_vectorizer.joblib"
)

content_vectors = __import__(
    "scipy.sparse", fromlist=["load_npz"]
).load_npz(
    MODEL_DIR / "content_movie_vectors.npz"
)

# Content similarity matrix is small enough to calculate directly.
content_similarity = cosine_similarity(
    content_vectors,
    dense_output=True,
)

# Reproduce the same per-user chronological split.
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

# Build positive user-item matrix.
n_users = len(user_to_idx)
n_movies = len(movie_to_idx)

from scipy.sparse import csr_matrix

positive_train = train[
    train["rating"] >= RELEVANCE_THRESHOLD
]

rows = positive_train["user_id"].map(user_to_idx).to_numpy()
cols = positive_train["movie_id"].map(movie_to_idx).to_numpy()

user_item = csr_matrix(
    (
        np.ones(len(positive_train), dtype=np.float32),
        (rows, cols),
    ),
    shape=(n_users, n_movies),
)

def normalize_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)

    minimum = np.min(scores)
    maximum = np.max(scores)

    if maximum - minimum < 1e-12:
        return np.zeros_like(scores)

    return (scores - minimum) / (maximum - minimum)


def recommend(user_id, k=TOP_K):
    user_idx = user_to_idx[user_id]

    profile = user_item[user_idx].toarray().ravel()

    # Collaborative filtering scores.
    cf_scores = np.asarray(
        cf_similarity @ profile
    ).ravel()

    # Content-based scores from positively rated movies.
    liked_indices = np.flatnonzero(profile > 0)

    if len(liked_indices):
        content_scores = content_similarity[
            liked_indices
        ].mean(axis=0)
    else:
        content_scores = np.zeros(n_movies)

    # Exclude movies already rated during training.
    seen = train_seen.get(user_id, set())

    seen_indices = [
        movie_to_idx[movie_id]
        for movie_id in seen
        if movie_id in movie_to_idx
    ]

    eligible = np.ones(n_movies, dtype=bool)
    eligible[seen_indices] = False

    # Normalise only over eligible movies.
    hybrid_scores = np.zeros(n_movies)

    if eligible.any():
        cf_normalized = normalize_scores(
            cf_scores[eligible]
        )

        content_normalized = normalize_scores(
            content_scores[eligible]
        )

        hybrid_scores[eligible] = (
            CF_WEIGHT * cf_normalized
            + CONTENT_WEIGHT * content_normalized
        )

    top_indices = np.argsort(hybrid_scores)[::-1][:k]

    return [
        movie_ids[i]
        for i in top_indices
        if eligible[i]
    ]


def evaluate():
    precisions = []
    recalls = []
    ndcgs = []
    hit_rates = []
    average_precisions = []

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

        precisions.append(hit_count / TOP_K)
        recalls.append(hit_count / len(relevant))
        hit_rates.append(float(hit_count > 0))

        dcg = sum(
            hit / np.log2(rank + 2)
            for rank, hit in enumerate(hits)
        )

        ideal_count = min(len(relevant), TOP_K)

        idcg = sum(
            1 / np.log2(rank + 2)
            for rank in range(ideal_count)
        )

        ndcgs.append(dcg / idcg if idcg else 0)

        cumulative_hits = 0
        precision_sum = 0

        for rank, hit in enumerate(hits, start=1):
            if hit:
                cumulative_hits += 1
                precision_sum += cumulative_hits / rank

        average_precisions.append(
            precision_sum / min(len(relevant), TOP_K)
        )

    return {
        "evaluated_users": len(precisions),
        "precision_at_10": float(np.mean(precisions)),
        "recall_at_10": float(np.mean(recalls)),
        "ndcg_at_10": float(np.mean(ndcgs)),
        "hit_rate_at_10": float(np.mean(hit_rates)),
        "map_at_10": float(np.mean(average_precisions)),
        "collaborative_weight": CF_WEIGHT,
        "content_weight": CONTENT_WEIGHT,
    }


print("\nEvaluating hybrid recommendations...")

metrics = evaluate()

print("\nHYBRID MODEL RESULTS")
print("=" * 45)

for key, value in metrics.items():
    print(f"{key}: {value}")

with open(REPORT_DIR / "hybrid_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

joblib.dump(
    {
        "cf_weight": CF_WEIGHT,
        "content_weight": CONTENT_WEIGHT,
        "top_k": TOP_K,
        "relevance_threshold": RELEVANCE_THRESHOLD,
    },
    MODEL_DIR / "hybrid_config.joblib",
)

print("\nHybrid model evaluation completed.")
