
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, save_npz
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 10
RELEVANCE_THRESHOLD = 4
TRAIN_RATIO = 0.80

print("Loading MovieLens data...")

ratings = pd.read_parquet(DATA_DIR / "ratings.parquet")

# Chronological split for each user.
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

print(f"Training ratings: {len(train):,}")
print(f"Test ratings: {len(test):,}")

# Build consistent ID mappings.
user_ids = np.sort(ratings["user_id"].unique())
movie_ids = np.sort(ratings["movie_id"].unique())

user_to_idx = {v: i for i, v in enumerate(user_ids)}
movie_to_idx = {v: i for i, v in enumerate(movie_ids)}

n_users = len(user_ids)
n_movies = len(movie_ids)

def make_matrix(df):
    rows = df["user_id"].map(user_to_idx).to_numpy()
    cols = df["movie_id"].map(movie_to_idx).to_numpy()

    # Only positive preferences contribute to similarity.
    values = (
        df["rating"].to_numpy() >= RELEVANCE_THRESHOLD
    ).astype(np.float32)

    return csr_matrix(
        (values, (rows, cols)),
        shape=(n_users, n_movies),
    )

print("\nBuilding user-item interaction matrix...")

train_matrix = make_matrix(train)

# Item-item cosine similarity.
print("Computing item-item similarity...")

item_user_matrix = train_matrix.T

similarity = cosine_similarity(
    item_user_matrix,
    dense_output=True,
)

# Evaluation.
test_relevant = (
    test[test["rating"] >= RELEVANCE_THRESHOLD]
    .groupby("user_id")["movie_id"]
    .apply(set)
    .to_dict()
)

train_seen = (
    train.groupby("user_id")["movie_id"]
    .apply(set)
    .to_dict()
)

def recommend(user_id, k=TOP_K):
    user_idx = user_to_idx[user_id]

    # Positive training interactions form the user profile.
    profile = train_matrix[user_idx].toarray().ravel()

    scores = similarity @ profile

    # Exclude every movie already rated in training.
    seen = train_seen.get(user_id, set())

    if seen:
        seen_indices = [
            movie_to_idx[movie_id]
            for movie_id in seen
        ]
        scores[seen_indices] = -np.inf

    top_indices = np.argsort(scores)[::-1][:k]

    return [
        movie_ids[i]
        for i in top_indices
        if np.isfinite(scores[i])
    ]

def evaluate():
    precisions = []
    recalls = []
    ndcgs = []
    hit_rates = []
    average_precisions = []

    evaluated = 0

    for user_id, relevant in test_relevant.items():
        if not relevant or user_id not in user_to_idx:
            continue

        recommendations = recommend(user_id)

        if not recommendations:
            continue

        evaluated += 1

        hits = [
            1 if movie_id in relevant else 0
            for movie_id in recommendations
        ]

        hit_count = sum(hits)

        precisions.append(hit_count / TOP_K)
        recalls.append(hit_count / len(relevant))
        hit_rates.append(float(hit_count > 0))

        # NDCG@K
        dcg = sum(
            hit / np.log2(rank + 2)
            for rank, hit in enumerate(hits)
        )

        ideal_hits = min(len(relevant), TOP_K)

        idcg = sum(
            1 / np.log2(rank + 2)
            for rank in range(ideal_hits)
        )

        ndcgs.append(dcg / idcg if idcg else 0)

        # MAP@K
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
        "evaluated_users": evaluated,
        "precision_at_10": float(np.mean(precisions)),
        "recall_at_10": float(np.mean(recalls)),
        "ndcg_at_10": float(np.mean(ndcgs)),
        "hit_rate_at_10": float(np.mean(hit_rates)),
        "map_at_10": float(np.mean(average_precisions)),
    }

print("\nEvaluating collaborative filtering...")

metrics = evaluate()

print("\nCOLLABORATIVE FILTERING RESULTS")
print("=" * 45)

for key, value in metrics.items():
    print(f"{key}: {value}")

# Save model artifacts.
save_npz(
    MODEL_DIR / "item_similarity.npz",
    csr_matrix(similarity),
)

joblib.dump(
    {
        "user_to_idx": user_to_idx,
        "movie_to_idx": movie_to_idx,
        "user_ids": user_ids,
        "movie_ids": movie_ids,
    },
    MODEL_DIR / "item_cf_mappings.joblib",
)

with open(REPORT_DIR / "item_cf_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nModel artifacts and metrics saved successfully.")
