
from pathlib import Path
import json

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "ratings.parquet"
OUTPUT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K = 10
RELEVANCE_THRESHOLD = 4
MIN_RATINGS = 20
TRAIN_RATIO = 0.80


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

ratings = pd.read_parquet(DATA_PATH)

ratings = ratings.sort_values(
    ["user_id", "timestamp"]
).reset_index(drop=True)

print("Dataset loaded:", ratings.shape)


# --------------------------------------------------
# 3. TEMPORAL TRAIN-TEST SPLIT
# --------------------------------------------------

train_parts = []
test_parts = []

for user_id, group in ratings.groupby("user_id"):
    split_index = int(len(group) * TRAIN_RATIO)

    train_parts.append(group.iloc[:split_index])
    test_parts.append(group.iloc[split_index:])

train = pd.concat(train_parts, ignore_index=True)
test = pd.concat(test_parts, ignore_index=True)

print("\nTRAIN:", train.shape)
print("TEST:", test.shape)

# Ensure temporal separation for every user.
assert (
    train.groupby("user_id")["timestamp"].max()
    <= test.groupby("user_id")["timestamp"].min()
).all()


# --------------------------------------------------
# 4. BUILD POPULARITY MODEL
# --------------------------------------------------

movie_stats = (
    train.groupby("movie_id")
    .agg(
        rating_count=("rating", "count"),
        average_rating=("rating", "mean")
    )
)

global_average = train["rating"].mean()

# Bayesian weighted rating:
# Smooths movie ratings when a movie has few ratings.

movie_stats["weighted_score"] = (
    (
        movie_stats["rating_count"]
        / (movie_stats["rating_count"] + MIN_RATINGS)
    ) * movie_stats["average_rating"]
    +
    (
        MIN_RATINGS
        / (movie_stats["rating_count"] + MIN_RATINGS)
    ) * global_average
)

movie_stats = movie_stats.sort_values(
    "weighted_score",
    ascending=False
)

print("\nTOP 10 POPULAR MOVIES")
print(movie_stats.head(10))


# --------------------------------------------------
# 5. PREPARE USER HISTORY
# --------------------------------------------------

user_history = (
    train.groupby("user_id")["movie_id"]
    .apply(set)
    .to_dict()
)

# Only ratings >= 4 are considered relevant.
relevant_items = (
    test[test["rating"] >= RELEVANCE_THRESHOLD]
    .groupby("user_id")["movie_id"]
    .apply(set)
    .to_dict()
)


# --------------------------------------------------
# 6. GENERATE RECOMMENDATIONS
# --------------------------------------------------

recommendations = {}

all_movies = movie_stats.index.tolist()

for user_id in test["user_id"].unique():

    watched = user_history.get(user_id, set())

    candidates = [
        movie_id
        for movie_id in all_movies
        if movie_id not in watched
    ]

    recommendations[user_id] = candidates[:K]


# --------------------------------------------------
# 7. EVALUATION METRICS
# --------------------------------------------------

def evaluate_at_k(recommendations, relevant_items, k=10):

    precisions = []
    recalls = []
    ndcgs = []
    hit_rates = []

    for user_id, relevant in relevant_items.items():

        if not relevant:
            continue

        predicted = recommendations.get(user_id, [])[:k]

        hits = [
            1 if movie_id in relevant else 0
            for movie_id in predicted
        ]

        hit_count = sum(hits)

        precision = hit_count / k
        recall = hit_count / len(relevant)

        dcg = sum(
            hit / np.log2(position + 2)
            for position, hit in enumerate(hits)
        )

        ideal_hits = min(len(relevant), k)

        idcg = sum(
            1 / np.log2(position + 2)
            for position in range(ideal_hits)
        )

        ndcg = dcg / idcg if idcg > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
        hit_rates.append(float(hit_count > 0))

    return {
        f"Precision@{k}": float(np.mean(precisions)),
        f"Recall@{k}": float(np.mean(recalls)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        f"HitRate@{k}": float(np.mean(hit_rates)),
        "Evaluated_users": len(precisions)
    }


# --------------------------------------------------
# 8. RUN EVALUATION
# --------------------------------------------------

metrics = evaluate_at_k(
    recommendations,
    relevant_items,
    K
)

print("\n========== BASELINE RESULTS ==========")

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}"
          if isinstance(value, float)
          else f"{metric}: {value}")


# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------

results = pd.DataFrame([
    {
        "user_id": user_id,
        "recommended_movie_ids": ",".join(map(str, items))
    }
    for user_id, items in recommendations.items()
])

results.to_csv(
    OUTPUT_DIR / "popularity_recommendations.csv",
    index=False
)

with open(OUTPUT_DIR / "baseline_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

movie_stats.to_csv(
    OUTPUT_DIR / "popularity_model.csv"
)

print("\nArtifacts saved in:", OUTPUT_DIR)
print("Baseline training and evaluation completed.")
