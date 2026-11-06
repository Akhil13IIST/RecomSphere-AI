
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "ml-100k"

ratings = pd.read_csv(
    DATA / "u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

movies = pd.read_csv(
    DATA / "u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "title"]
)

# Convert timestamps
ratings["timestamp"] = pd.to_datetime(
    ratings["timestamp"], unit="s"
)

print("\n========== DATASET OVERVIEW ==========")
print("Ratings:", ratings.shape)
print("Movies:", movies.shape)
print("Users:", ratings["user_id"].nunique())
print("Movies rated:", ratings["movie_id"].nunique())

print("\n========== RATING STATISTICS ==========")
print(ratings["rating"].describe())

print("\n========== USER ACTIVITY ==========")
user_activity = ratings.groupby("user_id").size()

print(user_activity.describe())

print("\n========== MOVIE POPULARITY ==========")
movie_popularity = (
    ratings.groupby("movie_id")
    .agg(
        rating_count=("rating", "count"),
        average_rating=("rating", "mean")
    )
    .reset_index()
    .merge(movies, on="movie_id")
    .sort_values("rating_count", ascending=False)
)

print(movie_popularity.head(10))

print("\n========== USER-ITEM SPARSITY ==========")

total_possible = (
    ratings["user_id"].nunique()
    * ratings["movie_id"].nunique()
)

sparsity = 1 - len(ratings) / total_possible

print(f"Sparsity: {sparsity:.4%}")

print("\n========== TEMPORAL DISTRIBUTION ==========")

print(
    ratings["timestamp"]
    .dt.to_period("M")
    .value_counts()
    .sort_index()
)

print("\n========== DATA VALIDATION ==========")

assert ratings.isnull().sum().sum() == 0
assert ratings["rating"].between(1, 5).all()
assert ratings["user_id"].nunique() == 943
assert ratings["movie_id"].nunique() == 1682

print("All validation checks passed.")

# Save processed data
output = ROOT / "data" / "processed"
output.mkdir(parents=True, exist_ok=True)

ratings.to_parquet(output / "ratings.parquet", index=False)
movies.to_parquet(output / "movies.parquet", index=False)

movie_popularity.to_parquet(
    output / "movie_popularity.parquet",
    index=False
)

print("\nProcessed datasets saved successfully.")
