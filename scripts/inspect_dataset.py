from pathlib import Path
import pandas as pd

DATA_PATH = Path("AIML_03/data/raw/ml-100k/u.data")

ratings = pd.read_csv(
    DATA_PATH,
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

print("\n===== MOVIELENS DATASET =====")
print("Shape:", ratings.shape)
print("\nFirst 5 rows:")
print(ratings.head())

print("\nMissing values:")
print(ratings.isnull().sum())

print("\nUnique users:", ratings["user_id"].nunique())
print("Unique movies:", ratings["movie_id"].nunique())

print("\nRating distribution:")
print(ratings["rating"].value_counts().sort_index())

print("\nDate range:")
print(
    pd.to_datetime(ratings["timestamp"], unit="s").agg(["min", "max"])
)
