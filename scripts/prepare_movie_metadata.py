
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw" / "ml-100k"
PROCESSED_DIR = ROOT / "data" / "processed"

GENRES = [
    "unknown", "Action", "Adventure", "Animation",
    "Children", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]

columns = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
] + GENRES

print("Loading original MovieLens metadata...")

metadata = pd.read_csv(
    RAW_DIR / "u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    names=columns,
)

metadata["genres"] = metadata[GENRES].apply(
    lambda row: " ".join(
        genre for genre in GENRES if row[genre] == 1
    ),
    axis=1,
)

metadata["content"] = (
    metadata["title"].fillna("").str.replace(r"\(\d{4}\)", "", regex=True)
    + " "
    + metadata["genres"]
)

movies = pd.read_parquet(
    PROCESSED_DIR / "movies.parquet"
)

enriched = movies.merge(
    metadata[["movie_id", "release_date", "imdb_url", *GENRES,
              "genres", "content"]],
    on="movie_id",
    how="left",
    validate="one_to_one",
)

if enriched["genres"].isna().any():
    raise ValueError("Some movies have missing genre metadata.")

output_path = PROCESSED_DIR / "movies_enriched.parquet"

enriched.to_parquet(output_path, index=False)

print("\nMETADATA PROCESSING COMPLETE")
print("=" * 45)
print(f"Movies: {len(enriched):,}")
print(f"Genre columns: {len(GENRES)}")
print(f"Missing genres: {enriched['genres'].isna().sum()}")

print("\nSample movies:")
print(enriched[["movie_id", "title", "genres"]].head(10).to_string(index=False))

print(f"\nSaved: {output_path}")
