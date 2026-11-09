from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
data = root / "data" / "processed"
output = root / "models" / "collaborative" / "track_metadata.parquet"

train = pd.read_parquet(data / "train_interactions",
    columns=["track_id", "track_name", "artist_name"])
valid = pd.read_parquet(data / "validation_interactions",
    columns=["track_id", "track_name", "artist_name"])

metadata = pd.concat([train, valid], ignore_index=True)
metadata = metadata.drop_duplicates("track_id", keep="first")
metadata.to_parquet(output, compression="zstd", index=False)

print(f"Tracks: {len(metadata):,}")
print(f"Metadata size: {output.stat().st_size / 1024**2:.2f} MB")
