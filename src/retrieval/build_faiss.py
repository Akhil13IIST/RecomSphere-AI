import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "retrieval"

CHECKPOINT = MODEL_DIR / "two_tower.pt"
TRACK_IDS_FILE = MODEL_DIR / "track_ids.npy"
USER_IDS_FILE = MODEL_DIR / "user_ids.npy"

INDEX_FILE = MODEL_DIR / "tracks.faiss"
INFO_FILE = MODEL_DIR / "faiss_metadata.json"

print("Loading trained embeddings...")
start = time.time()

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True
)

track_vectors = checkpoint["track_embedding.weight"].numpy().astype(
    np.float32, copy=True
)

track_ids = np.load(TRACK_IDS_FILE, allow_pickle=True)
user_ids = np.load(USER_IDS_FILE, allow_pickle=True)

assert len(track_vectors) == len(track_ids), "Track mapping mismatch!"

print(f"Tracks: {len(track_ids):,}")
print(f"Embedding dimension: {track_vectors.shape[1]}")

print("\nNormalizing track embeddings...")
faiss.normalize_L2(track_vectors)

print("\nBuilding exact FAISS index...")
index = faiss.IndexFlatIP(track_vectors.shape[1])
index.add(track_vectors)

print(f"Indexed vectors: {index.ntotal:,}")

print("\nSaving index...")
faiss.write_index(index, str(INDEX_FILE))

metadata = {
    "index_type": "IndexFlatIP",
    "similarity": "cosine_similarity",
    "embedding_dim": int(track_vectors.shape[1]),
    "num_tracks": int(index.ntotal),
    "num_users": int(len(user_ids))
}

INFO_FILE.write_text(json.dumps(metadata, indent=4))

print("\nFAISS INDEX BUILD COMPLETE")
print(f"Index: {INDEX_FILE}")
print(f"Metadata: {INFO_FILE}")
print(f"Time: {(time.time() - start) / 60:.2f} minutes")
