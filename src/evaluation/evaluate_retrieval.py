import time
from pathlib import Path
from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "retrieval"
DATA_DIR = ROOT / "data" / "processed"

print("Loading mappings and FAISS index...")
start = time.time()

user_ids = np.load(MODEL_DIR / "user_ids.npy", allow_pickle=True)
track_ids = np.load(MODEL_DIR / "track_ids.npy", allow_pickle=True)

user_to_idx = {str(v): i for i, v in enumerate(user_ids)}
track_to_idx = {str(v): i for i, v in enumerate(track_ids)}

checkpoint = torch.load(
    MODEL_DIR / "two_tower.pt",
    map_location="cpu",
    weights_only=True
)

user_vectors = checkpoint["user_embedding.weight"].numpy().astype(np.float32)
faiss.normalize_L2(user_vectors)

index = faiss.read_index(str(MODEL_DIR / "tracks.faiss"))

print("Loading interaction data...")
train = pd.read_parquet(DATA_DIR / "train_interactions",
                        columns=["user_id", "track_id"])
valid = pd.read_parquet(DATA_DIR / "validation_interactions",
                        columns=["user_id", "track_id"])
test = pd.read_parquet(DATA_DIR / "test_interactions",
                       columns=["user_id", "track_id"])

# Build history from training and validation periods.
seen = defaultdict(set)

for frame in (train, valid):
    for u, t in zip(frame["user_id"], frame["track_id"]):
        ui = user_to_idx.get(str(u))
        ti = track_to_idx.get(str(t))
        if ui is not None and ti is not None:
            seen[ui].add(ti)

# Build relevant held-out test tracks.
relevant = defaultdict(set)

for u, t in zip(test["user_id"], test["track_id"]):
    ui = user_to_idx.get(str(u))
    ti = track_to_idx.get(str(t))

    if ui is not None and ti is not None and ti not in seen[ui]:
        relevant[ui].add(ti)

eval_users = sorted(u for u in relevant if relevant[u])

print(f"Evaluation users: {len(eval_users):,}")
print(f"Relevant test interactions: {sum(map(len, relevant.values())):,}")

if not eval_users:
    raise RuntimeError("No eligible test interactions found.")

# Retrieve 500 candidates per user, then remove previously seen tracks.
print("\nSearching FAISS index...")
queries = user_vectors[eval_users].copy()
distances, indices = index.search(queries, 500)

precisions, recalls, ndcgs = [], [], []

for row, ui in enumerate(eval_users):
    recommendations = [
        int(ti) for ti in indices[row]
        if ti >= 0 and int(ti) not in seen[ui]
    ][:10]

    targets = relevant[ui]
    hits = [1 if ti in targets else 0 for ti in recommendations]

    precision = sum(hits) / 10
    recall = sum(hits) / len(targets)

    dcg = sum(
        hit / np.log2(rank + 2)
        for rank, hit in enumerate(hits)
    )
    ideal_hits = min(len(targets), 10)
    idcg = sum(
        1 / np.log2(rank + 2)
        for rank in range(ideal_hits)
    )
    ndcg = dcg / idcg if idcg else 0.0

    precisions.append(precision)
    recalls.append(recall)
    ndcgs.append(ndcg)

print("\n" + "=" * 45)
print("RETRIEVAL EVALUATION RESULTS")
print("=" * 45)
print(f"Users evaluated: {len(eval_users):,}")
print(f"Precision@10: {np.mean(precisions):.5f}")
print(f"Recall@10:    {np.mean(recalls):.5f}")
print(f"NDCG@10:      {np.mean(ndcgs):.5f}")
print(f"Time:         {(time.time() - start) / 60:.2f} minutes")
print("=" * 45)

