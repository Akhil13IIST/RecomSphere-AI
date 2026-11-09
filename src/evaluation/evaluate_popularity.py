import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "retrieval"

start = time.time()

print("Loading model vocabularies...")
user_ids = np.load(MODEL_DIR / "user_ids.npy", allow_pickle=True)
track_ids = np.load(MODEL_DIR / "track_ids.npy", allow_pickle=True)

user_to_idx = {str(v): i for i, v in enumerate(user_ids)}
track_to_idx = {str(v): i for i, v in enumerate(track_ids)}

print("Loading interactions...")
train = pd.read_parquet(
    DATA / "train_interactions",
    columns=["user_id", "track_id", "play_count"]
)
valid = pd.read_parquet(
    DATA / "validation_interactions",
    columns=["user_id", "track_id", "play_count"]
)
test = pd.read_parquet(
    DATA / "test_interactions",
    columns=["user_id", "track_id"]
)

history = pd.concat([train, valid], ignore_index=True)

# Exact same ID mappings and history exclusion as two-tower evaluation.
seen = defaultdict(set)

for u, t in zip(history.user_id, history.track_id):
    ui = user_to_idx.get(str(u))
    ti = track_to_idx.get(str(t))
    if ui is not None and ti is not None:
        seen[ui].add(ti)

relevant = defaultdict(set)

for u, t in zip(test.user_id, test.track_id):
    ui = user_to_idx.get(str(u))
    ti = track_to_idx.get(str(t))

    if ui is not None and ti is not None and ti not in seen[ui]:
        relevant[ui].add(ti)

# Popularity scores from training + validation only.
popularity = history.groupby("track_id")["play_count"].sum()

ranked_tracks = sorted(
    (
        (track_to_idx[str(t)], count)
        for t, count in popularity.items()
        if str(t) in track_to_idx
    ),
    key=lambda x: x[1],
    reverse=True
)

ranked_track_indices = [t for t, _ in ranked_tracks]

eval_users = sorted(u for u in relevant if relevant[u])

precisions, recalls, ndcgs = [], [], []

print(f"Evaluation users: {len(eval_users):,}")
print(f"Relevant test tracks: {sum(len(v) for v in relevant.values()):,}")
print("Evaluating popularity recommendations...")

for ui in eval_users:
    recommendations = [
        ti for ti in ranked_track_indices
        if ti not in seen[ui]
    ][:10]

    hits = [int(ti in relevant[ui]) for ti in recommendations]

    precisions.append(sum(hits) / 10)
    recalls.append(sum(hits) / len(relevant[ui]))

    dcg = sum(
        hit / np.log2(rank + 2)
        for rank, hit in enumerate(hits)
    )

    ideal = min(len(relevant[ui]), 10)
    idcg = sum(
        1 / np.log2(rank + 2)
        for rank in range(ideal)
    )

    ndcgs.append(dcg / idcg if idcg else 0)

print("\n" + "=" * 45)
print("POPULARITY BASELINE - MATCHED EVALUATION")
print("=" * 45)
print(f"Users evaluated: {len(eval_users):,}")
print(f"Precision@10: {np.mean(precisions):.5f}")
print(f"Recall@10:    {np.mean(recalls):.5f}")
print(f"NDCG@10:      {np.mean(ndcgs):.5f}")
print(f"Time:         {(time.time()-start)/60:.2f} minutes")
print("=" * 45)

