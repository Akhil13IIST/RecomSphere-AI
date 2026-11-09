import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "retrieval"
OUTPUT_DIR = ROOT / "models" / "collaborative"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K_NEIGHBORS = 50
TOP_K = 10

start = time.time()

print("Loading mappings...")
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

del train, valid

print("Mapping interactions...")
history["u"] = history["user_id"].astype(str).map(user_to_idx)
history["t"] = history["track_id"].astype(str).map(track_to_idx)
history = history.dropna(subset=["u", "t"])

history["u"] = history["u"].astype(np.int32)
history["t"] = history["t"].astype(np.int32)

# Implicit feedback with logarithmic play-count scaling.
history["weight"] = np.log1p(
    history["play_count"].astype(np.float32)
)

matrix = csr_matrix(
    (
        history["weight"].to_numpy(dtype=np.float32),
        (
            history["u"].to_numpy(),
            history["t"].to_numpy()
        )
    ),
    shape=(len(user_ids), len(track_ids)),
    dtype=np.float32
)

matrix.sum_duplicates()

del history

print("Interaction matrix:", matrix.shape)
print("Non-zero interactions:", matrix.nnz)

# IDF-style weighting reduces the influence of globally popular tracks.
print("Applying track IDF weighting...")
track_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel()

idf = np.log1p(
    matrix.shape[0] / (1 + track_frequency)
).astype(np.float32)

matrix = matrix.multiply(idf).tocsr()
matrix = normalize(matrix, norm="l2", axis=1)
# Save trained collaborative filtering artifacts
from scipy.sparse import save_npz

save_npz(OUTPUT_DIR / "user_item_matrix.npz", matrix)

np.save(
    OUTPUT_DIR / "user_similarity.npy",
    similarity if "similarity" in globals() else np.zeros((0, 0))
)
print("Computing user similarity...")
similarity = cosine_similarity(matrix, dense_output=True)
np.fill_diagonal(similarity, 0)
from scipy.sparse import save_npz

save_npz(OUTPUT_DIR / "user_item_matrix.npz", matrix)
np.save(OUTPUT_DIR / "user_similarity.npy", similarity)
np.save(OUTPUT_DIR / "user_ids.npy", user_ids)
np.save(OUTPUT_DIR / "track_ids.npy", track_ids)

print("Collaborative filtering model artifacts saved.")

print("Generating recommendations...")

test["u"] = test["user_id"].astype(str).map(user_to_idx)
test["t"] = test["track_id"].astype(str).map(track_to_idx)

relevant = defaultdict(set)

for u, t in zip(test["u"], test["t"]):
    if pd.notna(u) and pd.notna(t):
        relevant[int(u)].add(int(t))

seen = [
    set(matrix[u].indices)
    for u in range(matrix.shape[0])
]

# Evaluate on users with at least one eligible test track.
eligible = {
    u: targets - seen[u]
    for u, targets in relevant.items()
}

eval_users = sorted(
    u for u, targets in eligible.items() if targets
)

print("Evaluation users:", len(eval_users))

precisions, recalls, ndcgs = [], [], []

for count, u in enumerate(eval_users, 1):
    neighbors = np.argsort(similarity[u])[-K_NEIGHBORS:]
    neighbor_scores = similarity[u, neighbors]

    scores = np.asarray(
        matrix[neighbors].T @ neighbor_scores
    ).ravel()

    scores[list(seen[u])] = -np.inf

    candidates = np.argpartition(
        scores, -TOP_K
    )[-TOP_K:]

    recommendations = candidates[
        np.argsort(scores[candidates])[::-1]
    ]

    targets = eligible[u]

    hits = [
        int(t in targets)
        for t in recommendations
    ]

    precisions.append(sum(hits) / TOP_K)
    recalls.append(sum(hits) / len(targets))

    dcg = sum(
        hit / np.log2(rank + 2)
        for rank, hit in enumerate(hits)
    )

    ideal = min(len(targets), TOP_K)
    idcg = sum(
        1 / np.log2(rank + 2)
        for rank in range(ideal)
    )

    ndcgs.append(dcg / idcg if idcg else 0)

    if count % 200 == 0:
        print(f"Evaluated {count}/{len(eval_users)} users")

print("\n" + "=" * 45)
print("USER-BASED COLLABORATIVE FILTERING")
print("=" * 45)
print(f"Users evaluated: {len(eval_users)}")
print(f"Neighbors: {K_NEIGHBORS}")
print(f"Precision@10: {np.mean(precisions):.5f}")
print(f"Recall@10:    {np.mean(recalls):.5f}")
print(f"NDCG@10:      {np.mean(ndcgs):.5f}")
print(f"Time:         {(time.time()-start)/60:.2f} minutes")
print("=" * 45)
