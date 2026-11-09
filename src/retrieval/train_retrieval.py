import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DATA = ROOT / "data" / "processed" / "train_interactions"
VALID_DATA = ROOT / "data" / "processed" / "validation_interactions"
OUTPUT = ROOT / "models" / "retrieval"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Resource-conscious configuration
EMBEDDING_DIM = 64
BATCH_SIZE = 1024
EPOCHS = 3
LEARNING_RATE = 0.05
TEMPERATURE = 0.07
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True

print("\nLoading training interactions...")

train_df = pd.read_parquet(
    TRAIN_DATA,
    columns=["user_id", "track_id", "play_count"]
)
valid_df = pd.read_parquet(
    VALID_DATA,
    columns=["user_id", "track_id", "play_count"]
)

df = pd.concat([train_df, valid_df], ignore_index=True)
df = df.groupby(
    ["user_id", "track_id"], as_index=False
)["play_count"].sum()

del train_df, valid_df

print(f"Training pairs: {len(df):,}")

# Build vocabularies from training data only
user_codes, user_ids = pd.factorize(df["user_id"], sort=True)
track_codes, track_ids = pd.factorize(df["track_id"], sort=True)

num_users = len(user_ids)
num_tracks = len(track_ids)

print(f"Unique users: {num_users:,}")
print(f"Unique tracks: {num_tracks:,}")

users = torch.from_numpy(user_codes.astype(np.int64))
tracks = torch.from_numpy(track_codes.astype(np.int64))

weights = np.log1p(df["play_count"].to_numpy(dtype=np.float32))
weights = weights / max(weights.mean(), 1e-8)
weights = np.clip(weights, 0.5, 2.0)
sample_weights = torch.from_numpy(weights.astype(np.float32))

del df, user_codes, track_codes, weights

# Save vocabularies for inference and evaluation
np.save(OUTPUT / "user_ids.npy", np.asarray(user_ids, dtype=str))
np.save(OUTPUT / "track_ids.npy", np.asarray(track_ids, dtype=str))

class TwoTowerRetrieval(nn.Module):
    def __init__(self, n_users, n_tracks, dim):
        super().__init__()

        self.user_embedding = nn.Embedding(
            n_users, dim, sparse=True
        )
        self.track_embedding = nn.Embedding(
            n_tracks, dim, sparse=True
        )

        nn.init.normal_(self.user_embedding.weight, std=0.02)
        nn.init.normal_(self.track_embedding.weight, std=0.02)

    def forward(self, user_ids, track_ids):
        user_vectors = F.normalize(
            self.user_embedding(user_ids), dim=1
        )
        track_vectors = F.normalize(
            self.track_embedding(track_ids), dim=1
        )
        return user_vectors, track_vectors


model = TwoTowerRetrieval(
    num_users, num_tracks, EMBEDDING_DIM
).to(device)

optimizer = torch.optim.SparseAdam(
    model.parameters(), lr=LEARNING_RATE
)

n = len(users)

print("\nStarting two-tower training...")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Embedding dimension: {EMBEDDING_DIM}")

for epoch in range(EPOCHS):
    model.train()
    start_time = time.time()

    permutation = torch.randperm(n)
    total_loss = 0.0
    batches = 0

    for start in range(0, n, BATCH_SIZE):
        indices = permutation[start:start + BATCH_SIZE]

        batch_users = users[indices].to(device)
        batch_tracks = tracks[indices].to(device)
        batch_weights = sample_weights[indices].to(device)

        user_vectors, track_vectors = model(
            batch_users, batch_tracks
        )

        logits = (
            user_vectors @ track_vectors.T
        ) / TEMPERATURE

        batch_size = len(batch_users)
        diagonal = torch.eye(
            batch_size, dtype=torch.bool, device=device
        )

        # Mask other known positives in this batch.
        same_user = (
            batch_users[:, None] == batch_users[None, :]
        )
        same_track = (
            batch_tracks[:, None] == batch_tracks[None, :]
        )

        false_negatives = (
            (same_user | same_track) & ~diagonal
        )

        logits = logits.masked_fill(false_negatives, -1e9)

        labels = torch.arange(batch_size, device=device)

        per_example_loss = F.cross_entropy(
            logits, labels, reduction="none"
        )

        loss = (
            per_example_loss * batch_weights
        ).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1

        if batches % 500 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {batches}/{(n + BATCH_SIZE - 1)//BATCH_SIZE} | "
                f"Loss: {loss.item():.4f}"
            )

    elapsed = time.time() - start_time
    average_loss = total_loss / max(batches, 1)

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS} completed | "
        f"Average loss: {average_loss:.4f} | "
        f"Time: {elapsed / 60:.1f} min\n"
    )

# Save trained model
checkpoint = {
    key: value.detach().cpu()
    for key, value in model.state_dict().items()
}

torch.save(checkpoint, OUTPUT / "two_tower.pt")

metadata = {
    "model": "TwoTowerRetrieval",
    "embedding_dim": EMBEDDING_DIM,
    "num_users": num_users,
    "num_tracks": num_tracks,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "temperature": TEMPERATURE,
    "seed": SEED,
    "training_pairs": n
}

with open(OUTPUT / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print("\nTWO-TOWER TRAINING COMPLETE")
print(f"Model saved: {OUTPUT / 'two_tower.pt'}")
print(f"Metadata saved: {OUTPUT / 'metadata.json'}")

if device.type == "cuda":
    print(
        f"Peak GPU memory: "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
    )




