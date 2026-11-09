from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from fastapi import FastAPI, HTTPException, Query

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "collaborative"

K_NEIGHBORS = 50

user_ids = None
track_ids = None
user_to_idx = None
user_item_matrix = None
user_similarity = None
metadata_path = MODEL_DIR / "track_metadata.parquet"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global user_ids, track_ids, user_to_idx
    global user_item_matrix, user_similarity

    print("Loading collaborative filtering model...")

    user_ids = np.load(MODEL_DIR / "user_ids.npy", allow_pickle=True)

    with np.load(MODEL_DIR / "track_ids_compact.npz", allow_pickle=False) as saved:
        track_ids = saved["track_ids"]

    user_to_idx = {str(v): i for i, v in enumerate(user_ids)}

    user_item_matrix = load_npz(MODEL_DIR / "user_item_matrix.npz")

    user_similarity = np.load(
        MODEL_DIR / "user_similarity.npy",
        mmap_mode="r"
    )

    print(f"Loaded {len(user_ids):,} users")
    print(f"Loaded {len(track_ids):,} tracks")
    print("Collaborative filtering model ready.")

    yield

    print("API shutting down.")


app = FastAPI(
    title="RecomSphere AI",
    description="Music recommendation API powered by user-based collaborative filtering.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    return {
        "project": "RecomSphere AI",
        "status": "running",
        "model": "User-Based Collaborative Filtering",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": user_item_matrix is not None,
        "num_users": len(user_ids),
        "num_tracks": len(track_ids)
    }


@app.get("/users")
def get_users(limit: int = Query(default=50, ge=1, le=971)):
    return {
        "total_users": len(user_ids),
        "users": [str(u) for u in user_ids[:limit]]
    }


@app.get("/recommend/{user_id}")
def recommend(
    user_id: str,
    top_k: int = Query(default=10, ge=1, le=50)
):
    ui = user_to_idx.get(user_id)

    if ui is None:
        raise HTTPException(
            status_code=404,
            detail=f"User '{user_id}' not found in the training catalogue."
        )

    neighbors = np.argsort(user_similarity[ui])[-K_NEIGHBORS:]
    neighbor_scores = user_similarity[ui, neighbors]

    scores = np.asarray(
        user_item_matrix[neighbors].T @ neighbor_scores
    ).ravel()

    seen_tracks = user_item_matrix[ui].indices
    scores[seen_tracks] = -np.inf

    candidates = np.argpartition(scores, -top_k)[-top_k:]
    ranked = candidates[np.argsort(scores[candidates])[::-1]]

    selected_ids = [
        track_ids[idx].tobytes().decode("ascii")
        for idx in ranked
    ]

    metadata = pd.read_parquet(
        metadata_path,
        columns=["track_id", "track_name", "artist_name"],
        filters=[("track_id", "in", selected_ids)]
    )

    metadata_map = {
        str(row.track_id): (str(row.track_name), str(row.artist_name))
        for row in metadata.itertuples(index=False)
    }

    recommendations = []

    for rank, (track_idx, track_id) in enumerate(
        zip(ranked, selected_ids), start=1
    ):
        name, artist = metadata_map.get(
            track_id, ("Unknown track", "Unknown artist")
        )

        recommendations.append({
            "rank": rank,
            "track_id": track_id,
            "track_name": name,
            "artist_name": artist,
            "score": round(float(scores[track_idx]), 6)
        })

    return {
        "user_id": user_id,
        "model": "User-Based Collaborative Filtering",
        "recommendation_count": len(recommendations),
        "recommendations": recommendations
    }
