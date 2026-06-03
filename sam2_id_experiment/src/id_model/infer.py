from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sam2_id_experiment.src.id_model.model import SmallMedakaEmbeddingNet


def embed_crops(
    *,
    dataset_dir: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 128,
    device: str = "cpu",
) -> dict:
    import h5py

    dataset_dir = Path(dataset_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(model_path, map_location=device)
    embedding_dim = int(checkpoint.get("embedding_dim", 64))
    model = SmallMedakaEmbeddingNet(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with h5py.File(dataset_dir / "crops.h5", "r") as h5:
        crops = np.asarray(h5["crops"][...], dtype=np.float32) / 255.0
    meta = pd.read_parquet(dataset_dir / "crop_metadata.parquet")

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            batch = torch.from_numpy(crops[start:start + batch_size]).permute(0, 3, 1, 2).to(device)
            embeddings.append(model(batch).cpu().numpy())
    crop_embeddings = np.concatenate(embeddings, axis=0)
    crop_path = out_dir / "crop_embeddings.npy"
    np.save(crop_path, crop_embeddings)

    track_rows: list[dict] = []
    track_embeddings: list[np.ndarray] = []
    for tid, group in meta.groupby("tracklet_id"):
        emb = crop_embeddings[group.index.to_numpy()]
        mean_emb = emb.mean(axis=0)
        track_embeddings.append(mean_emb)
        track_rows.append({
            "tracklet_id": int(tid),
            "embedding_index": len(track_embeddings) - 1,
            "n_crops": int(len(group)),
            "assigned_identity": "unassigned",
        })
    track_path = out_dir / "tracklet_embeddings.npy"
    np.save(track_path, np.stack(track_embeddings, axis=0) if track_embeddings else np.zeros((0, embedding_dim)))
    assignments = pd.DataFrame(track_rows)
    assign_path = out_dir / "tracklet_identity_assignments.parquet"
    assignments.to_parquet(assign_path, index=False)
    return {
        "crop_embeddings": str(crop_path),
        "tracklet_embeddings": str(track_path),
        "assignments": str(assign_path),
    }
