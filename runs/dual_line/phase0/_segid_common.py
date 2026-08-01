"""Shared loaders for Phase-0 dual-line diagnostics.

Replicates the model-loading + forward helpers from
``sam2_id_experiment/scripts/44_apply_clean_female_anchor.py`` (module name starts
with a digit, so it cannot be imported normally). Kept byte-identical in behaviour
so P(female)/embeddings match the production female-anchor path exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.append(str(_REPO))

from sam2_id_experiment.src.id_model.resnet_embedder import (  # noqa: E402
    EmbeddingClassifier,
    ResNet18Embedder,
)

SEGID = Path("Z:/StephensU/medaka/runs/udmt_validation/segid/K_1_2")
FEMALE_MODEL = SEGID / "model_female_clean" / "model.pt"
EMBEDDER_MODEL = SEGID / "model_resnet_full" / "model.pt"


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_embedder(path: str | Path, dev: str) -> ResNet18Embedder:
    ck = torch.load(path, map_location=dev)
    m = ResNet18Embedder(embedding_dim=int(ck.get("embedding_dim", 128)), pretrained=False).to(dev)
    m.load_state_dict(ck["model_state"])
    return m.eval()


def load_classifier(path: str | Path, dev: str) -> EmbeddingClassifier:
    ck = torch.load(path, map_location=dev)
    m = EmbeddingClassifier(ResNet18Embedder(embedding_dim=int(ck.get("embedding_dim", 128)), pretrained=False), 2)
    m.load_state_dict(ck["model_state"])
    return m.to(dev).eval()


def forward(model, crops_uint8: np.ndarray, dev: str, head: bool, batch_size: int = 512) -> np.ndarray:
    """crops_uint8: (N,H,W,3) uint8 BGR. head=True -> P(female); head=False -> (N,D) embeddings."""
    out = []
    with torch.no_grad():
        for s in range(0, len(crops_uint8), batch_size):
            x = torch.from_numpy(crops_uint8[s:s + batch_size].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(dev)
            z = torch.softmax(model(x), 1)[:, 1] if head else model(x)
            out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0)


def load_crops(h5_path: str | Path, rows: np.ndarray) -> np.ndarray:
    """rows must be sorted + unique for h5py fancy indexing."""
    with h5py.File(h5_path, "r") as h:
        key = "crops" if "crops" in h else list(h.keys())[0]
        return np.asarray(h[key][rows], dtype=np.uint8)
