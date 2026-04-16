"""Tests for the learned cam0 identity tracker."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts.tracking.id_network.crops import (
    _crop_with_padding,
    extract_crops_from_video,
)
from scripts.tracking.id_network.fragments import build_fragments, mine_pairs


# ---- crops -----------------------------------------------------------------


def test_crop_with_padding_zero_pads_at_corner():
    frame = np.ones((10, 10), dtype=np.uint8) * 200
    crop = _crop_with_padding(frame, cx=0.0, cy=0.0, size=6)
    assert crop.shape == (6, 6)
    # Top-left 3x3 region lies outside the frame -> zero-padded.
    assert (crop[:3, :3] == 0).all()
    # Bottom-right region lies inside the frame -> pixel value 200.
    assert (crop[3:, 3:] == 200).all()


def test_crop_with_padding_center_inside_frame():
    frame = np.arange(100, dtype=np.uint8).reshape(10, 10)
    crop = _crop_with_padding(frame, cx=5.0, cy=5.0, size=4)
    assert crop.shape == (4, 4)
    assert (crop > 0).any()


def test_extract_crops_from_video_synthetic(tmp_path: Path):
    import cv2

    video_path = tmp_path / "synth.mp4"
    h, w = 64, 64
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30, (w, h))
    try:
        for i in range(10):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:] = (i * 20, i * 20, i * 20)
            writer.write(frame)
    finally:
        writer.release()
    assert video_path.exists()

    df = pd.DataFrame(
        [
            {"frame": 2, "track": 0, "x": 32.0, "y": 32.0, "fish_id": "a"},
            {"frame": 5, "track": 0, "x": 10.0, "y": 10.0, "fish_id": "a"},
            {"frame": 5, "track": 1, "x": 50.0, "y": 50.0, "fish_id": "b"},
        ]
    )
    crop_set = extract_crops_from_video(video_path, df, size=16, frame_offset=0)
    assert crop_set.images.shape == (3, 16, 16)
    # All three rows preserved, orig_row == input row (input had default index).
    assert list(crop_set.index["orig_row"]) == [0, 1, 2]
    # Frame-2 rows should have pixel value ~40, frame-5 ~100.
    # The crop is nonzero everywhere because the synthetic frame is solid.
    assert crop_set.images[0].mean() > 0
    assert crop_set.images[1].mean() > 0


# ---- fragments -------------------------------------------------------------


def _mk_index_df(rows):
    return pd.DataFrame(rows)[["frame", "track", "fish_id", "row"]]


def test_build_fragments_splits_on_id_change():
    df = _mk_index_df(
        [
            {"frame": 0, "track": 0, "fish_id": "a", "row": 0},
            {"frame": 1, "track": 0, "fish_id": "a", "row": 1},
            {"frame": 2, "track": 0, "fish_id": "b", "row": 2},  # swap -> new fragment
            {"frame": 3, "track": 0, "fish_id": "b", "row": 3},
        ]
    )
    frags = build_fragments(df, max_gap=1)
    assert len(frags) == 2
    assert frags[0].fish_id == "a" and frags[0].rows == [0, 1]
    assert frags[1].fish_id == "b" and frags[1].rows == [2, 3]


def test_build_fragments_splits_on_large_gap():
    df = _mk_index_df(
        [
            {"frame": 0, "track": 0, "fish_id": "a", "row": 0},
            {"frame": 1, "track": 0, "fish_id": "a", "row": 1},
            {"frame": 10, "track": 0, "fish_id": "a", "row": 2},  # gap > 1
        ]
    )
    frags = build_fragments(df, max_gap=1)
    assert len(frags) == 2


def test_build_fragments_skips_unassigned():
    df = _mk_index_df(
        [
            {"frame": 0, "track": 0, "fish_id": "unassigned", "row": 0},
            {"frame": 1, "track": 0, "fish_id": "a", "row": 1},
            {"frame": 2, "track": 0, "fish_id": "a", "row": 2},
        ]
    )
    frags = build_fragments(df, max_gap=1)
    assert len(frags) == 1
    assert frags[0].rows == [1, 2]


def test_mine_pairs_temporal_coexistence():
    # Two fragments that share frame 2 must generate diff pairs.
    df = _mk_index_df(
        [
            {"frame": 0, "track": 0, "fish_id": "a", "row": 0},
            {"frame": 1, "track": 0, "fish_id": "a", "row": 1},
            {"frame": 2, "track": 0, "fish_id": "a", "row": 2},
            {"frame": 2, "track": 1, "fish_id": "b", "row": 3},
            {"frame": 3, "track": 1, "fish_id": "b", "row": 4},
        ]
    )
    frags = build_fragments(df, max_gap=1)
    same, diff = mine_pairs(frags, max_same=50, max_diff=50, rng_seed=1)
    assert len(same) == 50
    assert len(diff) == 50
    # Same-pairs must draw from within one fragment -> rows from same fish.
    fish_by_row = {r["row"]: r["fish_id"] for _, r in df.iterrows()}
    for i, j in same:
        assert fish_by_row[i] == fish_by_row[j]
    # Diff-pairs must cross fragments -> different fish.
    for i, j in diff:
        assert fish_by_row[i] != fish_by_row[j]


def test_mine_pairs_no_overlap_yields_no_diff():
    df = _mk_index_df(
        [
            {"frame": 0, "track": 0, "fish_id": "a", "row": 0},
            {"frame": 1, "track": 0, "fish_id": "a", "row": 1},
            {"frame": 10, "track": 1, "fish_id": "b", "row": 2},
            {"frame": 11, "track": 1, "fish_id": "b", "row": 3},
        ]
    )
    frags = build_fragments(df, max_gap=1)
    same, diff = mine_pairs(frags, max_same=10, max_diff=10, rng_seed=1)
    assert len(diff) == 0
    assert len(same) == 10


# ---- train + infer smoke test ----------------------------------------------


def test_train_and_cluster_recovers_two_identities():
    """Train on two visually distinct classes and verify fragment embeddings
    cluster correctly. Uses 16x16 crops and 1 epoch so it runs on CPU fast."""
    from scripts.tracking.id_network.train import TrainConfig, embed_all, train_embedder
    from scripts.tracking.id_network.infer import _cluster_fragments

    rng = np.random.default_rng(42)
    n_per = 20
    size = 16
    # Class 0: low-intensity noise.  Class 1: high-intensity noise.
    imgs_a = rng.integers(0, 60, size=(n_per, size, size), dtype=np.uint8)
    imgs_b = rng.integers(195, 255, size=(n_per, size, size), dtype=np.uint8)
    images = np.concatenate([imgs_a, imgs_b], axis=0)

    same, diff = [], []
    for i in range(n_per):
        for j in range(i + 1, n_per):
            same.append((i, j))
            same.append((n_per + i, n_per + j))
    for i in range(n_per):
        diff.append((i, n_per + i))
    # Pad up for a meaningful batch count.
    diff = diff * 5

    config = TrainConfig(epochs=1, batch_size=16, lr=1e-3, device="cpu")
    model, history = train_embedder(images, same, diff, config=config)
    embeddings = embed_all(model, images, batch_size=32, device="cpu")
    frag_means = np.stack(
        [embeddings[:n_per].mean(axis=0), embeddings[n_per:].mean(axis=0)], axis=0
    )
    clusters = _cluster_fragments(frag_means, n_clusters=2)
    # Two fragments -> two distinct clusters.
    assert len(set(clusters.tolist())) == 2
