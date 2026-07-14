from __future__ import annotations

import importlib

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from sam2_id_experiment.src.id_model.geometry_contrastive import (
    GeometryPairDataset,
    StratifiedPairBatchSampler,
    assign_whole_block_splits,
    load_explicit_pairs,
    load_samples,
    margin_contrastive_loss,
    select_distance_threshold,
    write_fit_membership,
    write_split_manifest,
)


def _write_minimal_contract(root, samples: pd.DataFrame, positives: pd.DataFrame, negatives: pd.DataFrame):
    images = np.zeros((len(samples), 16, 16, 3), dtype=np.uint8)
    for idx in range(len(images)):
        images[idx, 4:12, 3:13] = (20 + idx, 50, 100)
    with h5py.File(root / "crops.h5", "w") as h5:
        h5.create_dataset("crops", data=images)
        h5.attrs["color_order"] = "RGB"
    samples.to_parquet(root / "samples.parquet", index=False)
    positives.to_parquet(root / "positive_pairs.parquet", index=False)
    negatives.to_parquet(root / "negative_pairs.parquet", index=False)


def test_whole_block_split_is_deterministic_and_normalizes_validation():
    samples = pd.DataFrame(
        {
            "sample_id": range(8),
            "crop_index": range(8),
            "block_id": np.repeat(["a", "b", "c", "d"], 2),
        }
    )
    first, block_col = assign_whole_block_splits(samples, seed=7, block_column="block_id")
    second, _ = assign_whole_block_splits(samples.sample(frac=1, random_state=2), seed=7, block_column="block_id")
    assert block_col == "block_id"
    assert first.groupby("block_id")["split"].nunique().max() == 1
    assert first.set_index("sample_id")["split"].to_dict() == second.set_index("sample_id")["split"].to_dict()

    supplied = samples.copy()
    supplied["split"] = supplied["block_id"].map(
        {"a": "train", "b": "train", "c": "validation", "d": "test"}
    )
    normalized, _ = assign_whole_block_splits(supplied, block_column="block_id")
    assert set(normalized["split"]) == {"train", "calibration", "test"}


def test_explicit_pairs_keep_cross_camera_and_filter_adjacent_same_fragment(tmp_path):
    samples = pd.DataFrame(
        {
            "sample_id": [10, 11, 12, 13, 14, 15],
            "crop_index": range(6),
            "frame": [0, 4, 0, 0, 20, 20],
            "view": ["top", "top", "side", "top", "top", "top"],
            "block_id": ["a"] * 6,
            "split": ["train"] * 6,
        }
    )
    positives = pd.DataFrame(
        {
            "sample_a_id": [10, 10, 10],
            "sample_b_id": [11, 12, 14],
            "pair_type": ["same_fragment", "cross_camera", "same_fragment"],
        }
    )
    negatives = pd.DataFrame(
        {"sample_a_id": [10, 14], "sample_b_id": [13, 15], "pair_type": ["simultaneous", "simultaneous"]}
    )
    _write_minimal_contract(tmp_path, samples, positives, negatives)
    loaded, _ = load_samples(tmp_path)
    pairs, audit, _ = load_explicit_pairs(tmp_path, loaded, min_positive_frame_gap=10)
    assert audit["positive_too_close_removed"] == 1
    assert set(pairs[pairs.label == 1]["pair_subtype"]) == {"same_fragment", "cross_camera"}
    assert (pairs[pairs.label == 0]["frame_a"] == pairs[pairs.label == 0]["frame_b"]).all()


def test_non_simultaneous_negative_is_rejected(tmp_path):
    samples = pd.DataFrame(
        {
            "sample_id": [0, 1, 2],
            "crop_index": [0, 1, 2],
            "frame": [0, 20, 4],
            "view": ["top"] * 3,
            "block_id": ["a"] * 3,
            "split": ["train"] * 3,
        }
    )
    positives = pd.DataFrame({"sample_a_id": [0], "sample_b_id": [1], "pair_type": ["same_fragment"]})
    negatives = pd.DataFrame({"sample_a_id": [0], "sample_b_id": [2]})
    _write_minimal_contract(tmp_path, samples, positives, negatives)
    loaded, _ = load_samples(tmp_path)
    with pytest.raises(ValueError, match="not simultaneous"):
        load_explicit_pairs(tmp_path, loaded, min_positive_frame_gap=1)


def test_stratified_sampler_retains_both_positive_sources_and_balances_labels():
    pairs = pd.DataFrame(
        {
            "label": [0] * 20 + [1] * 20 + [1] * 2,
            "pair_subtype": ["simultaneous_frame"] * 20 + ["same_fragment"] * 20 + ["cross_camera"] * 2,
        }
    )
    sampler = StratifiedPairBatchSampler(pairs, batch_size=12, seed=3, max_steps=1)
    batch = next(iter(sampler))
    selected = pairs.iloc[batch]
    assert int((selected.label == 0).sum()) == 6
    assert int((selected.label == 1).sum()) == 6
    assert set(selected[selected.label == 1].pair_subtype) == {"same_fragment", "cross_camera"}


def test_margin_loss_and_calibration_threshold_have_expected_direction():
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    good_b = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    bad_b = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    labels = torch.tensor([1.0, 0.0])
    assert margin_contrastive_loss(a, good_b, labels).item() == pytest.approx(0.0)
    assert margin_contrastive_loss(a, bad_b, labels).item() > 0
    selected = select_distance_threshold(np.array([1, 1, 0, 0]), np.array([0.1, 0.2, 1.0, 1.2]))
    assert 0.2 < selected["threshold"] < 1.0
    assert selected["balanced_accuracy"] == pytest.approx(1.0)


def test_rgb_crop_reader_preserves_black_background(tmp_path):
    samples = pd.DataFrame({"sample_id": [0, 1], "crop_index": [0, 1]})
    positives = pd.DataFrame({"sample_a_id": [0], "sample_b_id": [1]})
    negatives = pd.DataFrame({"sample_a_id": [0], "sample_b_id": [1]})
    _write_minimal_contract(tmp_path, samples, positives, negatives)
    pairs = pd.DataFrame({"crop_index_a": [0], "crop_index_b": [1], "label": [1.0]})
    dataset = GeometryPairDataset(tmp_path / "crops.h5", pairs, augment=True)
    crop_a, _, _ = dataset[0]
    assert crop_a.shape == (3, 16, 16)
    assert torch.count_nonzero(crop_a[:, 0, 0]) == 0


def test_declared_bgr_contract_fails_closed(tmp_path):
    samples = pd.DataFrame({"sample_id": [0, 1], "crop_index": [0, 1]})
    positives = pd.DataFrame({"sample_a_id": [0], "sample_b_id": [1]})
    negatives = pd.DataFrame({"sample_a_id": [0], "sample_b_id": [1]})
    _write_minimal_contract(tmp_path, samples, positives, negatives)
    with h5py.File(tmp_path / "crops.h5", "r+") as h5:
        h5["crops"].attrs["color_order"] = "BGR"
    with pytest.raises(ValueError, match="requires RGB"):
        load_samples(tmp_path)


def test_missing_color_order_contract_fails_closed(tmp_path):
    samples = pd.DataFrame({"sample_id": [0, 1], "crop_index": [0, 1]})
    positives = pd.DataFrame({"sample_a_id": [0], "sample_b_id": [1]})
    negatives = pd.DataFrame({"sample_a_id": [0], "sample_b_id": [1]})
    _write_minimal_contract(tmp_path, samples, positives, negatives)
    with h5py.File(tmp_path / "crops.h5", "r+") as h5:
        del h5.attrs["color_order"]
    with pytest.raises(ValueError, match="explicitly declare color_order=RGB"):
        load_samples(tmp_path)


def test_synthetic_cpu_contract_and_split_manifest_hash(tmp_path):
    script = importlib.import_module(
        "sam2_id_experiment.scripts.49_train_geometry_contrastive_embedding"
    )
    dataset = script.write_synthetic_smoke_contract(tmp_path / "synthetic", seed=5)
    samples, _ = load_samples(dataset)
    assigned, block_col = assign_whole_block_splits(
        samples, seed=5, block_column="split_block_id"
    )
    path_a, digest_a = write_split_manifest(
        assigned, tmp_path / "manifest_a.parquet", block_column=block_col
    )
    _, digest_b = write_split_manifest(
        assigned.sample(frac=1, random_state=10), tmp_path / "manifest_b.parquet", block_column=block_col
    )
    assert path_a.exists()
    assert digest_a == digest_b
    assert len(digest_a) == 64

    train_pairs = pd.DataFrame(
        {"crop_index_a": [0, 1], "crop_index_b": [1, 2]}
    )
    # Use crops known to be in one train block to exercise the exact-fit manifest.
    train_block = assigned.loc[assigned["split"] == "train", block_col].iloc[0]
    train_rows = assigned[assigned[block_col] == train_block].head(3).copy()
    train_pairs["crop_index_a"] = [train_rows.crop_index.iloc[0], train_rows.crop_index.iloc[1]]
    train_pairs["crop_index_b"] = [train_rows.crop_index.iloc[1], train_rows.crop_index.iloc[2]]
    fit_path, fit_digest, membership = write_fit_membership(
        assigned,
        train_pairs,
        tmp_path / "fit_membership.parquet",
        block_column=block_col,
    )
    assert fit_path.exists() and len(fit_digest) == 64
    assert set(membership["split"]) == {"train"}
    assert set(membership["crop_index"]) == set(train_pairs[["crop_index_a", "crop_index_b"]].to_numpy().ravel())
