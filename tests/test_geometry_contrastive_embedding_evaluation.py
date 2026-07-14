from __future__ import annotations

import importlib
import json

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from sam2_id_experiment.src.evaluation.geometry_contrastive_embedding import (
    cross_camera_retrieval,
    score_pairs,
    temporal_block_bootstrap_metrics,
    trusted_prototype_similarities,
    validate_split_integrity,
)


evaluator = importlib.import_module(
    "sam2_id_experiment.scripts.50_evaluate_geometry_contrastive_embedding"
)


def test_new_trainer_checkpoint_architecture_loads(monkeypatch, tmp_path):
    from sam2_id_experiment.src.id_model.geometry_contrastive import (
        GeometryContrastiveEmbedder,
    )

    source = GeometryContrastiveEmbedder(
        embedding_dim=8, pretrained=False, train_backbone=False, normalize=True
    )
    checkpoint = {
        "architecture": "resnet18_pair_embedding",
        "model_state": source.state_dict(),
        "embedding_dim": 8,
        "train_backbone": False,
        "normalize_embeddings": True,
    }
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)
    crops_path = tmp_path / "crops.h5"
    with h5py.File(crops_path, "w") as h5:
        h5.create_dataset("crops", data=np.full((1, 32, 32, 3), 127, dtype=np.uint8))
    embeddings, source_info = evaluator._infer_embeddings(
        tmp_path / "checkpoint_best.pt",
        crops_path,
        n_metadata=1,
        max_crops=2,
        batch_size=1,
        device="cpu",
    )
    assert embeddings.shape == (1, 8)
    assert np.linalg.norm(embeddings[0]) == pytest.approx(1.0, abs=1e-5)
    assert source_info["architecture"] == "resnet18_pair_embedding"


def _minimal_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [0, 1, 2],
            "evaluation_split": ["train", "calibration", "test"],
            "split_block_id": ["b_train", "b_cal", "b_test"],
            "frame": [0, 10, 20],
            "view": ["top", "top", "top"],
            "local_fragment_id": ["f_train", "f_cal", "f_test"],
        }
    )


def test_leakage_checks_fail_closed_for_overlap_and_fit_membership():
    metadata = _minimal_metadata()
    leaking = metadata.copy()
    leaking.loc[2, "split_block_id"] = "b_cal"
    with pytest.raises(ValueError, match="window leakage"):
        validate_split_integrity(
            leaking,
            split_col="evaluation_split",
            crop_id_col="sample_id",
            window_col="split_block_id",
            frame_col="frame",
            camera_col="view",
            fragment_col="local_fragment_id",
            fit_ids={0},
        )

    synchronized_leak = pd.DataFrame(
        {
            "sample_id": [0, 1, 2, 3],
            "evaluation_split": ["train", "calibration", "calibration", "test"],
            "split_block_id": ["train_a", "cal_a", "cal_b", "test_a"],
            "frame": [10, 10, 20, 30],
            "view": ["top", "side", "top", "top"],
            "local_fragment_id": ["a", "b", "c", "d"],
        }
    )
    with pytest.raises(ValueError, match="frame leakage"):
        validate_split_integrity(
            synchronized_leak,
            split_col="evaluation_split",
            crop_id_col="sample_id",
            window_col="split_block_id",
            frame_col="frame",
            camera_col="view",
            fragment_col="local_fragment_id",
            fit_ids={0},
        )
    with pytest.raises(ValueError, match="model-fit leakage"):
        validate_split_integrity(
            metadata,
            split_col="evaluation_split",
            crop_id_col="sample_id",
            window_col="split_block_id",
            frame_col="frame",
            camera_col="view",
            fragment_col="local_fragment_id",
            fit_ids={0, 1},
        )


def test_pair_scoring_rejects_cross_split_constraints():
    metadata = _minimal_metadata()
    pairs = pd.DataFrame({"sample_a_id": [0], "sample_b_id": [1], "label": [1]})
    with pytest.raises(ValueError, match="cross train/calibration/test"):
        score_pairs(
            metadata,
            pairs,
            np.eye(3),
            split_col="evaluation_split",
            crop_id_col="sample_id",
            pair_a_col="sample_a_id",
            pair_b_col="sample_b_id",
        )


def test_same_frame_cross_camera_retrieval_and_prototypes_are_similarity_only():
    metadata = pd.DataFrame(
        {
            "sample_id": range(8),
            "view": ["top", "top", "side", "side"] * 2,
            "frame": [10] * 4 + [11] * 4,
            "reference_identity": ["a", "b", "a", "b"] * 2,
            "local_fragment_id": [f"fragment_{index}" for index in range(8)],
        }
    )
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]] * 2)
    rows, metrics = cross_camera_retrieval(
        metadata,
        embeddings,
        identity_col="reference_identity",
        camera_col="view",
        unit_col="local_fragment_id",
        frame_col="frame",
        same_frame_only=True,
    )
    assert len(rows) == 8
    assert metrics["rank1"] == pytest.approx(1.0)
    assert metrics["map"] == pytest.approx(1.0)
    prototypes = trusted_prototype_similarities(
        metadata,
        embeddings,
        prototype_key_col="reference_identity",
        fragment_col="local_fragment_id",
        trusted_mask=np.arange(8) < 4,
        query_mask=np.arange(8) >= 4,
    )
    assert set(prototypes) == {
        "fragment_id",
        "prototype_key",
        "similarity",
        "n_fragment_samples",
        "n_prototype_samples",
    }
    assert not any("predicted" in column or "assignment" in column for column in prototypes)
    with pytest.raises(ValueError, match="geometry IDs"):
        trusted_prototype_similarities(
            metadata.assign(global_id=metadata["reference_identity"]),
            embeddings,
            prototype_key_col="GLOBAL_ID",
            fragment_col="local_fragment_id",
            trusted_mask=np.arange(8) < 4,
            query_mask=np.arange(8) >= 4,
        )
    with pytest.raises(ValueError, match="geometry IDs"):
        trusted_prototype_similarities(
            metadata.assign(geometry_source_idx_local=np.arange(8) % 2),
            embeddings,
            prototype_key_col="geometry_source_idx_local",
            fragment_col="local_fragment_id",
            trusted_mask=np.arange(8) < 4,
            query_mask=np.arange(8) >= 4,
        )


def _write_synthetic_builder_contract(root):
    metadata_rows = []
    embedding_rows = []
    positive_rows = []
    negative_rows = []
    sample_id = 0
    by_key = {}
    bases = {"train": 0, "calibration": 100, "test": 200}
    for split, base in bases.items():
        for frame in (base, base + 1):
            for view in ("top", "side"):
                for identity, vector in (("fish_a", [1.0, 0.05]), ("fish_b", [0.05, 1.0])):
                    by_key[(split, frame, view, identity)] = sample_id
                    metadata_rows.append(
                        {
                            "sample_id": sample_id,
                            "crop_index": sample_id,
                            "split": split,
                            "split_block_id": f"block_{split}",
                            "window": f"window_{split}",
                            "frame": frame,
                            "view": view,
                            "local_fragment_id": f"{split}_{view}_{identity}",
                            "reference_identity": identity,
                            "review_state": "query" if split == "test" else "clean",
                            "hard_crossing": frame % 2 == 1,
                        }
                    )
                    embedding_rows.append(
                        {
                            "sample_id": sample_id,
                            "embedding_000": vector[0] + (0.001 if view == "side" else 0.0),
                            "embedding_001": vector[1],
                        }
                    )
                    sample_id += 1
        for frame in (base, base + 1):
            for identity in ("fish_a", "fish_b"):
                positive_rows.append(
                    {
                        "sample_a_id": by_key[(split, frame, "top", identity)],
                        "sample_b_id": by_key[(split, frame, "side", identity)],
                        "pair_type": "synchronized_cross_camera",
                    }
                )
            for view in ("top", "side"):
                negative_rows.append(
                    {
                        "sample_a_id": by_key[(split, frame, view, "fish_a")],
                        "sample_b_id": by_key[(split, frame, view, "fish_b")],
                        "pair_type": "simultaneous_different_fish",
                    }
                )
    metadata = pd.DataFrame(metadata_rows)
    # Deliberately shuffle the trainer embedding export: evaluator must join by sample_id.
    embeddings = pd.DataFrame(embedding_rows).sample(frac=1, random_state=9)
    metadata.to_parquet(root / "samples.parquet", index=False)
    pd.DataFrame(positive_rows).to_parquet(root / "positive_pairs.parquet", index=False)
    pd.DataFrame(negative_rows).to_parquet(root / "negative_pairs.parquet", index=False)
    embeddings.to_parquet(root / "embeddings.parquet", index=False)
    membership_path = root / "fit_membership.parquet"
    metadata.loc[metadata["split"] == "train", ["sample_id"]].to_parquet(
        membership_path, index=False
    )
    (root / "report.json").write_text(
        json.dumps(
            {
                "fit_membership": {
                    "sha256": evaluator._canonical_table_sha256(membership_path),
                }
            }
        ),
        encoding="utf-8",
    )


def test_full_synthetic_builder_contract_smoke_without_checkpoint(tmp_path):
    _write_synthetic_builder_contract(tmp_path)
    output = tmp_path / "evaluation"
    evaluator.main(
        [
            "--dataset",
            str(tmp_path),
            "--embeddings",
            str(tmp_path / "embeddings.parquet"),
            "--fit-membership",
            str(tmp_path / "fit_membership.parquet"),
            "--out-dir",
            str(output),
            "--prototype-key-col",
            "reference_identity",
        ]
    )
    report = json.loads((output / "evaluation_report.json").read_text(encoding="utf-8"))
    assert report["thresholds_frozen_on_calibration"]["rule"] == "eer"
    assert report["pair_metrics"]["test"]["auroc"] == pytest.approx(1.0)
    assert report["retrieval"]["cross_camera"]["rank1"] == pytest.approx(1.0)
    assert report["retrieval"]["same_frame_cross_camera"]["rank1"] == pytest.approx(1.0)
    assert report["inputs"]["embedding_source"]["alignment"] == "joined_by_sample_id"
    assert report["fit_membership_hash"]["verified"] is True
    assert (output / "same_frame_cross_camera_retrieval.csv").exists()
    assert (output / "trusted_prototype_fragment_similarities.csv").exists()
    strata = pd.read_csv(output / "stratified_pair_metrics.csv")
    assert {"camera_pair", "window_pair", "hard_crossing_any"} <= set(strata["stratifier"])


def test_row_order_embedding_arrays_are_rejected(tmp_path):
    path = tmp_path / "embeddings.npy"
    np.save(path, np.eye(3, dtype=np.float32))
    with pytest.raises(ValueError, match="row-order embedding arrays"):
        evaluator._load_embeddings(path, _minimal_metadata(), crop_id_col="sample_id")


def test_temporal_block_bootstrap_is_deterministic_and_block_level():
    rows = []
    for block_index in range(4):
        for label, score in ((0, 0.1 + 0.01 * block_index), (1, 0.9 - 0.01 * block_index)):
            rows.append(
                {
                    "window_pair": f"block_{block_index}",
                    "label": label,
                    "cosine_similarity": score,
                    "same_probability": score,
                }
            )
    pairs = pd.DataFrame(rows)
    first, info = temporal_block_bootstrap_metrics(
        pairs,
        block_col="window_pair",
        label_col="label",
        probability_col="same_probability",
        decision_threshold_score=0.5,
        target_far_score=0.5,
        probability_threshold=0.5,
        replicates=50,
        seed=7,
    )
    second, _ = temporal_block_bootstrap_metrics(
        pairs,
        block_col="window_pair",
        label_col="label",
        probability_col="same_probability",
        decision_threshold_score=0.5,
        target_far_score=0.5,
        probability_threshold=0.5,
        replicates=50,
        seed=7,
    )
    pd.testing.assert_frame_equal(first, second)
    assert info["n_blocks"] == 4
    assert {"auroc", "average_precision", "precision_at_frozen_threshold"} <= set(first["metric"])
