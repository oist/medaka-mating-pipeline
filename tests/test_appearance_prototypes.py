import numpy as np
import pandas as pd
import pytest
import importlib
import json

from scripts.tracking.benchmark_global_fragment_stitching import (
    _load_appearance_calibration,
    _load_fragment_embeddings,
    _load_trusted_anchors,
)
from scripts.tracking.appearance_prototypes import (
    PooledEmbeddings,
    assign_simultaneous_fragments,
    attach_fragment_appearance,
    build_trusted_prototypes,
    pool_embeddings,
)
from scripts.tracking.global_fragment_stitching import EdgeFeatures, calculate_edge_cost


prepare_module = importlib.import_module(
    "sam2_id_experiment.scripts.51_prepare_fragment_appearance_evidence"
)


def test_pooling_balances_camera_views_instead_of_frame_counts():
    embeddings = np.asarray(
        [[1.0, 0.0]] * 9 + [[0.0, 1.0]],
        dtype=np.float32,
    )
    pooled = pool_embeddings(
        embeddings,
        ["fragment"] * 10,
        views=["top"] * 9 + ["side"],
    )
    np.testing.assert_allclose(
        pooled.embeddings[0],
        np.asarray([1.0, 1.0]) / np.sqrt(2.0),
        atol=1e-6,
    )
    assert pooled.sample_counts == (10,)


def test_prototypes_require_explicit_trusted_biological_labels():
    bank = build_trusted_prototypes(
        np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
        ["female", "female", "male_1", "male_1"],
        min_samples_per_identity=2,
    )
    assert bank.identities == ("female", "male_1")
    assert bank.sample_counts == (2, 2)


def test_hungarian_prototype_assignment_is_one_to_one_for_same_frame_fish():
    fragments = PooledEmbeddings(
        ("left", "right"),
        np.asarray([[0.99, 0.01], [0.01, 0.99]]),
        (5, 4),
    )
    prototypes = build_trusted_prototypes(
        np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        ["ID_1", "ID_1", "ID_2", "ID_2"],
        min_samples_per_identity=2,
    )
    assignments = assign_simultaneous_fragments(
        fragments, prototypes, max_distance=0.2, min_margin=0.1
    )
    assert [(row.fragment_id, row.candidate_identity, row.status) for row in assignments] == [
        ("left", "ID_1", "assigned"),
        ("right", "ID_2", "assigned"),
    ]


def test_ambiguous_prototype_match_abstains():
    fragments = PooledEmbeddings(("fish",), np.asarray([[1.0, 1.0]]), (3,))
    prototypes = build_trusted_prototypes(
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        ["ID_1", "ID_2"],
        min_samples_per_identity=1,
    )
    assignment = assign_simultaneous_fragments(
        fragments, prototypes, max_distance=0.5, min_margin=0.05
    )[0]
    assert assignment.candidate_identity is None
    assert assignment.status == "margin_reject"


def test_missing_fragment_embedding_is_neutral_and_present_embedding_changes_cost():
    raw_edges = (
        EdgeFeatures("a", "b", gap_frames=1),
        EdgeFeatures("a", "missing", gap_frames=1),
    )
    pooled = PooledEmbeddings(
        ("a", "b"),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        (4, 5),
    )
    attached = attach_fragment_appearance(
        raw_edges,
        pooled,
        calibrator_coefficient=10.0,
        calibrator_intercept=-5.0,
        same_probability_min=0.9,
        different_probability_max=0.1,
    )
    assert attached[0].appearance_cosine_distance == 1.0
    assert attached[0].appearance_state == "contradiction"
    assert attached[0].metadata["appearance_source"] == "pooled_metric_embedding"
    assert attached[1].appearance_cosine_distance is None
    assert calculate_edge_cost(attached[0]).terms["appearance"] > 0.0
    assert calculate_edge_cost(attached[1]).terms["appearance"] == 0.0
    ambiguous = EdgeFeatures(
        "a", "b", gap_frames=1, appearance_cosine_distance=0.5, appearance_state="unknown"
    )
    assert calculate_edge_cost(ambiguous).terms["appearance"] == 0.0


def test_benchmark_embedding_npz_contract(tmp_path):
    path = tmp_path / "fragment_embeddings.npz"
    np.savez_compressed(
        path,
        fragment_ids=np.asarray(["fragment_a", "fragment_b"]),
        embeddings=np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        sample_counts=np.asarray([7, 8]),
    )
    pooled = _load_fragment_embeddings(path)
    assert pooled is not None
    assert pooled.keys == ("fragment_a", "fragment_b")
    assert pooled.sample_counts == (7, 8)
    np.testing.assert_allclose(np.linalg.norm(pooled.embeddings, axis=1), 1.0)

    report = tmp_path / "evaluation_report.json"
    report.write_text(
        json.dumps(
            {
                "thresholds_frozen_on_calibration": {
                    "target_far_threshold_probability": 0.95,
                    "calibrator": {"coefficient": 8.0, "intercept": -4.0},
                }
            }
        ),
        encoding="utf-8",
    )
    calibration = _load_appearance_calibration(report)
    assert calibration == {
        "coefficient": 8.0,
        "intercept": -4.0,
        "same_probability_min": 0.95,
        "different_probability_max": pytest.approx(0.05),
    }


def test_benchmark_anchor_loader_requires_explicit_trust_schema(tmp_path):
    unsafe = tmp_path / "unsafe.csv"
    pd.DataFrame(
        {"component_id": ["a"], "geometry_id": [0]}
    ).to_csv(unsafe, index=False)
    with pytest.raises(ValueError, match="trusted anchor CSV missing columns"):
        _load_trusted_anchors(unsafe)

    safe = tmp_path / "safe.csv"
    pd.DataFrame(
        {
            "component_id": ["a", "b"],
            "trusted_identity": ["ID_1", "ID_2"],
            "anchor_status": ["trusted", "candidate"],
        }
    ).to_csv(safe, index=False)
    assert _load_trusted_anchors(safe) == {"a": "ID_1"}

    null_identity = tmp_path / "null_identity.csv"
    pd.DataFrame(
        {
            "component_id": ["a"],
            "trusted_identity": [None],
            "anchor_status": ["trusted"],
        }
    ).to_csv(null_identity, index=False)
    with pytest.raises(ValueError, match="non-null"):
        _load_trusted_anchors(null_identity)


def test_prepare_fragment_evidence_joins_provenance_and_pools_views():
    embedding_table = pd.DataFrame(
        {
            "sample_id": [0, 1, 2, 3],
            "embedding_000": [1.0, 1.0, 0.0, 0.0],
            "embedding_001": [0.0, 0.0, 1.0, 1.0],
        }
    )
    samples = pd.DataFrame(
        {
            "sample_id": [0, 1, 2, 3],
            "component_id_provenance": ["a", "a", "b", "b"],
            "view": ["top", "side", "top", "side"],
            "split": ["test"] * 4,
        }
    )
    pooled, provenance = prepare_module.prepare_fragment_embeddings(
        embedding_table,
        samples,
        fragment_column="component_id_provenance",
        splits=("test",),
        min_samples=2,
    )
    assert pooled.keys == ("a", "b")
    assert pooled.sample_counts == (2, 2)
    assert provenance.available_views.tolist() == [2, 2]

    with pytest.raises(ValueError, match="blank fragment/view provenance"):
        prepare_module.prepare_fragment_embeddings(
            embedding_table,
            samples.assign(component_id_provenance=["a", " ", "b", "b"]),
            fragment_column="component_id_provenance",
            splits=("test",),
            min_samples=2,
        )
