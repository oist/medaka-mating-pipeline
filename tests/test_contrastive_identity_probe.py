from __future__ import annotations

import numpy as np
import pandas as pd

from sam2_id_experiment.src.evaluation.contrastive_identity_probe import (
    apply_label_remap,
    best_label_remap,
    evaluate_crop_predictions,
    evaluate_tracklet_predictions,
    identity_centroids,
    majority_vote_by_tracklet,
    predict_by_centroid,
    transfer_nearest_identity,
)
from sam2_id_experiment.src.evaluation.joint_side_identity import (
    cross_view_tracklet_correspondence,
    sample_supervised_identity_pairs,
    tracklet_majority_table,
)


def test_transfer_nearest_identity_labels_same_frame_tracks():
    crops = pd.DataFrame(
        {
            "crop_id": [0, 1, 2],
            "frame": [10, 10, 11],
            "tracklet_id": [5, 6, 5],
            "centroid_x": [11.0, 90.0, 40.0],
            "centroid_y": [10.0, 50.0, 40.0],
        }
    )
    ref = pd.DataFrame(
        {
            "frame": [10, 10, 11],
            "track_id": [0, 1, 0],
            "x": [10.0, 100.0, 100.0],
            "y": [10.0, 50.0, 100.0],
        }
    )

    out = transfer_nearest_identity(crops, ref, max_distance_px=20)

    assert out["reference_identity"].tolist() == ["track_0", "track_1", ""]
    assert out["reference_label_status"].tolist() == ["labeled", "labeled", "too_far"]


def test_transfer_nearest_identity_accepts_projected_side_reference():
    metadata = pd.DataFrame({
        "crop_id": [0],
        "frame": [42],
        "tracklet_id": [7],
        "centroid_x": [22.0],
        "centroid_y": [33.0],
    })
    reference = pd.DataFrame({
        "frame": [42],
        "x": [20.0],
        "y": [30.0],
        "identity": ["pseudo_ID_2"],
        "view": ["side1_cam"],
    })

    out = transfer_nearest_identity(metadata, reference, max_distance_px=5.0)

    assert out.loc[0, "reference_identity"] == "pseudo_ID_2"
    assert out.loc[0, "reference_label_status"] == "labeled"


def test_crop_and_tracklet_metrics_majority_vote():
    meta = pd.DataFrame(
        {
            "frame": [0, 1, 2, 0, 1],
            "tracklet_id": [1, 1, 1, 2, 2],
        }
    )
    truth = ["A", "A", "A", "B", "B"]
    pred = ["A", "A", "B", "A", "A"]

    crop = evaluate_crop_predictions(truth, pred, margins=[0.5, 0.4, 0.1, 0.2, 0.3])
    votes = majority_vote_by_tracklet(meta, truth, pred)
    tracklet = evaluate_tracklet_predictions(votes)

    assert crop["crop_accuracy"] == 0.4
    assert votes.set_index("tracklet_id").loc[1, "correct"]
    assert not votes.set_index("tracklet_id").loc[2, "correct"]
    assert tracklet["tracklet_majority_accuracy"] == 0.5


def test_identity_centroid_prediction_returns_margin():
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    labels, centroids = identity_centroids(embeddings, ["A", "A", "B", "B"])
    pred, distance, margin = predict_by_centroid(
        np.asarray([[0.95, 0.05], [0.05, 0.95]], dtype=np.float32),
        labels,
        centroids,
    )

    assert pred.tolist() == ["A", "B"]
    assert np.all(distance < 0.01)
    assert np.all(margin > 0.8)


def test_best_label_remap_handles_permuted_identity_names():
    truth = ["A", "A", "B", "B", "C", "C"]
    pred = ["z", "z", "x", "x", "y", "y"]

    remap = best_label_remap(truth, pred)
    mapped = apply_label_remap(pred, remap)

    assert mapped.tolist() == truth


def test_supervised_identity_pairs_include_cross_view_positives():
    metadata = pd.DataFrame(
        {
            "crop_id": [0, 1, 2, 3, 4, 5],
            "source_view": ["side1", "side1", "side2", "side2", "side1", "side2"],
            "reference_identity": ["ID_1", "ID_1", "ID_1", "ID_1", "ID_2", "ID_2"],
        }
    )

    pairs = sample_supervised_identity_pairs(
        metadata,
        positive_pairs_per_identity=10,
        cross_view_positive_pairs_per_identity=10,
        negative_pairs_per_identity_pair=10,
        random_seed=1,
    )

    assert not pairs.empty
    assert set(pairs["label"]) == {0, 1}
    assert pairs["reason"].str.startswith("same_identity_cross_view").any()
    lookup = metadata.set_index("crop_id")
    cross_positive = pairs[
        pairs["label"].eq(1)
        & pairs["reason"].str.startswith("same_identity_cross_view")
    ]
    for row in cross_positive.to_dict("records"):
        left = lookup.loc[int(row["crop_a_id"])]
        right = lookup.loc[int(row["crop_b_id"])]
        assert left["reference_identity"] == right["reference_identity"]
        assert left["source_view"] != right["source_view"]


def test_tracklet_majority_table_keeps_camera_tracklet_ids_separate():
    metadata = pd.DataFrame(
        {
            "source_view": ["side1", "side1", "side2", "side2"],
            "tracklet_id": [0, 0, 0, 0],
            "frame": [10, 20, 10, 20],
        }
    )

    table = tracklet_majority_table(metadata, ["A", "A", "B", "B"], ["A", "A", "A", "B"])

    assert len(table) == 2
    assert set(table["source_view"]) == {"side1", "side2"}
    assert table.set_index("source_view").loc["side1", "correct"]
    assert not table.set_index("source_view").loc["side2", "correct"]


def test_cross_view_tracklet_correspondence_ranks_nearest_embedding_match():
    metadata = pd.DataFrame(
        {
            "source_view": ["side1", "side1", "side2", "side2"],
            "tracklet_id": [0, 1, 10, 11],
            "frame": [10, 10, 10, 10],
            "reference_identity": ["ID_1", "ID_2", "ID_1", "ID_2"],
        }
    )
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )

    matches = cross_view_tracklet_correspondence(metadata, embeddings)
    best = matches[matches["rank_for_left_tracklet"].eq(1)].sort_values("left_tracklet_id")

    assert best["same_truth_identity"].tolist() == [True, True]
    assert best["right_tracklet_id"].tolist() == [10, 11]
