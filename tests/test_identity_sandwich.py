import pandas as pd

from sam2_id_experiment.src.evaluation.identity_sandwich import (
    SandwichConfig,
    run_identity_sandwich_audit,
)


def _tracklet_rows(tracklet_id, frames, xy0, *, qc_status="clean_train"):
    rows = []
    for i, frame in enumerate(frames):
        rows.append(
            {
                "tracklet_id": tracklet_id,
                "frame": frame,
                "centroid_x": xy0[0] + i,
                "centroid_y": xy0[1],
                "qc_status": qc_status,
                "auto_qc_flag": qc_status != "clean_train",
                "expected_object_count_match": qc_status == "clean_train",
            }
        )
    return rows


def _prediction_rows(tracklet_id, frames, label, *, margin=0.5):
    return [
        {
            "tracklet_id": tracklet_id,
            "frame": frame,
            "accepted_predicted_identity": label,
            "predicted_identity": label,
            "reference_identity": label,
            "embedding_margin": margin,
        }
        for frame in frames
    ]


def test_identity_sandwich_fills_weak_middle_tracklet(tmp_path):
    tracklets = pd.DataFrame(
        _tracklet_rows(1, range(0, 6), (0, 0))
        + _tracklet_rows(2, range(6, 9), (6, 0), qc_status="manual_review")
        + _tracklet_rows(3, range(9, 15), (9, 0))
    )
    predictions = pd.DataFrame(
        _prediction_rows(1, range(0, 6), "ID_1")
        + _prediction_rows(3, range(9, 15), "ID_1")
    )

    result = run_identity_sandwich_audit(
        tracklets=tracklets,
        crop_predictions=predictions,
        config=SandwichConfig(max_gap_frames=20, max_speed_px_per_frame=5.0, min_crops=2),
        output_dir=tmp_path,
    )

    repairs = result["tracklet_identity_repair"]
    middle = repairs[repairs["tracklet_id"].eq(2)].iloc[0]
    assert middle["repaired_identity"] == "ID_1"
    assert middle["repair_action"] == "sandwich_filled"
    assert result["summary"]["sandwich_decision_counts"]["confirmed_identity_bridge"] >= 1
    assert (tmp_path / "sandwich_events.csv").exists()


def test_identity_sandwich_flags_conflicting_high_confidence_ids():
    tracklets = pd.DataFrame(
        _tracklet_rows(1, range(0, 6), (0, 0))
        + _tracklet_rows(2, range(9, 15), (9, 0))
    )
    predictions = pd.DataFrame(
        _prediction_rows(1, range(0, 6), "ID_1")
        + _prediction_rows(2, range(9, 15), "ID_2")
    )

    result = run_identity_sandwich_audit(
        tracklets=tracklets,
        crop_predictions=predictions,
        config=SandwichConfig(max_gap_frames=20, max_speed_px_per_frame=5.0, min_crops=2),
    )

    events = result["sandwich_events"]
    assert set(events["decision"]) == {"possible_identity_swap"}
    repairs = result["tracklet_identity_repair"]
    assert set(repairs["repair_action"]) == {"needs_review_possible_swap"}
