from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis.compare_anchor_coordinates import (
    apply_pose_qc_exclusions,
    build_centroid_reference,
    run_anchor_ablation,
)
from scripts.tracking.compare_motion_linkers import ComparisonConfig


def _mouth(frames=range(6)) -> pd.DataFrame:
    rows = []
    for frame in frames:
        rows.extend(
            [
                {
                    "frame": frame,
                    "human_identity": "H1",
                    "X": float(frame),
                    "Y": 0.0,
                    "Z": 0.0,
                },
                {
                    "frame": frame,
                    "human_identity": "H2",
                    "X": float(20 - frame),
                    "Y": 0.0,
                    "Z": 0.0,
                },
            ]
        )
    return pd.DataFrame(rows)


def _centroid(frames=range(6)) -> pd.DataFrame:
    rows = []
    for frame in frames:
        # Slot 0 follows H2 and slot 1 follows H1. Hungarian matching must use
        # XYZ rather than assuming geometry-slot or row-order identity.
        rows.extend(
            [
                {
                    "frame": frame,
                    "geometry_slot": 0,
                    "X": float(20 - frame) + 0.2,
                    "Y": 0.0,
                    "Z": 0.0,
                },
                {
                    "frame": frame,
                    "geometry_slot": 1,
                    "X": float(frame) - 0.1,
                    "Y": 0.0,
                    "Z": 0.0,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_centroid_reference_uses_one_to_one_xyz_matching_not_slot_order():
    reference, summary = build_centroid_reference(
        _mouth(range(2)), _centroid(range(2)), match_gate_mm=1.0
    )
    by_truth = reference.set_index(["frame", "human_identity"])

    assert by_truth.loc[(0, "H1"), "geometry_slot"] == "G2"
    assert by_truth.loc[(0, "H2"), "geometry_slot"] == "G1"
    assert by_truth.loc[(0, "H1"), "X"] == -0.1
    assert by_truth.loc[(0, "H2"), "X"] == 20.2
    assert reference.groupby("frame")["geometry_slot"].nunique().eq(2).all()
    assert summary["matched_rows"] == 4
    assert summary["match_coverage_of_finite_reviewed"] == 1.0
    assert summary["resolved_slot_map"] == {"0": "G1", "1": "G2"}


def test_pose_qc_excludes_slot_intervals_flagged_in_any_camera():
    reference, _summary = build_centroid_reference(
        _mouth(range(4)), _centroid(range(4)), match_gate_mm=1.0
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "event_type": "bad_mouth",
                "view_scope": "top_cam",
                "frame_start": 1,
                "frame_end": 2,
                "slot_a": "G2",
            },
            {
                "event_id": "E2",
                "event_type": "false_positive",
                "view_scope": "side2_cam",
                "frame_start": 3,
                "frame_end": 3,
                "slot_a": "G1",
            },
        ]
    )
    filtered, summary = apply_pose_qc_exclusions(
        reference, events, slots=["G1", "G2"]
    )

    excluded = filtered[filtered["pose_qc_excluded"]]
    assert set(zip(excluded["geometry_slot"], excluded["frame"])) == {
        ("G2", 1),
        ("G2", 2),
        ("G1", 3),
    }
    assert np.isnan(excluded[["X", "Y", "Z"]].to_numpy(dtype=float)).all()
    assert summary["excluded_matched_rows"] == 3
    assert summary["events_by_type"]["bad_mouth"] == 1
    assert summary["events_by_type"]["false_positive"] == 1
    assert summary["node_qc_conclusion_supported"] is True


def test_zero_node_labels_emit_explicit_no_conclusion_caveat():
    events = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "event_type": "false_positive",
                "view_scope": "side1_cam",
                "frame_start": 4,
                "frame_end": 4,
                "slot_a": "G1",
            }
        ]
    )
    result = run_anchor_ablation(
        mouth_reference=_mouth(),
        centroid_track=_centroid(),
        events=events,
        episodes=pd.DataFrame(
            [{"episode_id": "heldout", "start_frame": 3, "end_frame": 5}]
        ),
        comparison_config=ComparisonConfig(
            calibration_start_frame=0,
            calibration_end_frame=2,
            num_individuals=2,
            max_jump_mm=None,
            max_jump_hard_cap_mm=None,
            velocity_prediction_weight=1.0,
            flow_weight=0.0,
            acceleration_weight=0.0,
        ),
        centroid_match_gate_mm=1.0,
    )

    report = result["report"]
    assert set(report["comparisons"]) == {
        "mouth",
        "centroid",
        "centroid_pose_qc",
    }
    assert report["pose_qc"]["node_specific_events"] == 0
    assert report["pose_qc"]["node_qc_conclusion_supported"] is False
    assert "no conclusion" in report["pose_qc"]["caveat"]
    for anchor in report["comparisons"].values():
        assert set(anchor["methods"]) == {"position_only", "motion_aware"}
