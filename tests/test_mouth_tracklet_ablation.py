import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.reconstruction_3d.candidate_cache import CandidateWindow
from scripts.reconstruction_3d.deferred_identity_tracking import (
    DeferredTrackingConfig,
)
from scripts.reconstruction_3d.mouth_tracklet_ablation import (
    _a4_direct,
    _a4_filled,
    _admissible_retained_hypotheses,
    _slice_window,
    build_parser,
    build_a5_hints,
    run_a5_window,
)


def _window() -> CandidateWindow:
    mouth = np.zeros((5, 2, 2), dtype=float)
    score = np.ones((5, 2), dtype=float)
    return CandidateWindow(
        frame_start=10,
        frame_end=14,
        mouth_xy=mouth,
        tail_xy=mouth + 1,
        instance_score=score,
        mouth_score=score,
        tail_score=score,
        visibility=np.ones((5, 2), dtype=np.uint8),
        source_instance=np.zeros((5, 2), dtype=np.int16),
    )


def test_slice_window_preserves_absolute_frames_and_arrays():
    sliced = _slice_window(_window(), 11, 13)

    assert sliced.frame_start == 11
    assert sliced.frame_end == 13
    assert sliced.mouth_xy.shape == (3, 2, 2)


def test_admissible_hypotheses_require_retained_tracklet_in_declared_views():
    table = pd.DataFrame(
        [
            {
                "view_mask": 0b111,
                "tracklet_top_cam": 0,
                "tracklet_side1_cam": 0,
                "tracklet_side2_cam": 0,
            },
            {
                "view_mask": 0b111,
                "tracklet_top_cam": 0,
                "tracklet_side1_cam": -10,
                "tracklet_side2_cam": 0,
            },
            {
                "view_mask": 0b011,
                "tracklet_top_cam": 0,
                "tracklet_side1_cam": 0,
                "tracklet_side2_cam": -10,
            },
        ]
    )

    admissible = _admissible_retained_hypotheses(table)

    assert admissible.index.tolist() == [0, 1]
    assert len(admissible) == 2
    assert admissible.iloc[1].view_mask == 0b011


def _a4_row(
    frame,
    path_id,
    segment,
    hypothesis_id,
    x,
    *,
    mouth_confidence=0.9,
    view_count=3,
):
    return {
        "frame": frame,
        "path_id": path_id,
        "path_segment": segment,
        "hypothesis_id": hypothesis_id,
        "X": x,
        "Y": x + 0.1,
        "Z": x + 0.2,
        "mouth_confidence": mouth_confidence,
        "view_count": view_count,
        "group_eligible": True,
        "group_support_frames": 10,
        "group_median_ray_residual_mm2": 0.2,
        "group_mad_ray_residual_mm2": 0.05,
        "group_median_mouth_confidence": 0.85,
        "group_assignment_cost": -0.5,
        "group_retained_count_mismatch_fraction": 0.0,
    }


def _a2_row(a4, fish_id, source_index, identity_segment=0, *, xyz_offset=0):
    return {
        "frame": a4["frame"],
        "fish_id": fish_id,
        "identity_segment": identity_segment,
        "source_index": source_index,
        "hypothesis_id": a4["hypothesis_id"],
        "X": a4["X"] + xyz_offset,
        "Y": a4["Y"],
        "Z": a4["Z"],
    }


def test_a5_hints_are_exact_and_segment_bounded_with_forced_restart():
    rows = []
    geometry = []
    definitions = [
        (0, "PATH_1", 0, "h0", 0.0),
        (0, "PATH_2", 0, "h1", 10.0),
        (0, "PATH_3", 0, "h2", 20.0),
        (5, "PATH_1", 1, "h3", 1.0),
        (5, "PATH_2", 0, "h4", 11.0),
        (5, "PATH_3", 0, "h5", 21.0),
    ]
    for frame, path_id, segment, hypothesis_id, x in definitions:
        row = _a4_row(
            frame, path_id, segment, hypothesis_id, x
        )
        rows.append(row)
        fish_id = {
            "PATH_1": "ID_1",
            "PATH_2": "ID_2",
            "PATH_3": "ID_3",
        }[path_id]
        geometry.append(
            _a2_row(
                row,
                fish_id,
                int(path_id[-1]) - 1,
                identity_segment=1 if path_id == "PATH_1" and segment == 1 else 0,
            )
        )

    hints, resets, summary = build_a5_hints(
        pd.DataFrame(rows),
        pd.DataFrame(geometry),
        minimum_mapping_votes=1,
        minimum_vote_margin=1,
    )

    assert len(hints) == 6
    assert resets == [5]
    assert set(hints.segment_key) == {
        "PATH_1:0", "PATH_1:1", "PATH_2:0", "PATH_3:0"
    }
    assert (
        hints[hints.path_id.eq("PATH_1")]
        .sort_values("frame")
        .path_segment.tolist()
        == [0, 1]
    )
    assert summary["continuity_policy"].startswith("path_id:path_segment")


def test_a5_hints_reject_two_view_or_weak_group_rows():
    rows = [
        _a4_row(
            0, f"PATH_{index + 1}", 0, f"h{index}", float(index),
            view_count=2 if index == 0 else 3,
        )
        for index in range(3)
    ]
    selected = pd.DataFrame(rows)
    geometry = pd.DataFrame(
        [
            _a2_row(row, f"ID_{index + 1}", index)
            for index, row in enumerate(rows)
        ]
    )

    hints, _resets, summary = build_a5_hints(
        selected,
        geometry,
        minimum_mapping_votes=1,
        minimum_vote_margin=1,
    )

    assert len(hints) == 2
    assert summary["filter_counts"]["after_view_count"] == 2


def test_a5_votes_preserve_identity_through_x_order_crossing():
    selected_rows = []
    a2_rows = []
    x_by_path = {
        "PATH_1": [0.0, 5.0, 10.0],
        "PATH_2": [10.0, 5.5, 0.0],
        "PATH_3": [20.0, 20.0, 20.0],
    }
    for path_index, (path_id, values) in enumerate(x_by_path.items()):
        for frame, x in enumerate(values):
            row = _a4_row(
                frame, path_id, 0, f"{path_id}_{frame}", x
            )
            selected_rows.append(row)
            a2_rows.append(_a2_row(row, f"ID_{path_index + 1}", path_index))

    hints, _resets, _summary = build_a5_hints(
        pd.DataFrame(selected_rows), pd.DataFrame(a2_rows)
    )

    path1 = hints[hints.path_id.eq("PATH_1")].sort_values("frame")
    path2 = hints[hints.path_id.eq("PATH_2")].sort_values("frame")
    assert path1.identity.tolist() == ["ID_1"] * 3
    assert path2.identity.tolist() == ["ID_2"] * 3
    assert path1.X.tolist() == [0.0, 5.0, 10.0]
    assert path2.X.tolist() == [10.0, 5.5, 0.0]


def test_a5_rejects_temporally_overlapping_segments_mapped_to_same_identity():
    first_strong = _a4_row(0, "PATH_1", 0, "a0", 0.0)
    first_weak = _a4_row(
        2, "PATH_1", 0, "a2", 2.0, mouth_confidence=0.0
    )
    second_weak = _a4_row(
        1, "PATH_2", 0, "b1", 1.0, mouth_confidence=0.0
    )
    second_strong = _a4_row(3, "PATH_2", 0, "b3", 3.0)
    rows = [first_strong, first_weak, second_weak, second_strong]
    a2 = [
        _a2_row(row, "ID_1", 0)
        for row in rows
    ]

    hints, _resets, summary = build_a5_hints(
        pd.DataFrame(rows),
        pd.DataFrame(a2),
        minimum_mapping_votes=1,
        minimum_vote_margin=1,
    )

    assert hints.empty
    assert summary["filter_counts"]["rejected_overlap_identity_segments"] == 2
    assert all(
        row["overlap_identity_conflict"]
        for row in summary["mapping_vote_audit"]
    )


def test_a5_rejects_low_purity_or_xyz_mismatched_mapping():
    rows = [
        _a4_row(frame, "PATH_1", 0, f"h{frame}", float(frame))
        for frame in range(4)
    ]
    a2 = [
        _a2_row(row, "ID_1" if index < 2 else "ID_2", index % 3)
        for index, row in enumerate(rows)
    ]
    hints, _resets, summary = build_a5_hints(
        pd.DataFrame(rows),
        pd.DataFrame(a2),
        minimum_mapping_votes=1,
        minimum_vote_margin=0,
    )
    assert hints.empty
    assert not summary["mapping_vote_audit"][0]["accepted_by_vote"]

    exact_rows = [
        _a4_row(frame, "PATH_1", 0, f"x{frame}", float(frame))
        for frame in range(3)
    ]
    mismatched = [
        _a2_row(row, "ID_1", 0, xyz_offset=0.01)
        for row in exact_rows
    ]
    hints, _resets, summary = build_a5_hints(
        pd.DataFrame(exact_rows), pd.DataFrame(mismatched)
    )
    assert hints.empty
    assert summary["filter_counts"]["after_xyz_consistency"] == 0


def test_a4_direct_grid_and_fill_never_cross_path_segment_restart():
    selected = pd.DataFrame(
        [
            {
                **_a4_row(0, "PATH_1", 0, "h0", 0.0),
                "ray_residual_mm2": 0.1,
                "group_id": "g0",
                "view_mask": 7,
                "slot_cam0": 0,
                "slot_cam1": 0,
                "slot_cam2": 0,
            },
            {
                **_a4_row(2, "PATH_1", 1, "h2", 20.0),
                "ray_residual_mm2": 0.1,
                "group_id": "g2",
                "view_mask": 7,
                "slot_cam0": 0,
                "slot_cam1": 0,
                "slot_cam2": 0,
            },
        ]
    )
    direct = _a4_direct(selected, frame_start=0, frame_end=2)
    filled = _a4_filled(direct, max_gap_frames=100)

    assert len(direct) == 9
    middle = filled[
        filled.frame.eq(1) & filled.fish_id.eq("PATH_1")
    ].iloc[0]
    assert np.isnan(middle.X)
    assert middle.identity_segment == -1


def test_parser_exposes_a5_execution_action():
    args = build_parser().parse_args(
        [
            "a5",
            "--geometry", "geometry.parquet",
            "--geometry-summary", "geometry.json",
            "--hints", "hints.csv",
            "--forced-reset-frames", "resets.json",
            "--hints-summary", "hints.json",
            "--output-dir", "output",
            "--frame-start", "0",
            "--frame-end", "10",
            "--velocity-weight", "0.25",
            "--acceleration-weight", "0.05",
        ]
    )
    assert args.action == "a5"


def test_a5_runner_end_to_end_authenticates_and_honors_exact_hints(tmp_path):
    geometry_records = []
    selected_records = []
    a2_records = []
    for frame in range(3):
        for source_index in range(3):
            x = float(source_index * 20 + frame)
            hypothesis_id = f"f{frame}_h{source_index}"
            geometry_records.append(
                {
                    "frame": frame,
                    "view_count": 3,
                    "ray_residual_mm2": 0.1,
                    "hypothesis_id": hypothesis_id,
                    "X": x,
                    "Y": 1.0,
                    "Z": 2.0,
                    "slot_cam0": source_index,
                    "slot_cam1": source_index,
                    "slot_cam2": source_index,
                }
            )
            selected = _a4_row(
                frame,
                f"PATH_{source_index + 1}",
                0,
                hypothesis_id,
                x,
            )
            selected["Y"] = 1.0
            selected["Z"] = 2.0
            selected_records.append(selected)
            a2_records.append(
                _a2_row(selected, f"ID_{source_index + 1}", source_index)
            )
    geometry_path = tmp_path / "geometry.parquet"
    pd.DataFrame(geometry_records).to_parquet(geometry_path, index=False)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    geometry_summary_path = tmp_path / "geometry_summary.json"
    geometry_summary_path.write_text(
        json.dumps(
            {
                "output_sha256": digest(geometry_path),
                "calibration": {
                    "backend": "refractive",
                    "score_unit": "mm^2",
                    "provenance": {"resolved_backend": "refractive"},
                },
            }
        ),
        encoding="utf-8",
    )
    hints, resets, hints_summary = build_a5_hints(
        pd.DataFrame(selected_records), pd.DataFrame(a2_records)
    )
    hints_path = tmp_path / "hints.csv"
    resets_path = tmp_path / "resets.json"
    hints.to_csv(hints_path, index=False)
    resets_path.write_text(json.dumps(resets), encoding="utf-8")
    hints_summary["outputs"] = {
        "hints_sha256": digest(hints_path),
        "forced_reset_frames_sha256": digest(resets_path),
    }
    hints_summary_path = tmp_path / "hints_summary.json"
    hints_summary_path.write_text(
        json.dumps(hints_summary), encoding="utf-8"
    )
    output_dir = tmp_path / "a5"

    summary = run_a5_window(
        geometry_path=geometry_path,
        geometry_summary_path=geometry_summary_path,
        hints_path=hints_path,
        forced_reset_frames_path=resets_path,
        hints_summary_path=hints_summary_path,
        output_dir=output_dir,
        frame_start=0,
        frame_end=2,
        config=DeferredTrackingConfig(
            velocity_prediction_weight=0.25,
            acceleration_weight=0.05,
            motion_history_frames=5,
        ),
        command=["pytest-a5-e2e"],
    )
    direct = pd.read_csv(output_dir / "a5_direct.csv")
    application = pd.read_csv(
        output_dir / "a5_hint_application_audit.csv"
    )

    assert len(hints) == 9
    assert direct.groupby("frame").size().eq(3).all()
    assert direct[["X", "Y", "Z"]].notna().all().all()
    assert application.honored.astype(bool).all()
    assert summary["hints_honored"] == 9
    assert summary["calibration"]["backend"] == "refractive"
    assert all(
        digest(Path(value["path"])) == value["sha256"]
        for value in summary["outputs"].values()
    )
