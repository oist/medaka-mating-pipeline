from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.reconstruction_3d.compare_fixed_clip_candidates import (
    CandidateFiles,
    ClipWindow,
    summarize_candidate,
    summarize_identified,
    summarize_pre_link,
    summarize_trajectory,
    write_reports,
)


WINDOW = ClipWindow("event", 95000, 95001)


def _write_pre_link(
    path: Path,
    *,
    include_synthetic: bool = True,
) -> None:
    frames = {}
    for frame in range(WINDOW.start, WINDOW.end + 1):
        frame_payload = {
            "groups": {
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0],
                "c": [7.0, 8.0, 9.0],
            },
            "errors": {"a": 0.1, "b": 0.2, "c": 0.3},
        }
        if include_synthetic:
            frame_payload["synthetic_views"] = [-1, -1, -1]
        frames[str(frame)] = frame_payload
    payload = {
        "num_individuals": 3,
        "src_labels": ["a", "b", "c"],
        "calibration": {
            "triangulation": {
                "backend": "refractive",
                "score_unit": "mm^2",
                "provenance": {
                    "model_files": {
                        "cam0": "cam0.json",
                        "cam1": "cam1.json",
                        "cam2": "cam2.json",
                    }
                },
            }
        },
        "frames": frames,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_trajectory(path: Path) -> None:
    rows = []
    positions = {
        "ID_1": ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)),
        "ID_2": ((0.0, 5.0, 0.0), (1.0, 5.0, 0.0)),
        "ID_3": ((0.0, 9.0, 0.0), (1.0, 9.0, 0.0)),
    }
    for fish_id, points in positions.items():
        for frame, point in zip(
            range(WINDOW.start, WINDOW.end + 1),
            points,
        ):
            rows.append(
                {
                    "frame": frame,
                    "fish_id": fish_id,
                    "X": point[0],
                    "Y": point[1],
                    "Z": point[2],
                    "is_interpolated": False,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_identified(path: Path, *, unstable: bool = False) -> None:
    rows = []
    mapping = {"ID_1": "fish_a", "ID_2": "fish_b", "ID_3": "fish_c"}
    for fish_id, final_id in mapping.items():
        for frame in range(WINDOW.start, WINDOW.end + 1):
            if unstable and fish_id == "ID_1" and frame == WINDOW.end:
                final_id = "fish_b"
            rows.append(
                {
                    "frame": frame,
                    "fish_id": fish_id,
                    "final_fish_id": final_id,
                    "bind_source": "top_cam_2d",
                    "bind_score": 2.0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_pre_link_uses_bounded_nonzero_frame_denominator(
    tmp_path: Path,
) -> None:
    pre_link = tmp_path / "pre_link.json"
    _write_pre_link(pre_link)
    summary = summarize_pre_link(pre_link, window=WINDOW)
    assert summary["present_frames"] == 2
    assert summary["missing_frames"] == 0
    assert summary["all_three_valid_rate_expected"] == 1.0
    assert summary["calibration"]["backend"] == "refractive"
    assert summary["calibration"]["score_unit"] == "mm^2"
    assert summary["synthetic_view_provenance"] == "complete"
    assert summary["synthetic_slot_rate"] == 0.0


def test_missing_synthetic_provenance_is_unknown_not_zero(
    tmp_path: Path,
) -> None:
    pre_link = tmp_path / "legacy_pre_link.json"
    _write_pre_link(pre_link, include_synthetic=False)
    summary = summarize_pre_link(pre_link, window=WINDOW)
    assert summary["synthetic_view_provenance"] == "legacy_unknown"
    assert summary["synthetic_slot_count"] is None
    assert summary["synthetic_slot_rate"] is None


def test_trajectory_speed_denominator_is_consecutive_clip_pairs(
    tmp_path: Path,
) -> None:
    trajectory = tmp_path / "trajectory.csv"
    _write_trajectory(trajectory)
    summary = summarize_trajectory(trajectory, window=WINDOW)
    jump = summary["consecutive_jump_observed_mm"]
    assert summary["present_frames"] == 2
    assert summary["missing_frames"] == 0
    assert summary["fish_count"] == 3
    assert jump["n"] == 3
    assert jump["thresholds"]["8.0"]["count"] == 1
    assert jump["thresholds"]["8.0"]["rate"] == pytest.approx(1 / 3)


def test_identified_requires_stable_one_to_one_relabel(
    tmp_path: Path,
) -> None:
    stable_path = tmp_path / "stable.csv"
    unstable_path = tmp_path / "unstable.csv"
    _write_identified(stable_path)
    _write_identified(unstable_path, unstable=True)
    assert summarize_identified(
        stable_path, window=WINDOW
    )["stable_one_to_one_relabel"]
    assert not summarize_identified(
        unstable_path, window=WINDOW
    )["stable_one_to_one_relabel"]


def test_candidate_reports_refractive_structural_gate(
    tmp_path: Path,
) -> None:
    pre_link = tmp_path / "pre_link.json"
    trajectory = tmp_path / "trajectory.csv"
    identified = tmp_path / "identified.csv"
    _write_pre_link(pre_link)
    _write_trajectory(trajectory)
    _write_identified(identified)
    summary = summarize_candidate(
        CandidateFiles("baseline", pre_link, trajectory, identified),
        window=WINDOW,
        residual_gate=None,
        speed_thresholds=(8.0, 20.0),
        expected_backend="refractive",
        expected_score_unit="mm^2",
    )
    assert summary["structural_gate_pass"]

    output_json = tmp_path / "qc" / "fixed_clip_qc.json"
    output_csv = tmp_path / "qc" / "fixed_clip_qc.csv"
    output_md = tmp_path / "qc" / "FIXED_CLIP_QC.md"
    write_reports(
        summaries=[summary],
        baseline="baseline",
        output_json=output_json,
        output_csv=output_csv,
        output_md=output_md,
        expected_backend="refractive",
        expected_score_unit="mm^2",
        residual_gate=None,
        speed_thresholds=(8.0, 20.0),
    )
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["expected_calibration"]["score_unit"] == "mm^2"
    assert output_csv.is_file()
    assert "not an independent scientific promotion" in output_md.read_text(
        encoding="utf-8"
    )
