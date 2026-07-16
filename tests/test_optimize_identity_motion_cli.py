from __future__ import annotations

import importlib
import json
import sys
import types
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest


cli = importlib.import_module("sam2_id_experiment.scripts.52_optimize_identity_motion")


def _track(permutations: list[tuple[str, str, str]]) -> pd.DataFrame:
    rows = []
    for frame, labels in enumerate(permutations):
        for slot, label in enumerate(labels):
            rows.append(
                {
                    "frame": frame,
                    "geometry_slot": slot,
                    "id_final": label,
                    "id_source": "appearance",
                    "X": float(slot * 100),
                    "Y": 0.0,
                    "Z": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_speed_and_permutation_qc_detect_one_frame_swap() -> None:
    data = _track(
        [
            ("F", "M1", "M2"),
            ("F", "M1", "M2"),
            ("M1", "F", "M2"),
            ("F", "M1", "M2"),
        ]
    )

    speed = cli.speed_qc(
        data, id_column="id_final", jump_threshold_mm=8.0, source_fps=166.0
    )
    permutation = cli.permutation_qc(data, id_column="id_final")

    assert speed["all_ids"]["jumps_over_threshold"] == 4
    assert speed["all_ids"]["unique_jump_frames"] == 2
    assert speed["per_id"]["F"]["speed_mm_per_frame"]["max"] == pytest.approx(100.0)
    assert speed["per_id"]["F"]["abs_axis_mm_per_frame"]["Y"]["max"] == 0.0
    assert permutation["permutation_changes"] == 2
    assert permutation["change_frames"] == [2, 3]


def test_one_each_validity_flags_duplicate_identity() -> None:
    data = _track([("F", "M1", "M2"), ("F", "F", "M2")])

    result = cli.one_each_validity(data, id_column="id_final")

    assert result["total_frames"] == 2
    assert result["valid_frames"] == 1
    assert result["invalid_frames"] == 1
    assert result["invalid_frame_examples"] == [1]


def test_changed_spans_merge_only_consecutive_frames() -> None:
    data = _track([("F", "M1", "M2")] * 7)
    data["id_final_pre_motion"] = data["id_final"]
    for frame in (1, 2, 5):
        mask = (data["frame"] == frame) & data["geometry_slot"].isin([0, 1])
        data.loc[mask, "id_final"] = data.loc[mask, "id_final"].map(
            {"F": "M1", "M1": "F"}
        )

    spans = cli.changed_spans(
        data, before_column="id_final_pre_motion", after_column="id_final"
    )

    assert [(span["start_frame"], span["end_frame"]) for span in spans] == [(1, 2), (5, 5)]
    assert spans[0]["n_frames"] == 2
    assert spans[0]["changed_rows"] == 4
    assert spans[1]["geometry_slots"] == ["0", "1"]


def test_parser_exposes_optimizer_weights_and_gates(tmp_path) -> None:
    args = cli.build_parser().parse_args(
        [
            str(tmp_path / "input.csv"),
            "--output-csv",
            str(tmp_path / "output.csv"),
            "--velocity-weight",
            "2.5",
            "--tangent-weight",
            "4",
            "--hard-jump-mm",
            "7.5",
            "--max-gap-frames",
            "12",
        ]
    )

    assert args.velocity_weight == pytest.approx(2.5)
    assert args.tangent_weight == pytest.approx(4.0)
    assert args.hard_jump_mm == pytest.approx(7.5)
    assert args.max_gap_frames == 12
    assert args.strategy == "layered"
    assert args.max_flicker_frames == 166
    assert args.search_radius_frames == 350
    assert args.local_window_radius_frames == 166
    assert cli._config_kwargs(args) == {
        "velocity_weight": 2.5,
        "tangent_weight": 4.0,
        "hard_jump_mm": 7.5,
        "max_gap_frames": 12,
    }


def test_run_writes_auditable_csv_and_report(tmp_path, monkeypatch) -> None:
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    report_json = tmp_path / "report.json"
    data = _track(
        [
            ("F", "M1", "M2"),
            ("M1", "F", "M2"),
            ("F", "M1", "M2"),
        ]
    )
    data.to_csv(input_csv, index=False)

    fake_module = types.ModuleType(
        "sam2_id_experiment.src.id_model.motion_identity_optimizer"
    )

    @dataclass
    class FakeConfig:
        velocity_weight: float = 1.0

    def fake_optimize(track: pd.DataFrame, *, config: FakeConfig):
        optimized = track.copy()
        optimized["id_final"] = np.tile(np.asarray(cli.FISH_IDS, dtype=object), 3)
        return optimized, {"test_optimizer": True}

    fake_module.MotionIdentityConfig = FakeConfig
    fake_module.optimize_identity_track = fake_optimize
    monkeypatch.setitem(
        sys.modules,
        "sam2_id_experiment.src.id_model.motion_identity_optimizer",
        fake_module,
    )
    args = cli.build_parser().parse_args(
        [
            str(input_csv),
            "--output-csv",
            str(output_csv),
            "--report-json",
            str(report_json),
            "--strategy",
            "viterbi",
            "--velocity-weight",
            "3",
        ]
    )

    report = cli.run(args)
    output = pd.read_csv(output_csv)
    report_on_disk = json.loads(report_json.read_text(encoding="utf-8"))

    assert report["changed_rows"] == 2
    assert report["changed_frames"] == 1
    assert report["speed_qc"]["before"]["all_ids"]["jumps_over_threshold"] == 4
    assert report["speed_qc"]["after"]["all_ids"]["jumps_over_threshold"] == 0
    assert report["permutation_qc"]["after"]["permutation_changes"] == 0
    assert report["one_each_validity"]["after"]["valid_fraction"] == 1.0
    assert report_on_disk["optimizer_diagnostics"] == {"test_optimizer": True}
    assert output["id_final_pre_motion"].tolist() == data["id_final"].tolist()
    changed_rows = (output["frame"] == 1) & output["geometry_slot"].isin([0, 1])
    assert set(output.loc[changed_rows, "id_source"]) == {"motion_global"}
    assert set(output.loc[~changed_rows, "id_source"]) == {"appearance"}


def test_run_uses_conservative_layered_strategy_by_default(tmp_path) -> None:
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    data = _track(
        [
            ("F", "M1", "M2"),
            ("F", "M1", "M2"),
            ("M1", "F", "M2"),
            ("F", "M1", "M2"),
        ]
    )
    data.to_csv(input_csv, index=False)
    args = cli.build_parser().parse_args(
        [str(input_csv), "--output-csv", str(output_csv)]
    )

    report = cli.run(args)
    output = pd.read_csv(output_csv)

    assert report["strategy"] == "layered"
    assert report["speed_qc"]["after"]["all_ids"]["jumps_over_threshold"] == 0
    assert report["optimizer_diagnostics"]["removed_flicker_spans"] == 1
    changed = (output.frame == 2) & output.geometry_slot.isin([0, 1])
    assert set(output.loc[changed, "id_source"]) == {"motion_layered"}
