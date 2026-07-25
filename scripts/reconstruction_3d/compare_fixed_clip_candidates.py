from __future__ import annotations

"""Compare fixed-clip reconstruction candidates without inventing frame zero.

This scorer is designed for non-zero bounded clips and refractive calibration.
It separates pre-link observed geometry from spline-filled trajectories, reads
the residual unit from saved calibration provenance, and verifies that the
top-camera binding is a stable one-to-one relabel of geometry IDs.
"""

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClipWindow:
    name: str
    start: int
    end: int

    @property
    def expected_frames(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class CandidateFiles:
    name: str
    pre_link_json: Path
    trajectory_csv: Path
    identified_csv: Path | None


def parse_window(value: str) -> ClipWindow:
    name, separator, bounds = value.partition("=")
    start_text, bounds_separator, end_text = bounds.partition(":")
    if (
        not separator
        or not bounds_separator
        or not name.strip()
        or not start_text.strip()
        or not end_text.strip()
    ):
        raise ValueError(f"window must be NAME=START:END, got {value!r}")
    start, end = int(start_text), int(end_text)
    if start < 0 or end < start:
        raise ValueError(f"invalid inclusive frame bounds: {start}:{end}")
    return ClipWindow(name.strip(), start, end)


def parse_candidate(value: str) -> CandidateFiles:
    name, separator, raw_paths = value.partition("=")
    paths = [part.strip() for part in raw_paths.split(",")]
    if not separator or not name.strip() or len(paths) not in (2, 3):
        raise ValueError(
            "--candidate must be NAME=PRE_LINK_JSON,TRAJECTORY_CSV"
            "[,IDENTIFIED_CSV]"
        )
    if not paths[0] or not paths[1]:
        raise ValueError("pre-link JSON and trajectory CSV cannot be empty")
    return CandidateFiles(
        name=name.strip(),
        pre_link_json=Path(paths[0]),
        trajectory_csv=Path(paths[1]),
        identified_csv=Path(paths[2]) if len(paths) == 3 and paths[2] else None,
    )


def distribution(values: Iterable[float]) -> dict[str, int | float | None]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "n": 0,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }
    return {
        "n": int(len(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(np.max(array)),
    }


def _finite_xyz(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(array.shape == (3,) and np.isfinite(array).all())


def _calibration_provenance(payload: dict[str, Any]) -> dict[str, Any]:
    calibration = payload.get("calibration")
    if not isinstance(calibration, dict):
        return {
            "status": "legacy_unknown",
            "backend": None,
            "score_unit": None,
            "model_files": None,
        }
    triangulation = calibration.get("triangulation")
    if not isinstance(triangulation, dict):
        triangulation = calibration
    provenance = triangulation.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    backend = triangulation.get("backend") or provenance.get("backend")
    score_unit = (
        triangulation.get("score_unit") or provenance.get("score_unit")
    )
    model_files = provenance.get("model_files")
    status = (
        "complete"
        if backend is not None and score_unit is not None and model_files
        else "incomplete"
    )
    return {
        "status": status,
        "backend": str(backend) if backend is not None else None,
        "score_unit": str(score_unit) if score_unit is not None else None,
        "model_files": model_files,
    }


def summarize_pre_link(
    path: Path,
    *,
    window: ClipWindow,
    residual_gate: float | None = None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_frames = payload.get("frames", payload)
    if not isinstance(raw_frames, dict):
        raise ValueError(f"{path} does not contain a frame mapping")
    frames = {
        int(frame): info
        for frame, info in raw_frames.items()
        if window.start <= int(frame) <= window.end
    }
    labels = list(payload.get("src_labels") or ["a", "b", "c"])
    expected = range(window.start, window.end + 1)
    valid_counts: list[int] = []
    residuals: list[float] = []
    synthetic_slots: list[bool] = []
    synthetic_provenance_known = True
    for frame in expected:
        info = frames.get(frame)
        if not isinstance(info, dict):
            valid_counts.append(0)
            synthetic_provenance_known = False
            continue
        groups = info.get("groups") or {}
        valid_counts.append(
            sum(_finite_xyz(groups.get(label)) for label in labels)
        )
        errors = info.get("errors") or {}
        for label in labels:
            value = errors.get(label)
            if value is not None and np.isfinite(float(value)):
                residuals.append(float(value))
        if "synthetic_views" not in info:
            synthetic_provenance_known = False
        else:
            values = list(info.get("synthetic_views") or [])
            values.extend([-1] * max(0, len(labels) - len(values)))
            synthetic_slots.extend(int(value) >= 0 for value in values[: len(labels)])

    valid_array = np.asarray(valid_counts, dtype=int)
    residual_summary = distribution(residuals)
    residual_summary["gate"] = residual_gate
    residual_summary["gate_unit"] = (
        _calibration_provenance(payload).get("score_unit")
    )
    residual_summary["over_gate_count"] = (
        int(np.sum(np.asarray(residuals) > residual_gate))
        if residuals and residual_gate is not None
        else None
    )
    residual_summary["over_gate_rate"] = (
        float(np.mean(np.asarray(residuals) > residual_gate))
        if residuals and residual_gate is not None
        else None
    )
    present_valid = [
        count
        for frame, count in zip(expected, valid_counts)
        if frame in frames
    ]
    return {
        "path": str(path),
        "window": {
            "start": window.start,
            "end": window.end,
            "expected_frames": window.expected_frames,
        },
        "present_frames": int(len(frames)),
        "missing_frames": int(window.expected_frames - len(frames)),
        "frame_coverage": float(len(frames) / window.expected_frames),
        "source_labels": labels,
        "calibration": _calibration_provenance(payload),
        "all_three_valid_rate_expected": float(np.mean(valid_array == 3)),
        "all_three_valid_rate_present": (
            float(np.mean(np.asarray(present_valid) == 3))
            if present_valid
            else None
        ),
        "mean_valid_fish_per_expected_frame": float(np.mean(valid_array)),
        "valid_count_histogram": {
            str(count): int(np.sum(valid_array == count))
            for count in range(4)
        },
        "residual": residual_summary,
        "synthetic_view_provenance": (
            "complete" if synthetic_provenance_known else "legacy_unknown"
        ),
        "synthetic_slot_count": (
            int(sum(synthetic_slots))
            if synthetic_provenance_known
            else None
        ),
        "synthetic_slot_rate": (
            float(np.mean(synthetic_slots))
            if synthetic_provenance_known and synthetic_slots
            else (0.0 if synthetic_provenance_known else None)
        ),
    }


def _gap_lengths(missing: np.ndarray) -> list[int]:
    runs: list[int] = []
    current = 0
    for value in missing:
        if value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def _jump_summary(
    table: pd.DataFrame,
    *,
    require_observed: bool,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    jumps: list[float] = []
    for _, group in table.groupby("fish_id", sort=True):
        group = group.sort_values("frame")
        frame = group["frame"].to_numpy(dtype=int)
        xyz = group[["X", "Y", "Z"]].to_numpy(dtype=float)
        if len(group) < 2:
            continue
        keep = np.diff(frame) == 1
        keep &= np.isfinite(xyz[:-1]).all(axis=1)
        keep &= np.isfinite(xyz[1:]).all(axis=1)
        if require_observed:
            interpolated = group["_interpolated"].to_numpy(dtype=bool)
            keep &= ~interpolated[:-1]
            keep &= ~interpolated[1:]
        distances = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        jumps.extend(distances[keep].tolist())
    summary = distribution(jumps)
    jump_array = np.asarray(jumps, dtype=float)
    summary["thresholds"] = {
        str(threshold): {
            "count": int(np.sum(jump_array > threshold)),
            "denominator": int(len(jump_array)),
            "rate": (
                float(np.mean(jump_array > threshold))
                if len(jump_array)
                else None
            ),
        }
        for threshold in thresholds
    }
    return summary


def summarize_trajectory(
    path: Path,
    *,
    window: ClipWindow,
    speed_thresholds: tuple[float, ...] = (8.0, 20.0),
) -> dict[str, Any]:
    table = pd.read_csv(path)
    required = {"frame", "fish_id", "X", "Y", "Z"}
    missing_columns = required - set(table.columns)
    if missing_columns:
        raise ValueError(f"{path} missing columns: {sorted(missing_columns)}")
    table = table.loc[
        table["frame"].between(window.start, window.end)
    ].copy()
    table["frame"] = pd.to_numeric(table["frame"], errors="raise").astype(int)
    if table.duplicated(["frame", "fish_id"]).any():
        raise ValueError(f"{path} has duplicate frame/fish_id rows")
    interpolation_status = (
        "complete" if "is_interpolated" in table.columns else "legacy_unknown"
    )
    if "is_interpolated" in table:
        table["_interpolated"] = table["is_interpolated"].fillna(False).astype(bool)
    else:
        table["_interpolated"] = False
    fish_ids = sorted(str(value) for value in table["fish_id"].unique())
    finite = np.isfinite(table[["X", "Y", "Z"]]).all(axis=1)
    observed = finite & ~table["_interpolated"].to_numpy(dtype=bool)
    frame_index = pd.Index(
        range(window.start, window.end + 1), name="frame"
    )
    finite_per_frame = (
        table.assign(_finite=finite)
        .groupby("frame")["_finite"]
        .sum()
        .reindex(frame_index, fill_value=0)
    )
    observed_per_frame = (
        table.assign(_observed=observed)
        .groupby("frame")["_observed"]
        .sum()
        .reindex(frame_index, fill_value=0)
    )
    present_frames = int(table["frame"].nunique())
    gaps: dict[str, dict[str, Any]] = {}
    for fish_id in fish_ids:
        group = table.loc[
            table["fish_id"].astype(str).eq(fish_id)
        ].set_index("frame")
        xyz = group[["X", "Y", "Z"]].reindex(frame_index).to_numpy(float)
        missing = ~np.isfinite(xyz).all(axis=1)
        runs = _gap_lengths(missing)
        gaps[fish_id] = {
            "missing_frames": int(missing.sum()),
            "gap_count": int(len(runs)),
            "longest_gap_frames": int(max(runs)) if runs else 0,
        }
    return {
        "path": str(path),
        "present_frames": present_frames,
        "missing_frames": int(window.expected_frames - present_frames),
        "frame_coverage": float(present_frames / window.expected_frames),
        "fish_ids": fish_ids,
        "fish_count": int(len(fish_ids)),
        "row_count": int(len(table)),
        "finite_row_rate": float(finite.mean()) if len(table) else None,
        "all_three_finite_frame_rate": float(
            np.mean(finite_per_frame.to_numpy() == 3)
        ),
        "interpolation_provenance": interpolation_status,
        "interpolated_row_rate": (
            float(table["_interpolated"].mean())
            if interpolation_status == "complete" and len(table)
            else None
        ),
        "all_three_observed_frame_rate": (
            float(np.mean(observed_per_frame.to_numpy() == 3))
            if interpolation_status == "complete"
            else None
        ),
        "consecutive_jump_all_mm": _jump_summary(
            table,
            require_observed=False,
            thresholds=speed_thresholds,
        ),
        "consecutive_jump_observed_mm": (
            _jump_summary(
                table,
                require_observed=True,
                thresholds=speed_thresholds,
            )
            if interpolation_status == "complete"
            else None
        ),
        "gaps_by_fish": gaps,
    }


def summarize_identified(
    path: Path | None,
    *,
    window: ClipWindow,
) -> dict[str, Any]:
    if path is None:
        return {"status": "not_provided", "path": None}
    table = pd.read_csv(path)
    required = {
        "frame",
        "fish_id",
        "final_fish_id",
        "bind_source",
        "bind_score",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    table = table.loc[
        table["frame"].between(window.start, window.end)
    ].copy()
    if table.duplicated(["frame", "fish_id"]).any():
        raise ValueError(f"{path} has duplicate frame/fish_id rows")
    table["_geometry_id"] = table["fish_id"].astype(str)
    final = table["final_fish_id"].fillna("").astype(str).str.strip()
    final = final.mask(final.isin(["nan", "None"]), "")
    table["_final_id"] = final
    blank_final = int(table["_final_id"].eq("").sum())
    geometry_to_final = {
        str(geometry_id): sorted(
            value
            for value in group["_final_id"].unique()
            if value
        )
        for geometry_id, group in table.groupby("_geometry_id", sort=True)
    }
    final_to_geometry = {
        str(final_id): sorted(group["_geometry_id"].unique())
        for final_id, group in table.loc[
            table["_final_id"].ne("")
        ].groupby("_final_id", sort=True)
    }
    stable_one_to_one = bool(
        len(table)
        and blank_final == 0
        and all(len(values) == 1 for values in geometry_to_final.values())
        and all(len(values) == 1 for values in final_to_geometry.values())
        and len(geometry_to_final) == len(final_to_geometry)
    )
    source_counts = Counter(table["bind_source"].fillna("unknown").astype(str))
    scores = pd.to_numeric(table["bind_score"], errors="coerce").to_numpy(float)
    scores = scores[np.isfinite(scores)]
    expected_frames = set(range(window.start, window.end + 1))
    present_frames = set(pd.to_numeric(table["frame"], errors="raise").astype(int))
    return {
        "status": "complete",
        "path": str(path),
        "present_frames": int(len(present_frames)),
        "missing_frames": int(len(expected_frames - present_frames)),
        "frame_coverage": float(len(present_frames) / window.expected_frames),
        "bind_source_counts": dict(sorted(source_counts.items())),
        "top_cam_2d_row_rate": float(
            np.mean(table["bind_source"].astype(str).eq("top_cam_2d"))
        )
        if len(table)
        else None,
        "bind_score_px": distribution(scores),
        "blank_final_id_rows": blank_final,
        "geometry_to_final": geometry_to_final,
        "final_to_geometry": final_to_geometry,
        "stable_one_to_one_relabel": stable_one_to_one,
    }


def summarize_candidate(
    files: CandidateFiles,
    *,
    window: ClipWindow,
    residual_gate: float | None,
    speed_thresholds: tuple[float, ...],
    expected_backend: str,
    expected_score_unit: str,
) -> dict[str, Any]:
    pre_link = summarize_pre_link(
        files.pre_link_json,
        window=window,
        residual_gate=residual_gate,
    )
    trajectory = summarize_trajectory(
        files.trajectory_csv,
        window=window,
        speed_thresholds=speed_thresholds,
    )
    identified = summarize_identified(
        files.identified_csv,
        window=window,
    )
    calibration = pre_link["calibration"]
    checks = {
        "calibration_provenance_complete": (
            calibration["status"] == "complete"
        ),
        "expected_backend": calibration["backend"] == expected_backend,
        "expected_score_unit": (
            calibration["score_unit"] == expected_score_unit
        ),
        "pre_link_frame_complete": pre_link["missing_frames"] == 0,
        "synthetic_view_provenance_complete": (
            pre_link["synthetic_view_provenance"] == "complete"
        ),
        "trajectory_frame_complete": trajectory["missing_frames"] == 0,
        "three_geometry_ids": trajectory["fish_count"] == 3,
        "interpolation_provenance_complete": (
            trajectory["interpolation_provenance"] == "complete"
        ),
        "identified_frame_complete": (
            identified.get("status") == "complete"
            and identified.get("missing_frames") == 0
        ),
        "stable_one_to_one_relabel": bool(
            identified.get("stable_one_to_one_relabel")
        ),
    }
    return {
        "candidate": files.name,
        "window": {
            "name": window.name,
            "start": window.start,
            "end": window.end,
            "expected_frames": window.expected_frames,
        },
        "pre_link": pre_link,
        "trajectory": trajectory,
        "identified": identified,
        "structural_checks": checks,
        "structural_gate_pass": bool(all(checks.values())),
    }


def _number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _flat_row(summary: dict[str, Any]) -> dict[str, Any]:
    pre = summary["pre_link"]
    trajectory = summary["trajectory"]
    identified = summary["identified"]
    observed_jumps = trajectory.get("consecutive_jump_observed_mm") or {}
    thresholds = observed_jumps.get("thresholds") or {}
    return {
        "candidate": summary["candidate"],
        "window": summary["window"]["name"],
        "start": summary["window"]["start"],
        "end": summary["window"]["end"],
        "structural_gate_pass": summary["structural_gate_pass"],
        "calibration_backend": pre["calibration"]["backend"],
        "score_unit": pre["calibration"]["score_unit"],
        "prelink_all3_rate": pre["all_three_valid_rate_expected"],
        "residual_median": pre["residual"]["median"],
        "residual_p95": pre["residual"]["p95"],
        "residual_p99": pre["residual"]["p99"],
        "synthetic_slot_rate": pre["synthetic_slot_rate"],
        "all3_observed_rate": trajectory["all_three_observed_frame_rate"],
        "interpolated_row_rate": trajectory["interpolated_row_rate"],
        "observed_jump_p95_mm": observed_jumps.get("p95"),
        "observed_jump_p99_mm": observed_jumps.get("p99"),
        "observed_jump_max_mm": observed_jumps.get("maximum"),
        "observed_jump_over_8_rate": (
            thresholds.get("8.0", {}).get("rate")
        ),
        "observed_jump_over_20_rate": (
            thresholds.get("20.0", {}).get("rate")
        ),
        "top_cam_2d_row_rate": identified.get("top_cam_2d_row_rate"),
        "bind_score_p95_px": (
            identified.get("bind_score_px") or {}
        ).get("p95"),
        "stable_one_to_one_relabel": identified.get(
            "stable_one_to_one_relabel"
        ),
    }


def _delta_rows(
    rows: list[dict[str, Any]],
    *,
    baseline: str,
) -> list[dict[str, Any]]:
    by_name = {row["candidate"]: row for row in rows}
    if baseline not in by_name:
        raise ValueError(f"baseline {baseline!r} is not a candidate")
    baseline_row = by_name[baseline]
    metrics = (
        "prelink_all3_rate",
        "residual_p95",
        "all3_observed_rate",
        "interpolated_row_rate",
        "observed_jump_p99_mm",
        "observed_jump_over_8_rate",
        "top_cam_2d_row_rate",
        "bind_score_p95_px",
    )
    output: list[dict[str, Any]] = []
    for row in rows:
        delta = {
            "candidate": row["candidate"],
            "baseline": baseline,
        }
        for metric in metrics:
            value = _number(row.get(metric))
            reference = _number(baseline_row.get(metric))
            delta[f"{metric}_delta"] = (
                value - reference
                if value is not None and reference is not None
                else None
            )
        output.append(delta)
    return output


def write_reports(
    *,
    summaries: list[dict[str, Any]],
    baseline: str,
    output_json: Path,
    output_csv: Path,
    output_md: Path,
    expected_backend: str,
    expected_score_unit: str,
    residual_gate: float | None,
    speed_thresholds: tuple[float, ...],
) -> None:
    rows = [_flat_row(summary) for summary in summaries]
    deltas = _delta_rows(rows, baseline=baseline)
    payload = {
        "schema_version": "fixed_clip_reconstruction_qc_v1",
        "scientific_status": (
            "engineering_and_known_failure_gate_not_absolute_3d_ground_truth"
        ),
        "baseline": baseline,
        "expected_calibration": {
            "backend": expected_backend,
            "score_unit": expected_score_unit,
        },
        "residual_gate": residual_gate,
        "speed_thresholds_mm_per_frame": list(speed_thresholds),
        "candidates": summaries,
        "deltas_from_baseline": deltas,
        "limitations": [
            "Fish trajectories have no absolute 3D ground truth; this is a relative coverage/smoothness gate.",
            "A stable relabel is necessary but does not prove biological identity correctness.",
            "Scientific promotion requires independent event-disjoint manual labels.",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Fixed-clip 3D candidate QC",
        "",
        (
            f"Expected calibration: `{expected_backend}`, residual unit "
            f"`{expected_score_unit}`. Baseline: `{baseline}`."
        ),
        "",
        "| candidate | structural | pre-link all 3 | residual p95 | "
        "observed all 3 | jump p99 mm | >8 mm rate | top bind | 1:1 relabel |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(key: str) -> str:
            value = row.get(key)
            return "n/a" if value is None else f"{float(value):.4f}"

        lines.append(
            f"| {row['candidate']} | "
            f"{'pass' if row['structural_gate_pass'] else 'fail'} | "
            f"{fmt('prelink_all3_rate')} | "
            f"{fmt('residual_p95')} | "
            f"{fmt('all3_observed_rate')} | "
            f"{fmt('observed_jump_p99_mm')} | "
            f"{fmt('observed_jump_over_8_rate')} | "
            f"{fmt('top_cam_2d_row_rate')} | "
            f"{row.get('stable_one_to_one_relabel')} |"
        )
    lines.extend(
        [
            "",
            "This is a relative engineering/known-failure gate, not an "
            "independent scientific promotion result.",
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="NAME=PRE_LINK_JSON,TRAJECTORY_CSV[,IDENTIFIED_CSV]",
    )
    parser.add_argument("--window", required=True, metavar="NAME=START:END")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--expected-backend", default="refractive")
    parser.add_argument("--expected-score-unit", default="mm^2")
    parser.add_argument("--residual-gate", type=float, default=None)
    parser.add_argument(
        "--speed-thresholds",
        nargs="+",
        type=float,
        default=[8.0, 20.0],
    )
    args = parser.parse_args(argv)
    try:
        window = parse_window(args.window)
        candidates = [parse_candidate(value) for value in args.candidate]
    except ValueError as exc:
        parser.error(str(exc))
    if len({candidate.name for candidate in candidates}) != len(candidates):
        parser.error("candidate names must be unique")
    if args.baseline not in {candidate.name for candidate in candidates}:
        parser.error("--baseline must name one of the candidates")
    if args.residual_gate is not None and args.residual_gate < 0:
        parser.error("--residual-gate must be non-negative")
    speed_thresholds = tuple(sorted(set(args.speed_thresholds)))
    if not speed_thresholds or min(speed_thresholds) < 0:
        parser.error("--speed-thresholds must be non-negative")
    for candidate in candidates:
        for path in (
            candidate.pre_link_json,
            candidate.trajectory_csv,
            candidate.identified_csv,
        ):
            if path is not None and not path.is_file():
                parser.error(f"candidate artifact not found: {path}")

    summaries = [
        summarize_candidate(
            candidate,
            window=window,
            residual_gate=args.residual_gate,
            speed_thresholds=speed_thresholds,
            expected_backend=args.expected_backend,
            expected_score_unit=args.expected_score_unit,
        )
        for candidate in candidates
    ]
    write_reports(
        summaries=summaries,
        baseline=args.baseline,
        output_json=args.output_json,
        output_csv=args.output_csv,
        output_md=args.output_md,
        expected_backend=args.expected_backend,
        expected_score_unit=args.expected_score_unit,
        residual_gate=args.residual_gate,
        speed_thresholds=speed_thresholds,
    )
    print(
        pd.DataFrame([_flat_row(summary) for summary in summaries]).to_string(
            index=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
