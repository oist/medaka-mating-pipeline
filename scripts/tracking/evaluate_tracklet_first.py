from __future__ import annotations

"""Write compact QC metrics and a failure taxonomy for tracklet-first runs."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def _prelink_metrics(path: Path) -> tuple[dict, dict[int, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = {int(frame): info for frame, info in payload["frames"].items()}
    valid_counts = []
    residuals = []
    real_residuals = []
    real_all3 = []
    synthetic_slots = []
    for info in frames.values():
        valid = sum(
            bool(np.isfinite(np.asarray(info["groups"].get(label, []), float)).all())
            for label in payload.get("src_labels", ["a", "b", "c"])
        )
        recovered = [bool(value) for value in info.get("recovered", [])]
        valid_counts.append(valid)
        real_all3.append(valid == 3 and not any(recovered))
        synthetic_slots.extend(recovered)
        for source_idx, label in enumerate(payload.get("src_labels", ["a", "b", "c"])):
            value = info.get("errors", {}).get(label)
            if value is None or not np.isfinite(value):
                continue
            residuals.append(float(value))
            if not (source_idx < len(recovered) and recovered[source_idx]):
                real_residuals.append(float(value))
    counts = Counter(valid_counts)
    return {
        "frames": len(frames),
        "valid_fish_mean": float(np.mean(valid_counts)),
        "all3_any_fraction": float(np.mean(np.asarray(valid_counts) == 3)),
        "all3_real_fraction": float(np.mean(real_all3)),
        "valid_count_histogram": {str(i): int(counts[i]) for i in range(4)},
        "residual_median_px2": float(np.median(residuals)) if residuals else None,
        "residual_p95_px2": float(np.quantile(residuals, 0.95)) if residuals else None,
        "residual_p99_px2": float(np.quantile(residuals, 0.99)) if residuals else None,
        "real_residual_median_px2": float(np.median(real_residuals)) if real_residuals else None,
        "real_residual_p95_px2": float(np.quantile(real_residuals, 0.95)) if real_residuals else None,
        "synthetic_slot_fraction": float(np.mean(synthetic_slots)) if synthetic_slots else 0.0,
    }, frames


def _trajectory_metrics(path: Path) -> tuple[dict, pd.DataFrame]:
    table = pd.read_csv(path).sort_values(["fish_id", "frame"])
    finite = np.isfinite(table[["X", "Y", "Z"]]).all(axis=1)
    per_frame = table.assign(_finite=finite).groupby("frame")._finite.sum()
    jumps = []
    for _, group in table.groupby("fish_id"):
        xyz = group[["X", "Y", "Z"]].to_numpy(float)
        jumps.extend(np.linalg.norm(np.diff(xyz, axis=0), axis=1))
    jumps = np.asarray(jumps, float)
    return {
        "finite_row_fraction": float(finite.mean()),
        "all3_fraction": float((per_frame == 3).mean()),
        "frames_with_fewer_than_3": int((per_frame < 3).sum()),
        "speed_p95_mm_per_frame": float(np.nanquantile(jumps, 0.95)),
        "speed_max_mm_per_frame": float(np.nanmax(jumps)),
        "speed_over_8_rate": float(np.nanmean(jumps > 8.0)),
    }, table


def _window_metrics(
    name: str,
    start: int,
    end: int,
    pre_frames: dict[int, dict],
    trajectories: pd.DataFrame,
) -> dict:
    subset = {f: info for f, info in pre_frames.items() if start <= f <= end}
    valid = [
        sum(np.isfinite(np.asarray(info["groups"][label], float)).all() for label in ("a", "b", "c"))
        for info in subset.values()
    ]
    real = [
        count == 3 and not any(subset[f].get("recovered", []))
        for f, count in zip(subset, valid)
    ]
    t = trajectories[trajectories.frame.between(start, end)].copy()
    finite = np.isfinite(t[["X", "Y", "Z"]]).all(axis=1)
    per_frame = t.assign(_finite=finite).groupby("frame")._finite.sum()
    return {
        "name": name, "start": start, "end": end,
        "prelink_all3_any_fraction": float(np.mean(np.asarray(valid) == 3)) if valid else None,
        "prelink_all3_real_fraction": float(np.mean(real)) if real else None,
        "final_all3_fraction": float((per_frame == 3).mean()) if len(per_frame) else None,
    }


def evaluate(args: argparse.Namespace) -> dict:
    pre, pre_frames = _prelink_metrics(args.pre_link_json)
    trajectory, table = _trajectory_metrics(args.csv_3d)
    assignments = pd.read_csv(args.assignments)
    identified = pd.read_csv(args.identified_2d)
    dispositions = pd.read_csv(args.dispositions)
    key_cols = ["frame", "view", "candidate_slot"]
    duplicate_dispositions = int(dispositions.duplicated(key_cols).sum())
    internal_changes = int(
        (identified.groupby("node_id").final_fish_id.nunique() > 1).sum()
    )
    result = {
        "dataset": args.dataset,
        "prelink": pre,
        "trajectory": trajectory,
        "identity": {
            "tracklets": int(len(assignments)),
            "resolved_tracklets": int((assignments.assigned_identity.fillna("") != "").sum()),
            "unresolved_tracklets": int((assignments.assigned_identity.fillna("") == "").sum()),
            "locked_tracklet_internal_id_changes": internal_changes,
            "identified_2d_only_rows": int(
                (identified.observation_status == "assigned_2d_only").sum()
            ),
        },
        "dispositions": {
            "rows": int(len(dispositions)),
            "duplicate_frame_view_slot_rows": duplicate_dispositions,
            "reason_counts": {
                str(key): int(value)
                for key, value in dispositions.primary_reason.value_counts().items()
            },
        },
        "windows": [],
    }
    for spec in args.window:
        name, bounds = spec.split("=", 1)
        start, end = map(int, bounds.split(":"))
        result["windows"].append(
            _window_metrics(name, start, end, pre_frames, table)
        )
    if args.baseline_pre_link_json:
        baseline_pre, baseline_frames = _prelink_metrics(args.baseline_pre_link_json)
        result["baseline_prelink"] = baseline_pre
    if args.baseline_csv_3d:
        baseline_trajectory, baseline_table = _trajectory_metrics(args.baseline_csv_3d)
        result["baseline_trajectory"] = baseline_trajectory
    if args.baseline_pre_link_json and args.baseline_csv_3d:
        result["baseline_windows"] = []
        for spec in args.window:
            name, bounds = spec.split("=", 1)
            start, end = map(int, bounds.split(":"))
            result["baseline_windows"].append(
                _window_metrics(name, start, end, baseline_frames, baseline_table)
            )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--pre-link-json", type=Path, required=True)
    parser.add_argument("--csv-3d", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--identified-2d", type=Path, required=True)
    parser.add_argument("--dispositions", type=Path, required=True)
    parser.add_argument("--baseline-pre-link-json", type=Path)
    parser.add_argument("--baseline-csv-3d", type=Path)
    parser.add_argument("--window", action="append", default=[], metavar="NAME=START:END")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--failure-csv", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    assignments = pd.read_csv(args.assignments)
    unresolved = assignments[assignments.assigned_identity.fillna("") == ""].copy()
    unresolved["failure_type"] = "unresolved_identity_component"
    unresolved.to_csv(args.failure_csv, index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
