from __future__ import annotations

"""Compare two calibrations on identical SLEAP detections and settings.

The report deliberately separates three questions:

1. held-out rigid-object accuracy (the only absolute geometric evidence),
2. cross-camera assignment changes on round7 detections, and
3. downstream coverage and temporal smoothness.

Round7 fish tracks have no 3D ground truth, so they are a non-regression and
robustness test rather than an absolute calibration-accuracy test.
"""

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def _percentiles(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "median": None, "p90": None, "p95": None, "maximum": None}
    return {
        "n": int(len(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def summarize_tracks(path: Path, *, spike_threshold_mm: float) -> dict[str, object]:
    table = pd.read_csv(path)
    required = {"frame", "fish_id", "X", "Y", "Z"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    table = table.sort_values(["fish_id", "frame"])
    finite = np.all(np.isfinite(table[["X", "Y", "Z"]]), axis=1)
    speeds: list[float] = []
    gaps: dict[str, dict[str, int]] = {}
    for _, group in table.groupby("fish_id"):
        frame = group["frame"].to_numpy(int)
        xyz = group[["X", "Y", "Z"]].to_numpy(float)
        valid = np.all(np.isfinite(xyz), axis=1)
        run_lengths = []
        current = 0
        for missing_value in ~valid:
            if missing_value:
                current += 1
            elif current:
                run_lengths.append(current)
                current = 0
        if current:
            run_lengths.append(current)
        fish_id = str(group["fish_id"].iloc[0])
        gaps[fish_id] = {
            "missing_frames": int((~valid).sum()),
            "gap_count": int(len(run_lengths)),
            "longest_gap_frames": int(max(run_lengths)) if run_lengths else 0,
        }
        delta = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        keep = (np.diff(frame) == 1) & np.isfinite(delta)
        speeds.extend(delta[keep].tolist())
    speed = np.asarray(speeds, dtype=float)
    return {
        "frame_min": int(table["frame"].min()),
        "frame_max": int(table["frame"].max()),
        "n_frames": int(table["frame"].nunique()),
        "fish_ids": sorted(str(value) for value in table["fish_id"].unique()),
        "finite_row_rate": float(finite.mean()),
        "speed_mm_per_frame": _percentiles(speed),
        "speed_spike_threshold_mm": spike_threshold_mm,
        "speed_spike_rate": float(np.mean(speed > spike_threshold_mm)) if len(speed) else None,
        "gaps_by_fish": gaps,
    }


def _load_pre_link(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("frames", payload)


def _assignment_map(info: dict) -> dict[tuple[int, int, int], np.ndarray]:
    permutations = [info.get(f"perm_cam{camera}", []) for camera in range(3)]
    groups = info.get("groups", {})
    labels = list(groups)
    output: dict[tuple[int, int, int], np.ndarray] = {}
    for slot, label in enumerate(labels):
        if any(slot >= len(permutation) for permutation in permutations):
            continue
        key = tuple(int(permutation[slot]) for permutation in permutations)
        xyz = np.asarray(groups[label], dtype=float)
        if min(key) >= 0 and xyz.shape == (3,) and np.all(np.isfinite(xyz)):
            output[key] = xyz
    return output


def summarize_pre_link(frames: dict[str, dict], *, residual_threshold: float) -> dict[str, object]:
    valid_counts = []
    residuals = []
    recovered = 0
    slots = 0
    for info in frames.values():
        groups = info.get("groups", {})
        valid = 0
        for label, xyz in groups.items():
            slots += 1
            point = np.asarray(xyz, dtype=float)
            if point.shape == (3,) and np.all(np.isfinite(point)):
                valid += 1
            error = float(info.get("errors", {}).get(label, np.nan))
            if np.isfinite(error):
                residuals.append(error)
        valid_counts.append(valid)
        recovered += sum(bool(value) for value in info.get("recovered", []))
    residual_array = np.asarray(residuals, dtype=float)
    return {
        "n_frames": int(len(frames)),
        "all_3_valid_rate": float(np.mean(np.asarray(valid_counts) == 3)),
        "mean_valid_fish_per_frame": float(np.mean(valid_counts)),
        "recovered_slot_rate": float(recovered / slots) if slots else None,
        "reprojection_sum_squared_px": _percentiles(residual_array),
        "residual_threshold": residual_threshold,
        "residual_over_threshold_rate": (
            float(np.mean(residual_array > residual_threshold)) if len(residual_array) else None
        ),
    }


def compare_assignments(frames_a: dict[str, dict], frames_b: dict[str, dict]) -> dict[str, object]:
    common_frames = sorted(set(frames_a).intersection(frames_b), key=int)
    exact = 0
    shared_total = 0
    union_total = 0
    displacements = []
    changed_frames = []
    for frame in common_frames:
        assignments_a = _assignment_map(frames_a[frame])
        assignments_b = _assignment_map(frames_b[frame])
        keys_a = set(assignments_a)
        keys_b = set(assignments_b)
        if keys_a == keys_b:
            exact += 1
        else:
            changed_frames.append(int(frame))
        shared = keys_a.intersection(keys_b)
        shared_total += len(shared)
        union_total += len(keys_a.union(keys_b))
        displacements.extend(
            float(np.linalg.norm(assignments_a[key] - assignments_b[key]))
            for key in shared
        )
    changed_runs = []
    for frame in changed_frames:
        if not changed_runs or frame > changed_runs[-1][1] + 1:
            changed_runs.append([frame, frame])
        else:
            changed_runs[-1][1] = frame
    return {
        "n_common_frames": int(len(common_frames)),
        "exact_selected_triplet_set_rate": float(exact / len(common_frames)) if common_frames else None,
        "selected_triplet_jaccard": float(shared_total / union_total) if union_total else None,
        "n_shared_selected_triplets": int(shared_total),
        "changed_frame_count": int(len(changed_frames)),
        "changed_frame_runs": [
            {"start": start, "end": end, "length": end - start + 1}
            for start, end in changed_runs
        ],
        "same_detection_triplet_3d_displacement_mm": _percentiles(displacements),
    }


def assignment_disagreement_rows(frames_a: dict[str, dict], frames_b: dict[str, dict]):
    rows = []
    for frame in sorted(set(frames_a).intersection(frames_b), key=int):
        assignments_a = _assignment_map(frames_a[frame])
        assignments_b = _assignment_map(frames_b[frame])
        keys_a = set(assignments_a)
        keys_b = set(assignments_b)
        if keys_a == keys_b:
            continue
        shared = keys_a.intersection(keys_b)
        displacement = [
            float(np.linalg.norm(assignments_a[key] - assignments_b[key])) for key in shared
        ]
        rows.append(
            {
                "frame": int(frame),
                "triplets_a": json.dumps(sorted(keys_a)),
                "triplets_b": json.dumps(sorted(keys_b)),
                "shared_triplet_count": len(shared),
                "union_triplet_count": len(keys_a.union(keys_b)),
                "max_shared_triplet_displacement_mm": max(displacement) if displacement else np.nan,
            }
        )
    return rows


def compare_final_point_sets(csv_a: Path, csv_b: Path) -> dict[str, object]:
    a = pd.read_csv(csv_a)
    b = pd.read_csv(csv_b)
    frames = sorted(set(a["frame"]).intersection(b["frame"]))
    distances = []
    complete = 0
    for frame in frames:
        xyz_a = a[a["frame"] == frame][["X", "Y", "Z"]].to_numpy(float)
        xyz_b = b[b["frame"] == frame][["X", "Y", "Z"]].to_numpy(float)
        xyz_a = xyz_a[np.all(np.isfinite(xyz_a), axis=1)]
        xyz_b = xyz_b[np.all(np.isfinite(xyz_b), axis=1)]
        if not len(xyz_a) or not len(xyz_b):
            continue
        if len(xyz_a) == 3 and len(xyz_b) == 3:
            complete += 1
        cost = np.linalg.norm(xyz_a[:, None, :] - xyz_b[None, :, :], axis=2)
        rows, columns = linear_sum_assignment(cost)
        distances.extend(cost[rows, columns].tolist())
    return {
        "n_common_frames": int(len(frames)),
        "both_complete_frame_rate": float(complete / len(frames)) if frames else None,
        "hungarian_point_set_displacement_mm": _percentiles(distances),
    }


def _heldout_model(summary_path: Path | None, model_name: str | None):
    if summary_path is None or model_name is None:
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload["heldout_models"][model_name]


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def write_markdown(summary: dict[str, object], path: Path) -> None:
    a = summary["models"]["a"]
    b = summary["models"]["b"]
    assignment = summary["assignment_comparison"]
    point_sets = summary["final_point_set_comparison"]
    heldout_a = a.get("heldout")
    heldout_b = b.get("heldout")
    lines = [
        "# Round7 calibration comparison",
        "",
        "## Interpretation boundary",
        "",
        "Held-out rigid-object data provide the calibration accuracy evidence. The round7 fish data have no known 3D ground truth, so they test cross-camera assignment robustness, coverage, and temporal consistency only.",
        "",
        "## Results",
        "",
        f"| Metric | {a['label']} | {b['label']} |",
        "|---|---:|---:|",
    ]
    if heldout_a and heldout_b:
        lines.extend(
            [
                f"| Held-out Kabsch RMS (mm) | {_fmt(heldout_a['kabsch_residual_mm']['rms'])} | {_fmt(heldout_b['kabsch_residual_mm']['rms'])} |",
                f"| Held-out pairwise MAE (mm) | {_fmt(heldout_a['pairwise_distance_error_mm']['mae'])} | {_fmt(heldout_b['pairwise_distance_error_mm']['mae'])} |",
            ]
        )
    lines.extend(
        [
            f"| Round7 all-3-valid, pre-fill | {_fmt(100*a['pre_link']['all_3_valid_rate'], 2)}% | {_fmt(100*b['pre_link']['all_3_valid_rate'], 2)}% |",
            f"| Round7 reprojection P95 (sum squared px) | {_fmt(a['pre_link']['reprojection_sum_squared_px']['p95'], 1)} | {_fmt(b['pre_link']['reprojection_sum_squared_px']['p95'], 1)} |",
            f"| Final speed P95 (mm/frame) | {_fmt(a['tracks']['speed_mm_per_frame']['p95'])} | {_fmt(b['tracks']['speed_mm_per_frame']['p95'])} |",
            f"| Speed spikes > {a['tracks']['speed_spike_threshold_mm']:.1f} mm/frame | {_fmt(100*a['tracks']['speed_spike_rate'], 3)}% | {_fmt(100*b['tracks']['speed_spike_rate'], 3)}% |",
            "",
            "## Cross-model sensitivity",
            "",
            f"- Frames with exactly the same selected set of detection triplets: {_fmt(100*assignment['exact_selected_triplet_set_rate'], 2)}%.",
            f"- Selected-triplet Jaccard overlap: {_fmt(assignment['selected_triplet_jaccard'], 4)}.",
            f"- For identical detection triplets, 3D displacement median/P95: {_fmt(assignment['same_detection_triplet_3d_displacement_mm']['median'])} / {_fmt(assignment['same_detection_triplet_3d_displacement_mm']['p95'])} mm.",
            f"- Final per-frame point-set displacement median/P95: {_fmt(point_sets['hungarian_point_set_displacement_mm']['median'])} / {_fmt(point_sets['hungarian_point_set_displacement_mm']['p95'])} mm.",
            "",
            "## Adoption rule",
            "",
            "Use the new calibration only after the same comparison passes on stratified windows from all three experiments and on the full runs: held-out geometry must improve, pre-fill coverage and speed-spike rate must not regress materially, and every frame with a changed detection-triplet assignment near contact/crossing events should be reviewed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for suffix in ("a", "b"):
        parser.add_argument(f"--label-{suffix}", required=True)
        parser.add_argument(f"--csv-{suffix}", type=Path, required=True)
        parser.add_argument(f"--pre-link-{suffix}", type=Path, required=True)
        parser.add_argument(f"--heldout-model-{suffix}", default=None)
    parser.add_argument("--heldout-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--residual-threshold", type=float, default=5000.0)
    parser.add_argument("--spike-threshold-mm", type=float, default=8.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frames_a = _load_pre_link(args.pre_link_a)
    frames_b = _load_pre_link(args.pre_link_b)
    summary = {
        "scope": (
            "Held-out rigid-object metrics assess calibration geometry. Round7 fish metrics "
            "assess downstream consistency only because no fish 3D ground truth exists."
        ),
        "models": {
            "a": {
                "label": args.label_a,
                "tracks": summarize_tracks(args.csv_a, spike_threshold_mm=args.spike_threshold_mm),
                "pre_link": summarize_pre_link(frames_a, residual_threshold=args.residual_threshold),
                "heldout": _heldout_model(args.heldout_summary, args.heldout_model_a),
            },
            "b": {
                "label": args.label_b,
                "tracks": summarize_tracks(args.csv_b, spike_threshold_mm=args.spike_threshold_mm),
                "pre_link": summarize_pre_link(frames_b, residual_threshold=args.residual_threshold),
                "heldout": _heldout_model(args.heldout_summary, args.heldout_model_b),
            },
        },
        "assignment_comparison": compare_assignments(frames_a, frames_b),
        "final_point_set_comparison": compare_final_point_sets(args.csv_a, args.csv_b),
        "inputs": {
            "csv_a": str(args.csv_a),
            "csv_b": str(args.csv_b),
            "pre_link_a": str(args.pre_link_a),
            "pre_link_b": str(args.pre_link_b),
            "heldout_summary": str(args.heldout_summary) if args.heldout_summary else None,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "comparison_summary.json"
    markdown_path = args.output_dir / "CALIBRATION_COMPARISON.md"
    disagreement_path = args.output_dir / "assignment_disagreements.csv"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, markdown_path)
    pd.DataFrame(assignment_disagreement_rows(frames_a, frames_b)).to_csv(
        disagreement_path, index=False
    )
    print(json.dumps(summary, indent=2))
    print(f"Markdown report: {markdown_path}")
    print(f"Assignment review list: {disagreement_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
