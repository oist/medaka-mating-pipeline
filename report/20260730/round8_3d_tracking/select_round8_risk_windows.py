from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPERIMENTS = (
    "20251225_K_1_1_FCMC",
    "20251225_K_1_2_FCMC",
    "20251225_K_1_3_FCMC",
)
EXPECTED_FRAMES = 348_600
RESIDUAL_TARGETS: dict[str, tuple[tuple[int, float | None, str], ...]] = {
    "20251225_K_1_1_FCMC": (
        (198_209, 9579.575, "source_fish=c;three_view"),
        (592, 7279.140, "source_fish=b;synthetic_cam1"),
        (300_739, 6714.738, "source_fish=c;three_view"),
    ),
    "20251225_K_1_2_FCMC": (
        (39_839, 6646.176641765475, "prelink_residual_extreme"),
        (167_926, 6180.204909405345, "prelink_residual_extreme"),
        (343_773, 6103.997941199748, "prelink_residual_extreme"),
    ),
    "20251225_K_1_3_FCMC": (
        (1_805, 9623.513835656326, "prelink_residual_extreme"),
        (95_640, 9055.731879281328, "prelink_residual_extreme"),
        (111_197, 8300.955364363776, "prelink_residual_extreme"),
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Window:
    experiment: str
    reason: str
    center_frame: int
    start_frame: int
    end_frame: int
    metric_name: str
    metric_value: float | None
    source_fish_id: str = ""


def separated_rank(
    rows: Iterable[tuple[int, float, str]],
    *,
    count: int,
    min_spacing: int,
    reverse: bool,
) -> list[tuple[int, float, str]]:
    selected: list[tuple[int, float, str]] = []
    for frame, value, label in sorted(
        rows, key=lambda row: row[1], reverse=reverse
    ):
        if all(abs(frame - previous[0]) >= min_spacing for previous in selected):
            selected.append((frame, value, label))
            if len(selected) == count:
                break
    return selected


def top_contact_candidates(path: Path) -> tuple[list[tuple[int, float, str]], dict]:
    table = pd.read_csv(path, usecols=["frame", "track_id", "x", "y"])
    counts = table.groupby("frame", sort=False).size()
    exactly_three_frames = counts.index[counts.eq(3)]
    selected = table[table["frame"].isin(exactly_three_frames)].copy()
    selected = selected.sort_values(["frame", "track_id"], kind="stable")
    selected["slot"] = selected.groupby("frame", sort=False).cumcount()
    x = selected.pivot(index="frame", columns="slot", values="x")
    y = selected.pivot(index="frame", columns="slot", values="y")
    ids = selected.pivot(index="frame", columns="slot", values="track_id")
    frames = x.index.to_numpy(np.int64)
    xy = np.stack((x.to_numpy(float), y.to_numpy(float)), axis=-1)
    distances = np.column_stack(
        (
            np.linalg.norm(xy[:, 0] - xy[:, 1], axis=1),
            np.linalg.norm(xy[:, 0] - xy[:, 2], axis=1),
            np.linalg.norm(xy[:, 1] - xy[:, 2], axis=1),
        )
    )
    nearest = np.argmin(distances, axis=1)
    values = distances[np.arange(len(distances)), nearest]
    pairs = np.asarray(((0, 1), (0, 2), (1, 2)), dtype=np.int64)
    nearest_pairs = pairs[nearest]
    id_values = ids.astype(str).to_numpy()
    candidates = [
        (
            int(frame),
            float(value),
            f"{id_values[index, pair[0]]}|{id_values[index, pair[1]]}",
        )
        for index, (frame, value, pair) in enumerate(
            zip(frames, values, nearest_pairs, strict=True)
        )
    ]
    return candidates, {
        "frames_with_exactly_three_top_tracks": int(len(frames)),
        "nearest_pair_distance_px": {
            "min": float(np.min(values)),
            "p01": float(np.percentile(values, 1)),
            "median": float(np.median(values)),
        },
    }


def displacement_candidates(
    path: Path,
) -> tuple[list[tuple[int, float, str]], dict]:
    table = pd.read_csv(path, usecols=["frame", "fish_id", "X", "Y", "Z"])
    candidates: list[tuple[int, float, str]] = []
    values: list[float] = []
    for fish_id, group in table.groupby("fish_id", sort=True):
        ordered = group.sort_values("frame", kind="stable")
        frames = ordered["frame"].to_numpy(np.int64)
        xyz = ordered[["X", "Y", "Z"]].to_numpy(float)
        steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        consecutive = np.diff(frames) == 1
        finite = consecutive & np.isfinite(steps)
        values.extend(steps[finite].tolist())
        for frame, step in zip(frames[1:][finite], steps[finite], strict=True):
            candidates.append((int(frame), float(step), str(fish_id)))
    array = np.asarray(values, dtype=float)
    return candidates, {
        "adjacent_finite_step_count": int(array.size),
        "adjacent_step_mm": {
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)),
            "max": float(np.max(array)),
        },
    }


def geometry_only_runs(path: Path) -> tuple[list[tuple[int, float, str]], dict]:
    table = pd.read_csv(path, usecols=["frame", "bind_source"])
    by_frame = (
        table.assign(is_geometry=table["bind_source"].eq("geometry_only"))
        .groupby("frame", sort=True)["is_geometry"]
        .sum()
    )
    active = by_frame[by_frame > 0]
    runs: list[tuple[int, int, int, int]] = []
    if not active.empty:
        frames = active.index.to_numpy(np.int64)
        start = previous = int(frames[0])
        maximum_count = int(active.iloc[0])
        for frame in frames[1:]:
            frame = int(frame)
            if frame != previous + 1:
                runs.append(
                    (start, previous, previous - start + 1, maximum_count)
                )
                start = frame
                maximum_count = 0
            maximum_count = max(maximum_count, int(active.loc[frame]))
            previous = frame
        runs.append((start, previous, previous - start + 1, maximum_count))
    candidates = [
        (
            int((start + end) // 2),
            float(length),
            f"run={start}:{end};max_geometry_rows={maximum}",
        )
        for start, end, length, maximum in runs
    ]
    return candidates, {
        "geometry_only_rows": int(
            table["bind_source"].eq("geometry_only").sum()
        ),
        "frames_with_geometry_only": int(len(active)),
        "run_count": len(runs),
        "longest_run_frames": max((run[2] for run in runs), default=0),
    }


def add_windows(
    destination: list[Window],
    *,
    experiment: str,
    reason: str,
    candidates: list[tuple[int, float, str]],
    metric_name: str,
    count: int,
    min_spacing: int,
    reverse: bool,
    radius: int,
) -> None:
    for frame, value, label in separated_rank(
        candidates,
        count=count,
        min_spacing=min_spacing,
        reverse=reverse,
    ):
        destination.append(
            Window(
                experiment=experiment,
                reason=reason,
                center_frame=frame,
                start_frame=max(0, frame - radius),
                end_frame=min(EXPECTED_FRAMES - 1, frame + radius),
                metric_name=metric_name,
                metric_value=value,
                source_fish_id=label,
            )
        )


def append_fixed_window(
    destination: list[Window],
    *,
    experiment: str,
    reason: str,
    center_frame: int,
    metric_name: str,
    metric_value: float | None,
    label: str,
    radius: int,
) -> None:
    destination.append(
        Window(
            experiment=experiment,
            reason=reason,
            center_frame=center_frame,
            start_frame=max(0, center_frame - radius),
            end_frame=min(EXPECTED_FRAMES - 1, center_frame + radius),
            metric_name=metric_name,
            metric_value=metric_value,
            source_fish_id=label,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleap-root", required=True, type=Path)
    parser.add_argument("--reconstruction-root", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--radius", type=int, default=75)
    parser.add_argument("--min-spacing", type=int, default=900)
    args = parser.parse_args()

    windows: list[Window] = []
    report: dict = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "tag": args.tag,
        "window_radius_frames": args.radius,
        "temporal_spacing_frames": args.min_spacing,
        "interpretation": (
            "These are risk-ranked QC windows, not confirmed errors. Low 2D "
            "separation targets contacts/crossings; high 3D step targets motion "
            "outliers; geometry-only runs target intervals lacking direct "
            "top-camera identity evidence. Explicit residual extremes, normal "
            "start/mid/end controls, inference chunk boundaries, and the K2 "
            "ambiguous-extra reference are also included."
        ),
        "experiments": {},
    }
    for experiment in EXPERIMENTS:
        top_path = (
            args.sleap_root
            / experiment
            / args.tag
            / "top_cam.tracks.csv"
        )
        output_dir = args.reconstruction_root / experiment / args.tag
        csv3d = output_dir / f"{args.tag}_3d.csv"
        identified = output_dir / f"{args.tag}_3d_identified.csv"

        contact, contact_summary = top_contact_candidates(top_path)
        displacement, displacement_summary = displacement_candidates(csv3d)
        geometry, geometry_summary = geometry_only_runs(identified)
        count = 5 if experiment.endswith("K_1_3_FCMC") else 3
        add_windows(
            windows,
            experiment=experiment,
            reason="low_top_pair_separation",
            candidates=contact,
            metric_name="nearest_pair_px",
            count=count,
            min_spacing=args.min_spacing,
            reverse=False,
            radius=args.radius,
        )
        for frame, residual, label in RESIDUAL_TARGETS[experiment]:
            append_fixed_window(
                windows,
                experiment=experiment,
                reason="high_prelink_residual",
                center_frame=frame,
                metric_name="residual_mm2",
                metric_value=residual,
                label=label,
                radius=args.radius,
            )
        for reason, frame in (
            ("control_start", 50),
            ("control_midpoint", 174_300),
            ("control_end", EXPECTED_FRAMES - 51),
        ):
            append_fixed_window(
                windows,
                experiment=experiment,
                reason=reason,
                center_frame=frame,
                metric_name="control",
                metric_value=None,
                label="normal_control",
                radius=args.radius,
            )
        for boundary in (87_150, 174_300, 261_450):
            append_fixed_window(
                windows,
                experiment=experiment,
                reason="inference_chunk_boundary",
                center_frame=boundary,
                metric_name="boundary_frame",
                metric_value=float(boundary),
                label="four_chunk_array_boundary",
                radius=args.radius,
            )
        if experiment == "20251225_K_1_2_FCMC":
            append_fixed_window(
                windows,
                experiment=experiment,
                reason="ambiguous_extra_candidate_reference",
                center_frame=174_305,
                metric_name="source_prediction_visual_qc",
                metric_value=None,
                label="side-view ambiguous extra candidate",
                radius=args.radius,
            )
        add_windows(
            windows,
            experiment=experiment,
            reason="high_adjacent_3d_step",
            candidates=displacement,
            metric_name="step_mm",
            count=count,
            min_spacing=args.min_spacing,
            reverse=True,
            radius=args.radius,
        )
        add_windows(
            windows,
            experiment=experiment,
            reason="long_geometry_only_run",
            candidates=geometry,
            metric_name="run_length_frames",
            count=count,
            min_spacing=args.min_spacing,
            reverse=True,
            radius=args.radius,
        )
        report["experiments"][experiment] = {
            "inputs": {
                "top_tracks_csv": {
                    "path": str(top_path),
                    "sha256": sha256(top_path),
                },
                "final_3d_csv": {
                    "path": str(csv3d),
                    "sha256": sha256(csv3d),
                },
                "identified_3d_csv": {
                    "path": str(identified),
                    "sha256": sha256(identified),
                },
            },
            "top_contact_summary": contact_summary,
            "displacement_summary": displacement_summary,
            "geometry_only_summary": geometry_summary,
        }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(asdict(window) for window in windows)
    table = table.sort_values(
        ["experiment", "reason", "metric_value"],
        ascending=[True, True, False],
        kind="stable",
    )
    table.to_csv(args.output_csv, index=False)
    report["windows"] = table.to_dict("records")
    args.output_json.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "csv": str(args.output_csv),
                "csv_sha256": sha256(args.output_csv),
                "json": str(args.output_json),
                "json_sha256": sha256(args.output_json),
                "window_count": len(table),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
