"""Compare 3D identity methods against sparse female ground truth.

Each method supplies a 3D CSV and its coordinate definition (``mouth`` or
``centroid``). The corresponding point from the manually selected SLEAP pose is
used for per-view reprojection matching, so methods are not penalized merely
for reconstructing a different anatomical point.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.reconstruction_3d.calibration import Calibration


CAMERA_INDEX = {"side1": 1, "side2": 2}


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(len(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def _selected_point(row: pd.Series, prefix: str, anchor: str) -> np.ndarray:
    mouth = np.asarray([row[f"{prefix}_mouth_x"], row[f"{prefix}_mouth_y"]], float)
    if anchor == "mouth":
        return mouth
    tail = np.asarray([row[f"{prefix}_tail_x"], row[f"{prefix}_tail_y"]], float)
    visible = [point for point in (mouth, tail) if np.isfinite(point).all()]
    return np.mean(visible, axis=0) if visible else np.full(2, np.nan)


def _load_method(path: Path) -> tuple[pd.DataFrame, str]:
    header = pd.read_csv(path, nrows=0).columns
    id_column = next(
        (name for name in ("final_fish_id", "fish_id", "geometry_id") if name in header),
        None,
    )
    if id_column is None:
        raise ValueError(f"{path} has no fish identity column")
    usecols = ["frame", id_column, "X", "Y", "Z"]
    if "is_interpolated" in header:
        usecols.append("is_interpolated")
    table = pd.read_csv(path, usecols=usecols, low_memory=False)
    table["frame"] = pd.to_numeric(table["frame"], errors="raise").astype(np.int64)
    table["method_id"] = table[id_column].astype(str)
    if table.duplicated(["frame", "method_id"]).any():
        raise ValueError(f"{path} has duplicate frame/identity rows")
    if "is_interpolated" not in table:
        table["is_interpolated"] = False
    else:
        table["is_interpolated"] = table["is_interpolated"].astype(bool)
    return table, id_column


def _track_qc(table: pd.DataFrame) -> dict[str, Any]:
    xyz = table[["X", "Y", "Z"]].to_numpy(float)
    finite = np.isfinite(xyz).all(axis=1)
    observed = finite & ~table["is_interpolated"].to_numpy(bool)
    frames = np.sort(table["frame"].unique())
    ids = sorted(table["method_id"].unique())
    finite_counts = (
        pd.DataFrame({"frame": table["frame"], "finite": finite})
        .groupby("frame", sort=True)["finite"]
        .sum()
        .reindex(frames, fill_value=0)
    )
    observed_counts = (
        pd.DataFrame({"frame": table["frame"], "observed": observed})
        .groupby("frame", sort=True)["observed"]
        .sum()
        .reindex(frames, fill_value=0)
    )
    jumps: list[float] = []
    spike_count = 0
    consecutive_count = 0
    for method_id in ids:
        group = table.loc[table["method_id"].eq(method_id)].sort_values("frame")
        group_xyz = group[["X", "Y", "Z"]].to_numpy(float)
        group_frames = group["frame"].to_numpy(np.int64)
        valid_pair = (
            np.diff(group_frames).eq(1)
            if isinstance(np.diff(group_frames), pd.Series)
            else np.diff(group_frames) == 1
        )
        if len(group_xyz) > 1:
            valid_pair &= np.isfinite(group_xyz[:-1]).all(axis=1)
            valid_pair &= np.isfinite(group_xyz[1:]).all(axis=1)
            distances = np.linalg.norm(np.diff(group_xyz, axis=0), axis=1)
            kept = distances[valid_pair]
            jumps.extend(kept.tolist())
            consecutive_count += int(len(kept))
            spike_count += int(np.sum(kept > 8.0))
    return {
        "frame_min": int(frames.min()),
        "frame_max": int(frames.max()),
        "frame_count": int(len(frames)),
        "identity_count": int(len(ids)),
        "identities": ids,
        "finite_row_share": float(finite.mean()),
        "observed_noninterpolated_row_share": float(observed.mean()),
        "all_three_finite_frame_share": float(finite_counts.eq(3).mean()),
        "all_three_observed_frame_share": float(observed_counts.eq(3).mean()),
        "consecutive_jump_mm": _distribution(np.asarray(jumps, float)),
        "jump_over_8mm_count": spike_count,
        "jump_over_8mm_share": float(spike_count / max(consecutive_count, 1)),
    }


def _annotation_assignments(
    annotations: pd.DataFrame,
    table: pd.DataFrame,
    *,
    method: str,
    anchor: str,
    calibration: Calibration,
    gate_px: float,
    strict_gate_px: float,
) -> pd.DataFrame:
    frame_groups = {
        int(frame): group for frame, group in table.groupby("frame", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for _, annotation in annotations.iterrows():
        frame = int(annotation["frame"])
        group = frame_groups.get(frame)
        record: dict[str, Any] = {
            "method": method,
            "anchor": anchor,
            "annotation_id": str(annotation["annotation_id"]),
            "frame": frame,
            "gt_status": str(annotation["gt_status"]),
            "sex_label": str(annotation["sex_label"]),
        }
        choices: dict[str, str] = {}
        distances: dict[str, float] = {}
        if group is not None:
            xyz = group[["X", "Y", "Z"]].to_numpy(float)
            finite = np.isfinite(xyz).all(axis=1)
            projected = np.full((len(group), 3, 2), np.nan)
            if finite.any():
                projected[finite] = np.asarray(calibration.project_world(xyz[finite]), float)
            method_ids = group["method_id"].astype(str).to_numpy()
            for prefix in ("side1", "side2"):
                selected = _selected_point(annotation, prefix, anchor)
                view_points = projected[:, CAMERA_INDEX[prefix], :]
                costs = np.linalg.norm(view_points - selected[None, :], axis=1)
                if np.isfinite(costs).any():
                    index = int(np.nanargmin(costs))
                    distance = float(costs[index])
                    distances[prefix] = distance
                    if distance <= gate_px:
                        choices[prefix] = method_ids[index]
                record[f"{prefix}_distance_px"] = distance if np.isfinite(costs).any() else np.nan
                record[f"{prefix}_method_id"] = choices.get(prefix, "")
        available = [choices.get(prefix, "") for prefix in ("side1", "side2")]
        available = [value for value in available if value]
        if len(available) == 2 and available[0] == available[1]:
            status, consensus = "agree", available[0]
        elif len(available) == 2:
            status, consensus = "disagree", ""
        elif len(available) == 1:
            status, consensus = "one_view", available[0]
        else:
            status, consensus = "unmapped", ""
        strict = (
            status == "agree"
            and all(distances.get(prefix, np.inf) <= strict_gate_px for prefix in ("side1", "side2"))
        )
        record.update(
            {
                "mapping_status": status,
                "consensus_method_id": consensus,
                "strict_two_view": bool(strict),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def _assignment_summary(assignments: pd.DataFrame) -> dict[str, Any]:
    eligible = assignments.loc[
        assignments["gt_status"].eq("confirmed") & assignments["sex_label"].eq("F")
    ].copy()
    mapped = eligible.loc[eligible["consensus_method_id"].ne("")].sort_values("frame")
    strict = mapped.loc[mapped["strict_two_view"]]

    def stats(group: pd.DataFrame) -> dict[str, Any]:
        counts = Counter(group["consensus_method_id"].astype(str))
        best_id, best_count = counts.most_common(1)[0] if counts else ("", 0)
        values = group["consensus_method_id"].astype(str).to_numpy()
        switches = int(np.sum(values[1:] != values[:-1])) if len(values) > 1 else 0
        return {
            "count": int(len(group)),
            "identity_counts": dict(sorted(counts.items())),
            "best_fixed_id": best_id,
            "best_fixed_accuracy": float(best_count / max(len(group), 1)),
            "observed_run_count": switches + (1 if len(group) else 0),
            "observed_switch_count": switches,
        }

    return {
        "eligible_confirmed_female_annotations": int(len(eligible)),
        "mapping_status_counts": {
            str(key): int(value)
            for key, value in eligible["mapping_status"].value_counts().sort_index().items()
        },
        "mapped": stats(mapped),
        "strict_two_view": stats(strict),
    }


def _cohort_stats(group: pd.DataFrame) -> dict[str, Any]:
    ordered = group.sort_values("frame", kind="stable")
    counts = Counter(ordered["consensus_method_id"].astype(str))
    best_id, best_count = counts.most_common(1)[0] if counts else ("", 0)
    values = ordered["consensus_method_id"].astype(str).to_numpy()
    switches = int(np.sum(values[1:] != values[:-1])) if len(values) > 1 else 0
    return {
        "count": int(len(ordered)),
        "best_fixed_id": best_id,
        "best_fixed_accuracy": float(best_count / max(len(ordered), 1)),
        "observed_switch_count": switches,
    }


def _pairwise_common_cohorts(assignments: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    methods = sorted(assignments["method"].unique())
    columns = [
        "annotation_id", "frame", "method", "consensus_method_id", "strict_two_view"
    ]
    for left, right in itertools.combinations(methods, 2):
        pair = assignments.loc[assignments["method"].isin((left, right)), columns]
        present = (
            pair.assign(mapped=pair["consensus_method_id"].ne(""))
            .groupby("annotation_id", sort=False)["mapped"]
            .agg(["size", "sum"])
        )
        common_ids = set(present.index[(present["size"] == 2) & (present["sum"] == 2)])
        common = pair.loc[pair["annotation_id"].isin(common_ids)].copy()
        strict_ids = set(
            common.groupby("annotation_id", sort=False)["strict_two_view"]
            .all()
            .loc[lambda values: values]
            .index
        )
        for cohort, cohort_table in (
            ("common_mapped", common),
            ("common_strict", common.loc[common["annotation_id"].isin(strict_ids)]),
        ):
            for method in (left, right):
                stats = _cohort_stats(cohort_table.loc[cohort_table["method"].eq(method)])
                records.append(
                    {
                        "left_method": left,
                        "right_method": right,
                        "cohort": cohort,
                        "method": method,
                        **stats,
                    }
                )
    return pd.DataFrame(records)


def _event_summary(
    evidence_path: Path,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    evidence = pd.read_csv(evidence_path)
    if "subset" in evidence:
        evidence = evidence.loc[evidence["subset"].eq("strict_two_view_8px")]
    assignment_lookup = {
        (str(row.method), int(row.frame)): str(row.consensus_method_id)
        for row in assignments.itertuples()
    }
    records: list[dict[str, Any]] = []
    methods = sorted(assignments["method"].unique())
    for tracklet_id, group in evidence.groupby("clean_tracklet_id", sort=True):
        ordered = group.sort_values("frame", kind="stable").drop_duplicates("frame")
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        for method in methods:
            before = assignment_lookup.get((method, int(first["frame"])), "")
            after = assignment_lookup.get((method, int(last["frame"])), "")
            eligible = bool(before and after)
            records.append(
                {
                    "tracklet_id": str(tracklet_id),
                    "method": method,
                    "first_frame": int(first["frame"]),
                    "last_frame": int(last["frame"]),
                    "gt_geometry_id_first": str(first["consensus_geometry_id"]),
                    "gt_geometry_id_last": str(last["consensus_geometry_id"]),
                    "method_id_first": before,
                    "method_id_last": after,
                    "event_evaluable": eligible,
                    "female_id_preserved": bool(eligible and before == after),
                }
            )
    return pd.DataFrame(records)


def _parse_method(values: Sequence[str]) -> tuple[str, str, Path]:
    if len(values) != 3:
        raise ValueError("--method requires NAME ANCHOR CSV")
    name, anchor, path = values
    if anchor not in {"mouth", "centroid"}:
        raise ValueError("method anchor must be mouth or centroid")
    return name, anchor, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-csv", type=Path, required=True)
    parser.add_argument("--mixed-evidence-csv", type=Path, required=True)
    parser.add_argument("--calibration-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", nargs=3, action="append", required=True, metavar=("NAME", "ANCHOR", "CSV"))
    parser.add_argument("--gate-px", type=float, default=80.0)
    parser.add_argument("--strict-gate-px", type=float, default=8.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    annotations = pd.read_csv(args.annotations_csv, low_memory=False)
    numeric_columns = [
        "frame",
        "side1_mouth_x", "side1_mouth_y", "side1_tail_x", "side1_tail_y",
        "side2_mouth_x", "side2_mouth_y", "side2_tail_x", "side2_tail_y",
    ]
    for column in numeric_columns:
        annotations[column] = pd.to_numeric(annotations[column], errors="coerce")
    annotations["frame"] = annotations["frame"].astype(np.int64)
    calibration = Calibration(args.calibration_model_dir, backend="refractive")

    all_assignments: list[pd.DataFrame] = []
    summaries: dict[str, Any] = {}
    for raw in args.method:
        name, anchor, path = _parse_method(raw)
        table, id_column = _load_method(path)
        available_frames = set(table["frame"].unique())
        in_range = annotations.loc[annotations["frame"].isin(available_frames)].copy()
        assignments = _annotation_assignments(
            in_range,
            table,
            method=name,
            anchor=anchor,
            calibration=calibration,
            gate_px=args.gate_px,
            strict_gate_px=args.strict_gate_px,
        )
        all_assignments.append(assignments)
        summaries[name] = {
            "path": str(path.resolve()),
            "anchor": anchor,
            "id_column": id_column,
            "track_qc": _track_qc(table),
            "female_gt": _assignment_summary(assignments),
        }

    combined = pd.concat(all_assignments, ignore_index=True)
    events = _event_summary(args.mixed_evidence_csv, combined)
    event_counts = (
        events.groupby("method", sort=True)
        .agg(
            event_count=("tracklet_id", "size"),
            evaluable_events=("event_evaluable", "sum"),
            preserved_events=("female_id_preserved", "sum"),
        )
        .reset_index()
    )
    for row in event_counts.itertuples():
        summaries[str(row.method)]["mixed_event_results"] = {
            "event_count": int(row.event_count),
            "evaluable_events": int(row.evaluable_events),
            "preserved_events": int(row.preserved_events),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "annotation_assignments.csv", index=False)
    events.to_csv(args.output_dir / "mixed_event_results.csv", index=False)
    pairwise = _pairwise_common_cohorts(combined)
    pairwise.to_csv(args.output_dir / "pairwise_common_cohorts.csv", index=False)
    rows = []
    for method, summary in summaries.items():
        mapped = summary["female_gt"]["mapped"]
        strict = summary["female_gt"]["strict_two_view"]
        event = summary.get("mixed_event_results", {})
        rows.append(
            {
                "method": method,
                "anchor": summary["anchor"],
                "eligible_gt": summary["female_gt"]["eligible_confirmed_female_annotations"],
                "mapped_gt": mapped["count"],
                "mapped_best_fixed_accuracy": mapped["best_fixed_accuracy"],
                "mapped_switches": mapped["observed_switch_count"],
                "strict_gt": strict["count"],
                "strict_best_fixed_accuracy": strict["best_fixed_accuracy"],
                "strict_switches": strict["observed_switch_count"],
                "all_three_observed_frame_share": summary["track_qc"]["all_three_observed_frame_share"],
                "jump_over_8mm_share": summary["track_qc"]["jump_over_8mm_share"],
                "evaluable_mixed_events": event.get("evaluable_events", 0),
                "preserved_mixed_events": event.get("preserved_events", 0),
            }
        )
    pd.DataFrame(rows).to_csv(args.output_dir / "method_summary.csv", index=False)
    payload = {
        "schema_version": 1,
        "coordinate_fairness": (
            "Each method is reprojected against the corresponding mouth or visible-node "
            "centroid of the same manually selected SLEAP instance."
        ),
        "gate_px": args.gate_px,
        "strict_gate_px": args.strict_gate_px,
        "methods": summaries,
    }
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
