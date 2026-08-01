from __future__ import annotations

"""Independently recompute Round8 method QC from published artifacts."""

import argparse
import csv
import hashlib
import json
import math
import mmap
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.reconstruction_3d.calibration import Calibration


CAMERA_INDEX = {"side1": 1, "side2": 2}
FLOAT_TOKEN = rb"(?:-?Infinity|NaN|null|-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
ERRORS_PATTERN = re.compile(
    rb'"errors"\s*:\s*\{\s*"a"\s*:\s*('
    + FLOAT_TOKEN
    + rb')\s*,\s*"b"\s*:\s*('
    + FLOAT_TOKEN
    + rb')\s*,\s*"c"\s*:\s*('
    + FLOAT_TOKEN
    + rb")\s*\}"
)


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {
            "count": 0,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(len(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def parse_float_token(value: bytes) -> float:
    if value == b"null":
        return math.nan
    return float(value)


def prelink_metrics(path: Path) -> dict[str, Any]:
    per_identity: list[list[float]] = [[], [], []]
    all_three = 0
    frame_count = 0
    with path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            prefix = bytes(mapped[: min(len(mapped), 20_000)])
            backend_match = re.search(
                rb'"triangulation"\s*:\s*\{.*?"backend"\s*:\s*"([^"]+)"',
                prefix,
                flags=re.DOTALL,
            )
            unit_match = re.search(rb'"score_unit"\s*:\s*"([^"]+)"', prefix)
            for match in ERRORS_PATTERN.finditer(mapped):
                values = [parse_float_token(match.group(i)) for i in (1, 2, 3)]
                for index, value in enumerate(values):
                    per_identity[index].append(value)
                frame_count += 1
                all_three += int(all(np.isfinite(values)))
    arrays = [np.asarray(values, dtype=float) for values in per_identity]
    pooled = np.concatenate(arrays) if arrays else np.empty(0)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "frame_records": frame_count,
        "all_three_finite_frame_count": all_three,
        "all_three_finite_frame_share": (
            all_three / frame_count if frame_count else None
        ),
        "residual_score_unit": (
            unit_match.group(1).decode() if unit_match is not None else None
        ),
        "triangulation_backend": (
            backend_match.group(1).decode()
            if backend_match is not None
            else None
        ),
        "residual_pooled": distribution(pooled),
        "residual_by_prelink_label": {
            label: distribution(values)
            for label, values in zip(("a", "b", "c"), arrays, strict=True)
        },
    }


def load_method(path: Path) -> tuple[pd.DataFrame, str]:
    header = list(pd.read_csv(path, nrows=0).columns)
    id_column = next(
        (
            column
            for column in (
                "fish_id",
                "final_fish_id",
                "geometry_id",
                "path_id",
            )
            if column in header
        ),
        None,
    )
    if id_column is None:
        raise ValueError(f"{path} has no identity column")
    columns = ["frame", id_column, "X", "Y", "Z"]
    for optional in (
        "is_interpolated",
        "ray_residual_mm2",
        "identity_resolved",
        "is_unresolved",
        "assignment_status",
        "identity_status",
        "identity_segment",
        "path_segment",
    ):
        if optional in header:
            columns.append(optional)
    table = pd.read_csv(path, usecols=columns, low_memory=False)
    table["frame"] = pd.to_numeric(table.frame, errors="raise").astype(np.int64)
    table["base_method_id"] = table[id_column].astype(str)
    segment_column = next(
        (
            column
            for column in ("identity_segment", "path_segment")
            if column in table
        ),
        None,
    )
    if segment_column is None:
        table["method_id"] = table.base_method_id
    else:
        numeric_segment = pd.to_numeric(
            table[segment_column], errors="coerce"
        )
        segment_token = table[segment_column].astype("string").fillna("missing")
        integer_segment = numeric_segment.notna() & np.isclose(
            numeric_segment, np.round(numeric_segment)
        )
        segment_token.loc[integer_segment] = (
            np.round(numeric_segment.loc[integer_segment])
            .astype(np.int64)
            .astype(str)
        )
        table["method_id"] = (
            table.base_method_id
            + "#"
            + segment_column
            + "="
            + segment_token.astype(str)
        )
    if "is_interpolated" not in table:
        table["is_interpolated"] = False
    else:
        table["is_interpolated"] = table.is_interpolated.fillna(False).astype(bool)
    return table, id_column


def method_track_metrics(table: pd.DataFrame) -> dict[str, Any]:
    xyz = table[["X", "Y", "Z"]].to_numpy(dtype=float)
    finite = np.isfinite(xyz).all(axis=1)
    interpolated = table.is_interpolated.to_numpy(dtype=bool)
    direct = finite & ~interpolated
    if "ray_residual_mm2" in table:
        ray_residual_mm2 = pd.to_numeric(
            table.ray_residual_mm2, errors="coerce"
        ).to_numpy(dtype=float)
        direct_ray_residual_mm2 = ray_residual_mm2[
            direct & np.isfinite(ray_residual_mm2)
        ]
    else:
        direct_ray_residual_mm2 = np.empty(0, dtype=float)
    frame_min = int(table.frame.min())
    frame_max = int(table.frame.max())
    frame_index = np.arange(frame_min, frame_max + 1)
    finite_counts = (
        pd.Series(finite.astype(np.int8), index=table.frame)
        .groupby(level=0)
        .sum()
        .reindex(frame_index, fill_value=0)
    )
    direct_counts = (
        pd.Series(direct.astype(np.int8), index=table.frame)
        .groupby(level=0)
        .sum()
        .reindex(frame_index, fill_value=0)
    )
    direct_jumps: list[np.ndarray] = []
    finite_including_filled_jumps: list[np.ndarray] = []
    segment_boundary_distances: list[float] = []
    segment_boundary_frame_gaps: list[int] = []
    jump_group_columns = ["method_id"]
    if "identity_segment" in table:
        table = table.copy()
        table["_jump_identity_segment"] = (
            table.identity_segment.astype("string").fillna("__missing_segment__")
        )
        jump_group_columns.append("_jump_identity_segment")
        for _identity, identity_rows in table.groupby("method_id", sort=True):
            identity_rows = identity_rows.sort_values("frame", kind="stable")
            segment_order = (
                identity_rows.groupby("_jump_identity_segment", sort=False)
                .frame.min()
                .sort_values(kind="stable")
                .index.tolist()
            )
            for previous_segment, current_segment in zip(
                segment_order[:-1], segment_order[1:], strict=True
            ):
                previous = identity_rows[
                    identity_rows._jump_identity_segment.eq(previous_segment)
                ].sort_values("frame", kind="stable")
                current = identity_rows[
                    identity_rows._jump_identity_segment.eq(current_segment)
                ].sort_values("frame", kind="stable")
                previous_xyz = previous[["X", "Y", "Z"]].to_numpy(dtype=float)
                current_xyz = current[["X", "Y", "Z"]].to_numpy(dtype=float)
                previous_direct = (
                    np.isfinite(previous_xyz).all(axis=1)
                    & ~previous.is_interpolated.to_numpy(dtype=bool)
                )
                current_direct = (
                    np.isfinite(current_xyz).all(axis=1)
                    & ~current.is_interpolated.to_numpy(dtype=bool)
                )
                if not previous_direct.any() or not current_direct.any():
                    continue
                previous_index = int(np.flatnonzero(previous_direct)[-1])
                current_index = int(np.flatnonzero(current_direct)[0])
                previous_point = previous_xyz[previous_index]
                current_point = current_xyz[current_index]
                previous_frame = int(previous.iloc[previous_index].frame)
                current_frame = int(current.iloc[current_index].frame)
                segment_boundary_distances.append(
                    float(np.linalg.norm(current_point - previous_point))
                )
                segment_boundary_frame_gaps.append(
                    int(current_frame - previous_frame)
                )
    for _identity_segment, group in table.groupby(
        jump_group_columns, sort=True, dropna=False
    ):
        group = group.sort_values("frame", kind="stable")
        frames = group.frame.to_numpy(dtype=np.int64)
        points = group[["X", "Y", "Z"]].to_numpy(dtype=float)
        if len(group) < 2:
            continue
        valid = np.diff(frames) == 1
        valid &= np.isfinite(points[:-1]).all(axis=1)
        valid &= np.isfinite(points[1:]).all(axis=1)
        all_values = np.linalg.norm(np.diff(points, axis=0), axis=1)
        finite_including_filled_jumps.append(all_values[valid])
        direct_rows = (
            np.isfinite(points).all(axis=1)
            & ~group.is_interpolated.to_numpy(dtype=bool)
        )
        direct_valid = valid & direct_rows[:-1] & direct_rows[1:]
        direct_jumps.append(all_values[direct_valid])
    jump_values = (
        np.concatenate(direct_jumps) if direct_jumps else np.empty(0)
    )
    filled_jump_values = (
        np.concatenate(finite_including_filled_jumps)
        if finite_including_filled_jumps
        else np.empty(0)
    )

    def spike_rates(values: np.ndarray) -> dict[str, dict[str, float | int]]:
        return {
            f"ge_{str(threshold).replace('.', '_')}mm": {
                "count": int((values >= threshold).sum()),
                "share": float(
                    (values >= threshold).mean() if len(values) else 0.0
                ),
            }
            for threshold in (2.0, 4.0, 6.0, 7.5)
        }

    def unresolved_status_mask(series: pd.Series) -> np.ndarray:
        normalized = series.astype("string").fillna("").str.strip().str.lower()
        return normalized.str.match(
            r"^(?:unresolved|unassigned|deferred|ambiguous)(?:$|[_:\-\s])"
        ).to_numpy()

    explicit_unresolved = np.zeros(len(table), dtype=bool)
    if "identity_resolved" in table:
        explicit_unresolved |= (
            ~table.identity_resolved.fillna(False).astype(bool).to_numpy()
        )
    if "is_unresolved" in table:
        explicit_unresolved |= (
            table.is_unresolved.fillna(False).astype(bool).to_numpy()
        )
    if "assignment_status" in table:
        explicit_unresolved |= unresolved_status_mask(table.assignment_status)
    if "identity_status" in table:
        explicit_unresolved |= unresolved_status_mask(table.identity_status)
    unavailable_or_unresolved = ~finite | explicit_unresolved
    return {
        "frame_min": frame_min,
        "frame_max": frame_max,
        "frame_count": int(len(frame_index)),
        "row_count": int(len(table)),
        "identity_count": int(table.method_id.nunique()),
        "identities": sorted(table.method_id.unique().tolist()),
        "base_identity_count": int(table.base_method_id.nunique()),
        "base_identities": sorted(table.base_method_id.unique().tolist()),
        "finite_row_share": float(finite.mean()),
        "direct_noninterpolated_row_share": float(direct.mean()),
        "direct_ray_residual_mm2": distribution(direct_ray_residual_mm2),
        "all_three_finite_frame_count": int((finite_counts == 3).sum()),
        "all_three_finite_frame_share": float((finite_counts == 3).mean()),
        "all_three_direct_noninterpolated_frame_count": int(
            (direct_counts == 3).sum()
        ),
        "all_three_direct_noninterpolated_frame_share": float(
            (direct_counts == 3).mean()
        ),
        "nonfinite_row_count": int((~finite).sum()),
        "nonfinite_row_share": float((~finite).mean()),
        "explicit_identity_unresolved_row_count": int(explicit_unresolved.sum()),
        "explicit_identity_unresolved_row_share": float(
            explicit_unresolved.mean()
        ),
        "unavailable_or_unresolved_row_count": int(
            unavailable_or_unresolved.sum()
        ),
        "unavailable_or_unresolved_row_share": float(
            unavailable_or_unresolved.mean()
        ),
        "consecutive_jump_mm": distribution(jump_values),
        "consecutive_direct_jump_spike_rates": spike_rates(jump_values),
        "consecutive_finite_including_filled_jump_mm": distribution(
            filled_jump_values
        ),
        "segment_boundary_direct_distance_mm": distribution(
            np.asarray(segment_boundary_distances, dtype=float)
        ),
        "segment_boundary_direct_frame_gap": distribution(
            np.asarray(segment_boundary_frame_gaps, dtype=float)
        ),
        "segment_boundary_direct_pair_count": int(
            len(segment_boundary_distances)
        ),
        "consecutive_jump_ge_7_5mm_count": int((jump_values >= 7.5).sum()),
        "consecutive_jump_gt_8mm_count": int((jump_values > 8.0).sum()),
        "consecutive_jump_gt_8mm_share": float(
            (jump_values > 8.0).mean() if len(jump_values) else 0.0
        ),
    }


def selected_point(row: pd.Series, prefix: str, anchor: str) -> np.ndarray:
    mouth = np.asarray(
        [row[f"{prefix}_mouth_x"], row[f"{prefix}_mouth_y"]], dtype=float
    )
    if anchor == "mouth":
        return mouth
    tail = np.asarray(
        [row[f"{prefix}_tail_x"], row[f"{prefix}_tail_y"]], dtype=float
    )
    finite = [point for point in (mouth, tail) if np.isfinite(point).all()]
    return np.mean(finite, axis=0) if finite else np.full(2, np.nan)


def female_assignments(
    annotations: pd.DataFrame,
    table: pd.DataFrame,
    *,
    method: str,
    anchor: str,
    calibration: Calibration,
    gate_px: float = 80.0,
    strict_gate_px: float = 8.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = annotations[
        annotations.gt_status.eq("confirmed") & annotations.sex_label.eq("F")
    ].copy()
    needed_frames = set(int(value) for value in eligible.frame)
    by_frame = {
        int(frame): group
        for frame, group in table[table.frame.isin(needed_frames)].groupby(
            "frame", sort=False
        )
    }
    rows: list[dict[str, Any]] = []
    for annotation in eligible.itertuples(index=False):
        annotation_row = pd.Series(annotation._asdict())
        frame = int(annotation.frame)
        group = by_frame.get(frame)
        record: dict[str, Any] = {
            "method": method,
            "anchor": anchor,
            "annotation_id": str(annotation.annotation_id),
            "frame": frame,
        }
        choices: dict[str, str] = {}
        distances: dict[str, float] = {}
        if group is not None:
            xyz = group[["X", "Y", "Z"]].to_numpy(dtype=float)
            valid = np.isfinite(xyz).all(axis=1)
            projections = np.full((len(group), 3, 2), np.nan, dtype=float)
            if valid.any():
                projections[valid] = np.asarray(
                    calibration.project_world(xyz[valid]), dtype=float
                )
            identities = group.method_id.astype(str).to_numpy()
            for prefix in ("side1", "side2"):
                target = selected_point(annotation_row, prefix, anchor)
                costs = np.linalg.norm(
                    projections[:, CAMERA_INDEX[prefix], :] - target[None, :],
                    axis=1,
                )
                distance = (
                    float(np.nanmin(costs))
                    if np.isfinite(costs).any()
                    else math.nan
                )
                distances[prefix] = distance
                if np.isfinite(costs).any():
                    index = int(np.nanargmin(costs))
                    if distance <= gate_px:
                        choices[prefix] = identities[index]
                record[f"{prefix}_distance_px"] = distance
                record[f"{prefix}_method_id"] = choices.get(prefix, "")
        available = [
            choices[prefix] for prefix in ("side1", "side2") if prefix in choices
        ]
        if len(available) == 2 and available[0] == available[1]:
            status, consensus = "agree", available[0]
        elif len(available) == 2:
            status, consensus = "disagree", ""
        elif len(available) == 1:
            status, consensus = "one_view", available[0]
        else:
            status, consensus = "unmapped", ""
        record["mapping_status"] = status
        record["consensus_method_id"] = consensus
        record["strict_two_view"] = bool(
            status == "agree"
            and distances.get("side1", math.inf) <= strict_gate_px
            and distances.get("side2", math.inf) <= strict_gate_px
        )
        rows.append(record)
    assignments = pd.DataFrame(rows).sort_values("frame", kind="stable")

    def summarize(group: pd.DataFrame) -> dict[str, Any]:
        mapped = group[group.consensus_method_id.ne("")].copy()
        identities = mapped.consensus_method_id.astype(str).to_numpy()
        counts = Counter(identities)
        best_label, best_count = counts.most_common(1)[0] if counts else ("", 0)
        return {
            "count": int(len(group)),
            "mapped_count": int(len(mapped)),
            "identity_counts": dict(sorted(counts.items())),
            "best_fixed_identity": best_label,
            "selective_mapped_best_fixed_accuracy": (
                float(best_count / len(mapped)) if len(mapped) else 0.0
            ),
            "full_eligible_best_fixed_accuracy": (
                float(best_count / len(group)) if len(group) else 0.0
            ),
            "observed_identity_changes": int(
                np.sum(identities[1:] != identities[:-1])
                if len(identities) > 1
                else 0
            ),
        }

    summary = {
        "gate_px": gate_px,
        "strict_gate_px": strict_gate_px,
        "mapped": summarize(assignments),
        "strict_two_view": summarize(assignments[assignments.strict_two_view]),
        "critical_event_bracket": assignments[
            assignments.frame.isin((34146, 34736))
        ].to_dict(orient="records"),
        "limitation": (
            "The ledger labels only the female. A male-to-male exchange cannot "
            "change any female assignment and is therefore unobservable."
        ),
    }
    return assignments, summary


def normalized_scientific_hash(table: pd.DataFrame) -> str:
    columns = ["frame", "method_id", "X", "Y", "Z", "is_interpolated"]
    normalized = table[columns].sort_values(
        ["frame", "method_id"], kind="stable"
    )
    digest = hashlib.sha256()
    for row in normalized.itertuples(index=False, name=None):
        line = (
            f"{int(row[0])},{row[1]},"
            f"{float(row[2]).hex()},{float(row[3]).hex()},{float(row[4]).hex()},"
            f"{int(bool(row[5]))}\n"
        )
        digest.update(line.encode("ascii"))
    return digest.hexdigest()


def parse_method_spec(values: list[str]) -> tuple[str, str, Path, Path | None, Path | None]:
    name, anchor, csv_path, prelink_path, linked_path = values
    if anchor not in {"mouth", "centroid"}:
        raise ValueError(f"invalid anchor {anchor}")
    return (
        name,
        anchor,
        Path(csv_path),
        None if prelink_path == "-" else Path(prelink_path),
        None if linked_path == "-" else Path(linked_path),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        nargs=5,
        action="append",
        required=True,
        metavar=("NAME", "ANCHOR", "CSV", "PRELINK_OR_DASH", "LINKED_OR_DASH"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations = pd.read_csv(args.annotations, low_memory=False)
    numeric = [
        "frame",
        "side1_mouth_x",
        "side1_mouth_y",
        "side1_tail_x",
        "side1_tail_y",
        "side2_mouth_x",
        "side2_mouth_y",
        "side2_tail_x",
        "side2_tail_y",
    ]
    for column in numeric:
        annotations[column] = pd.to_numeric(annotations[column], errors="coerce")
    calibration = Calibration(args.calibration_dir, backend="refractive")
    method_results: dict[str, Any] = {}
    assignment_tables: list[pd.DataFrame] = []
    event_tables: list[pd.DataFrame] = []
    for raw_spec in args.method:
        name, anchor, csv_path, prelink_path, linked_path = parse_method_spec(
            raw_spec
        )
        table, id_column = load_method(csv_path)
        assignments, female = female_assignments(
            annotations,
            table,
            method=name,
            anchor=anchor,
            calibration=calibration,
        )
        assignment_tables.append(assignments)
        event = table[
            table.frame.between(34540, 34565, inclusive="both")
        ].copy()
        event.insert(0, "method", name)
        event_tables.append(event)
        method_results[name] = {
            "anchor": anchor,
            "csv": {
                "path": str(csv_path),
                "size_bytes": csv_path.stat().st_size,
                "sha256": sha256(csv_path),
                "id_column": id_column,
                "normalized_scientific_sha256": normalized_scientific_hash(table),
            },
            "track_qc": method_track_metrics(table),
            "female_gt": female,
            "pre_link_qc": (
                prelink_metrics(prelink_path) if prelink_path is not None else None
            ),
            "linked_json": (
                {
                    "path": str(linked_path),
                    "size_bytes": linked_path.stat().st_size,
                    "sha256": sha256(linked_path),
                }
                if linked_path is not None
                else None
            ),
        }

    (args.output_dir / "independent_method_metrics.json").write_text(
        json.dumps(method_results, indent=2, allow_nan=True), encoding="utf-8"
    )
    pd.concat(assignment_tables, ignore_index=True).to_csv(
        args.output_dir / "independent_female_assignments.csv", index=False
    )
    pd.concat(event_tables, ignore_index=True).to_csv(
        args.output_dir / "critical_event_method_rows.csv", index=False
    )
    summary_rows = []
    for name, result in method_results.items():
        track = result["track_qc"]
        female = result["female_gt"]["mapped"]
        prelink_residual = result["pre_link_qc"]
        direct_residual = track["direct_ray_residual_mm2"]
        use_direct_residual = direct_residual["count"] > 0
        summary_rows.append(
            {
                "method": name,
                "anchor": result["anchor"],
                "female_mapped": female["mapped_count"],
                "female_full_eligible_accuracy": female[
                    "full_eligible_best_fixed_accuracy"
                ],
                "female_selective_mapped_accuracy": female[
                    "selective_mapped_best_fixed_accuracy"
                ],
                "female_observed_identity_changes": female[
                    "observed_identity_changes"
                ],
                "all_three_direct_share": track[
                    "all_three_direct_noninterpolated_frame_share"
                ],
                "explicit_identity_unresolved_share": track[
                    "explicit_identity_unresolved_row_share"
                ],
                "nonfinite_row_share": track["nonfinite_row_share"],
                "jump_p95_mm": track["consecutive_jump_mm"]["p95"],
                "jump_p99_mm": track["consecutive_jump_mm"]["p99"],
                "jump_ge_7_5mm": track["consecutive_jump_ge_7_5mm_count"],
                "jump_gt_8mm": track["consecutive_jump_gt_8mm_count"],
                "residual_evidence": (
                    "authenticated_strict_mouth_direct_csv"
                    if use_direct_residual
                    else (
                        "centroid_pre_link_json"
                        if prelink_residual is not None
                        else None
                    )
                ),
                "residual_unit": (
                    "mm^2"
                    if use_direct_residual
                    else (
                        prelink_residual["residual_score_unit"]
                        if prelink_residual
                        else None
                    )
                ),
                "residual_p95": (
                    direct_residual["p95"]
                    if use_direct_residual
                    else (
                        prelink_residual["residual_pooled"]["p95"]
                        if prelink_residual
                        else None
                    )
                ),
                "residual_p99": (
                    direct_residual["p99"]
                    if use_direct_residual
                    else (
                        prelink_residual["residual_pooled"]["p99"]
                        if prelink_residual
                        else None
                    )
                ),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "independent_method_summary.csv", index=False
    )
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
