from __future__ import annotations

"""Run a geometry-preserving structural benchmark of global fragment stitching.

The benchmark consumes existing tracklet-first artifacts.  It never rewrites
3D coordinates and deliberately supplies no biological labels unless an
independently validated anchor table is added in a future experiment.  Its
component reduction is therefore a structural diagnostic, not ID accuracy.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.tracking.appearance_prototypes import (
    PooledEmbeddings,
    attach_fragment_appearance,
)
from scripts.tracking.global_fragment_stitching import (
    EdgeFeatures,
    Fragment,
    calculate_edge_cost,
    solve_constrained_path_cover,
)


def _load_best(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames", payload)
    return {int(frame): info for frame, info in frames.items()}


def _component_observations(
    assignments: pd.DataFrame,
    dispositions: pd.DataFrame,
    best: dict[int, dict],
) -> tuple[pd.DataFrame, dict[str, int]]:
    node_to_component = assignments.set_index("node_id")["component_id"].astype(str).to_dict()
    rows: list[dict[str, object]] = []
    for row in dispositions.to_dict("records"):
        node = str(row.get("tracklet_node", ""))
        component = node_to_component.get(node)
        if component is None:
            continue
        try:
            frame = int(float(row["frame"]))
            source_idx = int(float(row["selected_source_idx"]))
        except (KeyError, TypeError, ValueError):
            continue
        if source_idx < 0 or source_idx > 2 or frame not in best:
            continue
        label = chr(ord("a") + source_idx)
        point = np.asarray(best[frame].get("groups", {}).get(label, []), dtype=float)
        residual = best[frame].get("errors", {}).get(label)
        if point.shape != (3,) or not np.isfinite(point).all():
            continue
        rows.append(
            {
                "component_id": component,
                "frame": frame,
                "source_idx": source_idx,
                "X": float(point[0]),
                "Y": float(point[1]),
                "Z": float(point[2]),
                "residual_px2": float(residual) if residual is not None else np.nan,
            }
        )
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw, {"raw_rows": 0, "same_frame_source_conflicts": 0}
    cardinality = raw.groupby(["component_id", "frame"])["source_idx"].nunique()
    conflicts = set(cardinality[cardinality.gt(1)].index)
    raw["same_frame_source_conflict"] = [
        (component, frame) in conflicts
        for component, frame in zip(raw.component_id, raw.frame)
    ]
    clean = raw[~raw.same_frame_source_conflict].copy()
    clean = (
        clean.groupby(["component_id", "frame", "source_idx"], as_index=False)
        .agg(X=("X", "median"), Y=("Y", "median"), Z=("Z", "median"), residual_px2=("residual_px2", "median"))
        .sort_values(["component_id", "frame"])
    )
    return clean, {
        "raw_rows": int(len(raw)),
        "clean_component_frames": int(len(clean)),
        "same_frame_source_conflicts": int(len(conflicts)),
    }


def _fragment_table(assignments: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    fragments = (
        assignments.groupby("component_id", as_index=False)
        .agg(
            frame_start=("frame_start", "min"),
            frame_end=("frame_end", "max"),
            node_count=("node_id", "nunique"),
            view_count=("view", "nunique"),
        )
    )
    if observations.empty:
        fragments["geometry_frames"] = 0
        return fragments
    geometry = observations.groupby("component_id", as_index=False).agg(
        geometry_frame_start=("frame", "min"),
        geometry_frame_end=("frame", "max"),
        geometry_frames=("frame", "nunique"),
        median_residual_px2=("residual_px2", "median"),
    )
    return fragments.merge(geometry, on="component_id", how="left").fillna(
        {"geometry_frames": 0}
    )


def _velocity(group: pd.DataFrame, *, at_end: bool) -> np.ndarray | None:
    group = group.sort_values("frame").drop_duplicates("frame")
    if len(group) < 2:
        return None
    pair = group.iloc[-2:] if at_end else group.iloc[:2]
    dt = int(pair.iloc[1].frame) - int(pair.iloc[0].frame)
    if dt <= 0:
        return None
    return (
        pair.iloc[1][["X", "Y", "Z"]].to_numpy(float)
        - pair.iloc[0][["X", "Y", "Z"]].to_numpy(float)
    ) / dt


def _angle(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator < 1e-12:
        return None
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _silhouette_support(
    cross_view_edges: pd.DataFrame | None,
    node_to_component: dict[str, str],
) -> dict[tuple[str, str], float]:
    support: dict[tuple[str, str], float] = {}
    if cross_view_edges is None or cross_view_edges.empty:
        return support
    table = cross_view_edges[cross_view_edges.get("edge_source", "").eq("silhouette_continuity")]
    for row in table.to_dict("records"):
        left = node_to_component.get(str(row.get("node_a", "")))
        right = node_to_component.get(str(row.get("node_b", "")))
        if left is None or right is None or left == right:
            continue
        pair = tuple(sorted((left, right)))
        value = row.get("support_weight", 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        support[pair] = max(support.get(pair, 0.0), value)
    return support


def build_edges(
    fragments: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    silhouette_support: dict[tuple[str, str], float],
    max_gap_frames: int,
    fragment_embeddings: PooledEmbeddings | None = None,
    appearance_calibration: dict[str, float] | None = None,
) -> list:
    by_component = {
        str(component): group.sort_values("frame")
        for component, group in observations.groupby("component_id")
    }
    records = fragments.sort_values(["frame_start", "frame_end", "component_id"]).to_dict("records")
    features: list[EdgeFeatures] = []
    for source in records:
        source_id = str(source["component_id"])
        source_obs = by_component.get(source_id)
        if source_obs is None or source_obs.empty:
            continue
        source_last = source_obs.iloc[-1]
        source_velocity = _velocity(source_obs, at_end=True)
        for target in records:
            target_id = str(target["component_id"])
            if int(target["frame_start"]) <= int(source["frame_end"]):
                continue
            gap = int(target["frame_start"]) - int(source["frame_end"]) - 1
            if gap > max_gap_frames:
                continue
            target_obs = by_component.get(target_id)
            if target_obs is None or target_obs.empty:
                continue
            target_first = target_obs.iloc[0]
            if int(target_first.frame) <= int(source_last.frame):
                continue
            bridge_dt = int(target_first.frame) - int(source_last.frame)
            left_xyz = source_last[["X", "Y", "Z"]].to_numpy(float)
            right_xyz = target_first[["X", "Y", "Z"]].to_numpy(float)
            bridge = (right_xyz - left_xyz) / max(1, bridge_dt)
            target_velocity = _velocity(target_obs, at_end=False)
            angles = [value for value in (_angle(source_velocity, bridge), _angle(bridge, target_velocity)) if value is not None]
            predicted_error = None
            if source_velocity is not None:
                predicted_error = float(
                    np.linalg.norm(right_xyz - (left_xyz + source_velocity * bridge_dt))
                )
            residuals = np.asarray(
                [source_last.residual_px2, target_first.residual_px2], dtype=float
            )
            residuals = residuals[np.isfinite(residuals) & (residuals >= 0)]
            reprojection = float(np.sqrt(residuals).mean()) if len(residuals) else None
            support = silhouette_support.get(tuple(sorted((source_id, target_id))))
            feature = EdgeFeatures(
                source_id=source_id,
                target_id=target_id,
                gap_frames=gap,
                endpoint_displacement_mm=float(np.linalg.norm(right_xyz - left_xyz)),
                motion_prediction_error_mm=predicted_error,
                velocity_angle_deg=max(angles) if angles else None,
                reprojection_error_px=reprojection,
                silhouette_iou=support,
                mask_overlap=support,
                silhouette_state="support" if support is not None else "unknown",
                metadata={"silhouette_support_proxy": support},
            )
            features.append(feature)
    if fragment_embeddings is not None:
        calibration = appearance_calibration or {}
        features = list(
            attach_fragment_appearance(
                features,
                fragment_embeddings,
                calibrator_coefficient=calibration.get("coefficient"),
                calibrator_intercept=calibration.get("intercept"),
                same_probability_min=calibration.get("same_probability_min"),
                different_probability_max=calibration.get("different_probability_max"),
            )
        )
    return [calculate_edge_cost(feature) for feature in features]


def _load_fragment_embeddings(path: Path | None) -> PooledEmbeddings | None:
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as payload:
        if "embeddings" not in payload:
            raise ValueError("fragment embedding NPZ needs an 'embeddings' array")
        key_name = next((name for name in ("fragment_ids", "component_ids", "keys") if name in payload), None)
        if key_name is None:
            raise ValueError(
                "fragment embedding NPZ needs 'fragment_ids', 'component_ids', or 'keys'"
            )
        keys = tuple(str(value) for value in payload[key_name].tolist())
        counts = (
            tuple(int(value) for value in payload["sample_counts"].tolist())
            if "sample_counts" in payload
            else tuple(1 for _ in keys)
        )
        return PooledEmbeddings(keys, payload["embeddings"], counts)


def _load_appearance_calibration(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    frozen = report.get("thresholds_frozen_on_calibration", {})
    calibrator = frozen.get("calibrator", {})
    same_min = float(frozen.get("target_far_threshold_probability"))
    different_max = float(1.0 - same_min)
    values = {
        "coefficient": float(calibrator.get("coefficient")),
        "intercept": float(calibrator.get("intercept")),
        "same_probability_min": same_min,
        "different_probability_max": different_max,
    }
    if values["coefficient"] <= 0 or not 0 <= different_max < same_min <= 1:
        raise ValueError("appearance evaluation report has invalid calibration-frozen thresholds")
    return values


def _load_trusted_anchors(path: Path | None) -> dict[str, str]:
    """Load only explicitly reviewed biological anchors.

    The deliberately narrow schema prevents a reconstruction-local geometry ID
    column from being mistaken for biological ground truth.
    """
    if path is None:
        return {}
    table = pd.read_csv(path)
    required = {"component_id", "trusted_identity", "anchor_status"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            "trusted anchor CSV missing columns: " + ", ".join(sorted(missing))
        )
    table = table[table.anchor_status.astype(str).str.lower().eq("trusted")].copy()
    if table.empty:
        return {}
    if table[["component_id", "trusted_identity"]].isna().any().any():
        raise ValueError("trusted anchors must have non-null component_id and trusted_identity")
    table["component_id"] = table.component_id.astype(str).str.strip()
    table["trusted_identity"] = table.trusted_identity.astype(str).str.strip()
    if table.component_id.eq("").any() or table.trusted_identity.eq("").any():
        raise ValueError("trusted anchors must have non-empty component_id and trusted_identity")
    conflicts = table.groupby("component_id").trusted_identity.nunique()
    if conflicts.gt(1).any():
        ids = ", ".join(conflicts[conflicts.gt(1)].index.astype(str))
        raise ValueError(f"trusted anchor components have conflicting labels: {ids}")
    return (
        table.drop_duplicates("component_id")
        .set_index("component_id")["trusted_identity"]
        .to_dict()
    )


def _edge_row(edge, selected: bool) -> dict[str, object]:
    row = {
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "feasible": edge.feasible,
        "selected": selected,
        "total_cost": edge.total_cost if math.isfinite(edge.total_cost) else np.nan,
        "reason": edge.reason,
    }
    row.update(edge.terms)
    for name in (
        "gap_frames",
        "endpoint_displacement_mm",
        "motion_prediction_error_mm",
        "velocity_angle_deg",
        "reprojection_error_px",
        "silhouette_iou",
        "mask_overlap",
        "appearance_cosine_distance",
        "appearance_state",
        "silhouette_state",
    ):
        row[name] = getattr(edge.features, name)
    row["appearance_same_probability"] = edge.features.metadata.get(
        "appearance_same_probability"
    )
    return row


def _speed_spike_count(table: pd.DataFrame, threshold_mm: float) -> int:
    identity_col = next(
        (column for column in ("final_fish_id", "fish_id", "global_id") if column in table.columns),
        None,
    )
    groups = table.groupby(identity_col, dropna=False) if identity_col else [("all", table)]
    spikes = 0
    for _identity, group in groups:
        ordered = group.sort_values("frame")
        xyz = ordered[["X", "Y", "Z"]].to_numpy(float)
        valid = np.isfinite(xyz).all(axis=1)
        delta = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        adjacent_valid = valid[:-1] & valid[1:]
        spikes += int(np.sum(adjacent_valid & (delta > float(threshold_mm))))
    return spikes


def _assert_exact_xyz_equal(before: pd.DataFrame, after: pd.DataFrame) -> None:
    required = ["X", "Y", "Z"]
    if any(column not in before.columns or column not in after.columns for column in required):
        raise ValueError("XYZ equality check requires X, Y, and Z in both tables")
    if len(before) != len(after) or not np.array_equal(
        before[required].to_numpy(float),
        after[required].to_numpy(float),
        equal_nan=True,
    ):
        raise RuntimeError("XYZ safety failure: reconstructed coordinates changed")


def _write_and_verify_xyz_copy(
    source: Path | None,
    output: Path,
    *,
    speed_spike_threshold_mm: float,
) -> dict[str, object]:
    if source is None:
        return {
            "available": False,
            "xyz_exact_equal": None,
            "geometry_modified": None,
            "speed_spike_count_input": None,
            "speed_spike_count_output": None,
            "speed_spike_change": None,
            "reason": "--canonical-3d was not supplied",
        }
    table = pd.read_csv(source, float_precision="round_trip")
    required = {"frame", "X", "Y", "Z"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError("canonical 3D table missing columns: " + ", ".join(sorted(missing)))
    table.to_csv(output, index=False, float_format="%.17g")
    written = pd.read_csv(output, float_precision="round_trip")
    _assert_exact_xyz_equal(table, written)
    before_spikes = _speed_spike_count(table, speed_spike_threshold_mm)
    after_spikes = _speed_spike_count(written, speed_spike_threshold_mm)
    if after_spikes != before_spikes:
        raise RuntimeError("speed-spike safety failure: copied output changed the spike count")
    return {
        "available": True,
        "xyz_exact_equal": True,
        "geometry_modified": False,
        "speed_spike_threshold_mm": float(speed_spike_threshold_mm),
        "speed_spike_count_input": int(before_spikes),
        "speed_spike_count_output": int(after_spikes),
        "speed_spike_change": int(after_spikes - before_spikes),
        "output": str(output),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    assignments = pd.read_csv(args.assignments)
    dispositions = pd.read_csv(args.dispositions)
    best = _load_best(args.best_pre_link)
    cross_edges = pd.read_csv(args.cross_view_edges) if args.cross_view_edges else None
    observations, observation_qc = _component_observations(assignments, dispositions, best)
    fragment_table = _fragment_table(assignments, observations)
    fragment_embeddings = _load_fragment_embeddings(args.fragment_embeddings)
    appearance_calibration = _load_appearance_calibration(args.appearance_evaluation_report)
    if fragment_embeddings is not None and appearance_calibration is None:
        raise ValueError(
            "--fragment-embeddings requires --appearance-evaluation-report so ambiguous appearance can abstain"
        )
    trusted_anchors = _load_trusted_anchors(args.trusted_anchors)
    unknown_anchors = sorted(set(trusted_anchors) - set(fragment_table.component_id.astype(str)))
    if unknown_anchors:
        raise ValueError(
            "trusted anchors reference unknown components: " + ", ".join(unknown_anchors[:10])
        )
    node_to_component = assignments.set_index("node_id")["component_id"].astype(str).to_dict()
    support = _silhouette_support(cross_edges, node_to_component)
    edge_costs = build_edges(
        fragment_table,
        observations,
        silhouette_support=support,
        max_gap_frames=args.max_gap_frames,
        fragment_embeddings=fragment_embeddings,
        appearance_calibration=appearance_calibration,
    )
    fragments = [
        Fragment(
            str(row.component_id), int(row.frame_start), int(row.frame_end),
            trusted_identity=trusted_anchors.get(str(row.component_id)),
        )
        for row in fragment_table.itertuples(index=False)
    ]
    identity_labels = (
        tuple(sorted(set(trusted_anchors.values())))
        if trusted_anchors
        else ("ID_1", "ID_2", "ID_3")
    )
    result = solve_constrained_path_cover(
        fragments,
        edge_costs,
        identity_labels=identity_labels,
        break_cost=args.break_cost,
    )
    selected_pairs = {(edge.source_id, edge.target_id) for edge in result.selected_edges}
    edge_table = pd.DataFrame(
        [_edge_row(edge, (edge.source_id, edge.target_id) in selected_pairs) for edge in edge_costs]
    )
    fragment_table["global_component_id"] = fragment_table.component_id.astype(str).map(
        result.component_by_fragment
    )
    fragment_table["biological_identity"] = fragment_table.component_id.astype(str).map(
        result.fragment_identities
    ).fillna("")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fragment_table.to_csv(args.output_dir / "fragment_provenance.csv", index=False)
    observations.to_csv(args.output_dir / "component_3d_observations.csv", index=False)
    edge_table.to_csv(args.output_dir / "global_edge_costs.csv", index=False)
    edge_table[edge_table.selected].to_csv(args.output_dir / "selected_global_edges.csv", index=False)
    xyz_safety = _write_and_verify_xyz_copy(
        args.canonical_3d,
        args.output_dir / "canonical_3d_xyz_unchanged.csv",
        speed_spike_threshold_mm=args.speed_spike_threshold_mm,
    )
    reason_counts = Counter(edge.reason for edge in edge_costs if not edge.feasible)
    summary = {
        "benchmark_kind": "structural_no_biological_ground_truth",
        "geometry_modified": xyz_safety["geometry_modified"],
        "xyz_exact_equal": xyz_safety["xyz_exact_equal"],
        "input_components": int(fragment_table.component_id.nunique()),
        "output_path_components": int(fragment_table.global_component_id.nunique()),
        "component_reduction": int(
            fragment_table.component_id.nunique() - fragment_table.global_component_id.nunique()
        ),
        "candidate_edges": int(len(edge_costs)),
        "feasible_edges": int(sum(edge.feasible for edge in edge_costs)),
        "selected_edges": int(len(result.selected_edges)),
        "selected_silhouette_support_edges": int(
            sum(edge.features.silhouette_state == "support" for edge in result.selected_edges)
        ),
        "candidate_edges_with_appearance": int(
            sum(edge.features.appearance_cosine_distance is not None for edge in edge_costs)
        ),
        "selected_edges_with_appearance": int(
            sum(
                edge.features.appearance_cosine_distance is not None
                for edge in result.selected_edges
            )
        ),
        "candidate_appearance_states": dict(
            Counter(edge.features.appearance_state for edge in edge_costs)
        ),
        "selected_appearance_states": dict(
            Counter(edge.features.appearance_state for edge in result.selected_edges)
        ),
        "trusted_anchor_fragments": int(len(trusted_anchors)),
        "biological_identity_labels": list(identity_labels) if trusted_anchors else [],
        "biologically_labelled_fragments": int(len(result.fragment_identities)),
        "biological_id_accuracy": None,
        "identity_conflicts": None,
        "speed_spike_change": xyz_safety["speed_spike_change"],
        "xyz_safety": xyz_safety,
        "solver_status": result.solver_status,
        "objective": result.objective,
        "edge_rejection_reasons": dict(reason_counts),
        "observation_qc": observation_qc,
        "caveat": (
            "Component reduction is not identity accuracy. Only labels propagated from "
            "the explicitly supplied trusted-anchor table are biological identities."
            if trusted_anchors
            else "No independently validated biological anchors were available. "
            "Component reduction is not identity accuracy and must not be used as such."
        ),
        "inputs": {
            "assignments": str(args.assignments),
            "dispositions": str(args.dispositions),
            "best_pre_link": str(args.best_pre_link),
            "cross_view_edges": str(args.cross_view_edges) if args.cross_view_edges else None,
            "fragment_embeddings": (
                str(args.fragment_embeddings) if args.fragment_embeddings else None
            ),
            "appearance_evaluation_report": (
                str(args.appearance_evaluation_report)
                if args.appearance_evaluation_report
                else None
            ),
            "trusted_anchors": str(args.trusted_anchors) if args.trusted_anchors else None,
            "canonical_3d": str(args.canonical_3d) if args.canonical_3d else None,
        },
    }
    (args.output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--dispositions", type=Path, required=True)
    parser.add_argument("--best-pre-link", type=Path, required=True)
    parser.add_argument("--cross-view-edges", type=Path)
    parser.add_argument(
        "--fragment-embeddings",
        type=Path,
        help=(
            "NPZ with fragment_ids/component_ids/keys, embeddings, and optional "
            "sample_counts. Appearance remains label-free edge evidence."
        ),
    )
    parser.add_argument(
        "--appearance-evaluation-report",
        type=Path,
        help=(
            "Script-50 evaluation_report.json. Its calibration-frozen logistic model and "
            "target-FAR threshold define support/contradiction; intermediate scores abstain."
        ),
    )
    parser.add_argument(
        "--trusted-anchors",
        type=Path,
        help=(
            "CSV with component_id,trusted_identity,anchor_status. Only rows with "
            "anchor_status=trusted can name biological identities."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--canonical-3d",
        type=Path,
        help="Canonical reconstructed 3D CSV copied and round-trip checked for exact XYZ equality.",
    )
    parser.add_argument("--speed-spike-threshold-mm", type=float, default=8.0)
    parser.add_argument("--max-gap-frames", type=int, default=100)
    parser.add_argument("--break-cost", type=float, default=3.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
