from __future__ import annotations

"""Scene-adaptive identity linking from conservative tracklets in all views.

Concrete 2D tracklets are the identity primitives. Low-error 3D assignments
connect simultaneous tracklets across cameras; a tracklet that spans a scene
where another view breaks transfers identity across that break. Components are
never merged when their tracklets represent different fish in the same frame.
Optional appearance predictions label components that geometry cannot connect.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.reconstruction_3d.cross_frame_tracking import apply_cross_frame_tracking
from scripts.reconstruction_3d.fill_missing_3d import (
    best_by_frame_to_dfs,
    dfs_to_single_csv,
    ensure_all_frames_and_fill_spline,
)
from scripts.reconstruction_3d.load_2d_slp import (
    _instance_point,
    _instances_in_slot_order,
)
from scripts.reconstruction_3d.reconstruct_from_slp import (
    _save_best_tracked_json,
    load_best_pre_link_json,
)
from scripts.reconstruction_3d.tracklet_first_assignment import (
    propagate_identity_domains,
)

VIEWS = ("top_cam", "side1_cam", "side2_cam")
PERM_KEY = {"top_cam": "perm_cam0", "side1_cam": "perm_cam1", "side2_cam": "perm_cam2"}


def _key(value) -> str:
    try:
        x = float(value)
        if np.isfinite(x) and x.is_integer():
            return str(int(x))
    except (TypeError, ValueError):
        pass
    return str(value)


def _parse_view_paths(values: list[str], flag: str) -> dict[str, Path]:
    out = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{flag} expects view=path, got {value!r}")
        view, path = value.split("=", 1)
        if view not in VIEWS:
            raise ValueError(f"unknown view {view!r}")
        out[view] = Path(path)
    return out


def match_partial_tracks_to_slots(
    slot_xy: np.ndarray,
    track_xy: np.ndarray,
    track_nodes: list[str],
    *,
    match_distance_px: float,
) -> dict[int, str]:
    """Match any number of clean tracklet points to reconstruction slots."""
    slot_xy = np.asarray(slot_xy, float)
    track_xy = np.asarray(track_xy, float)
    if not len(slot_xy) or not len(track_xy):
        return {}
    valid_slots = np.where(np.isfinite(slot_xy).all(axis=1))[0]
    valid_tracks = np.where(np.isfinite(track_xy).all(axis=1))[0]
    if not len(valid_slots) or not len(valid_tracks):
        return {}
    cost = np.linalg.norm(
        track_xy[valid_tracks, None, :] - slot_xy[valid_slots][None, :, :], axis=2
    )
    rows, cols = linear_sum_assignment(cost)
    out = {}
    for r, c in zip(rows, cols):
        if float(cost[r, c]) <= match_distance_px:
            out[int(valid_slots[int(c)])] = track_nodes[int(valid_tracks[int(r)])]
    return out


def load_tracklet_slot_map(
    view: str,
    slp_path: Path,
    tracklet_csv: Path,
    *,
    start_frame: int,
    end_frame: int,
    match_distance_px: float = 5.0,
    anchor_node: str | None = None,
    require_anchor_node: bool = False,
) -> tuple[dict[int, dict[int, str]], dict[int, float], pd.DataFrame, dict]:
    """Map concrete tracklet rows onto the exact filtered candidate slots."""
    import sleap_io as sio

    table = pd.read_csv(tracklet_csv)
    required = {"frame", "tracklet_id", "x", "y"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{tracklet_csv} missing columns: {sorted(missing)}")
    table = table[(table.frame >= start_frame) & (table.frame <= end_frame)].copy()
    table["view"] = view
    table["tracklet_key"] = table["tracklet_id"].map(_key)
    table["node_id"] = view + ":" + table["tracklet_key"]
    by_frame = {int(f): g for f, g in table.groupby("frame")}

    labels = sio.load_slp(str(slp_path))
    node_names = [node.name for node in labels.skeletons[0].nodes]
    slot_map = {}
    separation_px = {}
    body_lengths = []
    matched_rows = 0
    for lf in labels.labeled_frames:
        frame = int(lf.frame_idx)
        if frame < start_frame or frame > end_frame:
            continue
        tf = by_frame.get(frame)
        if tf is None:
            continue
        ordered = _instances_in_slot_order(
            lf.instances,
            node_names=node_names,
            anchor_node=anchor_node,
            require_anchor_node=require_anchor_node,
        )
        slot_xy = np.full((len(ordered), 2), np.nan)
        for i, inst in enumerate(ordered):
            points = np.asarray(inst.numpy(), float)
            x, y, _score = _instance_point(
                inst,
                node_names=node_names,
                anchor_node=anchor_node,
                require_anchor_node=require_anchor_node,
            )
            if np.isfinite((x, y)).all():
                slot_xy[i] = (x, y)
            finite = np.isfinite(points).all(axis=1)
            if finite.sum() >= 2:
                body_lengths.append(float(np.max(
                    np.linalg.norm(points[finite, None, :] - points[None, finite, :], axis=2)
                )))
        tf = tf.drop_duplicates("node_id", keep="last")
        matched = match_partial_tracks_to_slots(
            slot_xy,
            tf[["x", "y"]].to_numpy(float),
            tf["node_id"].astype(str).tolist(),
            match_distance_px=match_distance_px,
        )
        if matched:
            slot_map[frame] = matched
            matched_rows += len(matched)
            matched_xy = slot_xy[list(matched)]
            if len(matched_xy) >= 2:
                distances = np.linalg.norm(
                    matched_xy[:, None, :] - matched_xy[None, :, :], axis=2
                )
                distances[np.eye(len(distances), dtype=bool)] = np.inf
                separation_px[frame] = float(distances.min())

    body_length = float(np.median(body_lengths)) if body_lengths else 1.0
    separation_bl = {
        frame: value / max(body_length, 1e-6)
        for frame, value in separation_px.items()
    }

    meta = (
        table.groupby(["node_id", "view", "tracklet_key"], as_index=False)
        .agg(frame_start=("frame", "min"), frame_end=("frame", "max"), n_rows=("frame", "size"))
    )
    qc = {
        "view": view,
        "tracklet_rows": int(len(table)),
        "matched_rows": int(matched_rows),
        "matched_fraction": float(matched_rows / max(len(table), 1)),
        "tracklets": int(table["node_id"].nunique()),
        "body_length_px": body_length,
        "anchor_node": anchor_node or "visible_node_centroid",
        "require_anchor_node": bool(require_anchor_node),
    }
    return slot_map, separation_bl, meta, qc


def load_silhouette_link_edges(
    view: str,
    path: str | Path,
    *,
    min_weight: float = 0.50,
    min_support_rows: int = 2,
    max_gap_frames: int = 3,
) -> pd.DataFrame:
    """Load trusted within-view mask-continuity edges for 3D validation."""
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}")
    table = pd.read_csv(path)
    required = {
        "left_tracklet_id",
        "right_tracklet_id",
        "gap_frames",
        "temporal_overlap_frames",
        "silhouette_link_status",
        "support_rows",
        "mean_silhouette_weight",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    table = table[
        table["silhouette_link_status"].eq("candidate")
        & pd.to_numeric(table["temporal_overlap_frames"], errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(table["gap_frames"], errors="coerce").between(0, max_gap_frames)
        & pd.to_numeric(table["support_rows"], errors="coerce").ge(min_support_rows)
        & pd.to_numeric(table["mean_silhouette_weight"], errors="coerce").ge(min_weight)
    ].copy()
    if table.empty:
        return pd.DataFrame(columns=[
            "node_a", "node_b", "support_rows", "support_weight",
            "trusted_mask_segment_id", "gap_frames", "edge_source",
        ])
    table["node_a"] = view + ":" + table["left_tracklet_id"].map(_key)
    table["node_b"] = view + ":" + table["right_tracklet_id"].map(_key)
    table["support_weight"] = pd.to_numeric(
        table["mean_silhouette_weight"], errors="coerce"
    ).fillna(0.0)
    table["edge_source"] = "silhouette_continuity"
    return table[[
        "node_a", "node_b", "support_rows", "support_weight",
        "trusted_mask_segment_id", "gap_frames", "edge_source",
    ]].reset_index(drop=True)


def build_observations(
    best: dict[int, dict],
    slot_maps: dict[str, dict[int, dict[int, str]]],
    quality_maps: dict[str, dict[int, float]],
    *,
    num_individuals: int,
    max_residual: float,
    hardcore_mm: float,
) -> pd.DataFrame:
    rows = []
    labels = [chr(ord("a") + i) for i in range(num_individuals)]
    for frame, info in best.items():
        points = [np.asarray(info.get("groups", {}).get(lb, []), float) for lb in labels]
        valid = [p.shape == (3,) and np.isfinite(p).all() for p in points]
        hard = False
        for i in range(num_individuals):
            for j in range(i + 1, num_individuals):
                if valid[i] and valid[j] and np.linalg.norm(points[i] - points[j]) < hardcore_mm:
                    hard = True
        for source_idx, lb in enumerate(labels):
            err = info.get("errors", {}).get(lb)
            err = float(err) if err is not None else float("inf")
            usable = bool(valid[source_idx] and np.isfinite(err) and err <= max_residual and not hard)
            for view, per_frame in slot_maps.items():
                perm = list(info.get(PERM_KEY[view]) or range(num_individuals))
                if source_idx >= len(perm):
                    continue
                node = per_frame.get(frame, {}).get(perm[source_idx])
                if node is not None:
                    rows.append({
                        "frame": int(frame), "source_idx": int(source_idx),
                        "view": view, "node_id": node,
                        "X": float(points[source_idx][0]) if valid[source_idx] else np.nan,
                        "Y": float(points[source_idx][1]) if valid[source_idx] else np.nan,
                        "Z": float(points[source_idx][2]) if valid[source_idx] else np.nan,
                        "residual": err, "geometry_usable": usable,
                        "hardcore": hard,
                        "view_quality": float(quality_maps.get(view, {}).get(frame, 0.0)),
                    })
    return pd.DataFrame(rows)


class _DSU:
    def __init__(self, nodes: list[str]):
        self.parent = {n: n for n in nodes}
        self.members = {n: {n} for n in nodes}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> str:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra].update(self.members.pop(rb))
        return ra


def build_constrained_tracklet_graph(
    observations: pd.DataFrame,
    node_ids: list[str],
    *,
    min_overlap_frames: int = 3,
    consistency_min: float = 0.95,
    silhouette_edges: pd.DataFrame | None = None,
    silhouette_max_3d_jump_mm: float = 8.0,
    silhouette_max_turn_deg: float = 120.0,
) -> tuple[dict[str, str], pd.DataFrame]:
    """Join cross-view tracklets while enforcing simultaneous-fish conflicts."""
    same = defaultdict(int)
    different = defaultdict(int)
    residuals = defaultdict(list)
    usable = observations[observations["geometry_usable"].astype(bool)]
    for _frame, fg in usable.groupby("frame"):
        records = fg.drop_duplicates(["view", "node_id", "source_idx"]).to_dict("records")
        for i, a in enumerate(records):
            for b in records[i + 1:]:
                pair = tuple(sorted((str(a["node_id"]), str(b["node_id"]))))
                if a["source_idx"] == b["source_idx"]:
                    if a["view"] == b["view"]:
                        continue
                    same[pair] += 1
                    residuals[pair].append(max(float(a["residual"]), float(b["residual"])))
                else:
                    different[pair] += 1

    pairs = sorted(set(same) | set(different))
    edge_rows = []
    for pair in pairs:
        ns, nd = same[pair], different[pair]
        consistency = ns / max(ns + nd, 1)
        edge_rows.append({
            "node_a": pair[0], "node_b": pair[1],
            "same_frames": int(ns), "different_frames": int(nd),
            "consistency": float(consistency),
            "median_residual": float(np.median(residuals[pair])) if residuals[pair] else np.nan,
            "eligible": bool(ns >= min_overlap_frames and consistency >= consistency_min),
            "edge_source": "cross_view_geometry",
            "support_weight": float(consistency),
            "trusted_mask_segment_id": np.nan,
            "gap_frames": np.nan,
        })
    edges = pd.DataFrame(edge_rows)
    if silhouette_edges is not None and not silhouette_edges.empty:
        soft_rows = []
        for row in silhouette_edges.to_dict("records"):
            a, b = sorted((str(row["node_a"]), str(row["node_b"])))
            if a not in node_ids or b not in node_ids or a == b:
                continue
            geometry = _validate_silhouette_edge_geometry(observations, a, b)
            step_limit = float(silhouette_max_3d_jump_mm) * max(
                1, int(geometry.get("frame_gap", row.get("gap_frames", 0))) + 1
            )
            distance = float(geometry.get("distance_mm", np.inf))
            turn = float(geometry.get("turn_deg", np.nan))
            if not bool(geometry.get("available", False)):
                validation = str(geometry.get("reason", "missing_3d_endpoint"))
            elif distance > step_limit:
                validation = "distance_exceeds_gate"
            elif np.isfinite(turn) and turn > silhouette_max_turn_deg:
                validation = "turn_exceeds_gate"
            else:
                validation = "compatible"
            geometry_ok = validation == "compatible"
            soft_rows.append({
                "node_a": a,
                "node_b": b,
                "same_frames": int(row.get("support_rows", 0)),
                "different_frames": 0,
                "consistency": 1.0,
                "median_residual": np.nan,
                "eligible": bool(geometry_ok),
                "edge_source": "silhouette_continuity",
                "support_weight": float(row.get("support_weight", 0.0)),
                "trusted_mask_segment_id": row.get("trusted_mask_segment_id", np.nan),
                "gap_frames": row.get("gap_frames", np.nan),
                "silhouette_3d_distance_mm": geometry.get("distance_mm", np.nan),
                "silhouette_3d_turn_deg": geometry.get("turn_deg", np.nan),
                "silhouette_3d_validation": validation,
            })
        if soft_rows:
            soft = pd.DataFrame(soft_rows).sort_values(
                ["support_weight", "same_frames"], ascending=[False, False]
            ).drop_duplicates(["node_a", "node_b"], keep="first")
            edges = pd.concat([edges, soft], ignore_index=True, sort=False)
    if edges.empty:
        return {n: n for n in node_ids}, edges

    conflicts = {p for p, count in different.items() if count > 0}
    dsu = _DSU(node_ids)
    edges["source_priority"] = edges["edge_source"].map(
        {"cross_view_geometry": 1, "silhouette_continuity": 0}
    ).fillna(0)
    order = edges.sort_values(
        ["eligible", "source_priority", "consistency", "support_weight", "same_frames", "median_residual"],
        ascending=[False, False, False, False, False, True],
    )
    accepted = {}
    reasons = {}
    for idx, edge in order.iterrows():
        a, b = str(edge.node_a), str(edge.node_b)
        if not bool(edge.eligible):
            accepted[idx] = False
            reasons[idx] = (
                str(edge.get("silhouette_3d_validation", "insufficient_support"))
                if str(edge.edge_source) == "silhouette_continuity"
                else "insufficient_support"
            )
            continue
        ra, rb = dsu.find(a), dsu.find(b)
        if ra == rb:
            accepted[idx] = True; reasons[idx] = "already_connected"; continue
        blocked = any(tuple(sorted((x, y))) in conflicts for x in dsu.members[ra] for y in dsu.members[rb])
        if blocked:
            accepted[idx] = False; reasons[idx] = "cannot_link_conflict"; continue
        dsu.union(ra, rb)
        accepted[idx] = True
        reasons[idx] = (
            "silhouette_continuity_soft_edge"
            if str(edge.edge_source) == "silhouette_continuity"
            else "cross_view_overlap"
        )
    edges["accepted"] = [bool(accepted.get(i, False)) for i in edges.index]
    edges["reason"] = [reasons.get(i, "not_processed") for i in edges.index]
    component = {node: dsu.find(node) for node in node_ids}
    return component, edges


def _validate_silhouette_edge_geometry(
    observations: pd.DataFrame,
    node_a: str,
    node_b: str,
) -> dict[str, object]:
    required = {"frame", "node_id", "X", "Y", "Z", "geometry_usable"}
    if not required.issubset(observations.columns):
        return {"available": False, "reason": "missing_3d_columns"}
    groups = []
    for node in (node_a, node_b):
        group = observations[
            observations["node_id"].astype(str).eq(node)
            & observations["geometry_usable"].astype(bool)
        ].drop_duplicates("frame").sort_values("frame")
        group = group[np.isfinite(group[["X", "Y", "Z"]].to_numpy(float)).all(axis=1)]
        if group.empty:
            return {"available": False, "reason": "missing_3d_endpoint"}
        groups.append(group)
    left, right = groups
    if int(left.frame.min()) > int(right.frame.min()):
        left, right = right, left
    if int(left.frame.max()) >= int(right.frame.min()):
        return {"available": False, "reason": "temporal_overlap"}
    left_last = left.iloc[-1]
    right_first = right.iloc[0]
    p_left = left_last[["X", "Y", "Z"]].to_numpy(float)
    p_right = right_first[["X", "Y", "Z"]].to_numpy(float)
    frame_gap = int(right_first.frame) - int(left_last.frame) - 1
    distance = float(np.linalg.norm(p_right - p_left))
    bridge_dt = max(1, int(right_first.frame) - int(left_last.frame))
    bridge_velocity = (p_right - p_left) / bridge_dt
    angles = []
    if len(left) >= 2:
        previous = left.iloc[-2]
        dt = max(1, int(left_last.frame) - int(previous.frame))
        velocity = (p_left - previous[["X", "Y", "Z"]].to_numpy(float)) / dt
        angles.append(_vector_angle_deg(velocity, bridge_velocity))
    if len(right) >= 2:
        following = right.iloc[1]
        dt = max(1, int(following.frame) - int(right_first.frame))
        velocity = (following[["X", "Y", "Z"]].to_numpy(float) - p_right) / dt
        angles.append(_vector_angle_deg(bridge_velocity, velocity))
    finite_angles = [value for value in angles if np.isfinite(value)]
    return {
        "available": True,
        "reason": "compatible",
        "distance_mm": distance,
        "turn_deg": max(finite_angles) if finite_angles else np.nan,
        "frame_gap": frame_gap,
    }


def _vector_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator < 1e-12:
        return np.nan
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def label_components(
    best: dict[int, dict],
    observations: pd.DataFrame,
    component_by_node: dict[str, str],
    *,
    num_individuals: int,
    appearance: pd.DataFrame | None = None,
    appearance_threshold: float = 0.8,
) -> tuple[dict[str, str], dict[str, str], int]:
    """Seed canonical IDs, then label disconnected components by appearance."""
    obs = observations.copy()
    obs["component"] = obs["node_id"].map(component_by_node)
    usable = obs[obs["geometry_usable"].astype(bool)]
    comp_labels: dict[str, str] = {}
    comp_source: dict[str, str] = {}
    seed_frame = -1
    for frame in sorted(best):
        fg = usable[usable.frame == frame]
        component_sets = []
        for source_idx in range(num_individuals):
            values = set(fg.loc[fg.source_idx == source_idx, "component"].dropna().astype(str))
            if not values:
                break
            component_sets.append(values)
        if len(component_sets) != num_individuals:
            continue
        if any(
            component_sets[i] & component_sets[j]
            for i in range(num_individuals)
            for j in range(i + 1, num_individuals)
        ):
            continue
        points = [np.asarray(best[frame]["groups"][chr(ord("a") + i)], float) for i in range(num_individuals)]
        if not all(p.shape == (3,) and np.isfinite(p).all() for p in points):
            continue
        for rank, source_idx in enumerate(sorted(range(num_individuals), key=lambda i: points[i][0])):
            for component in component_sets[source_idx]:
                comp_labels[component] = f"ID_{rank + 1}"
                comp_source[component] = "seed_geometry"
        seed_frame = int(frame)
        break
    if seed_frame < 0:
        raise RuntimeError("No clean frame could seed three distinct tracklet components")

    if appearance is not None and not appearance.empty:
        app = appearance.copy()
        if "node_id" not in app.columns:
            app["node_id"] = app["view"].astype(str) + ":" + app["tracklet_id"].map(_key)
        label_col = next((c for c in ("appearance_label", "predicted_identity", "identity") if c in app.columns), None)
        conf_col = next((c for c in ("appearance_confidence", "confidence", "probability") if c in app.columns), None)
        if label_col and conf_col:
            app["component"] = app["node_id"].map(component_by_node)
            app = app[pd.to_numeric(app[conf_col], errors="coerce") >= appearance_threshold]
            for comp, group in app.dropna(subset=["component"]).groupby("component"):
                labels = group[label_col].astype(str)
                labels = labels[labels.isin([f"ID_{i+1}" for i in range(num_individuals)])]
                if labels.empty:
                    continue
                label = labels.mode().iloc[0]
                if comp not in comp_labels:
                    comp_labels[str(comp)] = label
                    comp_source[str(comp)] = "appearance"
    return comp_labels, comp_source, seed_frame


def label_components_from_tracked_geometry(
    observations: pd.DataFrame,
    component_by_node: dict[str, str],
    tracked_geometry: dict[int, dict],
    seed_component_labels: dict[str, str],
    *,
    alignment_frame: int,
    min_frames: int = 3,
    min_fraction: float = 1.0,
) -> tuple[dict[str, str], pd.DataFrame]:
    """Label disconnected components from consistent geometry-ID votes.

    The geometry-only temporal tracker supplies one identity per selected 3D
    source. Its internal ID ordering is aligned to the canonical seed-frame
    component labels before any vote is counted. Votes are reduced to one per
    component/frame, and a component is accepted only when every usable frame
    agrees by default. This extends identity beyond a single seed frame without
    allowing a mixed-ID tracklet to become a hard identity constraint.
    """
    source_to_raw_identity: dict[tuple[int, int], str] = {}
    for frame, info in tracked_geometry.items():
        for identity_index, source_index in enumerate(info.get("id_mapping", [])):
            if int(source_index) >= 0:
                source_to_raw_identity[(int(frame), int(source_index))] = (
                    f"ID_{identity_index + 1}"
                )

    obs = observations.copy()
    obs["component"] = obs["node_id"].map(component_by_node)
    obs["raw_identity"] = [
        source_to_raw_identity.get((int(frame), int(source_index)), "")
        for frame, source_index in zip(obs.frame, obs.source_idx)
    ]
    usable = obs[
        obs["geometry_usable"].astype(bool)
        & obs["component"].notna()
        & obs["raw_identity"].ne("")
    ].copy()

    alignment = usable[usable.frame.astype(int).eq(int(alignment_frame))].copy()
    alignment["canonical_identity"] = alignment.component.map(seed_component_labels)
    alignment = alignment[alignment.canonical_identity.notna()]
    raw_to_canonical: dict[str, str] = {}
    used_canonical: set[str] = set()
    for raw_identity, group in alignment.groupby("raw_identity", sort=True):
        counts = group.canonical_identity.astype(str).value_counts()
        if counts.empty or (len(counts) > 1 and counts.iloc[0] == counts.iloc[1]):
            continue
        canonical = str(counts.index[0])
        if canonical in used_canonical:
            continue
        raw_to_canonical[str(raw_identity)] = canonical
        used_canonical.add(canonical)

    usable["identity"] = usable.raw_identity.map(raw_to_canonical).fillna("")
    usable = usable[usable.identity.ne("")]
    frame_votes = usable[["component", "frame", "identity"]].drop_duplicates()
    per_frame_cardinality = frame_votes.groupby(["component", "frame"]).identity.nunique()
    conflicted_components = set(
        per_frame_cardinality[per_frame_cardinality.gt(1)].index.get_level_values(0)
    )

    labels: dict[str, str] = {}
    decisions: list[dict[str, object]] = []
    for component, group in frame_votes.groupby("component", sort=True):
        component = str(component)
        counts = group.identity.value_counts()
        total_frames = int(group.frame.nunique())
        winner = str(counts.index[0]) if len(counts) else ""
        winner_frames = int(counts.iloc[0]) if len(counts) else 0
        fraction = winner_frames / max(1, total_frames)
        accepted = (
            component not in conflicted_components
            and total_frames >= int(min_frames)
            and fraction >= float(min_fraction)
        )
        if accepted:
            labels[component] = winner
        decisions.append(
            {
                "component_id": component,
                "accepted": bool(accepted),
                "assigned_identity": winner if accepted else "",
                "winner_identity": winner,
                "winner_frames": winner_frames,
                "total_frames": total_frames,
                "winner_fraction": float(fraction),
                "same_frame_conflict": component in conflicted_components,
                "reason": (
                    "unanimous_temporal_geometry_vote"
                    if accepted
                    else "same_frame_conflict"
                    if component in conflicted_components
                    else "insufficient_frames"
                    if total_frames < int(min_frames)
                    else "mixed_temporal_geometry_vote"
                ),
            }
        )
    return labels, pd.DataFrame(decisions)


def propagate_component_labels_by_exclusion(
    slot_maps: dict[str, dict[int, dict[int, str]]],
    component_by_node: dict[str, str],
    component_labels: dict[str, str],
    *,
    num_individuals: int = 3,
) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    """Lock deterministic identities from same-view all-different scopes.

    Unlike the former post-link hint logic, these scopes are built from every
    concrete tracklet associated with a candidate slot, including detections
    that the initial 3D selector did not use.
    """
    components = sorted(set(component_by_node.values()))
    scopes: list[tuple[str, ...]] = []
    for by_frame in slot_maps.values():
        for slot_nodes in by_frame.values():
            active = tuple(
                dict.fromkeys(
                    component_by_node[node]
                    for node in slot_nodes.values()
                    if node in component_by_node
                )
            )
            if 2 <= len(active) <= num_individuals:
                scopes.append(active)
    resolved, sources, domains = propagate_identity_domains(
        components,
        locked_identities=component_labels,
        simultaneous_groups=scopes,
        identity_labels=tuple(f"ID_{i + 1}" for i in range(num_individuals)),
    )
    # Existing geometry/appearance provenance remains authoritative.  Only
    # genuinely new singleton domains receive the exclusion reason.
    new_sources = {
        component: source
        for component, source in sources.items()
        if component not in component_labels
    }
    return resolved, new_sources, domains


def apply_targeted_appearance_fallback(
    appearance: pd.DataFrame | None,
    component_by_node: dict[str, str],
    identity_domains: dict[str, set[str]],
    *,
    confidence_threshold: float = 0.8,
) -> tuple[dict[str, str], dict[str, float], pd.DataFrame]:
    """Label only unresolved components when appearance respects constraints."""
    if appearance is None or appearance.empty:
        return {}, {}, pd.DataFrame()
    app = appearance.copy()
    if "node_id" not in app.columns:
        app["node_id"] = app["view"].astype(str) + ":" + app["tracklet_id"].map(_key)
    label_col = next(
        (c for c in ("appearance_label", "predicted_identity", "identity") if c in app.columns),
        None,
    )
    confidence_col = next(
        (c for c in ("appearance_confidence", "confidence", "probability") if c in app.columns),
        None,
    )
    if label_col is None or confidence_col is None:
        return {}, {}, pd.DataFrame()
    app["component"] = app.node_id.map(component_by_node)
    accepted: dict[str, str] = {}
    confidence: dict[str, float] = {}
    decisions = []
    for component, group in app.dropna(subset=["component"]).groupby("component"):
        component = str(component)
        domain = identity_domains.get(component, set())
        # Singleton domains are strong geometry/tracklet locks and are never
        # appearance targets.
        if len(domain) <= 1:
            continue
        valid = group[
            pd.to_numeric(group[confidence_col], errors="coerce") >= confidence_threshold
        ]
        valid = valid[valid[label_col].astype(str).isin(domain)]
        if valid.empty:
            decisions.append({
                "component_id": component, "accepted": False,
                "reason": "below_threshold_or_outside_identity_domain",
            })
            continue
        counts = valid[label_col].astype(str).value_counts()
        if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
            decisions.append({
                "component_id": component, "accepted": False,
                "reason": "ambiguous_appearance_vote",
            })
            continue
        identity = str(counts.index[0])
        chosen = valid[valid[label_col].astype(str) == identity]
        conf = float(pd.to_numeric(chosen[confidence_col], errors="coerce").median())
        accepted[component] = identity
        confidence[component] = conf
        decisions.append({
            "component_id": component, "accepted": True,
            "identity": identity, "confidence": conf,
            "reason": "unresolved_component_appearance_fallback",
        })
    return accepted, confidence, pd.DataFrame(decisions)


def build_identified_2d_rows(
    tracklet_paths: dict[str, Path],
    component_by_node: dict[str, str],
    component_labels: dict[str, str],
    component_sources: dict[str, str],
    observations: pd.DataFrame,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> pd.DataFrame:
    """Return long-form reliable 2D observations independent of valid XYZ."""
    selected = set(
        zip(
            observations.loc[observations.geometry_usable, "frame"].astype(int),
            observations.loc[observations.geometry_usable, "node_id"].astype(str),
        )
    )
    parts = []
    for view, path in tracklet_paths.items():
        table = pd.read_csv(path).copy()
        if start_frame is not None:
            table = table[pd.to_numeric(table["frame"], errors="coerce").ge(start_frame)]
        if end_frame is not None:
            table = table[pd.to_numeric(table["frame"], errors="coerce").le(end_frame)]
        table["view"] = view
        table["node_id"] = view + ":" + table["tracklet_id"].map(_key)
        table["component_id"] = table.node_id.map(component_by_node)
        table["final_fish_id"] = table.component_id.map(component_labels).fillna("")
        table["identity_source"] = table.component_id.map(component_sources).fillna("unresolved")
        is_selected = np.asarray([
            (int(frame), str(node)) in selected
            for frame, node in zip(table.frame, table.node_id)
        ], dtype=bool)
        assigned = table.final_fish_id != ""
        table["observation_status"] = np.select(
            [assigned & is_selected, assigned],
            ["assigned_2d_valid_3d", "assigned_2d_only"],
            default="unresolved_detection",
        )
        table["reason_code"] = np.select(
            [assigned & is_selected, assigned],
            ["assigned_three_view", "partial_2d_identity_only"],
            default="unresolved_identity",
        )
        parts.append(table)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_component_hints(
    observations: pd.DataFrame,
    component_by_node: dict[str, str],
    component_labels: dict[str, str],
    *,
    num_individuals: int = 3,
) -> tuple[dict[int, dict[int, str]], pd.DataFrame]:
    obs = observations.copy()
    obs["component"] = obs["node_id"].map(component_by_node)
    obs["identity"] = obs["component"].map(component_labels)
    hints = {}
    conflicts = []
    id_labels = [f"ID_{i + 1}" for i in range(num_individuals)]
    for frame, frame_group in obs.groupby("frame"):
        evidence = {}
        for source_idx in range(num_individuals):
            group = frame_group[frame_group.source_idx == source_idx]
            labeled = group.dropna(subset=["identity"])
            by_label = {}
            for identity, ig in labeled.groupby("identity"):
                identity = str(identity)
                if identity not in id_labels:
                    continue
                best_row = ig.loc[ig.get("view_quality", pd.Series(1.0, index=ig.index)).idxmax()]
                by_label[identity] = {
                    "quality": float(best_row.get("view_quality", 1.0)),
                    "view": str(best_row.get("view", "")),
                }
            evidence[source_idx] = (group, by_label)

        conflicted_sources = {
            source_idx for source_idx, (_group, by_label) in evidence.items()
            if len(by_label) > 1
        }
        label_sources = defaultdict(list)
        for source_idx, (_group, by_label) in evidence.items():
            if len(by_label) == 1:
                label_sources[next(iter(by_label))].append(source_idx)
        for sources in label_sources.values():
            if len(sources) > 1:
                conflicted_sources.update(sources)

        for source_idx, (group, by_label) in evidence.items():
            if source_idx in conflicted_sources:
                qualities = sorted(
                    ((identity, data["quality"], data["view"]) for identity, data in by_label.items()),
                    key=lambda item: item[1], reverse=True,
                )
                conflicts.append({
                    "frame": int(frame), "source_idx": int(source_idx),
                    "identities": "|".join(sorted(by_label)),
                    "best_identity": qualities[0][0] if qualities else "",
                    "best_view": qualities[0][2] if qualities else "",
                    "best_quality_bl": qualities[0][1] if qualities else 0.0,
                    "quality_margin_bl": (
                        float(qualities[0][1] - qualities[1][1])
                        if len(qualities) > 1 else 0.0
                    ),
                    "nodes": "|".join(sorted(group.node_id.astype(str).unique())),
                    "resolution": "veto_conflicting_3d_triplet",
                })
                continue
            if len(by_label) == 1:
                hints.setdefault(int(frame), {})[int(source_idx)] = next(iter(by_label))
    return hints, pd.DataFrame(conflicts)


def veto_conflicting_sources(
    best: dict[int, dict],
    conflicts: pd.DataFrame,
) -> dict[int, dict]:
    """Set cross-view-inconsistent selected 3D groups to NaN before linking."""
    out = {
        frame: {**info, "groups": dict(info.get("groups", {}))}
        for frame, info in best.items()
    }
    if conflicts.empty:
        return out
    for row in conflicts.itertuples(index=False):
        frame = int(row.frame)
        source_idx = int(row.source_idx)
        label = chr(ord("a") + source_idx)
        if frame in out:
            out[frame]["groups"][label] = [float("nan")] * 3
    return out


def continuity_vetoes(
    observations: pd.DataFrame,
    tracked: dict[int, dict],
    *,
    protected_hints: dict[int, dict[int, str]] | None = None,
) -> pd.DataFrame:
    """Return minority-ID frame/source rows for any concrete tracklet that jumps."""
    source_to_id = {}
    for frame, info in tracked.items():
        for id_idx, source_idx in enumerate(info.get("id_mapping", [])):
            if source_idx >= 0:
                source_to_id[(int(frame), int(source_idx))] = f"ID_{id_idx + 1}"
    obs = observations.copy()
    obs["assigned_identity"] = [
        source_to_id.get((int(f), int(s)), "")
        for f, s in zip(obs.frame, obs.source_idx)
    ]
    rows = []
    for node, group in obs[obs.assigned_identity != ""].groupby("node_id"):
        counts = group.assigned_identity.value_counts()
        if len(counts) <= 1:
            continue
        modal = str(counts.index[0])
        minority = group[group.assigned_identity != modal]
        for row in minority.itertuples(index=False):
            if int(row.source_idx) in (protected_hints or {}).get(int(row.frame), {}):
                continue
            rows.append({
                "frame": int(row.frame), "source_idx": int(row.source_idx),
                "identities": str(row.assigned_identity),
                "best_identity": modal,
                "best_view": str(row.view),
                "best_quality_bl": float(getattr(row, "view_quality", 0.0)),
                "quality_margin_bl": 0.0,
                "nodes": str(node),
                "resolution": "veto_minority_2d_identity",
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(["frame", "source_idx"])


def continuity_violations(
    observations: pd.DataFrame,
    tracked: dict[int, dict],
) -> pd.DataFrame:
    rows = []
    source_to_id = {}
    for frame, info in tracked.items():
        for id_idx, source_idx in enumerate(info.get("id_mapping", [])):
            if source_idx >= 0:
                source_to_id[(int(frame), int(source_idx))] = f"ID_{id_idx + 1}"
    obs = observations.copy()
    obs["assigned_identity"] = [
        source_to_id.get((int(f), int(s)), "")
        for f, s in zip(obs.frame, obs.source_idx)
    ]
    for node, group in obs[obs.assigned_identity != ""].groupby("node_id"):
        ids = sorted(group.assigned_identity.unique())
        if len(ids) > 1:
            rows.append({
                "node_id": node, "identities": "|".join(ids),
                "frame_start": int(group.frame.min()), "frame_end": int(group.frame.max()),
                "n_rows": int(len(group)),
            })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pre_link_json", type=Path)
    p.add_argument("--slp", action="append", default=[], help="view=path; repeat")
    p.add_argument("--tracklets", action="append", default=[], help="view=csv; repeat")
    p.add_argument("--appearance", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--output-tag", default="multiview_tracklet")
    p.add_argument("--num-individuals", type=int, default=3)
    p.add_argument("--slot-match-px", type=float, default=5.0)
    p.add_argument("--max-residual", type=float, default=5000.0)
    p.add_argument("--hardcore-mm", type=float, default=12.0)
    p.add_argument("--min-cross-view-overlap", type=int, default=3)
    p.add_argument("--cross-view-consistency", type=float, default=0.95)
    p.add_argument("--appearance-threshold", type=float, default=0.8)
    p.add_argument("--max-jump-mm", type=float, default=8.0)
    p.add_argument("--max-jump-hard-cap-mm", type=float, default=80.0)
    p.add_argument("--max-gap-frames", type=int, default=100)
    args = p.parse_args(argv)

    slp = _parse_view_paths(args.slp, "--slp")
    tracklets = _parse_view_paths(args.tracklets, "--tracklets")
    if set(slp) != set(tracklets) or len(slp) < 2:
        p.error("provide matching --slp and --tracklets for at least two views")
    best = load_best_pre_link_json(args.pre_link_json)
    frames = sorted(best)
    slot_maps = {}
    quality_maps = {}
    meta_parts = []
    slot_qc = {}
    for view in slp:
        slot_maps[view], quality_maps[view], meta, qc = load_tracklet_slot_map(
            view, slp[view], tracklets[view],
            start_frame=frames[0], end_frame=frames[-1],
            match_distance_px=args.slot_match_px,
        )
        meta_parts.append(meta); slot_qc[view] = qc
    node_meta = pd.concat(meta_parts, ignore_index=True)
    observations = build_observations(
        best, slot_maps, quality_maps, num_individuals=args.num_individuals,
        max_residual=args.max_residual, hardcore_mm=args.hardcore_mm,
    )
    component, edges = build_constrained_tracklet_graph(
        observations, node_meta.node_id.astype(str).tolist(),
        min_overlap_frames=args.min_cross_view_overlap,
        consistency_min=args.cross_view_consistency,
    )
    print(
        f"graph: {len(node_meta)} nodes -> {len(set(component.values()))} components; "
        f"accepted edges={int(edges.get('accepted', pd.Series(dtype=bool)).sum())}",
        flush=True,
    )
    appearance = pd.read_csv(args.appearance) if args.appearance else None
    comp_labels, comp_source, seed = label_components(
        best, observations, component, num_individuals=args.num_individuals,
        appearance=appearance, appearance_threshold=args.appearance_threshold,
    )
    propagated, exclusion_sources, identity_domains = propagate_component_labels_by_exclusion(
        slot_maps,
        component,
        comp_labels,
        num_individuals=args.num_individuals,
    )
    comp_labels.update(propagated)
    comp_source.update(exclusion_sources)
    hints, conflicts = build_component_hints(
        observations, component, comp_labels,
        num_individuals=args.num_individuals,
    )
    # This command consumes a winner-only pre-link artifact, so discarded
    # alternatives are unavailable here.  Preserve the observation and its
    # reliable 2D identity instead of the former conflict->NaN veto.  The
    # canonical tracklet-first reconstruction path supplies the same hints to
    # the candidate-set optimizer and re-solves before reaching this linker.
    tracked = apply_cross_frame_tracking(
        best, num_individuals=args.num_individuals,
        max_jump_mm=args.max_jump_mm,
        max_jump_hard_cap_mm=args.max_jump_hard_cap_mm,
        max_gap_frames=args.max_gap_frames,
        source_id_hints_by_frame=hints,
        source_hints_override_motion_gate=False,
    )
    violations = continuity_violations(observations, tracked)
    continuity_veto_count = 0
    if not conflicts.empty:
        conflicts = conflicts.copy()
        conflicts["resolution"] = "preserved_for_candidate_reoptimization"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.output_tag
    best_path = args.output_dir / f"{tag}_best.json"
    _save_best_tracked_json(tracked, best_path, num_individuals=args.num_individuals)
    dfs = best_by_frame_to_dfs(
        tracked, shift_to_positive=False, ensure_non_negative=False,
        num_individuals=args.num_individuals,
    )
    dfs = ensure_all_frames_and_fill_spline(dfs, frames, max_gap_frames=args.max_gap_frames)
    csv_path = dfs_to_single_csv(dfs, set_key=tag, output_dir=str(args.output_dir))

    assignments = node_meta.copy()
    assignments["component_id"] = assignments.node_id.map(component)
    assignments["assigned_identity"] = assignments.component_id.map(comp_labels).fillna("")
    assignments["identity_source"] = assignments.component_id.map(comp_source).fillna("unresolved")
    assignments["identity_domain"] = assignments.component_id.map(
        lambda value: "|".join(sorted(identity_domains.get(value, set())))
    )
    violation_assignments = violations.merge(
        assignments[["node_id", "assigned_identity", "identity_source"]],
        on="node_id", how="left",
    ) if not violations.empty else pd.DataFrame()
    labeled_violations = (
        violation_assignments[violation_assignments.identity_source != "unresolved"]
        if not violation_assignments.empty else violation_assignments
    )
    assignments.to_csv(args.output_dir / f"{tag}_tracklet_assignments.csv", index=False)
    identified_2d = build_identified_2d_rows(
        tracklets,
        component,
        comp_labels,
        comp_source,
        observations,
    )
    identified_2d.to_csv(args.output_dir / f"{tag}_identified_2d.csv", index=False)
    edges.to_csv(args.output_dir / f"{tag}_cross_view_edges.csv", index=False)
    conflicts.to_csv(args.output_dir / f"{tag}_hint_conflicts.csv", index=False)
    violations.to_csv(args.output_dir / f"{tag}_continuity_violations.csv", index=False)
    unresolved = assignments[assignments.assigned_identity == ""].copy()
    unresolved["reason"] = "appearance_or_manual_identity_needed"
    violation_nodes = set(violations.get("node_id", pd.Series(dtype=str)).astype(str))
    unresolved["has_2d_continuity_violation"] = unresolved.node_id.astype(str).isin(violation_nodes)
    unresolved["appearance_priority"] = np.where(
        unresolved["has_2d_continuity_violation"], "high", "normal"
    )
    unresolved = unresolved.sort_values(
        ["has_2d_continuity_violation", "n_rows"], ascending=[False, False]
    )
    unresolved.to_csv(args.output_dir / f"{tag}_appearance_needed.csv", index=False)
    summary = {
        "seed_frame": seed,
        "slot_matching": slot_qc,
        "tracklet_nodes": int(len(assignments)),
        "graph_components": int(len(set(component.values()))),
        "labeled_components": int(len(comp_labels)),
        "exclusion_labeled_components": int(len(exclusion_sources)),
        "unresolved_tracklets": int(len(unresolved)),
        "accepted_cross_view_edges": int(edges.get("accepted", pd.Series(dtype=bool)).sum()),
        "frames_with_identity_hints": int(len(hints)),
        "hint_conflicts": int(len(conflicts)),
        "continuity_vetoed_frame_sources": int(continuity_veto_count),
        "two_d_continuity_violations": int(len(violations)),
        "labeled_two_d_continuity_violations": int(len(labeled_violations)),
        "best_json": str(best_path), "csv_3d": str(csv_path),
        "identified_2d_csv": str(args.output_dir / f"{tag}_identified_2d.csv"),
    }
    (args.output_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
