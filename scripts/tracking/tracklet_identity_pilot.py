from __future__ import annotations

"""Pilot experiment for grouping short tracklets into three fish identities.

This script is intentionally artifact-first: it compares identity-linking
methods on small, explicit frame windows using CSV/parquet files that already
exist or can be produced by other scripts in this repository. The expensive
parts of the pipeline (SLEAP, 3D reconstruction, SAM2, Detectron2) stay outside
this pilot and feed it with inspectable tables.

Typical first run::

    python -m scripts.tracking.tracklet_identity_pilot \
        --tracklets top_cam=runs/tracklet_bind/EXP/top_cam.tracklet_bound.csv \
        --tracklets side1_cam=runs/tracklet_bind/EXP/side1_cam.tracklet_bound.csv \
        --tracklets side2_cam=runs/tracklet_bind/EXP/side2_cam.tracklet_bound.csv \
        --csv-3d Z:/.../reconstruction_3d/EXP/round4_260520/round4_260520_3d_identified.csv \
        --windows 176300-181300 \
        --out-dir runs/tracklet_identity_pilot/EXP_f176300_181300

The method families are transparent heuristics, not final models:

* geometry_only: uses 3D-bound slot labels/confidence from tracklet tables.
* appearance_only: uses optional crop/mask classifier predictions.
* temporal_dynamics: links fragments by endpoint motion plausibility.
* hybrid: combines geometry, appearance, and temporal evidence.
* udmt_inspired: temporal/crossing-aware baseline that makes uncertain crossing
  periods explicit instead of forcing a label.
"""

import argparse
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.tracking.evaluate_3d_id_consistency import spike_metrics


RESERVED_LABELS = {"", "nan", "none", "null", "unknown", "mix", "crossing", "unassigned"}
METHODS = [
    "geometry_only",
    "appearance_only",
    "temporal_dynamics",
    "hybrid",
    "udmt_inspired",
]
ASSIGNED_PREFIX = "ID_"


@dataclass(frozen=True)
class PilotConfig:
    num_individuals: int = 3
    max_gap_frames: int = 500
    max_speed_px_per_frame: float = 12.0
    max_3d_speed_mm_per_frame: float = 10.0
    geometry_threshold: float = 0.60
    appearance_threshold: float = 0.60
    temporal_threshold: float = 0.35
    hybrid_threshold: float = 0.55
    udmt_threshold: float = 0.35
    hardcore_mm: float = 12.0
    event_pad_frames: int = 250
    easy_separation_mm: float = 40.0
    event_node_fraction_for_crossing: float = 0.40
    min_component_confidence: float = 0.05
    max_plot_events: int = 8


class DisjointSet:
    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p)


def parse_key_path(spec: str, *, default_key: str | None = None) -> tuple[str, Path]:
    if "=" in spec:
        key, value = spec.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty key in {spec!r}")
        return key, Path(value.strip())
    path = Path(spec)
    key = default_key or path.stem
    return key, path


def parse_window(text: str) -> dict:
    raw = text.strip()
    for sep in ("-", ":", ","):
        if sep in raw:
            a, b = raw.split(sep, 1)
            start = int(a.strip())
            end = int(b.strip())
            if end < start:
                raise ValueError(f"window end before start: {text}")
            return {
                "window_id": f"w{start}_{end}",
                "kind": "manual",
                "start": start,
                "end": end,
                "source": "cli",
                "reason": raw,
            }
    raise ValueError(f"window must look like START-END, got {text!r}")


def parse_windows(values: list[str] | None) -> pd.DataFrame:
    rows = [parse_window(v) for v in values or []]
    return pd.DataFrame(rows, columns=["window_id", "kind", "start", "end", "source", "reason"])


def _in_any_window(frames: pd.Series, windows: pd.DataFrame) -> pd.Series:
    if windows.empty:
        return pd.Series(True, index=frames.index)
    mask = pd.Series(False, index=frames.index)
    for row in windows.itertuples(index=False):
        mask |= frames.between(int(row.start), int(row.end))
    return mask


def _canonical_xy(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    out = df.copy()
    if "x" not in out.columns:
        for cand in ("centroid_x", "center_x", "mouth_x"):
            if cand in out.columns:
                out["x"] = out[cand]
                break
    if "y" not in out.columns:
        for cand in ("centroid_y", "center_y", "mouth_y"):
            if cand in out.columns:
                out["y"] = out[cand]
                break
    missing = {"frame", "x", "y"} - set(out.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)}")
    if "tracklet_id" not in out.columns:
        if "track_id" in out.columns:
            out["tracklet_id"] = out["track_id"]
        elif "identity_trajectory_id" in out.columns:
            out["tracklet_id"] = out["identity_trajectory_id"]
        else:
            raise ValueError(f"{label} needs tracklet_id or track_id")
    out["frame"] = pd.to_numeric(out["frame"], errors="coerce")
    if out["frame"].isna().any():
        raise ValueError(f"{label} has non-numeric frame values")
    out["frame"] = out["frame"].astype(int)
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    if "score" in out.columns:
        out["score"] = pd.to_numeric(out["score"], errors="coerce")
    else:
        out["score"] = np.nan
    return out


def load_tracklets(tracklet_specs: list[str], windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in tracklet_specs:
        view, path = parse_key_path(spec)
        df = _canonical_xy(read_table(path), label=str(path))
        if "view" not in df.columns:
            df["view"] = view
        df["source_path"] = str(path)
        df = df[_in_any_window(df["frame"], windows)].copy()
        rows.append(df)
    if not rows:
        raise ValueError("provide at least one --tracklets CSV/parquet")
    out = pd.concat(rows, ignore_index=True, sort=False)
    out["view"] = out["view"].fillna("").astype(str)
    out["tracklet_id"] = out["tracklet_id"].astype(str)
    return out.sort_values(["view", "tracklet_id", "frame"]).reset_index(drop=True)


def normalize_label(value, *, numeric_base: int = 0) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in RESERVED_LABELS:
        return ""
    try:
        if isinstance(value, (int, np.integer)) or (text.replace(".", "", 1).isdigit() and float(text).is_integer()):
            n = int(float(text))
            return f"{ASSIGNED_PREFIX}{n + 1 if numeric_base == 0 else n}"
    except Exception:
        pass
    if text.lower().startswith("id_"):
        suffix = text.split("_", 1)[1]
        return f"{ASSIGNED_PREFIX}{suffix}"
    return text


def _majority_label(values: Iterable, *, numeric_base: int = 0) -> tuple[str, float]:
    labels = [normalize_label(v, numeric_base=numeric_base) for v in values]
    labels = [v for v in labels if v]
    if not labels:
        return "", 0.0
    counts = Counter(labels)
    label, n = counts.most_common(1)[0]
    return label, float(n / max(len(labels), 1))


def _mean_numeric(values: pd.Series, default: float = np.nan) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    return float(x.mean()) if len(x) else default


def _event_overlap_count(frames: pd.Series, events: pd.DataFrame) -> int:
    if events.empty or frames.empty:
        return 0
    mask = pd.Series(False, index=frames.index)
    for ev in events.itertuples(index=False):
        mask |= frames.between(int(ev.start), int(ev.end))
    return int(mask.sum())


def _geometry_from_group(group: pd.DataFrame) -> tuple[str, float]:
    label = ""
    purity = 0.0
    numeric_base = 0
    for col in ("global_id", "geometry_id", "fish_id", "final_fish_id"):
        if col in group.columns:
            numeric_base = 0 if col in {"global_id"} else 1
            label, purity = _majority_label(group[col], numeric_base=numeric_base)
            if label:
                break
    if not label:
        return "", 0.0

    conf_parts = [purity]
    if "bind_consistency" in group.columns:
        conf_parts.append(float(np.clip(_mean_numeric(group["bind_consistency"], 0.0), 0.0, 1.0)))
    if "bind_resid" in group.columns:
        resid = _mean_numeric(group["bind_resid"], np.nan)
        if np.isfinite(resid):
            conf_parts.append(float(np.clip(1.0 - resid / 60.0, 0.0, 1.0)))
    if "bind_margin" in group.columns:
        margin = _mean_numeric(group["bind_margin"], np.nan)
        if np.isfinite(margin):
            conf_parts.append(float(np.clip(margin / 60.0, 0.0, 1.0)))
    if "low_conf" in group.columns:
        low = group["low_conf"].astype(str).str.lower().isin({"true", "1", "yes"}).mean()
        conf_parts.append(float(1.0 - low))
    return label, float(np.nanmean(conf_parts))


def _reference_from_group(group: pd.DataFrame) -> tuple[str, float]:
    for col in (
        "reference_identity",
        "ground_truth_identity",
        "manual_identity",
        "review_identity",
        "biological_identity",
    ):
        if col in group.columns:
            label, purity = _majority_label(group[col], numeric_base=1)
            if label:
                return label, purity
    return "", 0.0


def summarize_appearance(appearance: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "view",
        "tracklet_id",
        "appearance_label",
        "appearance_confidence",
        "appearance_purity",
        "appearance_n",
    ]
    if appearance is None or appearance.empty:
        return pd.DataFrame(columns=columns)
    df = appearance.copy()
    if "tracklet_id" not in df.columns:
        if "track_id" in df.columns:
            df["tracklet_id"] = df["track_id"]
        elif "identity_trajectory_id" in df.columns:
            df["tracklet_id"] = df["identity_trajectory_id"]
        else:
            raise ValueError("appearance table needs tracklet_id or track_id")
    if "view" not in df.columns:
        df["view"] = ""
    label_col = ""
    for col in (
        "accepted_predicted_identity",
        "predicted_identity",
        "model_identity",
        "assigned_identity",
        "identity",
        "label",
    ):
        if col in df.columns:
            label_col = col
            break
    if not label_col:
        raise ValueError("appearance table needs a predicted identity column")
    conf_col = ""
    for col in (
        "confidence",
        "prediction_confidence",
        "probability",
        "model_identity_purity",
        "accepted_probability",
        "score",
        "embedding_margin",
    ):
        if col in df.columns:
            conf_col = col
            break
    rows = []
    for keys, group in df.groupby(["view", "tracklet_id"], dropna=False, sort=True):
        view, tracklet_id = keys
        label, purity = _majority_label(group[label_col], numeric_base=1)
        conf = purity
        if conf_col:
            v = _mean_numeric(group[conf_col], np.nan)
            if np.isfinite(v):
                conf = float(np.clip(v, 0.0, 1.0)) if conf_col != "embedding_margin" else float(np.clip(v / 0.3, 0.0, 1.0))
        rows.append(
            {
                "view": str(view),
                "tracklet_id": str(tracklet_id),
                "appearance_label": label,
                "appearance_confidence": conf,
                "appearance_purity": purity,
                "appearance_n": int(len(group)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_nodes(tracklets: pd.DataFrame, events: pd.DataFrame, appearance: pd.DataFrame | None) -> pd.DataFrame:
    app = summarize_appearance(appearance)
    rows = []
    for (view, tid), group in tracklets.groupby(["view", "tracklet_id"], sort=True):
        ordered = group.sort_values("frame")
        finite = ordered[np.isfinite(ordered[["x", "y"]]).all(axis=1)]
        if finite.empty:
            continue
        first = finite.iloc[0]
        last = finite.iloc[-1]
        duration = max(1, int(last["frame"]) - int(first["frame"]))
        vx = (float(last["x"]) - float(first["x"])) / duration
        vy = (float(last["y"]) - float(first["y"])) / duration
        geom_label, geom_conf = _geometry_from_group(ordered)
        ref_label, ref_purity = _reference_from_group(ordered)
        event_rows = _event_overlap_count(ordered["frame"], events)
        rows.append(
            {
                "node_id": f"{view}:{tid}",
                "view": str(view),
                "tracklet_id": str(tid),
                "frame_start": int(ordered["frame"].min()),
                "frame_end": int(ordered["frame"].max()),
                "n_rows": int(len(ordered)),
                "start_x": float(first["x"]),
                "start_y": float(first["y"]),
                "end_x": float(last["x"]),
                "end_y": float(last["y"]),
                "mean_x": float(finite["x"].mean()),
                "mean_y": float(finite["y"].mean()),
                "vx": float(vx),
                "vy": float(vy),
                "mean_score": _mean_numeric(ordered["score"], np.nan),
                "geometry_label": geom_label,
                "geometry_confidence": geom_conf,
                "reference_label": ref_label,
                "reference_purity": ref_purity,
                "event_rows": int(event_rows),
                "event_fraction": float(event_rows / max(len(ordered), 1)),
                "source_path": str(ordered["source_path"].iloc[0]) if "source_path" in ordered.columns else "",
            }
        )
    nodes = pd.DataFrame(rows)
    if nodes.empty:
        return nodes
    if not app.empty:
        app_specific = app[app["view"].astype(str).ne("")]
        app_global = app[app["view"].astype(str).eq("")].drop(columns=["view"])
        if not app_specific.empty:
            nodes = nodes.merge(app_specific, on=["view", "tracklet_id"], how="left")
        if not app_global.empty:
            nodes = nodes.merge(app_global, on=["tracklet_id"], how="left", suffixes=("", "_global"))
            for col in ("appearance_label", "appearance_confidence", "appearance_purity", "appearance_n"):
                gcol = f"{col}_global"
                if gcol in nodes.columns:
                    nodes[col] = nodes.get(col, pd.Series(index=nodes.index)).fillna(nodes[gcol])
                    nodes = nodes.drop(columns=[gcol])
    for col, default in {
        "appearance_label": "",
        "appearance_confidence": 0.0,
        "appearance_purity": 0.0,
        "appearance_n": 0,
    }.items():
        if col not in nodes.columns:
            nodes[col] = default
        nodes[col] = nodes[col].fillna(default)
    return nodes.sort_values(["view", "frame_start", "tracklet_id"]).reset_index(drop=True)


def _interval_overlaps_events(start: int, end: int, events: pd.DataFrame) -> bool:
    if events.empty:
        return False
    a, b = min(start, end), max(start, end)
    for ev in events.itertuples(index=False):
        if a <= int(ev.end) and int(ev.start) <= b:
            return True
    return False


def _angle_score(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.75
    cosv = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float((cosv + 1.0) / 2.0)


def build_edges(nodes: pd.DataFrame, events: pd.DataFrame, cfg: PilotConfig) -> pd.DataFrame:
    rows = []
    if nodes.empty:
        return pd.DataFrame()
    for view, group in nodes.groupby("view", sort=True):
        ordered = group.sort_values("frame_start").reset_index(drop=True)
        records = ordered.to_dict("records")
        for i, a in enumerate(records):
            for b in records[i + 1:]:
                if int(b["frame_start"]) <= int(a["frame_end"]):
                    continue
                gap = int(b["frame_start"]) - int(a["frame_end"])
                if gap > int(cfg.max_gap_frames):
                    break
                bridge = np.array([float(b["start_x"]) - float(a["end_x"]), float(b["start_y"]) - float(a["end_y"])])
                dist = float(np.linalg.norm(bridge))
                speed = dist / max(gap, 1)
                gap_score = math.exp(-gap / max(float(cfg.max_gap_frames), 1.0))
                speed_score = math.exp(-speed / max(float(cfg.max_speed_px_per_frame), 1e-6))
                heading_score = _angle_score(np.array([float(a["vx"]), float(a["vy"])]), bridge)
                temporal_score = float(np.clip(0.55 * speed_score + 0.25 * gap_score + 0.20 * heading_score, 0.0, 1.0))

                geom_a = str(a.get("geometry_label", "") or "")
                geom_b = str(b.get("geometry_label", "") or "")
                geom_conf = float(np.nanmean([a.get("geometry_confidence", 0.0), b.get("geometry_confidence", 0.0)]))
                geometry_same = bool(geom_a and geom_b and geom_a == geom_b)
                geometry_conflict = bool(geom_a and geom_b and geom_a != geom_b)
                geometry_score = float(geom_conf if geometry_same else 0.0)

                app_a = str(a.get("appearance_label", "") or "")
                app_b = str(b.get("appearance_label", "") or "")
                app_conf = float(np.nanmean([a.get("appearance_confidence", 0.0), b.get("appearance_confidence", 0.0)]))
                appearance_same = bool(app_a and app_b and app_a == app_b)
                appearance_conflict = bool(app_a and app_b and app_a != app_b)
                appearance_score = float(app_conf if appearance_same else 0.0)

                event_overlap = _interval_overlaps_events(int(a["frame_end"]), int(b["frame_start"]), events)
                event_penalty = 0.20 if event_overlap else 0.0
                available = []
                if geom_a or geom_b:
                    available.append(("geometry", geometry_score))
                if app_a or app_b:
                    available.append(("appearance", appearance_score))
                available.append(("temporal", temporal_score))
                weighted = 0.0
                weight_total = 0.0
                for name, value in available:
                    w = {"geometry": 0.40, "appearance": 0.30, "temporal": 0.30}[name]
                    weighted += w * value
                    weight_total += w
                hybrid_score = float(np.clip(weighted / max(weight_total, 1e-6) - event_penalty, 0.0, 1.0))
                corroborated = geometry_same or appearance_same
                udmt_score = temporal_score
                if event_overlap and not corroborated:
                    udmt_score *= 0.40
                if geometry_conflict or appearance_conflict:
                    hybrid_score *= 0.45
                    udmt_score *= 0.50

                rows.append(
                    {
                        "edge_id": f"{a['node_id']}->{b['node_id']}",
                        "view": str(view),
                        "node_a": a["node_id"],
                        "node_b": b["node_id"],
                        "a_end": int(a["frame_end"]),
                        "b_start": int(b["frame_start"]),
                        "gap_frames": int(gap),
                        "endpoint_distance_px": dist,
                        "speed_px_per_frame": speed,
                        "heading_score": heading_score,
                        "temporal_score": temporal_score,
                        "geometry_same": geometry_same,
                        "geometry_conflict": geometry_conflict,
                        "geometry_score": geometry_score,
                        "appearance_same": appearance_same,
                        "appearance_conflict": appearance_conflict,
                        "appearance_score": appearance_score,
                        "event_overlap": bool(event_overlap),
                        "hybrid_score": hybrid_score,
                        "udmt_score": float(np.clip(udmt_score, 0.0, 1.0)),
                    }
                )
    return pd.DataFrame(rows)


def _method_selected_edges(edges: pd.DataFrame, method: str, cfg: PilotConfig) -> pd.DataFrame:
    if edges.empty:
        return edges.copy()
    if method == "geometry_only":
        mask = edges["geometry_score"].fillna(0.0) >= float(cfg.geometry_threshold)
    elif method == "appearance_only":
        mask = edges["appearance_score"].fillna(0.0) >= float(cfg.appearance_threshold)
    elif method == "temporal_dynamics":
        mask = (
            (edges["temporal_score"].fillna(0.0) >= float(cfg.temporal_threshold))
            & ~edges["event_overlap"].astype(bool)
        )
    elif method == "hybrid":
        mask = edges["hybrid_score"].fillna(0.0) >= float(cfg.hybrid_threshold)
    elif method == "udmt_inspired":
        mask = edges["udmt_score"].fillna(0.0) >= float(cfg.udmt_threshold)
    else:
        raise ValueError(method)
    out = edges[mask].copy()
    out["method"] = method
    return out


def _component_label(component_nodes: pd.DataFrame, method: str, cfg: PilotConfig) -> tuple[str, float, str]:
    evidence: list[tuple[str, float, str]] = []
    if method in {"geometry_only", "hybrid", "udmt_inspired"}:
        for row in component_nodes.itertuples(index=False):
            if str(row.geometry_label):
                evidence.append((str(row.geometry_label), float(row.geometry_confidence), "geometry"))
    if method in {"appearance_only", "hybrid", "udmt_inspired"}:
        for row in component_nodes.itertuples(index=False):
            if str(row.appearance_label):
                evidence.append((str(row.appearance_label), float(row.appearance_confidence), "appearance"))

    evidence = [(label, conf, src) for label, conf, src in evidence if label.lower() not in RESERVED_LABELS]
    if evidence:
        score_by_label: dict[str, float] = defaultdict(float)
        count_by_label: dict[str, int] = defaultdict(int)
        src_by_label: dict[str, set[str]] = defaultdict(set)
        for label, conf, src in evidence:
            score_by_label[label] += max(float(conf), float(cfg.min_component_confidence))
            count_by_label[label] += 1
            src_by_label[label].add(src)
        ordered = sorted(score_by_label, key=lambda lab: score_by_label[lab], reverse=True)
        best = ordered[0]
        if len(ordered) > 1 and score_by_label[ordered[1]] >= 0.75 * score_by_label[best]:
            return "mix", 0.0, "conflicting_evidence"
        conf = score_by_label[best] / max(count_by_label[best], 1)
        return best, float(np.clip(conf, 0.0, 1.0)), "+".join(sorted(src_by_label[best]))

    if method == "geometry_only":
        return "unknown", 0.0, "no_geometry"
    if method == "appearance_only":
        return "unknown", 0.0, "no_appearance"
    return "", 0.0, "temporal_component"


def assign_methods(nodes: pd.DataFrame, edges: pd.DataFrame, cfg: PilotConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_all = []
    assignment_rows = []
    node_ids = list(nodes["node_id"].astype(str))
    nodes_by_id = nodes.set_index("node_id", drop=False)
    for method in METHODS:
        selected = _method_selected_edges(edges, method, cfg)
        selected_all.append(selected)
        dsu = DisjointSet(node_ids)
        for edge in selected.itertuples(index=False):
            dsu.union(str(edge.node_a), str(edge.node_b))

        comp_members: dict[str, list[str]] = defaultdict(list)
        for nid in node_ids:
            comp_members[dsu.find(nid)].append(nid)

        comp_labels: dict[str, tuple[str, float, str]] = {}
        unlabeled_comps = []
        for comp_id, members in comp_members.items():
            c_nodes = nodes_by_id.loc[members]
            label, conf, evidence = _component_label(c_nodes, method, cfg)
            comp_labels[comp_id] = (label, conf, evidence)
            if not label:
                total_rows = int(c_nodes["n_rows"].sum())
                earliest = int(c_nodes["frame_start"].min())
                unlabeled_comps.append((comp_id, total_rows, earliest))

        if method in {"temporal_dynamics", "udmt_inspired", "hybrid"}:
            already = {
                label
                for label, _, _ in comp_labels.values()
                if label.startswith(ASSIGNED_PREFIX)
            }
            slots = [f"{ASSIGNED_PREFIX}{i}" for i in range(1, int(cfg.num_individuals) + 1)]
            remaining = [slot for slot in slots if slot not in already]
            for comp_id, _, _ in sorted(unlabeled_comps, key=lambda x: (-x[1], x[2])):
                if remaining:
                    comp_labels[comp_id] = (remaining.pop(0), 0.35, "temporal_rank")
                else:
                    comp_labels[comp_id] = ("unknown", 0.0, "extra_temporal_component")

        for nid in node_ids:
            comp_id = dsu.find(nid)
            label, conf, evidence = comp_labels[comp_id]
            node = nodes_by_id.loc[nid]
            if (
                method == "udmt_inspired"
                and float(node["event_fraction"]) >= float(cfg.event_node_fraction_for_crossing)
                and (not str(label).startswith(ASSIGNED_PREFIX) or conf < 0.75)
            ):
                label = "crossing"
                evidence = f"{evidence}+crossing_gate" if evidence else "crossing_gate"
            if not label:
                label = "unknown"
            assignment_rows.append(
                {
                    "method": method,
                    "node_id": nid,
                    "view": str(node["view"]),
                    "tracklet_id": str(node["tracklet_id"]),
                    "component_id": f"{method}:{comp_id}",
                    "assigned_identity": label,
                    "assignment_confidence": float(np.clip(conf, 0.0, 1.0)),
                    "assignment_evidence": evidence,
                    "is_assigned_id": bool(str(label).startswith(ASSIGNED_PREFIX)),
                    "is_unknown": label == "unknown",
                    "is_mix": label == "mix",
                    "is_crossing": label == "crossing",
                }
            )
    selected_edges = pd.concat(selected_all, ignore_index=True, sort=False) if selected_all else pd.DataFrame()
    assignments = pd.DataFrame(assignment_rows)
    return assignments, selected_edges


def expand_per_frame(tracklets: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    if tracklets.empty or assignments.empty:
        return pd.DataFrame()
    base = tracklets.copy()
    base["node_id"] = base["view"].astype(str) + ":" + base["tracklet_id"].astype(str)
    keep_cols = [
        "method",
        "node_id",
        "component_id",
        "assigned_identity",
        "assignment_confidence",
        "assignment_evidence",
        "is_assigned_id",
        "is_unknown",
        "is_mix",
        "is_crossing",
    ]
    return base.merge(assignments[keep_cols], on="node_id", how="inner")


def _speed_spike_rate(rows: pd.DataFrame, *, threshold: float) -> tuple[int, int, float]:
    spikes = 0
    transitions = 0
    if rows.empty:
        return 0, 0, float("nan")
    assigned = rows[rows["assigned_identity"].astype(str).str.startswith(ASSIGNED_PREFIX)].copy()
    for (_, identity), group in assigned.groupby(["view", "assigned_identity"], sort=False):
        ordered = group.sort_values("frame")
        rec = ordered[["frame", "x", "y"]].to_records(index=False)
        for a, b in zip(rec, rec[1:]):
            gap = int(b.frame) - int(a.frame)
            if gap != 1:
                continue
            speed = float(np.hypot(float(b.x) - float(a.x), float(b.y) - float(a.y)))
            transitions += 1
            spikes += int(speed > threshold)
    return spikes, transitions, float(spikes / transitions) if transitions else float("nan")


def _best_remapped_accuracy(df: pd.DataFrame) -> tuple[float, dict[str, str]]:
    valid = df[
        df["assigned_identity"].astype(str).str.startswith(ASSIGNED_PREFIX)
        & df["reference_label"].fillna("").astype(str).ne("")
    ].copy()
    if valid.empty:
        return float("nan"), {}
    pred = sorted(valid["assigned_identity"].astype(str).unique())
    ref = sorted(valid["reference_label"].astype(str).unique())
    counts = np.zeros((len(pred), len(ref)), dtype=float)
    for i, p in enumerate(pred):
        for j, r in enumerate(ref):
            counts[i, j] = float(((valid["assigned_identity"].astype(str) == p) & (valid["reference_label"].astype(str) == r)).sum())
    ri, ci = linear_sum_assignment(-counts)
    total = float(len(valid))
    correct = float(counts[ri, ci].sum())
    mapping = {pred[i]: ref[j] for i, j in zip(ri, ci)}
    return correct / max(total, 1.0), mapping


def _tracklet_reference_metrics(nodes: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    if nodes.empty or assignments.empty:
        return pd.DataFrame()
    merged = assignments.merge(nodes[["node_id", "reference_label", "n_rows"]], on="node_id", how="left")
    rows = []
    for method, group in merged.groupby("method", sort=False):
        labeled = group[group["reference_label"].fillna("").astype(str).ne("")]
        if labeled.empty:
            rows.append({"method": method, "literal_tracklet_accuracy": np.nan, "best_remapped_tracklet_accuracy": np.nan, "best_remap": {}})
            continue
        assigned = labeled[labeled["assigned_identity"].astype(str).str.startswith(ASSIGNED_PREFIX)]
        literal = float((assigned["assigned_identity"].astype(str) == assigned["reference_label"].astype(str)).mean()) if not assigned.empty else float("nan")
        best, mapping = _best_remapped_accuracy(labeled)
        rows.append({"method": method, "literal_tracklet_accuracy": literal, "best_remapped_tracklet_accuracy": best, "best_remap": mapping})
    return pd.DataFrame(rows)


def summarize_metrics(
    nodes: pd.DataFrame,
    assignments: pd.DataFrame,
    per_frame: pd.DataFrame,
    selected_edges: pd.DataFrame,
    events: pd.DataFrame,
    cfg: PilotConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    for method in METHODS:
        method_assign = assignments[assignments["method"] == method]
        method_rows = per_frame[per_frame["method"] == method]
        method_edges = selected_edges[selected_edges["method"] == method] if not selected_edges.empty else pd.DataFrame()
        for view in sorted(nodes["view"].unique()):
            nview = nodes[nodes["view"] == view]
            aview = method_assign[method_assign["view"] == view]
            rview = method_rows[method_rows["view"] == view]
            assigned = rview["assigned_identity"].astype(str).str.startswith(ASSIGNED_PREFIX) if not rview.empty else pd.Series(dtype=bool)
            frames = int(rview["frame"].nunique()) if not rview.empty else 0
            expected = frames * int(cfg.num_individuals)
            counts = rview[rview["assigned_identity"].astype(str).str.startswith(ASSIGNED_PREFIX)].groupby("frame")["assigned_identity"].nunique() if not rview.empty else pd.Series(dtype=float)
            expected_cov = float((counts == int(cfg.num_individuals)).mean()) if len(counts) else 0.0
            spikes, transitions, spike_rate = _speed_spike_rate(rview, threshold=float(cfg.max_speed_px_per_frame))
            metric_rows.append(
                {
                    "method": method,
                    "view": view,
                    "n_tracklets": int(len(nview)),
                    "n_components": int(aview["component_id"].nunique()) if not aview.empty else 0,
                    "n_selected_edges": int(len(method_edges[method_edges["view"] == view])) if not method_edges.empty else 0,
                    "n_assigned_ids": int(aview[aview["assigned_identity"].astype(str).str.startswith(ASSIGNED_PREFIX)]["assigned_identity"].nunique()) if not aview.empty else 0,
                    "tracklet_assigned_fraction": float(aview["is_assigned_id"].mean()) if not aview.empty else 0.0,
                    "row_assigned_fraction": float(assigned.mean()) if len(assigned) else 0.0,
                    "unknown_tracklet_fraction": float(aview["is_unknown"].mean()) if not aview.empty else 0.0,
                    "mix_tracklet_fraction": float(aview["is_mix"].mean()) if not aview.empty else 0.0,
                    "crossing_tracklet_fraction": float(aview["is_crossing"].mean()) if not aview.empty else 0.0,
                    "coverage_vs_expected_rows": float(assigned.sum() / max(expected, 1)) if len(assigned) else 0.0,
                    "expected_id_coverage": expected_cov,
                    "mean_assignment_confidence": float(aview["assignment_confidence"].mean()) if not aview.empty else np.nan,
                    "n_speed_spikes_px": int(spikes),
                    "n_speed_transitions_px": int(transitions),
                    "speed_spike_rate_px": spike_rate,
                }
            )

    metrics = pd.DataFrame(metric_rows)
    ref_metrics = _tracklet_reference_metrics(nodes, assignments)
    if not ref_metrics.empty:
        metrics = metrics.merge(ref_metrics.drop(columns=["best_remap"]), on="method", how="left")

    event_rows = []
    for method in METHODS:
        a = assignments[assignments["method"] == method].merge(
            nodes[["node_id", "frame_start", "frame_end"]], on="node_id", how="left"
        )
        for ev in events.itertuples(index=False):
            for view in sorted(nodes["view"].unique()):
                before = a[
                    (a["view"] == view)
                    & (a["frame_end"] <= int(ev.start))
                    & (a["frame_end"] >= int(ev.start) - int(cfg.max_gap_frames))
                ]
                after = a[
                    (a["view"] == view)
                    & (a["frame_start"] >= int(ev.end))
                    & (a["frame_start"] <= int(ev.end) + int(cfg.max_gap_frames))
                ]
                before_ids = set(before["assigned_identity"].astype(str))
                after_ids = set(after["assigned_identity"].astype(str))
                usable_before = {x for x in before_ids if x.startswith(ASSIGNED_PREFIX)}
                usable_after = {x for x in after_ids if x.startswith(ASSIGNED_PREFIX)}
                common = sorted(usable_before & usable_after)
                if common:
                    decision = "reconnected"
                elif usable_before and usable_after:
                    decision = "conflict"
                else:
                    decision = "unresolved"
                event_rows.append(
                    {
                        "method": method,
                        "view": view,
                        "event_id": getattr(ev, "event_id", ""),
                        "event_kind": getattr(ev, "kind", "event"),
                        "event_start": int(ev.start),
                        "event_end": int(ev.end),
                        "decision": decision,
                        "shared_assigned_ids": ";".join(common),
                        "before_assigned_ids": ";".join(sorted(usable_before)),
                        "after_assigned_ids": ";".join(sorted(usable_after)),
                    }
                )
    event_reconnections = pd.DataFrame(event_rows)

    view_rows = []
    if not per_frame.empty:
        for method, group in per_frame.groupby("method", sort=False):
            by_view_frame: dict[tuple[str, int], set[str]] = {}
            for (view, frame), sub in group.groupby(["view", "frame"], sort=False):
                labels = {
                    str(x)
                    for x in sub["assigned_identity"]
                    if str(x).startswith(ASSIGNED_PREFIX)
                }
                if labels:
                    by_view_frame[(str(view), int(frame))] = labels
            frames = sorted({frame for _, frame in by_view_frame})
            agree = 0
            comparable = 0
            for frame in frames:
                sets = [labels for (view, f), labels in by_view_frame.items() if f == frame]
                if len(sets) < 2:
                    continue
                comparable += 1
                agree += int(all(s == sets[0] for s in sets[1:]))
            view_rows.append(
                {
                    "method": method,
                    "comparable_frames": comparable,
                    "view_identity_set_agreement_fraction": float(agree / comparable) if comparable else np.nan,
                }
            )
    view_agreement = pd.DataFrame(view_rows)
    if not view_agreement.empty:
        metrics = metrics.merge(view_agreement, on="method", how="left")

    return metrics, event_reconnections, ref_metrics


def _contiguous_true_runs(mask: np.ndarray, frames: np.ndarray, *, merge_gap: int = 0) -> list[tuple[int, int]]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    runs = []
    s = p = int(idx[0])
    for i_raw in idx[1:]:
        i = int(i_raw)
        if int(frames[i]) - int(frames[p]) <= merge_gap + 1:
            p = i
        else:
            runs.append((int(frames[s]), int(frames[p])))
            s = p = i
    runs.append((int(frames[s]), int(frames[p])))
    return runs


def detect_events_from_3d(csv_3d: Path | None, cfg: PilotConfig) -> pd.DataFrame:
    if csv_3d is None:
        return _empty_events()
    df = read_table(csv_3d)
    required = {"frame", "fish_id", "X", "Y", "Z"}
    if required - set(df.columns):
        return _empty_events()
    df = df.copy()
    df["_fish_id_str"] = df["fish_id"].astype(str)
    piv = df.pivot_table(index="frame", columns="_fish_id_str", values=["X", "Y", "Z"], aggfunc="mean")
    frames = piv.index.to_numpy(dtype=int)
    ids = sorted(df["_fish_id_str"].dropna().astype(str).unique())
    events = []
    event_id = 0
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            try:
                pa = np.stack([piv[("X", a)].to_numpy(), piv[("Y", a)].to_numpy(), piv[("Z", a)].to_numpy()], axis=1)
                pb = np.stack([piv[("X", b)].to_numpy(), piv[("Y", b)].to_numpy(), piv[("Z", b)].to_numpy()], axis=1)
            except KeyError:
                continue
            d = np.linalg.norm(pa - pb, axis=1)
            mask = np.isfinite(d) & (d <= float(cfg.hardcore_mm))
            for start, end in _contiguous_true_runs(mask, frames, merge_gap=5):
                seg = d[(frames >= start) & (frames <= end)]
                events.append(
                    {
                        "event_id": f"3d_close_{event_id:05d}",
                        "kind": "3d_close_contact",
                        "pair": f"{a}|{b}",
                        "start": int(start),
                        "end": int(end),
                        "n_frames": int(end - start + 1),
                        "min_distance": float(np.nanmin(seg)) if len(seg) else np.nan,
                        "source": str(csv_3d),
                    }
                )
                event_id += 1
    return pd.DataFrame(events) if events else _empty_events()


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=["event_id", "kind", "pair", "start", "end", "n_frames", "min_distance", "source"])


def load_events(events_csv: Path | None, csv_3d: Path | None, cfg: PilotConfig) -> pd.DataFrame:
    if events_csv is not None:
        df = read_table(events_csv)
        out = df.copy()
        rename = {}
        for col in ("start_frame", "frame_start"):
            if col in out.columns and "start" not in out.columns:
                rename[col] = "start"
        for col in ("end_frame", "frame_end"):
            if col in out.columns and "end" not in out.columns:
                rename[col] = "end"
        out = out.rename(columns=rename)
        if "start" not in out.columns or "end" not in out.columns:
            raise ValueError("events table needs start/end or start_frame/end_frame columns")
        if "event_id" not in out.columns:
            out["event_id"] = [f"event_{i:05d}" for i in range(len(out))]
        if "kind" not in out.columns:
            out["kind"] = "manual_event"
        if "pair" not in out.columns:
            out["pair"] = ""
        if "min_distance" not in out.columns:
            out["min_distance"] = np.nan
        out["start"] = pd.to_numeric(out["start"], errors="coerce").astype(int)
        out["end"] = pd.to_numeric(out["end"], errors="coerce").astype(int)
        out["n_frames"] = out["end"] - out["start"] + 1
        out["source"] = str(events_csv)
        return out[["event_id", "kind", "pair", "start", "end", "n_frames", "min_distance", "source"]].sort_values("start")
    return detect_events_from_3d(csv_3d, cfg)


def suggest_windows(
    *,
    tracklets: pd.DataFrame | None,
    csv_3d: Path | None,
    events: pd.DataFrame,
    cfg: PilotConfig,
    n: int,
    window_size: int,
) -> pd.DataFrame:
    rows = []
    seen: set[tuple[int, int, str]] = set()

    def add(kind: str, start: int, end: int, reason: str, source: str) -> None:
        start = max(0, int(start))
        end = int(end)
        key = (start, end, kind)
        if key in seen or end < start:
            return
        seen.add(key)
        rows.append(
            {
                "window_id": f"{kind}_{start}_{end}",
                "kind": kind,
                "start": start,
                "end": end,
                "source": source,
                "reason": reason,
            }
        )

    if tracklets is not None and not tracklets.empty:
        fmin = int(tracklets["frame"].min())
        add("early", fmin, fmin + window_size - 1, "first available tracklet frames", "tracklets")

    for ev in events.head(max(n, 1)).itertuples(index=False):
        add(
            "crossing_or_occlusion",
            int(ev.start) - int(cfg.event_pad_frames),
            int(ev.end) + int(cfg.event_pad_frames),
            f"{getattr(ev, 'kind', 'event')} {getattr(ev, 'pair', '')}".strip(),
            "events",
        )

    if csv_3d is not None and Path(csv_3d).exists():
        try:
            df = read_table(csv_3d)
            required = {"frame", "fish_id", "X", "Y", "Z"}
            if not (required - set(df.columns)):
                df = df.copy()
                df["_fish_id_str"] = df["fish_id"].astype(str)
                piv = df.pivot_table(index="frame", columns="_fish_id_str", values=["X", "Y", "Z"], aggfunc="mean")
                frames = piv.index.to_numpy(dtype=int)
                ids = sorted(df["_fish_id_str"].dropna().astype(str).unique())
                min_dist = np.full(len(frames), np.inf)
                for i, a in enumerate(ids):
                    for b in ids[i + 1:]:
                        try:
                            pa = np.stack([piv[("X", a)].to_numpy(), piv[("Y", a)].to_numpy(), piv[("Z", a)].to_numpy()], axis=1)
                            pb = np.stack([piv[("X", b)].to_numpy(), piv[("Y", b)].to_numpy(), piv[("Z", b)].to_numpy()], axis=1)
                        except KeyError:
                            continue
                        d = np.linalg.norm(pa - pb, axis=1)
                        min_dist = np.minimum(min_dist, np.where(np.isfinite(d), d, np.inf))
                easy = np.isfinite(min_dist) & (min_dist >= float(cfg.easy_separation_mm))
                for start, end in _contiguous_true_runs(easy, frames, merge_gap=3):
                    if end - start + 1 >= max(50, window_size // 2):
                        mid = (start + end) // 2
                        add("easy_separated", mid - window_size // 2, mid + window_size // 2, f"min 3D separation >= {cfg.easy_separation_mm:g} mm", str(csv_3d))
                        break
                for _, group in df.sort_values("frame").groupby("fish_id", sort=False):
                    xyz = group[["X", "Y", "Z"]].to_numpy(dtype=float)
                    frames_g = group["frame"].to_numpy(dtype=int)
                    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
                    consec = np.diff(frames_g) == 1
                    spike_idx = np.where(consec & (speed > float(cfg.max_3d_speed_mm_per_frame)))[0]
                    if len(spike_idx):
                        f = int(frames_g[spike_idx[0] + 1])
                        add("speed_spike", f - window_size // 2, f + window_size // 2, "3D speed spike", str(csv_3d))
                        break
        except Exception as exc:
            logging.getLogger(__name__).warning("could not suggest 3D windows: %s", exc)

    return pd.DataFrame(rows).head(n) if rows else parse_windows([])


def evaluate_3d(csv_3d: Path | None) -> dict:
    if csv_3d is None or not Path(csv_3d).exists():
        return {"available": False}
    df = read_table(csv_3d)
    if {"frame", "X", "Y", "Z"} - set(df.columns):
        return {"available": False, "reason": "missing frame/X/Y/Z"}
    id_col = "final_fish_id" if "final_fish_id" in df.columns else "fish_id" if "fish_id" in df.columns else ""
    if not id_col:
        return {"available": False, "reason": "missing fish_id/final_fish_id"}
    out = spike_metrics(df, id_col=id_col)
    out["available"] = True
    out["id_col"] = id_col
    out["csv"] = str(csv_3d)
    return out


def _json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if not isinstance(obj, (str, bytes)):
        try:
            missing = pd.isna(obj)
            if isinstance(missing, (bool, np.bool_)) and missing:
                return None
        except Exception:
            pass
    return obj


def write_outputs(
    *,
    out_dir: Path,
    config: dict,
    windows: pd.DataFrame,
    events: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    assignments: pd.DataFrame,
    selected_edges: pd.DataFrame,
    per_frame: pd.DataFrame,
    metrics: pd.DataFrame,
    event_reconnections: pd.DataFrame,
    reference_metrics: pd.DataFrame,
    summary: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.json").write_text(json.dumps(_json_safe(config), indent=2) + "\n", encoding="utf-8")
    windows.to_csv(out_dir / "pilot_windows.csv", index=False)
    events.to_csv(out_dir / "event_windows.csv", index=False)
    nodes.to_csv(out_dir / "tracklet_nodes.csv", index=False)
    edges.to_csv(out_dir / "tracklet_edges.csv", index=False)
    assignments.to_csv(out_dir / "method_assignments.csv", index=False)
    selected_edges.to_csv(out_dir / "selected_edges.csv", index=False)
    per_frame.to_csv(out_dir / "per_frame_assignments.csv", index=False)
    metrics.to_csv(out_dir / "metrics_summary.csv", index=False)
    event_reconnections.to_csv(out_dir / "event_reconnections.csv", index=False)
    reference_metrics.to_csv(out_dir / "reference_metrics.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8")
    write_recommendation(out_dir, summary, metrics)


def _method_score(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    grouped = metrics.groupby("method", sort=False).agg(
        row_assigned_fraction=("row_assigned_fraction", "mean"),
        expected_id_coverage=("expected_id_coverage", "mean"),
        speed_spike_rate_px=("speed_spike_rate_px", "mean"),
        unknown_tracklet_fraction=("unknown_tracklet_fraction", "mean"),
        mix_tracklet_fraction=("mix_tracklet_fraction", "mean"),
        crossing_tracklet_fraction=("crossing_tracklet_fraction", "mean"),
        best_remapped_tracklet_accuracy=("best_remapped_tracklet_accuracy", "mean"),
    ).reset_index()
    acc = grouped["best_remapped_tracklet_accuracy"].fillna(0.0)
    grouped["pilot_score"] = (
        0.35 * grouped["row_assigned_fraction"].fillna(0.0)
        + 0.20 * grouped["expected_id_coverage"].fillna(0.0)
        + 0.25 * acc
        - 0.10 * grouped["speed_spike_rate_px"].fillna(0.0)
        - 0.05 * grouped["unknown_tracklet_fraction"].fillna(0.0)
        - 0.05 * grouped["mix_tracklet_fraction"].fillna(0.0)
    )
    return grouped.sort_values("pilot_score", ascending=False)


def write_recommendation(out_dir: Path, summary: dict, metrics: pd.DataFrame) -> None:
    ranked = _method_score(metrics)
    if ranked.empty:
        best = "undetermined"
        best_line = "No comparable metrics were produced."
    else:
        best = str(ranked.iloc[0]["method"])
        best_line = f"Best pilot method by transparent proxy score: `{best}`."
    lines = [
        "# Tracklet identity pilot recommendation",
        "",
        best_line,
        "",
        "This pilot compares method families on the selected small windows only. Treat the result as a triage signal, not a production claim.",
        "",
        "## Method ranking",
        "",
    ]
    if not ranked.empty:
        lines.extend(_markdown_table(ranked))
        lines.append("")
    lines.extend(
        [
            "## Baseline notes",
            "",
            "- `idtracker.ai v2`: not launched by this pilot. Direct integration requires raw top-camera video, animal-count setup, and a separate long-running environment. Use `--idtracker-csv` in a later scoring pass once an idtracker export exists.",
            "- `UDMT`: represented here by a reproducible UDMT-inspired crossing/temporal baseline that flags hard events and avoids forced labels through uncertain close-contact spans.",
            "",
            "## Next annotation if the pilot is weak",
            "",
            "- Annotate the event windows in `event_windows.csv` first, especially rows that are unresolved or conflicting in `event_reconnections.csv`.",
            "- Preferred label type: tracklet-pair same/different labels around before/after event fragments, plus crossing/mix labels for frames inside hard events.",
            "- If appearance methods are weak, add segmentation masks or fish-ID labels for 20-50 clean crops per fish per view, plus 10-20 hard-event examples.",
        ]
    )
    (out_dir / "recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for record in df.to_dict("records"):
        vals = []
        for col in cols:
            val = record.get(col, "")
            if isinstance(val, float):
                vals.append("" if not np.isfinite(val) else f"{val:.4g}")
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return rows


def _setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "pilot.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


def _plot_timeline(nodes: pd.DataFrame, assignments: pd.DataFrame, out_dir: Path) -> None:
    if nodes.empty or assignments.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib unavailable; skipping timeline plots")
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    palette = {
        "ID_1": "#1f77b4",
        "ID_2": "#ff7f0e",
        "ID_3": "#2ca02c",
        "unknown": "#9e9e9e",
        "mix": "#d62728",
        "crossing": "#9467bd",
    }
    base = nodes.merge(assignments, on=["node_id", "view", "tracklet_id"], how="inner")
    for method, group in base.groupby("method", sort=False):
        ordered = group.sort_values(["view", "frame_start", "tracklet_id"]).reset_index(drop=True)
        fig_h = max(4, min(16, 0.22 * len(ordered) + 2))
        fig, ax = plt.subplots(figsize=(14, fig_h))
        y_labels = []
        y_pos = []
        for yi, row in enumerate(ordered.itertuples(index=False)):
            label = str(row.assigned_identity)
            color = palette.get(label, "#17becf")
            ax.broken_barh([(int(row.frame_start), int(row.frame_end) - int(row.frame_start) + 1)], (yi - 0.35, 0.7), facecolors=color)
            y_labels.append(f"{row.view}:{row.tracklet_id}")
            y_pos.append(yi)
        ax.set_yticks(y_pos[:: max(1, len(y_pos) // 30)])
        ax.set_yticklabels(y_labels[:: max(1, len(y_labels) // 30)], fontsize=7)
        ax.set_xlabel("frame")
        ax.set_ylabel("tracklet")
        ax.set_title(f"Tracklet timeline colored by {method}")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=k) for k, c in palette.items()]
        ax.legend(handles=handles, loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"timeline_{method}.png", dpi=150)
        plt.close(fig)


def _plot_edges(edges: pd.DataFrame, selected_edges: pd.DataFrame, out_dir: Path) -> None:
    if edges.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib unavailable; skipping edge plots")
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(edges["gap_frames"], edges["speed_px_per_frame"], c=edges["hybrid_score"], cmap="viridis", s=18, alpha=0.75)
    if not selected_edges.empty:
        hy = selected_edges[selected_edges["method"] == "hybrid"]
        if not hy.empty:
            ax.scatter(hy["gap_frames"], hy["speed_px_per_frame"], facecolors="none", edgecolors="red", s=50, label="hybrid selected")
    ax.set_xlabel("gap frames")
    ax.set_ylabel("endpoint speed (px/frame)")
    ax.set_title("Tracklet edge diagnostics")
    fig.colorbar(sc, ax=ax, label="hybrid score")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "edge_scores.png", dpi=150)
    plt.close(fig)


def _plot_3d(csv_3d: Path | None, out_dir: Path) -> None:
    if csv_3d is None or not Path(csv_3d).exists():
        return
    df = read_table(csv_3d)
    if {"frame", "X", "Y", "Z"} - set(df.columns):
        return
    id_col = "final_fish_id" if "final_fish_id" in df.columns else "fish_id" if "fish_id" in df.columns else None
    if id_col is None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib unavailable; skipping 3D plot")
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for identity, group in df.sort_values("frame").groupby(id_col, sort=True):
        sub = group[np.isfinite(group[["X", "Y", "Z"]]).all(axis=1)]
        if sub.empty:
            continue
        step = max(1, len(sub) // 5000)
        sub = sub.iloc[::step]
        ax.plot(sub["X"], sub["Y"], sub["Z"], lw=0.7, label=str(identity))
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"3D trajectory colored by {id_col}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "trajectory_3d.png", dpi=150)
    plt.close(fig)


def _plot_confusion(nodes: pd.DataFrame, assignments: pd.DataFrame, out_dir: Path) -> None:
    merged = assignments.merge(nodes[["node_id", "reference_label"]], on="node_id", how="left")
    merged = merged[
        merged["assigned_identity"].astype(str).str.startswith(ASSIGNED_PREFIX)
        & merged["reference_label"].fillna("").astype(str).ne("")
    ]
    if merged.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for method, group in merged.groupby("method", sort=False):
        table = pd.crosstab(group["assigned_identity"], group["reference_label"])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(table.to_numpy(), cmap="Blues")
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels(table.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels(table.index)
        ax.set_xlabel("reference")
        ax.set_ylabel("assigned")
        ax.set_title(f"Confusion: {method}")
        for (i, j), val in np.ndenumerate(table.to_numpy()):
            ax.text(j, i, str(int(val)), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(plot_dir / f"confusion_{method}.png", dpi=150)
        plt.close(fig)


def _plot_event_panels(per_frame: pd.DataFrame, events: pd.DataFrame, out_dir: Path, cfg: PilotConfig) -> None:
    if per_frame.empty or events.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plot_dir = out_dir / "plots" / "event_panels"
    plot_dir.mkdir(parents=True, exist_ok=True)
    palette = {"ID_1": "#1f77b4", "ID_2": "#ff7f0e", "ID_3": "#2ca02c", "unknown": "#9e9e9e", "mix": "#d62728", "crossing": "#9467bd"}
    for ev in events.head(int(cfg.max_plot_events)).itertuples(index=False):
        lo = int(ev.start) - int(cfg.event_pad_frames)
        hi = int(ev.end) + int(cfg.event_pad_frames)
        for method in METHODS:
            sub = per_frame[(per_frame["method"] == method) & per_frame["frame"].between(lo, hi)]
            if sub.empty:
                continue
            views = sorted(sub["view"].unique())
            fig, axs = plt.subplots(1, len(views), figsize=(5 * len(views), 4), squeeze=False)
            for ax, view in zip(axs[0], views):
                v = sub[sub["view"] == view]
                for label, group in v.groupby("assigned_identity", sort=False):
                    ax.scatter(group["x"], group["y"], s=8, alpha=0.75, color=palette.get(str(label), "#17becf"), label=str(label))
                ax.set_title(f"{view}")
                ax.set_aspect("equal", adjustable="datalim")
                ax.invert_yaxis()
            axs[0, -1].legend(fontsize=7, loc="best")
            fig.suptitle(f"{method}: event {getattr(ev, 'event_id', '')} frames {lo}-{hi}")
            fig.tight_layout()
            fig.savefig(plot_dir / f"event_{getattr(ev, 'event_id', 'event')}_{method}.png", dpi=150)
            plt.close(fig)


def write_plots(
    *,
    out_dir: Path,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    selected_edges: pd.DataFrame,
    assignments: pd.DataFrame,
    per_frame: pd.DataFrame,
    events: pd.DataFrame,
    csv_3d: Path | None,
    cfg: PilotConfig,
) -> None:
    _plot_timeline(nodes, assignments, out_dir)
    _plot_edges(edges, selected_edges, out_dir)
    _plot_3d(csv_3d, out_dir)
    _plot_confusion(nodes, assignments, out_dir)
    _plot_event_panels(per_frame, events, out_dir, cfg)


def render_event_videos(
    *,
    per_frame: pd.DataFrame,
    events: pd.DataFrame,
    video_specs: list[str],
    out_dir: Path,
    method: str,
    cfg: PilotConfig,
    fps: float,
    max_videos: int,
) -> pd.DataFrame:
    rows = []
    video_map = dict(parse_key_path(v) for v in video_specs)
    manifest_path = out_dir / "videos" / "video_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not video_map:
        rows.append({"status": "skipped", "reason": "no --video specs", "path": ""})
        out = pd.DataFrame(rows)
        out.to_csv(manifest_path, index=False)
        return out
    try:
        import cv2
    except ImportError:
        rows.append({"status": "skipped", "reason": "opencv unavailable", "path": ""})
        out = pd.DataFrame(rows)
        out.to_csv(manifest_path, index=False)
        return out

    colors = {"ID_1": (255, 80, 30), "ID_2": (30, 160, 255), "ID_3": (60, 200, 80), "unknown": (180, 180, 180), "mix": (40, 40, 220), "crossing": (180, 80, 180)}
    made = 0
    for view, path in video_map.items():
        if made >= max_videos:
            break
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            rows.append({"status": "skipped", "view": view, "reason": "could not open video", "path": str(path)})
            continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        for ev in events.head(max_videos).itertuples(index=False):
            if made >= max_videos:
                break
            start = max(0, int(ev.start) - int(cfg.event_pad_frames))
            end = int(ev.end) + int(cfg.event_pad_frames)
            out_path = out_dir / "videos" / f"{method}_{view}_{getattr(ev, 'event_id', made)}.mp4"
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                rows.append({"status": "skipped", "view": view, "reason": "could not open writer", "path": str(out_path)})
                continue
            frame_rows = per_frame[(per_frame["method"] == method) & (per_frame["view"] == view)]
            lookup = {int(f): g for f, g in frame_rows.groupby("frame", sort=False)}
            for frame_idx in range(start, end + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    break
                for row in lookup.get(frame_idx, pd.DataFrame()).itertuples(index=False):
                    label = str(row.assigned_identity)
                    color = colors.get(label, (240, 240, 0))
                    x = int(round(float(row.x)))
                    y = int(round(float(row.y)))
                    cv2.circle(frame, (x, y), 6, color, 2)
                    cv2.putText(frame, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                cv2.putText(frame, f"{method} {view} frame {frame_idx}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(frame)
            writer.release()
            rows.append({"status": "written", "view": view, "event_id": getattr(ev, "event_id", made), "start": start, "end": end, "path": str(out_path)})
            made += 1
        cap.release()
    out = pd.DataFrame(rows)
    out.to_csv(manifest_path, index=False)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pilot comparison of tracklet identity grouping methods.")
    p.add_argument("--tracklets", action="append", default=[], help="Tracklet table as view=path. Can be repeated.")
    p.add_argument("--csv-3d", type=Path, default=None, help="Optional *_3d.csv or *_3d_identified.csv for events and 3D QC.")
    p.add_argument("--appearance", type=Path, default=None, help="Optional crop/mask ID predictions with tracklet_id.")
    p.add_argument("--events-csv", type=Path, default=None, help="Optional event table with start/end columns.")
    p.add_argument("--idtracker-csv", type=Path, default=None, help="Optional future idtracker.ai export. Recorded in config; direct scoring is not part of the first pilot.")
    p.add_argument("--windows", action="append", default=[], help="Frame window START-END. Repeat for multiple small windows.")
    p.add_argument("--window-table", type=Path, default=None, help="Optional CSV/parquet with start/end window columns.")
    p.add_argument("--auto-windows", type=int, default=0, help="Use suggested windows instead of a full run.")
    p.add_argument("--window-size", type=int, default=5000)
    p.add_argument("--allow-full-run", action="store_true", help="Permit running over every input frame when no windows are provided.")
    p.add_argument("--dry-run", action="store_true", help="Only inspect inputs and write suggested windows/config.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--paths-config", default=None)
    p.add_argument("--num-individuals", type=int, default=3)
    p.add_argument("--max-gap-frames", type=int, default=500)
    p.add_argument("--max-speed-px-per-frame", type=float, default=12.0)
    p.add_argument("--max-3d-speed-mm-per-frame", type=float, default=10.0)
    p.add_argument("--hardcore-mm", type=float, default=12.0)
    p.add_argument("--easy-separation-mm", type=float, default=40.0)
    p.add_argument("--geometry-threshold", type=float, default=0.60)
    p.add_argument("--appearance-threshold", type=float, default=0.60)
    p.add_argument("--temporal-threshold", type=float, default=0.35)
    p.add_argument("--hybrid-threshold", type=float, default=0.55)
    p.add_argument("--udmt-threshold", type=float, default=0.35)
    p.add_argument("--render-videos", action="store_true")
    p.add_argument("--video", action="append", default=[], help="Optional video as view=path for annotated event clips.")
    p.add_argument("--video-method", default="hybrid", choices=METHODS)
    p.add_argument("--video-fps", type=float, default=20.0)
    p.add_argument("--max-videos", type=int, default=3)
    p.add_argument("--force", action="store_true", help="Overwrite an existing completed output directory.")
    return p


def _windows_from_table(path: Path) -> pd.DataFrame:
    df = read_table(path)
    rename = {}
    for col in ("start_frame", "frame_start"):
        if col in df.columns and "start" not in df.columns:
            rename[col] = "start"
    for col in ("end_frame", "frame_end"):
        if col in df.columns and "end" not in df.columns:
            rename[col] = "end"
    df = df.rename(columns=rename)
    if "start" not in df.columns or "end" not in df.columns:
        raise ValueError("window table needs start/end columns")
    if "window_id" not in df.columns:
        df["window_id"] = [f"w{i:03d}" for i in range(len(df))]
    if "kind" not in df.columns:
        df["kind"] = "table"
    if "source" not in df.columns:
        df["source"] = str(path)
    if "reason" not in df.columns:
        df["reason"] = ""
    return df[["window_id", "kind", "start", "end", "source", "reason"]].copy()


def _default_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir is not None:
        return args.out_dir
    local = load_local_paths(args.paths_config)
    root = Path(choose_path(local.runs_dir if local else None, "runs") or "runs")
    return root / "tracklet_identity_pilot"


def _config_from_args(args: argparse.Namespace) -> PilotConfig:
    return PilotConfig(
        num_individuals=args.num_individuals,
        max_gap_frames=args.max_gap_frames,
        max_speed_px_per_frame=args.max_speed_px_per_frame,
        max_3d_speed_mm_per_frame=args.max_3d_speed_mm_per_frame,
        geometry_threshold=args.geometry_threshold,
        appearance_threshold=args.appearance_threshold,
        temporal_threshold=args.temporal_threshold,
        hybrid_threshold=args.hybrid_threshold,
        udmt_threshold=args.udmt_threshold,
        hardcore_mm=args.hardcore_mm,
        easy_separation_mm=args.easy_separation_mm,
    )


def run(args: argparse.Namespace) -> dict:
    cfg = _config_from_args(args)
    out_dir = _default_out_dir(args)
    done_marker = out_dir / "summary.json"
    if done_marker.exists() and not args.force and not args.dry_run:
        raise SystemExit(f"{done_marker} already exists; pass --force to rerun")
    _setup_logging(out_dir)
    logging.info("starting tracklet identity pilot")

    windows = parse_windows(args.windows)
    if args.window_table is not None:
        windows = pd.concat([windows, _windows_from_table(args.window_table)], ignore_index=True)

    events = load_events(args.events_csv, args.csv_3d, cfg)

    inspect_tracklets = None
    if args.tracklets:
        # For dry-run/autowindow inspection before final filtering, load all rows
        # by using an empty window table.
        inspect_tracklets = load_tracklets(args.tracklets, parse_windows([]))

    if args.auto_windows:
        suggested = suggest_windows(
            tracklets=inspect_tracklets,
            csv_3d=args.csv_3d,
            events=events,
            cfg=cfg,
            n=int(args.auto_windows),
            window_size=int(args.window_size),
        )
        windows = pd.concat([windows, suggested], ignore_index=True)

    if windows.empty and not args.allow_full_run and not args.dry_run:
        raise SystemExit("Refusing an implicit full run. Pass --windows START-END, --auto-windows N, or --allow-full-run.")

    config = {
        "args": vars(args),
        "pilot_config": asdict(cfg),
        "methods": METHODS,
        "idtracker_ai_v2": {
            "requested_csv": str(args.idtracker_csv) if args.idtracker_csv else None,
            "status": "external_baseline_placeholder",
            "reason": "direct idtracker.ai v2 execution is a separate raw-video job; this pilot records/scans artifacts only",
        },
    }

    if args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config_used.json").write_text(json.dumps(_json_safe(config), indent=2) + "\n", encoding="utf-8")
        windows.to_csv(out_dir / "pilot_windows.csv", index=False)
        events.to_csv(out_dir / "event_windows.csv", index=False)
        summary = {
            "dry_run": True,
            "n_tracklet_rows_available": int(len(inspect_tracklets)) if inspect_tracklets is not None else 0,
            "n_events": int(len(events)),
            "n_windows": int(len(windows)),
            "outputs": {
                "config": str(out_dir / "config_used.json"),
                "windows": str(out_dir / "pilot_windows.csv"),
                "events": str(out_dir / "event_windows.csv"),
            },
        }
        (out_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8")
        logging.info("dry-run complete")
        return summary

    tracklets = load_tracklets(args.tracklets, windows)
    appearance = read_table(args.appearance) if args.appearance is not None else None
    nodes = build_nodes(tracklets, events, appearance)
    edges = build_edges(nodes, events, cfg)
    assignments, selected_edges = assign_methods(nodes, edges, cfg)
    per_frame = expand_per_frame(tracklets, assignments)
    metrics, event_reconnections, reference_metrics = summarize_metrics(
        nodes, assignments, per_frame, selected_edges, events, cfg
    )
    metrics_ranked = _method_score(metrics)
    three_d_metrics = evaluate_3d(args.csv_3d)
    summary = {
        "dry_run": False,
        "n_tracklet_rows": int(len(tracklets)),
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "n_selected_edges": int(len(selected_edges)),
        "n_events": int(len(events)),
        "n_windows": int(len(windows)),
        "methods": METHODS,
        "best_method_so_far": str(metrics_ranked.iloc[0]["method"]) if not metrics_ranked.empty else None,
        "method_ranking": metrics_ranked.to_dict("records") if not metrics_ranked.empty else [],
        "three_d_metrics": three_d_metrics,
        "output_dir": str(out_dir),
    }
    write_outputs(
        out_dir=out_dir,
        config=config,
        windows=windows,
        events=events,
        nodes=nodes,
        edges=edges,
        assignments=assignments,
        selected_edges=selected_edges,
        per_frame=per_frame,
        metrics=metrics,
        event_reconnections=event_reconnections,
        reference_metrics=reference_metrics,
        summary=summary,
    )
    write_plots(
        out_dir=out_dir,
        nodes=nodes,
        edges=edges,
        selected_edges=selected_edges,
        assignments=assignments,
        per_frame=per_frame,
        events=events,
        csv_3d=args.csv_3d,
        cfg=cfg,
    )
    if args.render_videos:
        render_event_videos(
            per_frame=per_frame,
            events=events,
            video_specs=args.video,
            out_dir=out_dir,
            method=args.video_method,
            cfg=cfg,
            fps=args.video_fps,
            max_videos=args.max_videos,
        )
    else:
        video_dir = out_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"status": "skipped", "reason": "use --render-videos with --video view=path", "path": ""}]).to_csv(
            video_dir / "video_manifest.csv", index=False
        )
    logging.info("pilot complete: %s", out_dir)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(_json_safe(summary), indent=2))


if __name__ == "__main__":
    main()
