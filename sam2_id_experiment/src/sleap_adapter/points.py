from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_sleap_points(
    slp_path: str | Path,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    anchor_node: str | None = None,
) -> pd.DataFrame:
    """Load one top-camera SLEAP point per visible instance.

    The point is the requested anchor node when available, otherwise the
    centroid of all finite nodes in the instance. Production tracking should
    pass ``anchor_node="mouth"``; centroid mode is kept for explicit comparison
    experiments. Frame numbers are preserved exactly as stored in the `.slp`
    file.
    """
    try:
        import sleap_io as sio
    except ImportError as exc:
        raise RuntimeError(
            "sleap-io is required to load .slp files. Run `uv sync --extra sleap` "
            "or use an environment with sleap-io installed."
        ) from exc

    labels = sio.load_slp(str(slp_path))
    node_names = [node.name for node in labels.skeletons[0].nodes] if labels.skeletons else []
    rows: list[dict] = []
    for lf in labels.labeled_frames:
        frame = int(lf.frame_idx)
        if start_frame is not None and frame < int(start_frame):
            continue
        if end_frame is not None and frame > int(end_frame):
            continue
        for instance_index, inst in enumerate(lf.instances):
            point = _instance_point(inst, node_names=node_names, anchor_node=anchor_node)
            if point is None:
                continue
            x, y, node_name = point
            score = getattr(inst, "score", None)
            rows.append({
                "frame": frame,
                "sleap_index": int(instance_index),
                "x": float(x),
                "y": float(y),
                "score": float(score) if score is not None else np.nan,
                "node": node_name,
            })
    return pd.DataFrame(rows, columns=["frame", "sleap_index", "x", "y", "score", "node"])


def _instance_point(inst, *, node_names: list[str], anchor_node: str | None) -> tuple[float, float, str] | None:
    pts = inst.numpy()
    if anchor_node is not None and anchor_node in node_names:
        idx = node_names.index(anchor_node)
        pt = pts[idx]
        if np.isfinite(pt).all():
            return float(pt[0]), float(pt[1]), anchor_node

    mask = np.isfinite(pts).all(axis=1)
    if not mask.any():
        return None
    centroid = pts[mask].mean(axis=0)
    return float(centroid[0]), float(centroid[1]), "centroid"
