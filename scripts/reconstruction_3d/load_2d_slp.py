from __future__ import annotations

"""
Load per-frame 2D points directly from three merged SLEAP `.slp` files.

Each merged file holds the raw, untracked predictions from one camera (top_cam /
side1_cam / side2_cam). This module converts them into the same
`points_by_frame[frame][cam] = [(x, y), ...]` shape the rest of the 3D
reconstruction pipeline expects.

Per-frame instance counts can vary from 0 to the model's `max_instances`; the
number of "slots" we keep per view is configurable via `num_individuals`.
Extra detections (beyond `num_individuals`) are dropped after sorting by
descending confidence score, and shortfalls are padded with NaN so the
downstream permutation enumeration always has N candidates per view.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


VIEW_TO_CAM = {"top_cam": "cam0", "side1_cam": "cam1", "side2_cam": "cam2"}


def _instance_point(
    inst,
    *,
    node_names: list[str] | None = None,
    anchor_node: str | None = None,
) -> tuple[float, float, float]:
    """Return (x, y, score) for an instance. x/y are NaN if nothing visible."""
    pts = inst.numpy()  # (n_nodes, 2)
    if anchor_node is not None and node_names is not None and anchor_node in node_names:
        pt = pts[node_names.index(anchor_node)]
        score = getattr(inst, "score", None)
        if np.isfinite(pt).all():
            return float(pt[0]), float(pt[1]), float(score) if score is not None else np.nan
    mask = np.isfinite(pts).all(axis=1)
    if not mask.any():
        return np.nan, np.nan, float(getattr(inst, "score", np.nan) or np.nan)
    centroid = pts[mask].mean(axis=0)
    score = getattr(inst, "score", None)
    return float(centroid[0]), float(centroid[1]), float(score) if score is not None else np.nan


def load_points_by_frame(
    slp_paths: Dict[str, Path | str],
    *,
    num_individuals: int = 3,
    num_candidates: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    anchor_node: str | None = None,
) -> Dict[int, Dict[str, List[Tuple[float, float]]]]:
    """
    Parameters
    ----------
    slp_paths : dict
        Mapping from view name (`top_cam`, `side1_cam`, `side2_cam`) to merged
        `.slp` path.
    num_individuals : int
        Number of fish slots to keep per view (when ``num_candidates`` is None).
        Extra detections are dropped (highest-score kept).
    num_candidates : int, optional
        Override the per-view slot count. Use this when downstream
        triangulation enumerates more candidates per view than there are real
        fish (e.g. ``-n 6`` SLEAP inference + reflection-rejecting 3D scoring).
        When None, defaults to ``num_individuals`` (the pre-existing behaviour).
    start_frame, end_frame : int, optional
        Inclusive frame-index filter.
    anchor_node : str, optional
        If provided, use this skeleton node's xy for each instance instead of
        the centroid of visible nodes. For the current medaka mouth-only SLEAP
        model, pass ``"mouth"`` to make the data contract explicit.

    Returns
    -------
    points_by_frame : dict
        `{frame_idx: {"cam0": [(x, y), ...], "cam1": ..., "cam2": ...}}` with
        ``num_candidates or num_individuals`` entries per view. Missing
        detections are ``(NaN, NaN)``.
    """
    slots = num_candidates if num_candidates is not None else num_individuals
    try:
        import sleap_io as sio
    except ImportError as exc:
        raise RuntimeError(
            "sleap-io is required. `uv sync --extra sleap` or `pip install sleap-io`."
        ) from exc

    per_view_frames: Dict[str, Dict[int, list]] = {}
    for view, path in slp_paths.items():
        if view not in VIEW_TO_CAM:
            raise ValueError(f"Unknown view {view!r}; expected one of {list(VIEW_TO_CAM)}")
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{view} .slp not found: {path}")
        labels = sio.load_slp(str(path))
        node_names = [node.name for node in labels.skeletons[0].nodes] if labels.skeletons else []
        by_frame: Dict[int, list] = {}
        for f in labels.labeled_frames:
            fi = int(f.frame_idx)
            if start_frame is not None and fi < start_frame:
                continue
            if end_frame is not None and fi > end_frame:
                continue
            pts: list[tuple[float, float, float]] = []
            for inst in f.instances:
                x, y, score = _instance_point(
                    inst,
                    node_names=node_names,
                    anchor_node=anchor_node,
                )
                if np.isfinite(x) and np.isfinite(y):
                    pts.append((x, y, score))
            if pts:
                by_frame[fi] = pts
        per_view_frames[view] = by_frame

    # Gather the union of frame indices across all three views.
    all_frames = sorted({fi for view in per_view_frames.values() for fi in view})

    result: Dict[int, Dict[str, List[Tuple[float, float]]]] = {}
    for fi in all_frames:
        result[fi] = {}
        for view, cam in VIEW_TO_CAM.items():
            raw = per_view_frames.get(view, {}).get(fi, [])
            # Keep the top-N by score, fill the rest with NaN.
            raw_sorted = sorted(
                raw,
                key=lambda r: (-(r[2] if np.isfinite(r[2]) else -np.inf), r[0]),
            )
            keep = raw_sorted[:slots]
            points: list[Tuple[float, float]] = [(x, y) for (x, y, _s) in keep]
            while len(points) < slots:
                points.append((np.nan, np.nan))
            result[fi][cam] = points
    return result
