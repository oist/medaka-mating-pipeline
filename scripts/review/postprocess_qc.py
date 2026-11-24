"""Quality checks of the correction steps, from ``versions.npz`` (:mod:`scripts.review.versions`).

* :func:`jitter_table` - per version and node: share of analysed fish-frames off their 9-frame
  median by > 2 mm, with acceleration > 3 mm/frame^2, and the largest single-frame step.
* :func:`change_breakdown` - where the post-pass replaced values (``node_source`` 5): by swimming
  speed, by distance to the nearest fish and by BORIS behaviour. Corrections concentrate where fish
  are close, i.e. during interactions; compare the analysis on measured frames only.
* :func:`measured_share` - share of fish-frames whose nodes were all triangulated (source 0), by
  distance to the nearest fish: the input of that sensitivity analysis.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.review.postprocess import NODE_BRIDGED
from scripts.review.reconstruct import rolling_median
from scripts.review.versions import VERSIONS, VERSIONS_FILE

SPEED_BINS = (0, 25, 50, 100, 200, 300, np.inf)        # mm/s
DISTANCE_BINS = (0, 10, 20, 40, np.inf)                  # mm, pectoral to the nearest fish's pectoral


def load_versions(folder: Path) -> dict:
    with np.load(Path(folder) / VERSIONS_FILE) as z:
        return {k: z[k] for k in z.files}


def _rows(v: dict) -> slice:
    start = int(v["analysis_start"])
    return slice(max(0, start - int(v["frames"][0])) if start >= 0 else 0, None)


def jitter_table(v: dict) -> pd.DataFrame:
    rows = []
    sl = _rows(v)
    names = [str(n) for n in v["node_names"]]
    for i, version in enumerate(VERSIONS):
        if version not in [str(p) for p in v["present"]]:
            continue
        for n, node in enumerate(names):
            dev2, acc3, steps = [], [], []
            for c in range(len(v["labels"])):
                x = v["nodes"][i, sl, c, n].astype(float)
                if not np.isfinite(x).any():
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    dev = np.linalg.norm(x - rolling_median(x, 4), axis=1)
                    acc = np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=1)
                    step = np.linalg.norm(np.diff(x, axis=0), axis=1)
                dev2.append(np.nanmean(dev > 2))
                acc3.append(np.nanmean(acc > 3))
                steps.append(np.nanmax(step))
            rows.append({"version": version, "node": node, "off_median_gt_2mm_%": 100 * float(np.mean(dev2)),
                         "acc_gt_3mm_%": 100 * float(np.mean(acc3)), "max_step_mm": float(np.max(steps))})
    return pd.DataFrame(rows).round(3)


def _speed_and_distance(v: dict, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Final version: ``(F, L)`` speed (mm/s, central difference over +-2 frames) and nearest-fish distance."""
    fin = VERSIONS.index("final")
    pos = v["nodes"][fin, :, :, int(v["position_node"])].astype(float)      # (F, L, 3)
    speed = np.full(pos.shape[:2], np.nan)
    speed[2:-2] = np.linalg.norm(pos[4:] - pos[:-4], axis=-1) / 4 * fps
    near = np.full(pos.shape[:2], np.inf)
    for a in range(pos.shape[1]):
        for b in range(pos.shape[1]):
            if a != b:
                near[:, a] = np.fmin(near[:, a], np.linalg.norm(pos[:, a] - pos[:, b], axis=-1))
    return speed, np.where(np.isfinite(near), near, np.nan)


def change_breakdown(v: dict, fps: float = 166.0, bouts: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    sl = _rows(v)
    fin = VERSIONS.index("final")
    changed = (v["source"][fin] == NODE_BRIDGED).any(axis=-1)[sl]            # (F, L)
    speed, near = (a[sl] for a in _speed_and_distance(v, fps))

    def by(values: np.ndarray, bins, label: str) -> pd.DataFrame:
        cat = pd.cut(values.ravel(), list(bins))
        df = pd.DataFrame({label: cat, "changed": changed.ravel()})
        out = df.groupby(label, observed=False)["changed"].agg(["mean", "size"])
        out = out.rename(columns={"mean": "changed_%", "size": "fish_frames"})
        out["changed_%"] *= 100
        out["share_of_data_%"] = 100 * out["fish_frames"] / max(1, len(df))
        return out.round(3)

    out = {"by_speed_mm_s": by(speed, SPEED_BINS, "speed_mm_s"),
           "by_nearest_fish_mm": by(near, DISTANCE_BINS, "nearest_mm")}
    if bouts is not None and len(bouts):
        frames = v["frames"][sl]
        rows = []
        inside_any = np.zeros(len(frames), dtype=bool)
        for cat in sorted(bouts["category"].unique()):
            m = np.zeros(len(frames), dtype=bool)
            for b in bouts[bouts["category"] == cat].itertuples(index=False):
                m |= (frames >= b.start) & (frames <= b.end)
            inside_any |= m
            if m.any():
                rows.append({"behaviour": cat, "frames": int(m.sum()), "changed_%": 100 * float(changed[m].mean()),
                             "mean_speed_mm_s": float(np.nanmean(speed[m]))})
        if (~inside_any).any():
            rows.append({"behaviour": "(no bout)", "frames": int((~inside_any).sum()),
                         "changed_%": 100 * float(changed[~inside_any].mean()),
                         "mean_speed_mm_s": float(np.nanmean(speed[~inside_any]))})
        out["by_behaviour"] = pd.DataFrame(rows).round(3)
    return out


def measured_share(v: dict, fps: float = 166.0) -> pd.DataFrame:
    """Share of fish-frames whose every node is a triangulated measurement (source 0), by distance."""
    sl = _rows(v)
    fin = VERSIONS.index("final")
    measured = (v["source"][fin] == 0).all(axis=-1)[sl]
    _, near = _speed_and_distance(v, fps)
    cat = pd.cut(near[sl].ravel(), list(DISTANCE_BINS))
    df = pd.DataFrame({"nearest_mm": cat, "measured": measured.ravel()})
    out = df.groupby("nearest_mm", observed=False)["measured"].agg(["mean", "size"])
    return out.rename(columns={"mean": "all_nodes_measured_%", "size": "fish_frames"}).assign(
        **{"all_nodes_measured_%": lambda d: 100 * d["all_nodes_measured_%"]}).round(3)
