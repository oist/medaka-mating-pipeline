"""The four versions of every fish's 3D nodes, for comparing what each correction step changed.

==========  ==================================================================================
initial     plain triangulation of the tracker's detections (no robust re-triangulation, no
            despiking) - ``initial_tracks.npz``, computed once from ``detections.npz``
machine     the review data (``review_data.npz``): two-view re-triangulation and despiking
            during the build; this is what the reviewer watched
manual      ``machine`` + the reviewer's ledger (identity swaps, head/tail flips, per-view fixes,
            re-estimated bodies, solved intervals)
final       ``manual`` + the machine post-pass (:mod:`scripts.review.postprocess`) - the export
==========  ==================================================================================

``initial`` and ``machine`` have no reviewer identity: each name (F / M1 / M2) follows the track it
was given to at the start, so comparing them with ``manual`` also shows the swaps. Everything is
written to ``versions.npz`` next to the export: ``nodes (V, F, L, N, 3)`` float32,
``source (V, F, L, N)`` int8 and ``track (V, F, L)`` int8 (the 3D track a name's data comes from),
with ``versions``, ``labels``, ``frames``, ``node_names``, ``edges``, ``head_node``/``tail_node``/
``position_node`` and ``analysis_start``.
"""

from __future__ import annotations

import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

VERSIONS = ("initial", "machine", "manual", "final")
VERSION_TITLES = {"initial": "initial (raw triangulation)", "machine": "machine (before review)",
                  "manual": "manual (after review)", "final": "final (+ post-pass)"}
INITIAL_FILE = "initial_tracks.npz"
VERSIONS_FILE = "versions.npz"
LABEL_ORDER = ("F", "M1", "M2")
POSITION_NODE = "pectoral"


def _fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def initial_tracks(folder: Path, cal=None, log=None) -> dict:
    """Plain triangulation of the tracker's assignment for the whole recording (cached in
    ``initial_tracks.npz``; rebuilt when ``review_data.npz`` changes). Needs the calibration."""
    folder = Path(folder)
    out = folder / INITIAL_FILE
    stamp = _fingerprint(folder / "review_data.npz")
    if out.exists():
        with np.load(out) as z:
            if str(z["review_data"]) == stamp:
                return {k: z[k] for k in z.files}
    from scripts.reconstruction_3d.calibration import Calibration
    from scripts.review.geometry import Detections, gather
    from scripts.review.reconstruct import RobustParams, reconstruct_tracks
    from scripts.review.review_data import FillParams, OrientationParams

    manifest = json.loads((folder / "review_manifest.json").read_text(encoding="utf-8"))

    def make(kind, key):
        names = {f.name for f in fields(kind)}
        return kind(**{k: v for k, v in (manifest.get(key) or {}).items() if k in names})

    t0 = time.time()
    if cal is None:
        cal = Calibration(str(manifest["sources"]["calibration_dir"]), backend="auto")
    with np.load(folder / "review_data.npz") as z:
        xyz = z["xyz"].astype(float)
        head, tail = int(z["head_node"]), int(z["tail_node"])
        frames = z["frames"]
    dets = Detections(folder)
    kp_raw = gather(dets.points(0, len(dets.frames)), dets.slots)
    robust = make(RobustParams, "robust_params")
    robust.enabled = False
    rec = reconstruct_tracks(cal, kp_raw, kp_raw, xyz, head, tail, make(OrientationParams, "orientation_params"),
                             make(FillParams, "fill_params"), robust, log=log)
    result = {"frames": frames, "nodes3d_filled": rec["nodes3d_filled"].astype(np.float32),
              "node_source": rec["node_source"].astype(np.int8), "review_data": np.array(stamp)}
    tmp = out.with_name("initial_tracks.partial.npz")
    np.savez_compressed(tmp, **result)
    tmp.replace(out)
    if log is not None:
        log(f"[versions] initial tracks ({time.time() - t0:.0f} s)")
    return result


def label_order(labels: dict[str, str], fish_ids: list[str]) -> list[str]:
    """Names in display order: F / M1 / M2 first, then any other name or unnamed track."""
    named = [lab for lab in LABEL_ORDER if lab in labels.values()]
    rest = [labels.get(f) or f for f in fish_ids if (labels.get(f) or f) not in named and labels.get(f) != "-"]
    return named + rest


def per_track(nodes: np.ndarray, source: np.ndarray, labels: dict[str, str], fish_ids: list[str],
              order: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(F, K, N, 3)`` track arrays -> per name, each name on the track it starts on (no swaps)."""
    n_frames = nodes.shape[0]
    by_name = {labels.get(f) or f: k for k, f in enumerate(fish_ids)}
    out = np.full((n_frames, len(order)) + nodes.shape[2:], np.nan, dtype=np.float32)
    src = np.full((n_frames, len(order), nodes.shape[2]), -1, dtype=np.int8)
    track = np.full((n_frames, len(order)), -1, dtype=np.int8)
    for c, name in enumerate(order):
        k = by_name.get(name)
        if k is not None:
            out[:, c], src[:, c], track[:, c] = nodes[:, k], source[:, k], k
    return out, src, track


def from_table(table: pd.DataFrame, frames: np.ndarray, node_names: list[str], fish_ids: list[str],
               order: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An exported table (one row per frame and corrected fish) -> per-name arrays."""
    n_frames, n_nodes = len(frames), len(node_names)
    out = np.full((n_frames, len(order), n_nodes, 3), np.nan, dtype=np.float32)
    src = np.full((n_frames, len(order), n_nodes), -1, dtype=np.int8)
    track = np.full((n_frames, len(order)), -1, dtype=np.int8)
    index = {f: k for k, f in enumerate(fish_ids)}
    for c, name in enumerate(order):
        g = table[(table["label"].astype(str).replace("", np.nan).fillna(table["fish_id"]) == name)]
        if g.empty:
            continue
        t = np.searchsorted(frames, g["frame"].to_numpy())
        for n, node in enumerate(node_names):
            out[t, c, n] = g[[f"{node}_X", f"{node}_Y", f"{node}_Z"]].to_numpy(np.float32)
            src[t, c, n] = g[f"{node}_source"].to_numpy()
        track[t, c] = g["source_track"].map(index).fillna(-1).to_numpy(np.int8)
    return out, src, track


def write_versions(path: Path, data: dict, initial: dict | None, manual: pd.DataFrame, final: pd.DataFrame,
                   labels: dict[str, str], start: int | None) -> dict:
    """Write ``versions.npz``; ``data`` is the (unedited) review data. Returns a small summary."""
    frames = data["frames"]
    fish_ids = [str(f) for f in data["fish_ids"]]
    node_names = [str(n) for n in data["node_names"]]
    order = label_order(labels, fish_ids)
    n_frames = len(frames)
    shape = (len(VERSIONS), n_frames, len(order), len(node_names))
    nodes = np.full(shape + (3,), np.nan, dtype=np.float32)
    source = np.full(shape, -1, dtype=np.int8)
    track = np.full(shape[:3], -1, dtype=np.int8)
    present = []
    if initial is not None:
        nodes[0], source[0], track[0] = per_track(initial["nodes3d_filled"], initial["node_source"], labels,
                                                  fish_ids, order)
        present.append("initial")
    nodes[1], source[1], track[1] = per_track(data["nodes3d_filled"], data["node_source"], labels, fish_ids, order)
    nodes[2], source[2], track[2] = from_table(manual, frames, node_names, fish_ids, order)
    nodes[3], source[3], track[3] = from_table(final, frames, node_names, fish_ids, order)
    present += ["machine", "manual", "final"]
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(
        tmp, nodes=nodes, source=source, track=track, versions=np.array(VERSIONS), present=np.array(present),
        labels=np.array(order), frames=frames, node_names=np.array(node_names), edges=data["edges"],
        head_node=int(data["head_node"]), tail_node=int(data["tail_node"]),
        position_node=node_names.index(POSITION_NODE) if POSITION_NODE in node_names else 0,
        analysis_start=-1 if start is None else int(start), fish_ids=np.array(fish_ids))
    tmp.replace(path)
    return {"file": str(path), "versions": present, "labels": order}
