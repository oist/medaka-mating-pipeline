"""Every raw SLEAP instance of the three views, unfiltered, for display (``slp_raw.npz`` in a review package).

``detections.npz`` holds what the 3D step could use: false positives are dropped there (span < 15 px,
duplicates within 12 px, no anchor node). This file keeps everything SLEAP wrote - every instance of every
frame (two when SLEAP found two, six when it found six) and every node it found (NaN = not found), in the
.slp's own order.

::

    python -m scripts.review.slp_raw <review package folder> [--slp-dir DIR]

The .slp folder defaults to ``sources.slp_dir`` of the package's ``review_manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scripts.review.review_data import DET_NONE, DET_SCALE, VIEWS, unpack_detections

RAW_FILE = "slp_raw.npz"


def load_slp_instances(slp_path: Path, start: int, end: int) -> tuple[dict[int, list[np.ndarray]], list[str],
                                                                      list[tuple[int, int]]]:
    """``{frame: [(N, 2) per instance]}`` for frames ``start..end`` - every instance, no filtering."""
    import sleap_io as sio

    labels = sio.load_slp(str(slp_path))
    skeleton = labels.skeletons[0]
    names = [n.name for n in skeleton.nodes]
    edges = [(names.index(e.source.name), names.index(e.destination.name)) for e in skeleton.edges]
    out: dict[int, list[np.ndarray]] = {}
    for lf in labels.labeled_frames:
        fi = int(lf.frame_idx)
        if start <= fi <= end and len(lf.instances):
            out.setdefault(fi, []).extend(np.asarray(inst.numpy(), dtype=np.float32)[:, :2] for inst in lf.instances)
    return out, names, edges


def pack(per_view: list[dict[int, list[np.ndarray]]], frames: np.ndarray, n_nodes: int) -> np.ndarray:
    """``(F, V, S, N, 2)`` int16 (pixels x ``DET_SCALE``, ``DET_NONE`` = none); S = most instances in a frame."""
    n_slots = max((len(d) for view in per_view for d in view.values()), default=1)
    det = np.full((len(frames), len(per_view), n_slots, n_nodes, 2), DET_NONE, dtype=np.int16)
    f0 = int(frames[0])
    lim = (np.iinfo(np.int16).max - 1) / DET_SCALE
    for v, view in enumerate(per_view):
        for fi, insts in view.items():
            t = fi - f0
            if 0 <= t < len(frames):
                for s, d in enumerate(insts):
                    q = np.round(np.clip(d, -lim, lim) * DET_SCALE)
                    det[t, v, s] = np.where(np.isfinite(d), q, DET_NONE).astype(np.int16)
    return det


def build(folder: Path, slp_dir: Path | None = None, log=print) -> Path:
    """Write ``slp_raw.npz`` into the review package ``folder`` (frames of its ``review_data.npz``)."""
    folder = Path(folder)
    if slp_dir is None:
        manifest = json.loads((folder / "review_manifest.json").read_text(encoding="utf-8"))
        slp_dir = Path(manifest["sources"]["slp_dir"])
    with np.load(folder / "review_data.npz") as z:
        frames = z["frames"]
    per_view, names, edges = [], None, None
    for view in VIEWS:
        t0 = time.time()
        insts, n, e = load_slp_instances(Path(slp_dir) / f"{view}.slp", int(frames[0]), int(frames[-1]))
        if names is not None and n != names:
            raise ValueError(f"{view}.slp skeleton {n} differs from {names}")
        names, edges = n, e
        per_view.append(insts)
        log(f"[slp_raw] {view}.slp: {len(insts)} frames with instances, "
            f"{sum(map(len, insts.values()))} instances ({time.time() - t0:.0f} s)")
    det = pack(per_view, frames, len(names))
    tmp = folder / "slp_raw.partial.npz"
    np.savez_compressed(tmp, frames=frames, det=det, scale=DET_SCALE, none=DET_NONE, node_names=np.array(names),
                        edges=np.array(edges, dtype=int).reshape(-1, 2), slp_dir=str(slp_dir))
    os.replace(tmp, folder / RAW_FILE)
    log(f"[slp_raw] wrote {folder / RAW_FILE}: {det.shape[2]} instance slots, nodes {names}")
    return folder / RAW_FILE


@dataclass
class RawSlp:
    """``slp_raw.npz`` of a review package."""

    folder: Path
    frames: np.ndarray = field(init=False)
    det: np.ndarray = field(init=False)
    node_names: list[str] = field(init=False)
    edges: list[tuple[int, int]] = field(init=False)

    def __post_init__(self):
        with np.load(Path(self.folder) / RAW_FILE) as z:
            self.frames, self.det = z["frames"], z["det"]
            self.node_names = [str(x) for x in z["node_names"]]
            self.edges = [tuple(int(a) for a in e) for e in z["edges"]]

    @classmethod
    def open(cls, folder: Path) -> RawSlp | None:
        return cls(folder) if (Path(folder) / RAW_FILE).exists() else None

    def points(self, frame: int) -> np.ndarray | None:
        """Every instance of ``frame`` as source pixels ``(V, S, N, 2)`` (NaN = none), or None."""
        if len(self.frames) == 0:
            return None
        row = frame - int(self.frames[0])
        if not 0 <= row < len(self.frames) or int(self.frames[row]) != frame:
            return None
        return unpack_detections(self.det[row])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("folder", type=Path, help="review package folder (with review_data.npz)")
    p.add_argument("--slp-dir", type=Path, default=None, help="folder of <view>.slp (default: from the manifest)")
    args = p.parse_args(argv)
    build(args.folder, args.slp_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
