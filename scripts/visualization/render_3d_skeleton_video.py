from __future__ import annotations

"""
Render a 3D mouth-tail skeleton mp4 for the entire experiment.

Inputs
------
* The three SLEAP `.slp` files (`top_cam.slp`, `side1_cam.slp`, `side2_cam.slp`).
* The `best.json` from `reconstruct_from_slp` (per-frame fish<->.slp-slot map).
* The calibration model folder (`imCoords_to_XYZ.joblib` + friends).

Pipeline
--------
1. Load `best.json` -> per-frame `cam{0,1,2}_to_id` (which detection slot in each
   view corresponds to which fish ID).
2. Load each .slp once; apply the same detection filtering and slot ordering as
   `load_2d_slp.py`. Keep mouth + tail (x, y) per instance instead of
   collapsing the rendered keypoints to a centroid.
3. For each rendered frame: look up the slot index per view per fish, build a
   (3, 2) imCoord triplet for mouth and for tail, and triangulate each via the
   calibration model.
4. Draw 3 fish as colored line segments (mouth dot + tail dot + connecting line)
   in a fixed 3D axes box, encode via imageio_ffmpeg + libx264.

Defaults
--------
* `--width --height 2048` (2K square).
* `--frame-step 10` (render every 10th source frame; 35 min @ 166 fps -> 34860
  rendered frames; at the default 30 fps playback that's ~19 min mp4 runtime).
* `--fps 30` mp4 playback fps. Source frame numbers shown on screen.

Usage::

    python -m scripts.visualization.render_3d_skeleton_video 20251225_K_1_2_FCMC \\
        --slp-dir Z:/.../sleap_output/20251225_K_1_2_FCMC/round4_260520 \\
        --best-json Z:/.../comparison/round4_260520_best.json \\
        --model-folder C:/Users/.../calibration_results/.../models \\
        --out Z:/.../comparison/round4_260520_skeleton_3d.mp4

    # 3-node models: draw mouth -> pectoral -> tail (head node first, tail last)
    ... --node-names mouth pectoral tail
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np


VIEWS = ("top_cam", "side1_cam", "side2_cam")
VIEW_TO_CAM = {"top_cam": "cam0", "side1_cam": "cam1", "side2_cam": "cam2"}
VIEW_TO_BEST_KEY = {"top_cam": "cam0_to_id", "side1_cam": "cam1_to_id",
                    "side2_cam": "cam2_to_id"}
FISH_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
TRIANGULATION_CACHE_SLOT_ORDER_VERSION = 3  # 3: two-view node triangulation


class SkeletonRenderer3D:
    """Persistent matplotlib 3D figure for per-frame mouth+tail skeleton rendering.

    Maintains arrow artists (one quiver per fish, remove+add per frame, since
    `Quiver3D` has no segment-update API) and optional fading trails (per-fish
    `Line3DCollection` of mid-body positions). Returns an RGB ndarray on each
    `update()`; intended to be combined into multi-panel composites.
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        xyz_lim: tuple[float, float],
        fish_ids: list[str],
        elev: float = 20.0,
        azim: float = -60.0,
        arrows: bool = True,
        trail_frames: int = 0,
        dpi: int = 100,
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgb
        from matplotlib.patches import Patch
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        self._plt = plt
        self._Line3DCollection = Line3DCollection

        self.n_fish = len(fish_ids)
        self.arrows = arrows
        self.trail_frames = max(0, int(trail_frames))
        self.colors = [to_rgb(FISH_COLORS[i % len(FISH_COLORS)])
                       for i in range(self.n_fish)]

        self.fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlim(*xyz_lim); self.ax.set_ylim(*xyz_lim); self.ax.set_zlim(*xyz_lim)
        self.ax.set_xlabel("X (mm)"); self.ax.set_ylabel("Y (mm)"); self.ax.set_zlabel("Z (mm)")
        self.ax.view_init(elev=elev, azim=azim)
        self.ax.set_box_aspect((1, 1, 1))
        self.title = self.ax.set_title("")

        # Body artists. Both modes update endpoints in place (no remove+add).
        # "arrows" mode draws a thick body Line3D + a large mouth marker scatter
        # + a small tail marker scatter -- visually an arrow (tail -> mouth)
        # without invoking ax.quiver (which has a scalar-input scaling bug in
        # matplotlib 3.10 when arrows appear and disappear across frames).
        self._body_lines: list[object] = []
        self._mouth_markers: list[object] = []  # arrows mode only
        self._tail_markers: list[object] = []   # arrows mode only
        for i, fid in enumerate(fish_ids):
            c = self.colors[i]
            if arrows:
                (line,) = self.ax.plot([0, 0], [0, 0], [0, 0],
                                       color=c, lw=3.5)
                self._body_lines.append(line)
                # Large mouth marker (the "head" of the arrow).
                self._mouth_markers.append(
                    self.ax.scatter([0], [0], [0],
                                    color=c, s=140, marker="o",
                                    edgecolors="k", linewidths=0.8,
                                    depthshade=False)
                )
                # Small tail marker.
                self._tail_markers.append(
                    self.ax.scatter([0], [0], [0],
                                    color=c, s=22, marker="o", depthshade=False)
                )
            else:
                (line,) = self.ax.plot([0, 0], [0, 0], [0, 0],
                                       color=c, lw=2.5,
                                       marker="o", ms=8)
                self._body_lines.append(line)
        handles = [Patch(color=self.colors[i], label=fid)
                   for i, fid in enumerate(fish_ids)]
        self.ax.legend(handles=handles, loc="upper right")

        # Trails: one Line3DCollection per fish with per-segment alpha.
        self._trail_buffers: list[collections.deque] = (
            [collections.deque(maxlen=self.trail_frames) for _ in range(self.n_fish)]
            if self.trail_frames > 0 else []
        )
        self._trail_artists: list[object] = []
        if self.trail_frames > 0:
            # Seed each collection with a tiny in-bounds segment so
            # add_collection3d's auto_scale_xyz call has data to look at
            # (empty segments hit a matplotlib 3.10 unpacking bug).
            seed = float(xyz_lim[0])
            dummy = [((seed, seed, seed), (seed + 0.001, seed + 0.001, seed + 0.001))]
            for _ in range(self.n_fish):
                lc = Line3DCollection(dummy, linewidths=1.6)
                self.ax.add_collection3d(lc)
                lc.set_segments([])  # clear; first real update will set them
                self._trail_artists.append(lc)

    def update(self, title_text: str, arr_frame: np.ndarray) -> np.ndarray:
        """Update artists for one frame.

        Parameters
        ----------
        title_text : str
            Title to display (e.g. ``"frame 12345 (601/34826)"``).
        arr_frame : np.ndarray, shape ``(n_fish, n_nodes, 3)``
            Nodes in ``--node-names`` order, head first: ``arr_frame[k, 0]`` =
            mouth XYZ, ``arr_frame[k, -1]`` = tail XYZ, any middle nodes
            (e.g. pectoral) in between. NaN signals a missing detection.

        Returns
        -------
        rgb : np.ndarray of shape ``(H, W, 3)`` uint8.
        """
        self.title.set_text(title_text)
        for k in range(self.n_fish):
            nodes = arr_frame[k]
            m = nodes[0]
            t = nodes[-1]
            finite = np.isfinite(nodes).all(axis=1)
            # Arrows need both ends; the polyline draws whatever nodes exist.
            valid = bool(finite[0] and finite[-1]) if self.arrows else bool(finite.any())
            if self.arrows:
                if valid:
                    self._body_lines[k].set_data_3d(
                        [t[0], m[0]], [t[1], m[1]], [t[2], m[2]]
                    )
                    self._mouth_markers[k]._offsets3d = (
                        np.array([m[0]]), np.array([m[1]]), np.array([m[2]])
                    )
                    self._tail_markers[k]._offsets3d = (
                        np.array([t[0]]), np.array([t[1]]), np.array([t[2]])
                    )
                else:
                    self._body_lines[k].set_data_3d([], [], [])
                    self._mouth_markers[k]._offsets3d = (
                        np.empty(0), np.empty(0), np.empty(0)
                    )
                    self._tail_markers[k]._offsets3d = (
                        np.empty(0), np.empty(0), np.empty(0)
                    )
            else:
                if valid:
                    # Polyline through every visible node (mouth -> ... -> tail);
                    # a missing node is skipped rather than hiding the fish.
                    pts = nodes[finite]
                    self._body_lines[k].set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
                else:
                    self._body_lines[k].set_data_3d([], [], [])
            if self.trail_frames > 0:
                buf = self._trail_buffers[k]
                if valid:
                    buf.append(nodes[finite].mean(axis=0))
                if len(buf) >= 2:
                    pts = np.asarray(buf)
                    segs = np.stack([pts[:-1], pts[1:]], axis=1)
                    n_seg = len(segs)
                    alphas = np.linspace(0.05, 0.55, n_seg)
                    cols = np.zeros((n_seg, 4))
                    cols[:, :3] = self.colors[k]
                    cols[:, 3] = alphas
                    self._trail_artists[k].set_segments(segs)
                    self._trail_artists[k].set_color(cols)
                else:
                    self._trail_artists[k].set_segments([])

        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()

    def close(self) -> None:
        self._plt.close(self.fig)


def _instances_in_reconstruction_slot_order(
    insts,
    *,
    node_names: list[str],
    anchor_node: str | None = None,
    filter_fp: bool = True,
    min_span_px: float = 15.0,
    dup_dist_px: float = 12.0,
) -> list:
    """Apply the reconstruction ingestion rules and return slot-ordered instances.

    ``best.json`` stores indices into the candidate arrays produced by
    :func:`scripts.reconstruction_3d.load_2d_slp.load_points_by_frame`.  The
    renderer must therefore repeat that loader's filtering, finite-centroid
    check, and score/X tie-break exactly before using those indices.
    """
    from scripts.reconstruction_3d.load_2d_slp import _instances_in_slot_order

    return _instances_in_slot_order(
        insts,
        node_names=node_names,
        anchor_node=anchor_node,
        filter_fp=filter_fp,
        min_span_px=min_span_px,
        dup_dist_px=dup_dist_px,
    )


def load_slp_node_xy(
    slp_path: Path,
    node_names: tuple[str, ...] = ("mouth", "tail"),
    *,
    anchor_node: str | None = None,
    filter_fp: bool = True,
    min_span_px: float = 15.0,
    dup_dist_px: float = 12.0,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> tuple[dict[int, np.ndarray], list[str]]:
    """Load .slp keypoints in the exact slot order used by reconstruction.

    The defaults mirror ``load_points_by_frame``: visible-node centroid for
    sorting plus the production collapsed/duplicate detection filter. This is
    required because ``best.json`` candidate indices refer to that filtered
    ordering, not to the raw order of instances in the ``.slp`` file.
    """
    import sleap_io as sio
    labels = sio.load_slp(str(slp_path))
    sk_nodes = [n.name for n in labels.skeletons[0].nodes]
    node_idx = [sk_nodes.index(n) for n in node_names]

    out: dict[int, np.ndarray] = {}
    for lf in labels.labeled_frames:
        fi = int(lf.frame_idx)
        if start_frame is not None and fi < start_frame:
            continue
        if end_frame is not None and fi > end_frame:
            continue
        insts = _instances_in_reconstruction_slot_order(
            lf.instances,
            node_names=sk_nodes,
            anchor_node=anchor_node,
            filter_fp=filter_fp,
            min_span_px=min_span_px,
            dup_dist_px=dup_dist_px,
        )
        items: list[np.ndarray] = []
        for inst in insts:
            pts = inst.numpy()  # (n_nodes, 2)
            sel = pts[node_idx]  # (len(node_names), 2)
            items.append(sel)
        if not items:
            continue
        arr = np.stack(items, axis=0)
        out[fi] = arr
    return out, sk_nodes


def build_triangulation(
    slp_per_view: dict[str, dict[int, np.ndarray]],
    best: dict,
    cal,
    fish_ids: list[str],
    node_names: tuple[str, ...],
    frames: list[int],
) -> np.ndarray:
    """Return ``(n_frames, n_fish, n_nodes, 3)`` triangulated 3D positions."""
    n_f = len(frames)
    n_fish = len(fish_ids)
    n_nodes = len(node_names)
    out = np.full((n_f, n_fish, n_nodes, 3), np.nan)

    cam_keys = ("cam0_to_id", "cam1_to_id", "cam2_to_id")
    view_order = ("top_cam", "side1_cam", "side2_cam")

    for fi_out, src in enumerate(frames):
        info = best.get(src) or best.get(str(src))
        if info is None:
            continue
        # For each fish, find its slot index per view.
        for k, fid in enumerate(fish_ids):
            slot_per_view: list[int | None] = []
            for cam_key in cam_keys:
                lst = info.get(cam_key) or []
                slot = None
                for idx, lbl in enumerate(lst):
                    if lbl == fid:
                        slot = idx
                        break
                slot_per_view.append(slot)
            # Triangulate each node independently from every view that saw it
            # (two views suffice). A fish occluded in one camera -- a
            # two-view completion in best.json, with no raw slot there -- or a
            # node missed in one view keeps its skeleton instead of vanishing.
            for n_i in range(n_nodes):
                triplet = np.full((3, 2), np.nan)
                for v_i, (view, slot) in enumerate(zip(view_order, slot_per_view)):
                    arr = slp_per_view[view].get(src)
                    if slot is None or arr is None or slot >= arr.shape[0]:
                        continue
                    triplet[v_i] = arr[slot, n_i]
                if np.isfinite(triplet).all(axis=1).sum() < 2:
                    continue
                try:
                    xyz = cal.compute_XYZ_from_imageCoord_triplet(triplet)
                except Exception:
                    continue
                if np.isfinite(xyz).all():
                    out[fi_out, k, n_i] = xyz
            # Fewer than two usable views for every node: keep the fish's
            # reconstructed 3D point as a degenerate marker so it stays visible.
            if not np.isfinite(out[fi_out, k]).any():
                point = np.asarray((info.get("groups") or {}).get(fid, []), dtype=float)
                if point.shape == (3,) and np.isfinite(point).all():
                    out[fi_out, k, :, :] = point
    return out


def interpolate_gaps(arr: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill NaN runs <= max_gap along axis 0 per (fish, node, axis)."""
    out = arr.copy()
    n_f, n_fish, n_nodes, n_xyz = out.shape
    for k in range(n_fish):
        for n_i in range(n_nodes):
            for c in range(n_xyz):
                s = out[:, k, n_i, c]
                isnan = np.isnan(s)
                if not isnan.any() or isnan.all():
                    continue
                # Walk gaps.
                i = 0
                while i < n_f:
                    if not isnan[i]:
                        i += 1
                        continue
                    j = i
                    while j < n_f and isnan[j]:
                        j += 1
                    # Gap is [i, j); endpoints i-1 and j are valid (if in range).
                    if i > 0 and j < n_f and (j - i) <= max_gap:
                        x0, x1 = i - 1, j
                        y0, y1 = s[x0], s[x1]
                        for kk in range(i, j):
                            t = (kk - x0) / (x1 - x0)
                            s[kk] = y0 + t * (y1 - y0)
                    i = j
                out[:, k, n_i, c] = s
    return out


def render_video(
    arr: np.ndarray,
    frame_indices: list[int],
    out_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    xyz_lim: tuple[float, float],
    fish_ids: list[str],
    elev: float,
    azim: float,
    arrows: bool = False,
    trail_frames: int = 0,
) -> None:
    import imageio.v2 as iio

    n_f = arr.shape[0]
    renderer = SkeletonRenderer3D(
        width=width, height=height, xyz_lim=xyz_lim, fish_ids=fish_ids,
        elev=elev, azim=azim, arrows=arrows, trail_frames=trail_frames,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(
        str(out_path), fps=fps, codec="libx264",
        quality=8, macro_block_size=1,
        pixelformat="yuv420p",
    )
    t0 = time.time()
    next_report = t0 + 5.0
    try:
        for f_i in range(n_f):
            src_frame = frame_indices[f_i]
            label = f"frame {src_frame}  ({f_i+1}/{n_f})"
            rgb = renderer.update(label, arr[f_i])
            writer.append_data(rgb)
            now = time.time()
            if now >= next_report:
                dt = now - t0
                rate = (f_i + 1) / dt
                eta = (n_f - f_i - 1) / max(rate, 1e-9)
                print(f"  rendered {f_i+1}/{n_f}  "
                      f"({rate:.2f} fps render, ETA {eta/60:.1f} min)")
                next_report = now + 30.0
    finally:
        writer.close()
        renderer.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment")
    ap.add_argument("--slp-dir", type=Path, required=True)
    ap.add_argument("--best-json", type=Path, required=True)
    ap.add_argument("--model-folder", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-frame", type=int, default=None)
    ap.add_argument("--end-frame", type=int, default=None)
    ap.add_argument("--frame-step", type=int, default=10,
                    help="Render every Nth source frame (default 10).")
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--height", type=int, default=2048)
    ap.add_argument("--fps", type=int, default=30, help="MP4 playback fps.")
    ap.add_argument("--xyz-lim", nargs=2, type=float, default=(0.0, 220.0))
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--node-names", nargs="+", default=["mouth", "tail"])
    ap.add_argument("--arrows", action="store_true",
                    help="Render each fish as a 3D arrow (tail -> mouth) "
                         "instead of a line+markers (default off for back-compat).")
    ap.add_argument("--trail-seconds", type=float, default=0.0,
                    help="Length of the fading per-fish trail in seconds of "
                         "source recording time (0 disables; default 0).")
    ap.add_argument("--source-fps", type=float, default=166.0,
                    help="Source recording fps; used to convert --trail-seconds "
                         "to a per-fish ring-buffer length (= trail_s * source_fps "
                         "/ frame_step). Default 166.")
    ap.add_argument("--triangulation-npz", type=Path, default=None,
                    help="Cache file. If exists, skip triangulation pass. "
                         "If not, write the (n_frames,n_fish,n_nodes,3) array here.")
    ap.add_argument("--fill-gap-frames", type=int, default=30,
                    help="Linearly interpolate NaN runs shorter than this many "
                         "rendered frames per (fish, node, axis). At step=10 / "
                         "166 fps source that's 30*10/166 ~= 1.8s. 0 disables.")
    args = ap.parse_args(argv)

    # Allow `-m scripts.visualization.render_3d_skeleton_video` style.
    if __package__ in (None, ""):
        sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.reconstruction_3d.calibration import Calibration

    print(f"experiment       : {args.experiment}")
    print(f"slp-dir          : {args.slp_dir}")
    print(f"best-json        : {args.best_json}")
    print(f"model-folder     : {args.model_folder}")
    print(f"out              : {args.out}")
    print(f"frame-step       : {args.frame_step}")
    print(f"resolution       : {args.width}x{args.height}")
    print()

    cache = args.triangulation_npz
    cache_loaded = False
    if cache and cache.exists():
        print(f"Loading triangulation cache: {cache}")
        data = np.load(cache, allow_pickle=False)
        cache_version = (
            int(np.asarray(data["slot_order_version"]).reshape(-1)[0])
            if "slot_order_version" in data.files
            else 1
        )
        if cache_version == TRIANGULATION_CACHE_SLOT_ORDER_VERSION:
            arr = data["xyz"]
            frame_indices = data["frames"].tolist()
            fish_ids = [str(x) for x in data["fish_ids"]]
            cache_loaded = True
        else:
            print(
                f"  stale cache slot-order version {cache_version}; "
                "recomputing with reconstruction-aligned slots."
            )
    if not cache_loaded:
        print(f"Loading best.json ...")
        with args.best_json.open("r", encoding="utf-8") as fh:
            best_payload = json.load(fh)
        fish_ids = list(best_payload.get("id_labels") or [])
        if not fish_ids:
            fish_ids = [f"ID_{i+1}" for i in range(int(best_payload.get("num_individuals", 3)))]
        best = {int(k): v for k, v in best_payload["frames"].items()}
        print(f"  best.json: {len(best)} frames, fish_ids={fish_ids}")

        slp_per_view: dict[str, dict[int, np.ndarray]] = {}
        for view in VIEWS:
            p = args.slp_dir / f"{view}.slp"
            print(f"Loading {p} ...")
            data_v, sk = load_slp_node_xy(p, node_names=tuple(args.node_names))
            slp_per_view[view] = data_v
            print(f"  {view}: {len(data_v)} frames, skeleton={sk}")

        # Frame range -> output frame list with stride.
        all_frames = sorted(set(best.keys()) & set(slp_per_view["top_cam"].keys())
                            & set(slp_per_view["side1_cam"].keys())
                            & set(slp_per_view["side2_cam"].keys()))
        if args.start_frame is not None:
            all_frames = [f for f in all_frames if f >= args.start_frame]
        if args.end_frame is not None:
            all_frames = [f for f in all_frames if f <= args.end_frame]
        frame_indices = all_frames[::max(1, args.frame_step)]
        print(f"  rendering {len(frame_indices)} frames "
              f"(every {args.frame_step}th from {len(all_frames)} sources)")

        print(f"Loading calibration: {args.model_folder} ...")
        cal = Calibration(str(args.model_folder))

        print(f"Triangulating mouth + tail per fish per frame ...")
        t0 = time.time()
        arr = build_triangulation(
            slp_per_view, best, cal, fish_ids,
            tuple(args.node_names), frame_indices,
        )
        print(f"  done in {time.time() - t0:.1f}s; shape={arr.shape}")

        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache, xyz=arr,
                frames=np.array(frame_indices, dtype=np.int64),
                fish_ids=np.array(fish_ids),
                slot_order_version=np.array(
                    [TRIANGULATION_CACHE_SLOT_ORDER_VERSION], dtype=np.int64
                ),
            )
            print(f"  cached -> {cache}")

    valid = np.isfinite(arr[..., 0]).mean()
    print(f"  valid 3D cells (raw): {valid*100:.2f}%")
    if args.fill_gap_frames > 0:
        arr = interpolate_gaps(arr, max_gap=args.fill_gap_frames)
        valid_f = np.isfinite(arr[..., 0]).mean()
        print(f"  valid 3D cells (after fill, max_gap={args.fill_gap_frames}): "
              f"{valid_f*100:.2f}%")
    print()
    print(f"Rendering mp4 ...")
    trail_frames = int(round(args.trail_seconds * args.source_fps / max(1, args.frame_step)))
    if args.trail_seconds > 0:
        print(f"  trails: {args.trail_seconds}s @ source {args.source_fps} fps "
              f"/ step {args.frame_step} = {trail_frames} rendered frames")
    if args.arrows:
        print("  arrows: enabled (tail -> mouth quiver)")
    render_video(
        arr, frame_indices, args.out,
        width=args.width, height=args.height, fps=args.fps,
        xyz_lim=tuple(args.xyz_lim), fish_ids=fish_ids,
        elev=args.elev, azim=args.azim,
        arrows=args.arrows, trail_frames=trail_frames,
    )
    print(f"Done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
