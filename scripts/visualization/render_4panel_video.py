from __future__ import annotations

"""
Render a 4-panel mp4 with the three camera views + the 3D skeleton view.

Layout (2K x 2K = 2048 x 2048; each panel is 1024 x 1024):

    +---------------+---------------+
    |   top_cam     |   side1_cam   |
    +---------------+---------------+
    |   side2_cam   |     3D        |
    +---------------+---------------+

Per fish in each camera panel: a colored tail->mouth arrow with the fish ID
label. Per fish in the 3D panel: a 3D arrow (quiver) plus an optional fading
trail of the past N seconds of mid-body positions.

The 2D arrow source data is the **raw SLEAP keypoints** (mouth + tail xy per
instance), mapped to fish IDs via the per-frame slot table in ``best.json``.
3D arrows reuse the triangulation cache produced by
``render_3d_skeleton_video.py`` (or are computed on the fly).

Usage::

    python -m scripts.visualization.render_4panel_video 20251225_K_1_2_FCMC \
        --slp-dir Z:/.../sleap_output/<exp>/<round> \
        --best-json Z:/.../comparison/<round>_best.json \
        --model-folder C:/Users/.../models \
        --video-dir Z:/.../medaka_202512/<exp> \
        --triangulation-npz Z:/.../<round>_skeleton_xyz.npz \
        --out Z:/.../comparison/<round>_4panel.mp4 \
        --trail-seconds 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 2D drawing helpers
# ---------------------------------------------------------------------------

def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return (b, g, r)


def draw_arrows_on_image(
    img_bgr: np.ndarray,
    arrows: list[tuple[str, np.ndarray, np.ndarray, tuple[int, int, int]]],
    *,
    thickness: int = 4,
    tip_length: float = 0.30,
    font_scale: float = 0.9,
) -> np.ndarray:
    """Draw tail->mouth arrows + ID labels on ``img_bgr`` in place. Returns it.

    ``arrows`` is a list of ``(label, mouth_xy, tail_xy, color_bgr)`` in image
    pixel coordinates (i.e. already scaled to ``img_bgr.shape``).
    """
    import cv2
    for label, mouth, tail, color in arrows:
        if mouth is None or tail is None:
            continue
        pt_tail = (int(round(float(tail[0]))), int(round(float(tail[1]))))
        pt_mouth = (int(round(float(mouth[0]))), int(round(float(mouth[1]))))
        # Black halo first so the arrow is readable on any background.
        cv2.arrowedLine(img_bgr, pt_tail, pt_mouth, (0, 0, 0),
                        thickness + 2, cv2.LINE_AA, 0, tip_length)
        cv2.arrowedLine(img_bgr, pt_tail, pt_mouth, color,
                        thickness, cv2.LINE_AA, 0, tip_length)
        # ID label near the mouth, offset so it doesn't sit on the arrowhead.
        label_pt = (pt_mouth[0] + 12, pt_mouth[1] - 12)
        cv2.putText(img_bgr, label, label_pt, cv2.FONT_HERSHEY_DUPLEX,
                    font_scale, (0, 0, 0), max(2, thickness), cv2.LINE_AA)
        cv2.putText(img_bgr, label, label_pt, cv2.FONT_HERSHEY_DUPLEX,
                    font_scale, color, max(1, thickness - 2), cv2.LINE_AA)
    return img_bgr


def label_panel_top(img_bgr: np.ndarray, text: str, font_scale: float = 1.1) -> None:
    """Stamp a panel title in the top-left corner."""
    import cv2
    pt = (20, 40)
    cv2.putText(img_bgr, text, pt, cv2.FONT_HERSHEY_DUPLEX,
                font_scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img_bgr, text, pt, cv2.FONT_HERSHEY_DUPLEX,
                font_scale, (255, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 2D arrow lookup
# ---------------------------------------------------------------------------

def lookup_2d_arrow(
    best_frame: dict,
    slp_per_view: dict[str, dict[int, np.ndarray]],
    src_frame: int,
    view: str,
    fish_id: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return ``(mouth_xy, tail_xy)`` for ``fish_id`` in ``view`` at ``src_frame``."""
    from scripts.visualization.render_3d_skeleton_video import VIEW_TO_BEST_KEY
    key = VIEW_TO_BEST_KEY[view]
    lst = best_frame.get(key) or []
    slot = None
    for idx, lbl in enumerate(lst):
        if lbl == fish_id:
            slot = idx
            break
    if slot is None:
        return None, None
    arr = slp_per_view[view].get(src_frame)
    if arr is None or slot >= arr.shape[0]:
        return None, None
    mouth = arr[slot, 0]
    tail = arr[slot, 1]
    if not (np.isfinite(mouth).all() and np.isfinite(tail).all()):
        return None, None
    return mouth, tail


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment")
    ap.add_argument("--slp-dir", type=Path, required=True,
                    help="Dir containing top_cam.slp / side1_cam.slp / side2_cam.slp.")
    ap.add_argument("--best-json", type=Path, required=True)
    ap.add_argument("--model-folder", type=Path, required=True,
                    help="Calibration model dir (used only when --triangulation-npz "
                         "is absent and we have to recompute).")
    ap.add_argument("--video-dir", type=Path, required=True,
                    help="Experiment dir under medaka_data_dir, containing "
                         "{top_cam,side1_cam,side2_cam}/cam*.mp4")
    ap.add_argument("--triangulation-npz", type=Path, default=None,
                    help="Cache from render_3d_skeleton_video. If missing, we "
                         "recompute (slower; needs --model-folder).")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-frame", type=int, default=None)
    ap.add_argument("--end-frame", type=int, default=None)
    ap.add_argument("--frame-step", type=int, default=10)
    ap.add_argument("--panel-size", type=int, default=1024,
                    help="Width/height of each of the 4 panels.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--xyz-lim", nargs=2, type=float, default=(0.0, 220.0))
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--node-names", nargs="+", default=["mouth", "tail"])
    ap.add_argument("--trail-seconds", type=float, default=5.0)
    ap.add_argument("--source-fps", type=float, default=166.0)
    ap.add_argument("--fill-gap-frames", type=int, default=30)
    args = ap.parse_args(argv)

    # Imports that pull heavy deps (cv2, matplotlib) only after CLI parsing.
    if __package__ in (None, ""):
        sys.path.append(str(Path(__file__).resolve().parents[2]))
    import cv2
    import imageio.v2 as iio
    from scripts.visualization.render_3d_skeleton_video import (
        FISH_COLORS, SkeletonRenderer3D, VIEWS,
        load_slp_node_xy, build_triangulation, interpolate_gaps,
    )

    cap_paths = {
        "top_cam":   args.video_dir / "top_cam"   / "cam0_top.mp4",
        "side1_cam": args.video_dir / "side1_cam" / "cam1_side.mp4",
        "side2_cam": args.video_dir / "side2_cam" / "cam2_side.mp4",
    }
    for v, p in cap_paths.items():
        if not p.exists():
            sys.exit(f"video not found: {p}")

    print(f"experiment   : {args.experiment}")
    print(f"slp-dir      : {args.slp_dir}")
    print(f"best-json    : {args.best_json}")
    print(f"video-dir    : {args.video_dir}")
    print(f"out          : {args.out}")
    print(f"frame-step   : {args.frame_step}")
    print(f"panel size   : {args.panel_size}x{args.panel_size}")
    print(f"trail        : {args.trail_seconds}s -> "
          f"{int(round(args.trail_seconds * args.source_fps / max(1, args.frame_step)))} rendered frames")
    print()

    # 1) best.json
    print("Loading best.json ...")
    with args.best_json.open("r", encoding="utf-8") as fh:
        best_payload = json.load(fh)
    fish_ids = list(best_payload.get("id_labels")
                    or [f"ID_{i+1}" for i in range(int(best_payload.get("num_individuals", 3)))])
    best = {int(k): v for k, v in best_payload["frames"].items()}
    print(f"  {len(best)} frames, fish_ids={fish_ids}")

    # 2) Per-view .slp (mouth + tail xy per instance, slot-sorted by score).
    slp_per_view: dict[str, dict[int, np.ndarray]] = {}
    for view in VIEWS:
        slp_path = args.slp_dir / f"{view}.slp"
        print(f"Loading {slp_path} ...")
        data_v, sk = load_slp_node_xy(slp_path, node_names=tuple(args.node_names))
        slp_per_view[view] = data_v
        print(f"  {view}: {len(data_v)} frames, skeleton={sk}")

    # 3) Frame index list.
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

    # 4) 3D triangulation: load cache or compute on the fly.
    if args.triangulation_npz and args.triangulation_npz.exists():
        print(f"Loading triangulation cache: {args.triangulation_npz}")
        data = np.load(args.triangulation_npz, allow_pickle=False)
        arr3d = data["xyz"]
        cached_frames = data["frames"].tolist()
        if cached_frames != frame_indices:
            print(f"  WARNING: cache covers {len(cached_frames)} frames; "
                  f"current selection covers {len(frame_indices)} frames -- "
                  f"recomputing to match.")
            arr3d = None
        else:
            cached_fish = [str(x) for x in data["fish_ids"]]
            if cached_fish != fish_ids:
                print(f"  WARNING: cache fish_ids {cached_fish} != "
                      f"current {fish_ids}; recomputing.")
                arr3d = None
    else:
        arr3d = None

    if arr3d is None:
        from scripts.reconstruction_3d.calibration import Calibration
        print(f"Loading calibration: {args.model_folder} ...")
        cal = Calibration(str(args.model_folder))
        print(f"Triangulating mouth + tail per fish per frame ...")
        t0 = time.time()
        arr3d = build_triangulation(
            slp_per_view, best, cal, fish_ids,
            tuple(args.node_names), frame_indices,
        )
        print(f"  done in {time.time() - t0:.1f}s; shape={arr3d.shape}")
        if args.triangulation_npz:
            args.triangulation_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.triangulation_npz, xyz=arr3d,
                frames=np.array(frame_indices, dtype=np.int64),
                fish_ids=np.array(fish_ids),
            )
            print(f"  cached -> {args.triangulation_npz}")

    if args.fill_gap_frames > 0:
        arr3d = interpolate_gaps(arr3d, max_gap=args.fill_gap_frames)
    valid3d = np.isfinite(arr3d[..., 0]).mean()
    print(f"  3D fill rate (post gap-fill): {valid3d*100:.2f}%")

    # 5) Renderer + video writer.
    panel = args.panel_size
    out_w = panel * 2
    out_h = panel * 2
    trail_frames = int(round(args.trail_seconds * args.source_fps / max(1, args.frame_step)))
    renderer3d = SkeletonRenderer3D(
        width=panel, height=panel, xyz_lim=tuple(args.xyz_lim),
        fish_ids=fish_ids, elev=args.elev, azim=args.azim,
        arrows=True, trail_frames=trail_frames,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(
        str(args.out), fps=args.fps, codec="libx264",
        quality=8, macro_block_size=1,
        pixelformat="yuv420p",
    )

    # 6) Open video captures.
    caps: dict[str, "cv2.VideoCapture"] = {}
    cap_widths: dict[str, int] = {}
    cap_heights: dict[str, int] = {}
    cur_idx: dict[str, int] = {}
    for v, p in cap_paths.items():
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            sys.exit(f"failed to open {p}")
        caps[v] = cap
        cap_widths[v] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_heights[v] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cur_idx[v] = 0
        print(f"  {v}: {cap_widths[v]}x{cap_heights[v]} @ "
              f"{cap.get(cv2.CAP_PROP_FRAME_COUNT):.0f} frames")

    fish_colors_bgr = [_hex_to_bgr(FISH_COLORS[i % len(FISH_COLORS)])
                       for i in range(len(fish_ids))]
    panel_views = ("top_cam", "side1_cam", "side2_cam")  # NW, NE, SW; SE = 3D

    print()
    print(f"Rendering 4-panel mp4 -> {args.out}")
    t0 = time.time()
    next_report = t0 + 5.0
    try:
        for f_i, src_frame in enumerate(frame_indices):
            # Read one frame per camera at src_frame (sequential grab+retrieve).
            cam_panels: dict[str, np.ndarray] = {}
            best_frame_info = best.get(src_frame, {})
            for view in panel_views:
                cap = caps[view]
                while cur_idx[view] < src_frame:
                    cap.grab()
                    cur_idx[view] += 1
                ok, frame_bgr = cap.read()
                cur_idx[view] += 1
                if not ok or frame_bgr is None:
                    frame_bgr = np.zeros((cap_heights[view], cap_widths[view], 3),
                                          dtype=np.uint8)
                # Build arrows in source pixel coords.
                arrows = []
                for k, fid in enumerate(fish_ids):
                    m, t = lookup_2d_arrow(best_frame_info, slp_per_view,
                                           src_frame, view, fid)
                    arrows.append((fid, m, t, fish_colors_bgr[k]))
                # Arrow geometry scaled to source resolution; downsamples to
                # ~ (thickness * panel/src) px in the final mp4.
                src_w = cap_widths[view]
                arr_thickness = max(4, int(round(src_w * 0.0075)))
                font_scale = max(0.9, src_w * 0.0015)
                draw_arrows_on_image(frame_bgr, arrows,
                                     thickness=arr_thickness,
                                     tip_length=0.28,
                                     font_scale=font_scale)
                label_panel_top(frame_bgr, view, font_scale=font_scale + 0.2)
                if frame_bgr.shape[0] != panel or frame_bgr.shape[1] != panel:
                    frame_bgr = cv2.resize(frame_bgr, (panel, panel),
                                           interpolation=cv2.INTER_AREA)
                cam_panels[view] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # 3D panel.
            label_txt = f"frame {src_frame}  ({f_i+1}/{len(frame_indices)})"
            panel3d_rgb = renderer3d.update(label_txt, arr3d[f_i])
            if panel3d_rgb.shape[0] != panel or panel3d_rgb.shape[1] != panel:
                panel3d_rgb = cv2.resize(panel3d_rgb, (panel, panel),
                                         interpolation=cv2.INTER_AREA)

            # 2x2 composite: NW=top, NE=side1, SW=side2, SE=3D.
            canvas = np.empty((out_h, out_w, 3), dtype=np.uint8)
            canvas[0:panel, 0:panel] = cam_panels["top_cam"]
            canvas[0:panel, panel:out_w] = cam_panels["side1_cam"]
            canvas[panel:out_h, 0:panel] = cam_panels["side2_cam"]
            canvas[panel:out_h, panel:out_w] = panel3d_rgb
            writer.append_data(canvas)

            now = time.time()
            if now >= next_report:
                dt = now - t0
                rate = (f_i + 1) / dt
                eta = (len(frame_indices) - f_i - 1) / max(rate, 1e-9)
                print(f"  rendered {f_i+1}/{len(frame_indices)}  "
                      f"({rate:.2f} fps, ETA {eta/60:.1f} min)")
                next_report = now + 30.0
    finally:
        writer.close()
        for cap in caps.values():
            cap.release()
        renderer3d.close()
    print(f"Done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
