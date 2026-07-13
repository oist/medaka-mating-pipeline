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
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX, font_scale, max(1, thickness - 2)
        )
        x = min(max(4, pt_mouth[0] + 12), max(4, img_bgr.shape[1] - text_w - 4))
        y = min(
            max(text_h + baseline + 4, pt_mouth[1] - 12),
            img_bgr.shape[0] - baseline - 4,
        )
        label_pt = (int(x), int(y))
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


SEGMENTATION_COLORS_BGR = (
    (255, 170, 60),
    (80, 180, 255),
    (180, 90, 220),
    (80, 210, 150),
    (230, 180, 80),
    (160, 160, 255),
)


def load_segmentation_masks(
    specs: list[str],
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    frame_ranges: list[tuple[int, int]] | None = None,
) -> dict[str, dict[int, list[dict]]]:
    """Load only the requested HDF5 mask frames for each camera view."""
    import h5py

    by_view: dict[str, dict[int, list[dict]]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--segmentation-masks expects view=path, got {spec!r}")
        view, value = spec.split("=", 1)
        if view not in ("top_cam", "side1_cam", "side2_cam"):
            raise ValueError(f"Unknown segmentation-mask view {view!r}")
        path = Path(value)
        frames: dict[int, list[dict]] = {}
        with h5py.File(path, "r") as h5:
            for frame_key in sorted(key for key in h5 if key.startswith("frame_")):
                frame = int(frame_key.split("_")[1])
                if start_frame is not None and frame < start_frame:
                    continue
                if end_frame is not None and frame > end_frame:
                    continue
                if frame_ranges and not any(
                    range_start <= frame <= range_end
                    for range_start, range_end in frame_ranges
                ):
                    continue
                instances = []
                group = h5[frame_key]
                for index, instance_key in enumerate(
                    sorted(key for key in group if key.startswith("instance_"))
                ):
                    instance = group[instance_key]
                    instances.append(
                        {
                            "mask": np.asarray(instance["mask"][...], dtype=bool),
                            "bbox": tuple(
                                int(value) for value in np.asarray(instance["bbox"][...]).tolist()
                            ),
                            "instance_id": int(instance.attrs.get("instance_id", index)),
                            "score": float(instance.attrs.get("score", np.nan)),
                        }
                    )
                frames[frame] = instances
        by_view[view] = frames
    return by_view


def draw_segmentation_masks(
    img_bgr: np.ndarray,
    instances: list[dict],
    *,
    alpha: float = 0.22,
    contour_thickness: int = 2,
    draw_labels: bool = True,
) -> np.ndarray:
    """Overlay translucent instance masks without hiding SLEAP arrows."""
    import cv2

    alpha = float(np.clip(alpha, 0.0, 1.0))
    for index, instance in enumerate(instances):
        mask = np.asarray(instance["mask"], dtype=np.uint8)
        if mask.shape != img_bgr.shape[:2]:
            mask = cv2.resize(
                mask,
                (img_bgr.shape[1], img_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        selected = mask.astype(bool)
        if not selected.any():
            continue
        color = SEGMENTATION_COLORS_BGR[index % len(SEGMENTATION_COLORS_BGR)]
        blended = (
            (1.0 - alpha) * img_bgr[selected].astype(np.float32)
            + alpha * np.asarray(color, dtype=np.float32)
        )
        img_bgr[selected] = np.clip(blended, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            img_bgr,
            contours,
            -1,
            color,
            max(1, int(contour_thickness)),
            cv2.LINE_AA,
        )
        if draw_labels:
            x0, y0, _x1, _y1 = instance.get("bbox", (0, 0, 0, 0))
            score = float(instance.get("score", np.nan))
            label = f"SEG {instance.get('instance_id', index)}"
            if np.isfinite(score):
                label += f" {score:.2f}"
            point = (max(3, int(x0)), max(16, int(y0) - 4))
            cv2.putText(
                img_bgr, label, point, cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (0, 0, 0), 3, cv2.LINE_AA,
            )
            cv2.putText(
                img_bgr, label, point, cv2.FONT_HERSHEY_SIMPLEX,
                0.42, color, 1, cv2.LINE_AA,
            )
    return img_bgr


def load_tracklet_boundary_events(
    specs: list[str],
    *,
    hold_frames: int = 5,
) -> tuple[dict[str, dict[int, list[dict]]], list[dict]]:
    """Load ``view=tracklets.csv`` specs and expand starts/ends for display."""
    import pandas as pd

    by_view: dict[str, dict[int, list[dict]]] = {}
    exact_events: list[dict] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--tracklet-boundaries expects view=path, got {spec!r}")
        view, value = spec.split("=", 1)
        if view not in ("top_cam", "side1_cam", "side2_cam"):
            raise ValueError(f"Unknown boundary view {view!r}")
        path = Path(value)
        table = pd.read_csv(path)
        required = {"frame", "tracklet_id"}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        expanded = by_view.setdefault(view, {})
        summary = table.groupby("tracklet_id")["frame"].agg(["min", "max"])
        for tracklet_id, row in summary.iterrows():
            for kind, boundary_frame in (("START", int(row["min"])), ("END", int(row["max"]))):
                event = {
                    "view": view,
                    "kind": kind,
                    "tracklet_id": str(tracklet_id),
                    "boundary_frame": boundary_frame,
                    "source_path": str(path),
                }
                exact_events.append(event)
                for frame in range(boundary_frame - hold_frames, boundary_frame + hold_frames + 1):
                    expanded.setdefault(frame, []).append(event)
    exact_events.sort(key=lambda e: (e["boundary_frame"], e["view"], e["kind"], e["tracklet_id"]))
    return by_view, exact_events


def draw_tracklet_boundary_indicator(
    img_bgr: np.ndarray,
    events: list[dict],
    *,
    source_frame: int,
) -> None:
    """Draw a persistent, readable warning on a camera panel near a boundary."""
    if not events:
        return
    import cv2

    h, w = img_bgr.shape[:2]
    thickness = max(5, int(round(min(h, w) * 0.008)))
    cv2.rectangle(img_bgr, (2, 2), (w - 3, h - 3), (0, 0, 255), thickness)
    shown = sorted(
        events,
        key=lambda e: (abs(source_frame - int(e["boundary_frame"])), e["kind"]),
    )[:3]
    banner_h = 38 + 32 * len(shown)
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (0, 52), (w, min(h - 1, 52 + banner_h)), (0, 0, 80), -1)
    cv2.addWeighted(overlay, 0.78, img_bgr, 0.22, 0, img_bgr)
    cv2.putText(
        img_bgr, "TRACKLET BOUNDARY", (18, 82), cv2.FONT_HERSHEY_DUPLEX,
        0.85, (255, 255, 255), 2, cv2.LINE_AA,
    )
    for line_i, event in enumerate(shown):
        boundary = int(event["boundary_frame"])
        delta = source_frame - boundary
        suffix = "exact" if delta == 0 else f"{delta:+d} fr"
        text = f"{event['kind']} T{event['tracklet_id']} @ {boundary} ({suffix})"
        color = (0, 255, 255) if event["kind"] == "START" else (255, 128, 255)
        cv2.putText(
            img_bgr, text, (18, 114 + line_i * 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.72, color, 2, cv2.LINE_AA,
        )


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


def lookup_unassigned_2d_arrows(
    best_frame: dict,
    slp_per_view: dict[str, dict[int, np.ndarray]],
    src_frame: int,
    view: str,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Return finite raw SLEAP arrows whose slots have no 3D identity label."""
    from scripts.visualization.render_3d_skeleton_video import VIEW_TO_BEST_KEY

    labels = best_frame.get(VIEW_TO_BEST_KEY[view]) or []
    assigned_slots = {i for i, label in enumerate(labels) if label is not None}
    arr = slp_per_view[view].get(src_frame)
    if arr is None or arr.ndim != 3 or arr.shape[1] < 2:
        return []
    out = []
    for slot in range(arr.shape[0]):
        if slot in assigned_slots:
            continue
        mouth = arr[slot, 0]
        tail = arr[slot, 1]
        if np.isfinite(mouth).all() and np.isfinite(tail).all():
            out.append((mouth, tail, slot))
    return out


def display_labels_from_identified_csv(
    csv_path: Path,
    fish_ids: list[str],
    *,
    id_col: str = "final_fish_id",
) -> list[str]:
    """Map geometry fish IDs from best.json to labels from an identified CSV."""
    import pandas as pd

    df = pd.read_csv(csv_path, usecols=lambda c: c in {"fish_id", id_col})
    if "fish_id" not in df.columns or id_col not in df.columns:
        raise ValueError(
            f"{csv_path} must contain columns 'fish_id' and {id_col!r}."
        )
    out: list[str] = []
    for fid in fish_ids:
        sub = df.loc[df["fish_id"].astype(str) == str(fid), id_col].dropna()
        if sub.empty:
            out.append(str(fid))
            continue
        label = str(sub.astype(str).mode().iloc[0])
        out.append(label)
    return out


def video_writer_kwargs(encoder: str, fps: int) -> dict:
    """Return ImageIO/FFmpeg writer options for CPU or NVIDIA encoding."""
    common = {
        "fps": fps,
        "codec": encoder,
        "macro_block_size": 1,
        "pixelformat": "yuv420p",
    }
    if encoder == "h264_nvenc":
        # Avoid ImageIO's generic quality -> qscale translation, which is not
        # supported by NVENC. CQ 19 is visually close to the former quality=8
        # libx264 output while moving video compression off the CPU.
        return {
            **common,
            "quality": None,
            "output_params": [
                "-preset", "p4",
                "-tune", "hq",
                "-rc", "vbr",
                "-cq", "19",
                "-b:v", "0",
            ],
        }
    if encoder == "libx264":
        return {**common, "quality": 8}
    raise ValueError(f"unsupported video encoder: {encoder}")


def parse_frame_ranges(values: list[str]) -> list[tuple[int, int]]:
    ranges = []
    for value in values:
        separator = ":" if ":" in value else "-"
        if separator not in value:
            raise ValueError(f"--frame-range expects START:END, got {value!r}")
        start_text, end_text = value.split(separator, 1)
        start, end = int(start_text), int(end_text)
        if end < start:
            raise ValueError(f"frame range ends before it starts: {value!r}")
        ranges.append((start, end))
    return ranges


def load_detection_dispositions(path: Path | None) -> dict[tuple[int, str], list[dict]]:
    """Load explicit per-detection states emitted by tracklet-first reconstruction."""
    if path is None:
        return {}
    import pandas as pd
    table = pd.read_csv(path).fillna("")
    required = {"frame", "view", "candidate_slot", "disposition", "primary_reason"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} missing disposition columns: {sorted(missing)}")
    out: dict[tuple[int, str], list[dict]] = {}
    for row in table.to_dict("records"):
        out.setdefault((int(row["frame"]), str(row["view"])), []).append(row)
    return out


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
    ap.add_argument(
        "--frame-range",
        action="append",
        default=[],
        metavar="START:END",
        help=(
            "Render only selected inclusive ranges while loading the enclosing "
            "SLP interval once; repeat to concatenate QC windows."
        ),
    )
    ap.add_argument("--frame-step", type=int, default=10)
    ap.add_argument("--panel-size", type=int, default=1024,
                    help="Width/height of each of the 4 panels.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument(
        "--arrow-thickness",
        type=int,
        default=None,
        help="Source-image arrow thickness. Default scales with video width.",
    )
    ap.add_argument(
        "--arrow-font-scale",
        type=float,
        default=None,
        help="SLEAP identity-label font scale. Default scales with video width.",
    )
    ap.add_argument(
        "--video-encoder",
        choices=("libx264", "h264_nvenc"),
        default="libx264",
        help=(
            "FFmpeg H.264 encoder (default: libx264). Use h264_nvenc on an "
            "NVIDIA GPU to offload compression from the CPU."
        ),
    )
    ap.add_argument("--xyz-lim", nargs=2, type=float, default=(0.0, 220.0))
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--node-names", nargs="+", default=["mouth", "tail"])
    ap.add_argument("--trail-seconds", type=float, default=5.0)
    ap.add_argument("--source-fps", type=float, default=166.0)
    ap.add_argument("--fill-gap-frames", type=int, default=30)
    ap.add_argument("--identified-csv", type=Path, default=None,
                    help="Optional *_3d_identified.csv; if provided, draw "
                         "labels from --display-id-col instead of geometry IDs.")
    ap.add_argument("--display-id-col", default="final_fish_id",
                    help="Column in --identified-csv to use as displayed ID label.")
    ap.add_argument(
        "--tracklet-boundaries", action="append", default=[],
        help="Tracklet table as view=path; repeat for each camera view.",
    )
    ap.add_argument(
        "--boundary-hold-frames", type=int, default=5,
        help="Show each boundary warning for +/- this many source frames (default 5).",
    )
    ap.add_argument(
        "--show-unassigned-sleap", action="store_true",
        help=(
            "Draw filtered SLEAP detections not referenced by the frame's "
            "3D identity mapping as gray UNASSIGNED arrows."
        ),
    )
    ap.add_argument(
        "--detection-dispositions",
        type=Path,
        default=None,
        help=(
            "Tracklet-first disposition CSV. Renders assigned-2D-only, "
            "synthetic, unresolved, and reflection/extra states distinctly."
        ),
    )
    ap.add_argument(
        "--segmentation-masks",
        action="append",
        default=[],
        metavar="VIEW=H5",
        help="Instance-mask HDF5 to overlay; repeat for each available view.",
    )
    ap.add_argument(
        "--mask-alpha",
        type=float,
        default=0.22,
        help="Segmentation fill opacity in [0,1] (default: 0.22).",
    )
    ap.add_argument(
        "--mask-contour-thickness",
        type=int,
        default=2,
        help="Segmentation contour thickness in source pixels (default: 2).",
    )
    ap.add_argument(
        "--no-mask-labels",
        action="store_true",
        help="Hide per-instance SEG labels while retaining mask fills/contours.",
    )
    args = ap.parse_args(argv)
    selected_ranges = parse_frame_ranges(args.frame_range)

    # Imports that pull heavy deps (cv2, matplotlib) only after CLI parsing.
    if __package__ in (None, ""):
        sys.path.append(str(Path(__file__).resolve().parents[2]))
    import cv2
    import imageio.v2 as iio
    from scripts.visualization.render_3d_skeleton_video import (
        FISH_COLORS, SkeletonRenderer3D, VIEWS,
        load_slp_node_xy, build_triangulation, interpolate_gaps,
        TRIANGULATION_CACHE_SLOT_ORDER_VERSION,
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
    print(f"encoder      : {args.video_encoder}")
    print(f"trail        : {args.trail_seconds}s -> "
          f"{int(round(args.trail_seconds * args.source_fps / max(1, args.frame_step)))} rendered frames")
    print(f"unassigned   : {'gray SLEAP overlays enabled' if args.show_unassigned_sleap else 'hidden'}")
    print()

    # 1) best.json
    print("Loading best.json ...")
    with args.best_json.open("r", encoding="utf-8") as fh:
        best_payload = json.load(fh)
    fish_ids = list(best_payload.get("id_labels")
                    or [f"ID_{i+1}" for i in range(int(best_payload.get("num_individuals", 3)))])
    display_ids = fish_ids
    if args.identified_csv is not None:
        display_ids = display_labels_from_identified_csv(
            args.identified_csv,
            fish_ids,
            id_col=args.display_id_col,
        )
    best = {int(k): v for k, v in best_payload["frames"].items()}
    print(f"  {len(best)} frames, fish_ids={fish_ids}")
    if display_ids != fish_ids:
        print(f"  display labels: {dict(zip(fish_ids, display_ids))}")

    boundary_by_view, boundary_events = load_tracklet_boundary_events(
        args.tracklet_boundaries,
        hold_frames=max(0, args.boundary_hold_frames),
    )
    disposition_by_frame_view = load_detection_dispositions(args.detection_dispositions)
    segmentation_by_view = load_segmentation_masks(
        args.segmentation_masks,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        frame_ranges=selected_ranges,
    )
    if boundary_events:
        print(
            f"  tracklet boundaries: {len(boundary_events)} exact start/end events; "
            f"display hold +/-{max(0, args.boundary_hold_frames)} source frames"
        )
    for view, frames in segmentation_by_view.items():
        instance_count = sum(len(instances) for instances in frames.values())
        print(
            f"  segmentation {view}: {len(frames)} frames, "
            f"{instance_count} instances"
        )

    # 2) Per-view .slp (mouth + tail xy in reconstruction-filtered slot order).
    slp_per_view: dict[str, dict[int, np.ndarray]] = {}
    for view in VIEWS:
        slp_path = args.slp_dir / f"{view}.slp"
        print(f"Loading {slp_path} ...")
        data_v, sk = load_slp_node_xy(
            slp_path,
            node_names=tuple(args.node_names),
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
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
    if selected_ranges:
        all_frames = [
            frame
            for frame in all_frames
            if any(start <= frame <= end for start, end in selected_ranges)
        ]
    frame_indices = all_frames[::max(1, args.frame_step)]
    print(f"  rendering {len(frame_indices)} frames "
          f"(every {args.frame_step}th from {len(all_frames)} sources)")

    # 4) 3D triangulation: load cache or compute on the fly.
    if args.triangulation_npz and args.triangulation_npz.exists():
        print(f"Loading triangulation cache: {args.triangulation_npz}")
        data = np.load(args.triangulation_npz, allow_pickle=False)
        cache_version = (
            int(np.asarray(data["slot_order_version"]).reshape(-1)[0])
            if "slot_order_version" in data.files
            else 1
        )
        if cache_version != TRIANGULATION_CACHE_SLOT_ORDER_VERSION:
            print(
                f"  WARNING: stale cache slot-order version {cache_version}; "
                "recomputing."
            )
            arr3d = None
        else:
            arr3d = data["xyz"]
            cached_frames = data["frames"].tolist()
        if arr3d is not None and cached_frames != frame_indices:
            print(f"  WARNING: cache covers {len(cached_frames)} frames; "
                  f"current selection covers {len(frame_indices)} frames -- "
                  f"recomputing to match.")
            arr3d = None
        elif arr3d is not None:
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
                slot_order_version=np.array(
                    [TRIANGULATION_CACHE_SLOT_ORDER_VERSION], dtype=np.int64
                ),
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
        fish_ids=display_ids, elev=args.elev, azim=args.azim,
        arrows=True, trail_frames=trail_frames,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(
        str(args.out), **video_writer_kwargs(args.video_encoder, args.fps)
    )

    # 6) Open video captures.
    caps: dict[str, "cv2.VideoCapture"] = {}
    cap_widths: dict[str, int] = {}
    cap_heights: dict[str, int] = {}
    cur_idx: dict[str, int] = {}
    first_render_frame = frame_indices[0] if frame_indices else 0
    for v, p in cap_paths.items():
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            sys.exit(f"failed to open {p}")
        if first_render_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_render_frame)
        caps[v] = cap
        cap_widths[v] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_heights[v] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cur_idx[v] = first_render_frame
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
                draw_segmentation_masks(
                    frame_bgr,
                    segmentation_by_view.get(view, {}).get(src_frame, []),
                    alpha=args.mask_alpha,
                    contour_thickness=args.mask_contour_thickness,
                    draw_labels=not args.no_mask_labels,
                )
                # Build arrows in source pixel coords.
                arrows = []
                for k, fid in enumerate(fish_ids):
                    m, t = lookup_2d_arrow(best_frame_info, slp_per_view,
                                           src_frame, view, fid)
                    arrows.append((display_ids[k], m, t, fish_colors_bgr[k]))
                explicit_dispositions = disposition_by_frame_view.get((src_frame, view), [])
                if explicit_dispositions:
                    arr = slp_per_view[view].get(src_frame)
                    color_by_id = {
                        str(fid): fish_colors_bgr[k] for k, fid in enumerate(fish_ids)
                    }
                    for disposition in explicit_dispositions:
                        state = str(disposition.get("disposition", ""))
                        if state == "assigned_2d_valid_3d":
                            continue
                        slot = int(disposition["candidate_slot"])
                        if arr is None or slot < 0 or slot >= arr.shape[0] or arr.shape[1] < 2:
                            continue
                        mouth, tail = arr[slot, 0], arr[slot, 1]
                        if not (np.isfinite(mouth).all() and np.isfinite(tail).all()):
                            continue
                        identity = str(disposition.get("final_fish_id", ""))
                        if state == "assigned_2d_only":
                            label = f"{identity} [2D]"
                            color = color_by_id.get(identity, (0, 200, 255))
                        elif state == "assigned_3d_synthetic_missing_view":
                            label = f"{identity} [2V->3D]"
                            color = (255, 255, 0)
                        elif state == "unresolved_detection":
                            label, color = "UNRESOLVED", (0, 180, 255)
                        else:
                            label, color = "EXTRA/REFL", (120, 120, 120)
                        arrows.append((label, mouth, tail, color))
                elif args.show_unassigned_sleap:
                    for m, t, _slot in lookup_unassigned_2d_arrows(
                        best_frame_info, slp_per_view, src_frame, view
                    ):
                        arrows.append(("UNASSIGNED", m, t, (160, 160, 160)))
                # Arrow geometry scaled to source resolution; downsamples to
                # ~ (thickness * panel/src) px in the final mp4.
                src_w = cap_widths[view]
                arr_thickness = (
                    max(1, int(args.arrow_thickness))
                    if args.arrow_thickness is not None
                    else max(4, int(round(src_w * 0.0075)))
                )
                font_scale = (
                    max(0.2, float(args.arrow_font_scale))
                    if args.arrow_font_scale is not None
                    else max(0.9, src_w * 0.0015)
                )
                draw_arrows_on_image(frame_bgr, arrows,
                                     thickness=arr_thickness,
                                     tip_length=0.28,
                                     font_scale=font_scale)
                panel_label = (
                    f"{view} + SEG"
                    if view in segmentation_by_view
                    else view
                )
                label_panel_top(frame_bgr, panel_label, font_scale=max(0.8, font_scale + 0.2))
                draw_tracklet_boundary_indicator(
                    frame_bgr,
                    boundary_by_view.get(view, {}).get(src_frame, []),
                    source_frame=src_frame,
                )
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
