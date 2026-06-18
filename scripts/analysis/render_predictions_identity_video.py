"""
Render the REAL per-view SLEAP keypoint predictions (mouth + tail), colored by
each fish's unified 3D identity, over a 5k-style window.

Unlike ``render_3d_projection_video`` (which draws a single reprojected
center-of-mass dot per fish), this shows the actual node predictions read from
the per-view ``.slp`` files, so you can see where the model put mouth vs tail
and whether identities stay on the right fish.

Layout (three views side by side, two rows):
* Top row  - live video with the real mouth/tail predictions drawn. Mouth =
  filled dot, tail = ring, joined by a thin segment. Each instance is colored by
  the unified 3D identity it was assigned to (nearest 3D reprojection within
  ``--assign-max-dist`` px); detections matching no 3D fish are drawn gray.
* Bottom row - the 3D-reconstructed identities reprojected into each view as
  colored dots (the "2D reprojection"), same identity colors.

Usage::

    python -m scripts.analysis.render_predictions_identity_video \
        --csv-3d <...>/round5_260611_f171800-176799_3d_identified.csv \
        --slp-dir <...>/round5_260611 \
        --model-folder calibration_results/.../models \
        --video-top <...>/top_cam/cam0_top.mp4 \
        --video-side1 <...>/side1_cam/cam1_side.mp4 \
        --video-side2 <...>/side2_cam/cam2_side.mp4 \
        --output <...>/round5_260611_f171800-176799_predictions_identity.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import sleap_io as sio

from scripts.reconstruction_3d.calibration import Calibration

# view -> (camera projection index, per-view .slp filename)
VIEWS = ["top", "side1", "side2"]
VIEW_CAM_IDX = {"top": 0, "side1": 1, "side2": 2}
VIEW_SLP_NAME = {"top": "top_cam.slp", "side1": "side1_cam.slp", "side2": "side2_cam.slp"}

# BGR identity colors (distinct, high-contrast); gray = unassigned/spurious.
ID_COLORS = [(255, 128, 0), (0, 165, 255), (40, 200, 40), (200, 0, 200)]
GRAY = (160, 160, 160)
TARGET_H = 360


def load_view_predictions(slp_path: Path) -> Dict[int, List[np.ndarray]]:
    """frame_idx -> list of (n_nodes, 2) mouth/tail arrays (full-res px)."""
    labels = sio.load_slp(str(slp_path))
    out: Dict[int, List[np.ndarray]] = {}
    for lf in labels.labeled_frames:
        insts = [inst.numpy()[:, :2] for inst in lf.instances]
        out[int(lf.frame_idx)] = insts
    return out


def reproject_identities(
    df_3d: pd.DataFrame, cal: Calibration, id_col: str
) -> Tuple[Dict[int, Dict[str, List[Tuple[int, float, float]]]], Dict[int, int]]:
    """frame -> view -> list of (identity, x, y); plus identity->color index."""
    ids = sorted(int(v) for v in df_3d[id_col].dropna().unique())
    id_to_cidx = {fid: i for i, fid in enumerate(ids)}
    per_frame: Dict[int, Dict[str, List[Tuple[int, float, float]]]] = {}
    for f, sub in df_3d.groupby("frame"):
        sub = sub.dropna(subset=["X", "Y", "Z"])
        if sub.empty:
            continue
        xyz = sub[["X", "Y", "Z"]].to_numpy(float)
        fids = sub[id_col].to_numpy()
        uv = cal.compute_imCoords_from_XYZ(xyz)  # (n, 3, 2)
        view_pts: Dict[str, List[Tuple[int, float, float]]] = {v: [] for v in VIEWS}
        for i in range(uv.shape[0]):
            if not np.isfinite(fids[i]):
                continue
            fid = int(fids[i])
            for v in VIEWS:
                x, y = float(uv[i, VIEW_CAM_IDX[v], 0]), float(uv[i, VIEW_CAM_IDX[v], 1])
                if np.isfinite(x) and np.isfinite(y):
                    view_pts[v].append((fid, x, y))
        per_frame[int(f)] = view_pts
    return per_frame, id_to_cidx


def assign_identity(centroid: np.ndarray, reproj: List[Tuple[int, float, float]],
                    max_dist: float) -> int | None:
    """Nearest reprojected identity to a real-instance centroid, gated by max_dist."""
    best_id, best_d = None, max_dist
    for fid, x, y in reproj:
        d = float(np.hypot(centroid[0] - x, centroid[1] - y))
        if d < best_d:
            best_d, best_id = d, fid
    return best_id


def draw_real(frame, insts, reproj, id_to_cidx, max_dist):
    for pts in insts:
        finite = pts[np.all(np.isfinite(pts), axis=1)]
        if finite.size == 0:
            continue
        centroid = finite.mean(axis=0)
        fid = assign_identity(centroid, reproj, max_dist)
        color = ID_COLORS[id_to_cidx[fid] % len(ID_COLORS)] if fid is not None else GRAY
        mouth = pts[0] if len(pts) > 0 and np.all(np.isfinite(pts[0])) else None
        tail = pts[1] if len(pts) > 1 and np.all(np.isfinite(pts[1])) else None
        if mouth is not None and tail is not None:
            cv2.line(frame, tuple(mouth.astype(int)), tuple(tail.astype(int)), color, 2, cv2.LINE_AA)
        if mouth is not None:  # mouth = filled dot
            cv2.circle(frame, tuple(mouth.astype(int)), 6, color, -1, cv2.LINE_AA)
        if tail is not None:   # tail = ring
            cv2.circle(frame, tuple(tail.astype(int)), 6, color, 2, cv2.LINE_AA)
    return frame


def draw_reproj(canvas, reproj, id_to_cidx):
    for fid, x, y in reproj:
        color = ID_COLORS[id_to_cidx[fid] % len(ID_COLORS)]
        cv2.circle(canvas, (int(x), int(y)), 8, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(x), int(y)), 8, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-3d", type=Path, required=True)
    ap.add_argument("--slp-dir", type=Path, required=True,
                    help="Directory holding top_cam.slp/side1_cam.slp/side2_cam.slp.")
    ap.add_argument("--model-folder", type=Path, required=True)
    ap.add_argument("--video-top", type=Path, required=True)
    ap.add_argument("--video-side1", type=Path, required=True)
    ap.add_argument("--video-side2", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=166.0)
    ap.add_argument("--assign-max-dist", type=float, default=60.0,
                    help="Max px from a real detection to a reprojected 3D fish "
                         "to inherit its identity color (else gray).")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_3d)
    id_col = "final_fish_id" if "final_fish_id" in df.columns else "fish_id"
    cal = Calibration(args.model_folder)
    reproj_by_frame, id_to_cidx = reproject_identities(df, cal, id_col)
    preds = {v: load_view_predictions(args.slp_dir / VIEW_SLP_NAME[v]) for v in VIEWS}

    f_min, f_max = int(df["frame"].min()), int(df["frame"].max())
    videos = {"top": args.video_top, "side1": args.video_side1, "side2": args.video_side2}
    caps = {v: cv2.VideoCapture(str(videos[v])) for v in VIEWS}
    for v, c in caps.items():
        if not c.isOpened():
            raise SystemExit(f"could not open {videos[v]}")

    w = int(caps["top"].get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(caps["top"].get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = TARGET_H / h
    new_w = int(w * scale)
    out_w, out_h = new_w * 3, TARGET_H * 2 + 30
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (out_w, out_h))
    if not writer.isOpened():
        raise SystemExit(f"could not open writer: {args.output}")

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    read_pos = {v: -10 for v in VIEWS}
    for idx in tqdm(range(f_min, f_max + 1), desc="Rendering", unit="frame", ncols=80):
        reproj = reproj_by_frame.get(idx, {v: [] for v in VIEWS})
        top_imgs, bot_imgs = [], []
        for v in VIEWS:
            cap = caps[v]
            if idx != read_pos[v] + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            read_pos[v] = idx
            if not ret or frame is None:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            rp = reproj.get(v, [])
            frame = draw_real(frame, preds[v].get(idx, []), rp, id_to_cidx, args.assign_max_dist)
            cv2.putText(frame, v, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            top_imgs.append(cv2.resize(frame, (new_w, TARGET_H)))

            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            canvas = draw_reproj(canvas, rp, id_to_cidx)
            cv2.putText(canvas, f"{v} (3D reproj)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
            bot_imgs.append(cv2.resize(canvas, (new_w, TARGET_H)))

        body = np.vstack([np.hstack(top_imgs), np.hstack(bot_imgs)])
        bar = np.zeros((30, out_w, 3), dtype=np.uint8)
        cv2.putText(bar, f"frame {idx}   mouth=dot tail=ring   gray=no 3D match",
                    (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        # identity legend swatches
        x0 = out_w - 360
        for fid, ci in sorted(id_to_cidx.items()):
            cv2.circle(bar, (x0, 15), 7, ID_COLORS[ci % len(ID_COLORS)], -1)
            cv2.putText(bar, f"ID {fid}", (x0 + 12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            x0 += 100
        writer.write(np.vstack([body, bar]))

    writer.release()
    for c in caps.values():
        c.release()
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()
