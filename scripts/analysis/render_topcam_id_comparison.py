"""
Render a side-by-side top-cam comparison video for human inspection:

  LEFT panel   : raw top-cam 2D track IDs (fragmenting, 40+ short tracks)
  RIGHT panel  : final 3D-bound fish IDs (3 stable IDs)

Both panels show the same underlying top-down video frame, with markers
and labels overlaid:

  * LEFT  - all top-cam "mouth" detections at this frame, colored and
            labeled by their raw `track_id` (from top_cam.tracks.csv).
  * RIGHT - each fish at this frame, drawn at:
              (a) the bound top-cam detection coords (when bind_source =
                  top_cam_2d) - identity is straight from the top-cam
                  tracker.
              (b) the calibration-projected (u, v) from (X, Y, Z) (when
                  bind_source = geometry_only and 3D is valid) - identity
                  was carried through occlusion by the 3D temporal linker.
              (c) nothing (when both are missing).
            Each marker is colored and labeled by `final_fish_id` (one of
            three stable values).

The right panel also draws short trailing tails (last N frames per fish)
and a per-frame indicator showing how many of the 3 fish are top_cam_2d-
anchored vs geometry_only at this frame, so it is visually obvious when
identity is being held by 3D geometry rather than the top camera.

Tuned for the 35-minute full-experiment run, but works on any segment via
`--start-frame` / `--end-frame`.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections import deque, defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError as e:
    sys.exit(f"opencv-python required: {e}")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.common.local_paths import load_local_paths  # noqa: E402
from scripts.reconstruction_3d.calibration import Calibration  # noqa: E402


FPS = 166

# 3 stable colors for final IDs (30, 31, 34) in BGR.
FINAL_ID_COLOR = {
    30: (40, 200, 255),
    31: (60, 220, 60),
    34: (220, 80, 220),
}


def raw_color(track_id: int) -> Tuple[int, int, int]:
    """Deterministic distinct color per top-cam track_id."""
    rng = np.random.RandomState(int(track_id) * 2654435761 % (2**32))
    h = int(rng.randint(0, 180))
    hsv = np.uint8([[[h, 220, 240]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_marker(img, x, y, color, label, label_color=(0, 0, 0), radius=10,
                with_tail=False, tail_points=None) -> None:
    if not (np.isfinite(x) and np.isfinite(y)):
        return
    xi, yi = int(round(x)), int(round(y))
    if with_tail and tail_points is not None and len(tail_points) >= 2:
        pts = np.asarray(tail_points, dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=False, color=color,
                      thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, (xi, yi), radius, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (xi, yi), radius, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    tx, ty = xi + radius + 4, yi + th // 2
    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                label_color, 1, cv2.LINE_AA)


def draw_text(img, text, org, color=(255, 255, 255), scale=0.8, thickness=2) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, help="top-cam mp4 path")
    ap.add_argument("--top-csv", required=True, help="top_cam.tracks.csv path")
    ap.add_argument("--id-csv", required=True,
                    help="_3d_identified.csv path (must have final_fish_id, "
                         "bind_source, source_top_id, X/Y/Z)")
    ap.add_argument("--calib-dir", default=None,
                    help="calibration model dir (XYZ_to_imCoords.joblib). "
                         "Defaults to paths.local.toml calibration_model_dir.")
    ap.add_argument("--paths-config", default="configs/paths.local.toml")
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=None,
                    help="exclusive; defaults to the full video")
    ap.add_argument("--tail-frames", type=int, default=30,
                    help="how many recent frames to draw as a tail (right panel)")
    ap.add_argument("--fourcc", default="mp4v",
                    help="output FOURCC; mp4v works everywhere, avc1/h264 if codec installed")
    ap.add_argument("--use-ffmpeg", action="store_true",
                    help="pipe frames to ffmpeg for H.264 encoding (much smaller output). "
                         "Requires ffmpeg on PATH.")
    ap.add_argument("--ffmpeg-crf", type=int, default=23,
                    help="ffmpeg CRF (constant rate factor); lower = higher quality. 18-28 typical.")
    ap.add_argument("--progress-every", type=int, default=1000)
    args = ap.parse_args()

    video_path = Path(args.video)
    top_csv = Path(args.top_csv)
    id_csv = Path(args.id_csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.calib_dir is None:
        paths = load_local_paths(Path(args.paths_config))
        calib_dir = paths.calibration_model_dir
    else:
        calib_dir = Path(args.calib_dir)
    calib = Calibration(calib_dir)
    if not calib.has_registration_model:
        sys.exit(f"ERROR: XYZ_to_imCoords.joblib missing in {calib_dir}")

    print(f"Loading {top_csv.name} ...")
    top = pd.read_csv(top_csv)
    top = top[top["node"] == "mouth"][["frame", "track_id", "x", "y"]].copy()
    top["track_id"] = top["track_id"].astype(int)
    top_by_frame = {int(f): sub[["track_id", "x", "y"]].to_numpy()
                    for f, sub in top.groupby("frame")}
    print(f"  {len(top):,} top-cam detections across {len(top_by_frame):,} frames")

    print(f"Loading {id_csv.name} ...")
    idf = pd.read_csv(id_csv)
    idf["final_fish_id"] = idf["final_fish_id"].astype(int)

    need_proj = (idf["bind_source"] == "geometry_only") & np.isfinite(
        idf[["X", "Y", "Z"]].to_numpy()
    ).all(axis=1)
    print(f"  projecting {int(need_proj.sum()):,} geometry_only rows to top_cam ...")
    idf["u_proj"] = np.nan
    idf["v_proj"] = np.nan
    if need_proj.any():
        xyz = idf.loc[need_proj, ["X", "Y", "Z"]].to_numpy(dtype=float)
        uv = calib.project_world_to_view(xyz, "top_cam")
        idf.loc[need_proj, "u_proj"] = uv[:, 0]
        idf.loc[need_proj, "v_proj"] = uv[:, 1]

    idf_top = idf[idf["bind_source"] == "top_cam_2d"].copy()
    top_match = top.rename(columns={"track_id": "source_top_id",
                                    "x": "u_top", "y": "v_top"})
    idf_top["source_top_id"] = idf_top["source_top_id"].astype("Int64")
    top_match["source_top_id"] = top_match["source_top_id"].astype("Int64")
    idf_top = idf_top.merge(top_match, on=["frame", "source_top_id"], how="left")

    idf_geo = idf[idf["bind_source"] != "top_cam_2d"].copy()
    idf_geo["u_top"] = idf_geo["u_proj"]
    idf_geo["v_top"] = idf_geo["v_proj"]

    keep_cols = ["frame", "final_fish_id", "u_top", "v_top", "bind_source"]
    idf_render = pd.concat([idf_top[keep_cols], idf_geo[keep_cols]], ignore_index=True)

    id_by_frame: dict[int, np.ndarray] = {}
    bind_by_frame: dict[int, list[str]] = {}
    for f, sub in idf_render.groupby("frame"):
        valid = np.isfinite(sub["u_top"].to_numpy()) & np.isfinite(sub["v_top"].to_numpy())
        sub_v = sub[valid]
        id_by_frame[int(f)] = sub_v[["final_fish_id", "u_top", "v_top"]].to_numpy()
        bind_by_frame[int(f)] = sub["bind_source"].tolist()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"failed to open video {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    print(f"  video: {width}x{height} @ {src_fps} fps, {total} frames")

    start = max(0, args.start_frame)
    end = total if args.end_frame is None else min(total, args.end_frame)
    if start >= end:
        sys.exit(f"empty range: start={start} end={end}")

    out_w = width * 2
    out_h = height

    ffmpeg_proc = None
    writer = None
    if args.use_ffmpeg:
        ffmpeg_bin = shutil.which("ffmpeg") or r"C:\ffmpeg\ffmpeg-8.0.1-full_build-shared\bin\ffmpeg.exe"
        if not Path(ffmpeg_bin).exists():
            sys.exit("--use-ffmpeg set but no ffmpeg on PATH or known location")
        cmd = [
            ffmpeg_bin, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{out_w}x{out_h}", "-pix_fmt", "bgr24",
            "-r", str(src_fps), "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast",
            "-crf", str(args.ffmpeg_crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_path),
        ]
        print(f"  piping to ffmpeg: {' '.join(cmd[:8])} ... -> {out_path}")
        ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    else:
        fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
        writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (out_w, out_h))
        if not writer.isOpened():
            sys.exit(f"failed to open writer at {out_path} (try --use-ffmpeg)")

    tails: dict[int, deque] = defaultdict(lambda: deque(maxlen=args.tail_frames))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    n_written = 0
    print(f"Rendering frames {start} .. {end-1} ({end - start:,} frames) -> {out_path}")
    for f in range(start, end):
        ok, frame = cap.read()
        if not ok:
            print(f"  early EOF at frame {f}")
            break

        left = frame.copy()
        right = frame.copy()

        dets = top_by_frame.get(f)
        if dets is not None:
            for tid, x, y in dets:
                tid = int(tid)
                color = raw_color(tid)
                draw_marker(left, x, y, color, label=str(tid),
                            label_color=color, radius=9)

        id_rows = id_by_frame.get(f)
        bind_list = bind_by_frame.get(f, [])
        n_top = sum(1 for s in bind_list if s == "top_cam_2d")
        n_geo = sum(1 for s in bind_list if s == "geometry_only")
        n_none = len(bind_list) - n_top - n_geo

        if id_rows is not None:
            for fid, ux, uy in id_rows:
                fid = int(fid)
                color = FINAL_ID_COLOR.get(fid, (255, 255, 255))
                tails[fid].append((int(round(ux)), int(round(uy))))
                draw_marker(right, ux, uy, color, label=f"ID {fid}",
                            label_color=color, radius=14,
                            with_tail=True, tail_points=list(tails[fid]))

        canvas = np.hstack([left, right])
        cv2.line(canvas, (width, 0), (width, height), (60, 60, 60), 2)

        ts = f / src_fps
        m, s = divmod(int(ts), 60)
        h, m = divmod(m, 60)
        clock = f"{h:02d}:{m:02d}:{s:02d}.{int((ts - int(ts))*1000):03d}"
        draw_text(canvas, f"frame {f:>6d}   t = {clock}", (20, 40), scale=1.0, thickness=2)
        draw_text(canvas, "LEFT: raw top-cam 2D track_id",
                  (20, height - 24), scale=0.7, thickness=2)
        draw_text(canvas, "RIGHT: final 3D-bound fish_id",
                  (width + 20, height - 56), scale=0.7, thickness=2)
        bind_msg = f"bind sources this frame: top_cam_2d={n_top}, geometry_only={n_geo}, missing={n_none}"
        bind_color = (255, 255, 255) if n_geo == 0 else (0, 200, 255)
        draw_text(canvas, bind_msg, (width + 20, height - 24),
                  color=bind_color, scale=0.7, thickness=2)

        if ffmpeg_proc is not None:
            ffmpeg_proc.stdin.write(canvas.tobytes())
        else:
            writer.write(canvas)
        n_written += 1
        if n_written % args.progress_every == 0:
            print(f"  {n_written:,} / {end - start:,}  (frame {f})")

    if ffmpeg_proc is not None:
        ffmpeg_proc.stdin.close()
        rc = ffmpeg_proc.wait()
        if rc != 0:
            print(f"  ffmpeg returned non-zero exit {rc}")
    else:
        writer.release()
    cap.release()
    print(f"Wrote {n_written:,} frames to {out_path}")


if __name__ == "__main__":
    main()
