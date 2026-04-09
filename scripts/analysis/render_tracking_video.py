"""
Overlay processed 2D trajectories on the three synchronized clip videos and
write a combined MP4.

The workflow mirrors the old `analyze_tracking_patterns` behavior:

1. load processed SLEAP CSV files
2. normalize identity labels per frame
3. optionally fill frame-level gaps across the clip range
4. draw the points on each synchronized camera view
5. export a three-view composite video
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

try:
    import cv2
except ImportError:
    cv2 = None


def normalise_individual_to_int(val) -> int:
    """Normalize the `individual` column to integer IDs."""
    if pd.isna(val):
        return -1
    if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
        return int(val)
    s = str(val).strip().lower()
    if s in ("a", "b", "c"):
        return ord(s) - ord("a")
    m = re.match(r"fish_(\d+)", s)
    if m:
        return int(m.group(1))
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return -1


def extract_clip_time_range(path: str | Path) -> Tuple[float, float] | None:
    """Extract the clip time range in seconds from a filename."""
    s = str(path)
    m = re.search(r"(\d+\.\d+)-(\d+\.\d+)", s)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    return None


def load_tracking_data(csv_path: Path) -> pd.DataFrame:
    """Load a SLEAP-exported CSV file that must contain `frame`, `x`, and `y`."""
    df = pd.read_csv(csv_path)
    for c in ["frame", "x", "y"]:
        if c not in df.columns:
            raise ValueError(f"CSV is missing required column: {c}")
    return df


def extract_centroid_per_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract centroids frame by frame and assign temporary individual IDs
    (`0/1/2`).
    """
    frames = sorted(df["frame"].unique())
    required = ["frame", "x", "y"]
    for c in required:
        if c not in df.columns:
            return pd.DataFrame()

    out: List[dict] = []
    prev: Dict[int, np.ndarray] = {}

    for fi in frames:
        sub = df[df["frame"] == fi]
        if sub.empty:
            continue
        pts = sub[["x", "y"]].values.astype(float)
        if "score" in sub.columns:
            sub = sub.sort_values("score", ascending=False)

        if not prev:
            n = min(3, len(sub))
            for k in range(n):
                r = sub.iloc[k]
                prev[k] = np.array([float(r["x"]), float(r["y"])])
                out.append({"frame": fi, "individual": k, "x": prev[k][0], "y": prev[k][1]})
            continue

        prev_pts = np.array([prev[i] for i in sorted(prev)])
        dist = cdist(pts, prev_pts)
        for _ in range(min(len(pts), len(prev_pts))):
            i, j = np.unravel_index(np.argmin(dist), dist.shape)
            d = dist[i, j]
            if d > 100:
                break
            dist[i, :] = np.inf
            dist[:, j] = np.inf
            ind_id = sorted(prev.keys())[j]
            x, y = float(sub.iloc[i]["x"]), float(sub.iloc[i]["y"])
            prev[ind_id] = np.array([x, y])
            out.append({"frame": fi, "individual": ind_id, "x": x, "y": y})
    return pd.DataFrame(out)


def complete_tracks_with_interpolation(
    individuals_df: pd.DataFrame,
    fps: float,
    clip_range: Tuple[float, float],
) -> pd.DataFrame:
    """Fill missing frames over the clip duration with linear interpolation."""
    if individuals_df.empty or fps <= 0:
        return individuals_df
    t0, t1 = clip_range
    dur = max(t1 - t0, 0.0)
    n = int(round(dur * fps))
    if n <= 0:
        return individuals_df

    f0 = int(individuals_df["frame"].min())
    full_frames = np.arange(f0, f0 + n, dtype=int)
    parts = []
    for ind_id in sorted(individuals_df["individual"].unique()):
        sub = individuals_df[individuals_df["individual"] == ind_id].set_index("frame").reindex(full_frames)
        sub["individual"] = ind_id
        sub["is_interpolated"] = sub["x"].isna() | sub["y"].isna()
        sub["x"] = sub["x"].interpolate(method="linear", limit_direction="both")
        sub["y"] = sub["y"].interpolate(method="linear", limit_direction="both")
        parts.append(sub.reset_index())
    return pd.concat(parts, ignore_index=True)


def identify_view_from_path(csv_path: Path) -> str | None:
    """PLAYER1 -> top, PLAYER2 -> side1, PLAYER3 -> side2."""
    n = csv_path.name.upper()
    if "PLAYER1" in n:
        return "top"
    if "PLAYER2" in n:
        return "side1"
    if "PLAYER3" in n:
        return "side2"
    return None


def extract_experiment_id_from_filename(name: str) -> str:
    """
    Remove extension and view/PLAYER suffixes to get a comparable experiment ID.
    """
    s = name.replace(".csv", "").replace(".predictions.csv", "")
    s = re.sub(r"_player[123]_(?:id_assigned|processed)$", "", s, flags=re.I)
    s = re.sub(r"PLAYER[123]", "", s, flags=re.I)
    s = re.sub(r"^(top|side)_[^_]+_\d+_", "", s, flags=re.I)
    return s.strip("_")


def _normalize_exp_id(s: str) -> str:
    """Normalize experiment IDs so repeated underscores do not matter."""
    return re.sub(r"_+", "_", s).strip("_")


def find_view_csvs(input_dir: Path, experiment_id: str) -> Dict[str, Path]:
    """
    Recursively find processed CSV files matching one experiment ID and return
    `{top|side1|side2: path}`.

    Only `*_processed.csv` files are used. The function does not fall back to
    raw SLEAP exports or `_id_assigned` files.
    """
    by_view: Dict[str, List[Path]] = {}
    seen_eids: set[str] = set()
    want_norm = _normalize_exp_id(experiment_id)
    for p in input_dir.rglob("*.csv"):
        view = identify_view_from_path(p)
        if view is None:
            continue
        eid = extract_experiment_id_from_filename(p.name)
        seen_eids.add(eid)
        if (eid == experiment_id or _normalize_exp_id(eid) == want_norm) and "processed" in p.name.lower():
            by_view.setdefault(view, []).append(p)
    if by_view:
        return {v: sorted(fs, key=lambda x: x.name)[-1] for v, fs in by_view.items() if fs}
    sample = sorted(seen_eids)[:8]
    raise FileNotFoundError(
        f"Could not find processed CSV files for experiment ID {experiment_id!r} under {input_dir}\n"
        f"This renderer now uses only *_processed.csv files.\n"
        f"Please confirm that scripts.tracking.process_tracking has already produced them.\n"
        f"Example experiment IDs detected in the directory (for reference only):\n  " + "\n  ".join(sample)
    )


def draw_one_frame(
    img: np.ndarray,
    individuals_df: pd.DataFrame,
    frame_idx: int,
    colors: List[Tuple[int, int, int]],
    simple_dots: bool = False,
) -> np.ndarray:
    """Draw points for one frame on one image."""
    out = img.copy()
    h, w = out.shape[:2]
    sub = individuals_df[individuals_df["frame"] == frame_idx].dropna(subset=["x", "y"])
    dot_color = (255, 255, 255) if simple_dots else None
    for _, row in sub.iterrows():
        k = int(row["individual"])
        x, y = int(round(float(row["x"]))), int(round(float(row["y"])))
        if not (0 <= x < w and 0 <= y < h):
            continue
        color = dot_color if simple_dots else colors[k % len(colors)]
        cv2.circle(out, (x, y), 8, color, 2)
        if not simple_dots:
            label = f"Ind {k}"
            cv2.putText(
                out, label, (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA,
            )
    return out


def draw_trail_panel(
    individuals_df: pd.DataFrame,
    frame_idx: int,
    trail_frames: int,
    width: int,
    height: int,
    colors: List[Tuple[int, int, int]],
    ref_wh: Tuple[int, int] | None,
    simple_dots: bool = False,
) -> np.ndarray:
    """Draw a trail-only panel showing the recent trajectory history."""
    out = np.zeros((height, width, 3), dtype=np.uint8)
    out[:] = (40, 40, 40)
    if individuals_df.empty or trail_frames <= 0:
        return out
    ref_w, ref_h = (ref_wh if ref_wh else (width, height))
    scale_x = width / max(ref_w, 1)
    scale_y = height / max(ref_h, 1)
    line_color = (200, 200, 200) if simple_dots else None

    start_frame = max(0, frame_idx - trail_frames)
    for ind_id in sorted(individuals_df["individual"].unique()):
        seg = individuals_df[
            (individuals_df["individual"] == ind_id)
            & (individuals_df["frame"] >= start_frame)
            & (individuals_df["frame"] <= frame_idx)
        ].sort_values("frame")
        seg = seg.dropna(subset=["x", "y"])
        c = line_color if simple_dots else colors[ind_id % len(colors)]
        if len(seg) < 2:
            pt = seg[["x", "y"]].values
            if len(pt) == 1:
                x, y = int(round(pt[0, 0] * scale_x)), int(round(pt[0, 1] * scale_y))
                cv2.circle(out, (x, y), 6, c, 2)
                if not simple_dots:
                    cv2.putText(out, f"Ind {ind_id}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1, cv2.LINE_AA)
            continue
        pts = seg[["x", "y"]].values
        pts_scaled = (pts * np.array([scale_x, scale_y])).astype(np.int32)
        pts_scaled = np.clip(pts_scaled, (0, 0), (width - 1, height - 1))
        cv2.polylines(out, [pts_scaled], False, c, 2, cv2.LINE_AA)
        x, y = int(pts_scaled[-1, 0]), int(pts_scaled[-1, 1])
        cv2.circle(out, (x, y), 6, c, 2)
        if not simple_dots:
            cv2.putText(out, f"Ind {ind_id}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1, cv2.LINE_AA)
    return out


def render_three_view_video(
    view_data: Dict[str, pd.DataFrame],
    video_paths: Dict[str, Path],
    output_path: Path,
    fps: float,
    trail_duration_s: float = 1.0,
    show_trail_row: bool = True,
    simple_dots: bool = False,
) -> None:
    """
    Top row: three synchronized videos with point overlays.
    Optional bottom row: three trail-only panels.
    """
    if cv2 is None:
        raise SystemExit("opencv-python is required: pip install opencv-python")

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x

    order = ["top", "side1", "side2"]
    caps = {}
    for v in order:
        if v not in video_paths or not video_paths[v].exists():
            raise FileNotFoundError(f"Missing view video: {v} -> {video_paths.get(v)}")
        cap = cv2.VideoCapture(str(video_paths[v]))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_paths[v]}")
        caps[v] = cap

    n_frames = int(min(caps[v].get(cv2.CAP_PROP_FRAME_COUNT) for v in order))
    target_h = 360
    scales = []
    ref_wh_per_view = []
    for v in order:
        w = int(caps[v].get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(caps[v].get(cv2.CAP_PROP_FRAME_HEIGHT)) or target_h
        scale = target_h / h
        new_w = int(w * scale)
        scales.append((new_w, target_h))
        ref_wh_per_view.append((w, h))
    out_w = sum(s[0] for s in scales)
    out_h_row = target_h
    trail_frames = max(1, int(round(trail_duration_s * fps)))
    extra_margin_h = 50
    n_rows = 2 if show_trail_row else 1
    total_h = out_h_row * n_rows + extra_margin_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, total_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    colors = [(0, 255, 0), (0, 165, 255), (255, 0, 0)]  # BGR

    for idx in tqdm(range(n_frames), desc="Writing video", unit="frame", ncols=80):
        row_video = []
        for i, v in enumerate(order):
            cap = caps[v]
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((out_h_row, scales[i][0], 3), dtype=np.uint8)
            elif len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if v in view_data and not view_data[v].empty:
                frame = draw_one_frame(frame, view_data[v], idx, colors, simple_dots=simple_dots)
            frame = cv2.resize(frame, (scales[i][0], scales[i][1]))
            cv2.putText(frame, v, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            row_video.append(frame)

        top_row = np.hstack(row_video)
        if show_trail_row:
            row_trail = []
            for i, v in enumerate(order):
                rw, rh = ref_wh_per_view[i]
                trail_img = draw_trail_panel(
                    view_data.get(v, pd.DataFrame()),
                    idx,
                    trail_frames,
                    scales[i][0],
                    scales[i][1],
                    colors,
                    (rw, rh),
                    simple_dots=simple_dots,
                )
                cv2.putText(trail_img, f"{v} (trail)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
                row_trail.append(trail_img)
            bottom_row = np.hstack(row_trail)
            body = np.vstack([top_row, bottom_row])
            frame_label_y = out_h_row * 2 + int(extra_margin_h * 0.7)
        else:
            body = top_row
            frame_label_y = out_h_row + int(extra_margin_h * 0.7)
        margin_band = np.zeros((extra_margin_h, out_w, 3), dtype=np.uint8)
        combined = np.vstack([body, margin_band])
        cv2.putText(
            combined, f"Frame: {idx}",
            (10, frame_label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(combined)

    for cap in caps.values():
        cap.release()
    writer.release()
    print(f"Wrote: {output_path}")


def resolve_video_paths(
    view_csvs: Dict[str, Path],
    video_dir: Path | None,
    video_top: Path | None,
    video_side1: Path | None,
    video_side2: Path | None,
) -> Dict[str, Path]:
    """Resolve the video path for each view from explicit arguments or a root directory."""
    explicit = {"top": video_top, "side1": video_side1, "side2": video_side2}
    out: Dict[str, Path] = {}
    for v, csv_path in view_csvs.items():
        if explicit.get(v) is not None and Path(explicit[v]).exists():
            out[v] = Path(explicit[v])
            continue
        if video_dir is None or not video_dir.exists():
            raise FileNotFoundError(
                f"Could not find the video for view {v}. Pass --video-dir or --video-{v}."
            )
        stem = csv_path.stem
        parent_name = csv_path.parent.name
        candidates = [
            video_dir / (stem + ".mp4"),
            video_dir / parent_name / (stem + ".mp4"),
            video_dir / parent_name / (csv_path.name.replace(".csv", ".mp4")),
        ]
        for cand in candidates:
            if cand.exists():
                out[v] = cand
                break
        if v in out:
            continue
        for f in video_dir.rglob("*.mp4"):
            if stem in f.stem or f.stem in stem:
                out[v] = f
                break
        if v in out:
            continue
        raise FileNotFoundError(
            f"Could not find the video for view {v}. Tried matching stem={stem!r} "
            f"and directory {parent_name} under {video_dir}. "
            f"You can also pass --video-{v} <path>."
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overlay processed 2D trajectories on clip videos and export a three-view MP4 (uses only *_processed.csv)",
    )
    ap.add_argument("--input-dir", type=Path, required=True, help="Directory containing processed CSV files (searched recursively)")
    ap.add_argument("--processed-dir", type=Path, default=None, help="Optional second processed directory; if provided it overrides matches from --input-dir")
    ap.add_argument("--experiment-id", type=str, required=True, help="Experiment ID")
    ap.add_argument("--video-dir", type=Path, default=None, help="Video root directory")
    ap.add_argument("--video-top", type=Path, default=None, help="Top-view video path")
    ap.add_argument("--video-side1", type=Path, default=None, help="Side1 video path")
    ap.add_argument("--video-side2", type=Path, default=None, help="Side2 video path")
    ap.add_argument("--fps", type=float, required=True, help="Video frame rate, used when filling frames")
    ap.add_argument("--output", type=Path, default=None, help="Output MP4 path (defaults under --output-dir)")
    ap.add_argument("--output-dir", type=Path, default=Path("D:/sleap_output"), help="Output directory")
    ap.add_argument("--trail-duration-s", type=float, default=1.0, help="Trail duration in seconds for the lower row")
    ap.add_argument("--no-trail-row", action="store_true", help="Hide the lower trail row")
    ap.add_argument("--simple-dots", action="store_true", help="Draw plain dots only, without identity labels")
    args = ap.parse_args()

    input_dir = args.input_dir
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    try:
        view_csvs = find_view_csvs(input_dir, args.experiment_id)
    except FileNotFoundError as e:
        raise SystemExit(e) from e

    if args.processed_dir and args.processed_dir.exists():
        try:
            from_processed = find_view_csvs(args.processed_dir, args.experiment_id)
            for v, path in from_processed.items():
                view_csvs[v] = path
        except FileNotFoundError:
            pass

    video_paths = resolve_video_paths(
        view_csvs,
        args.video_dir,
        args.video_top,
        args.video_side1,
        args.video_side2,
    )

    view_data: Dict[str, pd.DataFrame] = {}
    for v, csv_path in view_csvs.items():
        df = load_tracking_data(csv_path)
        if "individual" in df.columns:
            ind = df[["frame", "individual", "x", "y"]].copy()
        elif "fish_id" in df.columns:
            ind = df[["frame", "fish_id", "x", "y"]].copy()
            ind = ind.rename(columns={"fish_id": "individual"})
        else:
            ind = extract_centroid_per_frame(df)
        ind["individual"] = ind["individual"].apply(normalise_individual_to_int)
        ind = ind[ind["individual"] >= 0]
        if ind.empty:
            print(f"  Warning: {v} did not produce valid trajectories; skipping this view")
            continue
        clip_range = extract_clip_time_range(csv_path)
        if clip_range and args.fps > 0:
            ind = complete_tracks_with_interpolation(ind, args.fps, clip_range)
        view_data[v] = ind
        n = csv_path.name.lower()
        src = "processed" if "processed" in n else ("id_assigned" if "id_assigned" in n else "raw")
        print(f"  {v}: {len(ind)} rows, {ind['frame'].nunique()} frames ({src})")

    if not view_data:
        raise SystemExit("No view produced valid trajectories; cannot render a video")

    vp = {v: video_paths[v] for v in view_data if v in video_paths}
    if len(vp) != len(view_data):
        missing = set(view_data) - set(vp)
        raise SystemExit(f"Missing view video(s): {missing}")

    out_dir = args.output_dir
    out_path = args.output if args.output is not None else (out_dir / "tracking_id_video.mp4")
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_three_view_video(
        view_data, vp, out_path, args.fps,
        trail_duration_s=args.trail_duration_s,
        show_trail_row=not args.no_trail_row,
        simple_dots=args.simple_dots,
    )


if __name__ == "__main__":
    main()
