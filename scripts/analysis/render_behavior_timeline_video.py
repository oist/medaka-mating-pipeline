from __future__ import annotations

"""
Composite three-view video plus a BORIS behavior timeline.

Overview:

- input: top-view video path plus the corresponding BORIS event CSV
- automatic inference of side1 / side2 video paths from the top clip
- output:
  - first row: raw top / side1 / side2 videos placed side by side
  - second row: one behavior timeline spanning the full clip
    - different behaviors get different colors
    - the legend shows "color -> subject : behavior"
    - a moving cursor indicates the current time
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class BehaviorEvent:
    start_s: float
    end_s: float
    behavior: str
    subject: str

    @property
    def label(self) -> str:
        """Return the combined subject + behavior label used in the legend."""
        b = self.behavior or "UnknownBehavior"
        s = self.subject or "UnknownSubject"
        return f"{s}: {b}"


def infer_three_view_set_from_top(
    video_top: Path,
    video_side1: Optional[Path] = None,
    video_side2: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Infer side1 / side2 videos from the top-view video path.

    Expected directory layout:
        D:/videos_raw/<session>/
            top_cam/cam0_top.mp4
            side1_cam/cam1_side.mp4
            side2_cam/cam2_side.mp4

    Explicit `video_side1` / `video_side2` arguments take precedence.
    """
    video_top = Path(video_top).resolve()
    if not video_top.exists():
        raise FileNotFoundError(f"Top-view video does not exist: {video_top}")

    root = video_top.parents[1] if len(video_top.parents) >= 2 else video_top.parent

    out: Dict[str, Path] = {"top": video_top}

    if video_side1 is not None and Path(video_side1).exists():
        out["side1"] = Path(video_side1).resolve()
    if video_side2 is not None and Path(video_side2).exists():
        out["side2"] = Path(video_side2).resolve()
    if len(out) == 3:
        return out

    direct_candidates = {
        "side1": root / "side1_cam" / "cam1_side.mp4",
        "side2": root / "side2_cam" / "cam2_side.mp4",
    }
    for view, cand in direct_candidates.items():
        if view in out:
            continue
        if cand.exists():
            out[view] = cand.resolve()

    if len(out) == 3:
        return out

    all_mp4 = list(root.rglob("*.mp4"))

    def pick_one(candidates: List[Path]) -> Optional[Path]:
        if not candidates:
            return None
        return sorted(candidates, key=lambda p: str(p))[-1].resolve()

    if "side1" not in out:
        side1_cands = [
            p for p in all_mp4
            if re.search(r"(side1|cam1_side)", p.stem, flags=re.IGNORECASE)
        ]
        p1 = pick_one(side1_cands)
        if p1 is not None:
            out["side1"] = p1

    if "side2" not in out:
        side2_cands = [
            p for p in all_mp4
            if re.search(r"(side2|cam2_side)", p.stem, flags=re.IGNORECASE)
        ]
        p2 = pick_one(side2_cands)
        if p2 is not None:
            out["side2"] = p2

    missing = [v for v in ["side1", "side2"] if v not in out]
    if missing:
        raise FileNotFoundError(
            f"Could not infer the following view videos under {root}: {missing}\n"
            f"top: {video_top}\n"
            f"Please provide --video-side1 / --video-side2 explicitly."
        )

    return out


def _find_first_col(cols: List[str], *keywords: str) -> Optional[str]:
    """Fuzzy-match the first column containing all keywords."""
    lower_map = {c.lower(): c for c in cols}
    for lc, orig in lower_map.items():
        if all(k in lc for k in keywords):
            return orig
    return None


def load_boris_events(csv_path: Path, fps: float | None = None) -> List[BehaviorEvent]:
    """
    Load BORIS events and normalize them into a list of `BehaviorEvent`.

    Supported formats include:
    1. interval events with `Start/Stop/Duration`
    2. point events with `Time`
    3. frame-index events where `frame / fps` is used as time
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"BORIS event CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    cols = list(df.columns)

    start_col = (
        _find_first_col(cols, "start", "(s")
        or _find_first_col(cols, "start")
        or ("Start (s)" if "Start (s)" in df.columns else None)
    )
    stop_col = (
        _find_first_col(cols, "stop", "(s")
        or _find_first_col(cols, "stop")
        or None
    )
    duration_col = (
        _find_first_col(cols, "duration", "(s")
        or _find_first_col(cols, "duration")
        or None
    )

    time_col = None
    frame_col = None
    if start_col is None or start_col not in df.columns:
        time_col = (
            _find_first_col(cols, "time", "(s")
            or _find_first_col(cols, "time")
        )
        if time_col is None or time_col not in df.columns:
            frame_col = _find_first_col(cols, "frame")
            if frame_col is None or frame_col not in df.columns:
                raise ValueError(
                    "BORIS CSV does not contain Start/Time/Frame columns, "
                    f"so the event timeline cannot be inferred. Current columns: {cols}"
                )

    behavior_col = (
        _find_first_col(cols, "behavior")
        or _find_first_col(cols, "behav")
        or ("Behavior" if "Behavior" in df.columns else None)
    )
    if behavior_col is None or behavior_col not in df.columns:
        raise ValueError(
            f"BORIS CSV does not contain a behavior column such as 'Behavior'. Current columns: {cols}"
        )

    subject_col = (
        _find_first_col(cols, "subject")
        or _find_first_col(cols, "individual")
        or _find_first_col(cols, "id")
    )

    events: List[BehaviorEvent] = []
    for _, row in df.iterrows():
        start_s: Optional[float]
        if start_col and start_col in df.columns:
            try:
                start_s = float(row[start_col])
            except Exception:
                start_s = None
        elif time_col and time_col in df.columns:
            try:
                start_s = float(row[time_col])
            except Exception:
                start_s = None
        elif frame_col and frame_col in df.columns and fps and fps > 0:
            try:
                start_s = float(row[frame_col]) / float(fps)
            except Exception:
                start_s = None
        else:
            start_s = None

        if start_s is None or np.isnan(start_s):
            continue

        end_s: Optional[float] = None
        if stop_col and stop_col in df.columns:
            try:
                end_s = float(row[stop_col])
            except Exception:
                end_s = None
        if (end_s is None or not np.isfinite(end_s)) and duration_col and duration_col in df.columns:
            try:
                dur = float(row[duration_col])
                if np.isfinite(dur):
                    end_s = start_s + dur
            except Exception:
                end_s = None

        if end_s is None or not np.isfinite(end_s):
            end_s = start_s + (1.0 / float(fps) if fps and fps > 0 else 0.1)

        behavior = str(row[behavior_col]) if not pd.isna(row[behavior_col]) else ""
        if subject_col and subject_col in df.columns:
            subject_val = row[subject_col]
            subject = str(subject_val) if not pd.isna(subject_val) else ""
        else:
            subject = ""

        events.append(
            BehaviorEvent(
                start_s=float(start_s),
                end_s=float(end_s),
                behavior=behavior,
                subject=subject,
            )
        )

    return [e for e in events if np.isfinite(e.start_s) and np.isfinite(e.end_s) and e.end_s > e.start_s]


def _build_color_map(labels: List[str]) -> Dict[str, Tuple[int, int, int]]:
    """Assign one BGR color per label."""
    unique = sorted(set(labels))
    if not unique:
        return {}

    palette = [
        (0, 255, 0),
        (0, 165, 255),
        (255, 0, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
        (128, 0, 255),
        (0, 128, 255),
        (0, 255, 128),
        (255, 128, 0),
    ]
    color_map: Dict[str, Tuple[int, int, int]] = {}
    for i, label in enumerate(unique):
        color_map[label] = palette[i % len(palette)]
    return color_map


def build_timeline_base_image(
    events: List[BehaviorEvent],
    total_duration_s: float,
    width: int,
    *,
    bar_height: int = 60,
    legend_height: int = 80,
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int, int]], Tuple[int, int, int, int]]:
    """
    Build the timeline base image without the moving cursor and return:
    - base image
    - color map
    - bar bounding box `(x0, y0, x1, y1)`
    """
    h = bar_height + legend_height + 20
    img = np.zeros((h, width, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)

    if total_duration_s <= 0 or not events:
        x0, x1 = 40, width - 20
        y0, y1 = 20, 20 + bar_height
        cv2.rectangle(img, (x0, y0), (x1, y1), (80, 80, 80), 1)
        return img, {}, (x0, y0, x1, y1)

    x0, x1 = 40, width - 20
    y0, y1 = 20, 20 + bar_height
    bar_width = max(1, x1 - x0)

    labels = [e.label for e in events]
    color_map = _build_color_map(labels)

    cv2.rectangle(img, (x0, y0), (x1, y1), (100, 100, 100), 1)

    for e in events:
        start_s = max(0.0, min(e.start_s, total_duration_s))
        end_s = max(0.0, min(e.end_s, total_duration_s))
        if end_s <= start_s:
            continue

        xs = x0 + int(round(start_s / total_duration_s * bar_width))
        xe = x0 + int(round(end_s / total_duration_s * bar_width))
        xe = max(xs + 1, min(xe, x1))

        color = color_map.get(e.label, (200, 200, 200))
        cv2.rectangle(img, (xs, y0 + 2), (xe, y1 - 2), color, thickness=-1)

    legend_y = y1 + 10
    line_height = 20
    x_color = 40
    x_text = 65
    for i, label in enumerate(sorted(color_map.keys())):
        yy = legend_y + i * line_height
        if yy + line_height >= h:
            break
        color = color_map[label]
        cv2.rectangle(img, (x_color, yy), (x_color + 18, yy + 18), color, thickness=-1)
        cv2.putText(
            img,
            label,
            (x_text, yy + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    return img, color_map, (x0, y0, x1, y1)


def format_time_mm_ss_ff(frame_idx: int, fps: float) -> str:
    """Format the current time as `mm:ss:ff`."""
    if fps <= 0:
        return "00:00:00"
    t = frame_idx / fps
    minutes = int(t // 60)
    seconds = int(t % 60)
    frac = t - int(t)
    frames = int(round(frac * fps))
    return f"{minutes:02d}:{seconds:02d}:{frames:02d}"


def draw_cursor_on_timeline(
    base_img: np.ndarray,
    frame_idx: int,
    fps: float,
    total_duration_s: float,
    bar_box: Tuple[int, int, int, int],
) -> np.ndarray:
    """Draw the current-time cursor on top of the timeline base image."""
    x0, y0, x1, y1 = bar_box
    bar_width = max(1, x1 - x0)
    out = base_img.copy()

    if fps <= 0 or total_duration_s <= 0:
        return out

    t = frame_idx / fps
    t = max(0.0, min(t, total_duration_s))
    x = x0 + int(round(t / total_duration_s * bar_width))
    x = max(x0, min(x, x1))

    cv2.line(out, (x, y0 - 2), (x, y1 + 2), (255, 255, 255), 2, cv2.LINE_AA)

    time_str = format_time_mm_ss_ff(frame_idx, fps)
    cv2.putText(
        out,
        time_str,
        (x0, y0 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def render_three_view_with_behavior_timeline(
    video_top: Path | str,
    boris_csv: Path | str,
    output_path: Path | str,
    *,
    video_side1: Path | str | None = None,
    video_side2: Path | str | None = None,
    max_frames: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> Path:
    """
    Combine the three raw videos with one BORIS event timeline.

    Parameters
    ----------
    video_top:
        Top-view mp4 path (required)
    boris_csv:
        BORIS event CSV path (required)
    output_path:
        Output MP4 path
    video_side1 / video_side2:
        Optional explicit side-view video paths. If omitted, they are inferred.
    """
    if cv2 is None:
        raise SystemExit("opencv-python is required: pip install opencv-python")

    video_top = Path(video_top)
    boris_csv = Path(boris_csv)
    output_path = Path(output_path)

    views = infer_three_view_set_from_top(video_top, video_side1, video_side2)

    order = ["top", "side1", "side2"]
    caps: Dict[str, "cv2.VideoCapture"] = {}
    for v in order:
        p = views.get(v)
        if p is None or not p.exists():
            raise FileNotFoundError(f"Missing view video {v}: {p}")
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {v}: {p}")
        caps[v] = cap

    cap_top = caps["top"]
    fps = float(cap_top.get(cv2.CAP_PROP_FPS)) or 0.0
    if fps <= 0:
        fps = 30.0

    total_frames = int(min(caps[v].get(cv2.CAP_PROP_FRAME_COUNT) for v in order))
    if total_frames <= 0:
        raise RuntimeError("Video frame count is zero; cannot build the composite")

    sf = 0 if start_frame is None else max(0, int(start_frame))
    ef = total_frames - 1 if end_frame is None else min(total_frames - 1, int(end_frame))
    if max_frames is not None and max_frames > 0:
        ef = min(ef, sf + int(max_frames) - 1)
    if ef < sf:
        raise RuntimeError(
            f"Invalid frame range: start_frame={start_frame}, end_frame={end_frame}, max_frames={max_frames}"
        )

    total_duration_s = total_frames / fps

    target_h = 360
    scales: List[Tuple[int, int]] = []
    for v in order:
        cap = caps[v]
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or target_h
        scale = target_h / h
        new_w = int(w * scale)
        scales.append((new_w, target_h))

    out_w = sum(s[0] for s in scales)
    timeline_h = 160
    out_h = target_h + timeline_h

    events = load_boris_events(boris_csv, fps=fps)
    base_timeline, _, bar_box = build_timeline_base_image(
        events,
        total_duration_s=total_duration_s,
        width=out_w,
        bar_height=60,
        legend_height=80,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = lambda x, **kw: x

    for idx in tqdm(range(sf, ef + 1), desc="Writing behavior timeline video", unit="frame", ncols=80):
        row_video: List[np.ndarray] = []
        for i, v in enumerate(order):
            cap = caps[v]
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((target_h, scales[i][0], 3), dtype=np.uint8)
            elif len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            frame = cv2.resize(frame, (scales[i][0], scales[i][1]))
            cv2.putText(
                frame,
                v,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            row_video.append(frame)

        top_row = np.hstack(row_video)

        timeline = draw_cursor_on_timeline(
            base_timeline,
            frame_idx=idx,
            fps=fps,
            total_duration_s=total_duration_s,
            bar_box=bar_box,
        )

        if timeline.shape[0] < timeline_h:
            pad_h = timeline_h - timeline.shape[0]
            pad = np.zeros((pad_h, out_w, 3), dtype=np.uint8)
            pad[:] = (30, 30, 30)
            timeline = np.vstack([timeline, pad])
        elif timeline.shape[0] > timeline_h:
            timeline = timeline[:timeline_h, :, :]

        combined = np.vstack([top_row, timeline])
        writer.write(combined)

    for cap in caps.values():
        cap.release()
    writer.release()

    print(f"Wrote behavior visualization video: {output_path}")
    return output_path


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Infer the three synchronized views from the top-view clip, overlay a BORIS behavior "
            "timeline, and export a two-row composite video."
        )
    )
    p.add_argument("--video-top", type=Path, required=True, help="Top-view video path, for example D:/videos_raw/.../top_cam/cam0_top.mp4")
    p.add_argument("--boris-csv", type=Path, required=True, help="BORIS event CSV")
    p.add_argument("--output", type=Path, required=True, help="Output MP4 path")
    p.add_argument("--video-side1", type=Path, default=None, help="Optional side1 video path")
    p.add_argument("--video-side2", type=Path, default=None, help="Optional side2 video path")
    p.add_argument("--start-frame", type=int, default=None, help="First frame to include")
    p.add_argument("--end-frame", type=int, default=None, help="Last frame to include")
    p.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to write")
    args = p.parse_args()

    render_three_view_with_behavior_timeline(
        video_top=args.video_top,
        boris_csv=args.boris_csv,
        output_path=args.output,
        video_side1=args.video_side1,
        video_side2=args.video_side2,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
