from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.reconstruction_3d.calibration import Calibration
from sam2_id_experiment.src.sleap_adapter.points import load_sleap_points


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render top-camera baseline-vs-sandwich ID review videos around sandwich events."
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--repaired-csv", required=True)
    parser.add_argument("--events-csv", required=True)
    parser.add_argument("--repair-table", default=None)
    parser.add_argument("--tracklets", default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--slp", default=None, help="Optional top_cam .slp used to render mouth points.")
    parser.add_argument("--paths-config", default="configs/paths.local.toml")
    parser.add_argument("--calibration-model-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--event-pad-frames", type=int, default=100)
    parser.add_argument("--tail-frames", type=int, default=30)
    parser.add_argument("--resize-width", type=int, default=960)
    parser.add_argument("--fps-divisor", type=int, default=2, help="Write every Nth source frame.")
    parser.add_argument("--mouth-match-distance-px", type=float, default=80.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    import cv2

    local = load_local_paths(args.paths_config)
    calib_dir = choose_path(args.calibration_model_dir, local.calibration_model_dir if local else None)
    if calib_dir is None:
        raise SystemExit("calibration_model_dir required via paths config or --calibration-model-dir")
    raw_dir = local.raw_data_dir if local else None
    sleap_output_dir = local.sleap_output_dir if local else None
    video_path = Path(args.video) if args.video else _resolve_top_video(raw_dir, args.experiment)
    if video_path is None or not video_path.exists():
        raise FileNotFoundError(f"Could not resolve top-camera video for {args.experiment}: {video_path}")

    events = pd.read_csv(args.events_csv)
    start, end = _resolve_frame_range(
        events,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        pad=int(args.event_pad_frames),
    )

    calib = Calibration(str(calib_dir))
    baseline = _project_id_csv(Path(args.baseline_csv), calib=calib, start=start, end=end)
    repaired = _project_id_csv(Path(args.repaired_csv), calib=calib, start=start, end=end)
    slp_path = Path(args.slp) if args.slp else _resolve_top_slp(sleap_output_dir, args.experiment)
    mouth_points = _load_mouth_points(slp_path, start=start, end=end) if slp_path is not None and slp_path.exists() else {}
    if mouth_points:
        baseline = _snap_rows_to_mouth_points(
            baseline,
            mouth_points,
            max_distance_px=float(args.mouth_match_distance_px),
        )
        repaired = _snap_rows_to_mouth_points(
            repaired,
            mouth_points,
            max_distance_px=float(args.mouth_match_distance_px),
        )
    event_by_frame = _events_by_frame(events, start=start, end=end)
    repair_lookup = _repair_lookup(Path(args.repair_table)) if args.repair_table else {}
    tracklets_by_frame = (
        _tracklets_by_frame(Path(args.tracklets), repair_lookup=repair_lookup, start=start, end=end)
        if args.tracklets
        else {}
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 166.0
    divisor = max(1, int(args.fps_divisor))
    out_fps = src_fps / divisor
    panel_w = int(args.resize_width)
    scale = panel_w / max(1, width)
    panel_h = int(round(height * scale))
    out_size = (panel_w * 2, panel_h)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        out_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {out_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    tails = {
        "baseline": defaultdict(lambda: deque(maxlen=int(args.tail_frames))),
        "repaired": defaultdict(lambda: deque(maxlen=int(args.tail_frames))),
    }
    written = 0
    for frame_idx in range(start, end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if (frame_idx - start) % divisor != 0:
            continue
        frame = cv2.resize(frame, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        left = frame.copy()
        right = frame.copy()
        _draw_id_rows(left, baseline.get(frame_idx, []), tails["baseline"], scale=scale, title="BASELINE 3D final_fish_id")
        _draw_id_rows(right, repaired.get(frame_idx, []), tails["repaired"], scale=scale, title="SANDWICH-REPAIRED final_fish_id")
        _draw_tracklet_rows(right, tracklets_by_frame.get(frame_idx, []), scale=scale)
        canvas = np.hstack([left, right])
        _draw_header(canvas, frame_idx=frame_idx, fps=src_fps, panel_w=panel_w)
        _draw_events(canvas, event_by_frame.get(frame_idx, []), x=20, y=80)
        writer.write(canvas)
        written += 1
    writer.release()
    cap.release()

    summary = {
        "output": str(out_path),
        "experiment": str(args.experiment),
        "video": str(video_path),
        "slp": str(slp_path) if slp_path is not None else "",
        "mouth_overlay": bool(mouth_points),
        "start_frame": int(start),
        "end_frame": int(end),
        "source_fps": float(src_fps),
        "output_fps": float(out_fps),
        "frames_written": int(written),
        "events": int(len(events)),
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _resolve_top_video(raw_dir: Path | None, experiment: str) -> Path | None:
    if raw_dir is None:
        return None
    candidates = [
        raw_dir / experiment / "top_cam" / "cam0_top.mp4",
        raw_dir / experiment / "top_cam" / "top_cam.mp4",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted((raw_dir / experiment / "top_cam").glob("*.mp4")) if (raw_dir / experiment / "top_cam").exists() else []
    return matches[0] if matches else candidates[0]


def _resolve_top_slp(sleap_output_dir: Path | None, experiment: str) -> Path | None:
    if sleap_output_dir is None:
        return None
    candidates = [
        sleap_output_dir / experiment / "round4_260520" / "top_cam.slp",
        sleap_output_dir / experiment / "top_cam.slp",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted((sleap_output_dir / experiment).glob("**/top_cam.slp")) if (sleap_output_dir / experiment).exists() else []
    return matches[0] if matches else candidates[0]


def _resolve_frame_range(events: pd.DataFrame, *, start_frame: int | None, end_frame: int | None, pad: int) -> tuple[int, int]:
    if start_frame is not None and end_frame is not None:
        return int(start_frame), int(end_frame)
    if events.empty:
        if start_frame is None or end_frame is None:
            raise ValueError("Events are empty; provide --start-frame and --end-frame.")
        return int(start_frame), int(end_frame)
    start = int(events["frame_start"].min()) - int(pad) if start_frame is None else int(start_frame)
    end = int(events["frame_end"].max()) + int(pad) if end_frame is None else int(end_frame)
    return max(0, start), max(start, end)


def _project_id_csv(path: Path, *, calib: Calibration, start: int, end: int) -> dict[int, list[dict[str, Any]]]:
    df = pd.read_csv(path)
    df = df[(df["frame"].astype(int) >= int(start)) & (df["frame"].astype(int) <= int(end))].copy()
    df = df[np.isfinite(df[["X", "Y", "Z"]].to_numpy(dtype=float)).all(axis=1)].copy()
    if df.empty:
        return {}
    uv = calib.project_world_to_view(df[["X", "Y", "Z"]].to_numpy(dtype=float), "top_cam")
    df["u"] = uv[:, 0]
    df["v"] = uv[:, 1]
    id_col = "final_fish_id" if "final_fish_id" in df.columns else "fish_id"
    df["render_identity"] = df[id_col].fillna("").astype(str)
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in df.to_dict("records"):
        out[int(row["frame"])].append(row)
    return dict(out)


def _load_mouth_points(slp_path: Path, *, start: int, end: int) -> dict[int, list[dict[str, Any]]]:
    points = load_sleap_points(
        slp_path,
        start_frame=int(start),
        end_frame=int(end),
        anchor_node="mouth",
    )
    if points.empty:
        return {}
    return {
        int(frame): group[["x", "y", "sleap_index", "score", "node"]].to_dict("records")
        for frame, group in points.groupby("frame", sort=True)
    }


def _snap_rows_to_mouth_points(
    rows_by_frame: dict[int, list[dict[str, Any]]],
    mouth_by_frame: dict[int, list[dict[str, Any]]],
    *,
    max_distance_px: float,
) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for frame, rows in rows_by_frame.items():
        mouths = mouth_by_frame.get(int(frame), [])
        if not rows or not mouths:
            out[int(frame)] = [dict(row, render_source="projection") for row in rows]
            continue
        row_xy = np.asarray([[float(row["u"]), float(row["v"])] for row in rows], dtype=float)
        mouth_xy = np.asarray([[float(row["x"]), float(row["y"])] for row in mouths], dtype=float)
        dist = np.linalg.norm(row_xy[:, None, :] - mouth_xy[None, :, :], axis=2)
        assignments = _greedy_nearest(dist, max_distance_px=float(max_distance_px))
        snapped: list[dict[str, Any]] = []
        for row_idx, row in enumerate(rows):
            next_row = dict(row)
            mouth_idx = assignments.get(row_idx)
            if mouth_idx is not None:
                mouth = mouths[mouth_idx]
                next_row["u_projection"] = next_row["u"]
                next_row["v_projection"] = next_row["v"]
                next_row["u"] = float(mouth["x"])
                next_row["v"] = float(mouth["y"])
                next_row["render_source"] = "sleap_mouth"
                next_row["mouth_match_distance_px"] = float(dist[row_idx, mouth_idx])
                next_row["sleap_index"] = int(mouth["sleap_index"])
            else:
                next_row["render_source"] = "projection"
                next_row["mouth_match_distance_px"] = np.nan
            snapped.append(next_row)
        out[int(frame)] = snapped
    return out


def _greedy_nearest(dist: np.ndarray, *, max_distance_px: float) -> dict[int, int]:
    candidates: list[tuple[float, int, int]] = []
    for row_idx in range(dist.shape[0]):
        for col_idx in range(dist.shape[1]):
            d = float(dist[row_idx, col_idx])
            if np.isfinite(d) and d <= float(max_distance_px):
                candidates.append((d, row_idx, col_idx))
    assignments: dict[int, int] = {}
    used_cols: set[int] = set()
    for _d, row_idx, col_idx in sorted(candidates):
        if row_idx in assignments or col_idx in used_cols:
            continue
        assignments[row_idx] = col_idx
        used_cols.add(col_idx)
    return assignments


def _events_by_frame(events: pd.DataFrame, *, start: int, end: int) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events.to_dict("records"):
        lo = max(start, int(event["frame_start"]))
        hi = min(end, int(event["frame_end"]))
        for frame in range(lo, hi + 1):
            out[frame].append(event)
    return dict(out)


def _repair_lookup(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    repair = pd.read_csv(path)
    return {int(row["tracklet_id"]): row for row in repair.to_dict("records")}


def _tracklets_by_frame(
    path: Path,
    *,
    repair_lookup: dict[int, dict[str, Any]],
    start: int,
    end: int,
) -> dict[int, list[dict[str, Any]]]:
    tracklets = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    tracklets = tracklets[(tracklets["frame"].astype(int) >= start) & (tracklets["frame"].astype(int) <= end)].copy()
    keep_ids = set(repair_lookup)
    if keep_ids:
        tracklets = tracklets[tracklets["tracklet_id"].astype(int).isin(keep_ids)].copy()
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in tracklets.to_dict("records"):
        tid = int(row["tracklet_id"])
        rep = repair_lookup.get(tid, {})
        row["repaired_identity"] = str(rep.get("repaired_identity", "") or "")
        row["repair_action"] = str(rep.get("repair_action", "") or "")
        out[int(row["frame"])].append(row)
    return dict(out)


def _draw_id_rows(img, rows: list[dict[str, Any]], tails, *, scale: float, title: str) -> None:
    import cv2

    _text(img, title, (16, img.shape[0] - 22), scale=0.65)
    for row in rows:
        label = str(row["render_identity"])
        color = _color_for_label(label)
        x = int(round(float(row["u"]) * scale))
        y = int(round(float(row["v"]) * scale))
        if not (0 <= x < img.shape[1] and 0 <= y < img.shape[0]):
            continue
        tails[label].append((x, y))
        pts = list(tails[label])
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, color, 2, cv2.LINE_AA)
        cv2.circle(img, (x, y), 9, color, 2, cv2.LINE_AA)
        cv2.circle(img, (x, y), 3, color, -1, cv2.LINE_AA)
        source = "mouth" if str(row.get("render_source", "")) == "sleap_mouth" else "proj"
        _text(img, f"{label} {source}", (x + 10, y - 8), color=color, scale=0.55)


def _draw_tracklet_rows(img, rows: list[dict[str, Any]], *, scale: float) -> None:
    import cv2

    for row in rows:
        action = str(row.get("repair_action", ""))
        if action not in {"sandwich_filled", "model_disagrees_current", "needs_review_possible_swap"}:
            continue
        x = int(round(float(row["centroid_x"]) * scale))
        y = int(round(float(row["centroid_y"]) * scale))
        identity = str(row.get("repaired_identity", ""))
        color = (0, 255, 255) if action == "sandwich_filled" else (0, 120, 255)
        cv2.rectangle(img, (x - 13, y - 13), (x + 13, y + 13), color, 2, cv2.LINE_AA)
        _text(img, f"t{int(row['tracklet_id'])} {identity} {action}", (x + 15, y + 14), color=color, scale=0.45)


def _draw_header(canvas, *, frame_idx: int, fps: float, panel_w: int) -> None:
    import cv2

    cv2.line(canvas, (panel_w, 0), (panel_w, canvas.shape[0]), (80, 80, 80), 2, cv2.LINE_AA)
    seconds = frame_idx / max(fps, 1e-6)
    _text(canvas, f"frame {frame_idx}  t={seconds:0.2f}s", (20, 38), scale=0.8)


def _draw_events(canvas, events: list[dict[str, Any]], *, x: int, y: int) -> None:
    for i, event in enumerate(events[:3]):
        decision = str(event.get("decision", ""))
        color = (0, 255, 255) if decision == "confirmed_identity_bridge" else (0, 150, 255)
        text = (
            f"{decision}: t{event.get('before_tracklet_id')} {event.get('before_identity')} -> "
            f"t{event.get('after_tracklet_id')} {event.get('after_identity')}"
        )
        _text(canvas, text, (x, y + i * 24), color=color, scale=0.55)


def _text(img, text: str, org: tuple[int, int], *, color=(255, 255, 255), scale=0.6) -> None:
    import cv2

    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _color_for_label(label: str) -> tuple[int, int, int]:
    digest = hashlib.md5(label.encode("utf-8")).digest()
    return int(60 + digest[0] % 180), int(60 + digest[1] % 180), int(60 + digest[2] % 180)


if __name__ == "__main__":
    main()
