from __future__ import annotations

"""
Frame-by-frame nearest-neighbor tracking from SLEAP predictions.

Each frame's predicted instances are matched to the previous frame's track
positions via the Hungarian algorithm under a max-distance gate. Tracks that
go unseen for more than `max_gap` frames are retired so a re-appearing fish
gets a fresh ID instead of stealing a stale one.

The model in use here predicts a single keypoint (`mouth`) per fish, so the
"position" is just that keypoint. With more keypoints later, swap in a
centroid in `_instance_position`.

Outputs a tidy CSV: frame, track_id, x, y, score, node.
"""

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths


@dataclass
class _ActiveTrack:
    track_id: int
    last_frame: int
    last_pos: tuple[float, float]
    history: list[tuple[int, float, float, float | None]] = field(default_factory=list)


@dataclass(frozen=True)
class MotionAwareConfig:
    enabled: bool = False
    rolling_window: int = 5
    close_contact_distance_px: float = 35.0
    close_contact_min_margin: float = 12.0
    max_speed_ratio: float = 4.0
    zigzag_speed_ratio: float = 3.0
    zigzag_heading_rad: float = 2.35
    speed_weight: float = 8.0
    direction_weight: float = 16.0
    angular_weight: float = 8.0
    score_weight: float = 4.0
    local_repair: bool = True


BASE_FIELDS = ["frame", "track_id", "x", "y", "score", "node"]
DIAGNOSTIC_FIELDS = [
    "match_cost",
    "motion_distance_px",
    "one_step_distance_px",
    "rolling_speed_px",
    "speed_ratio",
    "heading_change_rad",
    "close_contact",
    "nearest_neighbor_distance_px",
    "zigzag_flag",
    "ambiguity_score",
    "bridge_reason",
    "repair_action",
]


def _instance_position(
    inst, *, node_names: list[str] | None = None, anchor_node: str | None = None,
) -> tuple[float, float] | None:
    """Position of an instance for tracking.

    If `anchor_node` is given and that node is present + finite for this
    instance, return that node's xy directly. Otherwise fall back to the
    NaN-safe centroid of all visible nodes (legacy default behaviour).
    """
    pts = inst.numpy()  # (n_nodes, 2)
    if anchor_node is not None and node_names is not None:
        try:
            idx = node_names.index(anchor_node)
            pt = pts[idx]
            if np.isfinite(pt).all():
                return float(pt[0]), float(pt[1])
        except ValueError:
            pass
    mask = np.isfinite(pts).all(axis=1)
    if not mask.any():
        return None
    centroid = pts[mask].mean(axis=0)
    return float(centroid[0]), float(centroid[1])


def _record_detection(track: _ActiveTrack, frame: int, det: dict) -> None:
    track.last_frame = int(frame)
    track.last_pos = (float(det["x"]), float(det["y"]))
    score = det.get("score")
    track.history.append((
        int(frame),
        float(det["x"]),
        float(det["y"]),
        None if score is None or not np.isfinite(score) else float(score),
    ))


def _predict_position(track: _ActiveTrack, frame: int) -> np.ndarray:
    last = np.asarray(track.last_pos, dtype=float)
    if len(track.history) < 2:
        return last
    f0, x0, y0, _ = track.history[-2]
    f1, x1, y1, _ = track.history[-1]
    gap = max(1, int(f1) - int(f0))
    vel = np.asarray([x1 - x0, y1 - y0], dtype=float) / float(gap)
    return last + vel * float(max(1, int(frame) - int(track.last_frame)))


def _recent_speeds(track: _ActiveTrack, *, window: int) -> np.ndarray:
    hist = track.history[-max(2, int(window) + 1):]
    speeds: list[float] = []
    for prev, curr in zip(hist, hist[1:]):
        f0, x0, y0, _ = prev
        f1, x1, y1, _ = curr
        gap = max(1, int(f1) - int(f0))
        speeds.append(float(np.hypot(x1 - x0, y1 - y0)) / float(gap))
    return np.asarray(speeds, dtype=float)


def _last_velocity(track: _ActiveTrack) -> np.ndarray | None:
    if len(track.history) < 2:
        return None
    f0, x0, y0, _ = track.history[-2]
    f1, x1, y1, _ = track.history[-1]
    gap = max(1, int(f1) - int(f0))
    return np.asarray([x1 - x0, y1 - y0], dtype=float) / float(gap)


def _angle_between(v0: np.ndarray | None, v1: np.ndarray) -> float:
    if v0 is None:
        return 0.0
    n0 = float(np.linalg.norm(v0))
    n1 = float(np.linalg.norm(v1))
    if n0 < 1e-6 or n1 < 1e-6:
        return 0.0
    cos_angle = float(np.dot(v0, v1) / (n0 * n1))
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.arccos(cos_angle))


def _add_detection_spacing(
    detections: list[dict], *, close_contact_distance_px: float,
) -> None:
    if not detections:
        return
    if len(detections) == 1:
        detections[0]["nearest_neighbor_distance_px"] = np.nan
        detections[0]["close_contact"] = False
        return
    pts = np.asarray([[d["x"], d["y"]] for d in detections], dtype=float)
    dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest = np.min(dist, axis=1)
    for det, nn in zip(detections, nearest):
        det["nearest_neighbor_distance_px"] = float(nn)
        det["close_contact"] = bool(nn <= float(close_contact_distance_px))


def _motion_diagnostics(
    track: _ActiveTrack,
    det: dict,
    *,
    frame: int,
    cfg: MotionAwareConfig,
) -> dict:
    pred = _predict_position(track, frame)
    det_pos = np.asarray([det["x"], det["y"]], dtype=float)
    last_pos = np.asarray(track.last_pos, dtype=float)
    frame_gap = max(1, int(frame) - int(track.last_frame))
    one_step_distance = float(np.linalg.norm(det_pos - last_pos))
    one_step_speed = one_step_distance / float(frame_gap)
    motion_distance = float(np.linalg.norm(det_pos - pred))

    recent = _recent_speeds(track, window=cfg.rolling_window)
    rolling_speed = float(np.nanmedian(recent)) if recent.size else np.nan
    baseline_speed = rolling_speed if np.isfinite(rolling_speed) and rolling_speed > 1e-6 else one_step_speed
    speed_ratio = 1.0 if baseline_speed <= 1e-6 else one_step_speed / baseline_speed

    new_velocity = (det_pos - last_pos) / float(frame_gap)
    heading_change = _angle_between(_last_velocity(track), new_velocity)

    score = det.get("score")
    score_penalty = 0.0
    if score is not None and np.isfinite(score):
        score_penalty = max(0.0, 1.0 - float(score)) * float(cfg.score_weight)

    speed_penalty = max(0.0, speed_ratio - 1.0) * float(cfg.speed_weight)
    direction_penalty = (heading_change / np.pi) * float(cfg.direction_weight)
    angular_penalty = max(0.0, heading_change - (np.pi / 2.0)) * float(cfg.angular_weight)
    cost = motion_distance + speed_penalty + direction_penalty + angular_penalty + score_penalty

    close_contact = bool(det.get("close_contact", False))
    zigzag = bool(
        speed_ratio >= float(cfg.zigzag_speed_ratio)
        and heading_change >= float(cfg.zigzag_heading_rad)
    )
    if close_contact and speed_ratio >= float(cfg.max_speed_ratio):
        zigzag = True
    return {
        "match_cost": float(cost),
        "motion_distance_px": motion_distance,
        "one_step_distance_px": one_step_distance,
        "rolling_speed_px": rolling_speed,
        "speed_ratio": float(speed_ratio),
        "heading_change_rad": float(heading_change),
        "close_contact": close_contact,
        "nearest_neighbor_distance_px": det.get("nearest_neighbor_distance_px", np.nan),
        "zigzag_flag": zigzag,
        "ambiguity_score": 0.0,
        "bridge_reason": "",
        "repair_action": "",
    }


def _build_motion_cost_matrix(
    active: dict[int, _ActiveTrack],
    detections: list[dict],
    *,
    frame: int,
    cfg: MotionAwareConfig,
    max_distance: float,
) -> tuple[list[int], np.ndarray, list[list[dict]]]:
    track_ids = list(active.keys())
    diagnostics: list[list[dict]] = []
    cost = np.full((len(track_ids), len(detections)), 1e9, dtype=float)
    for ri, tid in enumerate(track_ids):
        row_diag: list[dict] = []
        for ci, det in enumerate(detections):
            diag = _motion_diagnostics(active[tid], det, frame=frame, cfg=cfg)
            row_diag.append(diag)
            if diag["motion_distance_px"] <= max_distance:
                cost[ri, ci] = float(diag["match_cost"])
        diagnostics.append(row_diag)
    return track_ids, cost, diagnostics


def _row_ambiguity(cost: np.ndarray, ri: int, ci: int) -> float:
    finite = cost[ri][np.isfinite(cost[ri]) & (cost[ri] < 1e9)]
    if finite.size <= 1:
        return 0.0
    ordered = np.sort(finite)
    margin = float(ordered[1] - ordered[0])
    if not np.isfinite(margin):
        return 0.0
    return float(1.0 / (1.0 + max(0.0, margin)))


def _should_bridge(diag: dict, *, cfg: MotionAwareConfig) -> str:
    close_contact = bool(diag.get("close_contact", False))
    if close_contact:
        margin = (1.0 / max(float(diag.get("ambiguity_score", 0.0)), 1e-9)) - 1.0
        if margin < float(cfg.close_contact_min_margin):
            return "close_contact_ambiguous"
    if close_contact and bool(diag.get("zigzag_flag", False)):
        return "zigzag_motion"
    return ""


def _local_zigzag_repair(rows: list[dict]) -> list[dict]:
    if not rows:
        return rows
    drop_indices: set[int] = set()
    by_track: dict[int, list[tuple[int, dict]]] = {}
    for idx, row in enumerate(rows):
        by_track.setdefault(int(row["track_id"]), []).append((idx, row))

    for items in by_track.values():
        items.sort(key=lambda item: int(item[1]["frame"]))
        for prev_item, curr_item, next_item in zip(items, items[1:], items[2:]):
            curr_idx, curr = curr_item
            if not bool(curr.get("zigzag_flag", False)):
                continue
            if not bool(curr.get("close_contact", False)):
                continue
            prev = prev_item[1]
            nxt = next_item[1]
            p0 = np.asarray([prev["x"], prev["y"]], dtype=float)
            p1 = np.asarray([curr["x"], curr["y"]], dtype=float)
            p2 = np.asarray([nxt["x"], nxt["y"]], dtype=float)
            via = float(np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1))
            bridge = float(np.linalg.norm(p2 - p0))
            if bridge + 1e-6 < via * 0.6:
                drop_indices.add(curr_idx)
    repaired: list[dict] = []
    for idx, row in enumerate(rows):
        if idx in drop_indices:
            continue
        if "repair_action" in row and bool(row.get("zigzag_flag", False)):
            row = dict(row)
            row["repair_action"] = row.get("repair_action", "")
        repaired.append(row)
    return repaired


def track(
    slp_path: Path,
    *,
    max_frames: int | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    max_distance: float = 100.0,
    max_tracks: int = 3,
    max_gap: int = 50,
    anchor_node: str | None = None,
    top_k_by_score: int | None = None,
    motion_aware: bool = False,
    motion_config: MotionAwareConfig | None = None,
) -> tuple[list[dict], dict]:
    """
    Run NN tracking on a merged .slp.

    Parameters
    ----------
    slp_path : Path
        Merged top/side .slp file.
    max_frames : int or None
        Stop after this many frames (by frame_idx). None = process all.
    max_distance : float
        Hungarian-gate distance threshold in pixels. Pairs farther than this
        are not matched.
    max_tracks : int
        Maximum number of concurrent tracks (3 for medaka: 1F + 2M).
    max_gap : int
        Retire a track once it has been unseen for more than this many frames.

    Returns
    -------
    rows : list of dict
        Tracking rows: frame, track_id, x, y, score.
    stats : dict
        Summary stats (n_frames, n_tracks, n_assignments, ...).
    """
    try:
        import sleap_io as sio
    except ImportError as exc:
        raise RuntimeError(
            "sleap-io is required (pip install sleap-io)"
        ) from exc

    labels = sio.load_slp(str(slp_path))
    if not labels.skeletons:
        raise RuntimeError(f"No skeleton found in {slp_path}")
    node_names = [n.name for n in labels.skeletons[0].nodes]
    if motion_aware and anchor_node is None and "mouth" in node_names:
        anchor_node = "mouth"
    # Record the coordinate source truthfully in tracks.csv. The production
    # default is the visible-node centroid, not the first skeleton node.
    node_name = anchor_node if (anchor_node in node_names) else "centroid"

    frames = sorted(labels.labeled_frames, key=lambda f: f.frame_idx)
    frames = [f for f in frames if f.frame_idx >= start_frame]
    if end_frame is not None:
        frames = [f for f in frames if f.frame_idx <= end_frame]
    if max_frames is not None:
        frames = frames[:max_frames]

    active: dict[int, _ActiveTrack] = {}
    next_id = 0
    rows: list[dict] = []
    n_assignments = 0
    n_new = 0
    n_bridged = 0
    n_zigzag = 0
    cfg = motion_config or MotionAwareConfig(enabled=motion_aware)
    if motion_aware and not cfg.enabled:
        cfg = MotionAwareConfig(
            enabled=True,
            rolling_window=cfg.rolling_window,
            close_contact_distance_px=cfg.close_contact_distance_px,
            close_contact_min_margin=cfg.close_contact_min_margin,
            max_speed_ratio=cfg.max_speed_ratio,
            zigzag_speed_ratio=cfg.zigzag_speed_ratio,
            zigzag_heading_rad=cfg.zigzag_heading_rad,
            speed_weight=cfg.speed_weight,
            direction_weight=cfg.direction_weight,
            angular_weight=cfg.angular_weight,
            score_weight=cfg.score_weight,
            local_repair=cfg.local_repair,
        )

    def _retire_stale(now: int) -> None:
        stale = [tid for tid, tr in active.items() if (now - tr.last_frame) > max_gap]
        for tid in stale:
            del active[tid]

    for f in frames:
        fi = int(f.frame_idx)
        _retire_stale(fi)

        # Optional score-filter: keep only the top-K highest-scoring instances
        # before tracking (recovers max_instances=K-at-inference behaviour at runtime).
        instances = list(f.instances)
        if top_k_by_score is not None and len(instances) > top_k_by_score:
            instances = sorted(
                instances,
                key=lambda inst: (getattr(inst, "score", None) or 0.0),
                reverse=True,
            )[:top_k_by_score]

        detections = []
        for inst in instances:
            pos = _instance_position(
                inst, node_names=node_names, anchor_node=anchor_node,
            )
            if pos is None:
                continue
            score = getattr(inst, "score", None)
            detections.append({"x": pos[0], "y": pos[1], "score": score})

        if not detections:
            continue

        if not active:
            # Seed up to max_tracks new tracks from this frame's detections.
            for det in detections[:max_tracks]:
                tid = next_id
                next_id += 1
                active[tid] = _ActiveTrack(
                    track_id=tid, last_frame=fi, last_pos=(det["x"], det["y"])
                )
                _record_detection(active[tid], fi, det)
                rows.append({
                    "frame": fi, "track_id": tid,
                    "x": det["x"], "y": det["y"],
                    "score": det["score"], "node": node_name,
                })
                n_new += 1
            continue

        det_pos = np.array([[d["x"], d["y"]] for d in detections])
        _add_detection_spacing(
            detections,
            close_contact_distance_px=cfg.close_contact_distance_px
            if cfg.enabled else 0.0,
        )

        # Hungarian on a copy where gated entries are very expensive so they
        # only get matched when nothing else is available; we then drop those
        # matches by checking the original distance.
        if cfg.enabled:
            track_ids, cost, diagnostics = _build_motion_cost_matrix(
                active, detections, frame=fi, cfg=cfg, max_distance=max_distance,
            )
            dist = np.asarray(
                [[d["motion_distance_px"] for d in row] for row in diagnostics],
                dtype=float,
            )
        else:
            track_ids = list(active.keys())
            track_pos = np.array([active[t].last_pos for t in track_ids])
            # Pairwise distance matrix (n_tracks x n_dets)
            diff = track_pos[:, None, :] - det_pos[None, :, :]
            dist = np.sqrt((diff ** 2).sum(axis=-1))
            cost = dist.copy()
            cost[dist > max_distance] = 1e9
            diagnostics = [[{} for _ in detections] for _ in track_ids]
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for ri, ci in zip(row_ind, col_ind):
            if dist[ri, ci] > max_distance:
                continue
            if cost[ri, ci] >= 1e9:
                continue
            tid = track_ids[ri]
            det = detections[ci]
            diag = dict(diagnostics[ri][ci])
            if cfg.enabled:
                diag["ambiguity_score"] = _row_ambiguity(cost, ri, ci)
                bridge_reason = _should_bridge(diag, cfg=cfg)
                if bridge_reason:
                    n_bridged += 1
                    if bool(diag.get("zigzag_flag", False)):
                        n_zigzag += 1
                    continue
            _record_detection(active[tid], fi, det)
            row = {
                "frame": fi, "track_id": tid,
                "x": det["x"], "y": det["y"],
                "score": det["score"], "node": node_name,
            }
            if cfg.enabled:
                row.update(diag)
                if bool(diag.get("zigzag_flag", False)):
                    n_zigzag += 1
            rows.append(row)
            matched_tracks.add(tid)
            matched_dets.add(ci)
            n_assignments += 1

        # Unmatched detections become new tracks if there is room.
        for ci, det in enumerate(detections):
            if ci in matched_dets:
                continue
            if len(active) >= max_tracks:
                continue
            tid = next_id
            next_id += 1
            active[tid] = _ActiveTrack(
                track_id=tid, last_frame=fi, last_pos=(det["x"], det["y"])
            )
            _record_detection(active[tid], fi, det)
            rows.append({
                "frame": fi, "track_id": tid,
                "x": det["x"], "y": det["y"],
                "score": det["score"], "node": node_name,
            })
            n_new += 1

    rows_before_repair = len(rows)
    if cfg.enabled and cfg.local_repair:
        rows = _local_zigzag_repair(rows)

    stats = {
        "frames_processed": len(frames),
        "rows": len(rows),
        "tracks_created": next_id,
        "assignments": n_assignments,
        "new_tracks": n_new,
        "bridged_matches": n_bridged,
        "zigzag_matches": n_zigzag,
        "local_repair_dropped_rows": rows_before_repair - len(rows),
        "max_distance_px": max_distance,
        "max_tracks": max_tracks,
        "max_gap": max_gap,
        "motion_aware": bool(cfg.enabled),
        "anchor_node": anchor_node,
        "position_source": (
            f"anchor_node:{anchor_node}"
            if anchor_node in node_names
            else "visible_node_centroid"
        ),
    }
    return rows, stats


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BASE_FIELDS)
    if any(any(key in row for key in DIAGNOSTIC_FIELDS) for row in rows):
        fieldnames.extend(DIAGNOSTIC_FIELDS)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Greedy Hungarian-NN frame-by-frame tracker for medaka."
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        default=None,
        help="Experiment folder name; auto-resolves the merged .slp via paths.local.toml.",
    )
    parser.add_argument("--view", default="top_cam",
                        help="View to track (default top_cam).")
    parser.add_argument("--slp", default=None,
                        help="Explicit merged .slp path (overrides experiment lookup).")
    parser.add_argument("--output", default=None,
                        help="Output CSV path. Defaults to <slp>.tracks.csv.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Process at most this many frames (after start/end filter).")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="First frame index to include (inclusive).")
    parser.add_argument("--end-frame", type=int, default=None,
                        help="Last frame index to include (inclusive).")
    parser.add_argument("--max-distance", type=float, default=100.0,
                        help="Pixel gate for NN matching (default 100).")
    parser.add_argument("--max-tracks", type=int, default=3,
                        help="Cap on concurrent track IDs (default 3 = 1F+2M).")
    parser.add_argument("--max-gap", type=int, default=50,
                        help="Retire a track unseen for more than this many frames.")
    parser.add_argument(
        "--anchor-node", default=None,
        help="If set, track on this skeleton node's xy directly instead of the "
             "centroid of all visible nodes (e.g. 'mouth' for the round2 mouth+tail model).",
    )
    parser.add_argument(
        "--top-k-by-score", type=int, default=None,
        help="If set, keep only the top-K instances per frame ranked by SLEAP "
             "instance.score before tracking (recovers n=K-at-inference behaviour).",
    )
    parser.add_argument(
        "--motion-aware", action="store_true",
        help="Use mouth-anchored motion-aware matching with close-contact bridging.",
    )
    parser.add_argument(
        "--close-contact-distance", type=float, default=35.0,
        help="Mouth-mouth distance below which assignments are treated as close-contact.",
    )
    parser.add_argument(
        "--close-contact-min-margin", type=float, default=12.0,
        help="Minimum cost margin required to accept a close-contact match.",
    )
    parser.add_argument(
        "--max-speed-ratio", type=float, default=4.0,
        help="Bridge a candidate match above this one-step/rolling-speed ratio.",
    )
    parser.add_argument(
        "--no-local-repair", action="store_true",
        help="Disable the conservative post-pass that drops obvious close-contact zigzag rows.",
    )
    parser.add_argument("--paths-config", default=None)
    parser.add_argument("--sleap-output-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    slp_path = Path(args.slp) if args.slp else None
    if slp_path is None:
        if args.experiment is None:
            parser.error("Pass an experiment name or --slp.")
        local = load_local_paths(args.paths_config)
        sleap_output_dir = choose_path(
            args.sleap_output_dir,
            local.sleap_output_dir if local else None,
        )
        if sleap_output_dir is None:
            parser.error(
                "sleap_output_dir is required (paths.local.toml or --sleap-output-dir)."
            )
        slp_path = Path(sleap_output_dir) / args.experiment / f"{args.view}.slp"

    if not slp_path.exists():
        parser.error(f"Merged .slp not found: {slp_path}")

    output = Path(args.output) if args.output else slp_path.with_suffix(".tracks.csv")

    print(f"Input  : {slp_path}")
    print(f"Output : {output}")
    print(f"Params : max_frames={args.max_frames}  max_distance={args.max_distance}px  "
          f"max_tracks={args.max_tracks}  max_gap={args.max_gap}")
    if args.motion_aware and args.anchor_node is None:
        print("Motion : enabled  anchor_node=mouth")
    elif args.motion_aware:
        print(f"Motion : enabled  anchor_node={args.anchor_node}")

    rows, stats = track(
        slp_path=slp_path,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        max_distance=args.max_distance,
        max_tracks=args.max_tracks,
        max_gap=args.max_gap,
        anchor_node=args.anchor_node,
        top_k_by_score=args.top_k_by_score,
        motion_aware=args.motion_aware,
        motion_config=MotionAwareConfig(
            enabled=args.motion_aware,
            close_contact_distance_px=args.close_contact_distance,
            close_contact_min_margin=args.close_contact_min_margin,
            max_speed_ratio=args.max_speed_ratio,
            local_repair=not args.no_local_repair,
        ),
    )
    write_csv(rows, output)

    print()
    print("Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
