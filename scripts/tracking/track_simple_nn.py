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


def _instance_position(inst) -> tuple[float, float] | None:
    """Centroid of an instance's visible nodes (NaN-safe)."""
    pts = inst.numpy()  # (n_nodes, 2)
    mask = np.isfinite(pts).all(axis=1)
    if not mask.any():
        return None
    centroid = pts[mask].mean(axis=0)
    return float(centroid[0]), float(centroid[1])


def track(
    slp_path: Path,
    *,
    max_frames: int | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    max_distance: float = 100.0,
    max_tracks: int = 3,
    max_gap: int = 50,
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
    node_name = labels.skeletons[0].nodes[0].name

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

    def _retire_stale(now: int) -> None:
        stale = [tid for tid, tr in active.items() if (now - tr.last_frame) > max_gap]
        for tid in stale:
            del active[tid]

    for f in frames:
        fi = int(f.frame_idx)
        _retire_stale(fi)

        detections = []
        for inst in f.instances:
            pos = _instance_position(inst)
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
                rows.append({
                    "frame": fi, "track_id": tid,
                    "x": det["x"], "y": det["y"],
                    "score": det["score"], "node": node_name,
                })
                n_new += 1
            continue

        track_ids = list(active.keys())
        track_pos = np.array([active[t].last_pos for t in track_ids])
        det_pos = np.array([[d["x"], d["y"]] for d in detections])

        # Pairwise distance matrix (n_tracks x n_dets)
        diff = track_pos[:, None, :] - det_pos[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))

        # Hungarian on a copy where gated entries are very expensive so they
        # only get matched when nothing else is available; we then drop those
        # matches by checking the original distance.
        cost = dist.copy()
        cost[dist > max_distance] = 1e9
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for ri, ci in zip(row_ind, col_ind):
            if dist[ri, ci] > max_distance:
                continue
            tid = track_ids[ri]
            det = detections[ci]
            active[tid].last_frame = fi
            active[tid].last_pos = (det["x"], det["y"])
            rows.append({
                "frame": fi, "track_id": tid,
                "x": det["x"], "y": det["y"],
                "score": det["score"], "node": node_name,
            })
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
            rows.append({
                "frame": fi, "track_id": tid,
                "x": det["x"], "y": det["y"],
                "score": det["score"], "node": node_name,
            })
            n_new += 1

    stats = {
        "frames_processed": len(frames),
        "rows": len(rows),
        "tracks_created": next_id,
        "assignments": n_assignments,
        "new_tracks": n_new,
        "max_distance_px": max_distance,
        "max_tracks": max_tracks,
        "max_gap": max_gap,
    }
    return rows, stats


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["frame", "track_id", "x", "y", "score", "node"]
        )
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

    rows, stats = track(
        slp_path=slp_path,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        max_distance=args.max_distance,
        max_tracks=args.max_tracks,
        max_gap=args.max_gap,
    )
    write_csv(rows, output)

    print()
    print("Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
