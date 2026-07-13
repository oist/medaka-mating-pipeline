from __future__ import annotations

"""Relink pre-triangulated 3D candidates using a reliable tracked camera view.

The view anchor is conservative: a frame supplies identity hints only when all
fish tracks are present, mutually separated, and uniquely matched to the
reconstruction-ordered SLEAP slots. Other frames retain the existing 3D linker.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.reconstruction_3d.cross_frame_tracking import apply_cross_frame_tracking
from scripts.reconstruction_3d.fill_missing_3d import (
    best_by_frame_to_dfs,
    dfs_to_single_csv,
    ensure_all_frames_and_fill_spline,
)
from scripts.reconstruction_3d.load_2d_slp import _instances_in_slot_order
from scripts.reconstruction_3d.reconstruct_from_slp import (
    _save_best_tracked_json,
    load_best_pre_link_json,
)


def _stable_track_key(value) -> str:
    try:
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value)


def match_tracks_to_slots(
    slot_xy: np.ndarray,
    track_xy: np.ndarray,
    track_keys: list[str],
    *,
    num_individuals: int,
    match_distance_px: float,
    min_separation_px: float,
) -> dict[int, str] | None:
    """Return ``slot_index -> track_key`` for one high-confidence frame."""
    slot_xy = np.asarray(slot_xy, dtype=float)
    track_xy = np.asarray(track_xy, dtype=float)
    if len(track_xy) != num_individuals or len(set(track_keys)) != num_individuals:
        return None
    if len(slot_xy) < num_individuals or not np.isfinite(track_xy).all():
        return None

    pair = np.linalg.norm(track_xy[:, None, :] - track_xy[None, :, :], axis=2)
    pair[np.eye(len(pair), dtype=bool)] = np.inf
    if float(pair.min()) < min_separation_px:
        return None

    valid_slots = np.where(np.isfinite(slot_xy).all(axis=1))[0]
    if len(valid_slots) < num_individuals:
        return None
    cost = np.linalg.norm(
        track_xy[:, None, :] - slot_xy[valid_slots][None, :, :], axis=2
    )
    rows, cols = linear_sum_assignment(cost)
    if len(rows) != num_individuals:
        return None
    if any(float(cost[r, c]) > match_distance_px for r, c in zip(rows, cols)):
        return None
    return {
        int(valid_slots[int(c)]): track_keys[int(r)]
        for r, c in zip(rows, cols)
    }


def build_slot_track_anchors(
    slp_path: Path,
    tracks_csv: Path,
    *,
    start_frame: int,
    end_frame: int,
    num_individuals: int = 3,
    slot_order_anchor_node: str | None = None,
    match_node: str = "centroid",
    match_distance_px: float = 20.0,
    min_separation_px: float = 60.0,
    require_min_track_frames: int = 500,
) -> tuple[dict[int, dict[int, str]], dict]:
    """Match stable 2D track keys to reconstruction candidate slots per frame."""
    try:
        import sleap_io as sio
    except ImportError as exc:
        raise RuntimeError("sleap-io is required to build view anchors") from exc

    tracks = pd.read_csv(tracks_csv)
    required = {"frame", "track_id", "x", "y"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"{tracks_csv} missing columns: {sorted(missing)}")
    tracks = tracks.copy()
    tracks["track_key"] = tracks["track_id"].map(_stable_track_key)
    sizes = tracks.groupby("track_key").size()
    keep = set(sizes[sizes >= require_min_track_frames].index)
    tracks = tracks[tracks["track_key"].isin(keep)]
    tracks = tracks[(tracks["frame"] >= start_frame) & (tracks["frame"] <= end_frame)]

    labels = sio.load_slp(str(slp_path))
    node_names = [node.name for node in labels.skeletons[0].nodes]
    if match_node != "centroid" and match_node not in node_names:
        raise ValueError(f"Anchor node {match_node!r} not in SLEAP skeleton {node_names}")
    match_idx = node_names.index(match_node) if match_node != "centroid" else None
    tracks_by_frame = {int(f): g for f, g in tracks.groupby("frame")}

    anchors: dict[int, dict[int, str]] = {}
    considered = 0
    for lf in labels.labeled_frames:
        frame = int(lf.frame_idx)
        if frame < start_frame or frame > end_frame:
            continue
        tf = tracks_by_frame.get(frame)
        if tf is None:
            continue
        considered += 1
        ordered = _instances_in_slot_order(
            lf.instances,
            node_names=node_names,
            anchor_node=slot_order_anchor_node,
        )
        slot_xy = np.full((len(ordered), 2), np.nan)
        for i, inst in enumerate(ordered):
            points = np.asarray(inst.numpy(), dtype=float)
            if match_idx is None:
                finite = np.isfinite(points).all(axis=1)
                point = points[finite].mean(axis=0) if finite.any() else np.full(2, np.nan)
            else:
                point = points[match_idx]
            if point.shape == (2,) and np.isfinite(point).all():
                slot_xy[i] = point
        tf = tf.drop_duplicates("track_key", keep="last")
        matched = match_tracks_to_slots(
            slot_xy,
            tf[["x", "y"]].to_numpy(dtype=float),
            tf["track_key"].astype(str).tolist(),
            num_individuals=num_individuals,
            match_distance_px=match_distance_px,
            min_separation_px=min_separation_px,
        )
        if matched is not None:
            anchors[frame] = matched

    qc = {
        "view": "top_cam",
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "match_node": match_node,
        "match_distance_px": float(match_distance_px),
        "min_separation_px": float(min_separation_px),
        "require_min_track_frames": int(require_min_track_frames),
        "frames_with_tracks": int(considered),
        "high_confidence_anchor_frames": int(len(anchors)),
        "high_confidence_fraction": float(len(anchors) / max(considered, 1)),
    }
    return anchors, qc


def infer_anchor_identity_map(
    best_by_frame: dict[int, dict],
    slot_track_anchors: dict[int, dict[int, str]],
    *,
    num_individuals: int,
) -> tuple[dict[str, str], int]:
    """Seed stable track keys with the linker's canonical left-to-right IDs."""
    for frame in sorted(best_by_frame):
        slot_keys = slot_track_anchors.get(frame)
        if not slot_keys:
            continue
        info = best_by_frame[frame]
        groups = info.get("groups", {})
        perm0 = list(info.get("perm_cam0") or range(num_individuals))
        points: list[np.ndarray] = []
        keys: list[str] = []
        for src_idx in range(num_individuals):
            point = np.asarray(groups.get(chr(ord("a") + src_idx), []), dtype=float)
            if point.shape != (3,) or not np.isfinite(point).all():
                break
            slot = perm0[src_idx] if src_idx < len(perm0) else -1
            key = slot_keys.get(slot)
            if key is None:
                break
            points.append(point)
            keys.append(key)
        if len(points) != num_individuals or len(set(keys)) != num_individuals:
            continue
        source_by_x = sorted(range(num_individuals), key=lambda i: points[i][0])
        mapping = {
            keys[src_idx]: f"ID_{rank + 1}"
            for rank, src_idx in enumerate(source_by_x)
        }
        return mapping, int(frame)
    raise RuntimeError("No frame could seed all anchor track keys to canonical IDs")


def build_source_id_hints(
    best_by_frame: dict[int, dict],
    slot_track_anchors: dict[int, dict[int, str]],
    anchor_identity_map: dict[str, str],
    *,
    num_individuals: int,
) -> dict[int, dict[int, str]]:
    hints: dict[int, dict[int, str]] = {}
    for frame, slot_keys in slot_track_anchors.items():
        info = best_by_frame.get(frame)
        if info is None:
            continue
        perm0 = list(info.get("perm_cam0") or range(num_individuals))
        frame_hints = {}
        for src_idx, slot in enumerate(perm0[:num_individuals]):
            identity = anchor_identity_map.get(slot_keys.get(slot, ""))
            if identity is not None:
                frame_hints[src_idx] = identity
        if frame_hints:
            hints[int(frame)] = frame_hints
    return hints


def _derive_tag(path: Path) -> str:
    suffix = "_best_pre_link"
    return path.stem[: -len(suffix)] if path.stem.endswith(suffix) else path.stem


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pre_link_json", type=Path)
    parser.add_argument("--anchor-slp", type=Path, required=True)
    parser.add_argument("--anchor-tracks-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--num-individuals", type=int, default=3)
    parser.add_argument(
        "--match-node", default="centroid",
        help="SLEAP point used by the 2D tracker (default: visible-node centroid).",
    )
    parser.add_argument("--match-distance-px", type=float, default=20.0)
    parser.add_argument("--min-separation-px", type=float, default=60.0)
    parser.add_argument("--anchor-min-track-frames", type=int, default=500)
    parser.add_argument("--max-jump-mm", type=float, default=8.0)
    parser.add_argument("--max-jump-hard-cap-mm", type=float, default=80.0)
    parser.add_argument("--max-gap-frames", type=int, default=100)
    parser.add_argument("--no-fill-missing", action="store_true")
    args = parser.parse_args(argv)

    best = load_best_pre_link_json(args.pre_link_json)
    frames = sorted(best)
    if not frames:
        parser.error("Pre-link JSON contains no frames")
    anchors, qc = build_slot_track_anchors(
        args.anchor_slp,
        args.anchor_tracks_csv,
        start_frame=frames[0],
        end_frame=frames[-1],
        num_individuals=args.num_individuals,
        match_node=args.match_node,
        match_distance_px=args.match_distance_px,
        min_separation_px=args.min_separation_px,
        require_min_track_frames=args.anchor_min_track_frames,
    )
    identity_map, seed_frame = infer_anchor_identity_map(
        best, anchors, num_individuals=args.num_individuals
    )
    hints = build_source_id_hints(
        best, anchors, identity_map, num_individuals=args.num_individuals
    )
    tracked = apply_cross_frame_tracking(
        best,
        num_individuals=args.num_individuals,
        max_jump_mm=args.max_jump_mm,
        max_jump_hard_cap_mm=args.max_jump_hard_cap_mm,
        max_gap_frames=args.max_gap_frames,
        source_id_hints_by_frame=hints,
    )

    out_dir = args.output_dir or args.pre_link_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.output_tag or f"{_derive_tag(args.pre_link_json)}_top_anchor"
    best_path = out_dir / f"{tag}_best.json"
    _save_best_tracked_json(tracked, best_path, num_individuals=args.num_individuals)

    dfs = best_by_frame_to_dfs(
        tracked,
        shift_to_positive=False,
        ensure_non_negative=False,
        num_individuals=args.num_individuals,
    )
    if not args.no_fill_missing:
        dfs = ensure_all_frames_and_fill_spline(
            dfs, frames, max_gap_frames=args.max_gap_frames
        )
    csv_path = dfs_to_single_csv(dfs, set_key=tag, output_dir=str(out_dir))

    qc.update({
        "seed_frame": int(seed_frame),
        "anchor_identity_map": identity_map,
        "frames_with_source_hints": int(len(hints)),
        "anchored_valid_3d_assignments": int(
            sum(sum(bool(v) for v in info.get("id_anchor", [])) for info in tracked.values())
        ),
    })
    summary_path = out_dir / f"{tag}_anchor_summary.json"
    summary_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")

    print(f"Anchor identity map : {identity_map} (seed frame {seed_frame})")
    print(
        f"High-confidence frames: {qc['high_confidence_anchor_frames']}/"
        f"{qc['frames_with_tracks']} ({100 * qc['high_confidence_fraction']:.1f}%)"
    )
    print(f"best JSON          : {best_path}")
    print(f"3D CSV             : {csv_path}")
    print(f"anchor summary     : {summary_path}")


if __name__ == "__main__":
    main()
