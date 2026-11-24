"""Run a per-frame 3D reconstruction made elsewhere through this repo's own linking and outputs.

The medaka-3d repository solves the cross-view correspondence per frame (36 hypotheses on several
nodes, point-to-ray residual). To compare it with ``reconstruct_from_slp`` on equal terms, only that
per-frame step may differ; everything after it is this repo's code with the batch pipeline's
arguments (``batch_reconstruct_3d.RECONSTRUCT_ARGS`` parsed by ``reconstruct_from_slp``'s own parser):

1. per-frame triples -> the pre-link format (``best_by_frame``),
2. ``apply_cross_frame_tracking`` (velocity/gap-aware Hungarian, 8 / 80 mm, 100 frames),
3. ``<tag>_best_pre_link.json``, ``<tag>_best.json``, ``<tag>_3d.csv`` (gap-filled like the original),
4. ``bind_topcam_id.bind`` -> ``<tag>_3d_identified.csv``.

The outputs have the layout ``scripts.review.review_data.build_review_data`` reads, so the result can be
turned into a review package and compared in ``scripts.review.compare_gui`` (reconstruction mode).

Detections are referenced by their slot in ``load_2d_slp``'s order (false positives dropped, sorted by
score), the order ``_best.json`` and the review build use; ``slot_arrays`` takes the first three slots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.reconstruction_3d.batch_reconstruct_3d import RECONSTRUCT_ARGS
from scripts.reconstruction_3d.cross_frame_tracking import apply_cross_frame_tracking
from scripts.reconstruction_3d.fill_missing_3d import (
    best_by_frame_to_dfs,
    dfs_to_single_csv,
    ensure_all_frames_and_fill_spline,
)
from scripts.reconstruction_3d.reconstruct_from_slp import (
    _save_best_tracked_json,
    _src_labels,
    build_parser,
    save_best_pre_link_json_to,
)


def batch_cli_args(experiment: str = "EXPERIMENT") -> argparse.Namespace:
    """The batch pipeline's ``reconstruct_from_slp`` arguments, parsed by its own parser."""
    return build_parser().parse_args([experiment, *RECONSTRUCT_ARGS])


def linking_kwargs(args: argparse.Namespace) -> dict:
    """``apply_cross_frame_tracking`` keyword arguments exactly as ``run_reconstruction_from_slp`` passes
    them on the (non tracklet-first) batch path."""
    return {
        "num_individuals": args.num_individuals,
        "max_jump_mm": args.max_jump_mm,
        "max_jump_hard_cap_mm": args.max_jump_hard_cap_mm,
        "max_gap_frames": args.max_gap_frames,
        "velocity_prediction_weight": args.motion_velocity_prediction_weight,
        "flow_weight": args.motion_tangent_weight,
        "acceleration_weight": args.motion_acceleration_weight,
        "motion_uncertainty_growth_per_frame": args.motion_uncertainty_growth_per_frame,
        "max_motion_gap_frames": args.max_motion_gap_frames,
        "motion_history_frames": args.motion_history_frames,
        "source_id_hints_by_frame": None,
        "source_hints_override_motion_gate": False,
    }


def load_slot_ordered(slp_path: Path, *, experiment: str | None = None,
                      view: str | None = None) -> tuple[dict[int, list[np.ndarray]], list[str], int]:
    """Every frame's instances in ``load_2d_slp``'s slot order, the node names and the frame count.

    With ``experiment`` and ``view`` the file must point at ``.../<experiment>/<view>/...``: side1 and
    side2 share every frame index, so a file of the other camera would otherwise go unnoticed.
    """
    import sleap_io as sio

    from scripts.reconstruction_3d.load_2d_slp import _instances_in_slot_order

    labels = sio.load_slp(str(slp_path))
    node_names = [n.name for n in labels.skeletons[0].nodes]
    if experiment is not None and view is not None:
        videos = [str(v.filename).replace("\\", "/") for v in labels.videos]
        if len(videos) != 1 or f"/{experiment}/{view}/" not in videos[0]:
            raise ValueError(f"{slp_path}: video {videos} is not {experiment}/{view}; refusing to mix cameras")
    out: dict[int, list[np.ndarray]] = {}
    for lf in labels.labeled_frames:
        insts = _instances_in_slot_order(lf.instances, node_names=node_names)
        if insts:
            out[int(lf.frame_idx)] = [np.asarray(inst.numpy(), dtype=np.float32) for inst in insts]
    shape = labels.videos[0].shape if labels.videos else None
    n_frames = int(shape[0]) if shape else (max(out) + 1 if out else 0)
    return out, node_names, n_frames


def slot_arrays(nodes_by_frame: dict[int, list[np.ndarray]], n_frames: int, n_nodes: int,
                n_animals: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """``frame -> [instance (N, 2), ...]`` in slot order -> the first ``n_animals`` slots
    ``(n_frames, n_animals, n_nodes, 2)`` (NaN-padded) and the slot count per frame."""
    points = np.full((n_frames, n_animals, n_nodes, 2), np.nan)
    counts = np.zeros(n_frames, dtype=np.int32)
    for frame, instances in nodes_by_frame.items():
        if 0 <= frame < n_frames:
            counts[frame] = len(instances)
            for k, inst in enumerate(instances[:n_animals]):
                points[frame, k] = np.asarray(inst, dtype=float)[:n_nodes]
    return points, counts


def visible_centroid(points: np.ndarray) -> np.ndarray:
    """``(..., N, 2)`` -> ``(..., 2)`` mean of the nodes with both coordinates finite, the point
    ``--use-visible-node-centroid`` triangulates (NaN when no node is visible)."""
    points = np.asarray(points, dtype=float)
    ok = np.isfinite(points).all(axis=-1, keepdims=True)
    n = ok.sum(axis=-2)
    total = np.where(ok, points, 0.0).sum(axis=-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, total / np.maximum(n, 1), np.nan)


def best_by_frame(frames: np.ndarray, xyz: np.ndarray, residual_mm: np.ndarray, source: np.ndarray, *,
                  xyz_bounds: tuple[float, float] | None = None) -> dict[int, dict]:
    """Per-frame triples -> the pre-link dict ``compute_best_assignments_per_frame`` returns.

    Args:
        xyz: ``(F, K, 3)`` mm, NaN where a slot has no 3D position.
        residual_mm: ``(F, K)`` point-to-ray RMS; written squared, the refractive scorer's mm^2 unit.
        source: ``(F, K, V)`` detection slot per view, -1 where the view took no part.
        xyz_bounds: slots with a coordinate outside ``[lo, hi]`` are dropped, as the batch run does.

    A slot seen by two views is ``recovered`` with ``missing_views`` naming the absent view; an empty
    slot has NaN xyz, ``inf`` error and ``-1`` in every ``perm_cam``.
    """
    xyz = np.asarray(xyz, dtype=float)
    n_frames, n_slots = xyz.shape[:2]
    n_views = source.shape[2]
    labels = _src_labels(n_slots)
    ok = np.isfinite(xyz).all(axis=-1)
    if xyz_bounds is not None:
        lo, hi = xyz_bounds
        ok &= ((xyz >= lo) & (xyz <= hi)).all(axis=-1)
    src = np.where(ok[..., None], source, -1).astype(int)
    absent = (src < 0) & ok[..., None]
    missing = np.where(absent.any(axis=-1), absent.argmax(axis=-1), -1)
    err = np.where(ok, np.asarray(residual_mm, dtype=float) ** 2, np.inf)
    nan3 = [float("nan")] * 3
    out: dict[int, dict] = {}
    for t, frame in enumerate(np.asarray(frames).tolist()):
        recovered = [bool(r) for r in absent[t].any(axis=-1)]
        views = [int(m) for m in missing[t]]
        out[int(frame)] = {
            "groups": {lb: ([float(v) for v in xyz[t, k]] if ok[t, k] else list(nan3)) for k, lb in enumerate(labels)},
            "errors": {lb: float(err[t, k]) for k, lb in enumerate(labels)},
            **{f"perm_cam{v}": [int(s) for s in src[t, :, v]] for v in range(n_views)},
            "recovered": recovered,
            "missing_views": views,
            "synthetic_views": list(views),
        }
    return out


def slot_ids(best_tracked: dict[int, dict], frames: np.ndarray, n_slots: int) -> np.ndarray:
    """``(F, K)`` linked ID index (0 = ``ID_1``) of every source slot, -1 where the slot is not linked."""
    out = np.full((len(frames), n_slots), -1, dtype=np.int16)
    for t, frame in enumerate(np.asarray(frames).tolist()):
        info = best_tracked.get(int(frame))
        if info is None:
            continue
        for id_idx, src_idx in enumerate(info.get("id_mapping") or []):
            if 0 <= src_idx < n_slots:
                out[t, src_idx] = id_idx
    return out


def link_and_write(best: dict[int, dict], *, out_dir: Path, tag: str, top_tracks_csv: Path | None,
                   calibration_dir: Path, all_frames, args: argparse.Namespace | None = None,
                   source: dict | None = None, log=print) -> dict:
    """Link ``best`` and write the files ``reconstruct_from_slp`` writes (see the module docstring).

    ``all_frames`` are the frames with any detection (the gap-filling grid, ``points_by_frame`` in
    ``reconstruct_from_slp``). Without ``top_tracks_csv`` the bind step is skipped.
    """
    args = args or batch_cli_args()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = args.num_individuals
    linking = linking_kwargs(args)
    meta = {
        "triangulation": {"backend": "external", "provenance": {"model_folder": str(calibration_dir)}},
        "per_frame_source": source or {},
        "temporal_linking": {"algorithm": "cross_frame_hungarian",
                             **{k: v for k, v in linking.items() if k != "source_id_hints_by_frame"}},
        "reconstruct_args": list(RECONSTRUCT_ARGS),
    }
    (out_dir / f"{tag}_reconstruction_provenance.json").write_text(json.dumps(meta, indent=2) + "\n",
                                                                     encoding="utf-8")
    log(f"[bridge] linking {len(best):,} frames ...")
    best_tracked = apply_cross_frame_tracking(best, **linking)
    save_best_pre_link_json_to(best, out_dir / f"{tag}_best_pre_link.json", num_individuals=n,
                               calibration_metadata=meta)
    best_json = out_dir / f"{tag}_best.json"
    _save_best_tracked_json(best_tracked, best_json, num_individuals=n, calibration_metadata=meta)
    shift = not args.no_shift_to_positive
    dfs = best_by_frame_to_dfs(best_tracked, shift_to_positive=shift, ensure_non_negative=False, num_individuals=n)
    if not args.no_fill_missing:
        dfs = ensure_all_frames_and_fill_spline(dfs, sorted(int(f) for f in all_frames),
                                                max_gap_frames=args.max_gap_frames, ensure_non_negative=shift)
    csv_3d = Path(dfs_to_single_csv(dfs, set_key=tag, output_dir=str(out_dir)))
    log(f"[bridge] wrote {best_json.name}, {csv_3d.name}")

    csv_identified = None
    if args.bind_top_id and top_tracks_csv is not None:
        from scripts.reconstruction_3d.bind_topcam_id import bind
        from scripts.reconstruction_3d.calibration import Calibration

        csv_identified = csv_3d.with_name(csv_3d.stem + "_identified.csv")
        bind(exp_3d_csv=csv_3d, top_tracks_csv=Path(top_tracks_csv),
             calib=Calibration(str(calibration_dir), backend=args.calibration_backend), out_csv=csv_identified,
             bind_distance=args.bind_distance, require_min_frames=args.bind_min_frames)
        log(f"[bridge] wrote {csv_identified.name}")
    return {"best_json": best_json, "best_pre_link_json": out_dir / f"{tag}_best_pre_link.json", "csv_3d": csv_3d,
            "csv_identified": csv_identified, "best_tracked": best_tracked, "linking": linking}
