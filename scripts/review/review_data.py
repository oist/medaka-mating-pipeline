"""Build the per-experiment review data consumed by the review GUI.

Inputs (one experiment, one round tag):

* ``<sleap_output_dir>/<exp>/<tag>/<view>.slp`` - raw SLEAP keypoints.
* ``<reconstruction_output_dir>/<exp>/<tag>/<tag>_best.json`` - per frame,
  which detection slot in each view belongs to which 3D fish (``cam*_to_id``).
* ``<tag>_3d.csv`` / ``<tag>_3d_identified.csv`` - linked 3D centroids,
  interpolation flags and the top-camera bind result.

Steps:

1. Recover every node (mouth / pectoral / tail ...) of the detections the 3D
   step used, in the exact slot order of ``load_2d_slp``.
   Where two fish overlap in a view (skeletons closer than ``overlap_px``),
   SLEAP often puts a node on the wrong fish; such views are flagged
   (``view_overlap``) so the GUI draws them as unreliable. With
   ``drop_view`` (off by default) the most-overlapped view of a fish seen in
   all three views is also left out of the node triangulation
   (``view_dropped``). On 20260112_P_1_1_FCMC (after the first 1.5 min) that
   changed body-length outliers only from 0.33% to 0.29% of fish-frames but
   raised the 99.9th percentile of per-frame mouth steps from 7.5 to 9.7 mm.
2. Head-tail auto-correction per fish:
   a. cross-view consistency - with the reference view fixed, try flipping
      each other view's mouth<->tail and keep the combination with the lowest
      ray residual for the head and tail points;
   b. temporal orientation - a two-state Viterbi over time per fish that
      trades view votes (keep SLEAP's labels), 3D continuity of head/tail and
      a swimming-direction prior;
   c. short swaps - head and tail exchanging places for fewer than
      ``short_flip_frames`` frames is undone, first per view in 2D (before
      triangulation), then in 3D (a fish turns continuously, its head never
      jumps onto its tail position). Remaining 3D exchanges split the
      track into ``orientation_segment`` runs and become ``head_tail_jump``
      risks; the GUI flips a whole segment at once.
3. Triangulate every node with the corrected labels (two-view fallback where a
   view is missing) and project the 3D centroids into all views. Complete
   every node in 3D wherever the fish has a 3D centroid (``nodes3d_filled``,
   provenance in ``node_source``): a node seen in one view is fitted to that
   view's 2D point at the fish's body-model distance; a node seen in no view
   is the body model alone (anchor + offset interpolated in time).
4. Derive machine risk windows (close approaches, jumps, interpolation gaps,
   missing views, high residual, unbound to the top track, orientation
   overrides, fish-count changes).

Outputs into ``<review_root>/<exp>/<tag>/``: ``review_data.npz``,
``machine_risk.csv`` and ``review_manifest.json``. Machine risk rows are never
edited later, so they can be compared against the human ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VIEWS = ("top_cam", "side1_cam", "side2_cam")
CAM_KEYS = ("cam0_to_id", "cam1_to_id", "cam2_to_id")
DATA_VERSION = 5
RISK_COLUMNS = ["risk_id", "frame_start", "frame_end", "frame_peak", "types", "fish", "score", "details"]


@dataclass
class OrientationParams:
    w_vote: float = 1.0          # cost per view whose SLEAP labels are overridden
    w_cont: float = 0.5          # cost per mm of head/tail displacement between kept frames
    w_motion: float = 1.0        # cost for swimming backwards at full speed
    v_ref_mm: float = 0.3        # mm/frame at which the motion prior reaches full weight (~50 mm/s at 166 fps)
    max_gap: int = 30            # frames; longer gaps break temporal continuity
    velocity_half_window: int = 5
    short_flip_frames: int = 8   # head/tail swapped for fewer frames (~50 ms) than this is undone: a fish
                                 # turns around continuously, its head never lands on its tail position
    end_on_fraction: float = 0.35  # 2D head-tail shorter than this x the view's median: seen end-on, skipped


@dataclass
class FillParams:
    sigma_px: float = 3.0        # single-view fit: reprojection residual scale (source px)
    sigma_mm: float = 4.0        # pull towards the body-model prediction (fixes depth along the ray)
    sigma_len_mm: float = 1.5    # pull towards the fish's median node-to-pectoral distance
    iterations: int = 8
    step_mm: float = 0.25        # finite-difference step of the projection Jacobian


@dataclass
class OcclusionParams:
    overlap_px: float = 20.0     # source px between two fish's skeletons in one view (body width ~15-20 px)
    drop_view: bool = False      # leave the most-overlapped view out of node triangulation


@dataclass
class RiskParams:
    close_mm: float = 15.0
    top_close_px: float = 60.0
    jump_mm: float = 8.0
    min_gap_frames: int = 5
    missing_view_min_frames: int = 20
    residual_mm2: float = 4.0
    residual_min_frames: int = 20
    unbound_min_frames: int = 50
    orientation_min_frames: int = 3
    pad_seconds: float = 0.3


# ---------------------------------------------------------------------------
# Loading


def load_slp_nodes(slp_path: Path, start: int, end: int) -> tuple[dict[int, list[np.ndarray]], list[str], list[tuple[int, int]]]:
    """Per frame, all nodes of each detection in the reconstruction's slot order."""
    import sleap_io as sio

    from scripts.reconstruction_3d.load_2d_slp import _instances_in_slot_order

    labels = sio.load_slp(str(slp_path))
    skeleton = labels.skeletons[0]
    node_names = [n.name for n in skeleton.nodes]
    edges = [(node_names.index(e.source.name), node_names.index(e.destination.name)) for e in skeleton.edges]
    out: dict[int, list[np.ndarray]] = {}
    for lf in labels.labeled_frames:
        fi = int(lf.frame_idx)
        if fi < start or fi > end:
            continue
        insts = _instances_in_slot_order(lf.instances, node_names=node_names)
        if insts:
            out[fi] = [np.asarray(inst.numpy(), dtype=np.float32) for inst in insts]
    return out, node_names, edges


def gather_keypoints(best_frames: dict, slp_by_view: list[dict], fish_ids: list[str], frames: np.ndarray,
                     n_nodes: int, slots: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Raw keypoints ``(F, K, V, N, 2)`` of the detections assigned to each fish.

    ``slots (F, K, V)`` (int8, optional) receives the detection slot of each assignment (-1 none).
    """
    index = {fid: k for k, fid in enumerate(fish_ids)}
    kp = np.full((len(frames), len(fish_ids), len(VIEWS), n_nodes, 2), np.nan, dtype=np.float32)
    present = np.zeros((len(frames), len(fish_ids), len(VIEWS)), dtype=bool)
    f0 = int(frames[0])
    for key, info in best_frames.items():
        fi = int(key)
        t = fi - f0
        for v, cam_key in enumerate(CAM_KEYS):
            dets = slp_by_view[v].get(fi)
            if not dets:
                continue
            for slot, fid in enumerate(info.get(cam_key) or []):
                k = index.get(fid)
                if k is None or slot >= len(dets):
                    continue
                kp[t, k, v] = dets[slot][:n_nodes]
                present[t, k, v] = np.isfinite(dets[slot]).all(axis=1).any()
                if slots is not None:
                    slots[t, k, v] = slot
    return kp, present


DET_SCALE = 10.0           # detections.npz stores pixels x 10 as int16 (0.1 px resolution)
DET_NONE = -32768


def detection_pack(slp_by_view: list[dict], frames: np.ndarray, n_nodes: int, max_slots: int = 8) -> np.ndarray:
    """Every detection of every view, in slot order: ``(F, V, S, N, 2)`` int16 (pixels x ``DET_SCALE``,
    ``DET_NONE`` = missing). The review GUI re-assigns detections to fish from this."""
    n_slots = min(max_slots, max((len(d) for view in slp_by_view for d in view.values()), default=1))
    det = np.full((len(frames), len(VIEWS), n_slots, n_nodes, 2), DET_NONE, dtype=np.int16)
    f0 = int(frames[0])
    for v, view in enumerate(slp_by_view):
        for fi, dets in view.items():
            t = fi - f0
            if not 0 <= t < len(frames):
                continue
            for slot, d in enumerate(dets[:n_slots]):
                d = np.asarray(d[:n_nodes], dtype=float)
                det[t, v, slot] = np.where(np.isfinite(d), np.round(d * DET_SCALE), DET_NONE).astype(np.int16)
    return det


def unpack_detections(det: np.ndarray) -> np.ndarray:
    """:func:`detection_pack` -> float pixels with NaN for missing."""
    return np.where(det == DET_NONE, np.nan, det.astype(np.float32) / DET_SCALE)


# ---------------------------------------------------------------------------
# Overlapping fish in one view


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = np.einsum("...i,...i->...", ab, ab)
    t = np.clip(np.einsum("...i,...i->...", p - a, ab) / np.where(denom > 0, denom, 1.0), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def _cross(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
    return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])


def segment_distance(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Distance between 2D segments ``a-b`` and ``c-d`` (``(..., 2)``); NaN if an end is missing."""
    dist = np.minimum.reduce([_point_segment_distance(a, c, d), _point_segment_distance(b, c, d),
                              _point_segment_distance(c, a, b), _point_segment_distance(d, a, b)])
    crossing = (_cross(a, b, c) * _cross(a, b, d) < 0) & (_cross(c, d, a) * _cross(c, d, b) < 0)
    return np.where(crossing, 0.0, dist)


def skeleton_distance(p: np.ndarray, q: np.ndarray, edges: list[tuple[int, int]]) -> np.ndarray:
    """Closest approach of two 2D skeletons ``(..., N, 2)`` over their edges and nodes; inf if unknown."""
    with np.errstate(invalid="ignore"):
        node = np.linalg.norm(p[..., :, None, :] - q[..., None, :, :], axis=-1)
        best = np.nan_to_num(node, nan=np.inf).min(axis=(-2, -1))
        for a, b in edges:
            for c, d in edges:
                seg = segment_distance(p[..., a, :], p[..., b, :], q[..., c, :], q[..., d, :])
                best = np.minimum(best, np.nan_to_num(seg, nan=np.inf))
    return best


def overlap_distance(kp: np.ndarray, edges: list[tuple[int, int]], chunk: int = 50_000) -> np.ndarray:
    """``(F, K, V)`` distance from each fish's skeleton to the nearest other fish in the same view."""
    n_frames, n_fish = kp.shape[:2]
    out = np.full((n_frames, n_fish, kp.shape[2]), np.inf)
    for s in range(0, n_frames, chunk):
        part = kp[s:s + chunk].astype(float)
        for i in range(n_fish):
            for j in range(i + 1, n_fish):
                d = skeleton_distance(part[:, i], part[:, j], edges)   # (f, V)
                out[s:s + chunk, i] = np.minimum(out[s:s + chunk, i], d)
                out[s:s + chunk, j] = np.minimum(out[s:s + chunk, j], d)
    return out


def drop_overlapped_views(kp: np.ndarray, present: np.ndarray, distance: np.ndarray,
                          overlap_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Leave the most-overlapped view out of a fish's node triangulation.

    Only fish seen in all three views lose a view, so two always remain. A node
    that the remaining views see fewer than twice keeps the dropped view.
    Returns the keypoints for triangulation and ``dropped (F, K, V)``.
    """
    candidate = np.where((distance < overlap_px) & present, distance, np.inf)
    worst = candidate.argmin(axis=-1)
    drop_any = np.isfinite(candidate.min(axis=-1)) & (present.sum(axis=-1) >= 3)
    dropped = np.zeros(present.shape, dtype=bool)
    t, k = np.nonzero(drop_any)
    dropped[t, k, worst[t, k]] = True
    out = kp.copy()
    out[dropped] = np.nan
    seen = np.isfinite(out).all(axis=-1).sum(axis=2)                  # (F, K, N)
    restore = dropped[:, :, :, None] & (seen < 2)[:, :, None, :]     # (F, K, V, N)
    out[restore] = kp[restore]
    return out, dropped


def orient_by_projection(kp: np.ndarray, projected_ends: np.ndarray, head: int, tail: int) -> np.ndarray:
    """True where swapping head and tail brings them closer to the projected 3D ends.

    ``kp (..., N, 2)``; ``projected_ends (..., 2, 2)`` = projected head, tail.
    With one detected end, that end is labelled by the nearer projected end.
    False where no detected end or a projected end is missing.
    """
    h, t = kp[..., head, :], kp[..., tail, :]
    ph, pt = projected_ends[..., 0, :], projected_ends[..., 1, :]
    with np.errstate(invalid="ignore"):
        keep = np.nansum([np.linalg.norm(h - ph, axis=-1), np.linalg.norm(t - pt, axis=-1)], axis=0)
        swap = np.nansum([np.linalg.norm(h - pt, axis=-1), np.linalg.norm(t - ph, axis=-1)], axis=0)
    ok = (np.isfinite(h).all(-1) | np.isfinite(t).all(-1)) & np.isfinite(projected_ends).all(axis=(-2, -1))
    return ok & (swap < keep)


# ---------------------------------------------------------------------------
# Head-tail orientation


def head_tail_nodes(node_names: list[str]) -> tuple[int, int]:
    head = node_names.index("mouth") if "mouth" in node_names else 0
    tail = node_names.index("tail") if "tail" in node_names else len(node_names) - 1
    return head, tail


def flip_nodes(kp: np.ndarray, flip: np.ndarray, head: int, tail: int) -> np.ndarray:
    """Swap head and tail nodes where ``flip`` is true. ``kp[..., N, 2]``, ``flip`` broadcast to ``kp[..., 0, 0]``."""
    out = kp.copy()
    f = np.broadcast_to(flip, kp.shape[:-2])
    out[f, head] = kp[f, tail]
    out[f, tail] = kp[f, head]
    return out


VIEW_FLIP_COMBOS = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]], dtype=bool)


def evaluate_chunked(cal, obs: np.ndarray, chunk: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """``cal.evaluate_hypotheses`` over ``obs (..., 3, 2)`` in bounded-memory chunks."""
    lead = obs.shape[:-2]
    flat = obs.reshape(-1, 3, 2)
    xyz = np.full((len(flat), 3), np.nan)
    score = np.full(len(flat), np.inf)
    for s in range(0, len(flat), chunk):
        xyz[s:s + chunk], score[s:s + chunk] = cal.evaluate_hypotheses(flat[s:s + chunk])
    return xyz.reshape(lead + (3,)), score.reshape(lead)


def cross_view_flips(cal, kp: np.ndarray, head: int, tail: int, chunk: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Per (frame, fish): view flips (relative to top) that minimise head+tail ray residual.

    Returns ``flips (F, K, V)`` and ``residual (F, K)`` (sum of head and tail
    scores; inf when fewer than two views see both ends).
    """
    n_frames, n_fish = kp.shape[:2]
    flips = np.zeros((n_frames, n_fish, len(VIEWS)), dtype=bool)
    best_res = np.full((n_frames, n_fish), np.inf)
    for s in range(0, n_frames, chunk):
        part = kp[s:s + chunk]                                    # (f, K, V, N, 2)
        h, t = part[:, :, :, head], part[:, :, :, tail]          # (f, K, V, 2)
        res = []
        for combo in VIEW_FLIP_COMBOS:
            c = combo[None, None, :, None]
            hh, tt = np.where(c, t, h), np.where(c, h, t)
            _, sh = cal.evaluate_hypotheses(hh)
            _, st = cal.evaluate_hypotheses(tt)
            res.append(sh + st)
        res = np.stack(res, axis=-1)                              # (f, K, 4)
        choice = np.argmin(res, axis=-1)
        flips[s:s + chunk] = VIEW_FLIP_COMBOS[choice]
        best_res[s:s + chunk] = np.take_along_axis(res, choice[..., None], axis=-1)[..., 0]
    return flips, best_res


def smoothed_velocity(xyz: np.ndarray, half: int) -> np.ndarray:
    """Central-difference velocity (mm/frame) over +-half frames, per fish. ``xyz (F, K, 3)``."""
    vel = np.full_like(xyz, np.nan)
    if len(xyz) > 2 * half:
        vel[half:-half] = (xyz[2 * half:] - xyz[:-2 * half]) / (2 * half)
    return vel


def viterbi_orientation(head: np.ndarray, tail: np.ndarray, votes_keep: np.ndarray, n_views: np.ndarray,
                        velocity: np.ndarray, p: OrientationParams,
                        fixed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Two-state orientation for one fish over time.

    ``head``/``tail`` ``(F, 3)`` are the cross-view-consistent ends in the
    reference labelling (state 0); state 1 swaps them. ``votes_keep (F,)``
    counts views whose SLEAP labels agree with state 0, ``n_views (F,)`` the
    views that saw the fish. ``fixed (F,)`` (0 / 1, -1 = free) pins the state at
    reviewer keyframes. Returns ``state (F,)`` (-1 where invalid) and the
    per-frame emission margin (cost of the other state minus the chosen one).
    """
    n = len(head)
    state = np.full(n, -1, dtype=np.int8)
    margin = np.full(n, np.nan)
    valid = np.flatnonzero(np.isfinite(head).all(1) & np.isfinite(tail).all(1) & (n_views > 0))
    if len(valid) == 0:
        return state, margin

    axis = head - tail
    speed = np.linalg.norm(velocity, axis=1)
    forward = np.einsum("ij,ij->i", np.nan_to_num(axis), np.nan_to_num(velocity))
    weight = np.clip(np.nan_to_num(speed) / p.v_ref_mm, 0.0, 1.0) * p.w_motion
    emit0 = p.w_vote * (n_views - votes_keep) + np.where(forward < 0, weight, 0.0)
    emit1 = p.w_vote * votes_keep + np.where(forward > 0, weight, 0.0)
    if fixed is not None:
        emit0 = np.where(fixed == 1, emit0 + 1e6, emit0)
        emit1 = np.where(fixed == 0, emit1 + 1e6, emit1)

    m = len(valid)
    cost = np.empty((m, 2))
    back = np.zeros((m, 2), dtype=np.int8)
    i0 = valid[0]
    cost[0] = (emit0[i0], emit1[i0])
    for j in range(1, m):
        a, b = valid[j - 1], valid[j]
        if b - a > p.max_gap:
            same = cross = 0.0
        else:
            same = p.w_cont * (np.linalg.norm(head[b] - head[a]) + np.linalg.norm(tail[b] - tail[a]))
            cross = p.w_cont * (np.linalg.norm(head[b] - tail[a]) + np.linalg.norm(tail[b] - head[a]))
        c0, c1 = cost[j - 1]
        from0, from1 = c0 + same, c1 + cross          # to state 0
        to1_from0, to1_from1 = c0 + cross, c1 + same  # to state 1
        back[j, 0] = 0 if from0 <= from1 else 1
        back[j, 1] = 0 if to1_from0 <= to1_from1 else 1
        cost[j, 0] = min(from0, from1) + emit0[b]
        cost[j, 1] = min(to1_from0, to1_from1) + emit1[b]

    s = int(np.argmin(cost[-1]))
    for j in range(m - 1, -1, -1):
        state[valid[j]] = s
        s = int(back[j, s])
    chosen = state[valid]
    margin[valid] = np.where(chosen == 0, emit1[valid] - emit0[valid], emit0[valid] - emit1[valid])
    return state, margin


def end_jumps(head: np.ndarray, tail: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where head and tail exchange places between consecutive valid frames of one fish.

    ``head``/``tail`` ``(F, D)``. Returns ``valid (F,)``, ``new_stretch (F,)``
    (first valid frame after a gap longer than ``max_gap`` - continuity is not
    judged across it) and ``jump (F,)`` (head is closer to the previous tail
    than to the previous head, and vice versa, summed).
    """
    valid = np.isfinite(head).all(axis=1) & np.isfinite(tail).all(axis=1)
    idx = np.flatnonzero(valid)
    new_stretch = np.zeros(len(head), dtype=bool)
    jump = np.zeros(len(head), dtype=bool)
    if len(idx) == 0:
        return valid, new_stretch, jump
    new_stretch[idx[0]] = True
    a, b = idx[:-1], idx[1:]
    gap = b - a > max_gap
    new_stretch[b[gap]] = True
    same = np.linalg.norm(head[b] - head[a], axis=1) + np.linalg.norm(tail[b] - tail[a], axis=1)
    cross = np.linalg.norm(head[b] - tail[a], axis=1) + np.linalg.norm(tail[b] - head[a], axis=1)
    jump[b] = (cross < same) & ~gap
    return valid, new_stretch, jump


def orientation_segments(head: np.ndarray, tail: np.ndarray, max_gap: int) -> np.ndarray:
    """Segment id per frame (-1 where head or tail is missing).

    A segment is a run over which head and tail move continuously; a new one
    starts after a gap longer than ``max_gap`` or where head and tail exchange
    places. Within a segment the head/tail labelling is either right
    throughout or wrong throughout, so a human fix applies to a whole segment.
    """
    valid, new_stretch, jump = end_jumps(head, tail, max_gap)
    seg = np.cumsum(new_stretch | jump) - 1
    return np.where(valid, seg, -1).astype(np.int32)


def short_flip_mask(head: np.ndarray, tail: np.ndarray, max_frames: int, max_gap: int) -> np.ndarray:
    """``(F,)`` True on frames whose head/tail should be swapped back.

    Within one continuous stretch, every frame gets a parity (number of
    head<->tail exchanges so far, mod 2). Segments spanning at least
    ``max_frames`` frames keep their parity; shorter ones take the parity of
    the nearest long segment before them (after them at the stretch start).
    Stretches without a long segment are left alone.
    """
    valid, new_stretch, jump = end_jumps(head, tail, max_gap)
    out = np.zeros(len(head), dtype=bool)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return out
    stretch = np.cumsum(new_stretch[idx])
    for s in np.unique(stretch):
        fr = idx[stretch == s]
        parity = np.cumsum(jump[fr]) % 2
        seg = np.cumsum(jump[fr])
        starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
        ends = np.r_[starts[1:], len(fr)] - 1
        long_ = (fr[ends] - fr[starts] + 1) >= max_frames
        if not long_.any():
            continue
        target = np.empty(len(starts), dtype=np.int64)
        last = None
        for i in range(len(starts)):
            if long_[i]:
                last = parity[starts[i]]
            target[i] = last if last is not None else -1
        first_long = parity[starts[np.flatnonzero(long_)[0]]]
        target[target < 0] = first_long
        for i in np.flatnonzero(~long_):
            if parity[starts[i]] != target[i]:
                out[fr[starts[i]:ends[i] + 1]] = True
    return out


# ---------------------------------------------------------------------------
# Completing every node in 3D

NODE_TRIANGULATED, NODE_ONE_VIEW, NODE_BODY_MODEL, NODE_MISSING = 0, 1, 2, -1


def interp_rows(values: np.ndarray) -> np.ndarray:
    """NaN rows of ``values (F, D)`` filled by linear interpolation in time (nearest value at the ends)."""
    out = values.copy()
    ok = np.isfinite(values).all(axis=1)
    if ok.sum() == 0 or ok.all():
        return out
    idx = np.arange(len(values))
    for d in range(values.shape[1]):
        out[~ok, d] = np.interp(idx[~ok], idx[ok], values[ok, d])
    return out


def single_view_fit(project, view: np.ndarray, obs: np.ndarray, prior: np.ndarray, anchor: np.ndarray,
                    length: np.ndarray, p: FillParams) -> np.ndarray:
    """3D points seen in one view only.

    Gauss-Newton on: reprojection in that view (``obs``, px), a prior
    position (``prior``, the body-model prediction - it fixes the depth along
    the camera ray) and, where ``length`` is finite, the distance to
    ``anchor``. ``project(points (n, m, 3)) -> (n, m, V, 2)``; ``view (n,)``.
    """
    pts = prior.astype(float).copy()
    n = len(pts)
    rows = np.arange(n)
    use_len = np.isfinite(length) & np.isfinite(anchor).all(axis=1)
    steps = np.eye(3) * p.step_mm
    a_prior = np.broadcast_to(np.eye(3) / p.sigma_mm, (n, 3, 3))
    for _ in range(p.iterations):
        proj = project(np.concatenate([pts[:, None], pts[:, None] + steps[None]], axis=1))[rows, :, view]  # (n,4,2)
        f0 = proj[:, 0]
        jac = np.swapaxes((proj[:, 1:] - f0[:, None]) / p.step_mm, 1, 2)                        # (n, 2, 3)
        img_ok = (np.isfinite(f0).all(axis=1) & np.isfinite(jac).all(axis=(1, 2)) & np.isfinite(obs).all(axis=1))
        a_img = np.where(img_ok[:, None, None], np.nan_to_num(jac) / p.sigma_px, 0.0)
        r_img = np.where(img_ok[:, None], np.nan_to_num(f0 - obs) / p.sigma_px, 0.0)
        diff = pts - np.nan_to_num(anchor)
        dist = np.linalg.norm(diff, axis=1)
        unit = diff / np.where(dist > 1e-9, dist, 1.0)[:, None]
        ok_len = use_len & (dist > 1e-9)
        a_len = np.where(ok_len[:, None, None], unit[:, None, :] / p.sigma_len_mm, 0.0)
        r_len = np.where(ok_len, (dist - np.nan_to_num(length)) / p.sigma_len_mm, 0.0)[:, None]
        a = np.concatenate([a_img, a_prior, a_len], axis=1)                                      # (n, 6, 3)
        r = np.concatenate([r_img, (pts - prior) / p.sigma_mm, r_len], axis=1)                  # (n, 6)
        ata = np.einsum("nij,nik->njk", a, a)
        atr = np.einsum("nij,ni->nj", a, r)
        delta = -np.linalg.solve(ata, atr[..., None])[..., 0]
        pts = pts + np.where(np.isfinite(delta), delta, 0.0)
    return pts


def fill_nodes(nodes3d: np.ndarray, kp_obs: np.ndarray, xyz: np.ndarray, head: int, tail: int, project,
               p: FillParams | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Every node in 3D wherever the fish has a 3D position.

    ``nodes3d (F, K, N, 3)`` triangulated (NaN where fewer than two views saw
    the node), ``kp_obs (F, K, V, N, 2)`` the head/tail-corrected 2D nodes,
    ``xyz (F, K, 3)`` the 3D centroid. The middle node (pectoral) is anchored
    to the centroid, mouth and tail to the pectoral node: a missing node is
    predicted as anchor + its offset interpolated in time (scaled to the
    fish's median length for mouth / tail); seen in one view, it is then
    fitted to that view's 2D point (:func:`single_view_fit`).

    Returns ``filled (F, K, N, 3)`` and ``source (F, K, N)``:
    ``NODE_TRIANGULATED`` (0), ``NODE_ONE_VIEW`` (1), ``NODE_BODY_MODEL`` (2,
    no view: anchor + interpolated offset) or ``NODE_MISSING`` (-1, no 3D
    centroid either).
    """
    p = p or FillParams()
    n_frames, n_fish, n_nodes, _ = nodes3d.shape
    filled = nodes3d.astype(float).copy()
    source = np.where(np.isfinite(nodes3d).all(axis=-1), NODE_TRIANGULATED, NODE_MISSING).astype(np.int8)
    seen = np.isfinite(kp_obs).all(axis=-1)                                  # (F, K, V, N)
    n_views = seen.sum(axis=2)                                               # (F, K, N)
    mids = [n for n in range(n_nodes) if n not in (head, tail)]
    mid = mids[0] if mids else None
    order = ([mid] if mid is not None else []) + [n for n in range(n_nodes) if n != mid]
    for k in range(n_fish):
        has_xyz = np.isfinite(xyz[:, k]).all(axis=1)
        for n in order:
            anchor = xyz[:, k] if n == mid or mid is None else filled[:, k, mid]
            tri = source[:, k, n] == NODE_TRIANGULATED
            todo = ~tri & has_xyz & np.isfinite(anchor).all(axis=1)
            if not tri.any() or not todo.any():
                continue
            offset = np.where(tri[:, None], filled[:, k, n] - anchor, np.nan)
            length = np.nan
            if n != mid and mid is not None:
                length = float(np.nanmedian(np.linalg.norm(offset, axis=1)))
            offset = interp_rows(offset)
            if np.isfinite(length):
                norm = np.linalg.norm(offset, axis=1)
                offset = offset / np.where(norm > 1e-9, norm, 1.0)[:, None] * length
            prior = anchor + offset
            one = np.flatnonzero(todo & (n_views[:, k, n] == 1))
            if len(one):
                view = np.argmax(seen[one, k, :, n], axis=1)
                filled[one, k, n] = single_view_fit(project, view, kp_obs[one, k, view, n].astype(float),
                                                    prior[one], anchor[one], np.full(len(one), length), p)
                source[one, k, n] = NODE_ONE_VIEW
            rest = todo & (n_views[:, k, n] != 1)
            filled[rest, k, n] = prior[rest]
            source[rest, k, n] = NODE_BODY_MODEL
    return filled, source


# ---------------------------------------------------------------------------
# Risk windows


def runs(mask: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    """Inclusive (start, end) index runs where ``mask`` is true and at least ``min_len`` long."""
    if not mask.any():
        return []
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a + 1 >= min_len]


def raw_risks(frames: np.ndarray, fish_ids: list[str], xyz: np.ndarray, is_interp: np.ndarray, present: np.ndarray,
              top_xy: np.ndarray, node_resid: np.ndarray, bound: np.ndarray, orient_override: np.ndarray,
              p: RiskParams, ht_jump: np.ndarray | None = None) -> list[dict]:
    rows: list[dict] = []

    def add(kind, a, b, fish, score, details, peak=None):
        rows.append({"type": kind, "start": int(frames[a]), "end": int(frames[b]),
                     "peak": int(frames[a if peak is None else peak]), "fish": fish,
                     "score": float(score), "details": details})

    n_fish = len(fish_ids)
    for i in range(n_fish):
        for j in range(i + 1, n_fish):
            pair = f"{fish_ids[i]}|{fish_ids[j]}"
            d3 = np.linalg.norm(xyz[:, i] - xyz[:, j], axis=1)
            d2 = np.linalg.norm(top_xy[:, i] - top_xy[:, j], axis=1)
            close = (np.nan_to_num(d3, nan=np.inf) < p.close_mm) | (np.nan_to_num(d2, nan=np.inf) < p.top_close_px)
            for a, b in runs(close):
                seg = np.nan_to_num(d3[a:b + 1], nan=np.inf)
                k = a + int(np.argmin(seg))
                if np.isfinite(seg[k - a]):
                    add("close", a, b, pair, seg[k - a], f"min 3D distance {seg[k - a]:.1f} mm", k)
                else:
                    add("close", a, b, pair, np.nan, "close in top view", k)

    for k, fid in enumerate(fish_ids):
        step = np.r_[np.nan, np.linalg.norm(np.diff(xyz[:, k], axis=0), axis=1)]
        for a, b in runs(np.nan_to_num(step) > p.jump_mm):
            add("jump", a, b, fid, float(np.nanmax(step[a:b + 1])), f"step {np.nanmax(step[a:b + 1]):.1f} mm/frame")
        for a, b in runs(is_interp[:, k], p.min_gap_frames):
            add("gap", a, b, fid, b - a + 1, f"interpolated {b - a + 1} frames")
        missing = (~present[:, k]).any(axis=1) & present[:, k].any(axis=1)
        for a, b in runs(missing, p.missing_view_min_frames):
            add("missing_view", a, b, fid, b - a + 1, f"seen in fewer than 3 views for {b - a + 1} frames")
        med = np.nanmedian(node_resid[:, k], axis=1)
        for a, b in runs(np.nan_to_num(med) > p.residual_mm2, p.residual_min_frames):
            add("residual", a, b, fid, float(np.nanmax(med[a:b + 1])), f"node residual up to {np.nanmax(med[a:b + 1]):.1f} mm^2")
        unbound = ~bound[:, k] & np.isfinite(xyz[:, k]).all(axis=1)
        for a, b in runs(unbound, p.unbound_min_frames):
            add("unbound", a, b, fid, b - a + 1, f"not bound to a top-camera track for {b - a + 1} frames")
        for a, b in runs(orient_override[:, k], p.orientation_min_frames):
            add("orientation", a, b, fid, b - a + 1, f"head/tail set against SLEAP majority for {b - a + 1} frames")
        if ht_jump is not None:
            for t in np.flatnonzero(ht_jump[:, k]):
                add("head_tail_jump", t, t, fid, 1.0, "head and tail exchange places")

    count = np.isfinite(xyz).all(axis=2).sum(axis=1)
    for t in np.flatnonzero(np.diff(count) != 0) + 1:
        add("fish_count", t, t, "", float(count[t]), f"fish with 3D: {count[t - 1]} -> {count[t]}")
    return rows


def merge_risks(rows: list[dict], pad_frames: int, first: int, last: int) -> pd.DataFrame:
    """Pad and union overlapping raw risks into review windows."""
    if not rows:
        return pd.DataFrame(columns=RISK_COLUMNS)
    rows = sorted(rows, key=lambda r: r["start"])
    merged: list[dict] = []
    for r in rows:
        a, b = max(first, r["start"] - pad_frames), min(last, r["end"] + pad_frames)
        if merged and a <= merged[-1]["frame_end"]:
            m = merged[-1]
            m["frame_end"] = max(m["frame_end"], b)
            m["members"].append(r)
        else:
            merged.append({"frame_start": a, "frame_end": b, "members": [r]})
    out = []
    for i, m in enumerate(merged):
        members = m["members"]
        lead = next((r for r in members if r["type"] == "close"), members[0])
        out.append({
            "risk_id": f"R{i + 1:05d}",
            "frame_start": m["frame_start"], "frame_end": m["frame_end"], "frame_peak": lead["peak"],
            "types": ";".join(sorted({r["type"] for r in members})),
            "fish": ";".join(sorted({r["fish"] for r in members if r["fish"]})),
            "score": lead["score"],
            "details": " | ".join(dict.fromkeys(r["details"] for r in members)),
        })
    return pd.DataFrame(out, columns=RISK_COLUMNS)


# ---------------------------------------------------------------------------
# Main build


def file_fingerprint(path: Path) -> dict:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return {"path": str(path), "size": path.stat().st_size, "sha256": h.hexdigest()}


def calibration_folder(recon_dir: Path, tag: str, fallback: Path) -> Path:
    prov = recon_dir / f"{tag}_reconstruction_provenance.json"
    try:
        folder = Path(json.loads(prov.read_text(encoding="utf-8"))["triangulation"]["provenance"]["model_folder"])
        if folder.is_dir():
            return folder
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return fallback


def build_review_data(exp: str, tag: str, slp_dir: Path, recon_dir: Path, out_dir: Path, calibration_dir: Path,
                      *, orient: OrientationParams | None = None, risk: RiskParams | None = None,
                      occlusion: OcclusionParams | None = None, fill: FillParams | None = None,
                      robust=None, interval=None, fps: float = 166.0, log=print) -> dict:
    from scripts.reconstruction_3d.calibration import Calibration

    warnings.filterwarnings("ignore", message="(Mean of empty slice|All-NaN slice encountered)")
    orient = orient or OrientationParams()
    risk = risk or RiskParams()
    occlusion = occlusion or OcclusionParams()
    fill = fill or FillParams()
    t_start = time.time()
    best_path = recon_dir / f"{tag}_best.json"
    csv_3d = recon_dir / f"{tag}_3d.csv"
    csv_id = recon_dir / f"{tag}_3d_identified.csv"
    best = json.loads(best_path.read_text(encoding="utf-8"))
    fish_ids = list(best["id_labels"])
    best_frames = best["frames"]
    keys = np.array(sorted(int(k) for k in best_frames))
    frames = np.arange(keys[0], keys[-1] + 1, dtype=np.int64)
    n_frames, n_fish = len(frames), len(fish_ids)
    log(f"[review_data] {exp} {tag}: frames {frames[0]}-{frames[-1]}, fish {fish_ids}")

    slp_by_view, node_names, edges = [], None, None
    for view in VIEWS:
        t0 = time.time()
        nodes, names, view_edges = load_slp_nodes(slp_dir / f"{view}.slp", int(frames[0]), int(frames[-1]))
        if node_names is None:
            node_names, edges = names, view_edges
        elif names != node_names:
            raise ValueError(f"{view}.slp skeleton {names} differs from {node_names}")
        slp_by_view.append(nodes)
        log(f"[review_data] loaded {view}.slp ({len(nodes)} frames, {time.time() - t0:.0f} s)")
    n_nodes = len(node_names)
    head, tail = head_tail_nodes(node_names)

    slots = np.full((n_frames, n_fish, len(VIEWS)), -1, dtype=np.int8)
    kp_raw, present = gather_keypoints(best_frames, slp_by_view, fish_ids, frames, n_nodes, slots)
    det = detection_pack(slp_by_view, frames, n_nodes)
    del slp_by_view

    # 3D centroids, interpolation flags and bind info from the reconstruction CSVs.
    xyz = np.full((n_frames, n_fish, 3), np.nan)
    is_interp = np.zeros((n_frames, n_fish), dtype=bool)
    bound = np.zeros((n_frames, n_fish), dtype=bool)
    bind_label: dict[str, str] = {}
    fish_index = {f: k for k, f in enumerate(fish_ids)}
    g = pd.read_csv(csv_3d)
    t_idx = g["frame"].to_numpy() - frames[0]
    k_idx = g["fish_id"].map(fish_index).to_numpy()
    ok = (t_idx >= 0) & (t_idx < n_frames) & ~pd.isna(k_idx)
    t_idx, k_idx = t_idx[ok], k_idx[ok].astype(int)
    xyz[t_idx, k_idx] = g.loc[ok, ["X", "Y", "Z"]].to_numpy()
    if "is_interpolated" in g:
        is_interp[t_idx, k_idx] = g.loc[ok, "is_interpolated"].astype(bool).to_numpy()
    if csv_id.exists():
        gi = pd.read_csv(csv_id)
        ti = gi["frame"].to_numpy() - frames[0]
        ki = gi["fish_id"].map(fish_index).to_numpy()
        okk = (ti >= 0) & (ti < n_frames) & ~pd.isna(ki)
        bound[ti[okk], ki[okk].astype(int)] = (gi.loc[okk, "bind_source"] == "top_cam_2d").to_numpy()
        bind_label = gi.groupby("fish_id")["final_fish_id"].first().astype(str).to_dict()

    t0 = time.time()
    overlap_dist = overlap_distance(kp_raw, edges)
    view_overlap = (overlap_dist < occlusion.overlap_px) & present
    if occlusion.drop_view:
        kp_tri, view_dropped = drop_overlapped_views(kp_raw, present, overlap_dist, occlusion.overlap_px)
    else:
        kp_tri, view_dropped = kp_raw, np.zeros(present.shape, dtype=bool)
    log(f"[review_data] overlapping fish: {view_overlap.any(axis=(1, 2)).mean():.1%} of frames, "
        f"a view left out of triangulation in {view_dropped.any(axis=(1, 2)).mean():.1%} ({time.time() - t0:.0f} s)")

    from scripts.review.reconstruct import IntervalParams, RobustParams, close_intervals, reconstruct_tracks

    cal = Calibration(str(calibration_dir), backend="auto")
    robust = robust or RobustParams()
    interval = interval or IntervalParams()
    t0 = time.time()
    rec = reconstruct_tracks(cal, kp_raw, kp_tri, xyz, head, tail, orient, fill, robust, log=log)
    kp, nodes3d, nodes_filled, node_source = rec["kp2d"], rec["nodes3d"], rec["nodes3d_filled"], rec["node_source"]
    state, margin, orient_override = rec["orientation_state"], rec["orientation_margin"], rec["orientation_override"]
    node_resid, ht_jump, auto_flipped = rec["node_resid"], rec["head_tail_jump"], rec["auto_flipped"]
    segments, view_fixed, short_fixed = rec["orientation_segment"], rec["view_fixed"], rec["short_flip_fixed"]
    log(f"[review_data] reconstruction: {int(ht_jump.sum())} head/tail jumps left, nodes from one view "
        f"{np.mean(node_source == NODE_ONE_VIEW):.3%}, body model {np.mean(node_source == NODE_BODY_MODEL):.3%} "
        f"({time.time() - t0:.0f} s)")
    proj = np.full((n_frames, n_fish, len(VIEWS), 2), np.nan)
    rows_xyz = np.argwhere(np.isfinite(xyz).all(axis=2))
    for s in range(0, len(rows_xyz), 200_000):
        r = rows_xyz[s:s + 200_000]
        proj[r[:, 0], r[:, 1]] = cal.project_world(xyz[r[:, 0], r[:, 1]])
    log(f"[review_data] node triangulation + projection ({time.time() - t0:.0f} s)")

    top_xy = np.nanmean(kp[:, :, 0], axis=2) if n_nodes else np.full((n_frames, n_fish, 2), np.nan)
    top_xy = np.where(np.isfinite(top_xy), top_xy, proj[:, :, 0])
    rows = raw_risks(frames, fish_ids, xyz, is_interp, present, top_xy, node_resid, bound, orient_override, risk,
                     ht_jump)
    intervals = close_intervals(frames, xyz, fish_ids, fps, interval)
    risks = merge_risks(rows, round(risk.pad_seconds * fps), int(frames[0]), int(frames[-1]))

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "review_data.partial.npz"
    np.savez_compressed(
        tmp,
        frames=frames, fish_ids=np.array(fish_ids), node_names=np.array(node_names),
        edges=np.array(edges, dtype=np.int16).reshape(-1, 2), head_node=head, tail_node=tail,
        kp2d=kp.astype(np.float32), view_present=present, auto_flipped=auto_flipped,
        nodes3d=nodes3d.astype(np.float32), node_resid=node_resid.astype(np.float32),
        view_left_out=rec["view_left_out"], despiked=rec["despiked"],
        xyz=xyz.astype(np.float32), is_interpolated=is_interp, bound=bound,
        proj2d=proj.astype(np.float32), orientation_state=state, orientation_margin=margin.astype(np.float32),
        orientation_override=orient_override, orientation_segment=segments, short_flip_fixed=short_fixed,
        nodes3d_filled=nodes_filled.astype(np.float32), node_source=node_source,
        missing_node_proj2d=rec["missing_node_proj2d"],
        view_overlap=view_overlap, view_dropped=view_dropped,
        machine_label=np.array([bind_label.get(f, f) for f in fish_ids]),
    )
    os.replace(tmp, out_dir / "review_data.npz")
    tmp = out_dir / "detections.partial.npz"
    np.savez_compressed(tmp, frames=frames, det=det, slots=slots, scale=DET_SCALE, none=DET_NONE)
    os.replace(tmp, out_dir / "detections.npz")
    risks.to_csv(out_dir / "machine_risk.csv", index=False)
    intervals.to_csv(out_dir / "machine_intervals.csv", index=False)

    valid_state = state >= 0
    manifest = {
        "data_version": DATA_VERSION, "experiment": exp, "round_tag": tag,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "fps": fps,
        "frame_range": [int(frames[0]), int(frames[-1])], "fish_ids": fish_ids, "node_names": node_names,
        "sources": {"best_json": file_fingerprint(best_path), "csv_3d": file_fingerprint(csv_3d),
                    "csv_identified": file_fingerprint(csv_id) if csv_id.exists() else None,
                    "slp_dir": str(slp_dir), "calibration_dir": str(calibration_dir)},
        "orientation_params": asdict(orient), "risk_params": asdict(risk), "occlusion_params": asdict(occlusion),
        "fill_params": asdict(fill), "robust_params": asdict(robust), "interval_params": asdict(interval),
        "summary": {
            "frames_with_overlap": int(view_overlap.any(axis=(1, 2)).sum()),
            "frames_with_view_dropped": int(view_dropped.any(axis=(1, 2)).sum()),
            "orientation_frames": int(valid_state.sum()),
            "frames_with_any_view_auto_flipped": int(auto_flipped.any(axis=2).sum()),
            "orientation_override_frames": int(orient_override.sum()),
            "short_flip_fixed_view_frames": int(view_fixed.sum()),
            "short_flip_fixed_fish_frames": int(short_fixed.sum()),
            "head_tail_jumps": int(ht_jump.sum()),
            "node_frames_from_two_views_after_disagreement": int((rec["view_left_out"] >= 0).sum()),
            "node_frames_despiked": int(rec["despiked"].sum()),
            "machine_intervals": int(len(intervals)),
            "node_source_fraction": {
                name: {kind: round(float(np.mean(node_source[:, :, n] == code)), 6)
                       for kind, code in (("triangulated", NODE_TRIANGULATED), ("one_view", NODE_ONE_VIEW),
                                          ("body_model", NODE_BODY_MODEL), ("missing", NODE_MISSING))}
                for n, name in enumerate(node_names)},
            "risk_windows": int(len(risks)),
            "risk_window_frames": int((risks["frame_end"] - risks["frame_start"] + 1).sum()) if len(risks) else 0,
            "risk_types": {t: int(sum(t in s.split(";") for s in risks["types"])) for t in
                           sorted({r["type"] for r in rows})},
            "seconds": round(time.time() - t_start, 1),
        },
    }
    (out_dir / "review_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"[review_data] {len(risks)} risk windows; done in {time.time() - t_start:.0f} s")
    return manifest


def main(argv: list[str] | None = None) -> int:
    from scripts.common.local_paths import load_local_paths
    from scripts.sleap.batch_bottomup_track import DEFAULT_ROUND_TAG, default_paths_config

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment")
    p.add_argument("--round-tag", default=DEFAULT_ROUND_TAG)
    p.add_argument("--paths-config", type=Path, default=None)
    p.add_argument("--review-root", type=Path, default=None, help="Default: <data_root>/review_packages.")
    p.add_argument("--drop-overlapped-view", action="store_true",
                   help="Leave a fish's most-overlapped view out of the node triangulation (see module docstring).")
    args = p.parse_args(argv)

    paths = load_local_paths(args.paths_config or default_paths_config())
    if paths is None:
        raise SystemExit("paths config not found")
    review_root = args.review_root or paths.data_root / "review_packages"
    recon_dir = paths.reconstruction_output_dir / args.experiment / args.round_tag
    meta = paths.medaka_data_dir / args.experiment / "experiment_metadata.json"
    fps = 166.0
    if meta.exists():
        fps = float(json.loads(meta.read_text(encoding="utf-8"))["views"]["top_cam"]["frame_rate"])
    build_review_data(
        args.experiment, args.round_tag,
        slp_dir=paths.sleap_output_dir / args.experiment / args.round_tag, recon_dir=recon_dir,
        out_dir=review_root / args.experiment / args.round_tag,
        calibration_dir=calibration_folder(recon_dir, args.round_tag, paths.calibration_model_dir), fps=fps,
        occlusion=OcclusionParams(drop_view=args.drop_overlapped_view),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
