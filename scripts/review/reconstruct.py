"""3D reconstruction of tracked fish from per-view 2D detections, shared by the review build, the GUI and
the export.

:func:`reconstruct_tracks` turns per-track 2D keypoints ``(F, K, V, N, 2)`` into oriented 3D nodes:

1. head/tail per view: cross-view consistency, a temporal Viterbi (optionally pinned at reviewer
   keyframes), short 2D swaps undone, reviewer per-view flips;
2. node triangulation; a node whose three views disagree is re-triangulated from the two views that
   agree with its own track over time (:func:`robust_nodes`); single-frame spikes are replaced by the
   local median (:func:`despike_nodes`);
3. short 3D head/tail swaps undone; every node completed (one-view fit or body model).

It also holds the reviewer-driven pieces that need the calibration: re-estimating a flagged body
(:func:`reestimate_body`), the 3D centroid of re-assigned detections (:func:`centroid_from_kp`,
:func:`group_fill`), detecting close co-moving intervals (:func:`close_intervals`) and re-assigning
detections inside such an interval between reviewer keyframes (:func:`solve_interval`).
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.review.review_data import (
    NODE_BODY_MODEL,
    NODE_MISSING,
    NODE_TRIANGULATED,
    VIEWS,
    FillParams,
    OrientationParams,
    cross_view_flips,
    end_jumps,
    evaluate_chunked,
    fill_nodes,
    flip_nodes,
    interp_rows,
    orient_by_projection,
    orientation_segments,
    short_flip_mask,
    smoothed_velocity,
    viterbi_orientation,
)

NODE_DESPIKED = 3        # node_source: single-frame spike replaced by the local median
NODE_REESTIMATED = 4     # node_source: body flagged by the reviewer, re-estimated from the body model


@dataclass
class RobustParams:
    resid_mm2: float = 4.0         # 3-view node residual above which two-view alternatives are tried
    improve_mm: float = 1.0        # an alternative must be this much closer to the node's own track
    median_half: int = 4           # frames each side for the node's temporal median
    spike_k: float = 5.0           # despike: residual to the neighbours' line > k * its robust spread ...
    spike_floor_mm: float = 2.0    # ... and at least this many mm
    spike_scale_half: int = 20     # frames each side for that spread
    enabled: bool = True           # False: plain triangulation (the 'initial' version for comparisons)


@dataclass
class IntervalParams:
    close_mm: float = 12.0         # machine intervals: 3D centroids closer than this ...
    min_seconds: float = 0.5       # ... for at least this long
    merge_gap_seconds: float = 0.2
    move_sigma_mm: float = 1.5     # solver: frame-to-frame centroid motion scale (cost is quadratic)
    max_move_mm: float = 15.0      # solver: motion beyond this costs no more (fish re-found after a gap)
    switch_cost: float = 5.0       # solver: cost of changing which detection a fish has in one view
    no_view_cost: float = 4.0      # solver: cost per fish with fewer than two views
    resid_sigma: float = 4.0       # solver: triangulation residual scale


# ---------------------------------------------------------------------------
# Temporal helpers


def rolling_median(track: np.ndarray, half: int) -> np.ndarray:
    """NaN-aware median over +-``half`` frames of ``track (F, D)``."""
    n = len(track)
    pad = np.full((half,) + track.shape[1:], np.nan)
    padded = np.concatenate([pad, track, pad])
    stack = np.stack([padded[i:i + n] for i in range(2 * half + 1)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(stack, axis=0)


def _shift(a: np.ndarray, s: int) -> np.ndarray:
    """``a`` shifted by ``s`` frames (``out[t] = a[t + s]``), NaN-padded."""
    out = np.full_like(a, np.nan)
    if s > 0:
        out[:-s] = a[s:]
    elif s < 0:
        out[-s:] = a[:s]
    else:
        out[:] = a
    return out


def despike_nodes(nodes: np.ndarray, p: RobustParams) -> np.ndarray:
    """``(F, K, N)`` True where a node is a one- or two-frame spike.

    A node's spike residual is its distance from the straight line through its neighbours, over
    spans of one and two frames (the larger of the two, so a two-frame spike is caught too). It is a
    spike when that residual exceeds ``spike_k`` x the residual's robust spread over
    +-``spike_scale_half`` frames, and ``spike_floor_mm``. Smooth fast motion (tail beats, turns)
    keeps a small residual because it is nearly linear over two frames.
    """
    n_frames, n_fish, n_nodes, _ = nodes.shape
    out = np.zeros((n_frames, n_fish, n_nodes), dtype=bool)
    for k in range(n_fish):
        for n in range(n_nodes):
            track = nodes[:, k, n].astype(float)
            r1 = np.linalg.norm(track - (_shift(track, -1) + _shift(track, 1)) / 2, axis=1)
            r2 = np.linalg.norm(track - (_shift(track, -2) + _shift(track, 2)) / 2, axis=1)
            resid = np.fmax(r1, r2)
            scale = 1.4826 * rolling_median(resid[:, None], p.spike_scale_half)[:, 0]
            spike = np.nan_to_num(resid) > np.maximum(p.spike_k * np.nan_to_num(scale, nan=np.inf),
                                                      p.spike_floor_mm)
            # A spike also raises its neighbours' residual (to about half): keep only frames near the
            # local peak (a two-frame spike has two equal peaks).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)       # frames with no node nearby
                peak = np.nanmax(np.stack([_shift(resid, s) for s in range(-2, 3)]), axis=0)
            out[:, k, n] = spike & (np.nan_to_num(resid) >= 0.75 * np.nan_to_num(peak))
    return out


def robust_nodes(cal, obs: np.ndarray, nodes: np.ndarray, resid: np.ndarray,
                 p: RobustParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-triangulate nodes whose three views disagree from the two views that agree with the track.

    ``obs (F, K, N, V, 2)``, ``nodes (F, K, N, 3)``, ``resid (F, K, N)``. A node seen in three views
    with a residual above ``resid_mm2`` gets the two-view alternative nearest to its temporal median,
    if that is at least ``improve_mm`` nearer than the three-view point. Returns nodes, residuals and
    ``dropped (F, K, N)`` = the view left out (-1 none).
    """
    nodes, resid = nodes.copy(), resid.copy()
    dropped = np.full(resid.shape, -1, dtype=np.int8)
    seen = np.isfinite(obs).all(axis=-1)                                       # (F, K, N, V)
    cand = np.argwhere((seen.sum(axis=-1) == 3) & (np.nan_to_num(resid, nan=0.0) > p.resid_mm2))
    if not len(cand):
        return nodes, resid, dropped
    ref = np.full(nodes.shape, np.nan)
    for k in range(nodes.shape[1]):
        for n in range(nodes.shape[2]):
            ref[:, k, n] = rolling_median(nodes[:, k, n].astype(float), p.median_half)
    t, k, n = cand.T
    target = ref[t, k, n]
    best = np.linalg.norm(nodes[t, k, n] - target, axis=1)
    best = np.where(np.isfinite(best), best - p.improve_mm, np.inf)
    choice = np.full(len(cand), -1)
    alt_xyz = np.full((len(cand), 3), np.nan)
    alt_res = np.full(len(cand), np.nan)
    for v in range(len(VIEWS)):
        o = obs[t, k, n].astype(float).copy()                                   # (m, V, 2)
        o[:, v] = np.nan
        xyz_v, res_v = evaluate_chunked(cal, o)
        d = np.linalg.norm(xyz_v - target, axis=1)
        better = np.isfinite(d) & (d < best)
        best = np.where(better, d, best)
        choice[better] = v
        alt_xyz[better], alt_res[better] = xyz_v[better], res_v[better]
    use = choice >= 0
    nodes[t[use], k[use], n[use]] = alt_xyz[use]
    resid[t[use], k[use], n[use]] = alt_res[use]
    dropped[t[use], k[use], n[use]] = choice[use]
    return nodes, resid, dropped


# ---------------------------------------------------------------------------
# Full reconstruction of a frame range


def reconstruct_tracks(cal, kp_raw: np.ndarray, kp_tri: np.ndarray, xyz: np.ndarray, head: int, tail: int,
                       orient: OrientationParams, fill: FillParams, robust: RobustParams | None = None, *,
                       extra_view_flip: np.ndarray | None = None, prefer_head: np.ndarray | None = None,
                       log=None) -> dict:
    """Oriented, completed 3D nodes of every track.

    ``kp_raw (F, K, V, N, 2)``: SLEAP labelling of the detections assigned to each track (display);
    ``kp_tri``: the same with views left out of triangulation set to NaN; ``xyz (F, K, 3)``: the 3D
    centroid (motion prior and completion anchor). ``extra_view_flip (F, K, V)`` = reviewer per-view
    head/tail flips; ``prefer_head (F, K, 3)`` = reviewer-confirmed head positions (NaN elsewhere),
    which pin the temporal orientation.
    """
    import time

    def say(msg):
        if log is not None:
            log(msg)

    robust = robust or RobustParams()
    n_frames, n_fish, n_views = kp_tri.shape[:3]
    t0 = time.time()
    view_flips, _ = cross_view_flips(cal, kp_tri, head, tail)
    kp_consistent = flip_nodes(kp_tri, view_flips, head, tail)
    head3d, _ = evaluate_chunked(cal, kp_consistent[:, :, :, head])
    tail3d, _ = evaluate_chunked(cal, kp_consistent[:, :, :, tail])
    end_seen = np.isfinite(kp_tri[:, :, :, head]).all(-1) & np.isfinite(kp_tri[:, :, :, tail]).all(-1)
    n_ends = end_seen.sum(axis=2)
    votes_keep = (end_seen & ~view_flips).sum(axis=2)
    velocity = smoothed_velocity(xyz, orient.velocity_half_window)
    state = np.full((n_frames, n_fish), -1, dtype=np.int8)
    margin = np.full((n_frames, n_fish), np.nan)
    for k in range(n_fish):
        fixed = None
        if prefer_head is not None and np.isfinite(prefer_head[:, k]).any():
            pref = prefer_head[:, k]
            d0 = np.linalg.norm(head3d[:, k] - pref, axis=1)
            d1 = np.linalg.norm(tail3d[:, k] - pref, axis=1)
            fixed = np.where(np.isfinite(d0) & np.isfinite(d1), (d1 < d0).astype(np.int8), -1).astype(np.int8)
        state[:, k], margin[:, k] = viterbi_orientation(head3d[:, k], tail3d[:, k], votes_keep[:, k],
                                                        n_ends[:, k], velocity[:, k], orient, fixed=fixed)
    final_flip = view_flips ^ (state == 1)[:, :, None]
    say(f"[reconstruct] head/tail orientation ({time.time() - t0:.0f} s)")

    kp_oriented = flip_nodes(kp_tri, final_flip, head, tail)
    view_fixed = np.zeros(final_flip.shape, dtype=bool)
    for k in range(n_fish):
        for v in range(n_views):
            h2, t2 = kp_oriented[:, k, v, head].astype(float), kp_oriented[:, k, v, tail].astype(float)
            # Fish pointing at the camera: head and tail project onto each other, "exchanging places" is noise.
            length = np.linalg.norm(h2 - t2, axis=1)
            if np.isfinite(length).any():
                h2[length < orient.end_on_fraction * np.nanmedian(length)] = np.nan
            view_fixed[:, k, v] = short_flip_mask(h2, t2, orient.short_flip_frames, orient.max_gap)
    final_flip ^= view_fixed
    if extra_view_flip is not None:
        final_flip ^= extra_view_flip

    t0 = time.time()
    obs = np.moveaxis(flip_nodes(kp_tri, final_flip, head, tail), 2, 3)   # (F, K, N, V, 2)
    nodes3d, node_resid = evaluate_chunked(cal, obs)
    if robust.enabled:
        nodes3d, node_resid, view_left_out = robust_nodes(cal, obs, nodes3d, node_resid, robust)
    else:
        view_left_out = np.full(node_resid.shape, -1, dtype=np.int8)

    short_fixed = np.zeros((n_frames, n_fish), dtype=bool)
    for k in range(n_fish):
        short_fixed[:, k] = short_flip_mask(nodes3d[:, k, head], nodes3d[:, k, tail],
                                            orient.short_flip_frames, orient.max_gap)
    if short_fixed.any():
        final_flip ^= short_fixed[:, :, None]
        t_s, k_s = np.nonzero(short_fixed)
        nodes3d[t_s, k_s, head], nodes3d[t_s, k_s, tail] = nodes3d[t_s, k_s, tail], nodes3d[t_s, k_s, head].copy()
        node_resid[t_s, k_s, head], node_resid[t_s, k_s, tail] = \
            node_resid[t_s, k_s, tail], node_resid[t_s, k_s, head].copy()
    node_resid = np.where(np.isfinite(node_resid), node_resid, np.nan)
    spikes = despike_nodes(nodes3d, robust) if robust.enabled else np.zeros(node_resid.shape, dtype=bool)
    say(f"[reconstruct] triangulation: {np.mean(view_left_out >= 0):.3%} node-frames re-triangulated from two "
        f"views, {spikes.mean():.3%} spikes replaced ({time.time() - t0:.0f} s)")

    votes_final = (end_seen & ~final_flip).sum(axis=2)
    orient_override = (state >= 0) & (2 * votes_final < n_ends)
    segments = np.full((n_frames, n_fish), -1, dtype=np.int32)
    ht_jump = np.zeros((n_frames, n_fish), dtype=bool)
    for k in range(n_fish):
        segments[:, k] = orientation_segments(nodes3d[:, k, head], nodes3d[:, k, tail], orient.max_gap)
        ht_jump[:, k] = end_jumps(nodes3d[:, k, head], nodes3d[:, k, tail], orient.max_gap)[2]

    t0 = time.time()
    clean = np.where(spikes[..., None], np.nan, nodes3d)
    kp_obs = flip_nodes(kp_tri, final_flip, head, tail)
    nodes_filled, node_source = fill_nodes(clean, np.where(spikes[:, :, None, :, None], np.nan, kp_obs), xyz,
                                           head, tail, cal.project_world, fill)
    node_source = np.where(spikes & (node_source != NODE_MISSING), NODE_DESPIKED, node_source).astype(np.int8)

    # Displayed keypoints: head/tail labelled to match the projection of the triangulated ends.
    display_flip = final_flip.copy()
    ends = nodes_filled[:, :, [head, tail]]
    rows_ok = np.argwhere(np.isfinite(ends).all(axis=(2, 3)))
    for s in range(0, len(rows_ok), 200_000):
        r = rows_ok[s:s + 200_000]
        proj_ends = np.moveaxis(cal.project_world(ends[r[:, 0], r[:, 1]]), 1, 2)   # (n, V, 2 ends, 2)
        by_proj = orient_by_projection(kp_raw[r[:, 0], r[:, 1]], proj_ends, head, tail)
        seen = np.isfinite(proj_ends).all(axis=(2, 3))
        display_flip[r[:, 0], r[:, 1]] = np.where(seen, by_proj, display_flip[r[:, 0], r[:, 1]])
    kp = flip_nodes(kp_raw, display_flip, head, tail)
    missing_proj = project_missing(cal, kp, nodes_filled)
    raw_end_seen = np.isfinite(kp_raw[:, :, :, head]).all(-1) & np.isfinite(kp_raw[:, :, :, tail]).all(-1)
    say(f"[reconstruct] completion + display ({time.time() - t0:.0f} s)")
    return {
        "kp2d": kp.astype(np.float32), "auto_flipped": display_flip & raw_end_seen,
        "nodes3d": nodes3d.astype(np.float32), "node_resid": node_resid.astype(np.float32),
        "nodes3d_filled": nodes_filled.astype(np.float32), "node_source": node_source,
        "missing_node_proj2d": missing_proj, "orientation_state": state,
        "orientation_margin": margin.astype(np.float32), "orientation_override": orient_override,
        "orientation_segment": segments, "short_flip_fixed": short_fixed, "head_tail_jump": ht_jump,
        "view_fixed": view_fixed, "view_left_out": view_left_out, "despiked": spikes,
    }


def project_missing(cal, kp: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Projection of each 3D node into the views where its 2D node is missing (NaN elsewhere)."""
    out = np.full(kp.shape, np.nan, dtype=np.float32)                        # (F, K, V, N, 2)
    need = ~np.isfinite(kp).all(axis=-1) & np.isfinite(nodes).all(axis=-1)[:, :, None, :]
    rows = np.argwhere(need.any(axis=2))
    for s in range(0, len(rows), 200_000):
        t, k, n = rows[s:s + 200_000].T
        proj = cal.project_world(nodes[t, k, n])                               # (m, V, 2)
        keep = need[t, k, :, n]
        out[t, k, :, n] = np.where(keep[..., None], proj, np.nan)
    return out


def project_points(cal, xyz: np.ndarray) -> np.ndarray:
    """``(..., 3)`` -> ``(..., V, 2)`` with NaN where the point is unknown."""
    out = np.full(xyz.shape[:-1] + (len(VIEWS), 2), np.nan, dtype=np.float32)
    ok = np.isfinite(xyz).all(axis=-1)
    if ok.any():
        out[ok] = cal.project_world(xyz[ok])
    return out


# ---------------------------------------------------------------------------
# Reviewer-driven re-estimation


def reestimate_body(nodes: np.ndarray, source: np.ndarray, flagged: np.ndarray, head: int,
                    tail: int) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild mouth and tail of one track on reviewer-flagged frames.

    ``nodes (F, N, 3)``, ``source (F, N)``, ``flagged (F,)``. The middle node (pectoral) is kept;
    mouth and tail are placed at the fish's median distance from it along the head and tail
    directions interpolated in time from the unflagged frames on either side of each flagged run.
    """
    nodes, source = nodes.astype(float).copy(), source.copy()
    if not flagged.any():
        return nodes, source
    mids = [n for n in range(nodes.shape[1]) if n not in (head, tail)]
    if not mids:
        return nodes, source
    mid = mids[0]
    good = ~flagged & np.isfinite(nodes).all(axis=(1, 2))
    if not good.any():
        return nodes, source
    for end in (head, tail):
        vec = nodes[:, end] - nodes[:, mid]
        length = float(np.nanmedian(np.linalg.norm(vec[good], axis=1)))
        direction = np.where(good[:, None], vec, np.nan)
        direction = interp_rows(direction)
        norm = np.linalg.norm(direction, axis=1)
        unit = direction / np.where(norm > 1e-9, norm, 1.0)[:, None]
        new = nodes[:, mid] + unit * length
        ok = flagged & np.isfinite(new).all(axis=1)
        nodes[ok, end] = new[ok]
        source[ok, end] = NODE_REESTIMATED
    return nodes, source


def centroid_from_kp(cal, kp: np.ndarray, head: int, tail: int) -> np.ndarray:
    """3D centroid as the production tracker defines it: per view the mean of mouth and tail (swap
    invariant), triangulated. ``kp (F, K, V, N, 2)`` -> ``(F, K, 3)``."""
    both = np.isfinite(kp[..., head, :]).all(-1) & np.isfinite(kp[..., tail, :]).all(-1)
    centre = np.where(both[..., None], (kp[..., head, :] + kp[..., tail, :]) / 2, np.nan)   # (F, K, V, 2)
    xyz, _ = evaluate_chunked(cal, centre)
    return xyz


def group_fill(xyz: np.ndarray, members: list[int]) -> np.ndarray:
    """Fill unknown centroids of fish moving together: the fish keeps its offset to a group mate
    (interpolated in time between frames where both are known) and moves with it - e.g. sinking
    together; else plain interpolation in time. ``xyz (F, K, 3)``."""
    out = xyz.copy()
    for k in members:
        missing = ~np.isfinite(out[:, k]).all(axis=1)
        if not missing.any():
            continue
        for j in members:
            if j == k:
                continue
            both = np.isfinite(xyz[:, k]).all(axis=1) & np.isfinite(xyz[:, j]).all(axis=1)
            if not both.any():
                continue
            offset = interp_rows(np.where(both[:, None], xyz[:, k] - xyz[:, j], np.nan))
            use = missing & np.isfinite(xyz[:, j]).all(axis=1)
            out[use, k] = xyz[use, j] + offset[use]
            missing &= ~use
        if missing.any() and np.isfinite(out[:, k]).any():
            out[:, k] = interp_rows(out[:, k])
    return out


# ---------------------------------------------------------------------------
# Close co-moving intervals


def close_intervals(frames: np.ndarray, xyz: np.ndarray, fish_ids: list[str], fps: float,
                    p: IntervalParams | None = None) -> pd.DataFrame:
    """Intervals where fish stay close (3D centroids within ``close_mm`` for ``min_seconds``),
    pairs overlapping in time merged into one group. Columns: interval_id, frame_start, frame_end,
    fish (``ID_a|ID_b[|ID_c]``), min_mm."""
    p = p or IntervalParams()
    n_fish = len(fish_ids)
    runs = []
    gap = int(round(p.merge_gap_seconds * fps))
    for a, b in itertools.combinations(range(n_fish), 2):
        d = np.nan_to_num(np.linalg.norm(xyz[:, a] - xyz[:, b], axis=1), nan=np.inf)
        close = d < p.close_mm
        edges = np.diff(np.r_[0, close.astype(np.int8), 0])
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
        merged: list[list[int]] = []
        for s, e in zip(starts, ends):
            if merged and s - merged[-1][1] <= gap:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        for s, e in merged:
            if e - s + 1 >= p.min_seconds * fps:
                runs.append([s, e, {a, b}, float(d[s:e + 1].min())])
    runs.sort(key=lambda r: r[0])
    groups: list[list] = []
    for s, e, members, dmin in runs:
        if groups and s <= groups[-1][1] + gap and groups[-1][2] & members:
            g = groups[-1]
            g[1], g[2], g[3] = max(g[1], e), g[2] | members, min(g[3], dmin)
        else:
            groups.append([s, e, set(members), dmin])
    rows = [{"interval_id": f"M{i + 1:03d}", "frame_start": int(frames[s]), "frame_end": int(frames[e]),
             "fish": "|".join(fish_ids[k] for k in sorted(members)), "min_mm": round(dmin, 1)}
            for i, (s, e, members, dmin) in enumerate(groups)]
    return pd.DataFrame(rows, columns=["interval_id", "frame_start", "frame_end", "fish", "min_mm"])


# ---------------------------------------------------------------------------
# Re-assigning detections inside an interval


def solve_interval(cal, cand: np.ndarray, keyframes: np.ndarray, p: IntervalParams | None = None,
                   key_positions: np.ndarray | None = None) -> dict:
    """Which detection belongs to which fish, per frame and view, inside one interval.

    ``cand (F, V, m, 2)``: per frame and view the 2D centroids (mean of mouth and tail) of the ``m``
    detections the group's fish currently have, in fish order (NaN = none). A state assigns, in every
    view, a permutation of those detections to the fish. Viterbi over frames with costs: triangulation
    residual of each fish's triplet, frame-to-frame motion of each fish's 3D centroid, a fixed cost
    each time a view's permutation changes (so fish only swap where the data clearly says so), and a
    cost for a fish seen in fewer than two views. ``keyframes (F,)`` True = reviewer-confirmed frames.
    There the state is forced: the one whose fish positions lie nearest ``key_positions (F, m, 3)``
    (what the reviewer confirmed), or the identity permutation where no positions are given.

    Returns ``perm (F, V, m)`` (fish -> index into that frame's detections), ``xyz (F, m, 3)``,
    ``changed (F,)`` (any view differs from the identity) and ``cost``.
    """
    p = p or IntervalParams()
    n_frames, n_views, m, _ = cand.shape
    perms = np.array(list(itertools.permutations(range(m))), dtype=np.int8)           # (P, m)
    n_perm = len(perms)
    states = np.array(list(itertools.product(range(n_perm), repeat=n_views)), dtype=np.int16)   # (S, V)
    n_states = len(states)
    # Every fish's possible triplets: one detection index per view (m ** V combinations).
    combos = np.array(list(itertools.product(range(m), repeat=n_views)), dtype=np.int8)        # (C, V)
    obs = np.stack([cand[:, v, combos[:, v]] for v in range(n_views)], axis=2)                  # (F, C, V, 2)
    xyz_c, res_c = evaluate_chunked(cal, obs.reshape(-1, n_views, 2))
    xyz_c = xyz_c.reshape(n_frames, len(combos), 3)
    res_c = res_c.reshape(n_frames, len(combos))
    combo_index = {tuple(c): i for i, c in enumerate(combos)}
    # state, fish -> combo
    det_of = perms[states]                                                                      # (S, V, m)
    combo_of = np.array([[combo_index[tuple(det_of[s, :, c])] for c in range(m)] for s in range(n_states)])
    pos = xyz_c[:, combo_of]                                                                    # (F, S, m, 3)
    known = np.isfinite(pos).all(axis=-1)
    resid = np.where(known, np.nan_to_num(res_c[:, combo_of], nan=0.0, posinf=0.0), 0.0)
    emit = (np.minimum(resid, 50.0) / p.resid_sigma).sum(axis=-1) + p.no_view_cost * (~known).sum(axis=-1)
    identity = int(np.flatnonzero((states == 0).all(axis=1))[0])          # permutation 0 is the identity
    big = 1e9
    for f in np.flatnonzero(keyframes):
        target = identity
        if key_positions is not None and np.isfinite(key_positions[f]).all(axis=-1).any():
            d = np.linalg.norm(pos[f] - key_positions[f][None], axis=-1)                    # (S, m)
            use = np.isfinite(key_positions[f]).all(axis=-1)
            dist = np.where(np.isfinite(d[:, use]), d[:, use], 1e6).sum(axis=1)
            target = int(np.argmin(dist + 1e-6 * (np.arange(n_states) != identity)))   # ties: keep identity
        emit[f] = big
        emit[f, target] = 0.0
    switches = (states[:, None, :] != states[None, :, :]).sum(axis=-1) * p.switch_cost          # (S, S)
    cost = emit[0].copy()
    back = np.zeros((n_frames, n_states), dtype=np.int16)
    for f in range(1, n_frames):
        d = np.linalg.norm(pos[f - 1][:, None] - pos[f][None, :], axis=-1)                     # (S, S, m)
        move = ((np.where(np.isfinite(d), np.minimum(d, p.max_move_mm), 0.0) / p.move_sigma_mm) ** 2).sum(axis=-1)
        total = cost[:, None] + switches + move
        back[f] = np.argmin(total, axis=0)
        cost = total[back[f], np.arange(n_states)] + emit[f]
    s = int(np.argmin(cost))
    path = np.empty(n_frames, dtype=np.int16)
    for f in range(n_frames - 1, -1, -1):
        path[f] = s
        s = int(back[f, s])
    return {"perm": det_of[path], "xyz": pos[np.arange(n_frames), path], "changed": path != identity,
            "cost": float(cost.min())}
