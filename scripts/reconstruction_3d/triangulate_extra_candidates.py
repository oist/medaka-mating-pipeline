from __future__ import annotations

"""
Triangulation when each view exposes more candidates than fish.

Background
----------
The default ``triangulate.compute_candidates`` keeps cam0 as the anchor and
enumerates per-fish permutations of cam1/cam2 — it expects exactly
``num_individuals`` candidates per view. When SLEAP runs at ``-n 6`` (= 2*nfish)
to recover real fish that would otherwise be displaced by reflections, the
cam0 anchor assumption no longer holds: any of the 6 cam0 detections might be
a reflection.

This module batch-scores all (cam0_i, cam1_j, cam2_k) triplets (default 6^3 =
216 per frame) through the calibration's ``imCoords_to_XYZ`` model, drops
triplets whose 3D point falls outside the tank cube (``xyz_bounds``), and
greedily picks ``num_individuals`` mutually-exclusive triplets minimising
reprojection error.

Output shape matches what ``select_best_all_frames`` returns, so this can
plug directly into the existing cross-frame tracker.
"""

from typing import Dict, List, Tuple

import numpy as np

from scripts.reconstruction_3d.calibration import Calibration
from scripts.reconstruction_3d.tracklet_first_assignment import (
    ReconstructionCandidate,
    candidate_identity,
    select_global_candidate_set,
)


def _default_labels(num_individuals: int) -> List[str]:
    return [chr(ord("a") + i) for i in range(num_individuals)]


def _pad_to_n(pts: List[Tuple[float, float]] | None, n: int) -> np.ndarray:
    out = np.full((n, 2), np.nan, dtype=float)
    if not pts:
        return out
    arr = np.asarray(pts, dtype=float)
    take = min(len(arr), n)
    out[:take] = arr[:take, :2]
    return out


def compute_best_assignments_per_frame(
    cal: Calibration,
    points_by_frame: Dict[int, Dict[str, List[Tuple[float, float]]]],
    *,
    scoring_cal: Calibration | None = None,
    xyz_bounds: Tuple[float, float] | None = None,
    num_individuals: int = 3,
    num_candidates: int = 6,
    recover_missing_cam0: bool = True,
    recover_missing_views: bool | None = None,
    max_residual_per_fish: float | None = None,
    tracklet_nodes_by_frame: Dict[int, Dict[str, Dict[int, str]]] | None = None,
    tracklet_identities: Dict[str, str] | None = None,
    motion_weight: float = 0.0,
    max_motion_mm: float | None = None,
    max_motion_hard_cap_mm: float | None = None,
    partial_view_penalty: float = 1_000_000.0,
    global_max_states: int | None = 512,
) -> Dict[int, Dict]:
    """Pick the best ``num_individuals`` mutually-exclusive 3D triplets per frame.

    Two passes:

    1. *Real triplets* — enumerate every (cam0_i, cam1_j, cam2_k) combination
       across all valid detections, triangulate, drop those outside
       ``xyz_bounds``, greedily pick the ``num_individuals`` mutually-exclusive
       triplets with the lowest reprojection error.
    2. *Cam0 recovery* (when ``recover_missing_cam0=True``) — if fewer than
       ``num_individuals`` real triplets were viable, the top-down (cam0)
       detection is presumed missing for the leftover fish. For each
       still-unused (cam1_j, cam2_k) pair, predict the synthetic cam0
       coordinate via ``Calibration.compute_XY_imcoords_from_XZ_YZ``,
       triangulate the completed triplet, score by reprojection error, and
       greedily fill remaining fish slots. Recovered fish carry
       ``perm_cam0[slot] = -1`` to mark the cam0 detection as synthetic.

    Returns a per-frame dict in the same shape as ``select_best_all_frames``::

        {
          frame: {
            "groups":     {"a": [x,y,z], "b": [...], "c": [...]},
            "errors":     {"a": float, "b": float, "c": float},
            "perm_cam0":  [i_a, i_b, i_c],   # cam0 detection indices, -1 if recovered
            "perm_cam1":  [j_a, j_b, j_c],   # cam1 detection indices used
            "perm_cam2":  [k_a, k_b, k_c],   # cam2 detection indices used
            "recovered":  [bool, bool, bool],  # True iff cam0 was synthesised
            "dropped_high_residual": [],
          }
        }

    Unfilled fish slots get ``NaN`` xyz, ``inf`` error and ``-1`` perm index.
    """
    scoring_cal = scoring_cal or cal
    if recover_missing_views is None:
        recover_missing_views = recover_missing_cam0
    if scoring_cal._model_xyz2im is None:
        raise RuntimeError(
            "Calibration must include XYZ_to_imCoords.joblib for "
            "compute_best_assignments_per_frame (we score by reprojection error)."
        )

    low_high = xyz_bounds
    low = low_high[0] if low_high is not None else None
    high = low_high[1] if low_high is not None else None

    K = num_individuals
    N = num_candidates
    labels = _default_labels(K)

    # (N^3, 3) index table over all (i,j,k) triplets.
    ii, jj, kk = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    idx_tab = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x

    best_by_frame: Dict[int, Dict] = {}
    last_xyz_by_identity: Dict[str, tuple[int, np.ndarray]] = {}

    for frame in tqdm(sorted(points_by_frame.keys()), desc="3D best-assign", unit="frame"):
        cams = points_by_frame[frame]
        pts0 = _pad_to_n(cams.get("cam0", []), N)
        pts1 = _pad_to_n(cams.get("cam1", []), N)
        pts2 = _pad_to_n(cams.get("cam2", []), N)

        # (N^3, 6) image-coord matrix per triplet, in cam0/cam1/cam2 order.
        ic = np.empty((N ** 3, 6), dtype=float)
        ic[:, 0:2] = pts0[idx_tab[:, 0]]
        ic[:, 2:4] = pts1[idx_tab[:, 1]]
        ic[:, 4:6] = pts2[idx_tab[:, 2]]
        finite_in = np.all(np.isfinite(ic), axis=1)

        # Batch triangulation (skip rows with NaN coords to keep the predictor sane).
        xyz = np.full((N ** 3, 3), np.nan, dtype=float)
        if finite_in.any():
            xyz[finite_in] = cal._model_im2xyz.predict(ic[finite_in])
        finite_xyz = finite_in & np.all(np.isfinite(xyz), axis=1)

        if low is not None and high is not None:
            in_box = np.all((xyz >= low) & (xyz <= high), axis=1)
            valid = finite_xyz & in_box
        else:
            valid = finite_xyz

        # Batch reprojection error.
        err = np.full(N ** 3, np.inf, dtype=float)
        if valid.any():
            scoring_xyz = scoring_cal._model_im2xyz.predict(ic[valid])
            reproj = scoring_cal._model_xyz2im.predict(scoring_xyz)
            diff = ic[valid] - reproj
            err[valid] = np.sum(diff ** 2, axis=1)

        slot_nodes_frame = (tracklet_nodes_by_frame or {}).get(frame, {})
        slot_nodes = [
            slot_nodes_frame.get("cam0", {}),
            slot_nodes_frame.get("cam1", {}),
            slot_nodes_frame.get("cam2", {}),
        ]
        identities = tracklet_identities or {}

        def _make_candidate(
            slots: tuple[int, int, int],
            point: np.ndarray,
            residual: float,
            *,
            synthetic_view: int | None = None,
        ) -> ReconstructionCandidate:
            identity, conflict, evidence = candidate_identity(
                slots, slot_nodes=slot_nodes, node_identities=identities
            )
            motion_cost = 0.0
            motion_rejected = False
            if identity is not None and identity in last_xyz_by_identity:
                last_frame, last_point = last_xyz_by_identity[identity]
                distance = float(np.linalg.norm(point - last_point))
                motion_cost = float(motion_weight * distance)
                gap = max(1, int(frame - last_frame))
                threshold = max_motion_mm * gap if max_motion_mm is not None else None
                if max_motion_hard_cap_mm is not None:
                    threshold = (
                        max_motion_hard_cap_mm if threshold is None
                        else min(threshold, max_motion_hard_cap_mm)
                    )
                motion_rejected = threshold is not None and distance > threshold
            return ReconstructionCandidate(
                slots=slots,
                xyz=tuple(float(v) for v in point),
                residual=float(residual + (partial_view_penalty if synthetic_view is not None else 0.0)),
                identity=identity,
                synthetic_view=synthetic_view,
                motion_cost=motion_cost,
                identity_conflict=conflict,
                motion_rejected=motion_rejected,
                metadata={"identity_evidence": evidence, "raw_residual": float(residual)},
            )

        candidates: list[ReconstructionCandidate] = []
        for r in np.where(valid & np.isfinite(err))[0]:
            raw_error = float(err[r])
            if max_residual_per_fish is not None and raw_error > max_residual_per_fish:
                continue
            slots = tuple(int(v) for v in idx_tab[r])
            candidates.append(_make_candidate(slots, xyz[r], raw_error))

        real_selection = select_global_candidate_set(
            candidates, max_selected=K, max_states=global_max_states
        )

        # If three-view geometry cannot reconstruct all fish, add calibrated
        # two-view hypotheses for whichever camera is absent.  Synthetic slots
        # are -1 and therefore never consume a real detection.
        if recover_missing_views and len(real_selection.selected) < K:
            pts = (pts0, pts1, pts2)
            helper = {
                0: scoring_cal._model_xy_from_xz_yz,
                1: scoring_cal._model_xz_from_xy_yz,
                2: scoring_cal._model_yz_from_xz_xy,
            }
            for missing_view in range(3):
                model = helper[missing_view]
                if model is None:
                    continue
                observed = [view for view in range(3) if view != missing_view]
                a_idx = np.where(np.isfinite(pts[observed[0]]).all(axis=1))[0]
                b_idx = np.where(np.isfinite(pts[observed[1]]).all(axis=1))[0]
                if not len(a_idx) or not len(b_idx):
                    continue
                aa, bb = np.meshgrid(a_idx, b_idx, indexing="ij")
                aa, bb = aa.ravel(), bb.ravel()
                # Pairwise helper feature order follows the calibration file
                # names.  The cam2 helper is the one non-camera-order case:
                # ``xz_xy_to_yz`` expects cam1 before cam0.
                if missing_view == 2:
                    pair_input = np.hstack([pts[1][bb], pts[0][aa]])
                else:
                    pair_input = np.hstack([pts[observed[0]][aa], pts[observed[1]][bb]])
                predicted = np.asarray(model.predict(pair_input), dtype=float).reshape(-1, 2)
                ic_p = np.empty((len(aa), 6), dtype=float)
                slots_rows: list[tuple[int, int, int]] = []
                for row, (a_slot, b_slot) in enumerate(zip(aa, bb)):
                    triplet = np.full((3, 2), np.nan, dtype=float)
                    triplet[observed[0]] = pts[observed[0]][a_slot]
                    triplet[observed[1]] = pts[observed[1]][b_slot]
                    triplet[missing_view] = predicted[row]
                    ic_p[row] = triplet.ravel()
                    slots = [-1, -1, -1]
                    slots[observed[0]] = int(a_slot)
                    slots[observed[1]] = int(b_slot)
                    slots_rows.append(tuple(slots))
                xyz_p = cal._model_im2xyz.predict(ic_p)
                scoring_xyz_p = scoring_cal._model_im2xyz.predict(ic_p)
                reproj_p = scoring_cal._model_xyz2im.predict(scoring_xyz_p)
                err_p = np.sum((ic_p - reproj_p) ** 2, axis=1)
                valid_p = np.isfinite(xyz_p).all(axis=1) & np.isfinite(err_p)
                if low is not None and high is not None:
                    valid_p &= np.all((xyz_p >= low) & (xyz_p <= high), axis=1)
                for row in np.where(valid_p)[0]:
                    raw_error = float(err_p[row])
                    if max_residual_per_fish is not None and raw_error > max_residual_per_fish:
                        continue
                    candidates.append(
                        _make_candidate(
                            slots_rows[row], xyz_p[row], raw_error,
                            synthetic_view=missing_view,
                        )
                    )

        selection = select_global_candidate_set(
            candidates, max_selected=K, max_states=global_max_states
        )
        chosen = list(selection.selected)
        for candidate in chosen:
            if candidate.identity is not None:
                last_xyz_by_identity[candidate.identity] = (
                    int(frame), np.asarray(candidate.xyz, dtype=float)
                )

        groups = {lb: [float("nan")] * 3 for lb in labels}
        errors = {lb: float("inf") for lb in labels}
        perm_cam0 = [-1] * K
        perm_cam1 = [-1] * K
        perm_cam2 = [-1] * K
        recovered = [False] * K
        synthetic_views = [-1] * K
        source_id_hints: dict[int, str] = {}
        for slot, candidate in enumerate(chosen):
            i, j, k = candidate.slots
            point = np.asarray(candidate.xyz, dtype=float)
            e = float(candidate.metadata.get("raw_residual", candidate.residual))
            was_recovered = candidate.synthetic_view is not None
            lb = labels[slot]
            groups[lb] = [float(point[0]), float(point[1]), float(point[2])]
            errors[lb] = e
            perm_cam0[slot] = i
            perm_cam1[slot] = j
            perm_cam2[slot] = k
            recovered[slot] = was_recovered
            synthetic_views[slot] = (
                int(candidate.synthetic_view) if candidate.synthetic_view is not None else -1
            )
            if candidate.identity is not None:
                source_id_hints[slot] = candidate.identity

        best_by_frame[frame] = {
            "groups": groups,
            "errors": errors,
            "perm_cam0": perm_cam0,
            "perm_cam1": perm_cam1,
            "perm_cam2": perm_cam2,
            "recovered": recovered,
            "synthetic_views": synthetic_views,
            "source_id_hints": source_id_hints,
            "selection_diagnostics": {
                "alternatives_considered": selection.alternatives_considered,
                "identity_conflicts": selection.rejected_identity_conflicts,
                "motion_gate_rejections": selection.rejected_motion_gate,
                "candidate_count": len(candidates),
                "selection_kind": "global_tracklet_constrained",
            },
            "dropped_high_residual": [],
        }

    return best_by_frame
