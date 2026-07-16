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
216 per frame) through the selected public calibration backend, drops triplets
whose 3D point falls outside the tank cube (``xyz_bounds``), and globally
selects ``num_individuals`` mutually-exclusive triplets. Two-view hypotheses
keep the absent camera row as NaN; each backend owns its reconstruction policy.

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
    max_hypothesis_score: float | None = None,
    max_residual_per_fish: float | None = None,
    tracklet_nodes_by_frame: Dict[int, Dict[str, Dict[int, str]]] | None = None,
    tracklet_identities: Dict[str, str] | None = None,
    motion_weight: float = 0.0,
    max_motion_mm: float | None = None,
    max_motion_hard_cap_mm: float | None = None,
    partial_view_penalty_score_units: float = 1_000_000.0,
    partial_view_penalty: float | None = None,
    global_max_states: int | None = 512,
    evaluation_batch_frames: int = 256,
) -> Dict[int, Dict]:
    """Pick the best ``num_individuals`` mutually-exclusive 3D triplets per frame.

    Two passes:

    1. *Real triplets* — enumerate every (cam0_i, cam1_j, cam2_k) combination
       across all valid detections, triangulate, drop those outside
       ``xyz_bounds``, then globally select the mutually-exclusive set with the
       lowest declared backend score.
    2. *Two-view recovery* — if fewer than ``num_individuals`` three-view
       hypotheses are viable, enumerate pairs for each absent camera. The
       absent row remains NaN. Refractive calibration intersects the two real
       rays directly; polynomial calibration may use its legacy pairwise
       helper internally. Recovered fish carry ``perm_camN[slot] = -1``.

    Returns a per-frame dict in the same shape as ``select_best_all_frames``::

        {
          frame: {
            "groups":     {"a": [x,y,z], "b": [...], "c": [...]},
            "errors":     {"a": float, "b": float, "c": float},
            "perm_cam0":  [i_a, i_b, i_c],   # cam0 detection indices, -1 if recovered
            "perm_cam1":  [j_a, j_b, j_c],   # cam1 detection indices used
            "perm_cam2":  [k_a, k_b, k_c],   # cam2 detection indices used
            "recovered":  [bool, bool, bool],  # True iff one camera was absent
            "missing_views": [int, int, int],  # absent camera index, else -1
            "dropped_high_residual": [],
          }
        }

    Unfilled fish slots get ``NaN`` xyz, ``inf`` error and ``-1`` perm index.
    """
    scoring_cal = scoring_cal or cal
    if recover_missing_views is None:
        recover_missing_views = recover_missing_cam0
    if max_hypothesis_score is not None and max_residual_per_fish is not None:
        raise ValueError(
            "Use max_hypothesis_score or legacy max_residual_per_fish, not both"
        )
    if max_residual_per_fish is not None:
        if scoring_cal.score_unit != "px^2":
            raise ValueError(
                "max_residual_per_fish is a legacy pixel-squared gate and cannot "
                f"be used with {scoring_cal.backend_name!r} scores in "
                f"{scoring_cal.score_unit}; use max_hypothesis_score"
            )
        max_hypothesis_score = max_residual_per_fish
    if partial_view_penalty is not None:
        partial_view_penalty_score_units = float(partial_view_penalty)
    if evaluation_batch_frames < 1:
        raise ValueError("evaluation_batch_frames must be at least one")
    # Scikit/joblib polynomial predictors can differ in their final floating
    # bits when BLAS sees a different row batch. Keep their historical
    # per-frame evaluation shape for strict reproducibility. Refractive ray
    # construction/solving is row-independent and exactly batch-stable.
    effective_batch_frames = (
        evaluation_batch_frames
        if cal.backend_name == "refractive"
        and scoring_cal.backend_name == "refractive"
        else 1
    )

    low_high = xyz_bounds
    low = low_high[0] if low_high is not None else None
    high = low_high[1] if low_high is not None else None

    K = num_individuals
    N = num_candidates
    labels = _default_labels(K)

    def _evaluate(observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized XYZ + scoring, fused when one backend can share rays."""
        if scoring_cal is cal and hasattr(cal, "evaluate_hypotheses"):
            xyz_values, score_values = cal.evaluate_hypotheses(
                observations, profile="production_fast"
            )
        else:
            xyz_values = cal.triangulate(observations)
            score_values = scoring_cal.score_hypotheses(
                observations, profile="production_fast"
            )
        return (
            np.asarray(xyz_values, dtype=float),
            np.asarray(score_values, dtype=float),
        )

    # (N^3, 3) index table over all (i,j,k) triplets.
    ii, jj, kk = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    idx_tab = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x

    best_by_frame: Dict[int, Dict] = {}
    last_xyz_by_identity: Dict[str, tuple[int, np.ndarray]] = {}
    frames = sorted(points_by_frame)

    def _full_evaluations():
        """Yield per-frame full-view evaluations from bounded frame batches."""
        for chunk_start in range(0, len(frames), effective_batch_frames):
            chunk_frames = frames[
                chunk_start : chunk_start + effective_batch_frames
            ]
            points_chunk: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            observations_chunk = np.empty(
                (len(chunk_frames), N ** 3, 3, 2), dtype=float
            )
            for local_index, frame in enumerate(chunk_frames):
                cams = points_by_frame[frame]
                pts0 = _pad_to_n(cams.get("cam0", []), N)
                pts1 = _pad_to_n(cams.get("cam1", []), N)
                pts2 = _pad_to_n(cams.get("cam2", []), N)
                points_chunk.append((pts0, pts1, pts2))
                observations_chunk[local_index, :, 0] = pts0[idx_tab[:, 0]]
                observations_chunk[local_index, :, 1] = pts1[idx_tab[:, 1]]
                observations_chunk[local_index, :, 2] = pts2[idx_tab[:, 2]]
            finite_chunk = np.isfinite(observations_chunk).all(axis=(2, 3))
            xyz_chunk = np.full(
                (len(chunk_frames), N ** 3, 3), np.nan, dtype=float
            )
            score_chunk = np.full(
                (len(chunk_frames), N ** 3), np.inf, dtype=float
            )
            if np.any(finite_chunk):
                flat_observations = observations_chunk.reshape(-1, 3, 2)
                flat_finite = finite_chunk.ravel()
                evaluated_xyz, evaluated_scores = _evaluate(
                    flat_observations[flat_finite]
                )
                xyz_chunk.reshape(-1, 3)[flat_finite] = evaluated_xyz
                score_chunk.ravel()[flat_finite] = evaluated_scores
            for local_index, frame in enumerate(chunk_frames):
                yield (
                    frame,
                    *points_chunk[local_index],
                    observations_chunk[local_index],
                    finite_chunk[local_index],
                    xyz_chunk[local_index],
                    score_chunk[local_index],
                )

    evaluation_iterator = tqdm(
        _full_evaluations(), total=len(frames), desc="3D best-assign", unit="frame"
    )
    for (
        frame,
        pts0,
        pts1,
        pts2,
        observations,
        finite_in,
        xyz,
        all_scores,
    ) in evaluation_iterator:
        # Full three-view hypotheses. Missing-view hypotheses are enumerated
        # separately below so their absent source slot stays explicit.
        finite_xyz = finite_in & np.all(np.isfinite(xyz), axis=1)

        if low is not None and high is not None:
            in_box = np.all((xyz >= low) & (xyz <= high), axis=1)
            valid = finite_xyz & in_box
        else:
            valid = finite_xyz

        # The scoring backend declares the unit; geometry and scoring may be
        # deliberately different when assignments are frozen to another model.
        err = np.full(N ** 3, np.inf, dtype=float)
        if valid.any():
            err[valid] = all_scores[valid]

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
                residual=float(
                    residual
                    + (
                        partial_view_penalty_score_units
                        if synthetic_view is not None
                        else 0.0
                    )
                ),
                identity=identity,
                synthetic_view=synthetic_view,
                motion_cost=motion_cost,
                identity_conflict=conflict,
                motion_rejected=motion_rejected,
                metadata={
                    "identity_evidence": evidence,
                    "raw_residual": float(residual),
                    "score_unit": scoring_cal.score_unit,
                    "triangulation_backend": cal.backend_name,
                    "scoring_backend": scoring_cal.backend_name,
                    "scoring_profile": "production_fast",
                    "observed_view_count": 2 if synthetic_view is not None else 3,
                },
            )

        candidates: list[ReconstructionCandidate] = []
        for r in np.where(valid & np.isfinite(err))[0]:
            raw_error = float(err[r])
            if max_hypothesis_score is not None and raw_error > max_hypothesis_score:
                continue
            slots = tuple(int(v) for v in idx_tab[r])
            candidates.append(_make_candidate(slots, xyz[r], raw_error))

        real_selection = select_global_candidate_set(
            candidates, max_selected=K, max_states=global_max_states
        )

        # If three-view geometry cannot reconstruct all fish, add calibrated
        # two-view hypotheses for whichever camera is absent.  Synthetic slots
        # are -1 and therefore never consume a real detection.
        needs_partial_recovery = (
            recover_missing_views and len(real_selection.selected) < K
        )
        if needs_partial_recovery:
            pts = (pts0, pts1, pts2)
            for missing_view in range(3):
                observed = [view for view in range(3) if view != missing_view]
                a_idx = np.where(np.isfinite(pts[observed[0]]).all(axis=1))[0]
                b_idx = np.where(np.isfinite(pts[observed[1]]).all(axis=1))[0]
                if not len(a_idx) or not len(b_idx):
                    continue
                aa, bb = np.meshgrid(a_idx, b_idx, indexing="ij")
                aa, bb = aa.ravel(), bb.ravel()
                observations_p = np.full((len(aa), 3, 2), np.nan, dtype=float)
                observations_p[:, observed[0], :] = pts[observed[0]][aa]
                observations_p[:, observed[1], :] = pts[observed[1]][bb]
                slots_rows: list[tuple[int, int, int]] = []
                for a_slot, b_slot in zip(aa, bb):
                    slots = [-1, -1, -1]
                    slots[observed[0]] = int(a_slot)
                    slots[observed[1]] = int(b_slot)
                    slots_rows.append(tuple(slots))
                xyz_p, err_p = _evaluate(observations_p)
                valid_p = np.isfinite(xyz_p).all(axis=1) & np.isfinite(err_p)
                if low is not None and high is not None:
                    valid_p &= np.all((xyz_p >= low) & (xyz_p <= high), axis=1)
                for row in np.where(valid_p)[0]:
                    raw_error = float(err_p[row])
                    if max_hypothesis_score is not None and raw_error > max_hypothesis_score:
                        continue
                    candidates.append(
                        _make_candidate(
                            slots_rows[row], xyz_p[row], raw_error,
                            synthetic_view=missing_view,
                        )
                    )

        selection = (
            select_global_candidate_set(
                candidates, max_selected=K, max_states=global_max_states
            )
            if needs_partial_recovery
            else real_selection
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
            "missing_views": synthetic_views,
            "synthetic_views": synthetic_views,
            "source_id_hints": source_id_hints,
            "selection_diagnostics": {
                "alternatives_considered": selection.alternatives_considered,
                "identity_conflicts": selection.rejected_identity_conflicts,
                "motion_gate_rejections": selection.rejected_motion_gate,
                "candidate_count": len(candidates),
                "selection_kind": "global_tracklet_constrained",
                "triangulation_backend": cal.backend_name,
                "scoring_backend": scoring_cal.backend_name,
                "scoring_profile": "production_fast",
                "score_unit": scoring_cal.score_unit,
                "max_hypothesis_score": max_hypothesis_score,
                "partial_view_penalty": partial_view_penalty_score_units,
                "partial_view_penalty_unit": scoring_cal.score_unit,
            },
            "score_unit": scoring_cal.score_unit,
            "triangulation_backend": cal.backend_name,
            "scoring_backend": scoring_cal.backend_name,
            "scoring_profile": "production_fast",
            "dropped_high_residual": [],
        }

    return best_by_frame
