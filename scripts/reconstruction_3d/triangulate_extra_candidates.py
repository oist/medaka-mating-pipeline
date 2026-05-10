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
    xyz_bounds: Tuple[float, float] | None = None,
    num_individuals: int = 3,
    num_candidates: int = 6,
    recover_missing_cam0: bool = True,
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
    if cal._model_xyz2im is None:
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
            reproj = cal._model_xyz2im.predict(xyz[valid])
            diff = ic[valid] - reproj
            err[valid] = np.sum(diff ** 2, axis=1)

        # Greedy: pick K mutually-exclusive triplets in ascending cost.
        # ``chosen`` rows: (cam0_idx, cam1_idx, cam2_idx, xyz, err, recovered)
        order = np.argsort(err, kind="stable")
        used_i: set[int] = set()
        used_j: set[int] = set()
        used_k: set[int] = set()
        chosen: List[tuple[int, int, int, np.ndarray, float, bool]] = []
        for r in order:
            if not np.isfinite(err[r]):
                break
            i, j, k = idx_tab[r]
            i_int, j_int, k_int = int(i), int(j), int(k)
            if i_int in used_i or j_int in used_j or k_int in used_k:
                continue
            chosen.append((i_int, j_int, k_int, xyz[r].copy(), float(err[r]), False))
            used_i.add(i_int)
            used_j.add(j_int)
            used_k.add(k_int)
            if len(chosen) == K:
                break

        # Pass 2: cam0 recovery. If we couldn't fill all K fish from real
        # triplets, try to recover any remaining fish whose top-down detection
        # is missing but whose side views have leftover unused detections.
        if (
            recover_missing_cam0
            and len(chosen) < K
            and cal._model_xy_from_xz_yz is not None
        ):
            avail_j = np.array(
                [j for j in range(N) if j not in used_j and np.all(np.isfinite(pts1[j]))],
                dtype=int,
            )
            avail_k = np.array(
                [k for k in range(N) if k not in used_k and np.all(np.isfinite(pts2[k]))],
                dtype=int,
            )
            if avail_j.size > 0 and avail_k.size > 0:
                jjr, kkr = np.meshgrid(avail_j, avail_k, indexing="ij")
                jjr = jjr.ravel()
                kkr = kkr.ravel()

                # Synthesise cam0 from (cam1, cam2): predict (xy) from (xz, yz).
                pair_input = np.hstack([pts1[jjr], pts2[kkr]])  # (n_pairs, 4)
                cam0_pred = np.asarray(cal._model_xy_from_xz_yz.predict(pair_input))
                if cam0_pred.ndim == 1:
                    cam0_pred = cam0_pred.reshape(-1, 2)

                ic_r = np.empty((len(jjr), 6), dtype=float)
                ic_r[:, 0:2] = cam0_pred
                ic_r[:, 2:4] = pts1[jjr]
                ic_r[:, 4:6] = pts2[kkr]
                finite_r = np.all(np.isfinite(ic_r), axis=1)

                xyz_r = np.full((len(jjr), 3), np.nan, dtype=float)
                if finite_r.any():
                    xyz_r[finite_r] = cal._model_im2xyz.predict(ic_r[finite_r])
                finite_xyz_r = finite_r & np.all(np.isfinite(xyz_r), axis=1)

                if low is not None and high is not None:
                    in_box_r = np.all((xyz_r >= low) & (xyz_r <= high), axis=1)
                    valid_r = finite_xyz_r & in_box_r
                else:
                    valid_r = finite_xyz_r

                err_r = np.full(len(jjr), np.inf, dtype=float)
                if valid_r.any():
                    reproj_r = cal._model_xyz2im.predict(xyz_r[valid_r])
                    err_r[valid_r] = np.sum((ic_r[valid_r] - reproj_r) ** 2, axis=1)

                order_r = np.argsort(err_r, kind="stable")
                for r in order_r:
                    if len(chosen) == K:
                        break
                    if not np.isfinite(err_r[r]):
                        break
                    j_idx = int(jjr[r])
                    k_idx = int(kkr[r])
                    if j_idx in used_j or k_idx in used_k:
                        continue
                    # cam0_idx = -1 marks the cam0 detection as synthesised.
                    chosen.append(
                        (-1, j_idx, k_idx, xyz_r[r].copy(), float(err_r[r]), True)
                    )
                    used_j.add(j_idx)
                    used_k.add(k_idx)

        groups = {lb: [float("nan")] * 3 for lb in labels}
        errors = {lb: float("inf") for lb in labels}
        perm_cam0 = [-1] * K
        perm_cam1 = [-1] * K
        perm_cam2 = [-1] * K
        recovered = [False] * K
        for slot, (i, j, k, point, e, was_recovered) in enumerate(chosen):
            lb = labels[slot]
            groups[lb] = [float(point[0]), float(point[1]), float(point[2])]
            errors[lb] = e
            perm_cam0[slot] = i
            perm_cam1[slot] = j
            perm_cam2[slot] = k
            recovered[slot] = was_recovered

        best_by_frame[frame] = {
            "groups": groups,
            "errors": errors,
            "perm_cam0": perm_cam0,
            "perm_cam1": perm_cam1,
            "perm_cam2": perm_cam2,
            "recovered": recovered,
            "dropped_high_residual": [],
        }

    return best_by_frame
