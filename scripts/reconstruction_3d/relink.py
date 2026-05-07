"""
Forward + backward re-linker for 3D fish identity tracking.

Re-runs the temporal linker (`apply_cross_frame_tracking`) twice — once
forward through frames, once backward — then reconciles the two
labelings. At each frame where the two passes disagree about which
detection belongs to which ID, we compare the *local trajectory jump*
each pass implies and trust the smoother one. The intuition: when two
fish pass close to each other, a greedy NN linker has roughly even odds
of swapping them; running the same linker forward and backward through
the same encounter typically only causes a swap on one side, so the
side that does *not* swap has the smaller per-ID position jump across
the encounter and is the one to trust.

Inputs:
    `_best_pre_link.json` produced by `reconstruct_from_slp.py` (or by
    `recover_pre_link.py` for runs predating that artifact).

Outputs (alongside the pre-link JSON):
    - `<tag>_3d_relinked.csv`
    - `<tag>_best_relinked.json`           (post-link, same shape as
                                             `_best.json`)
    - `<tag>_relink_decisions.csv`         (per σ-change diagnostic row)

CLI:
    python -m scripts.reconstruction_3d.relink \\
        Z:/StephensU/medaka/reconstruction_3d/20251225_K_1_2_FCMC_best_pre_link.json \\
        --max-jump-mm 30 --max-jump-hard-cap-mm 80 --max-gap-frames 100
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.reconstruction_3d.cross_frame_tracking import (
    apply_cross_frame_tracking,
)
from scripts.reconstruction_3d.fill_missing_3d import (
    best_by_frame_to_dfs,
    dfs_to_single_csv,
    ensure_all_frames_and_fill_spline,
)
from scripts.reconstruction_3d.reconstruct_from_slp import load_best_pre_link_json


# ---------------------------------------------------------------------------
# Permutation helpers (small n; tuples).
# ---------------------------------------------------------------------------

def _identity_perm(n: int) -> Tuple[int, ...]:
    return tuple(range(n))


def _compose(p: Tuple[int, ...], q: Tuple[int, ...]) -> Tuple[int, ...]:
    """Return r where r[i] = p[q[i]]."""
    return tuple(p[q[i]] for i in range(len(p)))


def _inverse(p: Tuple[int, ...]) -> Tuple[int, ...]:
    out = [0] * len(p)
    for i, v in enumerate(p):
        out[v] = i
    return tuple(out)


def _id_labels(n: int) -> List[str]:
    return [f"ID_{i + 1}" for i in range(n)]


# ---------------------------------------------------------------------------
# Backward linker.
# ---------------------------------------------------------------------------

def apply_cross_frame_tracking_backward(
    best_by_frame: Dict[int, Dict], **kwargs
) -> Dict[int, Dict]:
    """
    Run `apply_cross_frame_tracking` with frames in reverse order. We do
    this by remapping `t -> T_max - t` so the existing function (which
    sorts ascending) actually walks the reversed sequence, then mapping
    keys back.
    """
    frames = sorted(best_by_frame.keys())
    if not frames:
        return {}
    T = max(frames)
    remapped = {(T - f): info for f, info in best_by_frame.items()}
    result_remapped = apply_cross_frame_tracking(remapped, **kwargs)
    return {(T - rf): info for rf, info in result_remapped.items()}


# ---------------------------------------------------------------------------
# Position extraction + permutation alignment.
# ---------------------------------------------------------------------------

def _positions_from_result(info: Dict, n: int) -> np.ndarray:
    out = np.full((n, 3), np.nan)
    groups = info.get("groups", {}) or {}
    for i, lbl in enumerate(_id_labels(n)):
        pt = groups.get(lbl)
        if pt is None:
            continue
        arr = np.asarray(pt, dtype=float)
        if arr.shape == (3,) and not np.isnan(arr).any():
            out[i] = arr
    return out


def _best_alignment(
    fwd_pos: np.ndarray, bwd_pos: np.ndarray, n: int
) -> Tuple[Tuple[int, ...] | None, float]:
    """
    Return σ such that bwd_pos[σ(i)] best matches fwd_pos[i], plus the
    sum-of-distances cost. Skips frames where either side has any NaN
    (σ is undefined — the position sets are not directly comparable).
    """
    if np.isnan(fwd_pos).any() or np.isnan(bwd_pos).any():
        return None, float("inf")
    best_sigma: Tuple[int, ...] | None = None
    best_cost = float("inf")
    for sigma in permutations(range(n)):
        cost = 0.0
        for i in range(n):
            cost += float(np.linalg.norm(fwd_pos[i] - bwd_pos[sigma[i]]))
        if cost < best_cost:
            best_cost = cost
            best_sigma = tuple(sigma)
    return best_sigma, best_cost


# ---------------------------------------------------------------------------
# Reconciliation.
# ---------------------------------------------------------------------------

def relink_forward_backward(
    best_by_frame: Dict[int, Dict],
    *,
    num_individuals: int = 3,
    sigma_tolerance_mm: float = 5.0,
    **linker_kwargs,
) -> Tuple[Dict[int, Dict], List[Dict]]:
    """
    Run forward + backward linkers and reconcile.

    Output frames are keyed by `ID_1..ID_n` and contain `groups`
    (positions taken from the forward pass, possibly relabeled) plus
    bookkeeping (`source_index_for_id`, `pi_after_perm`).

    The second return value is a per-event decision log.

    `sigma_tolerance_mm` — minimum total alignment cost beneath which we
    accept a non-identity σ as legitimate (i.e., positions match cleanly
    after permutation). Above this threshold the alignment is suspect
    (likely two passes diverging on a degenerate frame) and we skip the
    σ-change check.
    """
    n = num_individuals
    fwd = apply_cross_frame_tracking(
        best_by_frame, num_individuals=n, **linker_kwargs
    )
    bwd = apply_cross_frame_tracking_backward(
        best_by_frame, num_individuals=n, **linker_kwargs
    )
    frames = sorted(fwd.keys())
    if not frames:
        return {}, []

    fwd_pos = {t: _positions_from_result(fwd[t], n) for t in frames}
    bwd_pos = {t: _positions_from_result(bwd[t], n) for t in frames}

    # Per-frame σ (None where undefined or alignment is dubious).
    sigma: Dict[int, Tuple[int, ...] | None] = {}
    for t in frames:
        s, cost = _best_alignment(fwd_pos[t], bwd_pos[t], n)
        if s is None or cost > sigma_tolerance_mm:
            sigma[t] = None
        else:
            sigma[t] = s

    # Walk frames; maintain π = output_id -> fwd_id.
    pi: Tuple[int, ...] = _identity_perm(n)
    prev_valid_sigma: Tuple[int, ...] | None = None
    prev_valid_t: int | None = None

    decisions: List[Dict] = []
    output: Dict[int, Dict] = {}
    id_lbls = _id_labels(n)

    for t in frames:
        s = sigma[t]
        if (
            s is not None
            and prev_valid_sigma is not None
            and s != prev_valid_sigma
            and prev_valid_t is not None
        ):
            # σ changed between prev_valid_t and t. Decide which side swapped
            # by comparing per-ID trajectory jump on each side.
            f_jump = float(
                np.linalg.norm(fwd_pos[t] - fwd_pos[prev_valid_t], axis=1).mean()
            )
            b_jump = float(
                np.linalg.norm(bwd_pos[t] - bwd_pos[prev_valid_t], axis=1).mean()
            )
            delta_sigma = _compose(s, _inverse(prev_valid_sigma))
            fwd_swapped = f_jump > b_jump
            if fwd_swapped:
                pi = _compose(pi, delta_sigma)
            decisions.append({
                "frame": int(t),
                "prev_valid_frame": int(prev_valid_t),
                "side_swapped": "fwd" if fwd_swapped else "bwd",
                "fwd_jump_mm": f_jump,
                "bwd_jump_mm": b_jump,
                "delta_sigma": list(delta_sigma),
                "pi_after": list(pi),
            })

        if s is not None:
            prev_valid_sigma = s
            prev_valid_t = t

        # Emit output positions: output_id i -> fwd's pi[i].
        f_pos_t = fwd_pos[t]
        out_groups: Dict[str, List[float]] = {}
        for i, lbl in enumerate(id_lbls):
            src_idx = pi[i]
            pt = f_pos_t[src_idx]
            if np.any(np.isnan(pt)):
                out_groups[lbl] = [float("nan")] * 3
            else:
                out_groups[lbl] = [float(v) for v in pt]
        output[t] = {
            "groups": out_groups,
            "perm_cam1": list(fwd[t].get("perm_cam1", [])),
            "perm_cam2": list(fwd[t].get("perm_cam2", [])),
            "source_index_for_id": [int(pi[i]) for i in range(n)],
        }

    return output, decisions


# ---------------------------------------------------------------------------
# Output writers (mirror reconstruct_from_slp.py conventions).
# ---------------------------------------------------------------------------

def _save_relinked_best_json(
    relinked: Dict[int, Dict],
    fwd_id_mapping_by_frame: Dict[int, List[int]],
    path: Path,
    *,
    num_individuals: int,
) -> None:
    """
    Write a post-link best.json that mirrors `_save_best_tracked_json`
    so the existing visualisers (rerun_viewer etc.) can consume it.

    The cam{0,1,2}_to_id mapping is derived from:
      - source_index_for_id[i]: the source label index in the *fwd*
        result that the relinked output's ID_(i+1) is sourced from.
      - fwd's id_mapping (passed as fwd_id_mapping_by_frame): which
        source label index fwd assigned to its own ID_i.
      - the perm_cam1/2 from the same frame (which detection slot each
        source label corresponds to in cam1/2).
    """
    id_labels = _id_labels(num_individuals)
    out: Dict[int, Dict] = {}
    for frame, info in relinked.items():
        src_idx_for_id = info.get("source_index_for_id") or list(
            range(num_individuals)
        )
        # fwd's id_mapping[i] = src lbl idx for fwd-ID_(i+1). Compose with
        # src_idx_for_id to figure out which detection slot in cam0 each
        # output ID corresponds to.
        # cam0 has no permutation (det_idx == src_lbl_idx), so:
        cam0_to_id: List[str | None] = [None] * num_individuals
        for out_i in range(num_individuals):
            src_idx = src_idx_for_id[out_i]
            if 0 <= src_idx < num_individuals:
                cam0_to_id[src_idx] = id_labels[out_i]

        perm_cam1 = info.get("perm_cam1") or list(range(num_individuals))
        perm_cam2 = info.get("perm_cam2") or list(range(num_individuals))
        cam1_to_id: List[str | None] = [None] * num_individuals
        cam2_to_id: List[str | None] = [None] * num_individuals
        for out_i in range(num_individuals):
            src_idx = src_idx_for_id[out_i]
            if 0 <= src_idx < num_individuals:
                if src_idx < len(perm_cam1):
                    det_idx = perm_cam1[src_idx]
                    if 0 <= det_idx < num_individuals:
                        cam1_to_id[det_idx] = id_labels[out_i]
                if src_idx < len(perm_cam2):
                    det_idx = perm_cam2[src_idx]
                    if 0 <= det_idx < num_individuals:
                        cam2_to_id[det_idx] = id_labels[out_i]

        out[int(frame)] = {
            "cam0_to_id": cam0_to_id,
            "cam1_to_id": cam1_to_id,
            "cam2_to_id": cam2_to_id,
            "groups": info.get("groups", {}),
        }

    payload = {
        "num_individuals": num_individuals,
        "id_labels": id_labels,
        "frames": out,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Comparison metric: large-jump rate per minute.
# ---------------------------------------------------------------------------

def jump_summary(
    csv_path: Path, *, jump_thresh_mm: float = 15.0, fps: float = 166.0
) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    out: Dict[str, float] = {}
    for fid, g in df.groupby("fish_id"):
        g = g.sort_values("frame").dropna(subset=["X"])
        if len(g) < 2:
            continue
        pos = g[["X", "Y", "Z"]].to_numpy()
        fr = g["frame"].to_numpy()
        dt = np.diff(fr)
        d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        # Adjacent-only jumps.
        d = d[dt == 1]
        n_large = int((d > jump_thresh_mm).sum())
        minutes = (g["frame"].max() - g["frame"].min() + 1) / fps / 60.0
        out[str(fid)] = n_large / max(minutes, 1e-9)
    return out


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def _derive_tag(pre_link_path: Path) -> str:
    stem = pre_link_path.stem
    if stem.endswith("_best_pre_link"):
        return stem[: -len("_best_pre_link")]
    if stem.endswith("_pre_link"):
        return stem[: -len("_pre_link")]
    return stem


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Forward+backward re-linker. Reads a pre-link JSON, runs the "
            "linker forward and backward, reconciles disagreements via "
            "trajectory smoothness, writes a relinked 3D CSV + best.json."
        )
    )
    parser.add_argument("pre_link_json", help="Path to <tag>_best_pre_link.json")
    parser.add_argument("--output-dir", default=None,
                        help="Default: directory of the pre-link JSON.")
    parser.add_argument("--output-tag", default=None,
                        help="Stem for output files; default derived from input.")
    parser.add_argument("--num-individuals", type=int, default=3)
    # Forward linker options (forwarded to apply_cross_frame_tracking).
    parser.add_argument("--max-jump-mm", type=float, default=30.0)
    parser.add_argument("--max-jump-hard-cap-mm", type=float, default=80.0)
    parser.add_argument("--max-gap-frames", type=int, default=100)
    parser.add_argument(
        "--sigma-tolerance-mm", type=float, default=5.0,
        help="Skip sigma-change check if alignment cost exceeds this.",
    )
    parser.add_argument(
        "--forward-only", action="store_true",
        help=(
            "Skip the backward pass and reconciliation; just re-run the "
            "forward linker on the pre-link candidates and write the result. "
            "Use this to quickly test linker-parameter changes against the "
            "same pre-link without re-triangulating."
        ),
    )
    parser.add_argument(
        "--no-fill-missing", action="store_true",
        help="Do not interpolate NaN gaps in the output 3D CSV.",
    )
    parser.add_argument(
        "--shift-to-positive", action="store_true",
        help="Shift coordinates so the minimum sits at origin (matches old default).",
    )
    args = parser.parse_args(argv)

    in_path = Path(args.pre_link_json).resolve()
    if not in_path.is_file():
        parser.error(f"Pre-link JSON not found: {in_path}")
    out_dir = Path(args.output_dir).resolve() if args.output_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.output_tag or _derive_tag(in_path)

    print(f"Loading pre-link  : {in_path}")
    best = load_best_pre_link_json(in_path)
    print(f"Frames            : {len(best)}")
    print(f"Linker options    : max_jump_mm={args.max_jump_mm}, "
          f"hard_cap={args.max_jump_hard_cap_mm}, gap={args.max_gap_frames}")

    if args.forward_only:
        fwd = apply_cross_frame_tracking(
            best,
            num_individuals=args.num_individuals,
            max_jump_mm=args.max_jump_mm,
            max_jump_hard_cap_mm=args.max_jump_hard_cap_mm,
            max_gap_frames=args.max_gap_frames,
        )
        # Decorate with empty source_index_for_id so the post-link writer
        # produces a usable best.json.
        relinked = {
            t: {
                "groups": info.get("groups", {}),
                "perm_cam1": list(info.get("perm_cam1", [])),
                "perm_cam2": list(info.get("perm_cam2", [])),
                "source_index_for_id": list(
                    info.get("id_mapping", list(range(args.num_individuals)))
                ),
            }
            for t, info in fwd.items()
        }
        decisions = []
        print("forward-only mode: skipping backward pass + reconciliation")
    else:
        relinked, decisions = relink_forward_backward(
            best,
            num_individuals=args.num_individuals,
            sigma_tolerance_mm=args.sigma_tolerance_mm,
            max_jump_mm=args.max_jump_mm,
            max_jump_hard_cap_mm=args.max_jump_hard_cap_mm,
            max_gap_frames=args.max_gap_frames,
        )
        n_fwd = sum(1 for d in decisions if d["side_swapped"] == "fwd")
        n_bwd = sum(1 for d in decisions if d["side_swapped"] == "bwd")
        print(f"sigma-change events: {len(decisions)}  (fwd-swap: {n_fwd}, "
              f"bwd-swap: {n_bwd})")

    # 3D CSV via the standard helper so column layout matches existing CSVs.
    dfs_3d = best_by_frame_to_dfs(
        relinked,
        shift_to_positive=args.shift_to_positive,
        ensure_non_negative=False,
        num_individuals=args.num_individuals,
    )
    if not args.no_fill_missing:
        all_frames = sorted(relinked.keys())
        dfs_3d = ensure_all_frames_and_fill_spline(dfs_3d, all_frames)
    csv_path = dfs_to_single_csv(
        dfs_3d, set_key=f"{tag}_relinked", output_dir=str(out_dir)
    )
    print(f"3D CSV            : {csv_path}")

    # Post-link JSON in the same shape as <tag>_best.json.
    relinked_best_json = out_dir / f"{tag}_best_relinked.json"
    _save_relinked_best_json(
        relinked, {}, relinked_best_json, num_individuals=args.num_individuals
    )
    print(f"best (relinked)   : {relinked_best_json}")

    # Decisions CSV.
    if decisions:
        dec_df = pd.DataFrame(decisions)
        dec_path = out_dir / f"{tag}_relink_decisions.csv"
        dec_df.to_csv(dec_path, index=False)
        print(f"decisions CSV     : {dec_path}")

    # Quick metric: large-jump rate per minute, before vs after.
    print("\nLarge-jump rate (jumps >15 mm/frame, per minute):")
    after = jump_summary(Path(csv_path), jump_thresh_mm=15.0)
    print(f"  After re-link  : {after}")
    orig_csv = out_dir / f"{tag}_3d.csv"
    if orig_csv.is_file():
        before = jump_summary(orig_csv, jump_thresh_mm=15.0)
        print(f"  Before (orig)  : {before}")


if __name__ == "__main__":
    main()
