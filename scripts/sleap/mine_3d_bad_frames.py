from __future__ import annotations

"""
Round-3 active-learning frame miner: identify *bad* frames where the round-2
3D reconstruction failed, then merge those frames with the existing
mouth+tail annotation seed into a single ``pkg.slp`` for re-annotation.

Failure modes targeted
----------------------
1. **Spike** — a fish's 3D position jumps abruptly between frames (likely
   ID swap or a single-frame mis-detection that triangulates to a wild
   location). Detected as ``||xyz_t - xyz_{t-1}|| > spike_mm``.
2. **Close-fish + spike** — two fish are within ``close_mm`` of each other
   AND a spike happens. Strongly indicates ID swap due to bad/partial
   detection across two adjacent fish.

Frames matching either category are scored, diversified across time, and
collapsed to one (frame, top, side1, side2) annotation tuple per pick — so
the user can refine all three views at the same moment together.

Output
------
A new ``pkg.slp`` written to ``--out``, containing:

- All ``LabeledFrame``\\s from the existing user-corrected seed
  (``--existing-seed``) verbatim (preserves embedded frames and user
  annotations).
- New labelled frames at each picked (view, frame) pair, populated with the
  round-2 SLEAP prediction's mouth+tail keypoints (the annotator only needs
  to verify / nudge them).
- ``SuggestionFrame`` entries spanning both sets so SLEAP-label's
  navigation walks every frame in order.

CLI
---
    python -m scripts.sleap.mine_3d_bad_frames \\
        --existing-seed "C:/Users/Makoto Hiroi/Downloads/annotation_seed.pkg.slp" \\
        --csv-3d "Z:/.../round2_n6_recovered/..._3d.csv" \\
        --slp-dir "Z:/.../sleap_output/<exp>/round2_trt_n6" \\
        --experiment 20251225_K_1_2_FCMC \\
        --out "C:/.../annotation_seed_round3.pkg.slp" \\
        --target-count 500

The ``--target-count`` is the total number of *new* annotation entries
(across views), so 500 ≈ 167 unique 3D frames × 3 views.
"""

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


VIEWS = ("top_cam", "side1_cam", "side2_cam")


# ---------------------------------------------------------------------------
# 3D failure-mode mining.
# ---------------------------------------------------------------------------

def compute_per_frame_signals(
    df_3d: pd.DataFrame,
    *,
    fps: float = 166.0,
) -> pd.DataFrame:
    """For each frame, compute max per-fish jump and min pairwise distance.

    Returns a frame-indexed DataFrame with columns:
        max_jump_mm, min_pair_mm, n_fish_finite, mean_jump_mm
    """
    df = df_3d.copy()
    df["frame"] = df["frame"].astype(int)
    fish_ids = sorted(df["fish_id"].dropna().unique().tolist())
    n_fish = len(fish_ids)

    frames = sorted(df["frame"].unique().tolist())
    frame_set = pd.Index(frames, name="frame")

    # Per-fish wide pivots indexed by frame.
    pivots: Dict[str, np.ndarray] = {}
    for fid in fish_ids:
        sub = df[df["fish_id"] == fid].set_index("frame").reindex(frame_set)
        pivots[fid] = sub[["X", "Y", "Z"]].to_numpy(dtype=float)

    # Frame-to-frame jump per fish (NaN if either endpoint is NaN).
    jumps: Dict[str, np.ndarray] = {}
    for fid in fish_ids:
        xyz = pivots[fid]
        prev = np.roll(xyz, 1, axis=0)
        prev[0] = np.nan
        d = np.linalg.norm(xyz - prev, axis=1)
        jumps[fid] = d

    # Pairwise distances between fish at each frame.
    pair_dists = np.full((len(frames), n_fish * (n_fish - 1) // 2), np.nan)
    pair_idx = 0
    for i in range(n_fish):
        for j in range(i + 1, n_fish):
            d = np.linalg.norm(pivots[fish_ids[i]] - pivots[fish_ids[j]], axis=1)
            pair_dists[:, pair_idx] = d
            pair_idx += 1

    jump_arr = np.stack([jumps[fid] for fid in fish_ids], axis=1)
    n_finite = sum(
        np.isfinite(pivots[fid]).all(axis=1).astype(int) for fid in fish_ids
    )

    out = pd.DataFrame({
        "frame": frames,
        "max_jump_mm": np.nanmax(jump_arr, axis=1),
        "mean_jump_mm": np.nanmean(jump_arr, axis=1),
        "min_pair_mm": np.nanmin(pair_dists, axis=1),
        "n_fish_finite": n_finite,
    })
    return out


def score_and_pick_bad_frames(
    signals: pd.DataFrame,
    *,
    spike_mm: float = 15.0,
    close_mm: float = 30.0,
    target_unique_frames: int = 167,
    min_frame_gap: int = 200,
    fps: float = 166.0,
    skip_seconds: float = 30.0,
    n_time_bins: int = 30,
) -> pd.DataFrame:
    """Score each frame by failure-mode strength, then pick a diversified subset.

    Categories (mutually inclusive, signals can fire together):
        - spike: max_jump_mm >= spike_mm
        - close+spike: spike AND min_pair_mm <= close_mm

    Score is ``max_jump_mm * (2 if close_and_spike else 1)`` so close-fish
    spikes win priority. Picks are spaced ≥ ``min_frame_gap`` frames apart and
    stratified into ``n_time_bins`` time bins so we don't all cluster in one
    behavioural window.
    """
    s = signals.copy()
    s["spike"] = s["max_jump_mm"].fillna(0) >= spike_mm
    s["close_fish"] = s["min_pair_mm"].fillna(np.inf) <= close_mm
    s["close_and_spike"] = s["spike"] & s["close_fish"]

    # Skip the first `skip_seconds` of the run (animals settling into the tank).
    skip_frames = int(round(fps * skip_seconds))
    s["category_score"] = np.where(
        s["close_and_spike"], 2.0, np.where(s["spike"], 1.0, 0.0)
    )
    s["raw_score"] = s["max_jump_mm"].fillna(0) * s["category_score"]
    candidate_mask = s["spike"] & (s["frame"] >= skip_frames)
    candidates = s[candidate_mask].copy()

    if candidates.empty:
        return s.assign(picked=False)

    # Stratify by time bin.
    f_min, f_max = int(candidates["frame"].min()), int(candidates["frame"].max())
    n_bins = max(1, n_time_bins)
    candidates["time_bin"] = (
        ((candidates["frame"] - f_min) / max(1, f_max - f_min) * n_bins)
        .astype(int)
        .clip(0, n_bins - 1)
    )

    # Within each bin, sort by descending score; round-robin across bins to
    # spread picks evenly. Enforce min_frame_gap between picks.
    candidates_by_bin: Dict[int, List[Tuple[int, float, bool]]] = defaultdict(list)
    for r in candidates.itertuples(index=False):
        candidates_by_bin[int(r.time_bin)].append(
            (int(r.frame), float(r.raw_score), bool(r.close_and_spike))
        )
    for b in candidates_by_bin:
        candidates_by_bin[b].sort(key=lambda x: -x[1])

    picked: List[int] = []
    picked_set: set[int] = set()
    bin_keys = sorted(candidates_by_bin.keys())
    bin_pos = {b: 0 for b in bin_keys}
    while len(picked) < target_unique_frames:
        progress = False
        for b in bin_keys:
            if len(picked) >= target_unique_frames:
                break
            pos = bin_pos[b]
            row_list = candidates_by_bin[b]
            while pos < len(row_list):
                f, score, _ = row_list[pos]
                pos += 1
                # Skip if too close to a previously picked frame.
                too_close = any(abs(f - q) < min_frame_gap for q in picked_set)
                if not too_close:
                    picked.append(f)
                    picked_set.add(f)
                    progress = True
                    break
            bin_pos[b] = pos
        if not progress:
            break

    out = s.copy()
    out["picked"] = out["frame"].isin(picked_set)
    return out


# ---------------------------------------------------------------------------
# Build the merged annotation seed.
# ---------------------------------------------------------------------------

def _read_predictions_at(slp_path: Path) -> Tuple["sio.Labels", Dict[int, "sio.LabeledFrame"]]:
    import sleap_io as sio
    labels = sio.load_slp(str(slp_path))
    by_frame = {int(lf.frame_idx): lf for lf in labels.labeled_frames}
    return labels, by_frame


def _make_predicted_to_user_instance(inst, target_skeleton):
    """Convert a predicted SLEAP instance into a fresh user instance with the
    same per-node coordinates on the target skeleton.

    Predictions don't show up as user labels in the SLEAP-label GUI; only
    Instance (not PredictedInstance) entries are presented for editing. We
    keep the predicted positions as starting points for the annotator.
    """
    import sleap_io as sio
    pts = inst.numpy()
    return sio.Instance.from_numpy(points_data=np.asarray(pts, dtype=float),
                                   skeleton=target_skeleton)


def build_combined_seed(
    *,
    existing_seed: Path,
    slp_dir: Path,
    picked_frames: List[int],
    out_path: Path,
    embed_new: bool = False,
) -> dict:
    """Concatenate the existing user-labeled seed with new bad-frame entries.

    Parameters
    ----------
    existing_seed
        Path to the round-2 ``annotation_seed.pkg.slp`` (user-corrected).
    slp_dir
        Directory holding the round-2 inference outputs:
        ``top_cam.slp``, ``side1_cam.slp``, ``side2_cam.slp``.
    picked_frames
        Sorted list of unique 3D frame indices identified as failure-mode
        candidates. The output adds 3 ``LabeledFrame``\\s per pick (one per
        view, all at the same frame index).
    out_path
        Destination ``.pkg.slp``.
    embed_new
        If True, embed the picked-video frames so the file is self-contained
        without network access. Doubles disk size; off by default since the
        existing seed already embeds 500 frames.
    """
    try:
        import sleap_io as sio
    except ImportError as exc:
        raise RuntimeError(
            "sleap-io is required. `uv sync --extra sleap`."
        ) from exc

    existing = sio.load_slp(str(existing_seed))
    skel = existing.skeletons[0]
    print(f"Existing seed: {len(existing.labeled_frames)} LFs, "
          f"{len(existing.videos)} videos, skeleton={[n.name for n in skel.nodes]}")

    # Reuse existing video objects so the merged file references one Video per
    # path. Map by stored filename to handle duplicate paths gracefully.
    video_by_name: Dict[str, "sio.Video"] = {}
    for v in existing.videos:
        video_by_name[str(v.filename)] = v

    # Open the round-2 inference SLPs (one per view).
    slp_by_view: Dict[str, Tuple["sio.Labels", Dict[int, "sio.LabeledFrame"]]] = {}
    for view in VIEWS:
        p = slp_dir / f"{view}.slp"
        if not p.exists():
            raise FileNotFoundError(f"missing round-2 inference: {p}")
        slp_by_view[view] = _read_predictions_at(p)

    # Build the new bad-frame LabeledFrames.
    new_lfs: List["sio.LabeledFrame"] = []
    skipped = 0
    for f_idx in picked_frames:
        for view in VIEWS:
            src_labels, src_by_frame = slp_by_view[view]
            src_lf = src_by_frame.get(int(f_idx))
            if src_lf is None or not src_lf.instances:
                skipped += 1
                continue
            # Resolve the video object: reuse from existing seed when possible.
            src_video = src_lf.video if hasattr(src_lf, "video") else src_labels.videos[0]
            stored = str(src_video.filename)
            if stored in video_by_name:
                target_video = video_by_name[stored]
            else:
                target_video = src_video
                video_by_name[stored] = target_video

            new_instances = [
                _make_predicted_to_user_instance(inst, skel)
                for inst in src_lf.instances
                if np.isfinite(inst.numpy()).all()
            ]
            if not new_instances:
                skipped += 1
                continue
            new_lfs.append(sio.LabeledFrame(
                video=target_video,
                frame_idx=int(f_idx),
                instances=new_instances,
            ))

    print(f"New bad-frame LFs: {len(new_lfs)} (skipped {skipped})")

    merged_lfs = list(existing.labeled_frames) + new_lfs
    merged_videos = list({id(v): v for v in video_by_name.values()}.values())
    suggestions = [
        sio.SuggestionFrame(video=lf.video, frame_idx=int(lf.frame_idx))
        for lf in merged_lfs
    ]
    merged = sio.Labels(
        skeletons=[skel],
        videos=merged_videos,
        labeled_frames=merged_lfs,
        suggestions=suggestions,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    embed_choice = "user" if embed_new else "user"  # user == only embed user-labelled
    print(f"Saving {len(merged_lfs)} total LFs to {out_path} (embed={embed_choice!r})")
    sio.save_slp(merged, str(out_path), embed=embed_choice)

    return {
        "out_path": str(out_path),
        "n_existing": len(existing.labeled_frames),
        "n_new": len(new_lfs),
        "n_picked_frames": len(picked_frames),
        "n_videos": len(merged_videos),
        "skipped_views": skipped,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--existing-seed", type=Path, required=True,
                    help="round-2 annotation_seed.pkg.slp (user-corrected, 500 frames).")
    ap.add_argument("--csv-3d", type=Path, required=True,
                    help="Round-2 reconstruction *_3d.csv (per-frame XYZ per fish).")
    ap.add_argument("--slp-dir", type=Path, required=True,
                    help="Dir with round-2 inference top_cam.slp / side1_cam.slp / side2_cam.slp.")
    ap.add_argument("--out", type=Path, required=True, help="Output pkg.slp path.")
    ap.add_argument("--target-count", type=int, default=500,
                    help="Total *new* annotation entries across views (default 500 ~= 167 frames * 3 views).")
    ap.add_argument("--spike-mm", type=float, default=15.0,
                    help="Frame-to-frame jump magnitude (mm) to qualify as a 3D spike.")
    ap.add_argument("--close-mm", type=float, default=30.0,
                    help="Pairwise inter-fish distance (mm) qualifying as 'close'.")
    ap.add_argument("--min-frame-gap", type=int, default=200,
                    help="Don't pick two frames within this many frames of each other.")
    ap.add_argument("--n-time-bins", type=int, default=30,
                    help="Stratify picks across this many equal-width time bins.")
    ap.add_argument("--fps", type=float, default=166.0)
    ap.add_argument("--skip-seconds", type=float, default=30.0,
                    help="Skip the first N seconds of the run when picking.")
    ap.add_argument("--mining-report", type=Path, default=None,
                    help="Optional path to write a CSV of the per-frame signals.")
    args = ap.parse_args(argv)

    print(f"Loading 3D CSV: {args.csv_3d}")
    df_3d = pd.read_csv(args.csv_3d)
    print(f"  rows={len(df_3d)}, fish_ids={sorted(df_3d['fish_id'].dropna().unique())}")

    print("Computing per-frame failure-mode signals...")
    sigs = compute_per_frame_signals(df_3d, fps=args.fps)
    print(f"  spikes (>= {args.spike_mm} mm): {(sigs['max_jump_mm'].fillna(0) >= args.spike_mm).sum()}")
    print(f"  close fish (<= {args.close_mm} mm): {(sigs['min_pair_mm'].fillna(np.inf) <= args.close_mm).sum()}")
    spike_close = ((sigs['max_jump_mm'].fillna(0) >= args.spike_mm) &
                   (sigs['min_pair_mm'].fillna(np.inf) <= args.close_mm)).sum()
    print(f"  spike + close-fish: {spike_close}")

    target_unique = max(1, math.ceil(args.target_count / len(VIEWS)))
    print(f"Picking up to {target_unique} unique 3D frames "
          f"({target_unique} * {len(VIEWS)} views ~= {target_unique * len(VIEWS)} annotations)")
    picked = score_and_pick_bad_frames(
        sigs,
        spike_mm=args.spike_mm,
        close_mm=args.close_mm,
        target_unique_frames=target_unique,
        min_frame_gap=args.min_frame_gap,
        fps=args.fps,
        skip_seconds=args.skip_seconds,
        n_time_bins=args.n_time_bins,
    )
    picked_frames = sorted(picked.loc[picked["picked"], "frame"].astype(int).tolist())
    n_close_picks = picked.loc[picked["picked"], :]
    n_close_picks = (
        (n_close_picks["min_pair_mm"].fillna(np.inf) <= args.close_mm).sum()
    )
    print(f"  picked {len(picked_frames)} unique frames, "
          f"of which {n_close_picks} are close-fish + spike")

    if args.mining_report is not None:
        args.mining_report.parent.mkdir(parents=True, exist_ok=True)
        picked.to_csv(args.mining_report, index=False)
        print(f"  mining report -> {args.mining_report}")

    summary = build_combined_seed(
        existing_seed=args.existing_seed,
        slp_dir=args.slp_dir,
        picked_frames=picked_frames,
        out_path=args.out,
    )
    print(summary)


if __name__ == "__main__":
    main()
