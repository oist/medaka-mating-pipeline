from __future__ import annotations

"""
Add a `mid_body` keypoint to a labels.pkg.slp file by bootstrapping from
existing mouth + tail labels.

Bootstrap formula: mid_body = 0.75*mouth + 0.25*tail
(~25% along the body axis from the snout; pectoral fin base in medaka).

For user instances missing either mouth or tail (NaN), mid_body is left NaN
and skipped — the annotator should fill these in manually.

Also writes a "hard cases" CSV ranking labeled frames by minimum mouth-mouth
distance between fish, so the annotator knows where to focus refinement.

Backups the original to `<input>.pkg.slp.bak2nodes` and overwrites the input
with the 3-node version, so downstream tools (prepare_retraining, SLEAP GUI,
sleap-nn train) see the updated skeleton without path changes.

Usage:
    python -m scripts.sleap.bootstrap_midbody \\
        Z:/StephensU/medaka/sleap_models/retrain_20260508/top/labels.pkg.slp \\
        Z:/StephensU/medaka/sleap_models/retrain_20260508/side/labels.pkg.slp \\
        --top-k-hard 30
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import sleap_io as sio


MID_BODY_FRAC = 0.25  # along body axis, from mouth


def _bootstrap_one(slp_path: Path, top_k_hard: int = 30) -> dict:
    """Rename the input .pkg.slp to <name>.bak2nodes, then write a thin new
    .pkg.slp with the 3-node skeleton that references the backup for video
    data. The backup preserves the embedded video frames; the new file is
    ~90 KB and contains the updated skeleton + bootstrapped mid_body."""
    print(f"\n=== {slp_path} ===")

    # Use a backup name with a real `.pkg.slp` extension. sleap-io's save_slp
    # normalizes/rewrites video.filename references that DON'T end in .pkg.slp
    # (e.g. .bak2nodes), so the saved thin file would reference a non-existent
    # path. Naming the backup `labels_orig.pkg.slp` keeps the reference stable.
    bak = slp_path.with_name(slp_path.stem.rsplit(".", 1)[0] + "_orig.pkg.slp")

    # If a previous attempt already produced a 3-node labels.pkg.slp, skip.
    quick_labels = sio.load_slp(str(slp_path), open_videos=False)
    quick_nodes = [n.name for n in quick_labels.skeletons[0].nodes]
    if "mid_body" in quick_nodes:
        print(f"  mid_body already present: nodes={quick_nodes}; skipping bootstrap")
        return {"status": "skip_already_3nodes"}
    if "mouth" not in quick_nodes or "tail" not in quick_nodes:
        raise RuntimeError(f"expected mouth+tail nodes, got {quick_nodes}")

    if bak.exists():
        print(f"  backup already exists at {bak.name}; refusing to overwrite")
        raise RuntimeError(
            f"{bak} exists. Either delete it (after confirming you don't need "
            f"the old 2-node labels) or rename it manually before re-running."
        )
    # Step 1: rename original -> labels_orig.pkg.slp (atomic).
    print(f"  rename {slp_path.name} -> {bak.name}")
    slp_path.rename(bak)

    # Step 2: load from backup (videos now reference the backup path).
    labels = sio.load_slp(str(bak))
    sk = labels.skeletons[0]

    sk.add_node("mid_body")
    sk.add_edge("mouth", "mid_body")
    sk.add_edge("mid_body", "tail")
    # Drop the legacy mouth-tail edge so the skeleton is a clean chain
    # mouth -> mid_body -> tail (anatomical body axis, no triangle).
    sk.edges = [e for e in sk.edges
                if not (e.source.name == "mouth" and e.destination.name == "tail")]
    sk.reorder_nodes(["mouth", "mid_body", "tail"])
    print(f"  skeleton nodes -> {[n.name for n in sk.nodes]}")
    print(f"  skeleton edges -> {[(e.source.name, e.destination.name) for e in sk.edges]}")

    mi = sk.index("mouth")
    bi = sk.index("mid_body")
    ti = sk.index("tail")

    n_user, n_filled, n_skipped_nan = 0, 0, 0
    for lf in labels.labeled_frames:
        for inst in lf.instances:
            inst.update_skeleton()  # grows points array to 3 rows
            if type(inst).__name__ != "Instance":
                continue
            n_user += 1
            pts = inst.numpy()  # shape (3, 2)
            m, t = pts[mi], pts[ti]
            if not (np.isfinite(m).all() and np.isfinite(t).all()):
                n_skipped_nan += 1
                continue
            mb = m + MID_BODY_FRAC * (t - m)
            pts[bi] = mb
            inst2 = sio.Instance.from_numpy(
                pts, skeleton=sk, track=inst.track,
                tracking_score=getattr(inst, "tracking_score", None),
            )
            idx = lf.instances.index(inst)
            lf.instances[idx] = inst2
            n_filled += 1

    print(f"  user instances : {n_user}")
    print(f"  mid_body filled: {n_filled}")
    if n_skipped_nan:
        print(f"  skipped (NaN mouth/tail): {n_skipped_nan}")

    # Step 3: save thin .pkg.slp at the original path. The new file is small
    # (~90 KB) and references `bak` for video data. SLEAP GUI and sleap-nn
    # both follow the reference transparently. We deliberately avoid embedding
    # frames because round-tripping a self-referential .pkg.slp through
    # embed="user" raises FileNotFoundError on the embedding pass.
    sio.save_slp(
        labels, str(slp_path),
        embed=False, restore_original_videos=False,
    )
    sz = slp_path.stat().st_size
    print(f"  wrote 3-node {slp_path.name}  ({sz/1024:.1f} KB, references {bak.name})")

    items = []
    for lf in labels.labeled_frames:
        mouths = []
        for inst in lf.instances:
            if type(inst).__name__ != "Instance":
                continue
            pts = inst.numpy()
            if np.isfinite(pts[mi]).all():
                mouths.append(pts[mi])
        if len(mouths) < 2:
            continue
        a = np.array(mouths)
        d = np.sqrt(((a[:, None, :] - a[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        items.append((float(d.min()), int(lf.frame_idx),
                      Path(lf.video.filename).name if lf.video.filename else "?"))
    items.sort(key=lambda x: x[0])

    hard_csv = slp_path.parent / "midbody_hardcases.csv"
    with hard_csv.open("w", encoding="utf-8") as f:
        f.write("rank,min_mouth_dist_px,frame_idx,video\n")
        for i, (d, fi, v) in enumerate(items[:top_k_hard]):
            f.write(f"{i+1},{d:.1f},{fi},{v}\n")
    print(f"  hard cases ({top_k_hard} closest encounters) -> {hard_csv}")

    return {
        "status": "ok",
        "n_user_instances": n_user,
        "n_filled": n_filled,
        "n_skipped_nan": n_skipped_nan,
        "hard_csv": str(hard_csv),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("slp", nargs="+", type=Path,
                   help="One or more labels.pkg.slp files to update in place.")
    p.add_argument("--top-k-hard", type=int, default=30,
                   help="Number of hardest close-encounter frames to list per view.")
    args = p.parse_args(argv)

    for slp in args.slp:
        if not slp.exists():
            print(f"[error] missing: {slp}", file=sys.stderr)
            return 2
        _bootstrap_one(slp, top_k_hard=args.top_k_hard)
    print("\nDone. Next steps:")
    print("  1. Open each labels.pkg.slp in SLEAP GUI (`sleap-label <path>`).")
    print("  2. Navigate to the frame_idx values in midbody_hardcases.csv for each view.")
    print("  3. Drag mid_body keypoint to the pectoral-fin base (top: midline x pectoral line; side: visible pectoral fin base).")
    print("  4. Save (Ctrl+S). Then retrain via run_round3_training --kinds centered_instance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
