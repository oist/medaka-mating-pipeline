from __future__ import annotations

"""
Mark the closest-encounter frames in a labels.pkg.slp as SLEAP
SuggestionFrames so they're navigable in the SLEAP GUI Suggestions panel
(Ctrl+Up / Ctrl+Down to jump between them; the panel also lists all).

Ranks all labeled frames by minimum mouth-mouth distance across user
instances; takes the top-K closest as suggestions.

Updates the .pkg.slp in place using the same thin-file save path that
bootstrap_midbody.py uses (no re-embedding of frames; video refs
preserved).

Usage:
    python -m scripts.sleap.mark_hardcase_suggestions \\
        Z:/StephensU/medaka/sleap_models/retrain_20260508/top/labels.pkg.slp \\
        Z:/StephensU/medaka/sleap_models/retrain_20260508/side/labels.pkg.slp \\
        --top-k 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import sleap_io as sio


def _hard_cases(labels: sio.Labels, top_k: int) -> list[tuple[float, sio.LabeledFrame]]:
    """Per-frame minimum mouth-mouth distance, ranked ascending."""
    sk = labels.skeletons[0]
    try:
        mi = sk.index("mouth")
    except ValueError:
        raise RuntimeError("skeleton has no 'mouth' node")
    items: list[tuple[float, sio.LabeledFrame]] = []
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
        items.append((float(d.min()), lf))
    items.sort(key=lambda x: x[0])
    return items[:top_k]


def _add_suggestions(slp_path: Path, top_k: int) -> dict:
    print(f"\n=== {slp_path} ===")
    labels = sio.load_slp(str(slp_path))
    existing_keys = {(id(s.video), int(s.frame_idx)) for s in labels.suggestions}
    print(f"  existing suggestions: {len(existing_keys)}")

    hard = _hard_cases(labels, top_k)
    print(f"  found {len(hard)} close-encounter frames (top-{top_k})")
    if hard:
        print(f"  min mouth-mouth distance range: {hard[0][0]:.1f} px (closest) "
              f"to {hard[-1][0]:.1f} px")

    added = 0
    skipped_dup = 0
    for min_d, lf in hard:
        key = (id(lf.video), int(lf.frame_idx))
        if key in existing_keys:
            skipped_dup += 1
            continue
        labels.suggestions.append(
            sio.SuggestionFrame(video=lf.video, frame_idx=int(lf.frame_idx))
        )
        existing_keys.add(key)
        added += 1
    print(f"  added {added} new suggestions  (skipped {skipped_dup} already-present duplicates)")
    print(f"  total suggestions now: {len(labels.suggestions)}")

    # Same thin-save flow as bootstrap_midbody — no re-embedding.
    sio.save_slp(
        labels, str(slp_path),
        embed=False, restore_original_videos=False,
    )
    sz = slp_path.stat().st_size
    print(f"  wrote {slp_path.name}  ({sz/1024:.1f} KB)")
    return {"added": added, "skipped_dup": skipped_dup, "total": len(labels.suggestions)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("slp", nargs="+", type=Path,
                   help="One or more labels.pkg.slp files to update in place.")
    p.add_argument("--top-k", type=int, default=30,
                   help="Number of hardest close-encounter frames to mark as suggestions.")
    args = p.parse_args(argv)

    for slp in args.slp:
        if not slp.exists():
            print(f"[error] missing: {slp}", file=sys.stderr)
            return 2
        _add_suggestions(slp, args.top_k)
    print("\nDone. Open in SLEAP GUI: the Suggestions panel will list these "
          "frames; Ctrl+Up / Ctrl+Down jumps between them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
