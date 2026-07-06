"""Motion-continuity debounce: remove physically-impossible brief id teleport-swaps from a resolved
3-fish interp track (the confident <50-frame flickers where two ids exchange onto the wrong geometry
slots at a near-crossing and snap back). Reusable post-process over the shared, unit-tested
``per_frame_resolve.debounce_track_teleports``. Pure CPU; no inference.

Run::
    .venv\\Scripts\\python.exe runs/dual_line/debounce_track.py \\
        --in-csv  .../round5_260611_3d_individual_id_interp.csv \\
        --out-csv .../round5_260611_3d_individual_id_interp_debounced.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from sam2_id_experiment.src.id_model import per_frame_resolve as pfr  # noqa: E402


def count_teleports(track, n, jump_mm):
    """# of frames where some fish's id_final position jumps > jump_mm (and the moments)."""
    tot = 0
    for name in ("F", "M1", "M2"):
        s = track[track.id_final == name].set_index("frame")
        p = np.full((n, 3), np.nan)
        p[s.index.to_numpy()] = s[["X", "Y", "Z"]].to_numpy()
        d = np.linalg.norm(np.diff(p, axis=0), axis=1)
        tot += int(np.nansum(d > jump_mm))
    return tot


def main(argv=None):
    ap = argparse.ArgumentParser(description="Debounce teleport-swap flickers from a 3-fish track.")
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--jump-mm", type=float, default=60.0)
    ap.add_argument("--max-flicker", type=int, default=166)
    args = ap.parse_args(argv)

    track = pd.read_csv(args.in_csv)
    n = int(track.frame.max()) + 1
    before = count_teleports(track, n, args.jump_mm)

    new_id, reverted = pfr.debounce_track_teleports(track, n, jump_mm=args.jump_mm, max_flicker=args.max_flicker)
    track["id_final_raw"] = track["id_final"]
    track["id_final"] = new_id
    # provenance: mark every (frame, slot) inside a reverted span
    fr = track.frame.to_numpy().astype(int)
    fixed = np.zeros(n, bool)
    for s, e in reverted:
        fixed[s:e + 1] = True
    track.loc[fixed[fr], "id_source"] = "motion_debounced"

    after = count_teleports(track, n, args.jump_mm)
    changed = int((track.id_final != track.id_final_raw).sum())
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    track.to_csv(args.out_csv, index=False)

    counts = track.groupby("frame").id_final.value_counts().unstack(fill_value=0)
    one_each = int(((counts.get("F", 0) == 1) & (counts.get("M1", 0) == 1) & (counts.get("M2", 0) == 1)).sum())
    print(f"teleport-frames (>{args.jump_mm:.0f}mm/frame):  before {before}  ->  after {after}  "
          f"({100*(before-after)/max(before,1):.0f}% removed)")
    print(f"reverted {len(reverted)} flicker spans; {changed} fish-frame labels changed")
    print(f"still exactly 1F+1M1+1M2: {one_each}/{track.frame.nunique()} frames")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
