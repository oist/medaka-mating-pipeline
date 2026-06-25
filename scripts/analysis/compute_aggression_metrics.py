"""Male-male aggression metrics from the dual-line 3-fish individual-ID track.

Was a ``raise SystemExit("TODO")`` placeholder; now a thin CLI over the shared, tested
``mate_choice_core``. Reads the interp 3-fish track and writes the male-male
close+fast (aggressive-approach) bout table + initiator summary — answerable only
because the two males are now distinguished (M1/M2).

Usage:
    .venv\\Scripts\\python.exe -m scripts.analysis.compute_aggression_metrics \\
        --tracks runs/dual_line/lineB_fulltimeline/step2/round5_260611_3d_individual_id_interp.csv \\
        --output runs/dual_line/lineB_fulltimeline/analysis/aggression_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis import mate_choice_core as mc


def compute(tracks: str, params: mc.BehaviourParams) -> tuple[dict, pd.DataFrame]:
    pos, _n = mc.load_individual_track(tracks)
    d = mc.compute_distances(pos, params.fps)
    bouts = mc.aggression_bouts(d, params)
    result = {
        "track": str(tracks),
        "params": params.as_dict(),
        "aggression_summary": mc.aggression_summary(bouts),
        "breakdown_vs_courtship": mc.behaviour_breakdown(d, params),
    }
    return result, pd.DataFrame(bouts)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Compute male-male aggression metrics.")
    ap.add_argument("--tracks", required=True, help="3-fish individual-ID track CSV (id_final/F,M1,M2)")
    ap.add_argument("--output", required=True, help="output JSON path")
    ap.add_argument("--agg-mm", type=float, default=mc.BehaviourParams.agg_mm)
    ap.add_argument("--agg-speed", type=float, default=mc.BehaviourParams.agg_speed)
    ap.add_argument("--min-bout", type=int, default=mc.BehaviourParams.min_bout)
    ap.add_argument("--fps", type=float, default=mc.FPS)
    args = ap.parse_args(argv)

    params = mc.BehaviourParams(fps=args.fps, agg_mm=args.agg_mm, agg_speed=args.agg_speed,
                                min_bout=args.min_bout)
    result, bouts = compute(args.tracks, params)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    bouts.to_csv(out.with_name(out.stem + "_bouts.csv"), index=False)
    print(json.dumps(result, indent=2))
    print(f"\nwrote {out} and {out.with_name(out.stem + '_bouts.csv')} ({len(bouts)} bouts)")


if __name__ == "__main__":
    main()
