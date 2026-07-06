"""Courtship / female-association metrics from the dual-line 3-fish individual-ID track.

Was a ``raise SystemExit("TODO")`` placeholder; now a thin CLI over the shared, tested
``mate_choice_core`` (the same core that backs ``analyze_mate_choice.py``). Reads the
interp 3-fish track (cols ``frame``, ``id_final`` ∈ {F,M1,M2}, ``X,Y,Z``) and writes the
per-male courtship/association metrics + the temporal-preference trajectory.

Usage:
    .venv\\Scripts\\python.exe -m scripts.analysis.compute_courtship_metrics \\
        --tracks runs/dual_line/lineB_fulltimeline/step2/round5_260611_3d_individual_id_interp.csv \\
        --output runs/dual_line/lineB_fulltimeline/analysis/courtship_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis import mate_choice_core as mc


def compute(tracks: str, params: mc.BehaviourParams, spawn_frame: int | None) -> tuple[dict, pd.DataFrame]:
    pos, _n = mc.load_individual_track(tracks)
    d = mc.compute_distances(pos, params.fps)
    table, trend = mc.temporal_preference(d, params)
    result = {
        "track": str(tracks),
        "params": params.as_dict(),
        "association": mc.association_metrics(d, params),
        "courtship_bouts": {m: {k: v for k, v in mc.courtship_bouts(d, params)[m].items() if k != "bouts"}
                            for m in ("M1", "M2")},
        "temporal_preference": trend,
        "spawn": mc.spawn_attribution(d, params, spawn_frame),
    }
    return result, table


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Compute per-male courtship / association metrics.")
    ap.add_argument("--tracks", required=True, help="3-fish individual-ID track CSV (id_final/F,M1,M2)")
    ap.add_argument("--output", required=True, help="output JSON path")
    ap.add_argument("--spawn-frame", type=int, default=None)
    ap.add_argument("--court-mm", type=float, default=mc.BehaviourParams.court_mm)
    ap.add_argument("--contact-mm", type=float, default=mc.BehaviourParams.contact_mm)
    ap.add_argument("--min-bout", type=int, default=mc.BehaviourParams.min_bout)
    ap.add_argument("--n-bins", type=int, default=mc.BehaviourParams.n_bins)
    ap.add_argument("--fps", type=float, default=mc.FPS)
    args = ap.parse_args(argv)

    params = mc.BehaviourParams(fps=args.fps, court_mm=args.court_mm, contact_mm=args.contact_mm,
                                min_bout=args.min_bout, n_bins=args.n_bins)
    result, table = compute(args.tracks, params, args.spawn_frame)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    table.to_csv(out.with_name(out.stem + "_temporal.csv"), index=False)
    print(json.dumps(result, indent=2))
    print(f"\nwrote {out} and {out.with_name(out.stem + '_temporal.csv')}")


if __name__ == "__main__":
    main()
