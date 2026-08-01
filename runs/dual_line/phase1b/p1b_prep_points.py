"""Phase-1b prep (run in main .venv).

Extract, for each probe event's frame range (+bracket), the per-view concrete-tracklet
points (frame, x, y, tracklet_id, global_id) from the whole-timeline tracklet-bound CSVs.
Writes a compact ``points_<view>.json`` ({frame: [[tracklet_id, x, y, global_id], ...]})
that the Detectron2 stage (p1b_detect.py, .venv-detectron2 -- no pandas) consumes to link
masks -> fish identity and label crops, exactly as the production ``33_build_segid_dataset``
linking does.

This is the QC/FP-rejection + identity labeling that the Phase-1a probe skipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TB_DIR = Path("Z:/StephensU/medaka/runs/udmt_validation/tracklet_bind_extend_smooth")
VIEWS = ["side1_cam", "side2_cam"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-json", default=str(HERE.parent / "phase1a" / "probe_events.json"),
                    help="reuse the exact Phase-1a 20-event set for an apples-to-apples re-probe")
    ap.add_argument("--bracket", type=int, default=100)
    ap.add_argument("--out-dir", default=str(HERE))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    events = json.loads(Path(args.events_json).read_text())
    spans = [(e["start"] - args.bracket, e["end"] + args.bracket) for e in events]
    lo = min(s for s, _ in spans)
    hi = max(e for _, e in spans)
    print(f"[prep] {len(events)} events; frame span {lo}..{hi}", flush=True)

    def in_any_span(frames: np.ndarray) -> np.ndarray:
        keep = np.zeros(len(frames), bool)
        for s, e in spans:
            keep |= (frames >= s) & (frames <= e)
        return keep

    summary = {}
    for view in VIEWS:
        csv = TB_DIR / f"{view}.tracklet_bound.csv"
        df = pd.read_csv(csv, usecols=["frame", "x", "y", "tracklet_id", "global_id"])
        df = df[(df.frame >= lo) & (df.frame <= hi)]
        df = df[in_any_span(df.frame.to_numpy())]
        pts: dict[str, list] = {}
        n_gid = 0
        for f, x, y, tid, gid in zip(df.frame, df.x, df.y, df.tracklet_id, df.global_id):
            g = int(gid) if pd.notna(gid) else -1
            n_gid += int(g >= 0)
            pts.setdefault(str(int(f)), []).append([int(tid), float(x), float(y), g])
        (out / f"points_{view}.json").write_text(json.dumps(pts))
        summary[view] = {"n_frames_with_points": len(pts), "n_points": int(len(df)),
                         "n_points_with_global_id": n_gid,
                         "n_tracklets": int(df.tracklet_id.nunique())}
        print(f"[prep] {view}: {len(pts)} frames, {len(df)} points, "
              f"{df.tracklet_id.nunique()} tracklets -> points_{view}.json", flush=True)
    (out / "prep_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
