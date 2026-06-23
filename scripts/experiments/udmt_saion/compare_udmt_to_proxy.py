from __future__ import annotations

"""Score native-UDMT tracks (<stem>-whole-filter5.npy) against the 3D identity,
using the SAME metric panel as the motion-proxy (scripts.experiments.
udmt_individual_tracking) so the two are directly comparable.

UDMT `.npy` is (num_fish, num_frames, 2) center XY in resized-image px
(resize_factor). UDMT frame t == original frame `start_frame + t`. We divide by
resize_factor to get original px, reproject the 3D `final_fish_id` into the view,
and compute ID-switches / consistency / coverage / IDF1 / female-vs-male.

Run (anywhere with calibration + the .npy)::

    python -m scripts.experiments.udmt_saion.compare_udmt_to_proxy \
        --npy <stem>-whole-filter5.npy --view side2_cam \
        --identified-3d-csv <..._3d_identified.csv> --start-frame 176300 \
        --resize-factor 0.8 --paths-config configs/paths.local.toml \
        --out report.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.experiments.udmt_individual_tracking import (
    identity_metrics, match_tracks_to_gt, reproject_gt,
)


def npy_to_tracks(npy_path: str, start_frame: int, resize_factor: float) -> pd.DataFrame:
    arr = np.load(npy_path)                       # (num_fish, num_frames, 2)
    rows = []
    for fid in range(arr.shape[0]):
        for t in range(arr.shape[1]):
            x, y = arr[fid, t]
            if np.isfinite(x) and np.isfinite(y):
                rows.append({"frame": int(start_frame + t), "track_id": int(fid),
                             "x": float(x) / resize_factor, "y": float(y) / resize_factor})
    return pd.DataFrame(rows, columns=["frame", "track_id", "x", "y"])


def build_parser():
    p = argparse.ArgumentParser(description="Score UDMT .npy vs 3D identity (proxy-comparable).")
    p.add_argument("--npy", required=True)
    p.add_argument("--view", required=True)
    p.add_argument("--identified-3d-csv", required=True)
    p.add_argument("--start-frame", type=int, required=True)
    p.add_argument("--resize-factor", type=float, default=0.8)
    p.add_argument("--female-fish", type=int, default=15)
    p.add_argument("--match-px", type=float, default=80.0)
    p.add_argument("--model-folder", default=None)
    p.add_argument("--paths-config", default=None)
    p.add_argument("--out", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    model_folder = args.model_folder
    if model_folder is None:
        local = load_local_paths(args.paths_config)
        model_folder = choose_path(None, local.calibration_model_dir if local else None)

    tracks = npy_to_tracks(args.npy, args.start_frame, args.resize_factor)
    if tracks.empty:
        raise SystemExit("empty UDMT tracks")
    end = int(tracks.frame.max())
    gt = reproject_gt(args.identified_3d_csv, args.view, model_folder, args.start_frame, end)
    matches = match_tracks_to_gt(tracks, gt, args.match_px)
    m = identity_metrics(tracks, gt, matches, args.female_fish)
    report = {"method": "udmt_native", "view": args.view,
              "window": [args.start_frame, end], "n_frames": int(tracks.frame.nunique()),
              "identity_vs_3d": m,
              "note": "native UDMT (XMem->ST-Net), Detectron2-seeded; scored vs 3D proxy "
                      "with the same panel as udmt_individual_tracking (motion proxy)."}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    pf = m["per_fish"]
    fem = [v for v in pf.values() if v["is_female"]]
    print(f"[{args.view}] UDMT: tracks {m['n_tracks']} switches {m['total_id_switches']} "
          f"consistency {m['track_identity_consistency']} cov {m['coverage']} IDF1~ {m['idf1_like']}")
    if fem:
        print(f"  female recall {fem[0]['recall']} consistency {fem[0]['id_consistency']}")
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
