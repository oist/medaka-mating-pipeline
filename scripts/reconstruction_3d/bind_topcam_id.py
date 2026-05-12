from __future__ import annotations

"""
Reprojection-NN bind: attach top-camera 2D track IDs to the 3D tracks produced
by `reconstruct_from_slp` (geometry-only IDs).

Per frame:
  1. Project each 3D point onto the top_cam image plane via
     `Calibration.project_world_to_view`.
  2. Hungarian-match the projected points against the top_cam tracks.csv rows
     for that frame, gated by `--bind-distance` pixels.
  3. Geometry slots with no top match (top occluded, top track filtered out by
     --require-min-frames, or projection outside the gate) get
     `source_top_id = NaN`, `bind_source = "geometry_only"` and KEEP the
     geometry stream's own 3D-temporal-link `fish_id`.

After the per-frame pass, each geometry `fish_id` is mapped to its modal
`source_top_id` over the full run and written to `final_fish_id` so that
downstream behavioral analysis sees a stable identity.

Output columns: frame, fish_id (geometry), X, Y, Z, source_top_id, bind_score,
bind_source, final_fish_id.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.reconstruction_3d.calibration import Calibration


def bind(
    exp_3d_csv: Path,
    top_tracks_csv: Path,
    *,
    calib: Calibration,
    out_csv: Path,
    bind_distance: float = 60.0,
    require_min_frames: int = 500,
) -> pd.DataFrame:
    geom = pd.read_csv(exp_3d_csv)
    top = pd.read_csv(top_tracks_csv)

    required_geom = {"frame", "fish_id", "X", "Y", "Z"}
    required_top = {"frame", "track_id", "x", "y"}
    missing_g = required_geom - set(geom.columns)
    missing_t = required_top - set(top.columns)
    if missing_g:
        raise ValueError(f"{exp_3d_csv} missing columns: {missing_g}")
    if missing_t:
        raise ValueError(f"{top_tracks_csv} missing columns: {missing_t}")

    sizes = top.groupby("track_id").size()
    keep = sizes[sizes >= require_min_frames].index
    n_dropped = len(sizes) - len(keep)
    if n_dropped:
        print(f"[bind] dropping {n_dropped} top tracks below {require_min_frames} frames")
    top = top[top["track_id"].isin(keep)].copy()

    top_by_frame = {int(fi): g for fi, g in top.groupby("frame")}

    rows: list[dict] = []
    for fi, gf in geom.groupby("frame"):
        gf = gf.reset_index(drop=True)
        xyz = gf[["X", "Y", "Z"]].to_numpy(dtype=float)
        valid_mask = np.isfinite(xyz).all(axis=1)

        proj = np.full((len(gf), 2), np.nan)
        if valid_mask.any():
            proj[valid_mask] = calib.project_world_to_view(xyz[valid_mask], "top_cam")

        tf = top_by_frame.get(int(fi))
        if tf is None or len(tf) == 0:
            for i in range(len(gf)):
                r = gf.iloc[i]
                rows.append({
                    "frame": int(fi),
                    "fish_id": r["fish_id"],
                    "X": float(r["X"]), "Y": float(r["Y"]), "Z": float(r["Z"]),
                    "source_top_id": np.nan,
                    "bind_score": np.nan,
                    "bind_source": "geometry_only",
                })
            continue

        top_xy = tf[["x", "y"]].to_numpy(dtype=float)
        top_ids = tf["track_id"].to_numpy()

        cost = np.full((len(gf), len(tf)), 1e9, dtype=float)
        if valid_mask.any():
            cost[valid_mask] = cdist(proj[valid_mask], top_xy)
        cost_for_lap = cost.copy()
        cost_for_lap[cost_for_lap > bind_distance] = 1e9
        row_ind, col_ind = linear_sum_assignment(cost_for_lap)

        used_geom: set[int] = set()
        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] > bind_distance:
                continue
            r = gf.iloc[ri]
            rows.append({
                "frame": int(fi),
                "fish_id": r["fish_id"],
                "X": float(r["X"]), "Y": float(r["Y"]), "Z": float(r["Z"]),
                "source_top_id": int(top_ids[ci]),
                "bind_score": float(cost[ri, ci]),
                "bind_source": "top_cam_2d",
            })
            used_geom.add(int(ri))

        for i in range(len(gf)):
            if i in used_geom:
                continue
            r = gf.iloc[i]
            rows.append({
                "frame": int(fi),
                "fish_id": r["fish_id"],
                "X": float(r["X"]), "Y": float(r["Y"]), "Z": float(r["Z"]),
                "source_top_id": np.nan,
                "bind_score": np.nan,
                "bind_source": "geometry_only",
            })

    df = pd.DataFrame(rows).sort_values(["frame", "fish_id"]).reset_index(drop=True)

    # Global Hungarian assignment on the (geometry fish_id) x (top track_id) tally
    # matrix — prevents two geometry IDs from collapsing onto one top track via
    # independent per-fish modes.
    bound_only = df.dropna(subset=["source_top_id"])
    if not bound_only.empty:
        tally = bound_only.pivot_table(
            index="fish_id", columns="source_top_id",
            values="bind_score", aggfunc="count", fill_value=0,
        )
        geom_ids = tally.index.tolist()
        top_ids_unique = tally.columns.tolist()
        cost = -tally.to_numpy(dtype=float)  # maximize tally => minimize negative
        if cost.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost)
            mapping: dict = {}
            for ri, ci in zip(row_ind, col_ind):
                if -cost[ri, ci] > 0:  # only accept positive-tally pairs
                    mapping[geom_ids[ri]] = int(top_ids_unique[ci])
        else:
            mapping = {}
    else:
        mapping = {}
        geom_ids = []

    print(f"[bind] global 1-to-1 assignment geometry fish_id -> top_track_id:")
    for fid in df["fish_id"].unique():
        tid = mapping.get(fid)
        print(f"  {fid}: {tid if tid is not None else 'unbound (fallback to geometry id)'}")

    fallback_str = df["fish_id"].astype(str)
    df["final_fish_id"] = df["fish_id"].map(mapping)
    df["final_fish_id"] = np.where(
        df["final_fish_id"].notna(),
        df["final_fish_id"].astype("Int64").astype(str),
        fallback_str,
    )

    bound = int((df["bind_source"] == "top_cam_2d").sum())
    total = len(df)
    print(f"[bind] {bound}/{total} ({100*bound/max(total,1):.1f}%) rows bound via top_cam_2d")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[bind] wrote {out_csv}")
    return df


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bind top-cam 2D track IDs to a 3D geometry CSV via reprojection."
    )
    p.add_argument("--exp-3d-csv", required=True, type=Path)
    p.add_argument("--top-tracks-csv", required=True, type=Path)
    p.add_argument("--out-csv", required=True, type=Path)
    p.add_argument("--bind-distance", type=float, default=60.0,
                   help="Max projection-NN cost in px to accept a bind (default 60).")
    p.add_argument("--require-min-frames", type=int, default=500,
                   help="Drop top tracks with fewer frames than this from the anchor pool.")
    p.add_argument("--paths-config", default=None)
    p.add_argument("--calibration-model-dir", default=None,
                   help="Calibration model folder; defaults to paths.local.toml entry.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    local = load_local_paths(args.paths_config)
    model_dir = choose_path(
        args.calibration_model_dir,
        local.calibration_model_dir if local else None,
    )
    if model_dir is None:
        raise SystemExit("calibration_model_dir required (paths.local.toml or --calibration-model-dir)")
    calib = Calibration(model_dir)
    bind(
        exp_3d_csv=args.exp_3d_csv,
        top_tracks_csv=args.top_tracks_csv,
        calib=calib,
        out_csv=args.out_csv,
        bind_distance=args.bind_distance,
        require_min_frames=args.require_min_frames,
    )


if __name__ == "__main__":
    main()
