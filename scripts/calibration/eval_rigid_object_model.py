"""Evaluate a calibration model against the known 12-ball rigid cube.

Given detected image coordinates (``rigid_object_image_coords.csv``) and the true
object geometry, reconstruct every ball to 3D with a model and score it two ways:

1. **Kabsch residual** (mm): rigid-fit the reconstructed points to the true object
   coordinates per pose; report per-ball residual. Measures shape fidelity.
2. **Pairwise-distance error** (mm): compare reconstructed inter-ball distances to
   the known object distances (cube edge=50, face-diagonal=70.71, space-diag=86.60).
   This is *fit-free* — a direct, absolute measure of metric accuracy and scale bias.

Both are computed only on the poses passed in (use a held-out split for an honest
comparison of base vs refined models). Prints a JSON summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.calibration.refine_calibration_rigid import apply_rigid, kabsch_rigid_fit
from scripts.reconstruction_3d.calibration import Calibration

VIEW_COLS = ["cam0_x", "cam0_y", "cam1_x", "cam1_y", "cam2_x", "cam2_y"]


def evaluate(image_coords_csv, geometry_csv, model_folder, pose_ids=None) -> dict:
    coords = pd.read_csv(image_coords_csv)
    coords["pose_id"] = coords["pose_id"].astype(str)
    if pose_ids is not None:
        pose_ids = set(map(str, pose_ids))
        coords = coords[coords["pose_id"].isin(pose_ids)]
    geom = pd.read_csv(geometry_csv)
    obj = {int(r.marker_id): np.array([r.object_x, r.object_y, r.object_z], float)
           for r in geom.itertuples(index=False)}
    calib = Calibration(model_folder)

    per_marker_resid, dist_err, dist_err_signed = [], [], []
    per_pose = []
    for pid, grp in coords.groupby("pose_id"):
        grp = grp.dropna(subset=VIEW_COLS)
        if len(grp) < 4:
            continue
        mids = grp["marker_id"].astype(int).tolist()
        feats = grp[VIEW_COLS].to_numpy(float)
        p3d = np.asarray(calib._model_im2xyz.predict(feats), float)
        pobj = np.array([obj[m] for m in mids])

        R, t = kabsch_rigid_fit(pobj, p3d)
        resid = np.linalg.norm(apply_rigid(pobj, R, t) - p3d, axis=1)
        per_marker_resid.extend(resid.tolist())

        # fit-free pairwise distances
        n = len(mids)
        pe = []
        for i in range(n):
            for j in range(i + 1, n):
                d_rec = float(np.linalg.norm(p3d[i] - p3d[j]))
                d_true = float(np.linalg.norm(pobj[i] - pobj[j]))
                e = d_rec - d_true
                dist_err.append(abs(e))
                dist_err_signed.append(e)
                pe.append(abs(e))
        per_pose.append({"pose_id": pid, "n_markers": n,
                         "kabsch_rms_mm": float(np.sqrt(np.mean(resid ** 2))),
                         "kabsch_max_mm": float(resid.max()),
                         "dist_mae_mm": float(np.mean(pe))})

    pr = np.array(per_marker_resid)
    de = np.array(dist_err)
    des = np.array(dist_err_signed)
    return {
        "model_folder": str(model_folder),
        "n_poses": int(len(per_pose)),
        "n_markers_total": int(len(pr)),
        "kabsch_resid_mm": {"mean": float(pr.mean()), "rms": float(np.sqrt((pr ** 2).mean())),
                            "median": float(np.median(pr)), "p95": float(np.percentile(pr, 95)),
                            "max": float(pr.max())},
        "pairwise_dist_err_mm": {"mae": float(de.mean()), "rms": float(np.sqrt((de ** 2).mean())),
                                 "p95": float(np.percentile(de, 95)), "max": float(de.max()),
                                 "signed_bias": float(des.mean())},
        "per_pose": per_pose,
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a calibration model on the 12-ball cube")
    p.add_argument("--image-coords-csv", required=True, type=Path)
    p.add_argument("--geometry-csv", required=True, type=Path)
    p.add_argument("--model-folder", required=True, type=Path)
    p.add_argument("--pose-list-file", type=Path, default=None,
                   help="optional text file with one pose_id per line (held-out set)")
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args(argv)
    pose_ids = None
    if args.pose_list_file:
        pose_ids = [ln.strip() for ln in args.pose_list_file.read_text().splitlines() if ln.strip()]
    rep = evaluate(args.image_coords_csv, args.geometry_csv, args.model_folder, pose_ids)
    summary = {k: v for k, v in rep.items() if k != "per_pose"}
    print(json.dumps(summary, indent=2))
    if args.out_json:
        args.out_json.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
