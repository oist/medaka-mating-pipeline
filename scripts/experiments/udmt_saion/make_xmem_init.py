from __future__ import annotations

"""Build UDMT's first-frame init from Detectron2 masks (replaces SEEM).

UDMT's `auto_initial_prediction.run_auto_initial_prediction` (SEEM) writes exactly
two artifacts that seed the native XMem->ST-Net pipeline:

  * `<tmp>/<stem>/masks/0000000.png`  -- PIL mode 'L', foreground=1 (BINARY; XMem
    runs num_objects=1, so all fish collapse to one class),
  * `<tmp>/<stem>/extracted-images/start_pos_array.txt` -- N rows `x,y` (`%.2f`,
    comma) in ORIGINAL-video pixels; row order == tracked-fish index.

This script reproduces those two outputs from our medaka Detectron2 instance masks
on the start frame, picking exactly the 3 real fish (gating reflections) by nearest
match to the 3D `final_fish_id` reprojected into the view. That is an automatic,
medaka-tuned substitute for UDMT's manual SAM click / SEEM init -- first frame only;
XMem + ST-Net do all the tracking.

Output is RESOLUTION-INDEPENDENT (original-frame binary PNG + original-px centroids
+ a meta json with the fish_id<->row mapping). The saion driver resizes the PNG to
the 480-shorter-side size XMem uses and writes the two files into the project tree.

Run::

    python -m scripts.experiments.udmt_saion.make_xmem_init \
        --view top_cam --masks <detectron2_masks.h5> \
        --identified-3d-csv <..._3d_identified.csv> --start-frame 176300 \
        --paths-config configs/paths.local.toml --out-dir <bundle>/init
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from scripts.common.local_paths import choose_path, load_local_paths


# ------------------------------- pure logic ------------------------------- #

def match_fish_to_masks(fish_xy: dict[int, tuple[float, float]],
                        mask_centroids: list[tuple[float, float]],
                        gate_px: float) -> dict[int, int]:
    """fish_id -> mask index, by nearest centroid within gate (greedy, unique).
    Reflections (masks not nearest to any fish) are left unmatched = dropped."""
    out: dict[int, int] = {}
    used: set[int] = set()
    # greedy by ascending best-distance so the cleanest matches win first
    order = sorted(fish_xy, key=lambda f: _nearest(fish_xy[f], mask_centroids, used)[1])
    for fid in order:
        mi, d = _nearest(fish_xy[fid], mask_centroids, used)
        if mi is not None and d <= gate_px:
            out[fid] = mi
            used.add(mi)
    return out


def _nearest(xy, centroids, used):
    best, bd = None, np.inf
    for i, c in enumerate(centroids):
        if i in used:
            continue
        d = float(np.hypot(xy[0] - c[0], xy[1] - c[1]))
        if d < bd:
            best, bd = i, d
    return best, bd


def build_binary_union(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Binary foreground (uint8, fg=1) = union of the chosen instance masks."""
    out = np.zeros(shape, dtype=np.uint8)
    for m in masks:
        out[np.asarray(m, dtype=bool)] = 1
    return out


def disk_mask(shape: tuple[int, int], center: tuple[float, float], radius: float) -> np.ndarray:
    """Filled disk (bool) at center -- a synthetic seed for a fish Detectron2 missed
    on the start frame (init-only; equivalent to UDMT's manual first-frame click)."""
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) <= radius ** 2


# ------------------------------- I/O glue --------------------------------- #

def _load_detectron2_frame(masks_h5: str, frame: int):
    """Read only the target frame group (fast; avoids iterating the whole h5)."""
    import h5py
    insts = []
    with h5py.File(masks_h5, "r") as h:
        key = f"frame_{int(frame):06d}"
        if key not in h:
            return insts
        g = h[key]
        for ik in sorted(k for k in g.keys() if k.startswith("instance_")):
            grp = g[ik]
            m = np.asarray(grp["mask"][()]).astype(bool)
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                continue
            insts.append({"mask": m, "centroid": (float(xs.mean()), float(ys.mean())),
                          "area": int(len(xs)), "score": float(grp.attrs.get("score", 1.0)),
                          "shape": m.shape})
    return insts


def _reproject_fish(csv_3d: str, view: str, model_folder: str, frame: int):
    import pandas as pd
    from scripts.reconstruction_3d.calibration import Calibration
    d = pd.read_csv(csv_3d)
    d = d[(d.frame == int(frame))].dropna(subset=["X", "Y", "Z"])
    calib = Calibration(str(model_folder))
    out = {}
    for fish, g in d.groupby("final_fish_id"):
        xyz = g[["X", "Y", "Z"]].to_numpy(dtype=float)[:1]
        uv = calib.project_world_to_view(xyz, view)[0]
        if np.isfinite(uv).all():
            out[int(fish)] = (float(uv[0]), float(uv[1]))
    return out


def build_parser():
    p = argparse.ArgumentParser(description="Detectron2 -> UDMT XMem first-frame init.")
    p.add_argument("--view", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--identified-3d-csv", required=True)
    p.add_argument("--start-frame", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-folder", default=None)
    p.add_argument("--paths-config", default=None)
    p.add_argument("--gate-px", type=float, default=120.0)
    p.add_argument("--num-fish", type=int, default=3)
    p.add_argument("--no-synthetic-seed", action="store_true",
                   help="error instead of synthetic-disk-seeding a fish Detectron2 missed on frame 0")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    model_folder = args.model_folder
    if model_folder is None:
        local = load_local_paths(args.paths_config)
        model_folder = choose_path(None, local.calibration_model_dir if local else None)
    if model_folder is None:
        raise SystemExit("need --model-folder or calibration_model_dir")

    insts = _load_detectron2_frame(args.masks, args.start_frame)
    if not insts:
        raise SystemExit(f"no Detectron2 masks at frame {args.start_frame}")
    fish_xy = _reproject_fish(args.identified_3d_csv, args.view, model_folder, args.start_frame)
    if len(fish_xy) < args.num_fish:
        raise SystemExit(f"only {len(fish_xy)} 3D fish at frame {args.start_frame}; need {args.num_fish}")

    centroids = [i["centroid"] for i in insts]
    match = match_fish_to_masks(fish_xy, centroids, args.gate_px)
    shape = insts[0]["shape"]

    # fish Detectron2 missed on the start frame -> synthetic disk seed at the known
    # 3D-reprojected position (init only), unless --no-synthetic-seed.
    matched_radii = [float(np.sqrt(insts[match[f]]["area"] / np.pi)) for f in match]
    seed_radius = float(np.median(matched_radii)) if matched_radii else 25.0
    synthetic = sorted(set(fish_xy) - set(match))[: args.num_fish - len(match)]
    if synthetic and args.no_synthetic_seed:
        raise SystemExit(f"matched only {len(match)}/{args.num_fish} fish to masks "
                         f"(gate {args.gate_px}px) and --no-synthetic-seed set; "
                         f"missing {synthetic} at frame {args.start_frame}")
    if len(match) + len(synthetic) < args.num_fish:
        raise SystemExit(f"only {len(match)} matched + {len(synthetic)} synthetic < "
                         f"{args.num_fish}; inspect frame {args.start_frame}")

    # row order = fish id sorted -> stable, recorded in meta for later scoring
    ordered_fish = sorted(set(match) | set(synthetic))[: args.num_fish]
    chosen_masks, row_centroids = [], []
    for f in ordered_fish:
        if f in match:
            chosen_masks.append(insts[match[f]]["mask"])
            row_centroids.append(insts[match[f]]["centroid"])
        else:
            chosen_masks.append(disk_mask(shape, fish_xy[f], seed_radius))
            row_centroids.append(fish_xy[f])
    # PER-INSTANCE label mask: fish at row i -> XMem object label i+1, so XMem propagates
    # 3 SEPARATE identity-carrying masks (native UDMT/SEEM behaviour). A binary union here
    # merges all fish into one XMem object and the tracker can no longer keep them apart.
    label_mask = np.zeros(shape, dtype=np.uint8)
    for i, m in enumerate(chosen_masks):
        label_mask[np.asarray(m, dtype=bool)] = i + 1

    # write resolution-independent artifacts (per-instance label PNG: 0,1,2,3 single channel)
    import cv2
    cv2.imwrite(str(out / f"{args.view}_init_mask.png"), label_mask)
    np.savetxt(out / f"{args.view}_start_pos.txt",
               np.array(row_centroids, dtype=float), fmt="%.2f", delimiter=",")
    meta = {
        "view": args.view, "start_frame": int(args.start_frame),
        "orig_shape_hw": [int(shape[0]), int(shape[1])],
        "num_fish": int(len(ordered_fish)),
        "row_to_final_fish_id": {i: int(f) for i, f in enumerate(ordered_fish)},
        "row_centroids_origpx": [[round(x, 2), round(y, 2)] for x, y in row_centroids],
        "synthetic_seed_fish": [int(f) for f in synthetic],
        "seed_radius_px": round(seed_radius, 1),
        "n_masks_in_frame": int(len(insts)),
        "n_reflections_dropped": int(len(insts) - len(match)),
        "gate_px": args.gate_px,
        "note": ("init mask is PER-INSTANCE labels (fish row i -> XMem object i+1); "
                 "XMem propagates one mask per fish. row order matches start_pos rows."),
    }
    (out / f"{args.view}_init_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
