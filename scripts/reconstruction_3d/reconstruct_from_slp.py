from __future__ import annotations

"""
Three-view 3D reconstruction + identity tracking, driven by merged SLEAP
`.slp` files (one per view) produced by the HPC pipeline.

For each frame:
    1. Collect up to `num_individuals` predicted 2D points per view.
    2. Enumerate every (cam1_perm × cam2_perm) assignment, triangulate each
       candidate triplet via the calibration model, and score it by the
       triplet's registration error (reprojection loss across views).
    3. Pick the minimum-cost candidate per frame.
    4. Link the chosen 3D points across frames with a Hungarian NN pass.
    5. Write `<experiment>_3d.csv` with columns [frame, fish_id, X, Y, Z].

CLI:
    python -m scripts.reconstruction_3d.reconstruct_from_slp 20251220_D_1_1_FCMC \\
        --paths-config configs/paths.local.toml \\
        --start-frame 171800 --end-frame 176799
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.reconstruction_3d.calibration import Calibration
from scripts.reconstruction_3d.cross_frame_tracking import (
    apply_cross_frame_tracking,
    apply_first_frame_id_only,
)
from scripts.reconstruction_3d.fill_missing_3d import (
    best_by_frame_to_dfs,
    dfs_to_single_csv,
    ensure_all_frames_and_fill_spline,
)
from scripts.reconstruction_3d.load_2d_slp import VIEW_TO_CAM, load_points_by_frame
from scripts.reconstruction_3d.select_best import select_best_all_frames
from scripts.reconstruction_3d.triangulate import compute_candidates


def run_reconstruction_from_slp(
    experiment_name: str,
    *,
    slp_dir: Path,
    model_folder: Path,
    output_dir: Path,
    num_individuals: int = 3,
    num_candidates: int | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    anchor_node: str | None = None,
    xyz_bounds: Tuple[float, float] | None = None,
    max_residual_per_fish: float | None = None,
    max_jump_mm: float | None = None,
    max_jump_hard_cap_mm: float | None = None,
    max_gap_frames: int | None = None,
    first_frame_id_only: bool = False,
    fill_missing: bool = True,
    shift_to_positive: bool = True,
    plot: bool = True,
    output_tag: str | None = None,
    save_best_json: bool = True,
    save_best_pre_link_json: bool = True,
) -> dict:
    slp_paths = {
        view: slp_dir / f"{view}.slp" for view in VIEW_TO_CAM
    }

    points_by_frame = load_points_by_frame(
        slp_paths,
        num_individuals=num_individuals,
        num_candidates=num_candidates,
        start_frame=start_frame,
        end_frame=end_frame,
        anchor_node=anchor_node,
    )
    if not points_by_frame:
        raise RuntimeError(
            f"No frames loaded from {slp_dir} (filters: "
            f"{start_frame}..{end_frame})."
        )

    cal = Calibration(str(model_folder))
    allow_permutations = bool(cal.has_geometric_scoring)

    if num_candidates is not None and num_candidates > num_individuals:
        # Each view exposes more candidates than fish: enumerate all triplets
        # through the calibration model, drop those outside xyz_bounds, and
        # greedily pick `num_individuals` mutually-exclusive triplets minimising
        # reprojection error. Skips select_best_all_frames since the picker is
        # already producing a single optimal candidate per frame.
        from scripts.reconstruction_3d.triangulate_extra_candidates import (
            compute_best_assignments_per_frame,
        )
        best = compute_best_assignments_per_frame(
            cal,
            points_by_frame,
            xyz_bounds=xyz_bounds,
            num_individuals=num_individuals,
            num_candidates=num_candidates,
        )
        if max_residual_per_fish is not None:
            for info in best.values():
                errors = info["errors"]
                groups = info["groups"]
                dropped = []
                for lb, err in errors.items():
                    if not np.isfinite(err) or err > max_residual_per_fish:
                        groups[lb] = [float("nan")] * 3
                        dropped.append(lb)
                info["dropped_high_residual"] = dropped
    else:
        candidates = compute_candidates(
            cal,
            points_by_frame,
            xyz_bounds=xyz_bounds,
            allow_permutations=allow_permutations,
            num_individuals=num_individuals,
        )
        best = select_best_all_frames(
            candidates,
            xyz_bounds=xyz_bounds,
            num_individuals=num_individuals,
            max_residual_per_fish=max_residual_per_fish,
        )
    if first_frame_id_only:
        best_tracked = apply_first_frame_id_only(best)
    else:
        best_tracked = apply_cross_frame_tracking(
            best,
            num_individuals=num_individuals,
            max_jump_mm=max_jump_mm,
            max_jump_hard_cap_mm=max_jump_hard_cap_mm,
            max_gap_frames=max_gap_frames,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = output_tag or experiment_name

    pre_link_json_path: Path | None = None
    if save_best_pre_link_json:
        pre_link_json_path = output_dir / f"{tag}_best_pre_link.json"
        save_best_pre_link_json_to(
            best, pre_link_json_path, num_individuals=num_individuals
        )

    best_json_path: Path | None = None
    if save_best_json:
        best_json_path = output_dir / f"{tag}_best.json"
        _save_best_tracked_json(
            best_tracked, best_json_path, num_individuals=num_individuals
        )

    dfs_3d = best_by_frame_to_dfs(
        best_tracked,
        shift_to_positive=shift_to_positive,
        ensure_non_negative=False,
        num_individuals=num_individuals,
    )
    if fill_missing and points_by_frame:
        all_frames = sorted(points_by_frame.keys())
        dfs_3d = ensure_all_frames_and_fill_spline(dfs_3d, all_frames)
    csv_path = dfs_to_single_csv(dfs_3d, set_key=tag, output_dir=str(output_dir))

    png_path = None
    if plot:
        from scripts.reconstruction_3d.visualize_3d import run_plot
        try:
            png_path = run_plot(str(output_dir), tag, xyz_lim=xyz_bounds)
        except Exception as exc:  # noqa: BLE001
            print(f"Plot failed: {exc}")

    return {
        "csv_path": csv_path,
        "png_path": png_path,
        "best_json_path": str(best_json_path) if best_json_path else None,
        "best_pre_link_json_path": (
            str(pre_link_json_path) if pre_link_json_path else None
        ),
        "points_by_frame": points_by_frame,
        "best_tracked": best_tracked,
        "num_frames": len(points_by_frame),
        "num_individuals": num_individuals,
    }


def _src_labels(num_individuals: int) -> list[str]:
    return [chr(ord("a") + i) for i in range(num_individuals)]


def save_best_pre_link_json_to(
    best: dict, path: Path, *, num_individuals: int
) -> None:
    """
    Per-frame pre-link artifact: the chosen min-cost candidate before any
    cross-frame ID linking. Re-linking experiments consume this so they can
    iterate without re-running triangulation.

    Format (compact JSON):
        {
          "num_individuals": 3,
          "src_labels": ["a", "b", "c"],
          "frames": {
            "<frame>": {
              "groups":  {"a":[x,y,z], "b":[x,y,z], "c":[x,y,z]},
              "errors":  {"a":<float|null>, ...},
              "perm_cam1": [..],
              "perm_cam2": [..],
              "dropped_high_residual": ["a", ...]
            }, ...
          }
        }
    """
    src = _src_labels(num_individuals)
    out: dict = {}
    nan = float("nan")
    for frame, info in best.items():
        groups = info.get("groups", {}) or {}
        errors = info.get("errors", {}) or {}
        # perm_cam0 is implicit identity in the legacy path (cam0 anchor) but
        # explicit when triangulating from a larger candidate pool than fish
        # count; persist whichever is provided so cross-view linking can map
        # back to the right .slp instance index.
        perm_cam0 = list(info.get("perm_cam0") or list(range(num_individuals)))
        out[int(frame)] = {
            "groups": {
                lb: [float(v) for v in (groups.get(lb) or [nan, nan, nan])]
                for lb in src
            },
            "errors": {
                lb: (
                    float(errors[lb])
                    if (lb in errors and errors[lb] is not None)
                    else None
                )
                for lb in src
            },
            "perm_cam0": perm_cam0,
            "perm_cam1": list(info.get("perm_cam1", []) or []),
            "perm_cam2": list(info.get("perm_cam2", []) or []),
            "recovered": [bool(r) for r in (info.get("recovered") or [])],
            "dropped_high_residual": list(
                info.get("dropped_high_residual", []) or []
            ),
        }
    payload = {
        "num_individuals": num_individuals,
        "src_labels": src,
        "frames": out,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        # Compact: no indent. 348k frames at ~150 B/frame ≈ 50 MB.
        json.dump(payload, fh, separators=(",", ":"))


def load_best_pre_link_json(path: Path) -> dict:
    """
    Load a pre-link JSON produced by `save_best_pre_link_json_to` into the
    same shape `select_best_all_frames` returns, so it can be fed straight
    into `apply_cross_frame_tracking` (or any alternate linker).
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    out: dict = {}
    n = int(payload.get("num_individuals", 3))
    for f_str, info in payload["frames"].items():
        out[int(f_str)] = {
            "groups": {k: list(v) for k, v in info["groups"].items()},
            "errors": {
                k: (float(v) if v is not None else None)
                for k, v in info["errors"].items()
            },
            "perm_cam0": list(info.get("perm_cam0") or list(range(n))),
            "perm_cam1": list(info["perm_cam1"]),
            "perm_cam2": list(info["perm_cam2"]),
            "dropped_high_residual": list(info.get("dropped_high_residual", [])),
        }
    return out


def _save_best_tracked_json(
    best_tracked: dict, path: Path, *, num_individuals: int
) -> None:
    """
    Write a compact per-frame mapping usable by the multiview visualiser:
    - `cam0/1/2_to_id`: list of ID names (or null) for each SLEAP instance
      slot, indexed by the detection's position in the per-view sort used
      in `load_2d_slp`. Length = num_individuals.
    """
    id_labels = [f"ID_{i + 1}" for i in range(num_individuals)]
    out: dict = {}
    for frame, info in best_tracked.items():
        id_mapping = info.get("id_mapping") or []
        perm_cam0 = info.get("perm_cam0") or list(range(num_individuals))
        perm_cam1 = info.get("perm_cam1") or list(range(num_individuals))
        perm_cam2 = info.get("perm_cam2") or list(range(num_individuals))

        # detection-slot -> label index. With more candidates than fish, a
        # given .slp instance may not map to any fish (-1).
        det_indices = [d for d in (*perm_cam0, *perm_cam1, *perm_cam2) if d >= 0]
        n_slots = max([num_individuals] + [d + 1 for d in det_indices])
        cam0_label = [-1] * n_slots
        cam1_label = [-1] * n_slots
        cam2_label = [-1] * n_slots
        for lbl_idx, det_idx in enumerate(perm_cam0):
            if 0 <= det_idx < n_slots:
                cam0_label[det_idx] = lbl_idx
        for lbl_idx, det_idx in enumerate(perm_cam1):
            if 0 <= det_idx < n_slots:
                cam1_label[det_idx] = lbl_idx
        for lbl_idx, det_idx in enumerate(perm_cam2):
            if 0 <= det_idx < n_slots:
                cam2_label[det_idx] = lbl_idx

        # id_mapping[i] = label idx for ID_(i+1); invert to label -> ID_name.
        label_to_id = [None] * num_individuals
        for id_idx, src_label in enumerate(id_mapping):
            if 0 <= src_label < num_individuals:
                label_to_id[src_label] = id_labels[id_idx]

        def _view_map(cam_label_list):
            return [
                label_to_id[k] if 0 <= k < num_individuals else None
                for k in cam_label_list
            ]

        # Recovered-by-id flag: indices align with id_labels (ID_1, ID_2, ...).
        recovered_by_id = [
            bool(r) for r in (info.get("recovered") or [False] * num_individuals)
        ]
        out[int(frame)] = {
            "cam0_to_id": _view_map(cam0_label),
            "cam1_to_id": _view_map(cam1_label),
            "cam2_to_id": _view_map(cam2_label),
            "recovered": recovered_by_id,
            "groups": {k: v for k, v in info.get("groups", {}).items()},
        }

    payload = {
        "num_individuals": num_individuals,
        "id_labels": id_labels,
        "frames": out,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "3D reconstruction + identity tracking from merged SLEAP .slp files."
        )
    )
    parser.add_argument(
        "experiment",
        help="Experiment folder name, e.g. 20251220_D_1_1_FCMC.",
    )
    parser.add_argument("--paths-config", default=None)
    parser.add_argument("--sleap-output-dir", default=None,
                        help="Overrides sleap_output_dir.")
    parser.add_argument(
        "--sleap-subdir",
        default=None,
        help=(
            "Optional subfolder inside the experiment SLEAP directory, e.g. "
            "round3_3node_full. Defaults to the experiment root."
        ),
    )
    parser.add_argument("--model-folder", default=None,
                        help="Overrides calibration_model_dir.")
    parser.add_argument("--output-dir", default=None,
                        help="Overrides reconstruction_output_dir.")
    parser.add_argument("--num-individuals", type=int, default=3,
                        help="Max fish slots per view (default 3).")
    parser.add_argument(
        "--num-candidates", type=int, default=None,
        help=(
            "Per-view candidate count to load when SLEAP was run with "
            "more detections per frame than fish (e.g. -n 6 to recover real "
            "fish that would otherwise be displaced by reflections). When set "
            "and > --num-individuals, triangulation switches to greedy "
            "mutually-exclusive triplet selection across N^3 candidate "
            "triplets (with xyz_bounds gating reflections). Default: same as "
            "--num-individuals (legacy behaviour)."
        ),
    )
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--anchor-node",
        default="mouth",
        help=(
            "Use this SLEAP node for canonical 2D/3D tracking points "
            "(default: mouth)."
        ),
    )
    parser.add_argument(
        "--use-visible-node-centroid",
        action="store_true",
        help=(
            "Use the centroid of all visible nodes instead of --anchor-node. "
            "This is the round4 production choice: the mouth+tail centroid is "
            "invariant to the known round4 mouth<->tail node swaps. Leave off "
            "(mouth anchor) only for mouth-vs-centroid comparison experiments."
        ),
    )
    parser.add_argument(
        "--xyz-bounds", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"),
        help="Accept only 3D points inside this cube. Default: no bound.",
    )
    parser.add_argument(
        "--max-residual-per-fish", type=float, default=None,
        help=(
            "Per-fish registration-error threshold (pixel^2 units). Points "
            "from the winning candidate whose error exceeds this are set to "
            "NaN. Default: no threshold."
        ),
    )
    parser.add_argument(
        "--max-jump-mm", type=float, default=None,
        help=(
            "Per-frame velocity gate in millimetres. Hungarian matches "
            "farther than this * gap_frames are rejected. Default: no gate."
        ),
    )
    parser.add_argument(
        "--max-jump-hard-cap-mm", type=float, default=None,
        help=(
            "Absolute ceiling on the per-match distance (combined with "
            "--max-jump-mm so the effective threshold at gap=N is "
            "min(max_jump_mm*N, hard_cap)). Prevents long NaN gaps from "
            "accepting a different fish as the same ID. Default: no cap."
        ),
    )
    parser.add_argument(
        "--max-gap-frames", type=int, default=None,
        help=(
            "Retire an ID whose last valid 3D position is older than this "
            "many frames. Default: never retire."
        ),
    )
    parser.add_argument(
        "--no-shift-to-positive", action="store_true",
        help="Do not shift coordinates so the minimum sits at origin.",
    )
    parser.add_argument(
        "--first-frame-id-only", action="store_true",
        help="Use the first-frame ID mapping for the whole clip instead of "
             "Hungarian NN across frames.",
    )
    parser.add_argument(
        "--no-fill-missing", action="store_true",
        help="Do not linearly interpolate missing 3D frames.",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip the trajectory plot.",
    )
    parser.add_argument(
        "--no-save-best-json", action="store_true",
        help="Skip writing the per-frame detection->ID mapping JSON.",
    )
    parser.add_argument(
        "--no-save-best-pre-link-json", action="store_true",
        help="Skip writing the pre-link per-frame candidate JSON.",
    )
    parser.add_argument(
        "--output-tag", default=None,
        help="Suffix for output files (default: experiment name).",
    )
    parser.add_argument(
        "--bind-top-id", action="store_true",
        help="After 3D temporal linking, attach top_cam 2D track IDs via "
             "reprojection-NN (writes <tag>_3d_identified.csv). Requires "
             "--top-tracks-csv.",
    )
    parser.add_argument(
        "--top-tracks-csv", default=None, type=Path,
        help="Path to top_cam.tracks.csv produced by track_simple_nn. "
             "Required when --bind-top-id is set.",
    )
    parser.add_argument(
        "--bind-distance", type=float, default=60.0,
        help="Max projection-NN gate in px for the bind step (default 60).",
    )
    parser.add_argument(
        "--bind-min-frames", type=int, default=500,
        help="Drop top tracks shorter than this from the bind anchor pool.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    local = load_local_paths(args.paths_config)
    sleap_output_dir = choose_path(
        args.sleap_output_dir,
        local.sleap_output_dir if local else None,
    )
    model_folder = choose_path(
        args.model_folder,
        local.calibration_model_dir if local else None,
    )
    output_dir = choose_path(
        args.output_dir,
        local.reconstruction_output_dir if local else None,
        sleap_output_dir,
    )
    if sleap_output_dir is None:
        parser.error(
            "sleap_output_dir is required (paths.local.toml or --sleap-output-dir)."
        )
    if model_folder is None:
        parser.error(
            "calibration_model_dir is required "
            "(paths.local.toml or --model-folder)."
        )

    slp_dir = Path(sleap_output_dir) / args.experiment
    if args.sleap_subdir:
        slp_dir = slp_dir / args.sleap_subdir
    if not slp_dir.is_dir():
        parser.error(f"SLEAP output folder not found: {slp_dir}")

    tag = args.output_tag or args.experiment
    if args.start_frame is not None or args.end_frame is not None:
        start = args.start_frame if args.start_frame is not None else "start"
        end = args.end_frame if args.end_frame is not None else "end"
        tag = f"{tag}_f{start}-{end}"

    result = run_reconstruction_from_slp(
        experiment_name=args.experiment,
        slp_dir=slp_dir,
        model_folder=Path(model_folder),
        output_dir=Path(output_dir),
        num_individuals=args.num_individuals,
        num_candidates=args.num_candidates,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        anchor_node=None if args.use_visible_node_centroid else args.anchor_node,
        xyz_bounds=tuple(args.xyz_bounds) if args.xyz_bounds else None,
        max_residual_per_fish=args.max_residual_per_fish,
        max_jump_mm=args.max_jump_mm,
        max_jump_hard_cap_mm=args.max_jump_hard_cap_mm,
        max_gap_frames=args.max_gap_frames,
        first_frame_id_only=args.first_frame_id_only,
        fill_missing=not args.no_fill_missing,
        shift_to_positive=not args.no_shift_to_positive,
        plot=not args.no_plot,
        output_tag=tag,
        save_best_json=not args.no_save_best_json,
        save_best_pre_link_json=not args.no_save_best_pre_link_json,
    )

    print()
    print(f"Frames processed : {result['num_frames']}")
    print(f"Num individuals  : {result['num_individuals']}")
    print(f"3D CSV           : {result['csv_path']}")
    if result["png_path"]:
        print(f"Trajectory plot  : {result['png_path']}")
    if result["best_json_path"]:
        print(f"best-tracked JSON: {result['best_json_path']}")
    if result["best_pre_link_json_path"]:
        print(f"pre-link JSON    : {result['best_pre_link_json_path']}")

    if args.bind_top_id:
        if args.top_tracks_csv is None:
            parser.error("--bind-top-id requires --top-tracks-csv.")
        if not args.top_tracks_csv.exists():
            parser.error(f"top tracks CSV not found: {args.top_tracks_csv}")
        from scripts.reconstruction_3d.bind_topcam_id import bind
        exp_3d_csv = Path(result["csv_path"])
        out_csv = exp_3d_csv.with_name(exp_3d_csv.stem + "_identified.csv")
        calib = Calibration(str(model_folder))
        bind(
            exp_3d_csv=exp_3d_csv,
            top_tracks_csv=args.top_tracks_csv,
            calib=calib,
            out_csv=out_csv,
            bind_distance=args.bind_distance,
            require_min_frames=args.bind_min_frames,
        )
        print(f"3D identified CSV: {out_csv}")


if __name__ == "__main__":
    main()
