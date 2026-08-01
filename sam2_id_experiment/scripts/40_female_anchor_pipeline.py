from __future__ import annotations

"""End-to-end female-anchor pipeline: tag a whole-length 3D reconstruction F/M.

This wraps the *existing, validated* seg-ID steps into one resumable command so a
new experiment can be sexed without hand-driving five scripts. It changes no
algorithm; it only removes the K_1_2-specific manual plumbing.

Stages (each skips when its output already exists; ``--force`` re-runs all,
``--stop-after`` halts after a named stage)::

  0 slot-points  reproject every fish slot in <tag>_3d_identified.csv into each
                 side view -> <work>/slot_points/<view>.tracklet_bound.csv
                 (frame,x,y,score,global_id,tracklet_id). global_id == the
                 geometry slot ID_{k+1}, matching how 38 joins sex back on.
  1 detectron2   dense side masks on N sampled windows (runs in the SEPARATE
                 .venv-detectron2 interpreter)  -> detectron2_masks.h5 per window
  2 link (33)    link masks to the reprojected slot points -> per-window crops
  3 pool (34)    pool windows+views -> <work>/pooled/{crops.h5,crop_metadata.parquet}
  4 anchor (38)  apply the (reused) female classifier -> per-fish P(female),
                 pick exactly one female, write <tag>_3d_identified_sexed.csv

The classifier only *samples* windows to vote per fish; the sex label it produces
is applied to the whole-length CSV (every row gets sex/female_prob).

Run (main .venv)::

    python -m sam2_id_experiment.scripts.40_female_anchor_pipeline \
        20251225_K_1_1_FCMC --paths-config configs/paths.local.toml
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.reconstruction_3d.calibration import Calibration
from sam2_id_experiment.src.config import VIEW_VIDEO_BASENAMES
from sam2_id_experiment.src.id_model.reproject_slot_points import build_slot_points

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGES = ["slot-points", "detectron2", "link", "pool", "anchor", "per-frame-resolve"]

# Machine-local defaults (the K_1_2-trained appearance models on this box). These
# are scratch paths, not network drives, and are fully overridable by flag.
# Default anchor is the SWAP-CORRECTED clean female model (script 44); the embedder
# is the frozen self-supervised model used to align the target's windows.
DEFAULT_FEMALE_MODEL = r"Z:\StephensU\medaka\runs\udmt_validation\segid\K_1_2\model_female_clean\model.pt"
DEFAULT_MALE_MODEL = r"Z:\StephensU\medaka\runs\udmt_validation\segid\K_1_2\model_male_clean\model.pt"
DEFAULT_EMBEDDER = r"Z:\StephensU\medaka\runs\udmt_validation\segid\K_1_2\model_resnet_full\model.pt"
DEFAULT_DETECTRON_MODEL_DIR = REPO_ROOT / "outputs" / "20251225_K_1_2_FCMC" / "side_detectron2_model_firstprompt"
DEFAULT_DETECTRON_VENV = REPO_ROOT / ".venv-detectron2" / "Scripts" / "python.exe"


def make_windows(total_frames: int, n: int, length: int, margin: int) -> list[tuple[int, int]]:
    """N windows of ``length`` frames spread evenly across [margin, total-length-margin].

    Bounds are INCLUSIVE: a window is ``(s, s + length - 1)`` so it spans exactly
    ``length`` frames (Detectron2 inference and script 33 both treat END as
    inclusive). Even spacing keeps consecutive windows from sharing a frame.
    """
    last_start = total_frames - length - margin
    if last_start <= margin:
        return [(margin, min(total_frames - 1, margin + length - 1))]
    if n <= 1:
        starts = [margin]
    else:
        step = (last_start - margin) / (n - 1)
        starts = [int(round(margin + i * step)) for i in range(n)]
    return [(s, s + length - 1) for s in starts]


def _h5_ok(path: Path, *, key: str | None = None, prefix: str | None = None) -> bool:
    """True only if the .h5 opens cleanly and has the expected content. Guards
    resume against a truncated/empty file left by a killed run."""
    if not path.exists():
        return False
    try:
        import h5py
        with h5py.File(path, "r") as h:
            keys = list(h.keys())
            if key is not None:
                return key in keys and getattr(h[key], "shape", (0,))[0] > 0
            if prefix is not None:
                return any(k.startswith(prefix) for k in keys)
            return True
    except Exception:
        return False


def parse_windows(spec: list[str]) -> list[tuple[int, int]]:
    out = []
    for item in spec:
        s, e = item.split(":")
        out.append((int(s), int(e)))
    return out


def run(cmd: list[str], log_path: Path, *, python: Path) -> None:
    """Run ``python -m ...`` (cmd[0] is the module path) streaming to a log file."""
    full = [str(python), "-m", *cmd]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  $ {python.name} -m {' '.join(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(full, cwd=str(REPO_ROOT), stdout=fh,
                              stderr=subprocess.STDOUT, env=os.environ.copy())
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
        raise SystemExit(f"stage failed (exit {proc.returncode}); log {log_path}\n--- tail ---\n{tail}")


def stage_slot_points(identified_csv: Path, calib: Calibration, views: list[str],
                      out_dir: Path, force: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = {v: out_dir / f"{v}.tracklet_bound.csv" for v in views}
    if not force and all(p.exists() for p in targets.values()):
        print(f"[0 slot-points] skip (exists) -> {out_dir}")
        return {v: str(p) for v, p in targets.items()}
    print(f"[0 slot-points] reprojecting {identified_csv.name} into {views}")
    df = pd.read_csv(identified_csv, usecols=["frame", "fish_id", "X", "Y", "Z"])
    tables = build_slot_points(df, calib, views)
    counts = {}
    for v, tbl in tables.items():
        tbl.to_csv(targets[v], index=False)
        counts[v] = int(len(tbl))
        print(f"  {v}: {len(tbl)} slot points -> {targets[v].name}")
    return {"paths": {v: str(p) for v, p in targets.items()}, "counts": counts}


def stage_detectron2(experiment: str, views: list[str], windows: list[tuple[int, int]],
                     medaka_root: Path, work: Path, model_dir: Path, d2_python: Path,
                     score_thresh: float, force: bool) -> list[dict]:
    seg = work / "segpilot"
    jobs = []
    for view in views:
        video = medaka_root / experiment / view / VIEW_VIDEO_BASENAMES[view]
        for s, e in windows:
            jdir = seg / f"{view}_f{s}_{e}"
            masks = jdir / "detectron2_masks.h5"
            jobs.append({"view": view, "window": [s, e], "masks": str(masks)})
            if not force and _h5_ok(masks, prefix="frame_"):
                print(f"[1 detectron2] skip {view} {s}-{e} (masks exist)")
                continue
            jdir.mkdir(parents=True, exist_ok=True)
            cfg = jdir / "config.yaml"
            cfg.write_text(
                f"experiment_name: {experiment}\n"
                f"view: {view}\n"
                f"video_path: {video.as_posix()}\n"
                f"output_dir: {jdir.as_posix()}\n"
                f"start_frame: {s}\n"
                f"end_frame: {e}\n"
                f"n_animals_expected: 3\n",
                encoding="utf-8",
            )
            print(f"[1 detectron2] {view} {s}-{e}")
            run([
                "sam2_id_experiment.scripts.13_run_detectron2_segmentation",
                "--config", str(cfg),
                "--detectron-config", str(model_dir / "detectron2_config.yaml"),
                "--model", str(model_dir / "model_final.pth"),
                "--output-masks", str(masks),
                "--frame-stride", "1",
                "--score-thresh", str(score_thresh),
                "--device", "cuda",
            ], jdir / "detectron2.log", python=d2_python)
    return jobs


def stage_link(experiment: str, views: list[str], windows: list[tuple[int, int]],
               medaka_root: Path, work: Path, slot_dir: Path, crop_size: int,
               force: bool) -> list[str]:
    seg = work / "segpilot"
    segid = work / "segid"
    window_dirs = []
    for s, e in windows:
        wdir = segid / f"win_{s}_{e}"
        window_dirs.append(str(wdir))
        for view in views:
            crops = wdir / f"{view}_crops.h5"
            meta = wdir / f"{view}_crop_metadata.parquet"
            if not force and _h5_ok(crops, key="images") and meta.exists():
                print(f"[2 link] skip {view} {s}-{e} (crops exist)")
                continue
            masks = seg / f"{view}_f{s}_{e}" / "detectron2_masks.h5"
            if not masks.exists():
                raise SystemExit(f"[2 link] missing masks for {view} {s}-{e}: {masks}")
            video = medaka_root / experiment / view / VIEW_VIDEO_BASENAMES[view]
            print(f"[2 link] {view} {s}-{e}")
            run([
                "sam2_id_experiment.scripts.33_build_segid_dataset",
                "--view", view,
                "--masks", str(masks),
                "--tracklet-bound-csv", str(slot_dir / f"{view}.tracklet_bound.csv"),
                "--video", str(video),
                "--window", str(s), str(e),
                "--out-dir", str(wdir),
                "--crop-size", str(crop_size),
            ], wdir / f"{view}_link.log", python=Path(sys.executable))
    return window_dirs


def stage_pool(window_dirs: list[str], views: list[str], slot_dir: Path,
               work: Path, force: bool) -> Path:
    pooled = work / "pooled"
    if not force and _h5_ok(pooled / "crops.h5", key="crops") and (pooled / "crop_metadata.parquet").exists():
        print(f"[3 pool] skip (exists) -> {pooled}")
        return pooled
    print(f"[3 pool] pooling {len(window_dirs)} window(s) x {len(views)} view(s)")
    run([
        "sam2_id_experiment.scripts.34_assemble_segid_training",
        "--segid-dir", *window_dirs,
        "--views", *views,
        "--tracklet-bound-dir", str(slot_dir),
        "--out-dir", str(pooled),
    ], pooled / "pool.log", python=Path(sys.executable))
    return pooled


def stage_anchor(pooled: Path, female_model: Path, identified_csv: Path,
                 work: Path, margin_threshold: float, force: bool) -> dict:
    out_dir = work / "female_anchor"
    report_path = out_dir / "female_anchor_report.json"
    if not force and report_path.exists():
        print(f"[4 anchor] skip (exists) -> {out_dir}")
        return json.loads(report_path.read_text(encoding="utf-8"))
    print(f"[4 anchor] applying female classifier {female_model}")
    run([
        "sam2_id_experiment.scripts.38_apply_female_anchor",
        "--dataset", str(pooled),
        "--model", str(female_model),
        "--out-dir", str(out_dir),
        "--identified-3d-csv", str(identified_csv),
        "--margin-threshold", str(margin_threshold),
    ], out_dir / "anchor.log", python=Path(sys.executable))
    return json.loads(report_path.read_text(encoding="utf-8"))


def stage_anchor_clean(pooled: Path, female_model: Path, embedder: Path, identified_csv: Path,
                       work: Path, margin_threshold: float, force: bool) -> dict:
    """Swap-corrected anchor (script 44): align the target's windows, apply the clean
    female model, call the female, and write a whole-track sexed CSV that follows her
    geometry slot through the linker's crossings (with per-row sex_confidence)."""
    out_dir = work / "female_anchor"
    report_path = out_dir / "female_anchor_clean_report.json"
    if not force and report_path.exists():
        print(f"[4 anchor-clean] skip (exists) -> {out_dir}")
        return json.loads(report_path.read_text(encoding="utf-8"))
    print(f"[4 anchor-clean] swap-corrected female anchor {female_model}")
    run([
        "sam2_id_experiment.scripts.44_apply_clean_female_anchor",
        "--target-dataset", str(pooled),
        "--model", str(female_model),
        "--embedder", str(embedder),
        "--out-dir", str(out_dir),
        "--identified-3d-csv", str(identified_csv),
        "--margin-threshold", str(margin_threshold),
    ], out_dir / "anchor_clean.log", python=Path(sys.executable))
    return json.loads(report_path.read_text(encoding="utf-8"))


def stage_per_frame_resolve(experiment: str, paths_config: str, identified_csv: Path, slot_dir: Path,
                            masks_dir: str, views: list[str], female_model: str, male_model: str,
                            top_model: str | None, top_view: str, stride: int, work: Path,
                            force: bool) -> str:
    """Stage 5: per-frame appearance resolution of the whole 3-fish track (script 45). Supersedes the
    window-anchor sexing — resolves every female + M1/M2 swap per frame (optionally fusing a top model).
    Needs FULL-LENGTH Detectron2 masks (the saion fan-out) under ``masks_dir``."""
    out_csv = identified_csv.with_name(identified_csv.stem + "_individual_id.csv")
    if not force and out_csv.exists():
        print(f"[5 per-frame-resolve] skip (exists) -> {out_csv}")
        return str(out_csv)
    cmd = ["sam2_id_experiment.scripts.45_per_frame_resolve", experiment,
           "--paths-config", paths_config,
           "--identified-3d-csv", str(identified_csv),
           "--masks-dir", str(masks_dir),
           "--slot-points-dir", str(slot_dir),
           "--views", *views,
           "--female-model", str(female_model),
           "--male-model", str(male_model),
           "--stride", str(stride),
           "--out-csv", str(out_csv)]
    if top_model:
        cmd += ["--top-model", str(top_model), "--top-view", top_view]
    print(f"[5 per-frame-resolve] resolving every frame -> {out_csv.name}")
    run(cmd, work / "per_frame_resolve.log", python=Path(sys.executable))
    return str(out_csv)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Whole-length female-anchor pipeline (reproject -> Detectron2 -> 33 -> 34 -> 44).")
    p.add_argument("experiment", help="e.g. 20251225_K_1_1_FCMC")
    p.add_argument("--paths-config", default="configs/paths.local.toml")
    p.add_argument("--round-tag", default="round5_260611")
    p.add_argument("--identified-3d-csv", default=None,
                   help="default: <reconstruction_output_dir>/<round_tag>/<experiment>/<round_tag>_3d_identified.csv")
    p.add_argument("--work-dir", default=None,
                   help="default: runs/female_anchor/<experiment>/<round_tag>")
    p.add_argument("--views", nargs="+", default=["side1_cam", "side2_cam"])
    p.add_argument("--female-model", default=DEFAULT_FEMALE_MODEL)
    p.add_argument("--female-embedder", default=DEFAULT_EMBEDDER,
                   help="frozen self-supervised embedder for window alignment (clean anchor)")
    p.add_argument("--legacy-anchor", action="store_true",
                   help="use the old per-slot 38 anchor (no swap correction) instead of the clean 44 anchor")
    p.add_argument("--detectron-model-dir", default=str(DEFAULT_DETECTRON_MODEL_DIR))
    p.add_argument("--detectron-venv", default=str(DEFAULT_DETECTRON_VENV))
    p.add_argument("--calibration-model-dir", default=None)
    # window sampling
    p.add_argument("--windows", nargs="+", default=None, metavar="START:END",
                   help="explicit windows; default = evenly-spread sampling")
    p.add_argument("--num-windows", type=int, default=6)
    p.add_argument("--window-len", type=int, default=5000)
    p.add_argument("--window-margin", type=int, default=10000)
    p.add_argument("--total-frames", type=int, default=None,
                   help="default: max frame in the identified CSV")
    # knobs
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--margin-threshold", type=float, default=0.15)
    # stage 5: per-frame appearance resolution (opt-in; needs full-length masks)
    p.add_argument("--per-frame-resolve", action="store_true",
                   help="stage 5: resolve every frame's F/M1/M2 with the per-frame appearance models "
                        "(supersedes the window-anchor sexing); requires --masks-dir")
    p.add_argument("--masks-dir", default=None, help="full-length Detectron2 masks (<view>.masks_<S>-<E>.h5) for stage 5")
    p.add_argument("--male-model", default=DEFAULT_MALE_MODEL, help="M1/M2 classifier for stage 5")
    p.add_argument("--top-model", default=None, help="optional top-view male model -> fuse to shrink the male-low core")
    p.add_argument("--top-view", default="top_cam")
    p.add_argument("--pfr-stride", type=int, default=10, help="per-frame-resolve sampling stride")
    p.add_argument("--force", action="store_true", help="re-run all stages")
    p.add_argument("--stop-after", choices=STAGES, default=None)
    p.add_argument("--dry-run", action="store_true", help="resolve+print plan, run nothing")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    local = load_local_paths(args.paths_config)

    recon_root = Path(local.reconstruction_output_dir)
    medaka_root = Path(local.medaka_data_dir)
    identified_csv = Path(args.identified_3d_csv) if args.identified_3d_csv else (
        recon_root / args.round_tag / args.experiment / f"{args.round_tag}_3d_identified.csv")
    if not identified_csv.exists():
        raise SystemExit(f"identified 3D CSV not found: {identified_csv}")
    work = Path(args.work_dir) if args.work_dir else (
        REPO_ROOT / "runs" / "female_anchor" / args.experiment / args.round_tag)
    work.mkdir(parents=True, exist_ok=True)
    slot_dir = work / "slot_points"

    calib_dir = choose_path(args.calibration_model_dir,
                            local.calibration_model_dir if local else None)
    if calib_dir is None:
        raise SystemExit("calibration model dir not configured "
                         "(set calibration_model_dir in paths.local.toml or pass --calibration-model-dir)")
    if not Path(calib_dir).is_absolute():
        calib_dir = REPO_ROOT / calib_dir

    # windows
    if args.windows:
        windows = parse_windows(args.windows)
    else:
        total = args.total_frames
        if total is None:
            total = int(pd.read_csv(identified_csv, usecols=["frame"])["frame"].max()) + 1
        windows = make_windows(total, args.num_windows, args.window_len, args.window_margin)

    print("=" * 70)
    print(f"female-anchor pipeline | {args.experiment} | {args.round_tag}")
    print(f"  identified : {identified_csv}")
    print(f"  work-dir   : {work}")
    print(f"  views      : {args.views}")
    print(f"  windows    : {windows}")
    print(f"  female-mdl : {args.female_model}")
    print(f"  d2 model   : {args.detectron_model_dir}")
    print("=" * 70, flush=True)

    if args.dry_run:
        print("[dry-run] nothing executed.")
        return

    def reached(stage: str) -> bool:
        return args.stop_after is not None and STAGES.index(stage) >= STAGES.index(args.stop_after) + 1

    calib = Calibration(calib_dir)
    slot_info = stage_slot_points(identified_csv, calib, args.views, slot_dir, args.force)
    if reached("slot-points"):
        print("[stop-after slot-points]"); return

    stage_detectron2(args.experiment, args.views, windows, medaka_root, work,
                     Path(args.detectron_model_dir), Path(args.detectron_venv),
                     args.score_thresh, args.force)
    if reached("detectron2"):
        print("[stop-after detectron2]"); return

    window_dirs = stage_link(args.experiment, args.views, windows, medaka_root, work,
                             slot_dir, args.crop_size, args.force)
    if reached("link"):
        print("[stop-after link]"); return

    pooled = stage_pool(window_dirs, args.views, slot_dir, work, args.force)
    if reached("pool"):
        print("[stop-after pool]"); return

    if args.legacy_anchor:
        report = stage_anchor(pooled, Path(args.female_model), identified_csv, work,
                              args.margin_threshold, args.force)
        female_id = report.get("female_fish_id")
        prob_by_slot = report.get("female_prob") or {}
        female_slot = report.get("female_fish_slot")
        female_scalar_prob = (prob_by_slot.get(female_slot, prob_by_slot.get(str(female_slot)))
                              if female_slot is not None else None)
    else:
        report = stage_anchor_clean(pooled, Path(args.female_model), Path(args.female_embedder),
                                    identified_csv, work, args.margin_threshold, args.force)
        female_id = report.get("female_aligned_id")
        prob_by_slot = report.get("female_prob_by_aligned_id") or {}
        female_scalar_prob = (prob_by_slot.get(female_id, prob_by_slot.get(str(female_id)))
                              if female_id is not None else None)

    # copy the sexed CSV next to the source recon CSV for downstream discoverability
    sexed_src = report.get("sexed_csv")
    sexed_dst = None
    if sexed_src and Path(sexed_src).exists():
        sexed_dst = identified_csv.with_name(identified_csv.stem + "_sexed.csv")
        if Path(sexed_src) != sexed_dst:
            import shutil
            shutil.copy2(sexed_src, sexed_dst)

    # stage 5 (opt-in): per-frame appearance resolution of the whole track (supersedes window-anchor sexing)
    per_frame_resolve_csv = None
    if args.per_frame_resolve and args.stop_after != "anchor":
        if not args.masks_dir:
            raise SystemExit("--per-frame-resolve requires --masks-dir (full-length Detectron2 masks)")
        per_frame_resolve_csv = stage_per_frame_resolve(
            args.experiment, args.paths_config, identified_csv, slot_dir, args.masks_dir, args.views,
            args.female_model, args.male_model, args.top_model, args.top_view, args.pfr_stride, work, args.force)

    manifest = {
        "experiment": args.experiment, "round_tag": args.round_tag,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "identified_3d_csv": str(identified_csv),
        "windows": windows, "views": args.views,
        "per_frame_resolve_csv": per_frame_resolve_csv,
        "slot_points": slot_info,
        "female_model": str(args.female_model),
        "anchor_method": "legacy_38" if args.legacy_anchor else "clean_44_swap_corrected",
        "female_call": {
            "female_fish_id": female_id,
            "female_prob": female_scalar_prob,
            "female_prob_by_slot": prob_by_slot,
            "confidence_margin": report.get("confidence_margin"),
            "ambiguous": report.get("ambiguous"),
            "sex_by_slot": report.get("sex_by_slot") or report.get("sex_by_aligned_id"),
            "per_window_female_raw_global_id": report.get("per_window_female_raw_global_id"),
            "sex_confidence_counts": report.get("sex_confidence_counts"),
            "dropped_windows": report.get("dropped_windows"),
        },
        "sexed_csv_workdir": sexed_src,
        "sexed_csv_beside_recon": str(sexed_dst) if sexed_dst else None,
    }
    (work / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 70)
    print("DONE")
    print(f"  female fish : {female_id} "
          f"(margin {report.get('confidence_margin')}, ambiguous={report.get('ambiguous')})")
    print(f"  P(female)   : {female_scalar_prob}  (by slot: {prob_by_slot})")
    print(f"  sexed CSV   : {sexed_dst or sexed_src}")
    if per_frame_resolve_csv:
        print(f"  3-fish ID   : {per_frame_resolve_csv}  (per-frame F/M1/M2, supersedes window-anchor sexing)")
    print(f"  manifest    : {work / 'pipeline_manifest.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
