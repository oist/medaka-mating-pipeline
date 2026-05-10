from __future__ import annotations

"""
Prepare a new retraining round: split the annotation seed pkg.slp by view, then
stage per-run sleap-nn configs by copying the hand-tuned templates under
``configs/sleap/training/`` and overriding only run-specific paths.

The training config templates are committed and version-controlled. This
script does NOT regenerate them via ``sleap-nn config --auto``; it only fills
in labels / cache / output paths so each round has a reproducible config.

Inputs
------
- An annotation seed pkg.slp whose video filenames contain
  ``top_cam`` / ``side1_cam`` / ``side2_cam`` (the layout written by
  ``scripts/sleap/mine_training_frames.py``).

Layout produced under ``--out-dir``
-----------------------------------
    <out>/
      top/
        labels.pkg.slp
        configs/
          centroid.yaml
          centered_instance.yaml
        runs/        # sleap-nn writes <ckpt_dir>/<run_name>/best.ckpt etc.
      side/
        labels.pkg.slp
        configs/{centroid.yaml, centered_instance.yaml}
        runs/

Disk cache lives on local SSD: ``C:\\sleap_cache\\<round>_<view>_<kind>``
(network drives like Z:\\ are too slow for hot training I/O).

Usage
-----
    python -m scripts.sleap.prepare_retraining \\
        --seed "C:/Users/Makoto Hiroi/Downloads/annotation_seed.pkg.slp" \\
        --out-dir "Z:/StephensU/medaka/sleap_models/retrain_20260508"

Optional ``--pretrained-from <prior_round_dir>`` will set
``model_config.pretrained_backbone_weights`` for fine-tuning round N+1 from
round N's ``best.ckpt`` per (view, kind).
"""

import argparse
import shutil
from pathlib import Path

import sleap_io as sio
from omegaconf import OmegaConf, open_dict


# Templates committed under configs/sleap/training/.
REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = REPO_ROOT / "configs" / "sleap" / "training"
TEMPLATE_FILES = {
    "centroid": TEMPLATE_DIR / "centroid.yaml",
    "centered_instance": TEMPLATE_DIR / "centered_instance.yaml",
}

LOCAL_CACHE_ROOT = Path("C:/sleap_cache")

VIEW_TAGS = {
    "top": ("top_cam",),
    "side": ("side1_cam", "side2_cam"),
}


def view_for(filename: str) -> str | None:
    fn = filename.lower().replace("\\", "/")
    for view, tags in VIEW_TAGS.items():
        if any(tag in fn for tag in tags):
            return view
    return None


def split_seed(seed_path: Path, out_dir: Path) -> dict[str, Path]:
    labels = sio.load_slp(str(seed_path))
    print(f"Loaded {len(labels.labeled_frames)} LFs from {seed_path}")

    by_view: dict[str, list] = {v: [] for v in VIEW_TAGS}
    skipped = 0
    for lf in labels.labeled_frames:
        v = view_for(str(lf.video.filename))
        if v is None:
            skipped += 1
            continue
        by_view[v].append(lf)
    if skipped:
        print(f"  warning: skipped {skipped} LFs whose filename matched no view tag")

    out_paths: dict[str, Path] = {}
    for view, lfs in by_view.items():
        if not lfs:
            print(f"  view '{view}': 0 frames, skipping")
            continue
        view_dir = out_dir / view
        view_dir.mkdir(parents=True, exist_ok=True)

        out_path = view_dir / "labels.pkg.slp"
        if out_path.exists():
            print(f"  view '{view}': {len(lfs)} frames, reusing existing {out_path}")
            out_paths[view] = out_path
            continue

        used_videos = list({id(lf.video): lf.video for lf in lfs}.values())
        sub = sio.Labels(
            labeled_frames=lfs,
            videos=used_videos,
            skeletons=labels.skeletons,
            tracks=labels.tracks,
            provenance=labels.provenance,
        )
        sio.save_slp(sub, str(out_path), embed="user")
        print(f"  view '{view}': {len(lfs)} frames -> {out_path}")
        out_paths[view] = out_path
    return out_paths


def stage_config(
    *,
    template_path: Path,
    out_path: Path,
    labels_path: Path,
    cache_path: Path,
    ckpt_dir: Path,
    run_name: str,
    pretrained_backbone_weights: Path | None = None,
) -> None:
    """Copy a template YAML and patch the run-specific paths."""
    cache_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(template_path)
    with open_dict(cfg):
        cfg.data_config.train_labels_path = [str(labels_path).replace("\\", "/")]
        cfg.data_config.cache_img_path = str(cache_path).replace("\\", "/")
        cfg.trainer_config.ckpt_dir = str(ckpt_dir).replace("\\", "/")
        cfg.trainer_config.run_name = run_name
        if pretrained_backbone_weights is not None:
            cfg.model_config.pretrained_backbone_weights = str(
                pretrained_backbone_weights
            ).replace("\\", "/")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_path)
    print(
        f"  staged {out_path.name}: labels={labels_path.name}, cache={cache_path}, "
        f"run_name={run_name}"
        + (
            f", pretrained={pretrained_backbone_weights}"
            if pretrained_backbone_weights
            else ""
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=Path, required=True, help="annotation_seed.pkg.slp")
    ap.add_argument("--out-dir", type=Path, required=True, help="root output dir")
    ap.add_argument(
        "--pretrained-from",
        type=Path,
        default=None,
        help=(
            "Optional path to a prior round dir laid out the same way "
            "(<view>/runs/<kind>/best.ckpt). When set, fills in "
            "model_config.pretrained_backbone_weights for fine-tuning."
        ),
    )
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    round_name = out_dir.name  # e.g. "retrain_20260508" -> used in cache path

    print(f"[1/2] Splitting {args.seed} by view")
    slp_by_view = split_seed(args.seed, out_dir)

    print(f"[2/2] Staging configs from templates {TEMPLATE_DIR}")
    train_cmds: list[str] = []
    for view, slp_path in slp_by_view.items():
        view_dir = out_dir / view
        for kind, template in TEMPLATE_FILES.items():
            cfg_out = view_dir / "configs" / f"{kind}.yaml"
            cache_path = LOCAL_CACHE_ROOT / f"{round_name}_{view}_{kind}"
            ckpt_dir = view_dir / "runs"

            pretrained = None
            if args.pretrained_from is not None:
                candidate = args.pretrained_from / view / "runs" / kind / "best.ckpt"
                if candidate.exists():
                    pretrained = candidate
                else:
                    print(
                        f"  note: --pretrained-from set but {candidate} not found; "
                        f"training {view}/{kind} from scratch"
                    )

            stage_config(
                template_path=template,
                out_path=cfg_out,
                labels_path=slp_path,
                cache_path=cache_path,
                ckpt_dir=ckpt_dir,
                run_name=kind,
                pretrained_backbone_weights=pretrained,
            )
            train_cmds.append(f"sleap-nn train {cfg_out.as_posix()}")

    print("\nDone. Train each model with:")
    for cmd in train_cmds:
        print(f"  {cmd}")


if __name__ == "__main__":
    main()
