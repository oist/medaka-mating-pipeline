from __future__ import annotations

"""
Batch SLEAP prediction across every medaka experiment folder.

Discovers experiment folders under `medaka_data_dir` (one folder per
experiment, each with `top_cam/cam0_top.mp4`, `side1_cam/cam1_side.mp4`,
`side2_cam/cam2_side.mp4`), then runs the configured top / side models on
each view. Existing `.slp` outputs are skipped unless `--force` is passed.

Typical usage:
    python -m scripts.sleap.batch_predict \\
      --paths-config configs/paths.local.toml

    python -m scripts.sleap.batch_predict \\
      --paths-config configs/paths.local.toml \\
      --experiments 20251220_D_1_1_FCMC --dry-run
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.sleap.batch_track import find_side_videos, run_sleap_tracking

DEFAULT_MODEL_CONFIG = REPO_ROOT / "configs" / "sleap" / "model_paths.yaml"


def load_model_paths(
    config_path: Path, models_root: Path
) -> tuple[list[Path], list[Path]]:
    """Resolve top (centroid, instance) and side (centroid, instance) model dirs."""
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    sleap_cfg = data.get("sleap", {})

    def _resolve(section: str) -> list[Path]:
        entry = sleap_cfg.get(section, {}) or {}
        centroid = entry.get("centroid")
        instance = entry.get("instance")
        if not centroid or not instance:
            raise ValueError(
                f"{config_path}: [sleap.{section}] must define 'centroid' and 'instance'"
            )
        paths = []
        for rel in (centroid, instance):
            p = Path(rel)
            if not p.is_absolute():
                p = models_root / p
            paths.append(p.resolve())
        return paths

    return _resolve("top"), _resolve("side")


def discover_experiments(
    data_root: Path, pattern: str | None = None
) -> list[tuple[str, Path]]:
    """Return sorted (experiment_name, top_video_path) pairs."""
    regex = re.compile(pattern) if pattern else None
    results: list[tuple[str, Path]] = []
    for top_video in sorted(data_root.glob("*/top_cam/cam0_top.mp4")):
        exp_name = top_video.parent.parent.name
        if regex is not None and not regex.search(exp_name):
            continue
        results.append((exp_name, top_video))
    return results


def output_path_for(out_root: Path, exp_name: str, view: str) -> Path:
    return out_root / exp_name / f"{view}.slp"


def _is_complete(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch SLEAP prediction across every medaka experiment folder. "
            "Reads data and model roots from paths.local.toml; writes .slp "
            "files under sleap_output_dir/<experiment>/{top_cam,side1_cam,side2_cam}.slp."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--paths-config",
        default=None,
        help="Path to paths.local.toml (default: configs/paths.local.toml).",
    )
    parser.add_argument(
        "--model-config",
        default=str(DEFAULT_MODEL_CONFIG),
        help="YAML file listing top + side centroid/instance model dirs.",
    )
    parser.add_argument(
        "--medaka-data-dir",
        default=None,
        help="Override medaka_data_dir from paths.local.toml.",
    )
    parser.add_argument(
        "--sleap-models-dir",
        default=None,
        help="Override sleap_models_dir from paths.local.toml.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override sleap_output_dir from paths.local.toml.",
    )
    parser.add_argument(
        "--experiments",
        default=None,
        help="Regex filter on experiment folder names (e.g. '20260109_.*').",
    )
    parser.add_argument("--max-instances", type=int, default=3)
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["cpu", "gpu", "first-gpu", "last-gpu"],
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the work plan without invoking sleap-track.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run inference even if the output .slp already exists.",
    )
    parser.add_argument(
        "--skip-missing-side",
        action="store_true",
        help="Skip experiments missing one or both side videos instead of erroring.",
    )
    args = parser.parse_args()

    local = load_local_paths(args.paths_config)

    data_dir_str = choose_path(
        args.medaka_data_dir, local.medaka_data_dir if local else None
    )
    models_dir_str = choose_path(
        args.sleap_models_dir, local.sleap_models_dir if local else None
    )
    output_dir_str = choose_path(
        args.output_dir, local.sleap_output_dir if local else None
    )

    missing = [
        name
        for name, value in [
            ("medaka_data_dir", data_dir_str),
            ("sleap_models_dir", models_dir_str),
            ("sleap_output_dir", output_dir_str),
        ]
        if not value
    ]
    if missing:
        raise SystemExit(
            f"Missing required paths: {', '.join(missing)}. "
            "Set them in paths.local.toml or pass the matching --flag."
        )

    data_dir = Path(data_dir_str).resolve()
    models_dir = Path(models_dir_str).resolve()
    output_dir = Path(output_dir_str).resolve()

    if not data_dir.exists():
        raise SystemExit(f"medaka_data_dir not found: {data_dir}")
    if not models_dir.exists():
        raise SystemExit(f"sleap_models_dir not found: {models_dir}")

    model_config = Path(args.model_config).resolve()
    if not model_config.exists():
        raise SystemExit(f"model config not found: {model_config}")

    top_models, side_models = load_model_paths(model_config, models_dir)
    for m in top_models + side_models:
        if not m.exists():
            raise SystemExit(f"Model directory does not exist: {m}")

    experiments = discover_experiments(data_dir, args.experiments)
    if not experiments:
        raise SystemExit(
            f"No experiments found under {data_dir} "
            f"(pattern={args.experiments!r})."
        )

    print(f"Data root   : {data_dir}")
    print(f"Models root : {models_dir}")
    print(f"Output root : {output_dir}")
    print(f"Top models  : {[str(m) for m in top_models]}")
    print(f"Side models : {[str(m) for m in side_models]}")
    print(f"Experiments : {len(experiments)}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str, str]] = []
    skipped = 0
    completed = 0

    for idx, (exp_name, top_video) in enumerate(experiments, 1):
        print(f"[{idx}/{len(experiments)}] {exp_name}")

        side1_video, side2_video = find_side_videos(top_video)

        views: list[tuple[str, Path | None, list[Path]]] = [
            ("top_cam", top_video, top_models),
            ("side1_cam", side1_video, side_models),
            ("side2_cam", side2_video, side_models),
        ]

        for view, video, models in views:
            out_path = output_path_for(output_dir, exp_name, view)
            if video is None:
                msg = f"  {view}: MISSING video"
                if args.skip_missing_side and view != "top_cam":
                    print(msg + " (skipped)")
                    continue
                print(msg)
                failures.append((exp_name, view, "video not found"))
                continue

            if _is_complete(out_path) and not args.force:
                print(f"  {view}: skip (exists) -> {out_path}")
                skipped += 1
                continue

            print(f"  {view}: {video} -> {out_path}")
            if args.dry_run:
                continue

            try:
                run_sleap_tracking(
                    video_path=video,
                    model_paths=models,
                    output_path=out_path,
                    max_instances=args.max_instances,
                    frames=None,
                    device=args.device,
                )
                completed += 1
            except Exception as exc:
                print(f"  {view}: FAILED ({exc})")
                failures.append((exp_name, view, str(exc)))

    print()
    print(f"Completed : {completed}")
    print(f"Skipped   : {skipped}")
    print(f"Failures  : {len(failures)}")
    for exp_name, view, reason in failures:
        print(f"  - {exp_name} / {view}: {reason}")

    if failures and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
