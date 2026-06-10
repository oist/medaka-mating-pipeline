from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.evaluation.identity_sandwich import (
    SandwichConfig,
    run_identity_sandwich_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and conservatively repair ID labels by sandwiching critical gaps between clean ID-model tracklets."
    )
    parser.add_argument("--experiment", default=None, help="Experiment name under --outputs-root.")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--tracklets", default=None, help="Explicit tracklets_qc parquet/csv.")
    parser.add_argument("--crop-predictions", default=None, help="Explicit crop_predictions parquet/csv from an ID probe.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-gap-frames", type=int, default=500)
    parser.add_argument("--max-speed-px-per-frame", type=float, default=12.0)
    parser.add_argument("--min-crops", type=int, default=5)
    parser.add_argument("--min-prediction-purity", type=float, default=0.75)
    parser.add_argument("--min-median-margin", type=float, default=0.05)
    parser.add_argument("--max-bridge-deviation-px", type=float, default=80.0)
    parser.add_argument("--prediction-col", default="accepted_predicted_identity")
    parser.add_argument("--fallback-prediction-col", default="predicted_identity")
    parser.add_argument("--current-identity-col", default="reference_identity")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outputs_root = Path(args.outputs_root)
    tracklets_path = Path(args.tracklets) if args.tracklets else _resolve_tracklets(outputs_root, args.experiment)
    predictions_path = (
        Path(args.crop_predictions)
        if args.crop_predictions
        else _resolve_crop_predictions(outputs_root, args.experiment)
    )
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.experiment:
        output_dir = outputs_root / args.experiment / "identity_sandwich_audit"
    else:
        output_dir = predictions_path.parent / "identity_sandwich_audit"

    cfg = SandwichConfig(
        max_gap_frames=int(args.max_gap_frames),
        max_speed_px_per_frame=float(args.max_speed_px_per_frame),
        min_crops=int(args.min_crops),
        min_prediction_purity=float(args.min_prediction_purity),
        min_median_margin=float(args.min_median_margin),
        max_bridge_deviation_px=float(args.max_bridge_deviation_px),
        prediction_col=str(args.prediction_col),
        fallback_prediction_col=str(args.fallback_prediction_col),
        current_identity_col=str(args.current_identity_col) if args.current_identity_col else None,
    )
    result = run_identity_sandwich_audit(
        tracklets=_read_table(tracklets_path),
        crop_predictions=_read_table(predictions_path),
        config=cfg,
        output_dir=output_dir,
    )
    summary = dict(result["summary"])
    summary["tracklets"] = str(tracklets_path)
    summary["crop_predictions"] = str(predictions_path)
    summary["output_dir"] = str(output_dir)
    (output_dir / "identity_sandwich_run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def _resolve_tracklets(outputs_root: Path, experiment: str | None) -> Path:
    if not experiment:
        raise ValueError("Use --tracklets or provide --experiment for automatic resolution.")
    exp_dir = outputs_root / experiment
    candidates = [
        exp_dir / "top_cam_f176300_181300_contrastive" / "sleap_filtered_detectron2_tracklets_qc.parquet",
        exp_dir / "top_cam_f176300_181300" / "sleap_filtered_detectron2_codex_smoke_tracklets_qc.parquet",
        exp_dir / "top_cam_f176300_181300" / "sleap_filtered_detectron2_tracklets_qc.parquet",
        exp_dir / "top_cam_f176300_181300" / "tracklets_qc.parquet",
        exp_dir / "top_cam" / "tracklets_qc.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(
        exp_dir.glob("top_cam*/**/*tracklets_qc.parquet"),
        key=lambda p: (0 if "sleap_filtered" in str(p).lower() else 1, str(p)),
    )
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve top-camera tracklets for {experiment} under {outputs_root}")


def _resolve_crop_predictions(outputs_root: Path, experiment: str | None) -> Path:
    if not experiment:
        raise ValueError("Use --crop-predictions or provide --experiment for automatic resolution.")
    exp_dir = outputs_root / experiment
    matches = sorted(exp_dir.glob("contrastive_identity_probe/**/*crop_predictions.parquet"))
    if not matches:
        raise FileNotFoundError(f"Could not resolve crop predictions for {experiment} under {outputs_root}")

    def rank(path: Path) -> tuple[int, str]:
        text = str(path).lower()
        score = 0
        if "level1" in text or "local_within" in text or "within_chunk" in text:
            score -= 10
        if "within_after" in text:
            score -= 5
        if "level2" in text or "level3" in text or "from_k1" in text:
            score += 10
        return score, str(path)

    return sorted(matches, key=rank)[0]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


if __name__ == "__main__":
    main()
