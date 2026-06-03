from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.config import config_section, ensure_output_dir, load_config
from sam2_id_experiment.src.tracklets.filter_rules import classify_tracklets
from sam2_id_experiment.src.tracklets.qc_metrics import rejected_intervals_from_qc

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify raw SAM2 tracklets for pseudo-label safety.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tracklets", required=True)
    parser.add_argument(
        "--review-labels",
        default=None,
        help=(
            "Optional CSV/parquet with frame, tracklet_id or sam2_track_id, and "
            "review_label columns. Labels conservatively override automatic QC."
        ),
    )
    parser.add_argument("--output-prefix", default="", help="Optional filename prefix, e.g. detectron2_.")
    return parser


def _video_shape(video_path: Path) -> tuple[int, int] | None:
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        return None
    return height, width


def _load_review_labels(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)


def _merge_review_labels(raw: pd.DataFrame, review: pd.DataFrame | None) -> pd.DataFrame:
    if review is None or review.empty:
        if "review_label" not in raw.columns:
            raw = raw.copy()
            raw["review_label"] = ""
        return raw
    required = {"frame", "review_label"}
    missing = required - set(review.columns)
    if missing:
        raise ValueError(f"Review labels missing columns: {', '.join(sorted(missing))}")
    key = "tracklet_id" if "tracklet_id" in review.columns else "sam2_track_id"
    if key not in review.columns:
        raise ValueError("Review labels need either tracklet_id or sam2_track_id.")
    review_small = review[["frame", key, "review_label"]].copy()
    out = raw.drop(columns=["review_label"], errors="ignore").merge(
        review_small,
        on=["frame", key],
        how="left",
    )
    out["review_label"] = out["review_label"].fillna("")
    return out


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    out_dir = ensure_output_dir(cfg)

    raw = pd.read_parquet(args.tracklets)
    raw = _merge_review_labels(raw, _load_review_labels(args.review_labels))
    qc_cfg = config_section(cfg, "qc")
    qc_cfg.setdefault("expected_object_count", cfg.n_animals_expected)
    per_frame, summary = classify_tracklets(
        raw,
        qc_cfg,
        image_shape=_video_shape(cfg.video_path),
    )

    prefix = str(args.output_prefix)
    qc_path = out_dir / f"{prefix}tracklets_qc.parquet"
    mask_qc_path = out_dir / f"{prefix}mask_qc_frames.parquet"
    mask_qc_csv_path = out_dir / f"{prefix}mask_qc_frames.csv"
    summary_path = out_dir / f"{prefix}tracklet_summary_qc.parquet"
    rejected_path = out_dir / f"{prefix}rejected_intervals.csv"
    report_path = out_dir / f"{prefix}qc_summary.json"

    per_frame.to_parquet(qc_path, index=False)
    per_frame.to_parquet(mask_qc_path, index=False)
    per_frame.to_csv(mask_qc_csv_path, index=False)
    summary.to_parquet(summary_path, index=False)
    rejected_intervals_from_qc(summary).to_csv(rejected_path, index=False)

    status_counts = summary["qc_status"].value_counts().to_dict() if not summary.empty else {}
    n_clean_frames = int(per_frame["qc_status"].eq("clean_train").sum()) if not per_frame.empty else 0
    flag_counts = {}
    if not per_frame.empty:
        for reason in per_frame["auto_qc_reason"].fillna("").astype(str):
            for part in reason.split(";"):
                if part:
                    flag_counts[part] = flag_counts.get(part, 0) + 1
    report = {
        "n_tracklets": int(summary.shape[0]),
        "n_mask_instances": int(per_frame.shape[0]),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "auto_qc_flag_counts": {str(k): int(v) for k, v in sorted(flag_counts.items())},
        "clean_train_instances": n_clean_frames,
        "estimated_pseudo_positive_contamination_rate": "not_estimated_requires_manual_audit",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
