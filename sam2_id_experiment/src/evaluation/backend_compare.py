from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sam2_id_experiment.src.sam2_adapter.mask_io import iter_mask_instances_h5


def write_backend_comparison(
    *,
    output_dir: str | Path,
    reference_masks: str | Path,
    candidate_masks: str | Path | None = None,
    tracklets_qc: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict:
    output_dir = Path(output_dir)
    report = {
        "output_dir": str(output_dir),
        "comparison_backend": "generic_mask_h5",
        "reference_masks": mask_file_summary(reference_masks),
    }
    if tracklets_qc is not None and Path(tracklets_qc).exists():
        qc = pd.read_parquet(tracklets_qc)
        report["reference_qc"] = {
            "n_instances": int(len(qc)),
            "qc_status_counts": {str(k): int(v) for k, v in qc["qc_status"].value_counts().to_dict().items()},
            "auto_qc_reason_counts": {
                str(k): int(v) for k, v in qc["auto_qc_reason"].fillna("").value_counts().head(20).to_dict().items()
            } if "auto_qc_reason" in qc else {},
        }
    if candidate_masks is None or not Path(candidate_masks).exists():
        report["candidate_status"] = "not_available"
        report["comparison"] = None
    else:
        report["candidate_status"] = "available"
        report["candidate_masks"] = mask_file_summary(candidate_masks)
        report["comparison"] = compare_mask_files(reference_masks, candidate_masks)

    path = Path(output_path) if output_path else output_dir / "backend_comparison.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def mask_file_summary(mask_path: str | Path) -> dict:
    frames: set[int] = set()
    n_instances = 0
    areas: list[int] = []
    masks_by_frame: dict[int, list[np.ndarray]] = {}
    for inst in iter_mask_instances_h5(mask_path):
        frame = int(inst.frame)
        frames.add(frame)
        n_instances += 1
        mask = np.asarray(inst.mask, dtype=bool)
        areas.append(int(mask.sum()))
        masks_by_frame.setdefault(frame, []).append(mask)
    instances_per_frame = [len(masks) for masks in masks_by_frame.values()]
    overlap_stats = _frame_overlap_summary(masks_by_frame)
    return {
        "path": str(mask_path),
        "n_frames": len(frames),
        "n_instances": n_instances,
        "mean_instances_per_frame": float(n_instances / len(frames)) if frames else 0.0,
        "instance_count_distribution": {
            str(k): int(v) for k, v in pd.Series(instances_per_frame, dtype="int64").value_counts().sort_index().to_dict().items()
        } if instances_per_frame else {},
        "mean_area": float(np.mean(areas)) if areas else 0.0,
        "median_area": float(np.median(areas)) if areas else 0.0,
        "mean_max_overlap_fraction": overlap_stats["mean_max_overlap_fraction"],
        "frames_with_overlap_fraction_gt_0_05": overlap_stats["frames_with_overlap_fraction_gt_0_05"],
    }


def compare_mask_files(reference_masks: str | Path, candidate_masks: str | Path) -> dict:
    ref_by_frame = _load_masks_by_frame(reference_masks)
    cand_by_frame = _load_masks_by_frame(candidate_masks)
    shared_frames = sorted(set(ref_by_frame) & set(cand_by_frame))
    if not shared_frames:
        return {"n_shared_frames": 0, "mean_best_iou": None, "mean_count_abs_error": None}

    best_ious: list[float] = []
    count_errors: list[int] = []
    for frame in shared_frames:
        refs = ref_by_frame[frame]
        cands = cand_by_frame[frame]
        count_errors.append(abs(len(refs) - len(cands)))
        if not refs or not cands:
            best_ious.extend([0.0] * len(refs))
            continue
        iou = np.asarray([[mask_iou(ref, cand) for cand in cands] for ref in refs])
        best_ious.extend(iou.max(axis=1).tolist())

    return {
        "n_shared_frames": len(shared_frames),
        "mean_best_iou": float(np.mean(best_ious)) if best_ious else None,
        "median_best_iou": float(np.median(best_ious)) if best_ious else None,
        "p10_best_iou": float(np.percentile(best_ious, 10)) if best_ious else None,
        "mean_count_abs_error": float(np.mean(count_errors)) if count_errors else None,
        "frames_with_count_mismatch": int(sum(error > 0 for error in count_errors)),
    }


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(left, right).sum() / union)


def _load_masks_by_frame(mask_path: str | Path) -> dict[int, list[np.ndarray]]:
    out: dict[int, list[np.ndarray]] = {}
    for inst in iter_mask_instances_h5(mask_path):
        out.setdefault(int(inst.frame), []).append(np.asarray(inst.mask, dtype=bool))
    return out


def _frame_overlap_summary(masks_by_frame: dict[int, list[np.ndarray]]) -> dict:
    max_overlaps: list[float] = []
    for masks in masks_by_frame.values():
        max_overlap = 0.0
        for i, left in enumerate(masks):
            left_area = int(left.sum())
            if left_area == 0:
                continue
            for right in masks[i + 1:]:
                right_area = int(right.sum())
                denom = min(left_area, right_area)
                if denom == 0:
                    continue
                overlap = float(np.logical_and(left, right).sum() / denom)
                max_overlap = max(max_overlap, overlap)
        max_overlaps.append(max_overlap)
    return {
        "mean_max_overlap_fraction": float(np.mean(max_overlaps)) if max_overlaps else 0.0,
        "frames_with_overlap_fraction_gt_0_05": int(sum(value > 0.05 for value in max_overlaps)),
    }
