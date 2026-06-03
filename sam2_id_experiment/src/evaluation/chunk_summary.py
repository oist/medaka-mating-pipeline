from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_chunk_summary(*, output_dir: str | Path, summary_path: str | Path | None = None) -> dict:
    out_dir = Path(output_dir)
    summary = {
        "output_dir": str(out_dir),
        "qc": _qc_summary(out_dir),
        "id_dataset": _dataset_summary(out_dir),
        "id_assignment": _assignment_summary(out_dir),
    }
    path = Path(summary_path) if summary_path is not None else out_dir / "chunk_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _qc_summary(out_dir: Path) -> dict:
    mask_qc = out_dir / "mask_qc_frames.csv"
    tracklets = out_dir / "tracklet_summary_qc.parquet"
    if not mask_qc.exists():
        return {"available": False}
    frames = pd.read_csv(mask_qc)
    summary = {
        "available": True,
        "n_mask_instances": int(len(frames)),
        "n_frames": int(frames["frame"].nunique()) if "frame" in frames else 0,
        "status_counts": _count_dict(frames["qc_status"]) if "qc_status" in frames else {},
        "status_counts_by_tracklet": _group_count_dict(frames, ["tracklet_id", "qc_status"]),
        "auto_qc_reason_counts": _count_dict(frames["auto_qc_reason"].fillna("")) if "auto_qc_reason" in frames else {},
    }
    if tracklets.exists():
        tdf = pd.read_parquet(tracklets)
        summary["tracklet_summary"] = tdf.to_dict("records")
    return summary


def _dataset_summary(out_dir: Path) -> dict:
    dataset = out_dir / "id_training_dataset"
    meta_path = dataset / "crop_metadata.parquet"
    pairs_path = dataset / "pairs.parquet"
    if not meta_path.exists():
        return {"available": False}
    meta = pd.read_parquet(meta_path)
    out = {
        "available": True,
        "n_crops": int(len(meta)),
        "crops_by_tracklet": _count_dict(meta["tracklet_id"]),
    }
    if pairs_path.exists():
        pairs = pd.read_parquet(pairs_path)
        out["n_pairs"] = int(len(pairs))
        out["pairs_by_label"] = _count_dict(pairs["label"])
    return out


def _assignment_summary(out_dir: Path) -> dict:
    inference = out_dir / "id_inference"
    assignments_path = inference / "tracklet_identity_assignments.parquet"
    cross_summary_path = inference / "cross_chunk_assignment_summary.json"
    if not assignments_path.exists():
        return {"available": False}
    assignments = pd.read_parquet(assignments_path)
    out = {
        "available": True,
        "n_tracklets": int(len(assignments)),
        "assigned_identity_counts": _count_dict(assignments["assigned_identity"])
        if "assigned_identity" in assignments
        else {},
    }
    if "assignment_status" in assignments:
        out["assignment_status_counts"] = _count_dict(assignments["assignment_status"])
    if cross_summary_path.exists():
        out["cross_chunk_summary"] = json.loads(cross_summary_path.read_text(encoding="utf-8"))
    return out


def _count_dict(series: pd.Series) -> dict:
    return {str(k): int(v) for k, v in series.value_counts().to_dict().items()}


def _group_count_dict(df: pd.DataFrame, columns: list[str]) -> dict:
    if df.empty:
        return {}
    grouped = df.groupby(columns).size()
    return {"|".join(str(part) for part in key): int(value) for key, value in grouped.to_dict().items()}
