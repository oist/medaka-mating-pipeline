from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


QC_BIND_WEIGHTS = {
    "clean_train": 1.0,
    "usable_inference": 0.5,
    "manual_review": 0.1,
    "reject": 0.0,
}


def export_segmentation_top_tracks(
    *,
    identity_tracks: pd.DataFrame,
    output_csv: str | Path,
    output_summary: str | Path | None = None,
    node_name: str = "segmentation_centroid",
) -> dict:
    if identity_tracks.empty:
        out = _empty_export()
    else:
        required = {"frame", "centroid_x", "centroid_y", "identity_trajectory_id", "tracklet_id", "qc_status"}
        missing = required - set(identity_tracks.columns)
        if missing:
            raise ValueError(f"Identity tracks missing columns: {', '.join(sorted(missing))}")
        tracks = identity_tracks.copy()
        trajectory_ids = sorted(str(v) for v in tracks["identity_trajectory_id"].dropna().unique())
        id_map = {trajectory_id: idx for idx, trajectory_id in enumerate(trajectory_ids)}
        track_label = tracks.get("assigned_identity", tracks["identity_trajectory_id"]).fillna("").astype(str)
        out = pd.DataFrame({
            "frame": tracks["frame"].astype(int),
            "track_id": tracks["identity_trajectory_id"].astype(str).map(id_map).astype(int),
            "x": tracks["centroid_x"].astype(float),
            "y": tracks["centroid_y"].astype(float),
            "score": tracks.get("confidence", pd.Series(1.0, index=tracks.index)).astype(float),
            "node": str(node_name),
            "track_label": track_label,
            "identity": track_label,
            "identity_trajectory_id": tracks["identity_trajectory_id"].astype(str),
            "source_tracklet_id": tracks["tracklet_id"].astype(int),
            "qc_status": tracks["qc_status"].fillna("").astype(str),
        })
        out["bind_weight"] = out["qc_status"].map(QC_BIND_WEIGHTS).fillna(0.25).astype(float)
        out.sort_values(["frame", "track_id", "source_tracklet_id"], inplace=True)

    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    summary = {
        "output_csv": str(path),
        "n_rows": int(len(out)),
        "n_tracks": int(out["track_id"].nunique()) if not out.empty else 0,
        "track_label_counts": {str(k): int(v) for k, v in out["track_label"].value_counts().to_dict().items()} if not out.empty else {},
        "qc_status_counts": {str(k): int(v) for k, v in out["qc_status"].value_counts().to_dict().items()} if not out.empty else {},
    }
    if output_summary is not None:
        summary_path = Path(output_summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _empty_export() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "frame", "track_id", "x", "y", "score", "node", "track_label",
        "identity", "identity_trajectory_id", "source_tracklet_id", "qc_status", "bind_weight",
    ])
