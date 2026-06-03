from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare mouth-anchor tracking against an experimental anchor mode."
    )
    parser.add_argument("--mode-a-label", required=True)
    parser.add_argument("--mode-a-tracks", required=True)
    parser.add_argument("--mode-a-identified-3d", default=None)
    parser.add_argument("--mode-b-label", required=True)
    parser.add_argument("--mode-b-tracks", required=True)
    parser.add_argument("--mode-b-identified-3d", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = [
        _mode_summary(
            label=args.mode_a_label,
            tracks_path=Path(args.mode_a_tracks),
            identified_3d_path=Path(args.mode_a_identified_3d) if args.mode_a_identified_3d else None,
        ),
        _mode_summary(
            label=args.mode_b_label,
            tracks_path=Path(args.mode_b_tracks),
            identified_3d_path=Path(args.mode_b_identified_3d) if args.mode_b_identified_3d else None,
        ),
    ]
    report = {"modes": rows}
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    flat = [_flatten(row) for row in rows]
    output_csv = Path(args.output_csv) if args.output_csv else output_json.with_suffix(".csv")
    pd.DataFrame(flat).to_csv(output_csv, index=False)
    print(json.dumps(report, indent=2))


def _mode_summary(
    *,
    label: str,
    tracks_path: Path,
    identified_3d_path: Path | None,
) -> dict:
    tracks = pd.read_csv(tracks_path)
    summary = {
        "label": label,
        "tracks": str(tracks_path),
        "top_rows": int(len(tracks)),
        "top_frames": int(tracks["frame"].nunique()) if "frame" in tracks else 0,
        "top_rows_by_label": _value_counts(tracks, "track_label"),
        "tracking_source_counts": _value_counts(tracks, "tracking_source"),
        "top_gap_count_by_label": _gap_counts(tracks, max_gap=False),
        "top_max_gap_by_label": _gap_counts(tracks, max_gap=True),
    }
    if identified_3d_path is not None:
        identified = pd.read_csv(identified_3d_path)
        summary.update({
            "identified_3d": str(identified_3d_path),
            "3d_rows": int(len(identified)),
            "3d_top_bind_rows": int(identified["bind_source"].eq("top_cam_2d").sum()) if "bind_source" in identified else 0,
            "3d_top_bind_pct": float(identified["bind_source"].eq("top_cam_2d").mean() * 100.0) if "bind_source" in identified else np.nan,
            "3d_geometry_only_rows": int(identified["bind_source"].eq("geometry_only").sum()) if "bind_source" in identified else 0,
            "3d_mapping": _mapping(identified),
            **_step_stats(identified),
        })
    return summary


def _value_counts(df: pd.DataFrame, column: str) -> dict:
    if column not in df:
        return {}
    return {str(k): int(v) for k, v in df[column].value_counts(dropna=False).items()}


def _gap_counts(tracks: pd.DataFrame, *, max_gap: bool) -> dict:
    if "track_label" not in tracks or "frame" not in tracks:
        return {}
    out = {}
    for label, group in tracks.sort_values(["track_label", "frame"]).groupby("track_label"):
        frames = np.sort(group["frame"].astype(int).unique())
        gaps = np.diff(frames)
        if max_gap:
            out[str(label)] = int(gaps.max()) if len(gaps) else 0
        else:
            out[str(label)] = int((gaps > 1).sum())
    return out


def _mapping(identified: pd.DataFrame) -> dict:
    if not {"fish_id", "final_fish_id"}.issubset(identified.columns):
        return {}
    pairs = identified[["fish_id", "final_fish_id"]].dropna().drop_duplicates()
    return {str(fish_id): str(final_id) for fish_id, final_id in pairs.itertuples(index=False)}


def _step_stats(identified: pd.DataFrame) -> dict:
    if not {"fish_id", "frame", "X", "Y", "Z"}.issubset(identified.columns):
        return {}
    per_id = []
    for fish_id, group in identified.sort_values(["fish_id", "frame"]).groupby("fish_id"):
        frames = group["frame"].astype(int).to_numpy()
        xyz = group[["X", "Y", "Z"]].astype(float).to_numpy()
        steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        gaps = np.diff(frames)
        valid = np.isfinite(steps) & (gaps == 1)
        values = steps[valid]
        if len(values) == 0:
            continue
        per_id.append({
            "fish_id": str(fish_id),
            "step_p95": float(np.nanpercentile(values, 95)),
            "step_p99": float(np.nanpercentile(values, 99)),
            "step_max": float(np.nanmax(values)),
            "step_gt_8mm": int((values > 8.0).sum()),
            "step_gt_20mm": int((values > 20.0).sum()),
        })
    if not per_id:
        return {}
    return {
        "3d_step_stats_by_id": per_id,
        "3d_step_p95_max_across_ids": float(max(row["step_p95"] for row in per_id)),
        "3d_step_p99_max_across_ids": float(max(row["step_p99"] for row in per_id)),
        "3d_step_max_across_ids": float(max(row["step_max"] for row in per_id)),
        "3d_step_gt_8mm_total": int(sum(row["step_gt_8mm"] for row in per_id)),
        "3d_step_gt_20mm_total": int(sum(row["step_gt_20mm"] for row in per_id)),
    }


def _flatten(row: dict) -> dict:
    return {key: value for key, value in row.items() if not isinstance(value, (dict, list))}


if __name__ == "__main__":
    main()
