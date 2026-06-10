from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.reconstruction_3d.bind_topcam_id import bind
from scripts.reconstruction_3d.calibration import Calibration
from scripts.tracking.evaluate_3d_id_consistency import evaluate_csv

from sam2_id_experiment.src.sleap_adapter.top_track_export import export_segmentation_top_tracks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export sandwich-repaired segmentation top tracks, bind them to 3D, and compare ID metrics."
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--tracklets", required=True)
    parser.add_argument("--repair-table", required=True)
    parser.add_argument("--exp-3d-csv", required=True)
    parser.add_argument("--baseline-identified-csv", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--paths-config", default="configs/paths.local.toml")
    parser.add_argument("--calibration-model-dir", default=None)
    parser.add_argument("--bind-distance", type=float, default=90.0)
    parser.add_argument("--bind-min-frames", type=int, default=1)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=["conservative", "assertive"],
        default="conservative",
        help="Conservative excludes model/current disagreements; assertive applies them as candidate repairs.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local = load_local_paths(args.paths_config)
    model_dir = choose_path(
        args.calibration_model_dir,
        local.calibration_model_dir if local else None,
    )
    if model_dir is None:
        raise SystemExit("calibration_model_dir required via paths config or --calibration-model-dir")

    identity_tracks = materialize_repaired_identity_tracks(
        tracklets=_read_table(Path(args.tracklets)),
        repair_table=_read_table(Path(args.repair_table)),
        mode=str(args.mode),
    )
    if args.frame_start is not None:
        identity_tracks = identity_tracks[identity_tracks["frame"].astype(int) >= int(args.frame_start)].copy()
    if args.frame_end is not None:
        identity_tracks = identity_tracks[identity_tracks["frame"].astype(int) <= int(args.frame_end)].copy()

    identity_tracks_path = output_dir / f"{args.experiment}.sandwich_{args.mode}_identity_tracks.parquet"
    identity_tracks.to_parquet(identity_tracks_path, index=False)

    top_tracks_csv = output_dir / f"{args.experiment}.sandwich_{args.mode}_top_tracks.csv"
    top_summary_json = output_dir / f"{args.experiment}.sandwich_{args.mode}_top_tracks_summary.json"
    top_summary = export_segmentation_top_tracks(
        identity_tracks=identity_tracks,
        output_csv=top_tracks_csv,
        output_summary=top_summary_json,
        node_name="sandwich_segmentation_centroid",
    )

    exp_3d_csv = Path(args.exp_3d_csv)
    if args.frame_start is not None or args.frame_end is not None:
        exp_3d_subset_csv = output_dir / f"{exp_3d_csv.stem}.f{args.frame_start or 'start'}_{args.frame_end or 'end'}.csv"
        _write_frame_subset(exp_3d_csv, exp_3d_subset_csv, frame_start=args.frame_start, frame_end=args.frame_end)
        exp_3d_csv = exp_3d_subset_csv

    repaired_3d_csv = output_dir / f"{args.experiment}.sandwich_{args.mode}_3d_identified.csv"
    bind(
        exp_3d_csv=exp_3d_csv,
        top_tracks_csv=top_tracks_csv,
        calib=Calibration(str(model_dir)),
        out_csv=repaired_3d_csv,
        bind_distance=float(args.bind_distance),
        require_min_frames=int(args.bind_min_frames),
    )

    comparison = compare_metrics(
        repaired_csv=repaired_3d_csv,
        baseline_csv=Path(args.baseline_identified_csv) if args.baseline_identified_csv else None,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        output_dir=output_dir,
    )
    summary = {
        "experiment": str(args.experiment),
        "mode": str(args.mode),
        "identity_tracks": str(identity_tracks_path),
        "top_tracks_csv": str(top_tracks_csv),
        "repaired_3d_csv": str(repaired_3d_csv),
        "n_identity_track_rows": int(len(identity_tracks)),
        "top_track_summary": top_summary,
        "comparison": comparison,
    }
    summary_path = output_dir / f"{args.experiment}.sandwich_{args.mode}_bind_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def materialize_repaired_identity_tracks(
    *,
    tracklets: pd.DataFrame,
    repair_table: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    required = {"tracklet_id", "frame", "centroid_x", "centroid_y", "qc_status"}
    missing = required - set(tracklets.columns)
    if missing:
        raise ValueError(f"Tracklets missing required columns: {sorted(missing)}")
    repair_required = {"tracklet_id", "repaired_identity", "repair_action"}
    repair_missing = repair_required - set(repair_table.columns)
    if repair_missing:
        raise ValueError(f"Repair table missing required columns: {sorted(repair_missing)}")

    keep_actions = {"model_confirmed", "sandwich_filled"}
    if mode == "assertive":
        keep_actions.add("model_disagrees_current")
    repairs = repair_table[repair_table["repair_action"].fillna("").astype(str).isin(keep_actions)].copy()
    repairs = repairs[repairs["repaired_identity"].fillna("").astype(str).ne("")]
    if repairs.empty:
        return pd.DataFrame(columns=list(tracklets.columns) + ["assigned_identity", "identity_trajectory_id", "repair_action"])

    out = tracklets.merge(
        repairs[["tracklet_id", "repaired_identity", "repair_action"]],
        on="tracklet_id",
        how="inner",
    ).copy()
    out["assigned_identity"] = out["repaired_identity"].fillna("").astype(str)
    out["identity_trajectory_id"] = (
        out["assigned_identity"].astype(str)
        + "_sandwich_t"
        + out["tracklet_id"].astype(int).astype(str)
    )
    out["original_qc_status"] = out["qc_status"].fillna("").astype(str)
    fill_mask = out["repair_action"].astype(str).eq("sandwich_filled") & out["qc_status"].fillna("").astype(str).eq("reject")
    out.loc[fill_mask, "qc_status"] = "usable_inference"
    out = out[out["assigned_identity"].fillna("").astype(str).ne("")]
    return out.sort_values(["frame", "assigned_identity", "tracklet_id"]).reset_index(drop=True)


def compare_metrics(
    *,
    repaired_csv: Path,
    baseline_csv: Path | None,
    frame_start: int | None,
    frame_end: int | None,
    output_dir: Path,
) -> dict:
    rows: list[dict] = []
    if baseline_csv is not None and baseline_csv.exists():
        baseline_subset = output_dir / f"{baseline_csv.stem}.pilot_subset.csv"
        _write_frame_subset(baseline_csv, baseline_subset, frame_start=frame_start, frame_end=frame_end)
        before = evaluate_csv(baseline_subset)
        before["label"] = "baseline"
        before["csv_path"] = str(baseline_subset)
        rows.append(before)
    repaired_subset = output_dir / f"{repaired_csv.stem}.pilot_subset.csv"
    _write_frame_subset(repaired_csv, repaired_subset, frame_start=frame_start, frame_end=frame_end)
    after = evaluate_csv(repaired_subset)
    after["label"] = "sandwich_repaired"
    after["csv_path"] = str(repaired_subset)
    rows.append(after)
    table = pd.DataFrame(rows)
    comparison_csv = output_dir / f"{repaired_csv.stem}.metric_comparison.csv"
    table.to_csv(comparison_csv, index=False)
    return {
        "comparison_csv": str(comparison_csv),
        "rows": rows,
        "delta_n_spikes_gt_10.0": _delta(rows, "n_spikes_gt_10.0"),
        "delta_n_spikes_gt_20.0": _delta(rows, "n_spikes_gt_20.0"),
        "delta_xyz_speed_p99": _delta(rows, "xyz_speed_p99"),
        "delta_xyz_speed_p999": _delta(rows, "xyz_speed_p999"),
    }


def _write_frame_subset(source: Path, target: Path, *, frame_start: int | None, frame_end: int | None) -> None:
    df = pd.read_csv(source)
    if frame_start is not None:
        df = df[df["frame"].astype(int) >= int(frame_start)].copy()
    if frame_end is not None:
        df = df[df["frame"].astype(int) <= int(frame_end)].copy()
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)


def _delta(rows: list[dict], key: str) -> float | None:
    if len(rows) < 2:
        return None
    return float(rows[-1].get(key, 0.0)) - float(rows[0].get(key, 0.0))


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


if __name__ == "__main__":
    main()
