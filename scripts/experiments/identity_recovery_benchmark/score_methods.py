from __future__ import annotations

"""Score identity trackers on one locked historical-video interval.

Example::

    python -m scripts.experiments.identity_recovery_benchmark.score_methods \
      --reference top_cam.gt_reprojected.csv --start-frame 176300 \
      --end-frame 181298 --idtracker-tidy trajectories_tidy.csv \
      --udmt-npy top_cam_trained.npy --udmt-resize-factor 0.8 --output-dir runs/...
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    CrossingEvent,
    crossing_proxy_metrics,
    derive_crossing_events,
    geometry_proxy_identity_metrics,
    load_idtracker_tidy,
    load_reference,
    load_tracks_csv,
    load_udmt_npy,
    localization_metrics,
)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _table_quality(table: pd.DataFrame, *, id_column: str) -> dict[str, object]:
    counts = table.groupby("frame")[id_column].nunique()
    duplicate_rows = int(table.duplicated(["frame", id_column]).sum())
    return {
        "rows": int(len(table)),
        "frame_min": int(table["frame"].min()) if len(table) else None,
        "frame_max": int(table["frame"].max()) if len(table) else None,
        "frames": int(table["frame"].nunique()),
        "distinct_ids": int(table[id_column].nunique()),
        "duplicate_frame_id_rows": duplicate_rows,
        "median_ids_per_frame": float(counts.median()) if len(counts) else None,
        "minimum_ids_per_frame": int(counts.min()) if len(counts) else None,
        "maximum_ids_per_frame": int(counts.max()) if len(counts) else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--idtracker-tidy", type=Path)
    parser.add_argument("--udmt-npy", type=Path)
    parser.add_argument("--udmt-resize-factor", type=float, default=0.8)
    parser.add_argument(
        "--multiview-tracks",
        type=Path,
        help="CSV with frame,track_id,x,y from the multi-view constrained method",
    )
    parser.add_argument("--event-threshold-px", type=float, default=35.0)
    parser.add_argument("--event-merge-gap", type=int, default=8)
    parser.add_argument("--event-min-close-frames", type=int, default=2)
    parser.add_argument("--event-context-frames", type=int, default=83)
    parser.add_argument("--localization-gates", type=float, nargs="+", default=(40.0, 60.0, 80.0))
    parser.add_argument("--proxy-gate-px", type=float, default=80.0)
    return parser


def _load_methods(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    methods: dict[str, pd.DataFrame] = {}
    if args.idtracker_tidy:
        methods["idtracker_ai_v6"] = load_idtracker_tidy(
            args.idtracker_tidy, frame_offset=args.start_frame
        )
    if args.udmt_npy:
        methods["native_udmt"] = load_udmt_npy(
            args.udmt_npy,
            frame_offset=args.start_frame,
            resize_factor=args.udmt_resize_factor,
        )
    if args.multiview_tracks:
        methods["multiview_constrained_contrastive"] = load_tracks_csv(
            args.multiview_tracks
        )
    if not methods:
        raise ValueError("at least one method output must be supplied")
    return methods


def score_method(
    name: str,
    tracks: pd.DataFrame,
    reference: pd.DataFrame,
    events: list[CrossingEvent],
    args: argparse.Namespace,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    localization = {
        str(int(gate) if float(gate).is_integer() else gate): localization_metrics(
            tracks,
            reference,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            gate_px=float(gate),
        )
        for gate in args.localization_gates
    }
    proxy, matched = geometry_proxy_identity_metrics(
        tracks,
        reference,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        gate_px=args.proxy_gate_px,
    )
    crossing = crossing_proxy_metrics(
        matched,
        events,
        proxy["global_track_to_proxy_fish"],
    )
    crossing.insert(0, "method", name)
    report = {
        "method": name,
        "interval": [args.start_frame, args.end_frame],
        "n_track_rows": int(len(tracks)),
        "n_distinct_tracks": int(tracks["track_id"].nunique()),
        "localization_identity_blind": localization,
        "identity_geometry_proxy_only": proxy,
        "crossing_geometry_proxy_only": {
            "n_fish_event_checks": int(len(crossing)),
            "pass_fraction": float(crossing["proxy_continuity_pass"].mean())
            if len(crossing)
            else float("nan"),
        },
    }
    return report, matched.assign(method=name), crossing


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_reference(args.reference)
    methods = _load_methods(args)
    events = derive_crossing_events(
        reference,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        distance_threshold_px=args.event_threshold_px,
        merge_gap_frames=args.event_merge_gap,
        min_close_frames=args.event_min_close_frames,
        required_context_frames=args.event_context_frames,
    )
    event_table = pd.DataFrame([event.to_dict() for event in events])
    event_table.to_csv(args.output_dir / "crossing_events.csv", index=False)

    reports: dict[str, object] = {}
    all_matches: list[pd.DataFrame] = []
    all_crossings: list[pd.DataFrame] = []
    for name, tracks in methods.items():
        report, matched, crossing = score_method(name, tracks, reference, events, args)
        reports[name] = report
        all_matches.append(matched)
        all_crossings.append(crossing)
        tracks.to_csv(args.output_dir / f"{name}.tracks.csv", index=False)

    if all_matches:
        pd.concat(all_matches, ignore_index=True).to_csv(
            args.output_dir / "per_position_proxy_matches.csv", index=False
        )
    if all_crossings:
        pd.concat(all_crossings, ignore_index=True).to_csv(
            args.output_dir / "crossing_proxy_diagnostics.csv", index=False
        )

    summary = {
        "schema_version": "identity_recovery_benchmark_v1",
        "claim_under_test": (
            "biological identity can be recovered automatically from internal visual "
            "and temporal structure in historical recordings"
        ),
        "primary_truth": "pending independent blinded human crossing audit",
        "reference_role": (
            "unordered localization/event selection plus explicitly provisional "
            "geometry-label proxy; not biological identity ground truth"
        ),
        "reference": str(args.reference.resolve()),
        "input_provenance": {
            "reference": {
                "path": str(args.reference.resolve()),
                "sha256": _sha256(args.reference),
            },
            "idtracker_tidy": (
                {"path": str(args.idtracker_tidy.resolve()), "sha256": _sha256(args.idtracker_tidy)}
                if args.idtracker_tidy
                else None
            ),
            "native_udmt": (
                {"path": str(args.udmt_npy.resolve()), "sha256": _sha256(args.udmt_npy)}
                if args.udmt_npy
                else None
            ),
            "multiview_tracks": (
                {
                    "path": str(args.multiview_tracks.resolve()),
                    "sha256": _sha256(args.multiview_tracks),
                }
                if args.multiview_tracks
                else None
            ),
        },
        "data_quality": {
            "reference": _table_quality(reference, id_column="fish_id"),
            "methods": {
                name: _table_quality(table, id_column="track_id")
                for name, table in methods.items()
            },
        },
        "interval": [args.start_frame, args.end_frame],
        "event_selection": {
            "distance_threshold_px": args.event_threshold_px,
            "merge_gap_frames": args.event_merge_gap,
            "minimum_close_frames": args.event_min_close_frames,
            "required_context_frames": args.event_context_frames,
            "n_events": len(events),
        },
        "methods": reports,
    }
    (args.output_dir / "benchmark_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    summary = run(build_parser().parse_args(argv))
    print(json.dumps(_json_safe(summary), indent=2))


if __name__ == "__main__":
    main()
