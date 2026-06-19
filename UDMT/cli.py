from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from UDMT.bidirectional import (
    BidirectionalConfig,
    CrossingConfig,
    annotate_crossing_rows,
    detect_crossing_events,
    repair_tracks_with_backward,
)
from UDMT.io import read_csv, write_jsonable_summary
from UDMT.metrics import TrackQualityConfig, score_track_table
from UDMT.refinement import RefinementConfig, refine_tracks_with_masks
from UDMT.resolver import ResolverConfig, resolve_crossings
from UDMT.tuning import candidates_from_csvs, rank_candidate_track_runs, ranking_to_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UDMT-inspired medaka identity-track utilities."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="Score one tidy top-camera track CSV.")
    score.add_argument("--tracks", required=True, type=Path)
    score.add_argument("--summary", type=Path)
    _add_quality_args(score)

    select = sub.add_parser("select", help="Rank precomputed candidate track CSVs.")
    select.add_argument("tracks", nargs="+", type=Path)
    select.add_argument("--summary", type=Path)
    select.add_argument("--table", type=Path)
    _add_quality_args(select)

    refine = sub.add_parser("refine", help="Refine track centers with mask centroids.")
    refine.add_argument("--tracks", required=True, type=Path)
    refine.add_argument("--masks", required=True, type=Path)
    refine.add_argument("--output", required=True, type=Path)
    refine.add_argument("--summary", type=Path)
    refine.add_argument("--max-distance", type=float, default=30.0)
    refine.add_argument("--min-mask-area", type=float, default=20.0)
    refine.add_argument("--max-mask-area", type=float, default=None)
    refine.add_argument("--min-quality", type=float, default=0.0)
    refine.add_argument("--blend-alpha", type=float, default=1.0)
    refine.add_argument("--strict-identity-match", action="store_true")

    repair = sub.add_parser("repair", help="Repair forward tracks with a backward pass.")
    repair.add_argument("--forward", required=True, type=Path)
    repair.add_argument("--backward", required=True, type=Path)
    repair.add_argument("--output", required=True, type=Path)
    repair.add_argument("--summary", type=Path)
    repair.add_argument("--votes", type=Path)
    repair.add_argument("--max-pair-distance", type=float, default=35.0)
    repair.add_argument("--max-anomaly-repair-distance", type=float, default=250.0)
    repair.add_argument("--max-speed", type=float, default=80.0)
    repair.add_argument("--max-speed-ratio", type=float, default=4.0)
    repair.add_argument("--min-mapping-votes", type=int, default=3)
    repair.add_argument("--consensus-blend", type=float, default=1.0)
    repair.add_argument("--repair-all-consensus", action="store_true")
    repair.add_argument(
        "--repair-crossings",
        action="store_true",
        help="Also use close-contact crossing rows as backward-consensus repair triggers.",
    )
    repair.add_argument("--crossing-distance", type=float, default=35.0)
    repair.add_argument("--crossing-min-duration", type=int, default=1)
    repair.add_argument("--drop-unrepaired-anomalies", action="store_true")

    crossings = sub.add_parser(
        "crossings",
        help="Detect smooth close-contact crossing events in 2D or 3D tracks.",
    )
    crossings.add_argument("--tracks", required=True, type=Path)
    crossings.add_argument("--events", required=True, type=Path)
    crossings.add_argument("--annotated", type=Path)
    crossings.add_argument("--id-column", default="track_id")
    crossings.add_argument(
        "--coordinate-columns",
        nargs="+",
        default=["x", "y"],
        help="Coordinate columns, e.g. x y for top-cam or X Y Z for 3D.",
    )
    crossings.add_argument("--distance-threshold", type=float, default=35.0)
    crossings.add_argument("--min-duration", type=int, default=1)
    crossings.add_argument("--merge-gap", type=int, default=0)
    crossings.add_argument("--dtw-window", type=int, default=20)
    crossings.add_argument("--dtw-threshold", type=float, default=None)
    crossings.add_argument(
        "--review-max-min-distance",
        type=float,
        default=None,
        help="Mark events whose minimum distance is at or below this value.",
    )
    crossings.add_argument(
        "--review-min-duration",
        type=int,
        default=None,
        help="Mark events lasting at least this many frames.",
    )
    crossings.add_argument(
        "--review-only",
        action="store_true",
        help="Write only events that pass the optional review gates.",
    )

    resolve = sub.add_parser(
        "resolve",
        help="Read-only: score crossing events with medaka cues -> suspect/review flag.",
    )
    resolve.add_argument("--events", required=True, type=Path)
    resolve.add_argument("--tracks-3d", required=True, type=Path)
    resolve.add_argument("--identified", type=Path, help="*_3d_identified.csv for bind_score cue.")
    resolve.add_argument("--pre-link", type=Path, help="*_best_pre_link.json for per-fish 3D reprojection-residual cue.")
    resolve.add_argument("--output", required=True, type=Path)
    resolve.add_argument("--summary", type=Path)
    resolve.add_argument("--id-column", default="fish_id")
    resolve.add_argument("--coordinate-columns", nargs="+", default=["X", "Y", "Z"])
    resolve.add_argument("--window-pad", type=int, default=30)
    resolve.add_argument("--margin-threshold", type=float, default=0.34)
    resolve.add_argument(
        "--review-only",
        action="store_true",
        dest="resolve_review_only",
        help="Score only events flagged review_candidate in the events table.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "score":
        report = score_track_table(read_csv(args.tracks), config=_quality_config(args))
        _emit_summary(report.to_dict(), args.summary)
        return

    if args.command == "select":
        ranking = rank_candidate_track_runs(
            candidates_from_csvs(args.tracks),
            config=_quality_config(args),
        )
        _emit_summary({"candidates": ranking}, args.summary)
        if args.table is not None:
            args.table.parent.mkdir(parents=True, exist_ok=True)
            ranking_to_frame(ranking).to_csv(args.table, index=False)
        return

    if args.command == "refine":
        refined, summary = refine_tracks_with_masks(
            read_csv(args.tracks),
            read_csv(args.masks),
            config=RefinementConfig(
                max_distance_px=args.max_distance,
                min_mask_area=args.min_mask_area,
                max_mask_area=args.max_mask_area,
                min_quality=args.min_quality,
                blend_alpha=args.blend_alpha,
                strict_identity_match=args.strict_identity_match,
            ),
        )
        _write_csv(refined, args.output)
        _emit_summary(summary.to_dict(), args.summary)
        return

    if args.command == "repair":
        repaired, summary, votes = repair_tracks_with_backward(
            read_csv(args.forward),
            read_csv(args.backward),
            config=BidirectionalConfig(
                max_pair_distance_px=args.max_pair_distance,
                max_anomaly_repair_distance_px=args.max_anomaly_repair_distance,
                max_speed_px_per_frame=args.max_speed,
                max_speed_ratio=args.max_speed_ratio,
                min_mapping_votes=args.min_mapping_votes,
                consensus_blend=args.consensus_blend,
                repair_anomalies_only=not args.repair_all_consensus,
                repair_crossings=args.repair_crossings,
                crossing_distance_px=args.crossing_distance,
                crossing_min_duration_frames=args.crossing_min_duration,
                drop_unrepaired_anomalies=args.drop_unrepaired_anomalies,
            ),
        )
        _write_csv(repaired, args.output)
        if args.votes is not None:
            _write_csv(votes, args.votes)
        _emit_summary(summary.to_dict(), args.summary)
        return

    if args.command == "crossings":
        cfg = CrossingConfig(
            id_column=args.id_column,
            coordinate_columns=tuple(args.coordinate_columns),
            distance_threshold=args.distance_threshold,
            min_duration_frames=args.min_duration,
            merge_gap_frames=args.merge_gap,
            dtw_window_frames=args.dtw_window,
            dtw_threshold=args.dtw_threshold,
            review_max_min_distance=args.review_max_min_distance,
            review_min_duration_frames=args.review_min_duration,
        )
        tracks = read_csv(args.tracks)
        events = detect_crossing_events(tracks, cfg)
        n_detected = int(len(events))
        n_review_candidates = (
            int(events["review_candidate"].fillna(False).sum())
            if "review_candidate" in events.columns
            else 0
        )
        if args.review_only:
            if "review_candidate" in events.columns:
                events = events[events["review_candidate"].fillna(False)].copy()
            else:
                events = events.iloc[0:0].copy()
        _write_csv(events, args.events)
        if args.annotated is not None:
            _write_csv(annotate_crossing_rows(tracks, events, cfg), args.annotated)
        _emit_summary(
            {
                "n_detected_events": n_detected,
                "n_review_candidates": n_review_candidates,
                "n_written_events": int(len(events)),
                "review_only": bool(args.review_only),
                "events_csv": str(args.events),
                "annotated_csv": None if args.annotated is None else str(args.annotated),
            },
            None,
        )
        return

    if args.command == "resolve":
        identified = read_csv(args.identified) if args.identified is not None else None
        decisions, summary = resolve_crossings(
            read_csv(args.events),
            read_csv(args.tracks_3d),
            identified=identified,
            pre_link=args.pre_link,
            config=ResolverConfig(
                id_column=args.id_column,
                coordinate_columns=tuple(args.coordinate_columns),
                window_pad_frames=args.window_pad,
                margin_threshold=args.margin_threshold,
                review_only=args.resolve_review_only,
            ),
        )
        _write_csv(decisions, args.output)
        _emit_summary(summary.to_dict(), args.summary)
        return

    raise SystemExit(f"Unknown command: {args.command}")


def _add_quality_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-individuals", type=int, default=3)
    parser.add_argument("--max-speed", type=float, default=80.0)
    parser.add_argument("--max-speed-ratio", type=float, default=4.0)
    parser.add_argument("--close-contact-distance", type=float, default=35.0)


def _quality_config(args: argparse.Namespace) -> TrackQualityConfig:
    return TrackQualityConfig(
        num_individuals=args.num_individuals,
        max_speed_px_per_frame=args.max_speed,
        max_speed_ratio=args.max_speed_ratio,
        close_contact_distance_px=args.close_contact_distance,
    )


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _emit_summary(summary: dict, path: Path | None) -> None:
    if path is not None:
        write_jsonable_summary(summary, path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
