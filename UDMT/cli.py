from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from UDMT.bidirectional import BidirectionalConfig, repair_tracks_with_backward
from UDMT.io import read_csv, write_jsonable_summary
from UDMT.metrics import TrackQualityConfig, score_track_table
from UDMT.refinement import RefinementConfig, refine_tracks_with_masks
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
    repair.add_argument("--drop-unrepaired-anomalies", action="store_true")
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
                drop_unrepaired_anomalies=args.drop_unrepaired_anomalies,
            ),
        )
        _write_csv(repaired, args.output)
        if args.votes is not None:
            _write_csv(votes, args.votes)
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
