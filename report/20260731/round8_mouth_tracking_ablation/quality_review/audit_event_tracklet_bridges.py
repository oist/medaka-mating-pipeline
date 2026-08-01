from __future__ import annotations

"""Audit reciprocal per-view bridges at the critical Round8 transition."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from scripts.reconstruction_3d.candidate_cache import load_candidate_window
from scripts.reconstruction_3d.conservative_mouth_tracklets import (
    ConservativeTrackletConfig,
    build_reciprocal_tracklets,
    frozen_body_length_px_from_training_controls,
)


VIEWS = ("top_cam", "side1_cam", "side2_cam")


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cohort-protocol", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    protocol = json.loads(args.cohort_protocol.read_text(encoding="utf-8"))
    training_controls = [
        tuple(map(int, window["frames"]))
        for window in protocol["training"]["matched_control_windows"]
    ]
    sweep = protocol["tracklet_cutoff_sweep"]
    records: list[dict[str, object]] = []
    body_lengths: dict[str, float] = {}
    for view in VIEWS:
        cache_path = args.cache_root / f"{view}.candidates.h5"
        control_windows = [
            load_candidate_window(cache_path, start, end)
            for start, end in training_controls
        ]
        body_length = frozen_body_length_px_from_training_controls(
            control_windows
        )
        body_lengths[view] = body_length
        event = load_candidate_window(cache_path, 34000, 35100)
        for separation, margin, gap in itertools.product(
            sweep["separation_body_lengths"],
            sweep["assignment_ambiguity_margin_body_lengths"],
            sweep["maximum_missing_gap_frames"],
        ):
            config = ConservativeTrackletConfig(
                separation_body_lengths=float(separation),
                assignment_ambiguity_margin_body_lengths=float(margin),
                maximum_missing_gap_frames=int(gap),
                uncertainty_growth_body_lengths_per_missing_frame=float(
                    sweep[
                        "uncertainty_growth_body_lengths_per_missing_frame"
                    ]
                ),
            )
            result = build_reciprocal_tracklets(
                event,
                config=config,
                frozen_body_length_px=body_length,
            )
            frame_rows: dict[int, pd.DataFrame] = {
                frame: result.dispositions[result.dispositions.frame.eq(frame)]
                for frame in (34547, 34548)
            }
            selected_slots: dict[int, int] = {}
            for frame in (34547, 34548):
                local = frame - event.frame_start
                finite = np.flatnonzero(
                    np.isfinite(event.mouth_xy[local]).all(axis=1)
                )
                if view == "top_cam":
                    matching = finite[
                        event.source_instance[local, finite] == 1
                    ]
                    if len(matching) != 1:
                        raise AssertionError(
                            f"{view} frame {frame}: expected source instance 1"
                        )
                    selected_slots[frame] = int(matching[0])
                else:
                    lower = finite[
                        event.mouth_xy[local, finite, 1] > 700.0
                    ]
                    if len(lower) != 1:
                        raise AssertionError(
                            f"{view} frame {frame}: expected one lower fused pose"
                        )
                    selected_slots[frame] = int(lower[0])
            selected = []
            for frame in (34547, 34548):
                slot = selected_slots[frame]
                row = frame_rows[frame][
                    frame_rows[frame].candidate_slot.eq(slot)
                ]
                if len(row) != 1:
                    raise AssertionError(
                        f"{view} frame {frame} slot {slot}: missing disposition"
                    )
                selected.append(row.iloc[0])
            left_id, right_id = map(int, (selected[0].tracklet_id, selected[1].tracklet_id))
            exact_bridge = result.bridges[
                result.bridges.source_frame.eq(34547)
                & result.bridges.source_slot.eq(selected_slots[34547])
                & result.bridges.target_frame.eq(34548)
                & result.bridges.target_slot.eq(selected_slots[34548])
            ]
            records.append(
                {
                    "view": view,
                    "separation_body_lengths": float(separation),
                    "assignment_margin_body_lengths": float(margin),
                    "maximum_missing_gap_frames": int(gap),
                    "frozen_body_length_px": body_length,
                    "slot_34547": selected_slots[34547],
                    "slot_34548": selected_slots[34548],
                    "disposition_34547": str(selected[0].disposition),
                    "disposition_34548": str(selected[1].disposition),
                    "tracklet_id_34547": left_id,
                    "tracklet_id_34548": right_id,
                    "same_retained_tracklet": bool(
                        left_id >= 0 and left_id == right_id
                    ),
                    "exact_reciprocal_bridge": bool(
                        len(exact_bridge)
                        and exact_bridge.reciprocal_accepted.astype(bool).any()
                    ),
                    "total_fragment_count": int(
                        result.summary["fragment_count"]
                    ),
                    "retained_coverage": float(
                        result.summary["retained_coverage"]
                    ),
                }
            )
    table = pd.DataFrame.from_records(records)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    summary = {
        "schema_version": 1,
        "created_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
        "command": [sys.executable, *sys.argv],
        "protocol_deviation": "pre-freeze critical-event safety analysis",
        "analysis_role": (
            "post_hoc_descriptive implementation audit only; never used for "
            "threshold selection and not an untouched confirmatory result"
        ),
        "parameter_selection_impact": "none",
        "critical_event_opened_before_final_threshold_freeze": True,
        "held_out_opened_by_this_audit": False,
        "cache_root": str(args.cache_root),
        "input_hashes": {
            "cohort_protocol": sha256(args.cohort_protocol),
            "audit_script": sha256(Path(__file__)),
            "cache_h5": {
                view: sha256(
                    args.cache_root / f"{view}.candidates.h5"
                )
                for view in VIEWS
            },
            "detail_csv": sha256(args.output_csv),
        },
        "configuration_universe": {
            "separation_body_lengths": list(
                sweep["separation_body_lengths"]
            ),
            "assignment_ambiguity_margin_body_lengths": list(
                sweep["assignment_ambiguity_margin_body_lengths"]
            ),
            "maximum_missing_gap_frames": list(
                sweep["maximum_missing_gap_frames"]
            ),
            "uncertainty_growth_body_lengths_per_missing_frame": float(
                sweep[
                    "uncertainty_growth_body_lengths_per_missing_frame"
                ]
            ),
        },
        "body_length_source": "training matched controls only",
        "frozen_body_length_px": body_lengths,
        "selected_observation": {
            "top_cam": "raw source instance 1 (stable mouth / lost tail)",
            "side1_cam": "only detected lower-region pose for the close pair",
            "side2_cam": "only detected lower-region pose for the close pair",
        },
        "configurations": int(len(table)),
        "same_retained_tracklet_counts": {
            view: int(
                table.loc[
                    table.view.eq(view), "same_retained_tracklet"
                ].sum()
            )
            for view in VIEWS
        },
        "exact_reciprocal_bridge_counts": {
            view: int(
                table.loc[
                    table.view.eq(view), "exact_reciprocal_bridge"
                ].sum()
            )
            for view in VIEWS
        },
        "interpretation_constraint": (
            "The accepted side-view edge is continuity of one fused close-pair "
            "detection, not a true biological-fish bridge. It must not be used "
            "alone as a high-confidence A5 identity constraint. Any reciprocal "
            "2D edge is only a purity proxy and must still survive cross-view "
            "mutual exclusion and 3D motion constraints."
        ),
    }
    args.output_json.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
