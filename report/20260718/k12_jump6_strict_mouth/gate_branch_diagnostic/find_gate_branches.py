"""Find temporal-ID branches caused by lowering the 3D jump gate to 6 mm.

The paired inputs must contain the same unlabeled direct 3D observations. A
branch is a change in the per-frame permutation mapping 6 mm IDs onto 8 mm IDs.
Its trigger is verified when the observation expected under the previous
permutation is outside the effective 6 mm gate but inside the 8 mm gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


IDS = ("ID_1", "ID_2", "ID_3")


def _load_track(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(path, low_memory=False).sort_values(["frame", "fish_id"])
    if table.duplicated(["frame", "fish_id"]).any():
        raise ValueError(f"duplicate frame/fish_id rows in {path}")
    if tuple(sorted(table["fish_id"].unique())) != IDS:
        raise ValueError(f"expected {IDS} in {path}")
    frame_count = int(table["frame"].max()) + 1
    expected_rows = frame_count * len(IDS)
    if len(table) != expected_rows:
        raise ValueError(f"expected {expected_rows} dense rows in {path}, got {len(table)}")
    xyz = table[["X", "Y", "Z"]].to_numpy(float).reshape(frame_count, 3, 3)
    interpolated = table["is_interpolated"].astype(bool).to_numpy().reshape(frame_count, 3)
    observed = np.isfinite(xyz).all(axis=2) & ~interpolated
    return xyz, observed


def _mapping_label(state: tuple[int, ...]) -> str:
    return ";".join(f"{IDS[index]}->{IDS[target]}" for index, target in enumerate(state))


def _common_states(
    xyz6: np.ndarray,
    observed6: np.ndarray,
    xyz8: np.ndarray,
    observed8: np.ndarray,
    *,
    tolerance_mm: float,
) -> list[tuple[int, tuple[int, ...]]]:
    states: list[tuple[int, tuple[int, ...]]] = []
    common = np.flatnonzero(observed6.all(axis=1) & observed8.all(axis=1))
    for frame in common:
        distances = np.linalg.norm(
            xyz6[frame, :, None, :] - xyz8[frame, None, :, :], axis=2
        )
        state = tuple(int(value) for value in np.argmin(distances, axis=1))
        if len(set(state)) != 3:
            raise ValueError(f"non-unique point mapping at frame {frame}")
        if max(distances[index, target] for index, target in enumerate(state)) > tolerance_mm:
            raise ValueError(f"paired point sets disagree at frame {frame}")
        states.append((int(frame), state))
    return states


def _state_runs(
    samples: list[tuple[int, tuple[int, ...]]]
) -> list[dict[str, object]]:
    if not samples:
        return []
    start = samples[0][0]
    previous_frame = start
    state = samples[0][1]
    runs: list[dict[str, object]] = []
    for frame, current_state in samples[1:]:
        if current_state != state:
            runs.append({"start": start, "end": previous_frame, "state": state})
            start = frame
            state = current_state
        previous_frame = frame
    runs.append({"start": start, "end": previous_frame, "state": state})
    return runs


def _confirmed_female_frames(path: Path | None) -> np.ndarray:
    if path is None or not path.exists():
        return np.empty(0, dtype=np.int64)
    table = pd.read_csv(path, low_memory=False)
    keep = np.ones(len(table), dtype=bool)
    if "gt_status" in table:
        keep &= table["gt_status"].astype(str).str.casefold().eq("confirmed").to_numpy()
    if "sex_label" in table:
        keep &= table["sex_label"].astype(str).str.casefold().eq("f").to_numpy()
    return np.sort(table.loc[keep, "frame"].astype(np.int64).unique())


def _annotation_context(
    frames: np.ndarray, start: int, end: int, trigger: int
) -> dict[str, int | None]:
    in_window = frames[(frames >= start) & (frames <= end)]
    before = frames[frames <= trigger]
    after = frames[frames >= trigger]
    previous = int(before[-1]) if len(before) else None
    following = int(after[0]) if len(after) else None
    return {
        "existing_confirmed_female_annotations": int(len(in_window)),
        "nearest_annotation_before": previous,
        "frames_from_previous_annotation": trigger - previous if previous is not None else None,
        "nearest_annotation_after": following,
        "frames_to_next_annotation": following - trigger if following is not None else None,
    }


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    xyz6, observed6 = _load_track(args.jump6_csv)
    xyz8, observed8 = _load_track(args.jump8_csv)
    if xyz6.shape != xyz8.shape:
        raise ValueError("paired track shapes differ")
    frame_count = len(xyz6)
    samples = _common_states(
        xyz6, observed6, xyz8, observed8, tolerance_mm=args.tolerance_mm
    )
    runs = _state_runs(samples)
    female_frames = _confirmed_female_frames(args.annotations_csv)

    transition_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for run_index in range(1, len(runs)):
        previous = runs[run_index - 1]
        current = runs[run_index]
        previous_frame = int(previous["end"])
        next_frame = int(current["start"])
        previous_state = tuple(previous["state"])
        current_state = tuple(current["state"])
        last_position = xyz6[previous_frame].copy()
        last_frame = np.full(3, previous_frame, dtype=np.int64)
        first_mismatch_seen: set[int] = set()
        verified: list[dict[str, object]] = []

        for frame in range(previous_frame + 1, next_frame + 1):
            for id6_index, expected8_index in enumerate(previous_state):
                if not observed8[frame, expected8_index]:
                    continue
                same_observation = bool(
                    observed6[frame, id6_index]
                    and np.linalg.norm(
                        xyz6[frame, id6_index] - xyz8[frame, expected8_index]
                    )
                    <= args.tolerance_mm
                )
                if same_observation:
                    last_position[id6_index] = xyz6[frame, id6_index]
                    last_frame[id6_index] = frame
                    continue
                if id6_index in first_mismatch_seen:
                    continue
                first_mismatch_seen.add(id6_index)
                gap = int(frame - last_frame[id6_index])
                distance = float(
                    np.linalg.norm(xyz8[frame, expected8_index] - last_position[id6_index])
                )
                gate6 = float(min(args.jump6_mm * gap, args.hard_cap_mm))
                gate8 = float(min(args.jump8_mm * gap, args.hard_cap_mm))
                gate_verified = distance > gate6 + args.tolerance_mm and distance <= gate8 + args.tolerance_mm
                if not gate_verified:
                    continue
                assigned8 = "missing"
                if observed6[frame, id6_index]:
                    distances = np.linalg.norm(
                        xyz8[frame] - xyz6[frame, id6_index], axis=1
                    )
                    assigned8 = IDS[int(np.argmin(distances))]
                verified.append(
                    {
                        "frame": int(frame),
                        "six_id": IDS[id6_index],
                        "expected_eight_id": IDS[expected8_index],
                        "six_assignment": assigned8,
                        "gap_frames": gap,
                        "distance_mm": distance,
                        "gate6_mm": gate6,
                        "gate8_mm": gate8,
                        "above_6mm_gate_mm": distance - gate6,
                        "below_8mm_gate_mm": gate8 - distance,
                        "gate_interval_fraction": (distance - gate6) / (gate8 - gate6),
                    }
                )

        if not verified:
            raise ValueError(
                f"state change {previous_frame}->{next_frame} lacks a verified gate trigger"
            )
        branch_id = f"B{run_index:03d}"
        trigger_frame = min(int(row["frame"]) for row in verified)
        trigger_rows = [row for row in verified if int(row["frame"]) == trigger_frame]
        review_start = max(0, trigger_frame - args.context_frames)
        review_end = min(frame_count - 1, trigger_frame + args.context_frames)
        annotation_context = _annotation_context(
            female_frames, review_start, review_end, trigger_frame
        )
        distances = [float(row["distance_mm"]) for row in verified]
        gate_fractions = [float(row["gate_interval_fraction"]) for row in verified]
        transition = {
            "branch_id": branch_id,
            "trigger_frame": trigger_frame,
            "trigger_time_seconds": trigger_frame / args.source_fps,
            "last_common_old_state_frame": previous_frame,
            "first_common_new_state_frame": next_frame,
            "new_state_last_common_frame": int(current["end"]),
            "new_state_span_frames": int(current["end"]) - next_frame + 1,
            "old_mapping_6_to_8": _mapping_label(previous_state),
            "new_mapping_6_to_8": _mapping_label(current_state),
            "verified_rejection_count": len(verified),
            "first_trigger_six_ids": ";".join(row["six_id"] for row in trigger_rows),
            "first_trigger_expected_eight_ids": ";".join(
                row["expected_eight_id"] for row in trigger_rows
            ),
            "first_trigger_distances_mm": ";".join(
                f"{float(row['distance_mm']):.6f}" for row in trigger_rows
            ),
            "minimum_verified_distance_mm": min(distances),
            "maximum_verified_distance_mm": max(distances),
            "maximum_gate_interval_fraction": max(gate_fractions),
            "review_start_frame": review_start,
            "review_end_frame": review_end,
            **annotation_context,
        }
        transition_rows.append(transition)
        for row in verified:
            detail_rows.append(
                {
                    "branch_id": branch_id,
                    **row,
                    "old_mapping_6_to_8": _mapping_label(previous_state),
                    "new_mapping_6_to_8": _mapping_label(current_state),
                }
            )

    transitions = pd.DataFrame(transition_rows)
    details = pd.DataFrame(detail_rows)

    window_rows: list[dict[str, object]] = []
    window_members: list[pd.Series] = []
    for _, row in transitions.sort_values("trigger_frame").iterrows():
        if window_members and int(row.trigger_frame) - int(window_members[-1].trigger_frame) > args.merge_gap_frames:
            member_table = pd.DataFrame(window_members)
            window_rows.append(
                _make_window_row(
                    member_table, len(window_rows) + 1, female_frames, frame_count, args
                )
            )
            window_members = []
        window_members.append(row)
    if window_members:
        window_rows.append(
            _make_window_row(
                pd.DataFrame(window_members), len(window_rows) + 1, female_frames, frame_count, args
            )
        )
    windows = pd.DataFrame(window_rows)

    gui_rows = []
    for row in transitions.itertuples(index=False):
        has_existing = int(row.existing_confirmed_female_annotations) > 0
        gui_rows.append(
            {
                "experiment": args.experiment,
                "frame": int(row.trigger_frame),
                "candidate_id": f"{args.experiment}:jump6_gate:{row.branch_id}",
                "source": "jump6_gate_branch",
                "priority": 2 if has_existing else 3,
                "score": float(row.maximum_gate_interval_fraction),
                "reason": (
                    f"{row.branch_id}: 6 mm gate rejected {row.first_trigger_six_ids} "
                    f"at frame {row.trigger_frame}; 8 mm accepted; mapping changed "
                    f"from {row.old_mapping_6_to_8} to {row.new_mapping_6_to_8}"
                ),
                "view_hint": "both_sides",
                "branch_id": row.branch_id,
                "review_start_frame": int(row.review_start_frame),
                "review_end_frame": int(row.review_end_frame),
                "old_mapping_6_to_8": row.old_mapping_6_to_8,
                "new_mapping_6_to_8": row.new_mapping_6_to_8,
                "trigger_six_ids": row.first_trigger_six_ids,
                "trigger_expected_eight_ids": row.first_trigger_expected_eight_ids,
                "trigger_distances_mm": row.first_trigger_distances_mm,
                "existing_confirmed_female_annotations": int(
                    row.existing_confirmed_female_annotations
                ),
            }
        )
    gui = pd.DataFrame(gui_rows)

    transitions.to_csv(args.output_dir / "gate_branch_transitions.csv", index=False)
    details.to_csv(args.output_dir / "gate_branch_trigger_details.csv", index=False)
    windows.to_csv(args.output_dir / "gate_branch_review_windows.csv", index=False)
    gui.to_csv(args.output_dir / "female_annotator_candidates.csv", index=False)

    state_counts: dict[str, int] = {}
    for _, state in samples:
        label = _mapping_label(state)
        state_counts[label] = state_counts.get(label, 0) + 1
    summary = {
        "schema_version": 1,
        "experiment": args.experiment,
        "definition": (
            "A verified branch is a change in the 6mm-to-8mm identity permutation "
            "preceded by an expected observation outside min(6*gap,80) mm but "
            "inside min(8*gap,80) mm."
        ),
        "sources": {
            "jump6_csv": str(args.jump6_csv),
            "jump8_csv": str(args.jump8_csv),
            "annotations_csv": str(args.annotations_csv) if args.annotations_csv else None,
        },
        "parameters": {
            "jump6_mm": args.jump6_mm,
            "jump8_mm": args.jump8_mm,
            "hard_cap_mm": args.hard_cap_mm,
            "tolerance_mm": args.tolerance_mm,
            "context_frames": args.context_frames,
            "merge_gap_frames": args.merge_gap_frames,
            "source_fps": args.source_fps,
        },
        "frame_count": frame_count,
        "common_all_three_direct_frames": len(samples),
        "permutation_state_counts": state_counts,
        "verified_branch_transitions": len(transitions),
        "verified_gate_rejection_details": len(details),
        "consolidated_review_windows": len(windows),
        "first_trigger_frame": int(transitions.trigger_frame.min()),
        "last_trigger_frame": int(transitions.trigger_frame.max()),
        "transitions_without_existing_annotation_in_context": int(
            transitions.existing_confirmed_female_annotations.eq(0).sum()
        ),
    }
    (args.output_dir / "gate_branch_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def _make_window_row(
    members: pd.DataFrame,
    window_number: int,
    female_frames: np.ndarray,
    frame_count: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    first_trigger = int(members.trigger_frame.min())
    last_trigger = int(members.trigger_frame.max())
    start = max(0, first_trigger - args.context_frames)
    end = min(frame_count - 1, last_trigger + args.context_frames)
    context = _annotation_context(female_frames, start, end, first_trigger)
    return {
        "window_id": f"W{window_number:02d}",
        "start_frame": start,
        "end_frame": end,
        "start_time_seconds": start / args.source_fps,
        "end_time_seconds": end / args.source_fps,
        "trigger_count": len(members),
        "trigger_frames": ";".join(str(int(value)) for value in members.trigger_frame),
        "branch_ids": ";".join(str(value) for value in members.branch_id),
        **context,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jump6-csv", type=Path, required=True)
    parser.add_argument("--jump8-csv", type=Path, required=True)
    parser.add_argument("--annotations-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment", default="20251225_K_1_2_FCMC")
    parser.add_argument("--jump6-mm", type=float, default=6.0)
    parser.add_argument("--jump8-mm", type=float, default=8.0)
    parser.add_argument("--hard-cap-mm", type=float, default=80.0)
    parser.add_argument("--tolerance-mm", type=float, default=1e-6)
    parser.add_argument("--context-frames", type=int, default=166)
    parser.add_argument("--merge-gap-frames", type=int, default=1000)
    parser.add_argument("--source-fps", type=float, default=166.0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
