from __future__ import annotations

"""Consolidate close-approach fragments into auditable separation-delimited encounters."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _normalise_reference(table: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "fish": "fish_id",
        "gx": "x",
        "gy": "y",
    }
    out = table.rename(columns={key: value for key, value in aliases.items() if key in table})
    required = {"frame", "fish_id", "x", "y"}
    if missing := required - set(out.columns):
        raise ValueError(f"reference missing {sorted(missing)}")
    out = out[["frame", "fish_id", "x", "y"]].copy()
    out["frame"] = pd.to_numeric(out["frame"], errors="raise").astype(int)
    out["fish_id"] = out["fish_id"].astype(str)
    out[["x", "y"]] = out[["x", "y"]].apply(pd.to_numeric, errors="coerce")
    return out[np.isfinite(out[["x", "y"]]).all(axis=1)]


def _true_runs(mask: pd.Series, minimum_length: int) -> list[tuple[int, int]]:
    mask = mask.astype(bool).sort_index()
    runs: list[tuple[int, int]] = []
    start: int | None = None
    previous: int | None = None
    for frame, value in mask.items():
        frame = int(frame)
        contiguous = previous is not None and frame == previous + 1
        if start is not None and (not value or not contiguous):
            end = int(previous)
            if end - start + 1 >= int(minimum_length):
                runs.append((start, end))
            start = None
        if value and start is None:
            start = frame
        previous = frame
    if start is not None and previous is not None and previous - start + 1 >= int(minimum_length):
        runs.append((start, previous))
    return runs


def derive_separation_context_encounters(
    events: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    separation_threshold_px: float,
    separation_hold_frames: int,
    source_fps: float | None = None,
) -> pd.DataFrame:
    """Return encounter rows with clear, sustained pair separation at both ends.

    Adjacent detector fragments are consolidated unless the same fish pair is at
    least ``separation_threshold_px`` apart for ``separation_hold_frames``
    consecutive frames between them. Clip bounds contain the last such clear run
    before an encounter and the first such run after it.
    """
    required = {"event_id", "start_frame", "end_frame", "fish_a", "fish_b"}
    if missing := required - set(events.columns):
        raise ValueError(f"events missing {sorted(missing)}")
    if int(separation_hold_frames) < 1:
        raise ValueError("separation_hold_frames must be positive")

    reference = _normalise_reference(reference)
    encounter_rows: list[dict[str, object]] = []
    for (fish_a, fish_b), pair_events in events.groupby(["fish_a", "fish_b"], sort=True):
        fish_a, fish_b = str(fish_a), str(fish_b)
        pair = reference[reference["fish_id"].isin([fish_a, fish_b])]
        pivot = pair.pivot(index="frame", columns="fish_id", values=["x", "y"])
        try:
            distance = np.hypot(
                pivot[("x", fish_a)] - pivot[("x", fish_b)],
                pivot[("y", fish_a)] - pivot[("y", fish_b)],
            )
        except KeyError as exc:
            raise ValueError(f"reference lacks fish pair {fish_a}, {fish_b}") from exc
        distance = pd.Series(distance, index=pivot.index)
        clear_runs = _true_runs(
            distance.ge(float(separation_threshold_px)).fillna(False),
            int(separation_hold_frames),
        )

        clusters: list[list[pd.Series]] = []
        for _, event in pair_events.sort_values(["start_frame", "end_frame"]).iterrows():
            if not clusters:
                clusters.append([event])
                continue
            previous_end = int(clusters[-1][-1]["end_frame"])
            current_start = int(event["start_frame"])
            separated_between = any(
                run_start > previous_end and run_end < current_start
                for run_start, run_end in clear_runs
            )
            if separated_between:
                clusters.append([event])
            else:
                clusters[-1].append(event)

        for cluster in clusters:
            encounter_start = int(cluster[0]["start_frame"])
            encounter_end = int(cluster[-1]["end_frame"])
            before = [run for run in clear_runs if run[1] < encounter_start]
            after = [run for run in clear_runs if run[0] > encounter_end]
            if not before or not after:
                ids = [str(row["event_id"]) for row in cluster]
                raise ValueError(f"no complete separation context for {ids}")
            pre_run = before[-1]
            post_run = after[0]
            clip_start = pre_run[1] - int(separation_hold_frames) + 1
            clip_end = post_run[0] + int(separation_hold_frames) - 1
            close_intervals = [
                {
                    "event_id": str(row["event_id"]),
                    "start_frame": int(row["start_frame"]),
                    "end_frame": int(row["end_frame"]),
                }
                for row in cluster
            ]
            encounter_rows.append(
                {
                    "start_frame": encounter_start,
                    "end_frame": encounter_end,
                    "clip_start_frame": clip_start,
                    "clip_end_frame": clip_end,
                    "fish_a": fish_a,
                    "fish_b": fish_b,
                    "source_event_ids": ";".join(item["event_id"] for item in close_intervals),
                    "close_intervals_json": json.dumps(close_intervals, separators=(",", ":")),
                    "separation_threshold_px": float(separation_threshold_px),
                    "separation_hold_frames": int(separation_hold_frames),
                    "source_fps": None if source_fps is None else float(source_fps),
                    "separation_hold_seconds": (
                        None
                        if source_fps is None
                        else float(separation_hold_frames) / float(source_fps)
                    ),
                    "pre_context_frames": encounter_start - clip_start,
                    "post_context_frames": clip_end - encounter_end,
                    "clip_frames": clip_end - clip_start + 1,
                    "pre_min_distance_px": float(distance.loc[clip_start:pre_run[1]].min()),
                    "post_min_distance_px": float(distance.loc[post_run[0]:clip_end].min()),
                }
            )

    output = pd.DataFrame(encounter_rows).sort_values("start_frame").reset_index(drop=True)
    output.insert(0, "event_id", [f"encounter_{index:02d}" for index in range(1, len(output) + 1)])
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--separation-threshold-px", type=float, default=100.0)
    parser.add_argument("--separation-hold-frames", type=int, default=166)
    parser.add_argument("--source-fps", type=float, default=166.0)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    events = pd.read_csv(args.events)
    reference = pd.read_csv(args.reference)
    encounters = derive_separation_context_encounters(
        events,
        reference,
        separation_threshold_px=float(args.separation_threshold_px),
        separation_hold_frames=int(args.separation_hold_frames),
        source_fps=float(args.source_fps),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encounters.to_csv(args.output, index=False)
    return {
        "schema_version": "separation_context_encounters_v1",
        "source_events": int(len(events)),
        "encounters": int(len(encounters)),
        "separation_threshold_px": float(args.separation_threshold_px),
        "separation_hold_frames": int(args.separation_hold_frames),
        "separation_hold_seconds": float(args.separation_hold_frames) / float(args.source_fps),
        "output": str(args.output.resolve()),
    }


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
