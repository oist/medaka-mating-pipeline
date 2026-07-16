from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.analysis.heldout_association_benchmark import (
    BenchmarkConfig,
    build_heldout_episodes,
    frame_split_masks,
    load_event_candidates,
    run_benchmark,
    write_benchmark_outputs,
)


def _synthetic_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference_rows = []
    track_rows = []
    identities = ("H1", "H2", "H3")
    labels = ("F", "M1", "M2")
    offsets = (0.0, 50.0, 100.0)
    for frame in range(30):
        for slot, (human, label, offset) in enumerate(zip(identities, labels, offsets)):
            xyz = (offset + 0.1 * frame, float(slot), 0.0)
            reference_rows.append(
                {"frame": frame, "human_identity": human, "X": xyz[0], "Y": xyz[1],
                 "Z": xyz[2], "mean_reprojection_error_px": 2.0,
                 "max_reprojection_error_px": 3.0,
                 "reviewed_tracklet_id": f"RT{slot}"}
            )
            baseline = label
            if 14 <= frame <= 16 and slot in (0, 1):
                baseline = {0: "M1", 1: "F"}[slot]
            track_rows.append(
                {"frame": frame, "geometry_slot": slot, "X": xyz[0], "Y": xyz[1],
                 "Z": xyz[2], "baseline_id": baseline, "optimized_id": label}
            )
    events = pd.DataFrame([{"frame": 15, "event_type": "pair_swap", "event_id": "E1"}])
    return pd.DataFrame(reference_rows), pd.DataFrame(track_rows), events


def test_episodes_merge_guard_overlap_and_split_has_no_leakage():
    raw = pd.DataFrame(
        [
            {"frame": 20, "event_type": "swap", "event_id": "a"},
            {"frame": 25, "event_type": "swap", "event_id": "b"},
            {"frame": 70, "event_type": "gap", "event_id": "c"},
        ]
    )
    episodes = build_heldout_episodes(
        load_event_candidates(raw), frame_min=0, frame_max=100,
        radius_frames=5, guard_frames=5,
    )
    assert len(episodes) == 2
    assert episodes.iloc[0].n_source_events == 2
    assert episodes.iloc[0].end_frame < episodes.iloc[1].start_frame
    assert episodes.iloc[0].guard_end_frame < episodes.iloc[1].guard_start_frame

    frames = np.arange(101)
    heldout, train = frame_split_masks(frames, episodes)
    assert not np.any(heldout & train)
    assert heldout[20] and heldout[25] and heldout[70]
    assert not train[episodes.iloc[0].guard_start_frame]


def test_optimized_labels_win_on_event_centered_holdout_and_remove_jumps():
    reference, track, events = _synthetic_tables()
    result = run_benchmark(
        reference=reference,
        baseline_track=track,
        baseline_label_col="baseline_id",
        optimized_label_col="optimized_id",
        events=events,
        config=BenchmarkConfig(
            episode_radius_frames=4, guard_frames=2, match_distance_mm=1.0
        ),
    )
    summaries = {row["method"]: row for row in result["method_summaries"]}
    assert summaries["optimized"]["heldout_identity_accuracy_on_matches"] == 1.0
    assert summaries["baseline"]["heldout_identity_accuracy_on_matches"] < 1.0
    assert summaries["baseline"]["jumps"]["jumps_gt20"] == 4
    assert summaries["optimized"]["jumps"]["jumps_gt8"] == 0
    assert summaries["baseline"]["permutation_validity"]["duplicate_label_frames"] == 0
    assert result["leakage_audit"]["heldout_training_intersection_rows"] == 0

    train_matches = result["matches"].query("method == 'baseline' and split == 'train'")
    assert not train_matches["frame"].between(9, 21).any()  # episode 11--19 + guard 2


def test_outputs_are_inspectable_and_json_serializable(tmp_path):
    reference, track, events = _synthetic_tables()
    result = run_benchmark(
        reference=reference, baseline_track=track, baseline_label_col="baseline_id",
        optimized_label_col="optimized_id", events=events,
        config=BenchmarkConfig(episode_radius_frames=3, guard_frames=2, match_distance_mm=1.0),
    )
    paths = write_benchmark_outputs(result, tmp_path)
    assert all(path.exists() for path in paths.values())
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["schema_version"] == "heldout_association_benchmark_v1"
    assert report["leakage_audit"]["heldout_training_intersection_rows"] == 0
    assert set(pd.read_csv(paths["method_summary"])["method"]) == {"baseline", "optimized"}
