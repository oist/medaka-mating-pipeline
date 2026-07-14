from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.experiments.identity_recovery_benchmark.build_audit_player import (
    _presentation_order,
    write_audit_player,
)
from scripts.experiments.identity_recovery_benchmark.common import (
    derive_crossing_events,
    geometry_proxy_identity_metrics,
    localization_metrics,
)
from scripts.experiments.identity_recovery_benchmark.build_final_report import (
    apply_locked_decision_rule,
)
from scripts.experiments.identity_recovery_benchmark.run_multiview_contrastive import (
    merge_events,
    solve_episode_permutation,
)
from scripts.experiments.identity_recovery_benchmark.make_separation_context_events import (
    derive_separation_context_encounters,
)
from scripts.experiments.identity_recovery_benchmark.summarize_audit import (
    run as summarize_audit,
    wilson_interval,
)


def _reference() -> pd.DataFrame:
    rows = []
    for frame in range(30):
        rows.extend(
            [
                {"frame": frame, "fish_id": "a", "x": float(frame), "y": 0.0},
                {"frame": frame, "fish_id": "b", "x": float(29 - frame), "y": 0.0},
                {"frame": frame, "fish_id": "c", "x": 100.0, "y": 100.0},
            ]
        )
    return pd.DataFrame(rows)


def test_crossing_selection_is_exhaustive_and_context_gated():
    events = derive_crossing_events(
        _reference(),
        start_frame=0,
        end_frame=29,
        distance_threshold_px=3.0,
        merge_gap_frames=1,
        min_close_frames=2,
        required_context_frames=5,
    )
    assert len(events) == 1
    assert {events[0].fish_a, events[0].fish_b} == {"a", "b"}
    assert events[0].n_close_frames == 4


def test_localization_is_identity_blind():
    ref = _reference()
    tracks = ref.rename(columns={"fish_id": "track_id"})[["frame", "track_id", "x", "y"]]
    tracks["track_id"] = tracks["track_id"].map({"a": "z", "b": "x", "c": "y"})
    score = localization_metrics(tracks, ref, start_frame=0, end_frame=29, gate_px=0.1)
    assert score["position_recall"] == 1.0
    assert score["position_precision"] == 1.0
    assert score["all_fish_frame_coverage"] == 1.0


def test_proxy_identity_score_detects_a_swap():
    ref = _reference()
    tracks = ref.rename(columns={"fish_id": "track_id"})[["frame", "track_id", "x", "y"]].copy()
    # Exchange labels a/b after the crossing while leaving localization perfect.
    mask = tracks["frame"] >= 15
    tracks.loc[mask, "track_id"] = tracks.loc[mask, "track_id"].replace({"a": "b", "b": "a"})
    localization = localization_metrics(tracks, ref, start_frame=0, end_frame=29, gate_px=0.1)
    proxy, matched = geometry_proxy_identity_metrics(
        tracks, ref, start_frame=0, end_frame=29, gate_px=0.1
    )
    assert localization["position_recall"] == 1.0
    assert proxy["idf1_like"] < 1.0
    assert proxy["proxy_switches_adjacent_frames"] == 2
    assert np.isfinite(matched["distance_px"]).all()


def test_episode_merging_and_one_to_one_appearance_permutation():
    events = pd.DataFrame(
        [
            {"event_id": "a", "start_frame": 100, "end_frame": 110},
            {"event_id": "b", "start_frame": 150, "end_frame": 160},
            {"event_id": "c", "start_frame": 400, "end_frame": 410},
        ]
    )
    episodes = merge_events(events, maximum_gap_frames=50)
    assert [episode.event_ids for episode in episodes] == [("a", "b"), ("c",)]

    pre = {
        "0": np.array([1.0, 0.0, 0.0]),
        "1": np.array([0.0, 1.0, 0.0]),
        "2": np.array([0.0, 0.0, 1.0]),
    }
    post = {"0": pre["1"], "1": pre["0"], "2": pre["2"]}
    mapping, _, _, _ = solve_episode_permutation(pre, post)
    assert mapping == {"0": "1", "1": "0", "2": "2"}


def test_wilson_interval_contains_observed_fraction():
    lower, upper = wilson_interval(5, 7)
    assert 0.0 <= lower < 5 / 7 < upper <= 1.0


def _audit_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    worksheet = pd.DataFrame(
        [
            {"event_id": "e1", "blinded_method": "A", "clip": "clips/e1_A.mp4", "continuity": "pass", "localization": "pass", "confidence_1_to_3": "3", "notes": ""},
            {"event_id": "e1", "blinded_method": "B", "clip": "clips/e1_B.mp4", "continuity": "switch", "localization": "pass", "confidence_1_to_3": "2", "notes": ""},
            {"event_id": "e2", "blinded_method": "A", "clip": "clips/e2_A.mp4", "continuity": "pass", "localization": "pass", "confidence_1_to_3": "2", "notes": ""},
            {"event_id": "e2", "blinded_method": "B", "clip": "clips/e2_B.mp4", "continuity": "uncertain", "localization": "uncertain", "confidence_1_to_3": "1", "notes": "ambiguous"},
        ]
    )
    key = worksheet[["event_id", "blinded_method", "clip"]].copy()
    key["method"] = ["method_a", "method_b", "method_a", "method_b"]
    return worksheet, key


def test_audit_summary_is_conservative_and_requires_confidence(tmp_path):
    worksheet, key = _audit_tables()
    worksheet_path = tmp_path / "worksheet.csv"
    key_path = tmp_path / "key.csv"
    worksheet.to_csv(worksheet_path, index=False)
    key.to_csv(key_path, index=False)
    report = summarize_audit(
        SimpleNamespace(
            worksheet=worksheet_path,
            blinding_key=key_path,
            output_dir=tmp_path / "results",
            allow_incomplete=False,
        )
    )
    by_method = {row["method"]: row for row in report["methods"]}
    assert by_method["method_a"]["conservative_success_fraction"] == 1.0
    assert by_method["method_b"]["conservative_success_fraction"] == 0.0
    assert by_method["method_b"]["events_scored"] == 2
    assert by_method["method_a"]["mean_confidence"] == 2.5
    pair = report["pairwise_comparisons"][0]
    assert pair["method_a_wins"] == 2
    assert pair["method_b_wins"] == 0
    assert pair["paired_success_rate_difference_a_minus_b"] == 1.0
    assert pair["exact_mcnemar_p_two_sided"] == 0.5
    assert report["event_units"] == 2
    assert "2 events" in report["pairwise_test"]

    worksheet.loc[0, "confidence_1_to_3"] = ""
    worksheet.to_csv(worksheet_path, index=False)
    with pytest.raises(ValueError, match="incomplete or invalid"):
        summarize_audit(
            SimpleNamespace(
                worksheet=worksheet_path,
                blinding_key=key_path,
                output_dir=tmp_path / "invalid",
                allow_incomplete=False,
            )
        )


def test_audit_player_is_blinded_and_separates_repeated_events(tmp_path):
    worksheet, _ = _audit_tables()
    order = _presentation_order(worksheet, seed=17)
    ordered_events = worksheet.iloc[order]["event_id"].tolist()
    assert all(left != right for left, right in zip(ordered_events, ordered_events[1:]))

    worksheet_path = tmp_path / "worksheet.csv"
    output_path = tmp_path / "AUDIT_PLAYER.html"
    worksheet.to_csv(worksheet_path, index=False)
    write_audit_player(worksheet_path, output_path, seed=17)
    html = output_path.read_text(encoding="utf-8")
    assert "AUDIT_WORKSHEET_COMPLETED.csv" in html
    assert "method_a" not in html and "method_b" not in html
    for clip in worksheet["clip"]:
        assert clip in html


def test_separation_context_consolidates_fragments_and_keeps_clear_endpoints():
    frames = np.arange(100)
    distance = np.full(100, 150.0)
    distance[20:61] = 30.0
    reference_rows = []
    for frame, pair_distance in zip(frames, distance):
        reference_rows.extend(
            [
                {"frame": frame, "fish_id": "a", "x": 0.0, "y": 0.0},
                {"frame": frame, "fish_id": "b", "x": pair_distance, "y": 0.0},
            ]
        )
    events = pd.DataFrame(
        [
            {"event_id": "e1", "start_frame": 25, "end_frame": 30, "fish_a": "a", "fish_b": "b"},
            {"event_id": "e2", "start_frame": 45, "end_frame": 50, "fish_a": "a", "fish_b": "b"},
        ]
    )
    encounters = derive_separation_context_encounters(
        events,
        pd.DataFrame(reference_rows),
        separation_threshold_px=100.0,
        separation_hold_frames=10,
    )
    assert len(encounters) == 1
    row = encounters.iloc[0]
    assert row["source_event_ids"] == "e1;e2"
    assert row["clip_start_frame"] == 10
    assert row["clip_end_frame"] == 70
    assert row["pre_min_distance_px"] >= 100.0
    assert row["post_min_distance_px"] >= 100.0


def test_locked_decision_rule_combines_localization_and_human_continuity():
    benchmark = {
        "methods": {
            "multiview_constrained_contrastive": {
                "localization_identity_blind": {
                    "80": {"position_recall": 0.96, "all_fish_frame_coverage": 0.90}
                }
            },
            "idtracker_ai_v6": {
                "localization_identity_blind": {
                    "80": {"position_recall": 0.85, "all_fish_frame_coverage": 0.62}
                }
            },
            "native_udmt": {
                "localization_identity_blind": {
                    "80": {"position_recall": 0.07, "all_fish_frame_coverage": 0.01}
                }
            },
        }
    }
    audit_methods = [
        {"method": method, "events_scored": 7, "conservative_successes": successes,
         "conservative_success_fraction": successes / 7, "wilson_95_lower": 0.1,
         "wilson_95_upper": 0.9, "mean_confidence": 2.5,
         "continuity_pass": successes, "continuity_switch": 0,
         "continuity_uncertain": 7 - successes, "localization_fail": 0,
         "localization_uncertain": 0, "localized_events": 7}
        for method, successes in [
            ("multiview_constrained_contrastive", 6),
            ("idtracker_ai_v6", 7),
            ("native_udmt", 7),
        ]
    ]
    rows = apply_locked_decision_rule(benchmark, audit_methods)
    by_method = {row["method_key"]: row for row in rows}
    assert by_method["multiview_constrained_contrastive"]["decision"] == "recovered in locked interval"
    assert "localization gate failed" in by_method["idtracker_ai_v6"]["decision"]
    assert "localization gate failed" in by_method["native_udmt"]["decision"]
