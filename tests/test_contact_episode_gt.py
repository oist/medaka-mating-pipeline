from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.contact_episode_gt import (
    ANNOTATION_COLUMNS,
    CANDIDATE_COLUMNS,
    LEGACY_ANNOTATION_COLUMNS,
    V1_1_ANNOTATION_COLUMNS,
    ContactAnnotationLedger,
    DetectorConfig,
    deserialize_clicks,
    detect_contact_episodes,
    empty_clicks,
    expand_frame_range,
    load_tracking_csv,
    make_annotation_row,
    make_click_evidence,
    manual_episode,
    pair_colors_for,
    pairwise_ifd,
    read_annotation_ledger,
    serialize_clicks,
    serialize_false_negative_detections,
    serialize_false_positive_detections,
    stable_pair_order,
    suggest_anchor_frames,
    suggest_tracking_assessment,
    validate_annotation_row,
    validate_false_negative_detections,
    validate_false_positive_detections,
    write_candidate_artifacts,
)


def _xyz() -> np.ndarray:
    return np.array(
        [
            [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 12.0]],
            [[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan], [0.0, 0.0, 8.0]],
        ]
    )


def test_pairwise_ifd_correctness_and_missing_values() -> None:
    direct, secondary, minimum, minimum_pair = pairwise_ifd(_xyz())
    np.testing.assert_allclose(direct[0], [5.0, 12.0, 13.0])
    assert np.isnan(direct[1, 0])
    assert direct[1, 1] == 8.0
    assert np.isnan(direct[1, 2])
    assert np.isnan(secondary).all()
    np.testing.assert_allclose(minimum, [5.0, 8.0])
    np.testing.assert_array_equal(minimum_pair, [0, 1])


def test_minimum_ifd_is_invariant_to_id_permutation() -> None:
    xyz = _xyz()[:1]
    first = pairwise_ifd(xyz)[2]
    permuted = pairwise_ifd(xyz[:, [2, 0, 1]])[2]
    np.testing.assert_allclose(first, permuted)


def test_interpolated_pairs_are_secondary_and_cannot_trigger_contact() -> None:
    xyz = _xyz()[:1]
    direct_mask = np.array([[True, False, True]])
    direct, secondary, minimum, _pair = pairwise_ifd(xyz, direct_mask=direct_mask)
    assert np.isnan(direct[0, 0])
    assert direct[0, 1] == 12.0
    assert np.isnan(direct[0, 2])
    np.testing.assert_allclose(secondary[0, [0, 2]], [5.0, 13.0])
    assert minimum[0] == 12.0


def test_hysteresis_entry_exit_and_missing_break() -> None:
    values = np.array([20.0, 11.0, 13.0, 16.0, 16.1, 11.0, np.nan, 11.0, 17.0])
    candidates = detect_contact_episodes(
        values,
        DetectorConfig(enter_mm=12, exit_mm=16, merge_gap_frames=0, context_frames=2),
    )
    assert candidates[["contact_start_frame", "contact_end_frame"]].values.tolist() == [
        [1, 3],
        [5, 5],
        [7, 7],
    ]


def test_interval_merge_context_expansion_and_clipping() -> None:
    values = np.array([20.0, 11.0, 17.0, 20.0, 11.0, 13.0, 17.0])
    candidates = detect_contact_episodes(
        values,
        DetectorConfig(enter_mm=12, exit_mm=16, merge_gap_frames=2, context_frames=20),
        experiment="exp",
        tracking_method="jump8",
    )
    assert len(candidates) == 1
    row = candidates.iloc[0]
    assert (row["contact_start_frame"], row["contact_end_frame"]) == (1, 5)
    assert row["merged_raw_interval_count"] == 2
    assert row["missing_direct_frames"] == 0
    assert (row["review_start_frame"], row["review_end_frame"]) == (0, 6)
    assert expand_frame_range(2, 5, 166, 0, 10) == (0, 10)


def test_stable_pair_order_and_colors() -> None:
    assert stable_pair_order(("ID_1", "ID_2", "ID_3")) == (
        ("ID_1", "ID_2"),
        ("ID_1", "ID_3"),
        ("ID_2", "ID_3"),
    )
    assert pair_colors_for(("ID_1", "ID_2", "ID_3")) == {
        "ID_1-ID_2": "#f28e2b",
        "ID_1-ID_3": "#9467bd",
        "ID_2-ID_3": "#17becf",
    }


def test_anchor_click_serialization_keeps_free_raw_click() -> None:
    clicks = empty_clicks()
    clicks["anchor_a"]["female"]["top_cam"] = make_click_evidence(
        experiment="exp", view="top_cam", frame=4, click_x=12.5, click_y=17.25
    )
    clicks["anchor_b"]["anchor_male"]["side2_cam"] = make_click_evidence(
        experiment="exp",
        view="side2_cam",
        frame=10,
        click_x=22,
        click_y=31,
        instance_index=2,
        anchor_xy=(21, 30),
        geometry_by_method={"jump6": "ID_2"},
    )
    restored = deserialize_clicks(serialize_clicks(clicks))
    free = restored["anchor_a"]["female"]["top_cam"]
    assert free["selection_status"] == "free_click"
    assert free["click_x"] == 12.5
    assert free["geometry_by_method"] == {}
    linked = restored["anchor_b"]["anchor_male"]["side2_cam"]
    assert linked["detection_key"] == "exp|side2_cam|10|2"


def test_false_positive_detection_serialization_is_stable_and_deduplicated() -> None:
    value = [
        {
            "view": "top_cam",
            "frame": 10,
            "instance_index": 2,
            "detection_key": "exp|top_cam|10|2",
            "marked_at_utc": "2026-01-01T00:00:00Z",
        },
        {
            "view": "top_cam",
            "frame": 10,
            "instance_index": 2,
            "detection_key": "exp|top_cam|10|2",
            "marked_at_utc": "later duplicate",
        },
    ]
    restored = validate_false_positive_detections(
        serialize_false_positive_detections(value)
    )
    assert len(restored) == 1
    assert restored[0]["detection_key"] == "exp|top_cam|10|2"


def test_false_negative_detection_round_trip_is_stable_and_deduplicated() -> None:
    value = [
        {
            "view": "side2_cam",
            "frame": 12,
            "click_x": 123.5,
            "click_y": 456.25,
            "marked_at_utc": "2026-07-21T00:00:00Z",
        },
        {
            "view": "side2_cam",
            "frame": 12,
            "click_x": 123.5,
            "click_y": 456.25,
            "marked_at_utc": "duplicate",
        },
    ]
    restored = validate_false_negative_detections(
        serialize_false_negative_detections(value)
    )
    assert restored == [value[0]]
    assert "instance_index" not in restored[0]


def test_false_positive_prediction_cannot_supply_saved_id_link(tmp_path) -> None:
    row = _annotation(tmp_path)
    clicks = deserialize_clicks(row["clicks_json"])
    linked = clicks["anchor_a"]["female"]["top_cam"]
    row["false_positive_detections_json"] = serialize_false_positive_detections(
        [
            {
                "view": "top_cam",
                "frame": 2,
                "instance_index": linked["instance_index"],
                "detection_key": linked["detection_key"],
                "marked_at_utc": "2026-01-01T00:00:00Z",
            }
        ]
    )
    with pytest.raises(ValueError, match="false-positive SLEAP predictions"):
        validate_annotation_row(row)


@pytest.mark.parametrize(
    ("anchor_a", "anchor_b", "expected"),
    [
        ({"female": "ID_1", "anchor_male": "ID_2"}, {"female": "ID_1", "anchor_male": "ID_2"}, "same_ids_no_swap"),
        ({"female": "ID_1", "anchor_male": "ID_2"}, {"female": "ID_2", "anchor_male": "ID_1"}, "female_involved_swap"),
        ({"female": "ID_1", "anchor_male": "ID_2"}, {"female": "ID_1", "anchor_male": "ID_3"}, "male_to_male_swap"),
        ({"female": "ID_1", "anchor_male": ""}, {"female": "ID_1", "anchor_male": "ID_3"}, "incomplete_association"),
    ],
)
def test_mapping_suggestions(anchor_a, anchor_b, expected) -> None:
    assert suggest_tracking_assessment(anchor_a, anchor_b) == expected


def _associated_click(experiment: str, view: str, frame: int, identity: str) -> dict:
    return make_click_evidence(
        experiment=experiment,
        view=view,
        frame=frame,
        click_x=20,
        click_y=30,
        instance_index=0,
        anchor_xy=(20, 30),
        geometry_by_method={"jump8": identity},
    )


def _annotation(tmp_path, *, assessment="correct_continuity", notes="") -> dict:
    episode = manual_episode(
        experiment="exp",
        tracking_method="jump8",
        start_frame=5,
        end_frame=7,
        frame_min=0,
        frame_max=20,
        context_frames=5,
    )
    clicks = empty_clicks()
    for phase, frame in (("anchor_a", 2), ("anchor_b", 10)):
        clicks[phase]["female"]["top_cam"] = _associated_click("exp", "top_cam", frame, "ID_1")
        clicks[phase]["anchor_male"]["top_cam"] = _associated_click("exp", "top_cam", frame, "ID_2")
    return make_annotation_row(
        experiment="exp",
        episode=episode,
        annotator="tester",
        anchor_a_frame=2,
        anchor_b_frame=10,
        clicks=clicks,
        contact_validity="confirmed_contact_crossing",
        tracker_assessment=assessment,
        assessed_tracking_label="jump8",
        evidence_quality="usable",
        issue_flags=["close_contact"],
        notes=notes,
    )


def test_anchor_suggestions_choose_nearest_direct_separated_frames(tmp_path) -> None:
    rows = []
    distances = {frame: (10.0 if 4 <= frame <= 8 else 20.0) for frame in range(13)}
    for frame, distance in distances.items():
        for fish_id, x in (("ID_1", 0.0), ("ID_2", distance), ("ID_3", 100.0)):
            rows.append(
                {
                    "frame": frame,
                    "fish_id": fish_id,
                    "X": x,
                    "Y": 0.0,
                    "Z": 0.0,
                    "is_interpolated": False,
                }
            )
    path = tmp_path / "anchor_tracks.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    store = load_tracking_csv(path, method="jump8")
    episode = manual_episode(
        experiment="exp",
        tracking_method="jump8",
        start_frame=5,
        end_frame=7,
        frame_min=0,
        frame_max=12,
        context_frames=20,
    )
    assert suggest_anchor_frames(store, episode, separation_mm=16.0) == (3, 9)


def test_schema_atomic_upsert_resume_and_undo(tmp_path) -> None:
    csv_path = tmp_path / "annotations.csv"
    audit_path = tmp_path / "annotations.audit.jsonl"
    ledger = ContactAnnotationLedger(csv_path, audit_path)
    first = _annotation(tmp_path, notes="first")
    assert ledger.upsert(first) == "add"
    edited = dict(first, notes="edited")
    assert ledger.upsert(edited) == "edit"
    assert len(read_annotation_ledger(csv_path)) == 1
    reopened = ContactAnnotationLedger(csv_path, audit_path)
    latest = reopened.latest("exp", "tester")
    assert latest is not None and latest["notes"] == "edited"
    restored = ledger.undo()
    assert restored is not None and restored["notes"] == "first"
    records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert [record["action"] for record in records] == ["add", "edit", "undo"]
    assert list(read_annotation_ledger(csv_path).columns) == ANNOTATION_COLUMNS


def test_schema_1_0_ledger_is_upgraded_to_neutral_anchors(tmp_path) -> None:
    current = _annotation(tmp_path)
    clicks = json.loads(current["clicks_json"])
    legacy_clicks = {
        "pre": clicks["anchor_a"],
        "post": clicks["anchor_b"],
    }
    legacy = {}
    for column in LEGACY_ANNOTATION_COLUMNS:
        if column == "pre_frame":
            legacy[column] = current["anchor_a_frame"]
        elif column == "post_frame":
            legacy[column] = current["anchor_b_frame"]
        else:
            legacy[column] = current[column]
    legacy["schema_version"] = "1.0"
    legacy["clicks_json"] = json.dumps(legacy_clicks, separators=(",", ":"))
    path = tmp_path / "legacy.csv"
    pd.DataFrame([legacy], columns=LEGACY_ANNOTATION_COLUMNS).to_csv(path, index=False)

    upgraded = read_annotation_ledger(path).iloc[0]
    assert upgraded["schema_version"] == "1.2"
    assert upgraded["anchor_a_frame"] == 2
    assert upgraded["anchor_b_frame"] == 10
    assert json.loads(upgraded["false_positive_detections_json"]) == []
    assert set(json.loads(upgraded["clicks_json"])) == {"anchor_a", "anchor_b"}


def test_schema_1_1_ledger_migrates_in_memory_without_rewriting(tmp_path) -> None:
    current = _annotation(tmp_path)
    v1_1 = {column: current[column] for column in V1_1_ANNOTATION_COLUMNS}
    v1_1["schema_version"] = "1.1"
    path = tmp_path / "v1_1.csv"
    pd.DataFrame([v1_1], columns=V1_1_ANNOTATION_COLUMNS).to_csv(path, index=False)
    before = path.read_bytes()

    upgraded = read_annotation_ledger(path)

    assert len(upgraded) == 1
    assert upgraded.iloc[0]["schema_version"] == "1.2"
    assert json.loads(upgraded.iloc[0]["false_negative_detections_json"]) == []
    assert path.read_bytes() == before


def test_suggestion_is_saved_separately_from_human_decision(tmp_path) -> None:
    row = _annotation(tmp_path, assessment="temporary_swap_flicker")
    assert row["automatic_mapping_suggestion"] == "same_ids_no_swap"
    assert row["tracker_assessment"] == "temporary_swap_flicker"


def test_tracking_csv_direct_interpolated_and_candidate_artifacts(tmp_path) -> None:
    rows = []
    for frame in range(4):
        for fish_index, fish_id in enumerate(("ID_1", "ID_2", "ID_3")):
            rows.append(
                {
                    "frame": frame,
                    "fish_id": fish_id,
                    "X": fish_index * (5 if frame == 1 else 20),
                    "Y": 0,
                    "Z": 0,
                    "is_interpolated": frame == 2 and fish_id == "ID_2",
                }
            )
    source = tmp_path / "tracks.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    store = load_tracking_csv(source, method="jump8")
    assert store.minimum_direct[1] == 5
    assert np.isnan(store.direct_distances[2, 0])
    assert store.interpolated_distances[2, 0] == 20
    config = DetectorConfig(context_frames=1)
    candidates = detect_contact_episodes(store, config, experiment="exp")
    manual = manual_episode(
        experiment="exp",
        tracking_method="jump8",
        start_frame=3,
        end_frame=3,
        frame_min=0,
        frame_max=3,
        context_frames=1,
        source_provenance=store.source_provenance,
    )
    candidates = pd.concat(
        [candidates, pd.DataFrame([manual], columns=CANDIDATE_COLUMNS)],
        ignore_index=True,
    )
    csv_path = tmp_path / "candidates.csv"
    summary_path = tmp_path / "summary.json"
    summary = write_candidate_artifacts(
        candidates,
        csv_path,
        summary_path,
        experiment="exp",
        tracking_method="jump8",
        config=config,
        source_provenance=store.source_provenance,
    )
    assert list(pd.read_csv(csv_path).columns) == CANDIDATE_COLUMNS
    assert summary["direct_observations_only"] is True
    assert summary["manual_candidate_count"] == 1
    assert summary["detected_candidate_count"] == len(candidates) - 1
    assert json.loads(summary_path.read_text())["candidate_count"] == len(candidates)
