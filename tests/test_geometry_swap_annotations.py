from __future__ import annotations

import pandas as pd
import pytest

from scripts.tracking.geometry_swap_annotations import (
    build_clean_tracklets,
    build_tracklet_constraints,
    false_positive_slots_at_frame,
    identity_state_at_frame,
    mapping_at_frame,
    node_qc_events_at_frame,
    normalize_events,
    replay_annotations,
    review_progress_checkpoint,
)


def _tracks() -> pd.DataFrame:
    rows = []
    for frame in range(8):
        for index, source in enumerate(("ID_1", "ID_2", "ID_3"), start=1):
            rows.append(
                {
                    "frame": frame,
                    "fish_id": source,
                    "X": float(frame + index * 10),
                    "Y": float(index),
                    "Z": 0.0,
                    "is_interpolated": frame == 6 and source == "ID_1",
                }
            )
    return pd.DataFrame(rows)


def _events(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_pair_swap_changes_mapping_at_transition_frame_and_preserves_geometry():
    source = _tracks()
    events = _events(
        {
            "event_id": "E1",
            "event_type": "pair_swap",
            "frame_start": 3,
            "frame_end": 3,
            "slot_a": "G1",
            "slot_b": "G2",
        }
    )
    result = replay_annotations(source, events)
    before = result[result["frame"].eq(2)].set_index("geometry_slot")
    after = result[result["frame"].eq(3)].set_index("geometry_slot")
    assert before.loc["G1", "human_identity"] == "H1"
    assert before.loc["G2", "human_identity"] == "H2"
    assert after.loc["G1", "human_identity"] == "H2"
    assert after.loc["G2", "human_identity"] == "H1"
    pd.testing.assert_series_equal(result["X"], source.sort_values(["frame", "fish_id"])["X"].reset_index(drop=True))


def test_view_scoped_swaps_are_independent_and_only_3d_changes_export():
    events = _events(
        {
            "event_id": "E1",
            "event_type": "pair_swap",
            "view_scope": "top_cam",
            "frame_start": 2,
            "frame_end": 2,
            "slot_a": "G1",
            "slot_b": "G2",
        },
        {
            "event_id": "E2",
            "event_type": "pair_swap",
            "view_scope": "3d",
            "frame_start": 3,
            "frame_end": 3,
            "slot_a": "G2",
            "slot_b": "G3",
        },
    )
    top, _active, _resolved = identity_state_at_frame(
        events, ["G1", "G2", "G3"], 4, view_scope="top_cam"
    )
    side, _active, _resolved = identity_state_at_frame(
        events, ["G1", "G2", "G3"], 4, view_scope="side1_cam"
    )
    track_3d, _active, _resolved = identity_state_at_frame(
        events, ["G1", "G2", "G3"], 4, view_scope="3d"
    )
    assert top == {"G1": "H2", "G2": "H1", "G3": "H3"}
    assert side == {"G1": "H1", "G2": "H2", "G3": "H3"}
    assert track_3d == {"G1": "H1", "G2": "H3", "G3": "H2"}

    exported = replay_annotations(_tracks(), events)
    frame = exported[exported["frame"].eq(4)].set_index("geometry_slot")
    assert frame["human_identity"].to_dict() == track_3d


def test_legacy_unscoped_events_are_migrated_to_global_scope():
    normalized = normalize_events(
        _events(
            {
                "event_id": "E1",
                "event_type": "reviewed_no_swap",
                "frame_start": 1,
                "frame_end": 1,
            }
        ),
        ["G1", "G2", "G3"],
    )
    assert normalized.loc[0, "view_scope"] == "all"


def test_single_slot_false_positive_does_not_invalidate_mapping_or_other_views():
    events = _events(
        {
            "event_id": "E1",
            "event_type": "false_positive",
            "view_scope": "top_cam",
            "frame_start": 2,
            "frame_end": 4,
            "slot_a": "G2",
        }
    )
    slots = ["G1", "G2", "G3"]
    assert false_positive_slots_at_frame(
        events, slots, 3, view_scope="top_cam"
    ) == {"G2": ["E1"]}
    assert false_positive_slots_at_frame(
        events, slots, 3, view_scope="side1_cam"
    ) == {}
    assert false_positive_slots_at_frame(
        events, slots, 5, view_scope="top_cam"
    ) == {}
    mapping, active, resolved = identity_state_at_frame(
        events, slots, 5, view_scope="top_cam"
    )
    assert mapping == {"G1": "H1", "G2": "H2", "G3": "H3"}
    assert active == []
    assert resolved

    # Camera-only FPs are diagnostic and must not remove valid 3D track rows.
    corrected = replay_annotations(_tracks(), events)
    assert corrected["label_status"].ne("ambiguous").all()


def test_node_qc_events_are_camera_slot_and_interval_specific():
    events = _events(
        {
            "event_id": "E1",
            "event_type": "bad_mouth",
            "view_scope": "top_cam",
            "frame_start": 10,
            "frame_end": 12,
            "slot_a": "G2",
        },
        {
            "event_id": "E2",
            "event_type": "bad_tail",
            "view_scope": "top_cam",
            "frame_start": 11,
            "frame_end": 11,
            "slot_a": "G2",
        },
        {
            "event_id": "E3",
            "event_type": "wrong_link",
            "view_scope": "side1_cam",
            "frame_start": 11,
            "frame_end": 14,
            "slot_a": "G1",
        },
    )
    slots = ["G1", "G2", "G3"]

    assert node_qc_events_at_frame(
        events, slots, 11, view_scope="top_cam"
    ) == {
        "G2": {"bad_mouth": ["E1"], "bad_tail": ["E2"]}
    }
    assert node_qc_events_at_frame(
        events, slots, 12, view_scope="top_cam"
    ) == {"G2": {"bad_mouth": ["E1"]}}
    assert node_qc_events_at_frame(
        events, slots, 11, view_scope="side1_cam"
    ) == {"G1": {"wrong_link": ["E3"]}}
    assert node_qc_events_at_frame(
        events, slots, 15, view_scope="side1_cam"
    ) == {}


@pytest.mark.parametrize("event_type", ["bad_mouth", "bad_tail", "wrong_link"])
def test_node_qc_event_validation_requires_one_camera_and_known_slot(event_type):
    slots = ["G1", "G2", "G3"]
    with pytest.raises(ValueError, match="must target one camera view"):
        normalize_events(
            _events(
                {
                    "event_id": "E1",
                    "event_type": event_type,
                    "view_scope": "3d",
                    "frame_start": 2,
                    "frame_end": 3,
                    "slot_a": "G1",
                }
            ),
            slots,
        )
    with pytest.raises(ValueError, match="exactly one slot"):
        normalize_events(
            _events(
                {
                    "event_id": "E3",
                    "event_type": event_type,
                    "view_scope": "top_cam",
                    "frame_start": 2,
                    "frame_end": 3,
                    "slot_a": "G1",
                    "slot_b": "G2",
                }
            ),
            slots,
        )
    with pytest.raises(ValueError, match=f"invalid {event_type} slot"):
        normalize_events(
            _events(
                {
                    "event_id": "E2",
                    "event_type": event_type,
                    "view_scope": "top_cam",
                    "frame_start": 2,
                    "frame_end": 3,
                    "slot_a": "G9",
                }
            ),
            slots,
        )


def test_review_progress_checkpoint_is_explicit_and_camera_specific():
    events = _events(
        {
            "event_id": "E1",
            "event_type": "review_progress",
            "view_scope": "top_cam",
            "frame_start": 200,
            "frame_end": 200,
            "created_at": "2026-07-15T01:00:00Z",
            "notes": "first pass",
        },
        {
            # A later checkpoint may deliberately move review backward. The
            # timestamp, not the frame-sorted ledger row, defines latest.
            "event_id": "E2",
            "event_type": "review_progress",
            "view_scope": "top_cam",
            "frame_start": 150,
            "frame_end": 150,
            "created_at": "2026-07-15T02:00:00Z",
            "notes": "resume here",
        },
        {
            "event_id": "E3",
            "event_type": "review_progress",
            "view_scope": "side1_cam",
            "frame_start": 80,
            "frame_end": 80,
            "created_at": "2026-07-15T03:00:00Z",
        },
    )
    slots = ["G1", "G2", "G3"]

    assert review_progress_checkpoint(
        events, slots, view_scope="top_cam"
    ) == {
        "event_id": "E2",
        "view_scope": "top_cam",
        "frame": 150,
        "created_at": "2026-07-15T02:00:00Z",
        "notes": "resume here",
    }
    assert review_progress_checkpoint(
        events, slots, view_scope="side1_cam"
    )["frame"] == 80
    assert review_progress_checkpoint(
        events, slots, view_scope="side2_cam"
    ) is None


def test_review_progress_validation_rejects_global_or_interval_events():
    slots = ["G1", "G2", "G3"]
    with pytest.raises(ValueError, match="must target one camera view"):
        normalize_events(
            _events(
                {
                    "event_id": "E1",
                    "event_type": "review_progress",
                    "view_scope": "all",
                    "frame_start": 3,
                    "frame_end": 3,
                }
            ),
            slots,
        )
    with pytest.raises(ValueError, match="camera-scoped"):
        normalize_events(
            _events(
                {
                    "event_id": "E3",
                    "event_type": "review_progress",
                    "view_scope": "side2_cam",
                    "frame_start": 3,
                    "frame_end": 3,
                    "slot_a": "G1",
                }
            ),
            slots,
        )
    with pytest.raises(ValueError, match="one checkpoint frame"):
        normalize_events(
            _events(
                {
                    "event_id": "E2",
                    "event_type": "review_progress",
                    "view_scope": "side2_cam",
                    "frame_start": 3,
                    "frame_end": 4,
                }
            ),
            slots,
        )


def test_ambiguous_and_interpolated_rows_are_not_clean_training_rows():
    events = _events(
        {
            "event_id": "E1",
            "event_type": "ambiguous",
            "frame_start": 2,
            "frame_end": 4,
        }
    )
    corrected = replay_annotations(_tracks(), events)
    ambiguous = corrected[corrected["frame"].between(2, 4)]
    assert ambiguous["human_identity"].isna().all()
    assert ambiguous["label_status"].eq("ambiguous").all()
    after = corrected[corrected["frame"].eq(5)]
    assert after["human_identity"].isna().all()
    assert after["label_status"].eq("unresolved").all()
    assert corrected.loc[
        corrected["frame"].eq(6) & corrected["geometry_slot"].eq("G1"),
        "label_status",
    ].item() == "unresolved"
    assert corrected.loc[
        corrected["frame"].eq(6) & corrected["geometry_slot"].eq("G1"),
        "is_interpolated",
    ].item()

    clean, manifest = build_clean_tracklets(corrected, events)
    assert not clean["frame"].between(2, 4).any()
    assert not ((clean["frame"].eq(6)) & clean["geometry_slot"].eq("G1")).any()
    # Nothing after the uncertainty is exported until a mapping anchor exists.
    assert manifest["clean_tracklet_id"].nunique() == 3
    assert manifest["end_frame"].max() == 1


def test_track_break_and_swap_split_fragments_even_when_frames_are_consecutive():
    events = _events(
        {
            "event_id": "E1",
            "event_type": "track_break",
            "frame_start": 2,
            "frame_end": 2,
        },
        {
            "event_id": "E2",
            "event_type": "pair_swap",
            "frame_start": 5,
            "frame_end": 5,
            "slot_a": "G1",
            "slot_b": "G2",
        },
    )
    corrected = replay_annotations(_tracks(), events)
    clean, _manifest = build_clean_tracklets(corrected, events)
    h3 = clean[clean["human_identity"].eq("H3")].set_index("frame")
    assert h3.loc[1, "clean_tracklet_id"] != h3.loc[2, "clean_tracklet_id"]
    h1 = clean[clean["human_identity"].eq("H1")].set_index("frame")
    assert h1.loc[4, "clean_tracklet_id"] != h1.loc[5, "clean_tracklet_id"]
    assert h1.loc[4, "geometry_slot"] == "G1"
    assert h1.loc[5, "geometry_slot"] == "G2"


def test_mapping_anchor_restarts_known_mapping_after_ambiguity():
    events = _events(
        {
            "event_id": "E1",
            "event_type": "ambiguous",
            "frame_start": 2,
            "frame_end": 3,
        },
        {
            "event_id": "E2",
            "event_type": "mapping_anchor",
            "frame_start": 4,
            "frame_end": 4,
            "mapping_after": "G1=H3;G2=H1;G3=H2",
        },
    )
    corrected = replay_annotations(_tracks(), events)
    frame = corrected[corrected["frame"].eq(4)].set_index("geometry_slot")
    assert frame["human_identity"].to_dict() == {"G1": "H3", "G2": "H1", "G3": "H2"}

    mapping, active = mapping_at_frame(events, ["G1", "G2", "G3"], 3)
    assert mapping == {"G1": "H1", "G2": "H2", "G3": "H3"}
    assert active == ["E1"]
    mapping, active = mapping_at_frame(events, ["G1", "G2", "G3"], 4)
    assert mapping == {"G1": "H3", "G2": "H1", "G3": "H2"}
    assert active == []
    _mapping, _active, resolved = identity_state_at_frame(
        events, ["G1", "G2", "G3"], 4
    )
    assert resolved


def test_normalize_events_rejects_non_pair_swap():
    bad = _events(
        {
            "event_id": "E1",
            "event_type": "pair_swap",
            "frame_start": 4,
            "frame_end": 4,
            "slot_a": "G1",
            "slot_b": "G1",
        }
    )
    try:
        normalize_events(bad, ["G1", "G2", "G3"])
    except ValueError as error:
        assert "invalid pair_swap" in str(error)
    else:
        raise AssertionError("invalid self-swap was accepted")


def test_cannot_link_constraints_use_only_coexisting_fragments():
    corrected = replay_annotations(_tracks(), None)
    clean, _manifest = build_clean_tracklets(corrected, None)
    constraints = build_tracklet_constraints(clean)
    # G1 is split around its interpolated frame, so both fragments coexist
    # with G2/G3 but never with each other.
    assert len(constraints) == 5
    assert not (constraints["tracklet_a"] == constraints["tracklet_b"]).any()
    assert not (
        constraints.apply(frozenset, axis=1) == frozenset({"T000001", "T000002"})
    ).any()
