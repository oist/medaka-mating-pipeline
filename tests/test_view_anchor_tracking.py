import numpy as np

from scripts.reconstruction_3d.cross_frame_tracking import apply_cross_frame_tracking
from scripts.reconstruction_3d.view_anchor_tracking import (
    build_source_id_hints,
    infer_anchor_identity_map,
    match_tracks_to_slots,
)


def _frame(points, perm=(0, 1, 2)):
    return {
        "groups": {
            "a": list(points[0]),
            "b": list(points[1]),
            "c": list(points[2]),
        },
        "perm_cam0": list(perm),
        "perm_cam1": list(perm),
        "perm_cam2": list(perm),
    }


def test_track_slot_match_requires_complete_separated_unique_tracks():
    slots = np.array([[100, 100], [300, 100], [500, 100], [900, 900]], float)
    tracks = np.array([[501, 100], [99, 100], [301, 100]], float)
    matched = match_tracks_to_slots(
        slots,
        tracks,
        ["fish_c", "fish_a", "fish_b"],
        num_individuals=3,
        match_distance_px=5,
        min_separation_px=60,
    )
    assert matched == {2: "fish_c", 0: "fish_a", 1: "fish_b"}

    too_close = tracks.copy()
    too_close[1] = too_close[0] + 20
    assert match_tracks_to_slots(
        slots,
        too_close,
        ["fish_c", "fish_a", "fish_b"],
        num_individuals=3,
        match_distance_px=500,
        min_separation_px=60,
    ) is None


def test_anchor_identity_map_respects_permuted_top_slots():
    best = {10: _frame(((30, 0, 0), (10, 0, 0), (20, 0, 0)), perm=(2, 0, 1))}
    anchors = {10: {0: "track_left", 1: "track_mid", 2: "track_right"}}
    identity_map, seed = infer_anchor_identity_map(best, anchors, num_individuals=3)
    assert seed == 10
    assert identity_map == {
        "track_left": "ID_1",
        "track_mid": "ID_2",
        "track_right": "ID_3",
    }
    assert build_source_id_hints(
        best, anchors, identity_map, num_individuals=3
    ) == {10: {0: "ID_3", 1: "ID_1", 2: "ID_2"}}


def test_view_hint_overrides_ambiguous_3d_gate_only_for_hinted_source():
    best = {
        0: _frame(((0, 0, 0), (100, 0, 0), (200, 0, 0))),
        1: _frame(((50, 0, 0), (101, 0, 0), (201, 0, 0))),
    }
    unanchored = apply_cross_frame_tracking(
        best, max_jump_mm=8, max_jump_hard_cap_mm=80
    )
    assert np.isnan(unanchored[1]["groups"]["ID_1"]).all()

    anchored = apply_cross_frame_tracking(
        best,
        max_jump_mm=8,
        max_jump_hard_cap_mm=80,
        source_id_hints_by_frame={1: {0: "ID_1"}},
    )
    assert anchored[1]["groups"]["ID_1"] == [50.0, 0.0, 0.0]
    assert anchored[1]["groups"]["ID_2"] == [101.0, 0.0, 0.0]
    assert anchored[1]["groups"]["ID_3"] == [201.0, 0.0, 0.0]
    assert anchored[1]["id_anchor"] == [True, False, False]
