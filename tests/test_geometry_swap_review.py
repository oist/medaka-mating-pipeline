from __future__ import annotations

import pandas as pd

from scripts.analysis.geometry_swap_review import (
    IndexedSlpPredictions,
    atomic_write_events,
    build_review_candidates,
    load_extra_candidate_table,
    load_geometry_table,
    parse_slot_specs,
    draw_raw_sleap_gray,
)
import numpy as np


def _geometry() -> pd.DataFrame:
    rows = []
    for frame in range(10):
        positions = {
            "G1": (float(frame), 0.0, 0.0),
            "G2": (float(9 - frame), 0.0, 0.0),
            "G3": (50.0, 50.0, 50.0),
        }
        for slot, xyz in positions.items():
            rows.append(
                {
                    "frame": frame,
                    "geometry_slot": slot,
                    "X": xyz[0],
                    "Y": xyz[1],
                    "Z": xyz[2],
                    "is_interpolated": slot == "G3" and frame in (4, 5),
                }
            )
    return pd.DataFrame(rows)


def test_candidate_builder_finds_crossing_and_interpolation_boundaries():
    candidates = build_review_candidates(
        _geometry(), close_mm=2.0, jump_mm=8.0, merge_frames=0,
        min_interpolation_frames=1,
    )
    assert candidates["source"].str.contains("close_approach").any()
    assert candidates["source"].str.contains("interpolation_boundary").sum() == 2
    close = candidates[candidates["source"].str.contains("close_approach")].iloc[0]
    assert "G1<->G2" in close["slots"]
    assert int(close["frame"]) in (4, 5)


def test_candidate_builder_imports_external_frame_columns():
    extra = pd.DataFrame({"center_frame": [2, 8]})
    candidates = build_review_candidates(
        _geometry(),
        close_mm=0.1,
        merge_frames=0,
        extra_candidate_tables=[("known_jumps", extra)],
    )
    imported = candidates[candidates["source"].str.contains("external:known_jumps")]
    assert imported["frame"].tolist() == [2, 8]


def test_motion_qc_json_loader_uses_only_final_jump_frames(tmp_path):
    path = tmp_path / "motion_qc.json"
    path.write_text(
        '{"speed_qc":{"after":{"per_id":{"F":{"jump_frames":[8,2]},'
        '"M1":{"jump_frames":[2,5]}}}}}',
        encoding="utf-8",
    )
    assert load_extra_candidate_table(path)["frame"].tolist() == [2, 5, 8]


def test_load_geometry_assigns_slots_and_parses_interpolation(tmp_path):
    path = tmp_path / "geometry.csv"
    pd.DataFrame(
        {
            "frame": [0, 0, 0],
            "fish_id": ["ID_2", "ID_1", "ID_3"],
            "X": [1, 2, 3],
            "Y": [1, 2, 3],
            "Z": [1, 2, 3],
            "is_interpolated": ["False", "True", "False"],
        }
    ).to_csv(path, index=False)
    geometry, mapping = load_geometry_table(path)
    assert mapping == {"ID_1": "G1", "ID_2": "G2", "ID_3": "G3"}
    assert geometry.loc[geometry["geometry_source_id"].eq("ID_1"), "is_interpolated"].item()


def test_atomic_event_save_is_resumable(tmp_path):
    path = tmp_path / "events.csv"
    events = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "event_type": "pair_swap",
                "frame_start": 5,
                "frame_end": 5,
                "slot_a": "G1",
                "slot_b": "G2",
            }
        ]
    )
    atomic_write_events(events, path, ["G1", "G2", "G3"])
    saved = pd.read_csv(path, keep_default_na=False)
    assert saved.loc[0, "event_type"] == "pair_swap"
    assert saved.loc[0, "view_scope"] == "all"
    assert not path.with_suffix(".csv.tmp").exists()


def test_slot_spec_parser_rejects_duplicate_display_slots():
    try:
        parse_slot_specs(["ID_1=G1", "ID_2=G1"])
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate display slots were accepted")


def test_raw_sleap_predictions_are_drawn_in_gray():
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    instances = np.array([[[10.0, 10.0], [20.0, 10.0]]])
    draw_raw_sleap_gray(image, instances, 1.0)
    assert tuple(image[10, 10]) == (165, 165, 165)


def test_indexed_slp_predictions_read_only_requested_frame_and_nodes(tmp_path):
    import h5py
    import json

    path = tmp_path / "predictions.slp"
    frame_dtype = np.dtype(
        [
            ("frame_id", "<u8"),
            ("video", "<u4"),
            ("frame_idx", "<u8"),
            ("instance_id_start", "<u8"),
            ("instance_id_end", "<u8"),
        ]
    )
    instance_dtype = np.dtype(
        [
            ("instance_id", "<i8"),
            ("instance_type", "u1"),
            ("frame_id", "<u8"),
            ("skeleton", "<u4"),
            ("track", "<i4"),
            ("from_predicted", "<i8"),
            ("score", "<f4"),
            ("point_id_start", "<u8"),
            ("point_id_end", "<u8"),
            ("tracking_score", "<f4"),
        ]
    )
    point_dtype = np.dtype(
        [
            ("x", "<f8"),
            ("y", "<f8"),
            ("visible", "?"),
            ("complete", "?"),
            ("score", "<f8"),
        ]
    )
    frames = np.array([(0, 0, 5, 0, 2)], dtype=frame_dtype)
    instances = np.array(
        [
            (0, 1, 0, 0, -1, -1, 0.9, 0, 2, np.nan),
            (1, 1, 0, 0, -1, -1, 0.8, 2, 4, np.nan),
        ],
        dtype=instance_dtype,
    )
    points = np.array(
        [
            (10, 11, True, False, 0.9),
            (20, 21, True, False, 0.8),
            (30, 31, True, False, 0.7),
            (40, 41, False, False, 0.6),
        ],
        dtype=point_dtype,
    )
    metadata = {
        "nodes": [{"name": "mouth"}, {"name": "tail"}],
        "skeletons": [{"nodes": [{"id": 0}, {"id": 1}]}],
    }
    with h5py.File(path, "w") as file:
        file.create_dataset("frames", data=frames)
        file.create_dataset("instances", data=instances)
        file.create_dataset("pred_points", data=points)
        group = file.create_group("metadata")
        group.attrs["json"] = json.dumps(metadata).encode()

    source = IndexedSlpPredictions(path)
    try:
        assert source.get(4) is None
        loaded = source.get(5)
        np.testing.assert_allclose(loaded[0], [[10, 11], [20, 21]])
        np.testing.assert_allclose(loaded[1, 0], [30, 31])
        assert np.isnan(loaded[1, 1]).all()
    finally:
        source.close()
