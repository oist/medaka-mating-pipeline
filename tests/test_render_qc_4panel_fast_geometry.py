from __future__ import annotations

import pandas as pd

from scripts.visualization import render_qc_4panel_fast as renderer
import numpy as np


def test_indexed_slp_loader_reads_only_requested_episode_and_closes(monkeypatch, tmp_path):
    from scripts.analysis import geometry_swap_review

    calls = {"frames": [], "closed": False}

    class FakeIndexedPredictions:
        def __init__(self, path, *, node_names, cache_size):
            assert path == tmp_path / "view.slp"
            assert node_names == ("mouth", "tail")
            assert cache_size == 3

        def get(self, frame):
            calls["frames"].append(frame)
            if frame == 3:
                return None
            return np.full((1, 2, 2), float(frame))

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(
        geometry_swap_review, "IndexedSlpPredictions", FakeIndexedPredictions
    )
    frames, nodes = renderer.load_indexed_slp_node_xy(
        tmp_path / "view.slp",
        node_names=("mouth", "tail"),
        start_frame=2,
        end_frame=4,
    )
    assert calls == {"frames": [2, 3, 4], "closed": True}
    assert list(frames) == [2, 4]
    assert nodes == ["mouth", "tail"]


def test_parse_identity_specs_preserves_requested_display_order():
    mapping = renderer.parse_identity_specs(["ID_2=G2", "ID_1=G1", "ID_3=G3"])
    assert list(mapping.items()) == [
        ("ID_2", "G2"),
        ("ID_1", "G1"),
        ("ID_3", "G3"),
    ]


def test_geometry_track_loader_uses_fish_id_and_interpolation_confidence(tmp_path):
    path = tmp_path / "geometry.csv"
    pd.DataFrame(
        [
            {"frame": 0, "fish_id": "ID_1", "X": 1, "Y": 2, "Z": 3, "is_interpolated": False},
            {"frame": 1, "fish_id": "ID_1", "X": 2, "Y": 3, "Z": 4, "is_interpolated": True},
            {"frame": 0, "fish_id": "ID_2", "X": 5, "Y": 6, "Z": 7, "is_interpolated": False},
            {"frame": 0, "fish_id": "ID_3", "X": 8, "Y": 9, "Z": 10, "is_interpolated": False},
        ]
    ).to_csv(path, index=False)
    mapping = {"ID_1": "G1", "ID_2": "G2", "ID_3": "G3"}
    try:
        renderer.configure_identities(mapping)
        pos, confident = renderer.load_track(
            path, 3, id_column="fish_id", identity_map=mapping
        )
        assert pos["G1"][0].tolist() == [1.0, 2.0, 3.0]
        assert confident["G1"].tolist() == [True, False, False]
        assert confident["G2"][0]
    finally:
        renderer.configure_identities(dict(renderer.DEFAULT_IDENTITIES))


def test_geometry_matches_instance_centroid_but_returns_mouth_instance():
    # G1 is close to instance 0's body centroid but closer to instance 1's mouth.
    # Matching on mouths would therefore choose the wrong fish.
    instances = np.array(
        [
            [[0.0, 0.0], [10.0, 0.0]],   # centroid x=5
            [[6.0, 0.0], [20.0, 0.0]],   # centroid x=13
        ]
    )
    centers = renderer.visible_node_centroids(instances)
    np.testing.assert_allclose(centers, [[5.0, 0.0], [13.0, 0.0]])
    assigned = renderer.match_projected_identities_to_instances(
        {"G1": np.array([5.0, 0.0]), "G2": np.array([13.0, 0.0])},
        instances,
        gate_px=20.0,
    )
    assert assigned == {"G1": 0, "G2": 1}


def test_mouth_and_tail_triangulation_use_public_calibration_backend():
    class PublicCalibrationOnly:
        def __init__(self):
            self.calls = []

        def triangulate(self, observations):
            values = np.asarray(observations, dtype=float)
            self.calls.append(values.copy())
            xy = values.mean(axis=1)
            return np.column_stack((xy, xy[:, :1]))

    mapping = {"ID_1": "G1"}
    try:
        renderer.configure_identities(mapping)
        # Reprojected track sits at the visible-node centroid, while the
        # returned pose retains mouth/tail endpoints for body-axis geometry.
        reproj = {
            "G1": {view: np.array([[1.0, 1.0]]) for view in renderer.VIEWS}
        }
        instance = np.array([[[0.0, 0.0], [2.0, 2.0]]])
        slp = {view: {0: instance.copy()} for view in renderer.VIEWS}
        calibration = PublicCalibrationOnly()

        visible, mouth_xyz, tail_xyz = renderer.precompute_overlay(
            reproj, slp, calibration, 0, 0, 1, gate=10.0
        )

        assert len(calibration.calls) == 2
        assert calibration.calls[0].shape == (1, 3, 2)
        np.testing.assert_allclose(visible["G1"]["top_cam"][0], instance[0])
        np.testing.assert_allclose(mouth_xyz["G1"][0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(tail_xyz["G1"][0], [2.0, 2.0, 2.0])
    finally:
        renderer.configure_identities(dict(renderer.DEFAULT_IDENTITIES))


def test_mouth_trail_indices_use_source_time_and_include_current_frame():
    np.testing.assert_array_equal(
        renderer._trail_source_indices(frame=1000, trail_frames=830, stride=5),
        np.arange(1000, 170, -5)[::-1],
    )
    indices = renderer._trail_source_indices(
        frame=1000, trail_frames=830, stride=5
    )
    assert indices[-1] == 1000
    assert indices[0] == 175
    assert len(indices) == 166


def test_trail_sample_stride_is_independent_of_rendered_frame_step():
    assert renderer._resolve_trail_stride(frame_step=1, trail_sample_step=5) == 5
    assert renderer._resolve_trail_stride(frame_step=1, trail_sample_step=None) == 1
    assert renderer._resolve_trail_stride(frame_step=5, trail_sample_step=None) == 5


def test_3d_panel_draws_direct_mouth_trail_and_tail_to_mouth_arrow():
    mapping = {"ID_1": "G1"}
    try:
        renderer.configure_identities(mapping)
        projector = renderer.Projector3D(160, -20.0, 220.0, 22.0, -60.0)
        base = projector.base_canvas()
        centroid = {"G1": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])}
        confident = {"G1": np.array([True, True])}
        mouth = {"G1": np.array([[10.0, 10.0, 10.0], [11.0, 11.0, 11.0]])}
        tail = {"G1": np.array([[8.0, 8.0, 8.0], [9.0, 9.0, 9.0]])}

        image = renderer.draw_3d_panel(
            projector,
            base,
            centroid,
            confident,
            mouth,
            tail,
            1,
            2,
            trail_stride=1,
            arrow_t=1,
        )

        assert image.shape == base.shape
        assert np.any(image != base)
    finally:
        renderer.configure_identities(dict(renderer.DEFAULT_IDENTITIES))


def test_2d_identity_marker_is_at_mouth_never_reprojected_centroid():
    mapping = {"ID_1": "G1"}
    try:
        renderer.configure_identities(mapping)
        image = np.full((80, 80, 3), 255, dtype=np.uint8)
        reproj = {"G1": {"top_cam": np.array([[50.0, 50.0]])}}
        vis2d = {
            "G1": {"top_cam": np.array([[[10.0, 10.0], [20.0, 10.0]]])}
        }
        renderer.draw_2d_view(image, "top_cam", 0, reproj, vis2d, 1.0)
        assert tuple(image[10, 10]) == renderer.ID_BGR["G1"]
        assert tuple(image[50, 50]) == (255, 255, 255)

        center_only = np.full((80, 80, 3), 255, dtype=np.uint8)
        renderer.draw_2d_view(
            center_only,
            "top_cam",
            0,
            reproj,
            vis2d,
            1.0,
            center_only=True,
        )
        assert tuple(center_only[10, 10]) == renderer.ID_BGR["G1"]
        assert tuple(center_only[10, 20]) == (255, 255, 255)

        no_mouth = np.full((80, 80, 3), 255, dtype=np.uint8)
        vis2d["G1"]["top_cam"][0, 0] = np.nan
        renderer.draw_2d_view(no_mouth, "top_cam", 0, reproj, vis2d, 1.0)
        assert np.all(no_mouth == 255)
    finally:
        renderer.configure_identities(dict(renderer.DEFAULT_IDENTITIES))
