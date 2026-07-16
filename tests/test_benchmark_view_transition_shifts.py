import numpy as np

from scripts.calibration.benchmark_view_transition_shifts import (
    RayBackend,
    extract_assignment_arrays,
    find_bounded_transition_runs,
    fit_affine_shift_correction,
    observation_state,
    transition_event_rows,
)
from scripts.calibration.refractive_geometry import FlatPortCamera
from scripts.calibration.evaluate_transition_weights_rigid_object import load_correction


def _camera(name, center, rotation_vector):
    return FlatPortCamera(
        name=name,
        fx=100.0,
        fy=100.0,
        cx=0.0,
        cy=0.0,
        k1=0.0,
        k2=0.0,
        rotation_vector=list(rotation_vector),
        camera_center=list(center),
        interface_normal=[0.0, 0.0, 1.0],
        interface_point=[0.0, 0.0, 0.0],
        glass_thickness_mm=0.0,
        layer_count=0,
    )


def test_extract_assignment_arrays_maps_slots_and_id_ordered_synthetic_views():
    payload = {
        "id_labels": ["ID_1", "ID_2"],
        "frames": {
            "3": {
                "cam0_to_id": ["ID_2", "ID_1"],
                "cam1_to_id": ["ID_1", None],
                "cam2_to_id": ["ID_2", None],
                "synthetic_views": [2, 1],
            }
        },
    }
    identities, slots, synthetic, saved_xyz = extract_assignment_arrays(payload)
    assert identities == ["ID_1", "ID_2"]
    np.testing.assert_array_equal(slots[3, 0], [1, 0, -1])
    np.testing.assert_array_equal(slots[3, 1], [0, -1, 0])
    np.testing.assert_array_equal(synthetic[3], [2, 1])
    assert np.isnan(saved_xyz[3]).all()


def test_extract_assignment_arrays_rejects_legacy_missing_provenance():
    payload = {
        "id_labels": ["ID_1"],
        "frames": {
            "0": {
                "cam0_to_id": ["ID_1"],
                "cam1_to_id": ["ID_1"],
                "cam2_to_id": [None],
                "groups": {"ID_1": [1.0, 2.0, 3.0]},
            }
        },
    }
    import pytest

    with pytest.raises(ValueError, match="lacks ID-ordered synthetic_views"):
        extract_assignment_arrays(payload)


def test_observation_state_and_bounded_transition_runs():
    coordinates = np.full((8, 1, 3, 2), np.nan)
    coordinates[[0, 1, 4, 5, 6, 7], 0] = 1.0
    coordinates[2:4, 0, [0, 2]] = 1.0
    synthetic = np.full((8, 1), -1, dtype=np.int8)
    synthetic[2:4, 0] = 1
    state = observation_state(coordinates, synthetic)
    np.testing.assert_array_equal(state[:, 0], [3, 3, 1, 1, 3, 3, 3, 3])
    runs = find_bounded_transition_runs(state)
    assert len(runs) == 1
    assert (runs[0].missing_view, runs[0].start_frame, runs[0].end_frame) == (
        1,
        2,
        3,
    )


def test_affine_shift_correction_recovers_position_dependent_bias():
    rng = np.random.default_rng(3)
    two = rng.normal(size=(500, 3)) * [20.0, 10.0, 5.0] + [90.0, 80.0, 70.0]
    matrix = np.array(
        [[0.01, 0.00, 0.00], [0.00, -0.02, 0.00], [0.00, 0.00, 0.03]]
    )
    three = two + two @ matrix + [0.4, -0.2, 0.1]
    correction = fit_affine_shift_correction(two, three, ridge_alpha=1e-6)
    error = np.linalg.norm(correction.correct(two) - three, axis=1)
    assert np.percentile(error, 95) < 1e-4


def test_ray_backend_two_views_intersect_known_point():
    # Camera 0 looks +Z. Camera 1 is placed on +X and looks approximately -X.
    cam0 = _camera("cam0", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    cam1 = _camera("cam1", [10.0, 0.0, 5.0], [0.0, -np.pi / 2, 0.0])
    cam2 = _camera("cam2", [0.0, 10.0, 5.0], [np.pi / 2, 0.0, 0.0])
    backend = RayBackend("test", (cam0, cam1, cam2))
    point = np.array([0.0, 0.0, 5.0])

    def project(camera):
        camera_xyz = camera.rotation_matrix @ (point - np.asarray(camera.camera_center))
        return 100.0 * camera_xyz[:2] / camera_xyz[2]

    pixels = np.stack([project(camera) for camera in (cam0, cam1, cam2)])[None]
    reconstructed = backend.reconstruct(pixels, (0, 1))[0]
    np.testing.assert_allclose(reconstructed, point, atol=1e-8)


def test_natural_event_split_is_holdout_only_when_whole_window_is_holdout():
    positions = np.zeros((5, 1, 3), dtype=float)
    positions[:, 0, 0] = np.arange(5)
    runs = find_bounded_transition_runs(np.array([[3], [3], [0], [3], [3]]))
    clean = np.ones((5, 1), dtype=bool)
    _frames, events = transition_event_rows(
        "model", ["ID_1"], runs, positions, positions, clean, np.ones(5, dtype=bool)
    )
    assert events[0]["split"] == "holdout"
    holdout = np.ones(5, dtype=bool)
    holdout[1] = False
    _frames, events = transition_event_rows(
        "model", ["ID_1"], runs, positions, positions, clean, holdout
    )
    assert events[0]["split"] == "fit_or_boundary"


def test_weight_payload_round_trip():
    correction = fit_affine_shift_correction(
        np.arange(90, dtype=float).reshape(30, 3),
        np.arange(90, dtype=float).reshape(30, 3) + [1.0, 2.0, 3.0],
        ridge_alpha=1e-6,
    )
    loaded = load_correction(correction.to_dict())
    probe = np.array([[4.0, 5.0, 6.0]])
    np.testing.assert_allclose(loaded.correct(probe), correction.correct(probe))
