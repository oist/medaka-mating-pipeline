from pathlib import Path

import numpy as np
from joblib import dump

from scripts.reconstruction_3d.calibration import Calibration
from scripts.reconstruction_3d.triangulate import compute_candidates


class OrthographicIm2XYZModel:
    def predict(self, arr):
        arr = np.asarray(arr, dtype=float)
        x = arr[:, 0]
        y = arr[:, 1]
        z = np.nanmean(arr[:, [3, 5]], axis=1)
        return np.column_stack([x, y, z])


class OrthographicXYZToImCoordsModel:
    def predict(self, arr):
        arr = np.asarray(arr, dtype=float)
        x = arr[:, 0]
        y = arr[:, 1]
        z = arr[:, 2]
        return np.column_stack([x, y, x, z, y, z])


class XYYZToXZModel:
    def predict(self, arr):
        arr = np.asarray(arr, dtype=float)
        return np.column_stack([arr[:, 0], arr[:, 3]])


class XZYZToXYModel:
    def predict(self, arr):
        arr = np.asarray(arr, dtype=float)
        return np.column_stack([arr[:, 0], arr[:, 2]])


class XZXYToYZModel:
    def predict(self, arr):
        arr = np.asarray(arr, dtype=float)
        return np.column_stack([arr[:, 3], arr[:, 1]])


def _write_models(root: Path, *, nested: bool, include_xyz2im: bool) -> Path:
    model_dir = root / "python_calibration_models" if nested else root
    model_dir.mkdir(parents=True, exist_ok=True)
    dump(OrthographicIm2XYZModel(), model_dir / "imCoords_to_XYZ.joblib")
    dump(XYYZToXZModel(), model_dir / "xy_yz_to_xz.joblib")
    dump(XZYZToXYModel(), model_dir / "xz_yz_to_xy.joblib")
    dump(XZXYToYZModel(), model_dir / "xz_xy_to_yz.joblib")
    if include_xyz2im:
        dump(OrthographicXYZToImCoordsModel(), model_dir / "XYZ_to_imCoords.joblib")
    return model_dir


def test_calibration_uses_pairwise_models_when_reprojection_model_is_missing(tmp_path):
    _write_models(tmp_path, nested=True, include_xyz2im=False)

    cal = Calibration(tmp_path)
    triplet = np.array([[10.0, 20.0], [10.0, 30.0], [20.0, 30.0]])
    bad_triplet = np.array([[10.0, 20.0], [14.0, 35.0], [20.0, 30.0]])

    assert cal.model_dir == tmp_path / "python_calibration_models"
    assert cal.has_pairwise_models
    assert not cal.has_registration_model
    assert cal.has_geometric_scoring
    assert np.allclose(
        cal.complete_triplet(np.array([[10.0, 20.0], [np.nan, np.nan], [20.0, 30.0]])),
        triplet,
    )
    assert cal.compute_triplet_registration_error(triplet) == 0.0
    assert cal.compute_triplet_registration_error(bad_triplet) > 0.0


def test_compute_candidates_recovers_one_missing_view_with_helper_models(tmp_path):
    _write_models(tmp_path, nested=False, include_xyz2im=True)
    cal = Calibration(tmp_path)

    points_by_frame = {
        0: {
            "cam0": [(1.0, 2.0), (4.0, 5.0), (7.0, 8.0)],
            "cam1": [(np.nan, np.nan), (4.0, 6.0), (7.0, 9.0)],
            "cam2": [(2.0, 3.0), (5.0, 6.0), (8.0, 9.0)],
        }
    }

    candidates = compute_candidates(cal, points_by_frame, allow_permutations=False)
    groups = candidates[0][0]["groups"]

    assert np.allclose(groups["a"], [1.0, 2.0, 3.0])
    assert np.allclose(groups["b"], [4.0, 5.0, 6.0])
    assert np.allclose(groups["c"], [7.0, 8.0, 9.0])
    assert np.isfinite(candidates[0][0]["cost"])
    assert candidates[0][0]["missing_views"] == [1, -1, -1]
    assert candidates[0][0]["recovered"] == [True, False, False]
