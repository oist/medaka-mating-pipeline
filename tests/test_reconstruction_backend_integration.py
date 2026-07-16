from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.reconstruction_3d.reconstruct_from_slp as reconstruction
from scripts.reconstruction_3d.bind_topcam_id import bind
from scripts.reconstruction_3d.reconstruct_from_slp import (
    _calibration_metadata,
    _resolve_score_gate,
    save_best_pre_link_json_to,
)
from scripts.reconstruction_3d.triangulate import compute_candidates
from scripts.reconstruction_3d.triangulate_extra_candidates import (
    compute_best_assignments_per_frame,
)


class _OrthographicPublicBackend:
    def __init__(self, backend_name: str = "polynomial") -> None:
        self.backend_name = backend_name
        self.score_unit = "mm^2" if backend_name == "refractive" else "px^2"
        self.provenance = {
            "model_files": [f"{backend_name}.json"],
            "score_definitions": {
                "production_fast": "orthographic_test_consistency",
                "legacy_combined": "orthographic_test_consistency",
            },
        }
        self.capabilities = {
            "triangulation": True,
            "three_view_triangulation": True,
            "two_view_triangulation": True,
            "direct_two_view_triangulation": backend_name == "refractive",
            "projection": True,
            "hypothesis_scoring": True,
        }
        self.evaluate_calls: list[tuple[tuple[int, ...], str]] = []
        self.project_calls = 0

    def require_capability(self, key: str) -> None:
        if not self.capabilities.get(key, False):
            raise RuntimeError(key)

    @staticmethod
    def _evaluate(observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        obs = np.asarray(observations, dtype=float)
        valid = np.isfinite(obs).all(axis=-1)
        enough = valid.sum(axis=-1) >= 2

        def pair_mean(a, b):
            stacked = np.stack([a, b], axis=-1)
            finite = np.isfinite(stacked)
            numerator = np.where(finite, stacked, 0.0).sum(axis=-1)
            denominator = finite.sum(axis=-1)
            return np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, np.nan),
                where=denominator > 0,
            )

        x = pair_mean(obs[..., 0, 0], obs[..., 1, 0])
        y = pair_mean(obs[..., 0, 1], obs[..., 2, 0])
        z = pair_mean(obs[..., 1, 1], obs[..., 2, 1])
        xyz = np.stack([x, y, z], axis=-1)
        xyz = np.where(enough[..., None], xyz, np.nan)

        terms = np.stack(
            [
                (obs[..., 0, 0] - obs[..., 1, 0]) ** 2,
                (obs[..., 0, 1] - obs[..., 2, 0]) ** 2,
                (obs[..., 1, 1] - obs[..., 2, 1]) ** 2,
            ],
            axis=-1,
        )
        finite_terms = np.isfinite(terms)
        scores = np.where(finite_terms, terms, 0.0).sum(axis=-1)
        scores = np.where(enough, scores, np.inf)
        return xyz, scores

    def evaluate_hypotheses(self, observations, *, profile):
        observations = np.asarray(observations, dtype=float)
        self.evaluate_calls.append((observations.shape, profile))
        return self._evaluate(observations)

    def triangulate(self, observations):
        return self._evaluate(np.asarray(observations, dtype=float))[0]

    def score_hypotheses(self, observations, *, profile):
        return self._evaluate(np.asarray(observations, dtype=float))[1]

    def project_world(self, xyz):
        self.project_calls += 1
        xyz = np.asarray(xyz, dtype=float)
        return np.stack(
            [xyz[..., [0, 1]], xyz[..., [0, 2]], xyz[..., [1, 2]]], axis=-2
        )


def _camera_points(*, include_reflections: bool = False) -> dict[str, list[tuple[float, float]]]:
    world = [(10.0, 20.0, 30.0), (50.0, 60.0, 70.0), (90.0, 100.0, 110.0)]
    points = {
        "cam0": [(x, y) for x, y, _ in world],
        "cam1": [(x, z) for x, _, z in world],
        "cam2": [(y, z) for _, y, z in world],
    }
    if include_reflections:
        points["cam0"].extend([(500.0, 500.0), (550.0, 550.0), (600.0, 600.0)])
        points["cam1"].extend([(500.0, 500.0), (550.0, 550.0), (600.0, 600.0)])
        points["cam2"].extend([(500.0, 500.0), (550.0, 550.0), (600.0, 600.0)])
    return points


def test_standard_selector_batches_and_uses_legacy_profile() -> None:
    backend = _OrthographicPublicBackend()
    candidates = compute_candidates(
        backend,
        {0: _camera_points()},
        num_individuals=3,
        allow_permutations=True,
    )
    assert len(backend.evaluate_calls) == 1
    assert backend.evaluate_calls[0][1] == "legacy_combined"
    assert backend.evaluate_calls[0][0] == (108, 3, 2)
    assert candidates[0][0]["scoring_profile"] == "legacy_combined"


@pytest.mark.parametrize("backend_name", ["polynomial", "refractive"])
def test_fast_selector_preserves_global_exclusivity_and_bounds(backend_name: str) -> None:
    backend = _OrthographicPublicBackend(backend_name)
    best = compute_best_assignments_per_frame(
        backend,
        {4: _camera_points(include_reflections=True)},
        xyz_bounds=(0.0, 200.0),
        num_individuals=3,
        num_candidates=6,
        max_hypothesis_score=1.0,
        global_max_states=None,
    )[4]
    assert backend.evaluate_calls[0] == ((216, 3, 2), "production_fast")
    for camera in range(3):
        slots = [slot for slot in best[f"perm_cam{camera}"] if slot >= 0]
        assert len(slots) == len(set(slots)) == 3
        assert max(slots) < 3
    assert all(
        np.all((np.asarray(point) >= 0.0) & (np.asarray(point) <= 200.0))
        for point in best["groups"].values()
    )
    assert best["score_unit"] == backend.score_unit
    assert best["scoring_profile"] == "production_fast"


def test_dense_frame_batching_is_equivalent_and_reduces_backend_calls() -> None:
    points = {frame: _camera_points() for frame in range(8)}
    scalar_backend = _OrthographicPublicBackend("refractive")
    scalar = compute_best_assignments_per_frame(
        scalar_backend,
        points,
        xyz_bounds=(0.0, 200.0),
        num_individuals=3,
        num_candidates=3,
        global_max_states=None,
        evaluation_batch_frames=1,
    )
    batched_backend = _OrthographicPublicBackend("refractive")
    batched = compute_best_assignments_per_frame(
        batched_backend,
        points,
        xyz_bounds=(0.0, 200.0),
        num_individuals=3,
        num_candidates=3,
        global_max_states=None,
        evaluation_batch_frames=4,
    )
    assert json.dumps(scalar, sort_keys=True) == json.dumps(batched, sort_keys=True)
    assert len(scalar_backend.evaluate_calls) == 8
    assert len(batched_backend.evaluate_calls) == 2


def test_polynomial_evaluation_keeps_historical_per_frame_batch_shape() -> None:
    backend = _OrthographicPublicBackend("polynomial")
    compute_best_assignments_per_frame(
        backend,
        {frame: _camera_points() for frame in range(4)},
        xyz_bounds=(0.0, 200.0),
        num_individuals=3,
        num_candidates=3,
        global_max_states=None,
        evaluation_batch_frames=256,
    )
    assert len(backend.evaluate_calls) == 4


def test_backend_specific_score_gates_reject_mixed_units() -> None:
    refractive = _OrthographicPublicBackend("refractive")
    assert _resolve_score_gate(
        refractive,
        max_residual_per_fish=None,
        max_residual_px2=None,
        max_ray_residual_mm2=0.5,
    ) == (0.5, "mm^2", "--max-ray-residual-mm2")
    with pytest.raises(ValueError, match="never converted"):
        _resolve_score_gate(
            refractive,
            max_residual_per_fish=None,
            max_residual_px2=5000.0,
            max_ray_residual_mm2=None,
        )


def test_pre_link_json_persists_backend_score_and_missing_view_provenance(tmp_path: Path) -> None:
    backend = _OrthographicPublicBackend("refractive")
    metadata = _calibration_metadata(
        backend,
        backend,
        score_gate=0.5,
        score_gate_unit="mm^2",
        score_gate_option="--max-ray-residual-mm2",
        scoring_profile="production_fast",
    )
    best = {
        0: {
            "groups": {"a": [1.0, 2.0, 3.0]},
            "errors": {"a": 0.1},
            "perm_cam0": [0],
            "perm_cam1": [0],
            "perm_cam2": [-1],
            "recovered": [True],
            "missing_views": [2],
            "synthetic_views": [2],
            "score_unit": "mm^2",
            "triangulation_backend": "refractive",
            "scoring_backend": "refractive",
            "scoring_profile": "production_fast",
        }
    }
    path = tmp_path / "best_pre_link.json"
    save_best_pre_link_json_to(
        best, path, num_individuals=1, calibration_metadata=metadata
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["calibration"]["assignment_scoring"]["score_unit"] == "mm^2"
    assert payload["calibration"]["score_gate"]["value"] == pytest.approx(0.5)
    assert payload["frames"]["0"]["missing_views"] == [2]
    assert payload["frames"]["0"]["scoring_profile"] == "production_fast"


def test_bind_projects_geometry_in_one_batch_and_writes_provenance(tmp_path: Path) -> None:
    geometry = tmp_path / "geometry.csv"
    tracks = tmp_path / "top.csv"
    output = tmp_path / "identified.csv"
    pd.DataFrame(
        [
            {"frame": frame, "fish_id": "a", "X": 10 + frame, "Y": 20, "Z": 30}
            for frame in range(4)
        ]
    ).to_csv(geometry, index=False)
    pd.DataFrame(
        [
            {"frame": frame, "track_id": 0, "x": 10 + frame, "y": 20}
            for frame in range(4)
        ]
    ).to_csv(tracks, index=False)
    backend = _OrthographicPublicBackend("refractive")
    result = bind(
        geometry,
        tracks,
        calib=backend,
        out_csv=output,
        bind_distance=1.0,
        require_min_frames=1,
    )
    assert backend.project_calls == 1
    assert set(result["bind_source"]) == {"top_cam_2d"}
    provenance_path = Path(result.attrs["projection_provenance_path"])
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["projection_backend"] == "refractive"
    assert provenance["bind_gate"] == {"value": 1.0, "unit": "px"}


@pytest.mark.parametrize(
    ("backend_name", "gate_kwargs"),
    [
        ("polynomial", {"max_residual_px2": 1.0}),
        ("refractive", {"max_ray_residual_mm2": 1.0}),
    ],
)
def test_fast_six_candidate_reconstruction_persists_resolved_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend_name: str,
    gate_kwargs: dict[str, float],
) -> None:
    backend = _OrthographicPublicBackend(backend_name)
    monkeypatch.setattr(
        reconstruction,
        "Calibration",
        lambda _folder, *, backend: globals_backend,
    )
    monkeypatch.setattr(
        reconstruction,
        "load_points_by_frame",
        lambda *_args, **_kwargs: {0: _camera_points(include_reflections=True)},
    )
    globals_backend = backend
    output_dir = tmp_path / backend_name
    result = reconstruction.run_reconstruction_from_slp(
        "experiment",
        slp_dir=tmp_path,
        model_folder=tmp_path,
        calibration_backend=backend_name,
        output_dir=output_dir,
        num_individuals=3,
        num_candidates=6,
        xyz_bounds=(0.0, 200.0),
        fill_missing=False,
        shift_to_positive=False,
        plot=False,
        **gate_kwargs,
    )
    provenance_path = Path(result["tracklet_artifacts"]["calibration_provenance"])
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["triangulation"]["backend"] == backend_name
    assert provenance["assignment_scoring"]["scoring_profile"] == "production_fast"
    assert Path(result["best_pre_link_json_path"]).exists()
