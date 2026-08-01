from pathlib import Path

import numpy as np
import pandas as pd

from scripts.reconstruction_3d.candidate_cache import CandidateWindow
from scripts.reconstruction_3d.mouth_ablation import (
    _tracked_to_direct,
    _write_filled_unshifted,
    relink_geometry,
    selected_geometry_to_best,
)
import pytest
from scripts.reconstruction_3d.windowed_tracklet_core import (
    enumerate_refractive_hypotheses,
)


class _FiniteViewCalibration:
    def evaluate_hypotheses(self, pixels, profile):
        assert profile == "production_fast"
        finite = np.isfinite(pixels).all(axis=2)
        means = np.nanmean(pixels, axis=1)
        xyz = np.column_stack([means, np.zeros(len(means))])
        residual = finite.sum(axis=1).astype(float)
        return xyz, residual


def _candidate_window(mouth: np.ndarray) -> CandidateWindow:
    score = np.where(np.all(np.isfinite(mouth), axis=2), 0.9, np.nan)
    return CandidateWindow(
        frame_start=0,
        frame_end=mouth.shape[0] - 1,
        mouth_xy=mouth,
        tail_xy=np.full_like(mouth, np.nan),
        instance_score=score,
        mouth_score=score,
        tail_score=np.full_like(score, np.nan),
        visibility=np.where(np.isfinite(score), 1, 0).astype(np.uint8),
        source_instance=np.where(np.isfinite(score), 0, -1).astype(np.int16),
    )


def test_declared_full_view_hypotheses_require_every_selected_mouth_finite():
    top = np.full((1, 4, 2), np.nan)
    side1 = np.full_like(top, np.nan)
    side2 = np.full_like(top, np.nan)
    top[0, :3] = [[1, 1], [2, 2], [3, 3]]
    side1[0, :2] = [[1, 1], [2, 2]]
    side2[0, :2] = [[1, 1], [2, 2]]
    windows = {
        "top_cam": _candidate_window(top),
        "side1_cam": _candidate_window(side1),
        "side2_cam": _candidate_window(side2),
    }
    tracklet_ids = {
        view: np.full((1, 4), -1, dtype=np.int64) for view in windows
    }

    hypotheses = enumerate_refractive_hypotheses(
        windows,
        tracklet_ids,
        _FiniteViewCalibration(),
        xyz_bounds=(-20, 220),
        batch_frames=1,
        include_two_view=True,
    )

    full = hypotheses[hypotheses.view_mask.eq(0b111)]
    assert (full.slot_cam0 < 3).all()
    assert (full.slot_cam1 < 2).all()
    assert (full.slot_cam2 < 2).all()
    pairs = hypotheses[hypotheses.view_count.eq(2)]
    assert len(pairs)
    assert (
        pairs[["slot_cam0", "slot_cam1", "slot_cam2"]].lt(0).sum(axis=1)
        == 1
    ).all()


def test_tracklet_ids_use_window_indices_after_first_evaluation_batch():
    mouth = np.asarray([[[1.0, 1.0]], [[2.0, 2.0]]])
    windows = {
        view: _candidate_window(mouth.copy())
        for view in ("top_cam", "side1_cam", "side2_cam")
    }
    tracklet_ids = {
        view: np.asarray([[10], [20]], dtype=np.int64)
        for view in windows
    }

    hypotheses = enumerate_refractive_hypotheses(
        windows,
        tracklet_ids,
        _FiniteViewCalibration(),
        xyz_bounds=(-20, 220),
        batch_frames=1,
        include_two_view=False,
    )

    assert hypotheses.set_index("frame").loc[0, "tracklet_top_cam"] == 10
    assert hypotheses.set_index("frame").loc[1, "tracklet_top_cam"] == 20


def test_selected_geometry_conversion_preserves_slots_and_mm2():
    selected = pd.DataFrame(
        [
            {
                "frame": 10,
                "view_count": 3,
                "ray_residual_mm2": 0.2,
                "hypothesis_id": "h0",
                "X": -1.0,
                "Y": 2.0,
                "Z": 3.0,
                "slot_cam0": 4,
                "slot_cam1": 2,
                "slot_cam2": 1,
            },
            {
                "frame": 10,
                "view_count": 2,
                "ray_residual_mm2": 0.1,
                "hypothesis_id": "h1",
                "X": 9.0,
                "Y": 8.0,
                "Z": 7.0,
                "slot_cam0": 0,
                "slot_cam1": -1,
                "slot_cam2": 5,
            },
        ]
    )
    best = selected_geometry_to_best(selected)[10]

    assert best["groups"]["a"] == [-1.0, 2.0, 3.0]
    assert best["perm_cam0"] == [4, 0, -1]
    assert best["perm_cam1"] == [2, -1, -1]
    assert best["errors"][:2] == [0.2, 0.1]
    assert np.isnan(best["errors"][2])
    assert best["synthetic_views"] == [-1, 1, -1]
    assert best["score_unit"] == "mm^2"


def test_owned_fill_preserves_calibrated_negative_coordinates(tmp_path: Path):
    direct = pd.DataFrame(
        [
            {
                "frame": frame,
                "fish_id": identity,
                "X": -5.0 + frame,
                "Y": 1.0,
                "Z": 2.0,
                "identity_status": "resolved",
                "identity_segment": 0,
            }
            for frame in (0, 2)
            for identity in ("ID_1", "ID_2", "ID_3")
        ]
    )
    output = _write_filled_unshifted(
        direct,
        output_path=tmp_path / "filled.csv",
        frame_start=0,
        frame_end=2,
        max_gap_frames=100,
    )

    interpolated = output[output.frame.eq(1)]
    assert (interpolated.X == -4.0).all()
    assert interpolated.is_interpolated.all()


def test_owned_fill_interpolates_explicit_internal_nan_rows(tmp_path: Path):
    direct = pd.DataFrame(
        [
            {
                "frame": frame,
                "fish_id": identity,
                "X": np.nan if frame == 1 else float(frame),
                "Y": np.nan if frame == 1 else 1.0,
                "Z": np.nan if frame == 1 else 2.0,
                "identity_status": "resolved",
                "identity_segment": 0,
            }
            for frame in (0, 1, 2)
            for identity in ("ID_1", "ID_2", "ID_3")
        ]
    )

    output = _write_filled_unshifted(
        direct,
        output_path=tmp_path / "filled_explicit_nan.csv",
        frame_start=0,
        frame_end=2,
        max_gap_frames=100,
    )

    interpolated = output[output.frame.eq(1)]
    assert np.isfinite(interpolated[["X", "Y", "Z"]]).all().all()
    assert (interpolated.X == 1.0).all()
    assert interpolated.is_interpolated.all()


def test_a2_rejects_tangent_or_acceleration_terms(tmp_path: Path):
    missing = tmp_path / "not_read.parquet"
    with pytest.raises(ValueError, match="tangent_weight must be 0"):
        relink_geometry(
            geometry_path=missing,
            output_dir=tmp_path,
            method="A2",
            tangent_weight=0.1,
        )
    with pytest.raises(ValueError, match="constant-velocity only"):
        relink_geometry(
            geometry_path=missing,
            output_dir=tmp_path,
            method="A2",
            acceleration_weight=0.1,
        )


def _geometry_row(frame, source_index, x):
    return {
        "frame": frame,
        "view_count": 3,
        "ray_residual_mm2": 0.1 + source_index / 100,
        "hypothesis_id": f"h{frame}_{source_index}",
        "X": x,
        "Y": float(source_index),
        "Z": 1.0,
        "slot_cam0": source_index,
        "slot_cam1": source_index,
        "slot_cam2": source_index,
    }


@pytest.mark.parametrize("method,velocity_weight", [("A1", 0.0), ("A2", 0.25)])
def test_relink_marks_all_fish_reseed_after_gap_over_budget(
    tmp_path: Path, method: str, velocity_weight: float
):
    geometry_path = tmp_path / f"{method.lower()}_geometry.parquet"
    pd.DataFrame(
        [
            _geometry_row(frame, source, source * 10.0 + frame / 204)
            for frame in (0, 102)
            for source in range(3)
        ]
    ).to_parquet(geometry_path, index=False)

    summary = relink_geometry(
        geometry_path=geometry_path,
        output_dir=tmp_path / method,
        method=method,
        frame_start=0,
        frame_end=102,
        velocity_weight=velocity_weight,
        max_gap_frames=100,
    )
    direct = pd.read_csv(tmp_path / method / f"{method.lower()}_direct.csv")
    filled = pd.read_csv(tmp_path / method / f"{method.lower()}_filled.csv")

    resumed = direct[direct.frame.eq(102)]
    assert resumed.identity_segment.tolist() == [1, 1, 1]
    assert set(resumed.identity_status) == {
        "new_segment_after_max_gap_reseed"
    }
    missing = direct[direct.frame.eq(50)]
    assert set(missing.identity_status) == {"unresolved_missing_geometry"}
    assert filled[filled.frame.eq(50)][["X", "Y", "Z"]].isna().all().all()
    assert summary["metrics"]["adjacent_steps_gt8_mm"] == 0
    assert summary["metrics"]["unresolved_row_fraction"] > 0


def test_relink_segments_only_the_fish_reseeded_after_partial_gap(
    tmp_path: Path,
):
    geometry_path = tmp_path / "partial_geometry.parquet"
    rows = [
        _geometry_row(0, source, source * 10.0)
        for source in range(3)
    ]
    rows.extend(
        _geometry_row(frame, source, source * 10.0 + frame / 1000)
        for frame in (50, 101)
        for source in (1, 2)
    )
    rows.extend(
        _geometry_row(102, source, source * 10.0 + 0.102)
        for source in range(3)
    )
    pd.DataFrame(rows).to_parquet(geometry_path, index=False)

    relink_geometry(
        geometry_path=geometry_path,
        output_dir=tmp_path / "partial",
        method="A2",
        frame_start=0,
        frame_end=102,
        velocity_weight=0.25,
        max_gap_frames=100,
    )
    direct = pd.read_csv(tmp_path / "partial" / "a2_direct.csv")
    resumed = direct[direct.frame.eq(102)].set_index("fish_id")

    assert resumed.loc["ID_1", "identity_segment"] == 1
    assert resumed.loc["ID_1", "identity_status"] == (
        "new_segment_after_max_gap_reseed"
    )
    assert resumed.loc["ID_2", "identity_segment"] == 0
    assert resumed.loc["ID_3", "identity_segment"] == 0


def test_direct_marks_partially_unassigned_identity_unresolved():
    tracked = {
        0: {
            "groups": {
                "ID_1": [np.nan, np.nan, np.nan],
                "ID_2": [1.0, 2.0, 3.0],
                "ID_3": [4.0, 5.0, 6.0],
            },
            "id_mapping": [-1, 0, 1],
            "identity_status": "resolved_after_lookahead",
            "identity_segment": 0,
            "synthetic_views": [-1, -1, -1],
        }
    }
    best = {
        0: {
            "errors": [0.1, 0.2, np.nan],
            "hypothesis_ids": ["h0", "h1", ""],
        }
    }

    direct = _tracked_to_direct(tracked, best).set_index("fish_id")

    assert direct.loc["ID_1", "identity_status"] == (
        "unresolved_unassigned_identity"
    )
    assert direct.loc["ID_2", "identity_status"] == "resolved_after_lookahead"
