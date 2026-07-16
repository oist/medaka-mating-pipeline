from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.tracking import fit_interpretable_association_cost as fitter


def _synthetic_clean_track(n_frames: int = 60) -> pd.DataFrame:
    rows = []
    for frame in range(n_frames):
        for identity_index, identity in enumerate(("H1", "H2", "H3")):
            x = 50.0 * identity_index + 0.4 * frame
            y = 3.0 * identity_index + 0.05 * np.sin(frame / 4.0)
            rows.append(
                {
                    "frame": frame,
                    "human_identity": identity,
                    "X": x,
                    "Y": y,
                    "Z": 0.1 * np.cos(frame / 5.0),
                    "mean_reprojection_error_px": 2.0 + 0.01 * frame,
                    "max_reprojection_error_px": 3.0 + 0.02 * frame,
                    "top_mouth_x": 100.0 + x,
                    "top_mouth_y": 200.0 + y,
                    "side1_mouth_x": 300.0 + x,
                    "side1_mouth_y": 400.0 + y,
                    "side2_mouth_x": 500.0 + x,
                    "side2_mouth_y": 600.0 + y,
                    "top_cam_match_px": 4.0 + 0.01 * frame,
                    "side1_cam_match_px": 5.0 + 0.01 * frame,
                    "side2_cam_match_px": 6.0 + 0.01 * frame,
                    "all_views_agree": True,
                }
            )
    return pd.DataFrame(rows)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    track_path = tmp_path / "clean_mouth_3d.csv"
    episodes_path = tmp_path / "heldout_episodes.csv"
    _synthetic_clean_track().to_csv(track_path, index=False)
    pd.DataFrame(
        [
            {
                "episode_id": "episode_001",
                "start_frame": 25,
                "end_frame": 30,
                "guard_start_frame": 23,
                "guard_end_frame": 32,
            }
        ]
    ).to_csv(episodes_path, index=False)
    return track_path, episodes_path


def test_pair_features_and_fitted_cost_have_expected_distance_sign(tmp_path: Path) -> None:
    track_path, episodes_path = _write_inputs(tmp_path)
    clean = pd.read_csv(track_path)
    pairs, summary = fitter.build_association_pairs(clean, max_frame_gap=10)

    same = pairs.query(
        "frame == 10 and previous_identity == 'H1' and candidate_identity == 'H1'"
    ).iloc[0]
    different = pairs.query(
        "frame == 10 and previous_identity == 'H1' and candidate_identity == 'H2'"
    ).iloc[0]
    assert same.distance_mm < different.distance_mm
    assert same.prediction_error_mm < different.prediction_error_mm
    assert same.label_same_identity == 1
    assert different.label_same_identity == 0
    assert summary["positive_pairs"] * 2 == summary["negative_pairs"]

    report, _ = fitter.run(
        track_path,
        episodes_path,
        max_frame_gap=10,
        context_margin_frames=3,
    )
    model = report["model"]
    assert model["standardized_logit"]["coefficients"]["distance_mm"] < 0
    assert model["raw_association_cost"]["weights"]["distance_mm"] > 0
    assert model["coefficient_sign_audit"]["distance_mm"]["matches_expectation"] is True
    assert model["metrics"]["heldout"]["top1_candidate_accuracy"] == 1.0
    assert model["metrics"]["heldout"]["roc_auc"] == 1.0
    assert model["unsupported_or_constant_features"]["all_views_agree"] == (
        "constant in training data"
    )
    assert "any_view_missing" in model["unsupported_or_constant_features"]


def test_episode_context_and_motion_history_are_excluded_from_training(tmp_path: Path) -> None:
    track_path, episodes_path = _write_inputs(tmp_path)
    report, pairs = fitter.run(
        track_path,
        episodes_path,
        max_frame_gap=10,
        context_margin_frames=3,
    )

    # Supplied guard is 23..32; the additional context margin expands this to 20..35.
    assert report["split_audit"]["training_exclusion_intervals"] == [[20, 35]]
    train = pairs[pairs["split"] == "train"]
    heldout = pairs[pairs["split"] == "heldout"]
    assert heldout["frame"].between(25, 30).all()
    assert not train["frame"].between(20, 35).any()
    for column in ("history_frame_1", "history_frame_2", "history_frame_3"):
        finite = train[column].dropna()
        assert not finite.between(20, 35).any()
    assert report["split_audit"]["heldout_training_current_frame_intersection"] == 0
    assert report["split_audit"]["training_pairs_with_excluded_history"] == 0


def test_run_is_deterministic_and_reports_missing_sources(tmp_path: Path) -> None:
    track_path, episodes_path = _write_inputs(tmp_path)
    clean = pd.read_csv(track_path)
    clean.drop(columns=["side2_cam_match_px"], inplace=True)
    clean.to_csv(track_path, index=False)

    first, first_pairs = fitter.run(
        track_path,
        episodes_path,
        max_frame_gap=10,
        context_margin_frames=3,
    )
    second, second_pairs = fitter.run(
        track_path,
        episodes_path,
        max_frame_gap=10,
        context_margin_frames=3,
    )

    first_json = json.dumps(fitter._jsonable(first), indent=2, sort_keys=True)
    second_json = json.dumps(fitter._jsonable(second), indent=2, sort_keys=True)
    assert first_json == second_json
    pd.testing.assert_frame_equal(first_pairs, second_pairs)
    support = first["pair_summary"]["source_feature_support"]
    assert support["side2_cam_match_px"]["missing_columns"] == ["side2_cam_match_px"]
    assert support["side2_cam_missing"]["missing_columns"] == ["side2_cam_match_px"]
    assert first["model"]["unsupported_or_constant_features"]["side2_cam_match_px"] == (
        "no finite training values"
    )
