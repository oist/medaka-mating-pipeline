from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.benchmark_reviewed_calibration_backends import (
    DLTBackend,
    build_episode_split,
    full_triplet_assignments,
    natural_transition_metrics,
    pair_association_metrics,
    sha256_file,
    validate_reviewed_data,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class _IdentityScoreBackend:
    backend_name = "test_backend"
    score_unit = "mm^2"

    def triangulate(self, observations: np.ndarray) -> np.ndarray:
        values = np.asarray(observations, dtype=float)
        x = np.nanmean(values[..., 0], axis=-1)
        return np.stack((x, np.full_like(x, 50.0), np.full_like(x, 50.0)), axis=-1)

    def score_hypotheses(
        self, observations: np.ndarray, xyz: np.ndarray | None = None
    ) -> np.ndarray:
        values = np.asarray(observations, dtype=float)
        return np.nanvar(values[..., 0], axis=-1)


def _review_rows(frames=(0, 1)) -> pd.DataFrame:
    rows = []
    for frame in frames:
        for view in ("top_cam", "side1_cam", "side2_cam"):
            for index, identity in enumerate(("H1", "H2", "H3"), start=1):
                rows.append({
                    "frame": frame, "view": view, "geometry_slot": f"G{index}",
                    "human_identity": identity, "mouth_x": 10.0 * index,
                    "mouth_y": 100.0 + index, "tail_x": 0.0, "tail_y": 0.0,
                    "centroid_x": 5.0, "centroid_y": 5.0,
                    "geometry_match_distance_px": 1.0,
                })
    return pd.DataFrame(rows)


def test_fixed_refractive_manifest_hashes_match_promoted_assets() -> None:
    directory = REPO_ROOT / "calibration_results" / "fixed_refractive_192x192x186"
    manifest = json.loads((directory / "calibration_manifest.json").read_text())
    for model in manifest["model_files"]:
        assert sha256_file(directory / model["path"]) == model["sha256"]
    assert manifest["view_order"] == ["top_cam", "side1_cam", "side2_cam"]
    assert manifest["hypothesis_score_unit"] == "mm^2"


def test_dlt_adapter_accepts_batched_two_view_rows() -> None:
    model_dir = (
        REPO_ROOT / "calibration_results" / "cali_20251217_manual"
        / "fullset_pinhole_dlt" / "models"
    )
    backend = DLTBackend(model_dir)
    xyz = np.array([[20.0, 30.0, 40.0], [80.0, 90.0, 100.0]])
    pixels = backend.project_world(xyz)
    pixels[:, 1] = np.nan
    reconstructed = backend.triangulate(pixels)
    scores = backend.score_hypotheses(pixels, reconstructed)
    assert reconstructed.shape == (2, 3)
    assert np.all(np.isfinite(reconstructed))
    assert np.all(np.isfinite(scores))


def test_episode_split_is_contiguous_and_heldout_by_block() -> None:
    episodes = build_episode_split(0, 10, episode_frames=3, cycle=3, holdout_index=2)
    assert episodes[["start_frame", "end_frame"]].values.tolist() == [
        [0, 2], [3, 5], [6, 8], [9, 10]
    ]
    assert episodes["split"].tolist() == ["tuning", "tuning", "heldout", "tuning"]


def test_pair_and_triplet_matching_use_backend_scores() -> None:
    backend = _IdentityScoreBackend()
    reviewed = _review_rows()
    # Deliberately scramble one camera's row order without changing provenance.
    side1 = reviewed["view"].eq("side1_cam")
    reviewed.loc[side1, "geometry_slot"] = reviewed.loc[side1, "geometry_slot"].map(
        {"G1": "G3", "G2": "G1", "G3": "G2"}
    )
    episodes = build_episode_split(0, 1, episode_frames=1, cycle=2, holdout_index=1)
    pairs = pair_association_metrics(backend, reviewed, episodes)
    assert (pairs.loc[pairs["split"].eq("all"), "pair_association_accuracy"] == 1.0).all()

    detail, selected, _ = full_triplet_assignments(
        backend, reviewed, episodes, xyz_bounds=(-20.0, 220.0)
    )
    assert detail["correct_groups"].tolist() == [3, 3]
    assert all(all(info["correct"]) for info in selected.values())


def test_natural_transition_empty_cohort_has_stable_schema() -> None:
    reviewed = _review_rows(frames=(0, 1, 2, 3, 4))
    clean = pd.DataFrame(columns=["frame", "human_identity", "X", "Y", "Z"])
    episodes = build_episode_split(0, 4)
    result = natural_transition_metrics(
        _IdentityScoreBackend(), reviewed, clean, episodes, set()
    )
    assert result.empty
    assert "trajectory_error_mm_p95" in result.columns


def test_reviewed_manifest_verification_and_mouth_provenance(tmp_path: Path) -> None:
    per_view = _review_rows(frames=(0,)).query("human_identity == 'H1'").copy()
    clean = pd.DataFrame([{
        "frame": 0, "human_identity": "H1", "X": 1.0, "Y": 2.0, "Z": 3.0,
        "top_mouth_x": 10.0, "top_mouth_y": 101.0,
        "side1_mouth_x": 10.0, "side1_mouth_y": 101.0,
        "side2_mouth_x": 10.0, "side2_mouth_y": 101.0,
        "mean_reprojection_error_px": 1.0, "max_reprojection_error_px": 1.0,
        "top_cam_match_px": 1.0, "side1_cam_match_px": 1.0,
        "side2_cam_match_px": 1.0,
        "coordinate_source": "three_view_reviewed_sleap_mouth",
        "all_views_agree": True, "reviewed_tracklet_id": "RT1",
        "label_status": "clean_reviewed_three_view",
        "filter_all_view_mapping_agreement": True,
        "filter_max_geometry_match_px": 30.0, "filter_max_reprojection_px": 15.0,
    }])
    files = {
        "corrected_per_view_sleap_mouth_tracks_f0_15000.csv": per_view,
        "clean_reviewed_multiview_mouth_3d_f0_15000.csv": clean,
    }
    artifacts = []
    for name, table in files.items():
        path = tmp_path / name
        table.to_csv(path, index=False)
        artifacts.append({
            "path": name, "byte_size": path.stat().st_size, "row_count": len(table),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    manifest = {
        "schema_version": "reviewed_consensus_dataset_manifest_v1",
        "coordinate_definition": {"point": "mouth"}, "artifacts": artifacts,
    }
    (tmp_path / "reviewed_consensus_manifest_v1.json").write_text(json.dumps(manifest))
    _, _, fingerprints = validate_reviewed_data(tmp_path)
    assert fingerprints["coordinate_provenance"]["maximum_pixel_difference"] == 0.0

    clean_gate_failure = clean.copy()
    clean_gate_failure.loc[0, "max_reprojection_error_px"] = 16.0
    path = tmp_path / "clean_reviewed_multiview_mouth_3d_f0_15000.csv"
    clean_gate_failure.to_csv(path, index=False)
    artifact = next(item for item in artifacts if item["path"] == path.name)
    artifact.update({
        "byte_size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
    (tmp_path / "reviewed_consensus_manifest_v1.json").write_text(json.dumps(manifest))
    try:
        validate_reviewed_data(tmp_path)
    except ValueError as exc:
        assert "reprojection gate" in str(exc)
    else:
        raise AssertionError("clean row above its QC gate was accepted")

    clean.loc[0, "top_mouth_x"] = 11.0
    clean.to_csv(path, index=False)
    artifact.update({
        "byte_size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
    (tmp_path / "reviewed_consensus_manifest_v1.json").write_text(json.dumps(manifest))
    try:
        validate_reviewed_data(tmp_path)
    except ValueError as exc:
        assert "provenance mismatch" in str(exc)
    else:
        raise AssertionError("mouth-coordinate mismatch was accepted")
