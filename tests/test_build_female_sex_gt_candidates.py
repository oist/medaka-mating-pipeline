from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.analysis.build_female_sex_gt_candidates as candidate_builder
from scripts.analysis.build_female_sex_gt_candidates import (
    OUTPUT_COLUMNS, REFRACTIVE_SOURCE_MODE, VIEWS, CandidateConfig,
    VideoMetadata, build_refractive_geometry_tracklets, generate_candidates,
    validate_clean_tracklets, validate_refractive_reconstruction_provenance)
from scripts.analysis.female_sex_gt import normalize_candidate_table
from scripts.analysis.geometry_swap_review import VIDEO_NAMES

EXPERIMENT = "20251225_K_1_2_FCMC"
WORLD = {
    "G1": np.array([60.0, 60.0, 60.0]),
    "G2": np.array([140.0, 120.0, 130.0]),
    "G3": np.array([230.0, 190.0, 200.0]),
}


class FakeCalibration:
    backend_name = "synthetic"
    score_unit = "synthetic"
    provenance = {"model_files": {"synthetic": "fixture"}}

    def project_world(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=float)
        return np.stack(
            (
                xyz[:, [0, 1]],
                xyz[:, [0, 2]],
                xyz[:, [1, 2]],
            ),
            axis=1,
        )


class FakeRefractiveCalibration(FakeCalibration):
    backend_name = "refractive"
    score_unit = "mm^2"
    provenance = {
        "model_files": {
            "cam0": "synthetic/cam0_bundle_refractive.json",
            "cam1": "synthetic/cam1_bundle_refractive.json",
            "cam2": "synthetic/cam2_bundle_refractive.json",
        }
    }


def _refractive_provenance() -> dict:
    model_files = FakeRefractiveCalibration.provenance["model_files"]
    return {
        "schema_version": 1,
        "camera_order": ["cam0", "cam1", "cam2"],
        "triangulation": {
            "backend": "refractive",
            "score_unit": "mm^2",
            "provenance": {"model_files": model_files},
        },
        "projection": {
            "backend": "refractive",
            "provenance": {"model_files": model_files},
        },
    }


class FakeSlp:
    def __init__(self, frames: dict[int, np.ndarray | None], extent_end: int = 319):
        self.frames = frames
        self._frame_indices = np.arange(extent_end + 1, dtype=np.int64)

    def get(self, frame: int, default=None):
        return self.frames.get(int(frame), default)

    def close(self) -> None:
        return None


def _projected(view: str) -> np.ndarray:
    xyz = np.stack([WORLD[key] for key in sorted(WORLD)])
    return FakeCalibration().project_world(xyz)[:, VIEWS.index(view), :]


def _poses(view: str) -> np.ndarray:
    centers = _projected(view)
    offset = np.array([4.0, 0.0])
    return np.stack((centers - offset, centers + offset), axis=1)


def _tables(frames: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_rows: list[dict] = []
    geometry_rows: list[dict] = []
    for frame in frames:
        for index, geometry_id in enumerate(sorted(WORLD), start=1):
            xyz = WORLD[geometry_id]
            clean_rows.append(
                {
                    "frame": frame,
                    "clean_tracklet_id": f"T{index:06d}",
                    "geometry_source_id": geometry_id,
                    "geometry_slot": geometry_id,
                    "human_identity": f"H{index}",
                    "X": xyz[0],
                    "Y": xyz[1],
                    "Z": xyz[2],
                    "label_status": "clean",
                    "mapping_resolved": True,
                    "is_interpolated": False,
                }
            )
            geometry_rows.append(
                {
                    "frame": frame,
                    "fish_id": geometry_id,
                    "X": xyz[0],
                    "Y": xyz[1],
                    "Z": xyz[2],
                }
            )
    return pd.DataFrame(clean_rows), pd.DataFrame(geometry_rows)


def _inputs(frames: list[int]):
    clean, geometry = _tables(frames)
    detections = {
        view: FakeSlp({frame: _poses(view) for frame in frames}) for view in VIEWS
    }
    videos = {
        view: VideoMetadata(
            path=Path(f"synthetic/{view}.mp4"),
            frame_count=320,
            width=320,
            height=240,
            reported_fps=166.0,
        )
        for view in VIEWS
    }
    return clean, geometry, detections, videos


def _generate(
    frames: list[int],
    *,
    config: CandidateConfig | None = None,
    mutate=None,
):
    clean, geometry, detections, videos = _inputs(frames)
    if mutate is not None:
        mutate(clean, geometry, detections, videos)
    return generate_candidates(
        experiment=EXPERIMENT,
        clean_tracklets=clean,
        geometry=geometry,
        calibration=FakeCalibration(),
        slp_predictions=detections,
        videos=videos,
        config=config
        or CandidateConfig(target_count=300, min_frame_spacing=1, seed=0),
    )


def test_only_frames_with_three_clean_simultaneous_tracklets_are_eligible():
    def mutate(clean, _geometry, _detections, _videos):
        clean.drop(
            clean[(clean["frame"] == 11) & (clean["geometry_source_id"] == "G3")].index,
            inplace=True,
        )

    result = _generate([10, 11], mutate=mutate)
    assert result.candidates["frame"].tolist() == [10]
    assert result.diagnostics["rejection_counts"][
        "not_three_simultaneous_clean_tracklets"
    ] == 1


def test_interpolated_ambiguous_unresolved_and_false_positive_rows_are_rejected():
    def mutate(clean, _geometry, _detections, _videos):
        clean.loc[
            (clean["frame"] == 10) & (clean["geometry_source_id"] == "G1"),
            ["label_status", "is_interpolated"],
        ] = ["interpolated", True]
        clean.loc[
            (clean["frame"] == 11) & (clean["geometry_source_id"] == "G1"),
            "label_status",
        ] = "ambiguous"
        clean.loc[
            (clean["frame"] == 12) & (clean["geometry_source_id"] == "G1"),
            ["label_status", "mapping_resolved"],
        ] = ["unresolved", False]
        clean.loc[
            (clean["frame"] == 13) & (clean["geometry_source_id"] == "G1"),
            "label_status",
        ] = "false_positive"

    result = _generate([10, 11, 12, 13], mutate=mutate)
    assert result.candidates.empty
    assert result.diagnostics["rejection_counts"][
        "interpolated_ambiguous_unresolved_or_false_positive"
    ] == 4


def test_missing_or_unknown_clean_status_fields_fail_loudly():
    clean, _geometry = _tables([10])
    with pytest.raises(ValueError, match="required status/provenance"):
        validate_clean_tracklets(clean.drop(columns="mapping_resolved"))
    clean.loc[0, "label_status"] = "legacy_unknown"
    with pytest.raises(ValueError, match="unknown meanings"):
        validate_clean_tracklets(clean)


def test_direct_refractive_geometry_builds_segments_and_breaks_unclean_rows():
    _clean, geometry = _tables([0, 1, 2, 3])
    geometry["is_interpolated"] = False
    geometry.loc[
        (geometry["frame"] == 1) & (geometry["fish_id"] == "G1"),
        "is_interpolated",
    ] = True
    geometry.loc[
        (geometry["frame"] == 1) & (geometry["fish_id"] == "G3"),
        ["X", "Y", "Z"],
    ] = np.nan
    geometry.loc[
        (geometry["frame"] == 2) & (geometry["fish_id"] == "G2"), "X"
    ] += 20.0

    tracklets, diagnostics = build_refractive_geometry_tracklets(
        geometry, max_local_jump_mm=8.0
    )
    by_key = tracklets.set_index(["fish_id", "frame"])
    assert by_key.loc[("G1", 1), "label_status"] == "interpolated"
    assert by_key.loc[("G3", 1), "label_status"] == "missing"
    assert by_key.loc[("G2", 1), "label_status"] == "unresolved"
    assert by_key.loc[("G2", 2), "label_status"] == "unresolved"
    assert by_key.loc[("G2", 3), "label_status"] == "unresolved"
    assert (
        by_key.loc[("G1", 0), "clean_tracklet_id"]
        != by_key.loc[("G1", 2), "clean_tracklet_id"]
    )
    assert diagnostics["over_threshold_jump_count"] == 2


def test_direct_mode_treats_present_all_nan_geometry_as_missing_not_absent():
    _reviewed, geometry, detections, videos = _inputs([0, 1])
    geometry["is_interpolated"] = False
    geometry.loc[
        (geometry["frame"] == 0) & (geometry["fish_id"] == "G2"),
        ["X", "Y", "Z"],
    ] = np.nan
    direct, source_diagnostics = build_refractive_geometry_tracklets(
        geometry, max_local_jump_mm=8.0
    )
    result = generate_candidates(
        experiment=EXPERIMENT,
        clean_tracklets=direct,
        geometry=geometry,
        calibration=FakeRefractiveCalibration(),
        slp_predictions=detections,
        videos=videos,
        config=CandidateConfig(target_count=10, min_frame_spacing=1),
        source_mode=REFRACTIVE_SOURCE_MODE,
        source_diagnostics=source_diagnostics,
    )
    assert result.candidates["frame"].tolist() == [1]
    assert result.diagnostics["rejection_counts"][
        "interpolated_ambiguous_unresolved_or_false_positive"
    ] == 1


def test_direct_mode_requires_matching_refractive_provenance():
    validated = validate_refractive_reconstruction_provenance(
        _refractive_provenance(), FakeRefractiveCalibration()
    )
    assert validated["triangulation_backend"] == "refractive"
    polynomial = _refractive_provenance()
    polynomial["triangulation"]["backend"] = "polynomial"
    with pytest.raises(ValueError, match="requires refractive"):
        validate_refractive_reconstruction_provenance(
            polynomial, FakeRefractiveCalibration()
        )
    with pytest.raises(ValueError, match="requires a refractive calibration"):
        validate_refractive_reconstruction_provenance(
            _refractive_provenance(), FakeCalibration()
        )


def test_missing_mouth_or_tail_in_any_view_is_rejected():
    def mutate(_clean, _geometry, detections, _videos):
        broken = detections["side1_cam"].frames[10].copy()
        broken[0, 0] = np.nan
        detections["side1_cam"].frames[10] = broken

    result = _generate([10], mutate=mutate)
    assert result.candidates.empty
    assert result.diagnostics["rejection_counts"][
        "missing_mouth_or_tail_in_a_view"
    ] == 1


def test_extra_reflection_detections_do_not_invalidate_correct_assignment():
    def mutate(_clean, _geometry, detections, _videos):
        for view in VIEWS:
            reflection = np.array([[[276.0, 30.0], [284.0, 30.0]]])
            detections[view].frames[10] = np.concatenate(
                (detections[view].frames[10], reflection), axis=0
            )

    result = _generate([10], mutate=mutate)
    assert result.candidates["frame"].tolist() == [10]
    assert result.candidates.loc[0, "top_visible_pose_count"] == 4


def test_failed_or_over_gate_geometry_association_is_rejected():
    def mutate(_clean, _geometry, detections, _videos):
        detections["top_cam"].frames[10] = detections["top_cam"].frames[10] + 500.0

    result = _generate([10], mutate=mutate)
    assert result.candidates.empty
    assert result.diagnostics["rejection_counts"][
        "failed_or_over_gate_geometry_association"
    ] == 1


def test_insufficient_separation_in_any_view_is_rejected():
    def mutate(_clean, _geometry, detections, _videos):
        poses = detections["side2_cam"].frames[10].copy()
        first_center = poses[0].mean(axis=0)
        poses[1] = np.stack(
            (first_center + np.array([16.0, 0.0]), first_center + np.array([24.0, 0.0]))
        )
        detections["side2_cam"].frames[10] = poses

    result = _generate(
        [10],
        config=CandidateConfig(
            target_count=10,
            min_frame_spacing=1,
            match_gate_px=120,
            min_view_separation_px=40,
        ),
        mutate=mutate,
    )
    assert result.candidates.empty
    assert result.diagnostics["rejection_counts"][
        "insufficient_fish_separation_in_a_view"
    ] == 1


def test_border_adjacent_pose_is_rejected():
    def mutate(_clean, _geometry, detections, _videos):
        poses = detections["top_cam"].frames[10].copy()
        poses[0] = np.array([[6.0, 60.0], [14.0, 60.0]])
        detections["top_cam"].frames[10] = poses

    result = _generate([10], mutate=mutate)
    assert result.candidates.empty
    assert result.diagnostics["rejection_counts"][
        "border_adjacent_pose_in_a_view"
    ] == 1


def test_quality_scores_and_ordering_are_deterministic():
    config = CandidateConfig(target_count=6, min_frame_spacing=2, seed=7)
    first = _generate(list(range(10, 31)), config=config).candidates
    second = _generate(list(range(10, 31)), config=config).candidates
    pd.testing.assert_frame_equal(first, second)
    expected = first.sort_values(
        ["priority", "score", "frame"],
        ascending=[False, False, True],
        kind="stable",
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(first, expected)


def test_temporal_spacing_is_enforced():
    selected = _generate(
        list(range(0, 31)),
        config=CandidateConfig(target_count=30, min_frame_spacing=5, seed=0),
    ).candidates
    frames = np.sort(selected["frame"].to_numpy(dtype=int))
    assert len(frames) > 1
    assert (np.diff(frames) >= 5).all()


def test_candidates_cover_multiple_clean_intervals_and_recording_sections():
    frames = [*range(10, 31), *range(150, 171), *range(290, 311)]
    selected = _generate(
        frames,
        config=CandidateConfig(target_count=9, min_frame_spacing=5, seed=3),
    ).candidates
    assert selected["clean_interval_id"].nunique() == 3
    assert set(selected["temporal_bin"]) == {"early", "middle", "late"}
    coverage = selected.groupby("clean_interval_id")["interval_section"].agg(set)
    assert all({"early", "middle", "late"} <= sections for sections in coverage)


def test_output_follows_annotator_contract_and_contains_no_identity_inference():
    candidates = _generate([10, 11, 12]).candidates
    assert list(candidates.columns) == OUTPUT_COLUMNS
    normalized = normalize_candidate_table(
        candidates, experiment=EXPERIMENT, max_frame=319
    )
    assert normalized["frame"].tolist() == candidates.sort_values("frame")[
        "frame"
    ].tolist()
    assert candidates["source"].eq("clean_tracklet_three_view").all()
    forbidden = ("sex", "female", "instance_id", "track_id")
    assert not any(token in column.lower() for column in candidates for token in forbidden)
    assert candidates["view_hint"].isin(
        ["both_sides", "side1_cam", "side2_cam"]
    ).all()
    mapping = json.loads(candidates.loc[0, "geometry_tracklet_map_json"])
    assert mapping == {
        "G1": "T000001",
        "G2": "T000002",
        "G3": "T000003",
    }


def test_synthetic_cli_smoke_writes_candidate_csv_and_summary_json(
    tmp_path, monkeypatch
):
    frames = [10, 11, 12]
    clean, geometry = _tables(frames)
    clean_path = tmp_path / "reviewed_clean_3d_tracklets.csv"
    geometry_path = tmp_path / "geometry.csv"
    clean.to_csv(clean_path, index=False)
    geometry.to_csv(geometry_path, index=False)
    slp_dir = tmp_path / "slp"
    video_dir = tmp_path / "videos"
    calibration_dir = tmp_path / "calibration"
    slp_dir.mkdir()
    calibration_dir.mkdir()
    for view in VIEWS:
        (slp_dir / f"{view}.slp").write_bytes(b"synthetic-slp")
        video_path = video_dir / view / VIDEO_NAMES[view]
        video_path.parent.mkdir(parents=True)
        video_path.write_bytes(b"synthetic-video")

    class CliSlp(FakeSlp):
        def __init__(self, path, *, node_names):
            view = Path(path).stem
            super().__init__({frame: _poses(view) for frame in frames})
            self.path = Path(path)
            self.node_names = node_names

    def video_metadata(path):
        return VideoMetadata(Path(path), 320, 320, 240, 166.0)

    monkeypatch.setattr(candidate_builder, "IndexedSlpPredictions", CliSlp)
    monkeypatch.setattr(candidate_builder, "_open_video_metadata", video_metadata)
    monkeypatch.setattr(
        candidate_builder,
        "Calibration",
        lambda _path, backend: FakeCalibration(),
    )
    monkeypatch.setattr(candidate_builder, "_git_commit", lambda: "synthetic")

    output_csv = tmp_path / "candidates.csv"
    summary_json = tmp_path / "candidates.summary.json"
    result = candidate_builder.main(
        [
            EXPERIMENT,
            "--clean-tracklets-csv",
            str(clean_path),
            "--geometry-csv",
            str(geometry_path),
            "--slp-dir",
            str(slp_dir),
            "--video-dir",
            str(video_dir),
            "--calibration-model-dir",
            str(calibration_dir),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--min-frame-spacing",
            "1",
        ]
    )
    assert result == 0
    saved = pd.read_csv(output_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert saved["frame"].tolist() == [11, 10, 12]
    assert summary["selected_candidate_count"] == 3
    assert summary["calibration"]["resolved_backend"] == "synthetic"
    assert summary["inputs"]["clean_tracklets_csv"]["sha256"]


def test_direct_refractive_cli_smoke_writes_segment_mappings(tmp_path, monkeypatch):
    frames = [10, 11, 12]
    _clean, geometry = _tables(frames)
    geometry["is_interpolated"] = False
    geometry_path = tmp_path / "round7_refractive_3d.csv"
    geometry.to_csv(geometry_path, index=False)
    provenance_path = tmp_path / "round7_refractive_reconstruction_provenance.json"
    provenance_path.write_text(
        json.dumps(_refractive_provenance()), encoding="utf-8"
    )
    slp_dir = tmp_path / "slp"
    video_dir = tmp_path / "videos"
    calibration_dir = tmp_path / "calibration"
    slp_dir.mkdir()
    calibration_dir.mkdir()
    for view in VIEWS:
        (slp_dir / f"{view}.slp").write_bytes(b"synthetic-slp")
        video_path = video_dir / view / VIDEO_NAMES[view]
        video_path.parent.mkdir(parents=True)
        video_path.write_bytes(b"synthetic-video")

    class CliSlp(FakeSlp):
        def __init__(self, path, *, node_names):
            view = Path(path).stem
            super().__init__({frame: _poses(view) for frame in frames})
            self.path = Path(path)
            self.node_names = node_names

    monkeypatch.setattr(candidate_builder, "IndexedSlpPredictions", CliSlp)
    monkeypatch.setattr(
        candidate_builder,
        "_open_video_metadata",
        lambda path: VideoMetadata(Path(path), 320, 320, 240, 166.0),
    )
    monkeypatch.setattr(
        candidate_builder,
        "Calibration",
        lambda _path, backend: FakeRefractiveCalibration(),
    )
    monkeypatch.setattr(candidate_builder, "_git_commit", lambda: "synthetic")

    output_csv = tmp_path / "direct_candidates.csv"
    summary_json = tmp_path / "direct_candidates.summary.json"
    result = candidate_builder.main(
        [
            EXPERIMENT,
            "--geometry-csv",
            str(geometry_path),
            "--slp-dir",
            str(slp_dir),
            "--video-dir",
            str(video_dir),
            "--calibration-model-dir",
            str(calibration_dir),
            "--calibration-backend",
            "refractive",
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--min-frame-spacing",
            "1",
        ]
    )
    assert result == 0
    saved = pd.read_csv(output_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert saved["geometry_source_mode"].eq(REFRACTIVE_SOURCE_MODE).all()
    mapping = json.loads(saved.loc[0, "geometry_tracklet_map_json"])
    assert mapping["G1"].startswith("R3D:G1:S")
    assert summary["geometry_source_mode"] == REFRACTIVE_SOURCE_MODE
    assert summary["reconstruction_provenance_validation"][
        "triangulation_backend"
    ] == "refractive"
    assert summary["inputs"]["clean_tracklets_csv"] is None
