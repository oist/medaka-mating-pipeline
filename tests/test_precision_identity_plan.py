from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from sam2_id_experiment.src.sam2_adapter.prompt_seed_from_3d import mine_3d_seed_prompts
from sam2_id_experiment.src.sleap_adapter.top_track_export import export_segmentation_top_tracks
from scripts.calibration.refine_calibration_rigid_object import run_rigid_object_refinement
from scripts.reconstruction_3d.bind_topcam_id import bind


class _IdentityProjectionCalibration:
    def project_world_to_view(self, xyz: np.ndarray, view: str) -> np.ndarray:
        assert view == "top_cam"
        return np.asarray(xyz, dtype=float)[:, :2]


class _OrthographicIm2XYZModel:
    def predict(self, arr):
        arr = np.asarray(arr, dtype=float)
        return np.column_stack([arr[:, 0], arr[:, 1], 0.5 * (arr[:, 3] + arr[:, 5])])


def _identity_tracklet_row(*, tracklet_id: int, frame: int, x: float, y: float, identity: str, seg: int) -> dict:
    return {
        "tracklet_id": tracklet_id,
        "sam2_track_id": tracklet_id,
        "instance_id": 0,
        "frame": frame,
        "centroid_x": x,
        "centroid_y": y,
        "bbox_x0": int(x),
        "bbox_y0": int(y),
        "bbox_x1": int(x) + 5,
        "bbox_y1": int(y) + 5,
        "area": 25,
        "confidence": 0.99,
        "qc_status": "clean_train",
        "qc_reason": "passed_clean_frame_rules",
        "identity_trajectory_id": f"{identity}_seg{seg}",
        "assigned_identity": identity,
    }


def test_mine_3d_seed_prompts_selects_clean_windows_and_negatives():
    rows = []
    for frame in range(5):
        rows.extend([
            {
                "frame": frame,
                "fish_id": "g0",
                "final_fish_id": "pseudo_ID_0",
                "X": frame,
                "Y": 0.0,
                "Z": 10.0,
                "bind_source": "top_cam_2d",
                "bind_score": 2.0,
            },
            {
                "frame": frame,
                "fish_id": "g1",
                "final_fish_id": "pseudo_ID_1",
                "X": 50.0 + frame,
                "Y": 0.0,
                "Z": 10.0,
                "bind_source": "top_cam_2d",
                "bind_score": 2.0,
            },
            {
                "frame": frame,
                "fish_id": "g2",
                "final_fish_id": "pseudo_ID_2",
                "X": 0.0,
                "Y": 50.0 + frame,
                "Z": 10.0,
                "bind_source": "top_cam_2d",
                "bind_score": 2.0,
            },
        ])
    pre_link = {
        "frames": {
            str(frame): {
                "errors": {"a": 100.0 if frame != 2 else 9999.0, "b": 100.0, "c": 100.0},
                "dropped_high_residual": [],
            }
            for frame in range(5)
        }
    }

    result = mine_3d_seed_prompts(
        identified_3d=pd.DataFrame(rows),
        best_pre_link=pre_link,
        calib=_IdentityProjectionCalibration(),
        config={
            "num_individuals": 3,
            "min_window_frames": 2,
            "max_residual_px2": 5000.0,
            "min_separation_mm": 20.0,
            "max_speed_mm_per_frame": 8.0,
            "max_bind_score_px": 10.0,
        },
        start_frame=0,
    )

    assert result.summary["n_clean_frames"] == 4
    assert result.windows["frame_start"].tolist() == [0, 3]
    assert set(result.seed_points["final_fish_id"]) == {"pseudo_ID_0", "pseudo_ID_1", "pseudo_ID_2"}
    prompts = result.prompts_payload["sam2"]["prompts"]
    assert len(prompts) == 6
    assert all(prompt["labels"].count(1) == 1 for prompt in prompts)
    assert all(prompt["labels"].count(0) == 1 for prompt in prompts)
    assert prompts[0]["absolute_frame"] in {1, 4}


def test_segmentation_top_export_pools_fragments_by_assigned_identity(tmp_path):
    tracks = pd.DataFrame([
        _identity_tracklet_row(tracklet_id=10, frame=0, x=10, y=20, identity="pseudo_ID_0", seg=0),
        _identity_tracklet_row(tracklet_id=11, frame=100, x=30, y=40, identity="pseudo_ID_0", seg=1),
    ])
    output_csv = tmp_path / "top_cam.seg_identity_tracks.csv"

    summary = export_segmentation_top_tracks(identity_tracks=tracks, output_csv=output_csv)
    out = pd.read_csv(output_csv)

    assert summary["n_tracks"] == 1
    assert out["track_id"].nunique() == 1
    assert set(out["identity_trajectory_id"]) == {"pseudo_ID_0_seg0", "pseudo_ID_0_seg1"}
    assert set(out["identity"]) == {"pseudo_ID_0"}


def test_bind_topcam_id_pools_short_segmentation_fragments_by_identity(tmp_path):
    geom_path = tmp_path / "geom.csv"
    top_path = tmp_path / "top.csv"
    out_path = tmp_path / "identified.csv"
    pd.DataFrame([
        {"frame": 0, "fish_id": "ID_1", "X": 10.0, "Y": 20.0, "Z": 1.0},
        {"frame": 1, "fish_id": "ID_1", "X": 11.0, "Y": 21.0, "Z": 1.0},
    ]).to_csv(geom_path, index=False)
    pd.DataFrame([
        {
            "frame": 0,
            "track_id": 10,
            "x": 10.0,
            "y": 20.0,
            "identity": "pseudo_ID_0",
            "track_label": "pseudo_ID_0",
            "identity_trajectory_id": "pseudo_ID_0_seg0",
            "qc_status": "clean_train",
            "bind_weight": 1.0,
        },
        {
            "frame": 1,
            "track_id": 11,
            "x": 11.0,
            "y": 21.0,
            "identity": "pseudo_ID_0",
            "track_label": "pseudo_ID_0",
            "identity_trajectory_id": "pseudo_ID_0_seg1",
            "qc_status": "clean_train",
            "bind_weight": 1.0,
        },
    ]).to_csv(top_path, index=False)

    out = bind(
        geom_path,
        top_path,
        calib=_IdentityProjectionCalibration(),
        out_csv=out_path,
        bind_distance=5.0,
        require_min_frames=2,
    )

    assert set(out["source_bind_key"]) == {"pseudo_ID_0"}
    assert set(out["source_identity_trajectory_id"]) == {"pseudo_ID_0_seg0", "pseudo_ID_0_seg1"}
    assert set(out["final_fish_id"]) == {"pseudo_ID_0"}


def test_rigid_object_refinement_trains_models_and_report(tmp_path):
    base_model = tmp_path / "base_model"
    base_model.mkdir()
    dump(_OrthographicIm2XYZModel(), base_model / "imCoords_to_XYZ.joblib")

    geometry = pd.DataFrame([
        {"marker_id": "m0", "object_x": 0.0, "object_y": 0.0, "object_z": 0.0},
        {"marker_id": "m1", "object_x": 10.0, "object_y": 0.0, "object_z": 0.0},
        {"marker_id": "m2", "object_x": 0.0, "object_y": 10.0, "object_z": 0.0},
        {"marker_id": "m3", "object_x": 0.0, "object_y": 0.0, "object_z": 10.0},
    ])
    geometry_csv = tmp_path / "rigid_object_geometry.csv"
    geometry.to_csv(geometry_csv, index=False)

    rows = []
    translations = [
        (0.0, 0.0, 0.0),
        (20.0, 0.0, 5.0),
        (0.0, 30.0, 10.0),
        (15.0, 25.0, 20.0),
    ]
    for pose_idx, translation in enumerate(translations):
        t = np.asarray(translation)
        for marker in geometry.to_dict("records"):
            xyz = np.asarray([marker["object_x"], marker["object_y"], marker["object_z"]]) + t
            rows.append({
                "pose_id": f"pose{pose_idx}",
                "marker_id": marker["marker_id"],
                "cam0_x": xyz[0],
                "cam0_y": xyz[1],
                "cam1_x": xyz[0],
                "cam1_y": xyz[2],
                "cam2_x": xyz[1],
                "cam2_y": xyz[2],
            })
    image_csv = tmp_path / "rigid_object_image_coords.csv"
    pd.DataFrame(rows).to_csv(image_csv, index=False)

    report = run_rigid_object_refinement(
        rigid_object_geometry_csv=geometry_csv,
        rigid_object_image_coords_csv=image_csv,
        base_model_folder=base_model,
        output_folder=tmp_path / "refined_models",
        results_root=tmp_path / "refined_results",
        min_markers_per_pose=4,
        holdout_fraction=0.0,
        alpha_range=[1e-6],
        poly_deg_range=[1],
        train_fraction=0.75,
        num_permutations=2,
        model_types=["polynomial"],
    )

    out_dir = Path(report["output_folder"])
    assert (out_dir / "imCoords_to_XYZ.joblib").exists()
    assert (out_dir / "XYZ_to_imCoords.joblib").exists()
    assert (out_dir / "xy_yz_to_xz.joblib").exists()
    assert Path(report["report_path"]).exists()
    assert Path(report["mapping_csv"]).exists()
    assert report["n_poses"] == 4
    assert report["mean_pose_residual_mm"] < 1e-6
