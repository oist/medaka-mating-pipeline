from __future__ import annotations

import yaml
import pandas as pd

from sam2_id_experiment.src.config import load_config
from sam2_id_experiment.src.production.side_camera_pipeline import (
    SidePilotSpec,
    build_side_pilot_commands,
    build_side_pilot_config,
)
from sam2_id_experiment.src.review.napari_tracklet_review import (
    REVIEW_SHORTCUTS,
    build_review_template,
    read_review_table,
    save_review_table,
    set_review_labels,
    tracklet_ids_for_frame,
)


def test_side_pilot_config_resolves_side_video_paths(tmp_path):
    data_root = tmp_path / "data"
    video_path = data_root / "20251225_K_1_2_FCMC" / "side1_cam" / "cam1_side.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"")
    template = {
        "experiment_name": "old",
        "view": "top_cam",
        "paths_config": "configs/paths.local.toml",
        "output_dir": "outputs/{experiment_name}/{view}",
        "n_animals_expected": 3,
        "sam2": {"prompts": [{"obj_id": 0}], "frame_cache_dir": "old"},
        "sleap_gate": {"view": "top_cam", "anchor_node": "mouth"},
        "hybrid_tracking": {"view": "top_cam", "anchor_node": "mouth"},
    }
    cfg = build_side_pilot_config(
        template,
        view="side1_cam",
        spec=SidePilotSpec(paths_config=""),
    )
    cfg["medaka_data_dir"] = str(data_root)
    config_path = tmp_path / "side1.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    loaded = load_config(config_path)

    assert loaded.view == "side1_cam"
    assert loaded.video_path == video_path
    assert loaded.output_dir.as_posix().endswith("20251225_K_1_2_FCMC/side1_cam_f176300_181300")
    assert "anchor_node" not in cfg["sleap_gate"]
    assert "anchor_node" not in cfg["hybrid_tracking"]
    assert cfg["sam2"]["prompts"] == []


def test_side_pilot_command_manifest_uses_existing_cli_shapes(tmp_path):
    cfg_path = tmp_path / "side1.yaml"
    cfg_path.write_text(
        yaml.safe_dump({
            "experiment_name": "20251225_K_1_2_FCMC",
            "view": "side1_cam",
            "output_dir": "outputs/{experiment_name}/{view}_f176300_181300",
        }),
        encoding="utf-8",
    )

    commands = build_side_pilot_commands(
        config_paths={"side1_cam": str(cfg_path)},
        identified_3d_csv="identified.csv",
        best_pre_link_json="best.json",
    )

    joined = "\n".join(commands)
    assert "--view side1_cam" in joined
    assert "--train-json outputs/20251225_K_1_2_FCMC/side_coco_merged/annotations.json" in joined
    assert "--output outputs/20251225_K_1_2_FCMC/side_coco_merged" in joined
    assert "detectron2_tracklets_raw.parquet" in joined


def test_napari_review_template_carries_optional_identity_columns():
    tracklets = pd.DataFrame([
        {
            "frame": 10,
            "tracklet_id": 3,
            "qc_status": "manual_review",
            "qc_reason": "near_crossing",
            "centroid_x": 12.0,
            "centroid_y": 20.0,
        }
    ])
    reference = pd.DataFrame([
        {"frame": 10, "x": 13.0, "y": 21.0, "identity": "pseudo_ID_1"},
    ])

    template = build_review_template(tracklets, reference_points=reference)

    assert template.loc[0, "review_label"] == ""
    assert template.loc[0, "assigned_identity"] == ""
    assert template.loc[0, "identity_review_label"] == ""
    assert template.loc[0, "reference_identity"] == "pseudo_ID_1"


def test_review_label_helper_saves_csv_changes(tmp_path):
    review = pd.DataFrame([
        {"frame": 100, "tracklet_id": 0, "review_label": "", "qc_status": "manual_review"},
        {"frame": 100, "tracklet_id": 1, "review_label": "", "qc_status": "manual_review"},
        {"frame": 110, "tracklet_id": 0, "review_label": "", "qc_status": "manual_review"},
    ])

    n_changed = set_review_labels(
        review,
        frame=100,
        tracklet_ids=[1],
        review_label="merged",
    )
    out = tmp_path / "napari_tracklet_review.csv"
    save_review_table(review, out)
    loaded = read_review_table(out)

    assert n_changed == 1
    assert loaded.loc[loaded["tracklet_id"].eq(1), "review_label"].iloc[0] == "merged"
    assert loaded.loc[loaded["frame"].eq(110), "review_label"].iloc[0] == ""


def test_review_shortcuts_cover_common_labels():
    assert REVIEW_SHORTCUTS["G"] == "good"
    assert REVIEW_SHORTCUTS["M"] == "merged"
    assert REVIEW_SHORTCUTS["U"] == "uncertain"
    assert REVIEW_SHORTCUTS["C"] == ""


def test_tracklet_ids_for_frame_supports_keyboard_review_rhythm():
    review = pd.DataFrame([
        {"frame": 100, "tracklet_id": 2, "review_label": ""},
        {"frame": 100, "tracklet_id": 0, "review_label": ""},
        {"frame": 110, "tracklet_id": 1, "review_label": ""},
    ])

    ids = tracklet_ids_for_frame(review, frame=100)
    n_changed = set_review_labels(review, frame=100, tracklet_ids=ids, review_label="good")

    assert ids == [0, 2]
    assert n_changed == 2
    assert review.loc[review["frame"].eq(100), "review_label"].tolist() == ["good", "good"]
    assert review.loc[review["frame"].eq(110), "review_label"].iloc[0] == ""
