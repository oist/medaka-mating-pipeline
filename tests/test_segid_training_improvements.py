from __future__ import annotations

import importlib

import numpy as np
import pandas as pd


segid_dataset = importlib.import_module("sam2_id_experiment.scripts.33_build_segid_dataset")
segid_assemble = importlib.import_module("sam2_id_experiment.scripts.34_assemble_segid_training")
identity_classifier = importlib.import_module("sam2_id_experiment.scripts.39_train_identity_classifier")


def test_square_crop_can_mask_background():
    img = np.full((8, 8, 3), 200, dtype=np.uint8)
    img[3:5, 3:5] = 50
    mask = np.zeros((8, 8), dtype=bool)
    mask[3:5, 3:5] = True

    crop = segid_dataset._square_crop(
        img,
        bbox=(3, 3, 5, 5),
        size=8,
        pad_frac=0.0,
        mask=mask,
        background_value=0,
    )

    assert crop.shape == (8, 8, 3)
    assert (crop == 0).any()
    assert (crop == 50).any()


def test_global_id_stats_reports_modal_purity(tmp_path):
    tb = pd.DataFrame(
        {
            "tracklet_id": [1, 1, 1, 1, 2, 2],
            "global_id": [0, 0, 0, 1, 2, 2],
        }
    )
    path = tmp_path / "top_cam.tracklet_bound.csv"
    tb.to_csv(path, index=False)

    stats = segid_assemble._global_id_stats(str(path))

    assert stats[1]["global_id"] == 0
    assert stats[1]["global_id_modal_n"] == 3
    assert stats[1]["global_id_n"] == 4
    assert stats[1]["global_id_purity"] == 0.75
    assert stats[2]["global_id_purity"] == 1.0


def test_window_split_holds_out_latest_window():
    meta = pd.DataFrame(
        {
            "crop_id": range(12),
            "qc_status": ["clean_train"] * 12,
            "view_tracklet": [0, 0, 1, 1, 2, 2, 10, 10, 11, 11, 12, 12],
            "tracklet_id": [0, 0, 1, 1, 2, 2, 10, 10, 11, 11, 12, 12],
            "frame": [0, 1, 0, 1, 0, 1, 100, 101, 100, 101, 100, 101],
            "window": ["early"] * 6 + ["late"] * 6,
            "global_id": [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2],
            "global_id_purity": [1.0] * 12,
        }
    )
    args = type(
        "Args",
        (),
        {
            "label_col": "global_id",
            "min_global_id_purity": 0.9,
            "min_tracklet_crops": 1,
            "mode": "all3",
            "classes": None,
            "female_gid": 1,
            "split": "window",
            "holdout_window": None,
            "val_frac": 0.25,
        },
    )()

    use, _class_values, _class_names = identity_classifier.prepare_labeled_metadata(meta, args)
    train, val, info = identity_classifier.make_split(use, args)

    assert info["kind"] == "window"
    assert info["holdout_window"] == "late"
    assert set(train["window"]) == {"early"}
    assert set(val["window"]) == {"late"}
