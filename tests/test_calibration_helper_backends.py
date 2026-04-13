from pathlib import Path

import pandas as pd

from scripts.calibration.lib.calibration_core import load_calibration_data, run_calibration_pipeline
from scripts.calibration.lib.visualize_coords_from_template import get_colors_from_template, parse_template


def _write_template(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "CORNERS",
                "0 0 0 corner_1",
                "190 0 0 corner_2",
                "190 190 0 corner_3",
                "0 190 0 corner_4",
                "BEADS_WAY1",
                "27.5 27.5 27.5 bead_001",
                "50 50 50 bead_002",
            ]
        ),
        encoding="utf-8",
    )


def _write_image_coords(path: Path, cam_name: str) -> None:
    rows = {
        "cam0": [
            "0 0 0 0 0 0 CORNER: corner_1",
            "1 190 0 190 0 0 CORNER: corner_2",
            "2 190 190 190 190 0 CORNER: corner_3",
            "3 0 190 0 190 0 CORNER: corner_4",
            "4 27 27 27.5 27.5 27.5 BEAD: bead_001",
            "5 50 50 50 50 50 BEAD: bead_002",
        ],
        "cam1": [
            "0 0 0 0 0 0 CORNER: corner_1",
            "1 190 0 190 0 0 CORNER: corner_2",
            "2 190 0 190 190 0 CORNER: corner_3",
            "3 0 0 0 190 0 CORNER: corner_4",
            "4 27 27 27.5 27.5 27.5 BEAD: bead_001",
            "5 50 50 50 50 50 BEAD: bead_002",
        ],
        "cam2": [
            "0 0 0 0 0 0 CORNER: corner_1",
            "1 0 0 190 0 0 CORNER: corner_2",
            "2 190 0 190 190 0 CORNER: corner_3",
            "3 190 0 0 190 0 CORNER: corner_4",
            "4 27 27 27.5 27.5 27.5 BEAD: bead_001",
            "5 50 50 50 50 50 BEAD: bead_002",
        ],
    }
    path.write_text(
        "# Manual annotation file\n# Format: index x y X Y Z label\n#\n" + "\n".join(rows[cam_name]) + "\n",
        encoding="utf-8",
    )


def test_template_helpers_parse_and_color(tmp_path):
    template = tmp_path / "cali_coords_template_way1.txt"
    _write_template(template)

    sections = parse_template(template)
    coords, colors = get_colors_from_template(template)

    assert "CORNERS" in sections
    assert "BEADS_WAY1" in sections
    assert coords.shape == (6, 3)
    assert colors[:4] == ["green"] * 4


def test_template_header_with_bracket_suffix_still_parses_as_corners(tmp_path):
    template = tmp_path / "cali_coords_template_way1.txt"
    template.write_text(
        "\n".join(
            [
                "[CORNERS] (black dots), clockwise marking",
                "corner_1,0,0,190",
                "corner_2,190,0,190",
                "[BEADS_WAY1]",
                "27.5,27.5,27.5",
            ]
        ),
        encoding="utf-8",
    )

    sections = parse_template(template)

    assert "CORNERS" in sections
    assert sections["CORNERS"][0].shape == (2, 3)


def test_load_calibration_data_and_run_pipeline(tmp_path):
    raw_dir = tmp_path / "calibration_raw"
    raw_dir.mkdir()
    template = raw_dir / "cali_coords_template_way1.txt"
    _write_template(template)

    image_dir = tmp_path / "calibration_results"
    image_dir.mkdir()
    for cam in ("cam0", "cam1", "cam2"):
        _write_image_coords(image_dir / f"cali_way1_{cam}_image_coords.txt", cam)

    template_3d, image_coords, point_labels, metadata = load_calibration_data(
        calibration_folder=str(raw_dir),
        image_coords_folder=str(image_dir),
        filename_keyword="way1",
        way_name="way1",
        beads_only=False,
    )

    assert template_3d.shape == (6, 3)
    assert image_coords.shape == (3, 6, 2)
    assert len(point_labels) == 6
    assert metadata["template_path"].endswith("cali_coords_template_way1.txt")

    out_dir = tmp_path / "models_out"
    results_dir = tmp_path / "results_out"
    result = run_calibration_pipeline(
        calibration_folder=str(raw_dir),
        image_coords_folder=str(image_dir),
        filename_keyword="way1",
        alpha_range=[1e-3],
        poly_deg_range=[1],
        cv_folds=2,
        train_fraction=0.8,
        num_permutations=2,
        output_folder=str(out_dir),
        beads_only=False,
        model_types=["polynomial"],
        results_root=str(results_dir),
    )

    assert (out_dir / "imCoords_to_XYZ.joblib").exists()
    assert (out_dir / "xy_yz_to_xz.joblib").exists()
    assert (out_dir / "xz_yz_to_xy.joblib").exists()
    assert (out_dir / "xz_xy_to_yz.joblib").exists()
    assert Path(result["mapping_csv"]).exists()
    df = pd.read_csv(result["mapping_csv"])
    assert {"X", "Y", "Z", "pred_X", "pred_Y", "pred_Z", "cam0_x", "cam1_x", "cam2_x"}.issubset(df.columns)


def test_load_calibration_data_keeps_extra_annotation_points_not_in_template(tmp_path):
    raw_dir = tmp_path / "calibration_raw"
    raw_dir.mkdir()
    template = raw_dir / "cali_coords_template_way1.txt"
    _write_template(template)

    image_dir = tmp_path / "calibration_results"
    image_dir.mkdir()
    extra_lines = {
        "cam0": "53 5 5 0.000000 0.000000 0.000000 CORNER_BOTTOM: corner_1_bottom",
        "cam1": "53 6 6 0.000000 0.000000 0.000000 CORNER_BOTTOM: corner_1_bottom",
        "cam2": "53 7 7 0.000000 0.000000 0.000000 CORNER_BOTTOM: corner_1_bottom",
    }
    for cam in ("cam0", "cam1", "cam2"):
        path = image_dir / f"cali_way1_{cam}_image_coords.txt"
        _write_image_coords(path, cam)
        with path.open("a", encoding="utf-8") as f:
            f.write(extra_lines[cam] + "\n")

    template_3d, image_coords, point_labels, metadata = load_calibration_data(
        calibration_folder=str(raw_dir),
        image_coords_folder=str(image_dir),
        filename_keyword="way1",
        way_name="way1",
        beads_only=False,
    )

    assert template_3d.shape == (7, 3)
    assert tuple(template_3d[-1]) == (0.0, 0.0, 0.0)
    assert point_labels[-1] == "CORNER_BOTTOM: corner_1_bottom"
