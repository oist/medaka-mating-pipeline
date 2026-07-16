from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.tracking import freeze_reviewed_consensus as freezer


def _write_review_fixture(root: Path) -> tuple[Path, Path]:
    review_dir = root / "review_0_15000_consensus"
    review_dir.mkdir(parents=True)
    ledger = root / "geometry_review_events.csv"
    ledger.write_text(
        "event_id,event_type,frame_start,frame_end\nE1,pair_swap,10,12\n",
        encoding="utf-8",
    )
    (review_dir / freezer.CAMERA_SUMMARY_NAME).write_text(
        json.dumps(
            {
                "schema_version": "camera_mapping_consensus_v1",
                "experiment": "20251225_K_1_2_FCMC",
                "start_frame": 0,
                "end_frame": 15000,
                "source_events": str(ledger),
            }
        ),
        encoding="utf-8",
    )
    (review_dir / freezer.CLEAN_SUMMARY_NAME).write_text(
        json.dumps(
            {
                "schema_version": "clean_reviewed_multiview_mouth_tracklets_v1",
                "start_frame": 0,
                "end_frame": 15000,
                "filters": {
                    "all_three_view_mappings_agree": True,
                    "max_geometry_to_sleap_centroid_match_px": 30.0,
                    "max_reprojection_error_px": 15.0,
                    "interpolation": "none",
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "frame": [0, 0, 0, 1, 1, 1],
            "identity": ["H1", "H2", "H3"] * 2,
            "X": [1.0, 2.0, 3.0, 1.1, 2.1, 3.1],
        }
    ).to_csv(review_dir / "clean_reviewed_multiview_mouth_3d_f0_15000.csv", index=False)
    pd.DataFrame(
        {
            "frame": [0, 0, 0],
            "identity": ["H1", "H2", "H3"],
            "X": [1.0, 2.0, 3.0],
        }
    ).to_csv(review_dir / "consensus_corrected_3d_f0_15000.csv", index=False)
    return review_dir, ledger


def test_build_manifest_hashes_artifacts_ledger_and_schema(tmp_path: Path) -> None:
    review_dir, ledger = _write_review_fixture(tmp_path)

    manifest = freezer.build_manifest(review_dir)

    assert manifest["schema_version"] == freezer.SCHEMA_VERSION
    assert manifest["dataset_version"] == "v1"
    assert manifest["frame_range"] == {
        "start_frame": 0,
        "end_frame": 15000,
        "inclusive": True,
        "frame_count": 15001,
        "summary_sources": {
            freezer.CAMERA_SUMMARY_NAME: [0, 15000],
            freezer.CLEAN_SUMMARY_NAME: [0, 15000],
        },
    }
    assert manifest["coordinate_definition"]["point"] == "mouth"
    assert manifest["gates"]["max_geometry_to_sleap_centroid_match_px"] == 30.0
    assert manifest["gates"]["max_reprojection_error_px"] == 15.0
    assert manifest["source_ledger"]["sha256"] == hashlib.sha256(
        ledger.read_bytes()
    ).hexdigest()
    assert manifest["source_ledger"]["byte_size"] == ledger.stat().st_size
    assert manifest["source_ledger"]["row_count"] == 1

    artifact_by_name = {item["path"]: item for item in manifest["artifacts"]}
    clean = artifact_by_name["clean_reviewed_multiview_mouth_3d_f0_15000.csv"]
    assert clean["row_count"] == 6
    assert clean["schema"]["columns"] == ["frame", "identity", "X"]
    assert clean["byte_size"] == (
        review_dir / "clean_reviewed_multiview_mouth_3d_f0_15000.csv"
    ).stat().st_size
    assert len(clean["sha256"]) == 64

    warning = manifest["warnings"][0]
    assert warning["severity"] == "unsafe"
    assert warning["artifacts"] == ["consensus_corrected_3d_f0_15000.csv"]
    assert "DIAGNOSTIC ONLY" in warning["message"]
    assert "unsafe" in warning["message"]


def test_freeze_refuses_silent_overwrite_and_force_is_explicit(tmp_path: Path) -> None:
    review_dir, _ = _write_review_fixture(tmp_path)

    output, first = freezer.freeze_manifest(review_dir)
    first_bytes = output.read_bytes()
    assert json.loads(first_bytes)["dataset_version"] == "v1"

    with pytest.raises(FileExistsError, match="--force"):
        freezer.freeze_manifest(review_dir)
    assert output.read_bytes() == first_bytes

    forced_output, forced = freezer.freeze_manifest(review_dir, force=True)
    assert forced_output == output
    assert forced["artifact_count"] == first["artifact_count"]
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == freezer.SCHEMA_VERSION


def test_manifest_rejects_inconsistent_summary_frame_ranges(tmp_path: Path) -> None:
    review_dir, _ = _write_review_fixture(tmp_path)
    (review_dir / "incorrect_summary.json").write_text(
        json.dumps({"start_frame": 1, "end_frame": 15000}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="disagree on frame range"):
        freezer.build_manifest(review_dir)


def test_manifest_requires_diagnostic_artifact_warning_target(tmp_path: Path) -> None:
    review_dir, _ = _write_review_fixture(tmp_path)
    (review_dir / "consensus_corrected_3d_f0_15000.csv").unlink()

    with pytest.raises(FileNotFoundError, match="diagnostic consensus_corrected_3d"):
        freezer.build_manifest(review_dir)
