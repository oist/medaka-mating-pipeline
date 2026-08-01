from __future__ import annotations

"""Independently audit completed training-only strict-mouth geometry windows."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("report/20260731/round8_mouth_tracking_ablation")
GEOMETRY_ROOT = ROOT / "training_geometry"
OUTPUT = ROOT / "quality_review" / "training_geometry_independent_audit.json"
EXPECTED = {
    "reviewed_event_1": (72371, 73027),
    "control_for_event_1": (32264, 32920),
    "reviewed_event_3": (174316, 180515),
    "control_for_event_3": (21840, 28039),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def distribution(values: Any) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "count": 0,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def normalized_configuration(configuration: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(configuration))
    normalized.pop("frame_start", None)
    normalized.pop("frame_end", None)
    return normalized


def audit_window(name: str, frame_start: int, frame_end: int) -> dict[str, Any]:
    folder = GEOMETRY_ROOT / name
    summary_path = folder / "geometry_summary.json"
    parquet_path = folder / "strict_mouth_selected_geometry.parquet"
    resource_path = folder / "geometry_resource_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    table = pd.read_parquet(parquet_path)
    expected_frames = frame_end - frame_start + 1
    xyz = table[["X", "Y", "Z"]].to_numpy(dtype=float)
    finite_xyz = np.all(np.isfinite(xyz), axis=1)
    finite_residual = np.isfinite(
        table.ray_residual_mm2.to_numpy(dtype=float)
    )
    frame_counts = (
        table.assign(_direct=finite_xyz & finite_residual)
        .groupby("frame", sort=True)
        ._direct.sum()
        .reindex(range(frame_start, frame_end + 1), fill_value=0)
    )
    observed_frames = set(table.frame.astype(int))
    missing_frames = sorted(
        set(range(frame_start, frame_end + 1)) - observed_frames
    )
    outside = table[
        ~table.frame.between(frame_start, frame_end, inclusive="both")
    ]
    duplicate_hypotheses = int(
        table.duplicated(["frame", "hypothesis_id"]).sum()
    )
    residual_by_view_count = {
        str(int(view_count)): distribution(group.ray_residual_mm2)
        for view_count, group in table.groupby("view_count", sort=True)
    }
    output_hash = sha256(parquet_path)
    resource = (
        json.loads(resource_path.read_text(encoding="utf-8"))
        if resource_path.exists()
        else None
    )
    return {
        "expected_interval_inclusive": [frame_start, frame_end],
        "expected_frame_count": expected_frames,
        "summary_path": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "parquet_path": str(parquet_path),
        "parquet_sha256": output_hash,
        "summary_status": summary.get("status"),
        "summary_output_hash_matches": (
            summary.get("output_sha256") == output_hash
        ),
        "summary_interval_matches": (
            summary.get("configuration", {}).get("frame_start") == frame_start
            and summary.get("configuration", {}).get("frame_end") == frame_end
        ),
        "rows": int(len(table)),
        "unique_frames_with_rows": int(table.frame.nunique()),
        "finite_xyz_and_residual_rows": int((finite_xyz & finite_residual).sum()),
        "nonfinite_xyz_or_residual_rows": int(
            (~(finite_xyz & finite_residual)).sum()
        ),
        "all_three_direct_frames": int((frame_counts == 3).sum()),
        "all_three_direct_fraction": float((frame_counts == 3).mean()),
        "frames_with_0_1_2_3plus_direct": {
            "0": int((frame_counts == 0).sum()),
            "1": int((frame_counts == 1).sum()),
            "2": int((frame_counts == 2).sum()),
            "3": int((frame_counts == 3).sum()),
            "greater_than_3": int((frame_counts > 3).sum()),
        },
        "missing_frame_count": int(len(missing_frames)),
        "first_missing_frames": missing_frames[:20],
        "outside_interval_row_count": int(len(outside)),
        "duplicate_frame_hypothesis_rows": duplicate_hypotheses,
        "view_count_rows": {
            str(int(key)): int(value)
            for key, value in table.view_count.value_counts().sort_index().items()
        },
        "ray_residual_mm2": distribution(table.ray_residual_mm2),
        "ray_residual_mm2_by_view_count": residual_by_view_count,
        "normalized_configuration": normalized_configuration(
            summary["configuration"]
        ),
        "normalized_configuration_sha256": stable_hash(
            normalized_configuration(summary["configuration"])
        ),
        "calibration_sha256": stable_hash(summary.get("calibration")),
        "cache_metadata_sha256": stable_hash(
            summary.get("cache_metadata")
        ),
        "code_sha256": summary.get("code_sha256"),
        "resource_summary": resource,
    }


def main() -> None:
    windows = {
        name: audit_window(name, *interval)
        for name, interval in EXPECTED.items()
    }
    normalized_hashes = {
        record["normalized_configuration_sha256"]
        for record in windows.values()
    }
    calibration_hashes = {
        record["calibration_sha256"] for record in windows.values()
    }
    cache_hashes = {
        record["cache_metadata_sha256"] for record in windows.values()
    }
    code_hashes = {
        stable_hash(record["code_sha256"]) for record in windows.values()
    }
    result = {
        "schema_version": 1,
        "scope": (
            "Independent audit of training-only geometry outputs. "
            "No validation, held-out, or critical-event method output was read."
        ),
        "windows": windows,
        "cross_window": {
            "normalized_configuration_equal": len(normalized_hashes) == 1,
            "calibration_provenance_equal": len(calibration_hashes) == 1,
            "candidate_cache_metadata_equal": len(cache_hashes) == 1,
            "code_hash_manifest_equal": len(code_hashes) == 1,
            "normalized_configuration_sha256": sorted(normalized_hashes),
            "calibration_sha256": sorted(calibration_hashes),
            "cache_metadata_sha256": sorted(cache_hashes),
            "code_manifest_sha256": sorted(code_hashes),
        },
        "overall_pass": bool(
            len(normalized_hashes) == 1
            and len(calibration_hashes) == 1
            and len(cache_hashes) == 1
            and len(code_hashes) == 1
            and all(
                record["summary_status"] == "completed"
                and record["summary_output_hash_matches"]
                and record["summary_interval_matches"]
                and record["outside_interval_row_count"] == 0
                and record["duplicate_frame_hypothesis_rows"] == 0
                for record in windows.values()
            )
        ),
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
