"""Create an authenticated A5-input copy of an A4 selected-observation table.

The only transformation permitted is adding the canonical empty schema when an
authenticated A4 scientific abstention wrote a zero-row, zero-column parquet.
Nonempty inputs are copied byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


REPO = Path(__file__).resolve().parents[3]
FROZEN_BUILDER_SHA256 = (
    "1e75e10654b5d6c233ec3ea5ff2f8c969dac3dd41027031cd332d3c88096d4e0"
)
FROZEN_BUILDER = REPO / "scripts" / "reconstruction_3d" / "mouth_tracklet_ablation.py"

CANONICAL_DTYPES = {
    "frame": "int32",
    "view_mask": "uint8",
    "X": "float64",
    "Y": "float64",
    "Z": "float64",
    "ray_residual_mm2": "float64",
    "residual_unit": "string",
    "slot_cam0": "int16",
    "tracklet_top_cam": "int64",
    "slot_cam1": "int16",
    "tracklet_side1_cam": "int64",
    "slot_cam2": "int16",
    "tracklet_side2_cam": "int64",
    "hypothesis_id": "string",
    "view_count": "int64",
    "mouth_confidence": "float64",
    "retained_tracklet_count_mismatch": "bool",
    "all_declared_tracklets_retained": "bool",
    "group_id": "string",
    "group_assignment_cost": "float64",
    "group_support_frames": "int64",
    "group_median_ray_residual_mm2": "float64",
    "group_mad_ray_residual_mm2": "float64",
    "group_median_mouth_confidence": "float64",
    "group_retained_count_mismatch_fraction": "float64",
    "group_eligible": "bool",
    "group_rejection_reason": "string",
    "path_id": "string",
    "path_segment": "int64",
}
EXPECTED_NONEMPTY_PARQUET_READ_DTYPES = {
    column: ("str" if dtype == "string" else dtype)
    for column, dtype in CANONICAL_DTYPES.items()
}
EXPECTED_EMPTY_PARQUET_READ_DTYPES = dict(CANONICAL_DTYPES)
BUILDER_REQUIRED_COLUMNS = {
    "frame",
    "path_id",
    "path_segment",
    "hypothesis_id",
    "X",
    "Y",
    "Z",
    "mouth_confidence",
    "view_count",
    "group_eligible",
    "group_support_frames",
    "group_median_ray_residual_mm2",
    "group_mad_ray_residual_mm2",
    "group_median_mouth_confidence",
    "group_assignment_cost",
    "group_retained_count_mismatch_fraction",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO)).replace("/", "\\")


def write_json_atomic(path: Path, value: object) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a4-summary", required=True, type=Path)
    parser.add_argument("--a4-selected", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "a4_selected_for_a5.parquet"
    summary_path = args.output_dir / "a4_selected_for_a5_summary.json"
    if target.exists() or summary_path.exists():
        raise FileExistsError("Refusing to overwrite existing adapter output")
    assert sha256(FROZEN_BUILDER) == FROZEN_BUILDER_SHA256
    a4_summary = json.loads(args.a4_summary.read_text(encoding="utf-8"))
    selected_record = a4_summary["outputs"]["selected"]
    assert Path(selected_record["path"]).resolve() == args.a4_selected.resolve()
    source_sha256 = sha256(args.a4_selected)
    assert source_sha256 == selected_record["sha256"]
    assert a4_summary["code_sha256"][
        "scripts\\reconstruction_3d\\mouth_tracklet_ablation.py"
    ] == FROZEN_BUILDER_SHA256

    source = pd.read_parquet(args.a4_selected)
    source_columns = list(source.columns)
    source_dtypes = {column: str(dtype) for column, dtype in source.dtypes.items()}
    temporary = target.with_name(f".{target.name}.tmp.parquet")
    if temporary.exists():
        raise FileExistsError(temporary)
    if a4_summary["status"] == "insufficient_three_path_solution":
        assert len(source) == 0
        if source_columns:
            assert source_columns == list(CANONICAL_DTYPES)
            assert source_dtypes == EXPECTED_EMPTY_PARQUET_READ_DTYPES
            shutil.copyfile(args.a4_selected, temporary)
            mode = "canonical_empty_byte_copy"
        else:
            canonical = pd.DataFrame(
                {
                    column: pd.Series([], dtype=dtype)
                    for column, dtype in CANONICAL_DTYPES.items()
                }
            )
            canonical.to_parquet(temporary, index=False)
            mode = "zero_row_zero_column_abstention_schema_normalized"
    else:
        assert a4_summary["status"] == "completed"
        assert source_columns == list(CANONICAL_DTYPES)
        assert source_dtypes == EXPECTED_NONEMPTY_PARQUET_READ_DTYPES
        assert len(source) > 0
        shutil.copyfile(args.a4_selected, temporary)
        mode = "nonempty_byte_identical_copy"
    os.replace(temporary, target)

    adapted = pd.read_parquet(target)
    assert list(adapted.columns) == list(CANONICAL_DTYPES)
    assert BUILDER_REQUIRED_COLUMNS <= set(adapted.columns)
    if a4_summary["status"] == "insufficient_three_path_solution":
        assert len(adapted) == 0
    else:
        assert sha256(target) == source_sha256
        assert len(adapted) == len(source)
    output_dtypes = {
        column: str(dtype) for column, dtype in adapted.dtypes.items()
    }
    expected_output_dtypes = (
        EXPECTED_EMPTY_PARQUET_READ_DTYPES
        if len(adapted) == 0
        else EXPECTED_NONEMPTY_PARQUET_READ_DTYPES
    )
    assert output_dtypes == expected_output_dtypes
    record = {
        "schema_version": "round8_a4_to_a5_schema_adapter_v1",
        "status": "completed",
        "mode": mode,
        "created_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "command": [sys.executable, *sys.argv],
        "adapter": {
            "path": relative(Path(__file__)),
            "sha256": sha256(Path(__file__)),
        },
        "frozen_builder": {
            "path": relative(FROZEN_BUILDER),
            "sha256": FROZEN_BUILDER_SHA256,
            "required_columns": sorted(BUILDER_REQUIRED_COLUMNS),
        },
        "a4_scientific_status": a4_summary["status"],
        "inputs": {
            "a4_summary": {
                "path": relative(args.a4_summary),
                "sha256": sha256(args.a4_summary),
            },
            "a4_selected": {
                "path": relative(args.a4_selected),
                "sha256": source_sha256,
                "rows": len(source),
                "columns": source_columns,
                "dtypes": source_dtypes,
            },
        },
        "output": {
            "path": relative(target),
            "sha256": sha256(target),
            "rows": len(adapted),
            "columns": list(adapted.columns),
            "dtypes": output_dtypes,
        },
        "canonical_schema": CANONICAL_DTYPES,
        "expected_output_parquet_read_dtypes": expected_output_dtypes,
        "scientific_rows_added_removed_or_changed": False,
        "identity_information_added": False,
        "thresholds_changed": False,
    }
    write_json_atomic(summary_path, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
