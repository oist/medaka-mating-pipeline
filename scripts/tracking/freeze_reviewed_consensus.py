from __future__ import annotations

"""Freeze a reviewed multiview-mouth consensus dataset as immutable manifest v1.

The manifest inventories every file already present in the review output
directory.  File hashes make later mutations detectable; the manifest itself
is never silently replaced.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA_VERSION = "reviewed_consensus_dataset_manifest_v1"
DATASET_VERSION = "v1"
DEFAULT_MANIFEST_NAME = "reviewed_consensus_manifest_v1.json"
CAMERA_SUMMARY_NAME = "camera_consensus_summary.json"
CLEAN_SUMMARY_NAME = "clean_reviewed_multiview_mouth_summary.json"
UNSAFE_CONSENSUS_PREFIX = "consensus_corrected_3d"


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON metadata {path}: {error}") from error


def _csv_metadata(path: Path) -> dict[str, Any]:
    try:
        table = pd.read_csv(path, low_memory=False)
    except Exception as error:
        raise ValueError(f"cannot read CSV artifact {path}: {error}") from error
    return {
        "row_count": int(len(table)),
        "schema": {
            "columns": [str(column) for column in table.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in table.dtypes.items()},
        },
    }


def _json_metadata(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    metadata: dict[str, Any] = {"json_type": type(payload).__name__}
    if isinstance(payload, dict):
        metadata["schema"] = {"top_level_keys": sorted(map(str, payload))}
    elif isinstance(payload, list):
        metadata["row_count"] = len(payload)
        keys = sorted(
            {
                str(key)
                for row in payload
                if isinstance(row, dict)
                for key in row
            }
        )
        if keys:
            metadata["schema"] = {"record_keys": keys}
    return metadata


def _jsonl_metadata(path: Path) -> dict[str, Any]:
    row_count = 0
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL artifact {path} at line {line_number}: {error}"
                ) from error
            row_count += 1
            if isinstance(row, dict):
                keys.update(map(str, row))
    metadata: dict[str, Any] = {"row_count": row_count}
    if keys:
        metadata["schema"] = {"record_keys": sorted(keys)}
    return metadata


def describe_artifact(path: Path, *, relative_to: Path) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": file_sha256(path),
        "byte_size": int(path.stat().st_size),
    }
    suffix = path.suffix.lower()
    if suffix == ".csv":
        artifact.update(_csv_metadata(path))
    elif suffix in {".json", ".ipynb"}:
        artifact.update(_json_metadata(path))
    elif suffix == ".jsonl":
        artifact.update(_jsonl_metadata(path))
    return artifact


def _resolve_source_ledger(
    review_dir: Path,
    camera_summary: dict[str, Any],
    explicit_path: Path | None,
) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"source ledger does not exist: {path}")
        return path

    recorded = camera_summary.get("source_events")
    if not recorded:
        raise ValueError(
            f"{CAMERA_SUMMARY_NAME} has no source_events; pass --source-ledger"
        )
    recorded_path = Path(str(recorded)).expanduser()
    candidates = [recorded_path]
    if not recorded_path.is_absolute():
        candidates.extend(
            [
                Path.cwd() / recorded_path,
                review_dir / recorded_path,
                review_dir.parent / recorded_path.name,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"cannot locate source ledger recorded as {recorded!r}; pass --source-ledger"
    )


def _frame_range(review_dir: Path) -> tuple[int, int, dict[str, list[int]]]:
    ranges: dict[str, list[int]] = {}
    for path in sorted(review_dir.glob("*.json"), key=lambda item: item.name):
        if path.name == DEFAULT_MANIFEST_NAME:
            continue
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        if "start_frame" in payload and "end_frame" in payload:
            ranges[path.name] = [int(payload["start_frame"]), int(payload["end_frame"])]
    if not ranges:
        raise ValueError("review directory contains no JSON summary with a frame range")
    unique = {tuple(value) for value in ranges.values()}
    if len(unique) != 1:
        raise ValueError(f"review summaries disagree on frame range: {ranges}")
    start, end = next(iter(unique))
    if start < 0 or end < start:
        raise ValueError(f"invalid review frame range: {start}..{end}")
    return start, end, ranges


def build_manifest(
    review_dir: Path,
    *,
    manifest_path: Path | None = None,
    source_ledger: Path | None = None,
) -> dict[str, Any]:
    review_dir = review_dir.expanduser().resolve()
    if not review_dir.is_dir():
        raise NotADirectoryError(f"review output directory does not exist: {review_dir}")
    camera_summary_path = review_dir / CAMERA_SUMMARY_NAME
    clean_summary_path = review_dir / CLEAN_SUMMARY_NAME
    for required in (camera_summary_path, clean_summary_path):
        if not required.is_file():
            raise FileNotFoundError(f"required review summary is missing: {required}")

    camera_summary = _read_json(camera_summary_path)
    clean_summary = _read_json(clean_summary_path)
    if not isinstance(camera_summary, dict) or not isinstance(clean_summary, dict):
        raise ValueError("review summaries must contain JSON objects")
    filters = clean_summary.get("filters")
    if not isinstance(filters, dict):
        raise ValueError(f"{CLEAN_SUMMARY_NAME} has no filters/gates object")
    start_frame, end_frame, range_sources = _frame_range(review_dir)
    ledger_path = _resolve_source_ledger(review_dir, camera_summary, source_ledger)

    output_path = (manifest_path or review_dir / DEFAULT_MANIFEST_NAME).resolve()
    artifacts = []
    for path in sorted(
        (item for item in review_dir.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(review_dir).as_posix(),
    ):
        if path.resolve() == output_path:
            continue
        artifacts.append(describe_artifact(path, relative_to=review_dir))
    if not artifacts:
        raise ValueError(f"review output directory contains no artifacts: {review_dir}")

    try:
        ledger_display = Path(os.path.relpath(ledger_path, review_dir)).as_posix()
    except ValueError:
        ledger_display = str(ledger_path)
    ledger_record = describe_artifact(ledger_path, relative_to=ledger_path.parent)
    ledger_record["path"] = ledger_display

    unsafe_artifacts = [
        artifact["path"]
        for artifact in artifacts
        if Path(artifact["path"]).name.startswith(UNSAFE_CONSENSUS_PREFIX)
    ]
    if not unsafe_artifacts:
        raise FileNotFoundError(
            "review dataset is missing the diagnostic consensus_corrected_3d artifact"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "immutability": {
            "mechanism": "content-addressed SHA-256 manifest",
            "rule": "Any artifact change invalidates this v1 manifest; create a new dataset version instead.",
        },
        "review_output_directory": str(review_dir),
        "experiment": camera_summary.get("experiment"),
        "frame_range": {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "inclusive": True,
            "frame_count": end_frame - start_frame + 1,
            "summary_sources": range_sources,
        },
        "coordinate_definition": {
            "point": "mouth",
            "description": "Per-view SLEAP mouth coordinates and multiview 3D reconstructed from those mouth points.",
        },
        "gates": filters,
        "source_ledger": ledger_record,
        "warnings": [
            {
                "severity": "unsafe",
                "artifacts": unsafe_artifacts,
                "message": (
                    "DIAGNOSTIC ONLY: consensus_corrected_3d is unsafe as ground truth, "
                    "training data, or a production trajectory because camera-consensus "
                    "relabeling can introduce 3D discontinuities. Use the gated clean "
                    "reviewed multiview-mouth dataset for trusted rows."
                ),
            }
        ],
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def freeze_manifest(
    review_dir: Path,
    *,
    output: Path | None = None,
    source_ledger: Path | None = None,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    review_dir = review_dir.expanduser().resolve()
    output_path = (output or review_dir / DEFAULT_MANIFEST_NAME).expanduser().resolve()
    if output_path.exists() and not force:
        raise FileExistsError(
            f"manifest already exists: {output_path}; pass --force to replace it explicitly"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        review_dir,
        manifest_path=output_path,
        source_ledger=source_ledger,
    )
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(output_path)
    return output_path, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_dir", type=Path, help="Reviewed consensus output directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Manifest path (default: REVIEW_DIR/{DEFAULT_MANIFEST_NAME}).",
    )
    parser.add_argument(
        "--source-ledger",
        type=Path,
        default=None,
        help="Override the source review-event ledger recorded by the camera summary.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace an existing manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output, manifest = freeze_manifest(
        args.review_dir,
        output=args.output,
        source_ledger=args.source_ledger,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "manifest": str(output),
                "dataset_version": manifest["dataset_version"],
                "artifact_count": manifest["artifact_count"],
                "source_ledger_sha256": manifest["source_ledger"]["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
