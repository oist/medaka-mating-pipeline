from __future__ import annotations

"""Independent contract and critical-event audit for Round8 candidate caches."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


VIEWS = ("top_cam", "side1_cam", "side2_cam")
COORDINATE_TOLERANCE_PX = 1e-9


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        attrs = {
            str(key): (
                value.item() if isinstance(value, np.generic) else value
            )
            for key, value in handle.attrs.items()
        }
        mouth_scores = np.asarray(handle["mouth_score"])
        finite_mouth_scores = np.isfinite(mouth_scores)
        attrs["finite_mouth_score_count"] = int(finite_mouth_scores.sum())
        attrs["finite_mouth_score_fraction"] = float(
            finite_mouth_scores.mean()
        )
        finite_mouth_rows = np.isfinite(np.asarray(handle["mouth_xy"])).all(axis=2)
        attrs["finite_mouth_coordinate_row_count"] = int(
            finite_mouth_rows.sum()
        )
        attrs["finite_mouth_rows_with_finite_score"] = int(
            (finite_mouth_rows & finite_mouth_scores).sum()
        )
    attrs["path"] = str(path)
    attrs["size_bytes"] = path.stat().st_size
    attrs["sha256"] = sha256(path)
    return attrs


def cache_event_match(
    path: Path, raw: pd.DataFrame, view: str, frame_start: int, frame_end: int
) -> dict[str, Any]:
    maximum_error = 0.0
    maximum_mouth_score_error = 0.0
    compared_points = 0
    frame_mismatches: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        cached = np.asarray(handle["mouth_xy"][frame_start : frame_end + 1])
        scores = np.asarray(handle["instance_score"][frame_start : frame_end + 1])
        cached_mouth_scores = np.asarray(
            handle["mouth_score"][frame_start : frame_end + 1]
        )
    for local, frame in enumerate(range(frame_start, frame_end + 1)):
        rows = raw[(raw.view == view) & (raw.frame == frame)].copy()
        rows["sort_score"] = rows.instance_score.fillna(-np.inf)
        rows = rows.sort_values(
            ["sort_score", "mouth_x"],
            ascending=[False, True],
            kind="stable",
        ).head(cached.shape[1])
        expected = rows[["mouth_x", "mouth_y"]].to_numpy(dtype=float)
        actual_mask = np.isfinite(cached[local]).all(axis=1)
        actual = cached[local][actual_mask]
        actual_scores = scores[local][actual_mask]
        actual_mouth_scores = cached_mouth_scores[local][actual_mask]
        actual_order = np.lexsort((actual[:, 0], -actual_scores))
        actual = actual[actual_order]
        actual_mouth_scores = actual_mouth_scores[actual_order]
        if expected.shape != actual.shape:
            frame_mismatches.append(
                {
                    "frame": frame,
                    "expected_count": int(len(expected)),
                    "actual_count": int(len(actual)),
                }
            )
            continue
        if len(expected):
            error = float(np.max(np.abs(expected - actual)))
            expected_mouth_scores = rows["mouth_score"].to_numpy(dtype=float)
            mouth_score_error = (
                float(
                    np.max(
                        np.abs(expected_mouth_scores - actual_mouth_scores)
                    )
                )
                if np.isfinite(expected_mouth_scores).all()
                and np.isfinite(actual_mouth_scores).all()
                else np.inf
            )
            maximum_error = max(maximum_error, error)
            maximum_mouth_score_error = max(
                maximum_mouth_score_error, mouth_score_error
            )
            compared_points += int(len(expected))
            if (
                error > COORDINATE_TOLERANCE_PX
                or mouth_score_error > COORDINATE_TOLERANCE_PX
            ):
                frame_mismatches.append(
                    {
                        "frame": frame,
                        "expected_count": int(len(expected)),
                        "actual_count": int(len(actual)),
                        "maximum_absolute_coordinate_error_px": error,
                        "maximum_absolute_mouth_score_error": mouth_score_error,
                    }
                )
    return {
        "frame_start": frame_start,
        "frame_end": frame_end,
        "compared_mouth_points": compared_points,
        "maximum_absolute_coordinate_error_px": maximum_error,
        "maximum_absolute_mouth_score_error": maximum_mouth_score_error,
        "coordinate_tolerance_px": COORDINATE_TOLERANCE_PX,
        "mismatch_count": len(frame_mismatches),
        "mismatches": frame_mismatches,
    }


def compare_cache_datasets(
    reference_path: Path, candidate_path: Path
) -> dict[str, Any]:
    datasets = (
        "mouth_xy",
        "tail_xy",
        "instance_score",
        "visibility",
        "source_instance",
    )
    equality: dict[str, bool] = {}
    with h5py.File(reference_path, "r") as reference, h5py.File(
        candidate_path, "r"
    ) as candidate:
        for name in datasets:
            equality[name] = bool(
                np.array_equal(
                    np.asarray(reference[name]),
                    np.asarray(candidate[name]),
                    equal_nan=True,
                )
            )
    return {
        "reference_path": str(reference_path),
        "reference_sha256": sha256(reference_path),
        "datasets_bit_exact": equality,
        "all_compared_datasets_bit_exact": all(equality.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--rejected-root", type=Path)
    parser.add_argument("--coordinate-reference-root", type=Path)
    parser.add_argument("--raw-event-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw = pd.read_csv(args.raw_event_csv)
    canonical: dict[str, Any] = {}
    for view in VIEWS:
        path = args.canonical_root / f"{view}.candidates.h5"
        attrs = metadata(path)
        contract = {
            "complete": int(attrs.get("complete", 0)) == 1,
            "anchor_node_mouth": attrs.get("anchor_node") == "mouth",
            "require_anchor_node": int(attrs.get("require_anchor_node", 0)) == 1,
            "filter_fp_disabled": int(attrs.get("filter_fp", 1)) == 0,
            "six_slots": int(attrs.get("num_candidates", 0)) == 6,
            "full_frame_count": int(attrs.get("frame_count", 0)) == 348600,
            "float64": attrs.get("coordinate_dtype") == "float64",
            "mouth_scores_available_for_all_finite_mouth_rows": (
                int(attrs["finite_mouth_rows_with_finite_score"])
                == int(attrs["finite_mouth_coordinate_row_count"])
                and int(attrs["finite_mouth_coordinate_row_count"]) > 0
            ),
        }
        canonical[view] = {
            "metadata": attrs,
            "contract": contract,
            "contract_pass": all(contract.values()),
            "critical_event_raw_match": cache_event_match(
                path, raw, view, 34540, 34565
            ),
        }
        if args.coordinate_reference_root is not None:
            canonical[view]["coordinate_reference_comparison"] = (
                compare_cache_datasets(
                    args.coordinate_reference_root
                    / f"{view}.candidates.h5",
                    path,
                )
            )

    rejected: dict[str, Any] | None = None
    if args.rejected_root is not None:
        rejected = {}
        for view in VIEWS:
            path = args.rejected_root / f"{view}.candidates.h5"
            attrs = metadata(path)
            rejected[view] = {
                "metadata": attrs,
                "filter_fp_enabled": int(attrs.get("filter_fp", 0)) == 1,
                "excluded_from_method_qc": True,
                "reason": (
                    "legacy filter_fp reads tail/body span and therefore violates "
                    "the strict mouth-only candidate-admission contract"
                ),
            }
    output = {
        "schema_version": 1,
        "canonical": canonical,
        "canonical_all_contracts_pass": all(
            item["contract_pass"] for item in canonical.values()
        ),
        "canonical_event_matches_raw_within_float64_tolerance": all(
            item["critical_event_raw_match"]["mismatch_count"] == 0
            and item["critical_event_raw_match"][
                "maximum_absolute_coordinate_error_px"
            ]
            <= COORDINATE_TOLERANCE_PX
            and item["critical_event_raw_match"][
                "maximum_absolute_mouth_score_error"
            ]
            <= COORDINATE_TOLERANCE_PX
            for item in canonical.values()
        ),
        "rejected_diagnostic": rejected,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
