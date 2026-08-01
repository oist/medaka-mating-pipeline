"""Independent training-control clean-motion and body-scale audit."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from scripts.reconstruction_3d.candidate_cache import (
    VIEW_NAMES,
    load_candidate_window,
)


ROOT = Path("report/20260731/round8_mouth_tracking_ablation")
CACHE_DIR = ROOT / "candidate_cache_strict_mouth_unfiltered_nodescores"
SCALES_PATH = ROOT / "frozen_body_length_scales.json"
OUTPUT_PATH = ROOT / "quality_review" / "training_clean_motion_audit.json"
CONTROL_INTERVALS = ((32264, 32920), (21840, 28039))
MINIMUM_SEPARATION_BODY_LENGTHS = 1.5
MINIMUM_MOUTH_SCORE = 0.3
HISTORY_FRAMES = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_positive_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values) & (values > 0)]
    if not len(finite):
        raise ValueError("no finite positive body lengths")
    median = float(np.median(finite))
    deviation = np.abs(finite - median)
    mad = float(np.median(deviation))
    if mad > 0:
        kept = finite[deviation <= 4.0 * 1.4826 * mad]
        if len(kept):
            median = float(np.median(kept))
    return median


def distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "n": 0,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "n": int(len(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def robust_velocity(frames: list[int], points: list[np.ndarray]) -> np.ndarray:
    frames = frames[-HISTORY_FRAMES:]
    points = points[-HISTORY_FRAMES:]
    slopes = []
    for left, right in combinations(range(len(frames)), 2):
        delta = frames[right] - frames[left]
        if delta:
            slopes.append((points[right] - points[left]) / delta)
    return (
        np.median(np.asarray(slopes, dtype=float), axis=0)
        if slopes
        else np.zeros(2, dtype=float)
    )


def main() -> None:
    scales_payload = json.loads(SCALES_PATH.read_text(encoding="utf-8"))
    scale_code_current = all(
        Path(path).exists() and sha256(Path(path)) == recorded
        for path, recorded in scales_payload["code_sha256"].items()
    )
    records: dict[str, Any] = {}
    for view in VIEW_NAMES:
        windows = [
            load_candidate_window(
                CACHE_DIR / f"{view}.candidates.h5", frame_start, frame_end
            )
            for frame_start, frame_end in CONTROL_INTERVALS
        ]
        length_values = np.concatenate(
            [
                np.linalg.norm(window.tail_xy - window.mouth_xy, axis=2).ravel()
                for window in windows
            ]
        )
        independently_estimated_scale = robust_positive_median(length_values)
        frozen_scale = float(scales_payload["body_length_px"][view])
        displacement_px: list[float] = []
        prediction_error_px: list[float] = []
        clean_frames = 0
        total_frames = 0
        for window in windows:
            history: dict[int, tuple[list[int], list[np.ndarray]]] = {}
            next_identity = 0
            for local, frame in enumerate(
                range(window.frame_start, window.frame_end + 1)
            ):
                total_frames += 1
                finite = np.all(np.isfinite(window.mouth_xy[local]), axis=1)
                scored = (
                    np.isfinite(window.mouth_score[local])
                    & (window.mouth_score[local] >= MINIMUM_MOUTH_SCORE)
                )
                slots = np.flatnonzero(finite & scored)
                points = window.mouth_xy[local, slots]
                if len(points) > 1:
                    pairwise = np.linalg.norm(
                        points[:, None] - points[None, :], axis=2
                    )
                    np.fill_diagonal(pairwise, np.inf)
                    keep = np.min(pairwise, axis=1) >= (
                        MINIMUM_SEPARATION_BODY_LENGTHS * frozen_scale
                    )
                    points = points[keep]
                if len(points) != 3:
                    history = {}
                    continue
                clean_frames += 1
                if not history:
                    for point in points:
                        history[next_identity] = (
                            [int(frame)], [point.copy()]
                        )
                        next_identity += 1
                    continue
                identity_ids = sorted(history)
                if any(
                    frame - history[identity][0][-1] != 1
                    for identity in identity_ids
                ):
                    history = {}
                    continue
                predicted = np.asarray(
                    [
                        history[identity][1][-1]
                        + robust_velocity(*history[identity])
                        for identity in identity_ids
                    ]
                )
                cost = np.linalg.norm(
                    predicted[:, None] - points[None, :], axis=2
                )
                rows, columns = linear_sum_assignment(cost)
                for row, column in zip(rows, columns, strict=True):
                    identity = identity_ids[int(row)]
                    frames, history_points = history[identity]
                    point = points[int(column)]
                    displacement_px.append(
                        float(np.linalg.norm(point - history_points[-1]))
                    )
                    if len(frames) >= 2:
                        prediction_error_px.append(float(cost[row, column]))
                    frames.append(int(frame))
                    history_points.append(point.copy())
                    history[identity] = (
                        frames[-HISTORY_FRAMES:],
                        history_points[-HISTORY_FRAMES:],
                    )
        records[view] = {
            "frozen_body_length_px": frozen_scale,
            "independently_estimated_body_length_px": independently_estimated_scale,
            "body_length_absolute_difference_px": abs(
                frozen_scale - independently_estimated_scale
            ),
            "control_frames": total_frames,
            "clean_frames": clean_frames,
            "clean_frame_fraction": clean_frames / total_frames,
            "displacement_px": distribution(displacement_px),
            "displacement_body_lengths": distribution(
                [value / frozen_scale for value in displacement_px]
            ),
            "constant_velocity_prediction_error_px": distribution(
                prediction_error_px
            ),
            "constant_velocity_prediction_error_body_lengths": distribution(
                [value / frozen_scale for value in prediction_error_px]
            ),
        }
    result = {
        "schema_version": 1,
        "scope": "training length-matched temporal controls only",
        "control_intervals_inclusive": [
            list(interval) for interval in CONTROL_INTERVALS
        ],
        "minimum_separation_body_lengths": MINIMUM_SEPARATION_BODY_LENGTHS,
        "minimum_mouth_score": MINIMUM_MOUTH_SCORE,
        "history_frames": HISTORY_FRAMES,
        "frozen_scales_path": str(SCALES_PATH),
        "frozen_scales_sha256": sha256(SCALES_PATH),
        "frozen_scale_recorded_code_hashes_current": scale_code_current,
        "cache_sha256": {
            view: sha256(CACHE_DIR / f"{view}.candidates.h5")
            for view in VIEW_NAMES
        },
        "views": records,
        "limitations": [
            "Clean-frame matching is a motion proxy, not biological ground truth.",
            "Length-matched temporal controls are not behavior- or risk-matched.",
            "Tail is used only to audit the frozen normalization scale.",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
