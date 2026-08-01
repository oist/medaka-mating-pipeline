"""Independent purity-first scoring of the 18-setting A4 cutoff sweep."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.reconstruction_3d.candidate_cache import (
    VIEW_NAMES,
    load_candidate_window,
)
from scripts.reconstruction_3d.conservative_mouth_tracklets import (
    ConservativeTrackletConfig,
    build_reciprocal_tracklets,
)


ROOT = Path("report/20260731/round8_mouth_tracking_ablation")
CACHE_DIR = ROOT / "candidate_cache_strict_mouth_unfiltered_nodescores"
SCALES_PATH = ROOT / "frozen_body_length_scales.json"
OUTPUT_ROOT = ROOT / "quality_review"
ANNOTATIONS = Path(
    "annotations/fish_sex_gt/20251225_K_1_2_FCMC/female_sex_gt.csv"
)
WINDOWS = {
    "reviewed_event_1": (72371, 73027),
    "control_for_event_1": (32264, 32920),
    "reviewed_event_3": (174316, 180515),
    "control_for_event_3": (21840, 28039),
}
SEPARATIONS = (0.35, 0.5, 0.65)
AMBIGUITY_MARGINS = (0.05, 0.1)
MAXIMUM_GAPS = (0, 2, 5)
ANNOTATION_GATE_PX = 12.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def config_name(separation: float, margin: float, gap: int) -> str:
    return (
        f"sep{int(round(separation * 100)):02d}_"
        f"margin{int(round(margin * 100)):02d}_gap{gap}"
    )


def training_annotations() -> pd.DataFrame:
    table = pd.read_csv(ANNOTATIONS, low_memory=False)
    table = table[
        table.gt_status.eq("confirmed") & table.sex_label.eq("F")
    ].copy()
    parts = []
    for window_name, (frame_start, frame_end) in WINDOWS.items():
        part = table[table.frame.between(frame_start, frame_end)].copy()
        part["training_window"] = window_name
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["training_window", "frame"], kind="stable"
    )


def main() -> None:
    scales_payload = json.loads(SCALES_PATH.read_text(encoding="utf-8"))
    for path, recorded_hash in scales_payload["code_sha256"].items():
        if not Path(path).exists() or sha256(Path(path)) != recorded_hash:
            raise ValueError(f"frozen scale provenance is stale: {path}")
    cache_hashes = {
        view: sha256(CACHE_DIR / f"{view}.candidates.h5")
        for view in VIEW_NAMES
    }
    if cache_hashes != scales_payload["cache_sha256"]:
        raise ValueError("frozen scales are bound to different candidate caches")
    windows = {
        window_name: {
            view: load_candidate_window(
                CACHE_DIR / f"{view}.candidates.h5",
                frame_start,
                frame_end,
            )
            for view in VIEW_NAMES
        }
        for window_name, (frame_start, frame_end) in WINDOWS.items()
    }
    annotations = training_annotations()
    results: dict[str, Any] = {}
    annotation_records: list[dict[str, Any]] = []
    for separation in SEPARATIONS:
        for margin in AMBIGUITY_MARGINS:
            for maximum_gap in MAXIMUM_GAPS:
                name = config_name(separation, margin, maximum_gap)
                configuration = ConservativeTrackletConfig(
                    separation_body_lengths=separation,
                    assignment_ambiguity_margin_body_lengths=margin,
                    maximum_missing_gap_frames=maximum_gap,
                    uncertainty_growth_body_lengths_per_missing_frame=0.1,
                    base_prediction_gate_body_lengths=1.0,
                    maximum_displacement_body_lengths_per_frame=1.0,
                    motion_history_frames=5,
                    minimum_mouth_score=0.3,
                    minimum_fragment_observations=2,
                )
                per_window_view: dict[str, Any] = {}
                finite_rows = 0
                retained_rows = 0
                fragments = 0
                accepted_bridges = 0
                nonreciprocal = 0
                false_merge_proxy = 0
                false_break_proxy = 0
                unresolved_rows = 0
                fragment_lengths: list[float] = []
                accepted_prediction_errors: list[float] = []
                accepted_assignment_margins: list[float] = []
                accepted_gap_frames: list[float] = []
                female_mapped = 0
                female_eligible = 0
                female_changes = 0
                female_tracklet_counts: Counter[str] = Counter()
                for window_name, view_windows in windows.items():
                    per_window_view[window_name] = {}
                    window_annotations = annotations[
                        annotations.training_window.eq(window_name)
                    ]
                    for view in VIEW_NAMES:
                        build = build_reciprocal_tracklets(
                            view_windows[view],
                            config=configuration,
                            frozen_body_length_px=float(
                                scales_payload["body_length_px"][view]
                            ),
                        )
                        summary = dict(build.summary)
                        disposition_counts = {
                            str(key): int(value)
                            for key, value in build.dispositions.disposition.value_counts().items()
                        }
                        lengths = (
                            build.tracklets.groupby("tracklet_id").size().to_numpy()
                            if len(build.tracklets)
                            else np.empty(0)
                        )
                        accepted = build.bridges[
                            build.bridges.reciprocal_accepted.astype(bool)
                        ]
                        for prefix in ("forward", "backward"):
                            accepted_prediction_errors.extend(
                                pd.to_numeric(
                                    accepted[
                                        f"{prefix}_prediction_error_body_lengths"
                                    ],
                                    errors="coerce",
                                ).tolist()
                            )
                            accepted_assignment_margins.extend(
                                pd.to_numeric(
                                    accepted[
                                        f"{prefix}_assignment_margin_body_lengths"
                                    ],
                                    errors="coerce",
                                ).tolist()
                            )
                        accepted_gap_frames.extend(
                            pd.to_numeric(
                                accepted.gap_frames, errors="coerce"
                            ).tolist()
                        )
                        finite_rows += int(summary["finite_candidate_rows"])
                        retained_rows += int(summary["retained_candidate_rows"])
                        fragments += int(summary["fragment_count"])
                        accepted_bridges += int(
                            summary["reciprocal_accepted_bridges"]
                        )
                        nonreciprocal += int(
                            summary["nonreciprocal_bridge_proposals"]
                        )
                        false_merge_proxy += int(summary["false_merge_proxy"])
                        false_break_proxy += int(summary["false_break_proxy"])
                        unresolved_rows += int(
                            disposition_counts.get(
                                "mouth_proximity_ambiguous", 0
                            )
                            + disposition_counts.get("short_or_unlinked", 0)
                        )
                        fragment_lengths.extend(lengths.tolist())
                        per_window_view[window_name][view] = {
                            "summary": summary,
                            "disposition_counts": disposition_counts,
                            "fragment_length": distribution(lengths.tolist()),
                            "accepted_prediction_error_body_lengths": distribution(
                                [
                                    *pd.to_numeric(
                                        accepted.forward_prediction_error_body_lengths,
                                        errors="coerce",
                                    ).tolist(),
                                    *pd.to_numeric(
                                        accepted.backward_prediction_error_body_lengths,
                                        errors="coerce",
                                    ).tolist(),
                                ]
                            ),
                            "accepted_assignment_margin_body_lengths": distribution(
                                [
                                    *pd.to_numeric(
                                        accepted.forward_assignment_margin_body_lengths,
                                        errors="coerce",
                                    ).tolist(),
                                    *pd.to_numeric(
                                        accepted.backward_assignment_margin_body_lengths,
                                        errors="coerce",
                                    ).tolist(),
                                ]
                            ),
                        }
                        if view == "top_cam":
                            continue
                        prefix = "side1" if view == "side1_cam" else "side2"
                        observed_keys: list[str] = []
                        window = view_windows[view]
                        for annotation in window_annotations.itertuples(index=False):
                            female_eligible += 1
                            local = int(annotation.frame - window.frame_start)
                            target = np.asarray(
                                [
                                    getattr(annotation, f"{prefix}_mouth_x"),
                                    getattr(annotation, f"{prefix}_mouth_y"),
                                ],
                                dtype=float,
                            )
                            points = window.mouth_xy[local]
                            distances = np.linalg.norm(
                                points - target[None, :], axis=1
                            )
                            slot = (
                                int(np.nanargmin(distances))
                                if np.isfinite(distances).any()
                                else -1
                            )
                            distance = (
                                float(distances[slot]) if slot >= 0 else np.nan
                            )
                            tracklet_id = (
                                int(build.tracklet_ids[local, slot])
                                if slot >= 0 and distance <= ANNOTATION_GATE_PX
                                else -1
                            )
                            key = (
                                f"{window_name}:{view}:{tracklet_id}"
                                if tracklet_id >= 0
                                else ""
                            )
                            if key:
                                female_mapped += 1
                                observed_keys.append(key)
                                female_tracklet_counts[key] += 1
                            annotation_records.append(
                                {
                                    "config": name,
                                    "training_window": window_name,
                                    "view": view,
                                    "annotation_id": str(
                                        annotation.annotation_id
                                    ),
                                    "frame": int(annotation.frame),
                                    "nearest_candidate_distance_px": distance,
                                    "retained_tracklet_key": key,
                                }
                            )
                        female_changes += int(
                            np.sum(
                                np.asarray(observed_keys[1:], dtype=object)
                                != np.asarray(observed_keys[:-1], dtype=object)
                            )
                            if len(observed_keys) > 1
                            else 0
                        )
                false_merge_rate = (
                    false_merge_proxy / accepted_bridges
                    if accepted_bridges
                    else 0.0
                )
                false_break_rate = (
                    false_break_proxy
                    / (accepted_bridges + false_break_proxy)
                    if accepted_bridges + false_break_proxy
                    else 0.0
                )
                results[name] = {
                    "configuration": {
                        "separation_body_lengths": separation,
                        "assignment_ambiguity_margin_body_lengths": margin,
                        "maximum_missing_gap_frames": maximum_gap,
                        "uncertainty_growth_body_lengths_per_missing_frame": 0.1,
                        "forward_backward_reciprocal_required": True,
                    },
                    "pooled": {
                        "finite_candidate_rows": finite_rows,
                        "retained_candidate_rows": retained_rows,
                        "retained_coverage": retained_rows / finite_rows,
                        "unresolved_rows": unresolved_rows,
                        "unresolved_fraction": unresolved_rows / finite_rows,
                        "fragment_count": fragments,
                        "fragment_length": distribution(fragment_lengths),
                        "reciprocal_accepted_bridges": accepted_bridges,
                        "nonreciprocal_bridge_proposals": nonreciprocal,
                        "false_merge_proxy_count": false_merge_proxy,
                        "false_merge_proxy_rate_per_accepted_bridge": false_merge_rate,
                        "false_break_proxy_count": false_break_proxy,
                        "false_break_proxy_rate": false_break_rate,
                        "accepted_prediction_error_body_lengths": distribution(
                            accepted_prediction_errors
                        ),
                        "accepted_assignment_margin_body_lengths": distribution(
                            accepted_assignment_margins
                        ),
                        "accepted_gap_frames": distribution(
                            accepted_gap_frames
                        ),
                        "female_annotation_view_rows": female_eligible,
                        "female_annotation_retained_rows": female_mapped,
                        "female_annotation_retained_fraction": (
                            female_mapped / female_eligible
                            if female_eligible
                            else 0.0
                        ),
                        "female_observed_tracklet_changes": female_changes,
                        "female_tracklet_repeat_observations": int(
                            sum(
                                count
                                for count in female_tracklet_counts.values()
                                if count > 1
                            )
                        ),
                    },
                    "windows": per_window_view,
                }

    ranking = sorted(
        results,
        key=lambda name: (
            results[name]["pooled"][
                "false_merge_proxy_rate_per_accepted_bridge"
            ],
            results[name]["pooled"]["false_merge_proxy_count"],
            results[name]["pooled"]["female_observed_tracklet_changes"],
            results[name]["pooled"]["false_break_proxy_rate"],
            -results[name]["pooled"]["female_annotation_retained_fraction"],
            -results[name]["pooled"]["retained_coverage"],
            -results[name]["configuration"]["separation_body_lengths"],
            -results[name]["configuration"][
                "assignment_ambiguity_margin_body_lengths"
            ],
            results[name]["configuration"]["maximum_missing_gap_frames"],
            name,
        ),
    )
    output = {
        "schema_version": 1,
        "scope": (
            "training reviewed events and length-matched controls only; no "
            "validation, held-out, or critical method output read"
        ),
        "configuration_count": len(results),
        "frozen_scales_sha256": sha256(SCALES_PATH),
        "cache_sha256": cache_hashes,
        "code_sha256": {
            "conservative_mouth_tracklets": sha256(
                Path(
                    "scripts/reconstruction_3d/conservative_mouth_tracklets.py"
                )
            ),
            "candidate_cache": sha256(
                Path("scripts/reconstruction_3d/candidate_cache.py")
            ),
        },
        "selection_rule": (
            "purity-first lexicographic ranking: false-merge proxy rate/count, "
            "female observed tracklet changes, false-break proxy rate, female "
            "retention, overall retention, then conservative deterministic "
            "tie-breaks"
        ),
        "ranking": ranking,
        "selected": ranking[0],
        "results": results,
        "proxy_limits": [
            "False-merge and false-break fields are motion/reciprocity proxies, not labeled biological errors.",
            "Female annotations are sparse and cannot expose male-to-male swaps.",
            "A lower proxy count can result from abstention; retained coverage and unresolved fraction must be read alongside it.",
            "Temporal controls are length-matched, not behavior- or risk-matched.",
        ],
    }
    (OUTPUT_ROOT / "training_tracklet_cutoff_sweep_audit.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "config": name,
                **results[name]["configuration"],
                **{
                    key: value
                    for key, value in results[name]["pooled"].items()
                    if not isinstance(value, dict)
                },
                **{
                    f"fragment_length_{key}": value
                    for key, value in results[name]["pooled"][
                        "fragment_length"
                    ].items()
                },
                **{
                    f"accepted_prediction_error_{key}": value
                    for key, value in results[name]["pooled"][
                        "accepted_prediction_error_body_lengths"
                    ].items()
                },
            }
            for name in ranking
        ]
    ).to_csv(
        OUTPUT_ROOT / "training_tracklet_cutoff_sweep_summary.csv",
        index=False,
    )
    pd.DataFrame(annotation_records).to_csv(
        OUTPUT_ROOT / "training_tracklet_cutoff_female_assignments.csv",
        index=False,
    )
    print(
        json.dumps(
            {
                "configuration_count": len(results),
                "selected": ranking[0],
                "top_5": ranking[:5],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
