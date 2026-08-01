"""Authenticate and score the final 72-run A4 training cutoff sweep."""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.reconstruction_3d.calibration import Calibration


ROOT = Path("report/20260731/round8_mouth_tracking_ablation")
SELF_PATH = Path(__file__)
SWEEP_ROOT = ROOT / "training_a4_cutoff_sweep"
COMPLETION_PATH = SWEEP_ROOT / "SWEEP_COMPLETED.json"
OUTPUT_ROOT = ROOT / "quality_review"
SCALES_PATH = ROOT / "frozen_body_length_scales.json"
RUNNER_PATH = Path(
    "scripts/reconstruction_3d/mouth_tracklet_ablation.py"
)
BASE_SCORER_PATH = OUTPUT_ROOT / "score_training_motion_sweep.py"
CALIBRATION_DIR = Path("calibration_results/fixed_refractive_192x192x186")
WINDOWS = {
    "reviewed_event_1": (72371, 73027),
    "control_for_event_1": (32264, 32920),
    "reviewed_event_3": (174316, 180515),
    "control_for_event_3": (21840, 28039),
}
CONFIGS = {
    f"s{int(separation * 100):03d}_m{int(margin * 100):03d}_g{gap}": {
        "separation_body_lengths": separation,
        "assignment_ambiguity_margin_body_lengths": margin,
        "maximum_missing_gap_frames": gap,
    }
    for separation in (0.35, 0.5, 0.65)
    for margin in (0.05, 0.1)
    for gap in (0, 2, 5)
}
VIEW_NAMES = ("top_cam", "side1_cam", "side2_cam")
SCIENTIFIC_STATUSES = {
    "completed",
    "insufficient_three_path_solution",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def distribution(values: list[float] | np.ndarray) -> dict[str, Any]:
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


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_base_scorer():
    spec = importlib.util.spec_from_file_location(
        "independent_a4_base_scorer", BASE_SCORER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the independent base scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticate_outputs(
    summary: dict[str, Any],
    folder: Path,
    *,
    frame_start: int,
    frame_end: int,
) -> dict[str, dict[str, str]]:
    authenticated: dict[str, dict[str, str]] = {}
    resolved_folder = folder.resolve()
    for name, item in summary["outputs"].items():
        path = Path(item["path"]).resolve()
        if resolved_folder not in path.parents:
            raise ValueError(f"output escapes run folder: {path}")
        if not path.exists():
            raise FileNotFoundError(f"missing advertised output: {path}")
        actual = sha256(path)
        if actual != item["sha256"]:
            raise ValueError(f"output hash mismatch: {path}")
        authenticated[name] = {
            "path": str(path),
            "sha256": actual,
        }
    required = {
        "admissible_hypotheses",
        "group_observations",
        "groups",
        "paths",
        "links",
        "dispositions",
        "selected",
        "direct",
        "filled",
    }
    required.update(
        f"{view}_{kind}"
        for view in VIEW_NAMES
        for kind in ("tracklets", "dispositions", "bridges", "summary")
    )
    if missing := sorted(required - set(authenticated)):
        raise ValueError(f"summary does not authenticate required outputs: {missing}")
    block_count = int(summary["hypothesis_blocks"])
    if len(authenticated) != 21 + 2 * block_count:
        raise ValueError(
            "output manifest cardinality differs from 21 final/per-view "
            f"artifacts plus two files per block: {folder}"
        )
    data_blocks: dict[tuple[int, int], str] = {}
    status_blocks: dict[tuple[int, int], str] = {}
    for name in authenticated:
        match = re.fullmatch(
            r"admissible_f(\d{6})_(\d{6})_(data|status)", name
        )
        if match is None:
            continue
        key = (int(match.group(1)), int(match.group(2)))
        target = data_blocks if match.group(3) == "data" else status_blocks
        if key in target:
            raise ValueError(f"duplicate hypothesis block output: {name}")
        target[key] = name
    if (
        len(data_blocks) != block_count
        or len(status_blocks) != block_count
        or set(data_blocks) != set(status_blocks)
    ):
        raise ValueError(f"incomplete hypothesis block data/status pairs: {folder}")
    cursor = frame_start
    block_rows = 0
    for (block_start, block_end) in sorted(data_blocks):
        if block_start != cursor or block_end < block_start or block_end > frame_end:
            raise ValueError(f"non-contiguous hypothesis block range: {folder}")
        data_name = data_blocks[(block_start, block_end)]
        status_name = status_blocks[(block_start, block_end)]
        status_path = Path(authenticated[status_name]["path"])
        block_status = json.loads(status_path.read_text(encoding="utf-8"))
        expected_block_hash = stable_hash(
            {
                "configuration_hash": summary["configuration_hash"],
                "frame_start": block_start,
                "frame_end": block_end,
            }
        )
        if (
            block_status.get("schema_version") != 1
            or block_status.get("status") != "completed"
            or int(block_status.get("frame_start", -1)) != block_start
            or int(block_status.get("frame_end", -1)) != block_end
            or block_status.get("block_configuration_hash")
            != expected_block_hash
            or block_status.get("output_sha256")
            != authenticated[data_name]["sha256"]
        ):
            raise ValueError(f"invalid hypothesis block status: {status_path}")
        data_rows = int(
            pq.ParquetFile(authenticated[data_name]["path"]).metadata.num_rows
        )
        if int(block_status.get("admissible_rows", -1)) != data_rows:
            raise ValueError(f"hypothesis block row-count mismatch: {status_path}")
        block_rows += data_rows
        cursor = block_end + 1
    if cursor != frame_end + 1:
        raise ValueError(f"hypothesis blocks do not cover full window: {folder}")
    aggregate_rows = int(
        pq.ParquetFile(
            authenticated["admissible_hypotheses"]["path"]
        ).metadata.num_rows
    )
    if (
        aggregate_rows != block_rows
        or aggregate_rows != int(summary["admissible_hypothesis_rows"])
    ):
        raise ValueError(f"aggregate/block admissible row mismatch: {folder}")
    return authenticated


def main() -> None:
    if len(CONFIGS) != 18:
        raise AssertionError("expected exactly 18 cutoff configurations")
    if not COMPLETION_PATH.exists():
        raise FileNotFoundError(
            "refusing to score before executor completion sentinel: "
            f"{COMPLETION_PATH}"
        )
    completion = json.loads(COMPLETION_PATH.read_text(encoding="utf-8"))
    if (
        completion.get("schema_version") != 1
        or completion.get("status") != "completed"
        or completion.get("execution_completion_status") != "completed"
        or int(completion.get("expected_configurations", -1)) != 18
        or int(completion.get("expected_windows", -1)) != 4
        or int(completion.get("expected_summaries", -1)) != 72
        or int(completion.get("observed_summaries", -1)) != 72
        or int(completion.get("validated_runs", -1)) != 72
        or completion.get("all_output_artifacts_authenticated") is not True
    ):
        raise ValueError(
            "completion sentinel does not declare a fully authenticated "
            "18-config x 4-window sweep: "
            f"{COMPLETION_PATH}"
        )
    expected_summary_paths = {
        (SWEEP_ROOT / window / config / "a4_summary.json").resolve()
        for window in WINDOWS
        for config in CONFIGS
    }
    discovered_summary_paths = {
        path.resolve()
        for path in SWEEP_ROOT.rglob("a4_summary.json")
    }
    if discovered_summary_paths != expected_summary_paths:
        missing = sorted(str(path) for path in expected_summary_paths - discovered_summary_paths)
        unexpected = sorted(
            str(path) for path in discovered_summary_paths - expected_summary_paths
        )
        raise ValueError(
            "sweep directory is not the exact 4-window x 18-config design: "
            f"missing={missing}, unexpected={unexpected}"
        )
    summary_hash_items = completion.get("run_summary_hashes")
    if not isinstance(summary_hash_items, list) or len(summary_hash_items) != 72:
        raise ValueError("completion sentinel must authenticate exactly 72 summaries")
    recorded_relative_paths = [
        str(item.get("relative_path", "")) for item in summary_hash_items
    ]
    sentinel_summary_hash_list_path_sorted = (
        recorded_relative_paths == sorted(recorded_relative_paths)
    )
    sentinel_summary_paths: set[Path] = set()
    for item in summary_hash_items:
        relative_path = Path(str(item.get("relative_path", "")))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe sentinel summary path: {relative_path}")
        summary_path = (SWEEP_ROOT / relative_path).resolve()
        if summary_path not in expected_summary_paths:
            raise ValueError(f"unexpected sentinel summary path: {relative_path}")
        if sha256(summary_path) != item.get("sha256"):
            raise ValueError(f"sentinel summary hash mismatch: {relative_path}")
        sentinel_summary_paths.add(summary_path)
    if sentinel_summary_paths != expected_summary_paths:
        raise ValueError("completion sentinel summary path set is incomplete")
    scorer = load_base_scorer()
    annotations = scorer.load_training_annotations()
    calibration = Calibration(CALIBRATION_DIR, backend="refractive")
    if calibration.backend_name != "refractive" or calibration.score_unit != "mm^2":
        raise ValueError("A4 scorer requires refractive mm^2 calibration")
    scales_hash = sha256(SCALES_PATH)
    runner_hash = sha256(RUNNER_PATH)
    if completion.get("frozen_scales_sha256") != scales_hash:
        raise ValueError("completion sentinel has stale frozen-scale hash")
    if completion.get("runner_sha256") != runner_hash:
        raise ValueError("completion sentinel has stale A4-runner hash")
    records: dict[str, Any] = {}
    assignment_parts: list[pd.DataFrame] = []
    total_authenticated_summaries = 0
    observed_status_counts: Counter[str] = Counter()

    for config_name, expected_config in CONFIGS.items():
        normalized_config_hashes: set[str] = set()
        code_manifest_hashes: set[str] = set()
        window_records: dict[str, Any] = {}
        pooled: dict[str, Any] = {
            "frames": 0,
            "rows": 0,
            "all_three_direct_frames": 0,
            "unavailable_rows": 0,
            "resolved_rows": 0,
            "selected_observations": 0,
            "eligible_groups": 0,
            "selected_groups": 0,
            "unassigned_eligible_groups": 0,
            "rejected_count_mismatch_groups": 0,
            "fragment_count": 0,
            "finite_candidate_rows": 0,
            "retained_candidate_rows": 0,
            "reciprocal_accepted_bridges": 0,
            "false_merge_proxy_count": 0,
            "false_break_proxy_count": 0,
            "female_eligible": 0,
            "female_mapped": 0,
            "female_correct": 0,
            "female_changes": 0,
            "wall_seconds": 0.0,
            "peak_rss_bytes_max": 0,
        }
        pooled_jumps: list[float] = []
        pooled_residuals: list[float] = []
        selected_group_residuals: list[float] = []
        selected_group_mads: list[float] = []
        selected_group_confidence: list[float] = []
        selected_group_support: list[float] = []
        fragment_lengths: list[float] = []
        accepted_prediction_errors: list[float] = []
        accepted_assignment_margins: list[float] = []

        for window_name, (frame_start, frame_end) in WINDOWS.items():
            folder = SWEEP_ROOT / window_name / config_name
            summary_path = folder / "a4_summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"missing A4 summary: {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            method_status = str(summary.get("status", ""))
            path_status = str(summary.get("path_summary", {}).get("status", ""))
            if (
                method_status not in SCIENTIFIC_STATUSES
                or path_status != method_status
            ):
                raise ValueError(
                    f"incomplete/crashed/unknown A4 method run: "
                    f"{window_name}/{config_name} status={method_status}, "
                    f"path_status={path_status}"
                )
            observed_status_counts[method_status] += 1
            configuration = summary["configuration"]
            if stable_hash(configuration) != summary["configuration_hash"]:
                raise ValueError(f"configuration hash mismatch: {summary_path}")
            if configuration["frozen_body_scales_sha256"] != scales_hash:
                raise ValueError(f"stale scale binding: {summary_path}")
            if (
                int(configuration["frame_start"]) != frame_start
                or int(configuration["frame_end"]) != frame_end
                or configuration["anchor"] != "strict_finite_mouth"
                or bool(configuration["tail_identity_features"])
                or int(configuration["num_candidates"]) != 6
                or int(configuration["num_individuals"]) != 3
                or list(configuration["xyz_bounds"]) != [-20.0, 220.0]
                or configuration["calibration"]["resolved_backend"] != "refractive"
                or configuration["calibration"]["score_unit"] != "mm^2"
            ):
                raise ValueError(f"controlled setting mismatch: {summary_path}")
            calibration_hashes = configuration["calibration_file_sha256"]
            if (
                configuration["calibration"]["file_sha256"] != calibration_hashes
                or summary["calibration"]["file_sha256"] != calibration_hashes
                or any(
                    not (CALIBRATION_DIR / filename).exists()
                    or sha256(CALIBRATION_DIR / filename) != recorded_hash
                    for filename, recorded_hash in calibration_hashes.items()
                )
            ):
                raise ValueError(f"stale calibration-file binding: {summary_path}")
            tracklet_config = configuration["tracklet_config"]
            for key, value in expected_config.items():
                if not np.isclose(float(tracklet_config[key]), float(value)):
                    raise ValueError(
                        f"config name/setting mismatch: {summary_path}, {key}"
                    )
            if (
                float(tracklet_config["uncertainty_growth_body_lengths_per_missing_frame"])
                != 0.1
                or int(tracklet_config["motion_history_frames"]) != 5
                or float(tracklet_config["minimum_mouth_score"]) != 0.3
                or int(tracklet_config["minimum_fragment_observations"]) != 2
                or not bool(
                    configuration["group_config"][
                        "reject_retained_count_mismatch"
                    ]
                )
                or int(configuration["path_config"]["number_of_paths"]) != 3
                or float(configuration["path_config"]["max_jump_mm"]) != 8.0
                or float(
                    configuration["path_config"]["max_jump_hard_cap_mm"]
                )
                != 80.0
                or int(
                    configuration["path_config"]["maximum_link_gap_frames"]
                )
                != 100
            ):
                raise ValueError(f"non-frozen A4 setting: {summary_path}")
            if summary["code_sha256"] != configuration["code_sha256"]:
                raise ValueError(f"duplicated code manifest differs: {summary_path}")
            if summary["code_sha256"].get(str(RUNNER_PATH).replace("/", "\\")) != runner_hash:
                raise ValueError(f"stale A4 runner binding: {summary_path}")
            for code_path, recorded_hash in summary["code_sha256"].items():
                if not Path(code_path).exists() or sha256(Path(code_path)) != recorded_hash:
                    raise ValueError(f"stale code binding: {summary_path}: {code_path}")
            authenticated = authenticate_outputs(
                summary,
                folder,
                frame_start=frame_start,
                frame_end=frame_end,
            )
            total_authenticated_summaries += 1

            normalized = dict(configuration)
            normalized.pop("frame_start")
            normalized.pop("frame_end")
            normalized_config_hashes.add(stable_hash(normalized))
            code_manifest_hashes.add(stable_hash(summary["code_sha256"]))

            direct = pd.read_csv(authenticated["direct"]["path"], low_memory=False)
            expected_rows = 3 * (frame_end - frame_start + 1)
            if (
                len(direct) != expected_rows
                or direct.duplicated(["frame", "fish_id"]).any()
                or set(direct.frame.astype(int))
                != set(range(frame_start, frame_end + 1))
                or set(direct.fish_id.astype(str))
                != {"PATH_1", "PATH_2", "PATH_3"}
                or not direct.groupby("frame").size().eq(3).all()
            ):
                raise ValueError(f"invalid direct grid: {summary_path}")
            track_qc = scorer.track_metrics(direct, frame_start, frame_end)
            jump_values = track_qc.pop("_jump_values")
            residual_values = track_qc.pop("_residual_values")
            pooled_jumps.extend(jump_values)
            pooled_residuals.extend(residual_values)
            pooled["frames"] += frame_end - frame_start + 1
            pooled["rows"] += len(direct)
            pooled["all_three_direct_frames"] += track_qc[
                "all_three_direct_frames"
            ]
            pooled["unavailable_rows"] += track_qc["unavailable_rows"]
            pooled["resolved_rows"] += len(direct) - track_qc["unavailable_rows"]

            window_annotations = annotations[
                annotations.training_window.eq(window_name)
            ]
            assignments = scorer.map_female_rows(
                window_annotations,
                direct,
                calibration,
                config_name=config_name,
                window_name=window_name,
            )
            identity = scorer.summarize_assignments(assignments)
            assignments["best_fixed_identity"] = identity[
                "best_fixed_identity"
            ]
            assignment_parts.append(assignments)
            pooled["female_eligible"] += identity["eligible_annotations"]
            pooled["female_mapped"] += identity["mapped_annotations"]
            pooled["female_correct"] += identity["best_fixed_correct"]
            pooled["female_changes"] += identity["observed_identity_changes"]

            groups = read_csv_safe(Path(authenticated["groups"]["path"]))
            dispositions = read_csv_safe(
                Path(authenticated["dispositions"]["path"])
            )
            paths = read_csv_safe(Path(authenticated["paths"]["path"]))
            links = read_csv_safe(Path(authenticated["links"]["path"]))
            selected = pd.read_parquet(authenticated["selected"]["path"])
            eligible_groups = (
                groups[groups.eligible.fillna(False).astype(bool)]
                if len(groups)
                else groups
            )
            selected_ids = set(
                dispositions.loc[
                    dispositions.selected.fillna(False).astype(bool),
                    "group_id",
                ].astype(str)
            ) if len(dispositions) else set()
            selected_groups = groups[
                groups.group_id.astype(str).isin(selected_ids)
            ] if len(groups) else groups
            if len(selected_groups) and (
                not selected_groups.eligible.fillna(False).astype(bool).all()
                or (
                    pd.to_numeric(
                        selected_groups[
                            "retained_tracklet_count_mismatch_fraction"
                        ],
                        errors="coerce",
                    )
                    > 0
                ).any()
            ):
                raise ValueError(f"selected impure/ineligible A4 group: {summary_path}")
            if len(selected) != int(
                summary["path_summary"].get("selected_observations", 0)
            ):
                raise ValueError(f"selected observation count mismatch: {summary_path}")
            if len(paths) and paths.path_id.nunique() != 3:
                raise ValueError(f"A4 does not contain three paths: {summary_path}")
            if method_status == "insufficient_three_path_solution":
                finite_direct_rows = int(
                    np.isfinite(
                        direct[["X", "Y", "Z"]].to_numpy(dtype=float)
                    ).all(axis=1).sum()
                )
                if (
                    len(selected)
                    or len(paths)
                    or len(links)
                    or finite_direct_rows
                    or not direct.identity_status.astype(str).eq(
                        "unassigned_tracklet_path"
                    ).all()
                ):
                    raise ValueError(
                        "scientific abstention contains a partial/implicit "
                        f"three-path assignment: {summary_path}"
                    )

            pooled["selected_observations"] += len(selected)
            pooled["eligible_groups"] += len(eligible_groups)
            pooled["selected_groups"] += len(selected_groups)
            pooled["unassigned_eligible_groups"] += (
                len(eligible_groups) - len(selected_groups)
            )
            pooled["rejected_count_mismatch_groups"] += int(
                groups.rejection_reason.fillna("").astype(str).str.contains(
                    "retained_tracklet_count_mismatch", regex=False
                ).sum()
                if len(groups)
                else 0
            )
            if len(selected_groups):
                selected_group_residuals.extend(
                    pd.to_numeric(
                        selected_groups.median_ray_residual_mm2,
                        errors="coerce",
                    ).tolist()
                )
                selected_group_mads.extend(
                    pd.to_numeric(
                        selected_groups.mad_ray_residual_mm2,
                        errors="coerce",
                    ).tolist()
                )
                selected_group_confidence.extend(
                    pd.to_numeric(
                        selected_groups.median_mouth_confidence,
                        errors="coerce",
                    ).tolist()
                )
                selected_group_support.extend(
                    pd.to_numeric(
                        selected_groups.support_frames, errors="coerce"
                    ).tolist()
                )

            per_view: dict[str, Any] = {}
            for view in VIEW_NAMES:
                view_summary = json.loads(
                    Path(authenticated[f"{view}_summary"]["path"]).read_text(
                        encoding="utf-8"
                    )
                )
                if stable_hash(view_summary) != stable_hash(
                    summary["per_view_tracklet_summary"][view]
                ):
                    raise ValueError(f"embedded per-view summary differs: {summary_path}")
                tracklets = pd.read_parquet(
                    authenticated[f"{view}_tracklets"]["path"]
                )
                view_dispositions = pd.read_parquet(
                    authenticated[f"{view}_dispositions"]["path"]
                )
                bridges = pd.read_parquet(
                    authenticated[f"{view}_bridges"]["path"]
                )
                lengths = (
                    tracklets.groupby("tracklet_id").size().to_numpy(dtype=int)
                    if len(tracklets)
                    else np.empty(0, dtype=int)
                )
                accepted = bridges[
                    bridges.reciprocal_accepted.fillna(False).astype(bool)
                ] if len(bridges) else bridges
                false_merge_actual = int(
                    (
                        accepted.reciprocal_accepted.fillna(False).astype(bool)
                        & (
                            (
                                pd.to_numeric(
                                    accepted.forward_prediction_error_body_lengths,
                                    errors="coerce",
                                )
                                > float(tracklet_config["base_prediction_gate_body_lengths"])
                            )
                            | (
                                pd.to_numeric(
                                    accepted.backward_prediction_error_body_lengths,
                                    errors="coerce",
                                )
                                > float(tracklet_config["base_prediction_gate_body_lengths"])
                            )
                        )
                    ).sum()
                    if len(accepted)
                    else 0
                )
                false_break_actual = int(
                    (
                        bridges.forward_proposed.fillna(False).astype(bool)
                        ^ bridges.backward_proposed.fillna(False).astype(bool)
                    ).sum()
                    if len(bridges)
                    else 0
                )
                if (
                    len(tracklets) != int(view_summary["retained_candidate_rows"])
                    or len(lengths) != int(view_summary["fragment_count"])
                    or len(accepted)
                    != int(view_summary["reciprocal_accepted_bridges"])
                    or false_merge_actual != int(view_summary["false_merge_proxy"])
                    or false_break_actual != int(view_summary["false_break_proxy"])
                ):
                    raise ValueError(f"per-view metric mismatch: {summary_path}/{view}")
                pooled["fragment_count"] += len(lengths)
                pooled["finite_candidate_rows"] += len(view_dispositions)
                pooled["retained_candidate_rows"] += len(tracklets)
                pooled["reciprocal_accepted_bridges"] += len(accepted)
                pooled["false_merge_proxy_count"] += false_merge_actual
                pooled["false_break_proxy_count"] += false_break_actual
                fragment_lengths.extend(lengths.tolist())
                for prefix in ("forward", "backward"):
                    accepted_prediction_errors.extend(
                        pd.to_numeric(
                            accepted[
                                f"{prefix}_prediction_error_body_lengths"
                            ],
                            errors="coerce",
                        ).tolist()
                        if len(accepted)
                        else []
                    )
                    accepted_assignment_margins.extend(
                        pd.to_numeric(
                            accepted[
                                f"{prefix}_assignment_margin_body_lengths"
                            ],
                            errors="coerce",
                        ).tolist()
                        if len(accepted)
                        else []
                    )
                per_view[view] = {
                    "summary": view_summary,
                    "fragment_length": distribution(lengths),
                    "disposition_counts": {
                        str(key): int(value)
                        for key, value in view_dispositions.disposition.value_counts().items()
                    },
                }

            pooled["wall_seconds"] += float(summary["wall_seconds"])
            pooled["peak_rss_bytes_max"] = max(
                pooled["peak_rss_bytes_max"], int(summary["peak_rss_bytes"])
            )
            window_records[window_name] = {
                "method_status": method_status,
                "summary_path": str(summary_path),
                "summary_sha256": sha256(summary_path),
                "configuration_hash": summary["configuration_hash"],
                "advertised_outputs_authenticated": len(authenticated),
                "track_qc": track_qc,
                "female_identity": identity,
                "group_qc": {
                    "groups": int(len(groups)),
                    "eligible_groups": int(len(eligible_groups)),
                    "selected_groups": int(len(selected_groups)),
                    "unassigned_eligible_groups": int(
                        len(eligible_groups) - len(selected_groups)
                    ),
                    "selected_observations": int(len(selected)),
                    "path_rows": int(len(paths)),
                    "link_rows": int(len(links)),
                    "all_views_break_links": int(
                        links.all_views_break.fillna(False).astype(bool).sum()
                        if len(links) and "all_views_break" in links
                        else 0
                    ),
                    "segment_restart_links": int(
                        links.segment_restart.fillna(False).astype(bool).sum()
                        if len(links) and "segment_restart" in links
                        else 0
                    ),
                },
                "per_view": per_view,
                "wall_seconds": float(summary["wall_seconds"]),
                "peak_rss_bytes": int(summary["peak_rss_bytes"]),
            }

        pooled["female_full_eligible_accuracy"] = (
            pooled["female_correct"] / pooled["female_eligible"]
            if pooled["female_eligible"]
            else 0.0
        )
        pooled["female_selective_mapped_accuracy"] = (
            pooled["female_correct"] / pooled["female_mapped"]
            if pooled["female_mapped"]
            else 0.0
        )
        pooled["all_three_direct_fraction"] = (
            pooled["all_three_direct_frames"] / pooled["frames"]
        )
        pooled["unavailable_row_fraction"] = (
            pooled["unavailable_rows"] / pooled["rows"]
        )
        pooled["retained_candidate_coverage"] = (
            pooled["retained_candidate_rows"] / pooled["finite_candidate_rows"]
        )
        pooled["eligible_group_selected_fraction"] = (
            pooled["selected_groups"] / pooled["eligible_groups"]
            if pooled["eligible_groups"]
            else 0.0
        )
        pooled["false_merge_proxy_rate"] = (
            pooled["false_merge_proxy_count"]
            / pooled["reciprocal_accepted_bridges"]
            if pooled["reciprocal_accepted_bridges"]
            else 0.0
        )
        pooled["false_break_proxy_rate"] = (
            pooled["false_break_proxy_count"]
            / (
                pooled["reciprocal_accepted_bridges"]
                + pooled["false_break_proxy_count"]
            )
            if (
                pooled["reciprocal_accepted_bridges"]
                + pooled["false_break_proxy_count"]
            )
            else 0.0
        )
        jump_array = np.asarray(pooled_jumps, dtype=float)
        pooled["jump_mm"] = distribution(jump_array)
        pooled["jump_ge_7_5mm_count"] = int((jump_array >= 7.5).sum())
        pooled["jump_gt_8mm_count"] = int((jump_array > 8.0).sum())
        pooled["direct_residual_mm2"] = distribution(pooled_residuals)
        pooled["selected_group_median_residual_mm2"] = distribution(
            selected_group_residuals
        )
        pooled["selected_group_residual_mad_mm2"] = distribution(
            selected_group_mads
        )
        pooled["selected_group_median_mouth_confidence"] = distribution(
            selected_group_confidence
        )
        pooled["selected_group_support_frames"] = distribution(
            selected_group_support
        )
        pooled["fragment_length"] = distribution(fragment_lengths)
        pooled["accepted_prediction_error_body_lengths"] = distribution(
            accepted_prediction_errors
        )
        pooled["accepted_assignment_margin_body_lengths"] = distribution(
            accepted_assignment_margins
        )
        records[config_name] = {
            "configuration": expected_config,
            "cross_window_normalized_configuration_equal": (
                len(normalized_config_hashes) == 1
            ),
            "cross_window_code_manifest_equal": len(code_manifest_hashes) == 1,
            "normalized_configuration_sha256": sorted(normalized_config_hashes),
            "code_manifest_sha256": sorted(code_manifest_hashes),
            "windows": window_records,
            "pooled": pooled,
        }

    ranking = sorted(
        CONFIGS,
        key=lambda name: (
            -records[name]["pooled"]["female_full_eligible_accuracy"],
            records[name]["pooled"]["female_changes"],
            records[name]["pooled"]["false_merge_proxy_rate"],
            records[name]["pooled"]["false_merge_proxy_count"],
            records[name]["pooled"]["jump_ge_7_5mm_count"],
            -records[name]["pooled"]["all_three_direct_fraction"],
            records[name]["pooled"]["false_break_proxy_rate"],
            records[name]["pooled"]["unavailable_row_fraction"],
            -records[name]["pooled"]["eligible_group_selected_fraction"],
            -records[name]["configuration"][
                "assignment_ambiguity_margin_body_lengths"
            ],
            records[name]["configuration"]["maximum_missing_gap_frames"],
            -records[name]["configuration"]["separation_body_lengths"],
            name,
        ),
    )
    result = {
        "schema_version": 1,
        "status": "training_only_a4_cutoff_selected",
        "scope": (
            "authenticated A4 method-level training runs only; validation, "
            "held-out, and critical method outputs were not read"
        ),
        "expected_run_count": 72,
        "authenticated_summary_count": total_authenticated_summaries,
        "scientific_status_counts": dict(sorted(observed_status_counts.items())),
        "selected": ranking[0],
        "ranking": ranking,
        "selection_rule": [
            "maximize full-eligible female identity accuracy",
            "minimize observed female identity changes",
            "minimize reciprocal false-merge proxy rate and count",
            "minimize direct jumps >=7.5 mm",
            "maximize all-three direct path coverage",
            "minimize nonreciprocal false-break proxy and unavailable rows",
            "maximize eligible-group selection coverage",
            "larger ambiguity margin, shorter maximum gap, larger separation",
        ],
        "frozen_scales_sha256": scales_hash,
        "a4_runner_path": str(RUNNER_PATH),
        "a4_runner_sha256": runner_hash,
        "completion_sentinel_path": str(COMPLETION_PATH),
        "completion_sentinel_sha256": sha256(COMPLETION_PATH),
        "completion_sentinel": completion,
        "completion_sentinel_summary_hash_list_path_sorted": (
            sentinel_summary_hash_list_path_sorted
        ),
        "selection_code_sha256": {
            str(SELF_PATH): sha256(SELF_PATH),
            str(BASE_SCORER_PATH): sha256(BASE_SCORER_PATH),
        },
        "training_annotations": {
            "path": str(scorer.ANNOTATIONS),
            "sha256": sha256(Path(scorer.ANNOTATIONS)),
        },
        "records": records,
        "limitations": [
            "Only 15 confirmed-female training annotations are available.",
            "Female-only annotations cannot expose male-to-male swaps.",
            "False-merge and false-break counts are motion/reciprocity proxies, not labeled biological errors.",
            "Lower residual and smoother paths are quality checks, not identity proof.",
            "Temporal controls are length-matched, not behavior- or risk-matched.",
        ],
    }
    if not sentinel_summary_hash_list_path_sorted:
        result["limitations"].append(
            "The completion sentinel's summary-hash list is not path-sorted; "
            "the reviewer validated its exact path set and every hash "
            "independently, so ordering has no scientific effect."
        )
    if total_authenticated_summaries != 72:
        raise ValueError(
            f"authenticated {total_authenticated_summaries}, expected 72"
        )
    sentinel_status_counts = {
        str(key): int(value)
        for key, value in completion.get(
            "scientific_solution_status_counts", {}
        ).items()
    }
    if sentinel_status_counts != dict(sorted(observed_status_counts.items())):
        raise ValueError(
            "completion sentinel status counts differ from authenticated summaries"
        )
    (OUTPUT_ROOT / "training_a4_method_sweep_audit.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "rank": index + 1,
                "config": name,
                **records[name]["configuration"],
                **{
                    key: value
                    for key, value in records[name]["pooled"].items()
                    if not isinstance(value, dict)
                },
                **{
                    f"direct_residual_mm2_{key}": value
                    for key, value in records[name]["pooled"][
                        "direct_residual_mm2"
                    ].items()
                },
                **{
                    f"jump_mm_{key}": value
                    for key, value in records[name]["pooled"][
                        "jump_mm"
                    ].items()
                },
            }
            for index, name in enumerate(ranking)
        ]
    ).to_csv(
        OUTPUT_ROOT / "training_a4_method_sweep_summary.csv",
        index=False,
    )
    pd.concat(assignment_parts, ignore_index=True).to_csv(
        OUTPUT_ROOT / "training_a4_method_female_assignments.csv",
        index=False,
    )
    print(
        json.dumps(
            {
                "authenticated_summary_count": total_authenticated_summaries,
                "selected": ranking[0],
                "top_5": ranking[:5],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
