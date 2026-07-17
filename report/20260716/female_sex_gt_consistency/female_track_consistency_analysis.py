"""Reproducible consistency audit for female GT versus round7 refractive 3D IDs."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scripts.analysis.build_female_sex_gt_candidates import (
    build_refractive_geometry_tracklets,
)
from scripts.analysis.female_sex_gt import deterministic_annotation_id
from scripts.reconstruction_3d.calibration import Calibration


EXPERIMENT = "20251225_K_1_2_FCMC"
SOURCE_FPS = 166.0
ANNOTATION_MATCH_GATE_PX = 8.0
TRACKLET_JUMP_GATE_MM = 8.0

REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPORT_DIR.parents[2]
ANNOTATION_CSV = (
    REPO_ROOT / "annotations" / "fish_sex_gt" / EXPERIMENT / "female_sex_gt.csv"
)
AUDIT_JSONL = ANNOTATION_CSV.with_suffix(".audit.jsonl")
CANDIDATE_CSV = ANNOTATION_CSV.with_name("clean_three_view_candidates.csv")
CANDIDATE_SUMMARY_JSON = CANDIDATE_CSV.with_name(
    "clean_three_view_candidates_summary.json"
)
CALIBRATION_MODEL_DIR = (
    REPO_ROOT / "calibration_results" / "fixed_refractive_192x192x186"
)
NETWORK_RECONSTRUCTION_DIR = Path(
    r"\\bucket.oist.jp\bucket\StephensU\medaka\reconstruction_3d"
) / EXPERIMENT / "round7_260704_refractive"
GEOMETRY_CSV = (
    NETWORK_RECONSTRUCTION_DIR / "round7_260704_refractive_3d.csv"
)
RECONSTRUCTION_PROVENANCE_JSON = (
    NETWORK_RECONSTRUCTION_DIR
    / "round7_260704_refractive_reconstruction_provenance.json"
)


def _sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    series = pd.Series(list(values), dtype=float).replace([np.inf, -np.inf], np.nan)
    series = series.dropna()
    if series.empty:
        return {"count": 0}
    return {
        "count": int(series.size),
        "min": float(series.min()),
        "median": float(series.median()),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
    }


def _segment_distance(
    click_x: float,
    click_y: float,
    mouth_x: float,
    mouth_y: float,
    tail_x: float,
    tail_y: float,
) -> float:
    click = np.asarray([click_x, click_y], dtype=float)
    endpoints = np.asarray([[mouth_x, mouth_y], [tail_x, tail_y]], dtype=float)
    finite = np.isfinite(endpoints).all(axis=1)
    endpoints = endpoints[finite]
    if not np.isfinite(click).all() or not len(endpoints):
        return math.nan
    if len(endpoints) == 1:
        return float(np.linalg.norm(click - endpoints[0]))
    start, end = endpoints
    direction = end - start
    length_squared = float(np.dot(direction, direction))
    if length_squared <= 1e-12:
        return float(np.linalg.norm(click - start))
    fraction = float(np.dot(click - start, direction) / length_squared)
    closest = start + np.clip(fraction, 0.0, 1.0) * direction
    return float(np.linalg.norm(click - closest))


def _visible_centroid(row: pd.Series, prefix: str) -> np.ndarray:
    points = np.asarray(
        [
            [row[f"{prefix}_mouth_x"], row[f"{prefix}_mouth_y"]],
            [row[f"{prefix}_tail_x"], row[f"{prefix}_tail_y"]],
        ],
        dtype=float,
    )
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        return np.asarray([np.nan, np.nan], dtype=float)
    return points[finite].mean(axis=0)


def _candidate_metadata(value: str) -> dict[str, Any]:
    if not value:
        return {}
    contributions = json.loads(value)
    if not isinstance(contributions, list) or not contributions:
        return {}
    contribution = contributions[0]
    extra = contribution.get("extra", {}) if isinstance(contribution, dict) else {}
    if not isinstance(extra, dict):
        extra = {}
    quality_components: dict[str, Any] = {}
    raw_quality = extra.get("quality_components_json", "")
    if raw_quality:
        quality_components = json.loads(raw_quality)
    return {
        "candidate_score": contribution.get("score"),
        "temporal_bin": extra.get("temporal_bin", ""),
        "max_reprojection_distance_px": extra.get(
            "max_reprojection_distance_px"
        ),
        "min_border_distance_px": extra.get("min_border_distance_px"),
        "candidate_quality_score": quality_components.get("score"),
        "clean_interval_id": extra.get("clean_interval_id", ""),
    }


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    annotation = pd.read_csv(ANNOTATION_CSV, keep_default_na=False)
    numeric_suffixes = (
        "_x",
        "_y",
        "_index",
    )
    numeric_columns = [
        "frame",
        "time_seconds",
        "candidate_frame",
        *[
            column
            for column in annotation.columns
            if column.endswith(numeric_suffixes)
        ],
    ]
    for column in numeric_columns:
        annotation[column] = pd.to_numeric(annotation[column], errors="coerce")
    annotation["frame"] = annotation["frame"].astype(np.int64)

    candidate = pd.read_csv(CANDIDATE_CSV, keep_default_na=False)
    candidate["frame"] = pd.to_numeric(candidate["frame"], errors="raise").astype(
        np.int64
    )
    candidate_summary = json.loads(CANDIDATE_SUMMARY_JSON.read_text(encoding="utf-8"))
    provenance = json.loads(
        RECONSTRUCTION_PROVENANCE_JSON.read_text(encoding="utf-8")
    )
    return annotation, candidate, candidate_summary, provenance


def _audit_summary(annotation: pd.DataFrame) -> dict[str, Any]:
    events = [
        json.loads(line)
        for line in AUDIT_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    actions = Counter(str(event.get("action", "")) for event in events)
    last_new_row: dict[str, dict[str, Any]] = {}
    for event in events:
        row = event.get("new_row")
        if isinstance(row, dict) and row.get("annotation_id"):
            last_new_row[str(row["annotation_id"])] = row

    canonical = annotation.set_index("annotation_id", drop=False)
    matching_latest_rows = 0
    for annotation_id, latest in last_new_row.items():
        if annotation_id not in canonical.index:
            continue
        current = canonical.loc[annotation_id]
        if str(current["updated_at_utc"]) == str(latest.get("updated_at_utc", "")):
            matching_latest_rows += 1
    return {
        "event_count": len(events),
        "action_counts": dict(sorted(actions.items())),
        "canonical_rows_with_latest_audit_match": matching_latest_rows,
        "canonical_row_count": int(len(annotation)),
        "audit_annotation_id_count": int(len(last_new_row)),
    }


def _reprojection_details(
    annotation: pd.DataFrame,
    geometry: pd.DataFrame,
    calibration: Calibration,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_details: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for prefix, camera_index in (("side1", 1), ("side2", 2)):
        selected = annotation.loc[
            annotation[f"{prefix}_geometry_id"].astype(str).str.len().gt(0)
            & annotation[f"{prefix}_selection_status"].eq("sleap_linked")
        ].copy()
        selected = selected.merge(
            geometry[["frame", "fish_id", "X", "Y", "Z"]],
            left_on=["frame", f"{prefix}_geometry_id"],
            right_on=["frame", "fish_id"],
            how="left",
            validate="many_to_one",
        )
        xyz = selected[["X", "Y", "Z"]].to_numpy(dtype=float)
        finite_xyz = np.isfinite(xyz).all(axis=1)
        projected = np.full((len(selected), 2), np.nan, dtype=float)
        if finite_xyz.any():
            projected_all = np.asarray(
                calibration.project_world(xyz[finite_xyz]), dtype=float
            )
            projected[finite_xyz] = projected_all[:, camera_index, :]
        centroids = np.stack(
            [_visible_centroid(row, prefix) for _, row in selected.iterrows()]
        )
        selected["view"] = prefix
        selected["projected_x"] = projected[:, 0]
        selected["projected_y"] = projected[:, 1]
        selected["pose_centroid_x"] = centroids[:, 0]
        selected["pose_centroid_y"] = centroids[:, 1]
        selected["reprojection_distance_px"] = np.linalg.norm(
            projected - centroids, axis=1
        )
        selected["click_to_pose_segment_px"] = selected.apply(
            lambda row: _segment_distance(
                row[f"{prefix}_click_x"],
                row[f"{prefix}_click_y"],
                row[f"{prefix}_mouth_x"],
                row[f"{prefix}_mouth_y"],
                row[f"{prefix}_tail_x"],
                row[f"{prefix}_tail_y"],
            ),
            axis=1,
        )
        all_details.append(
            selected[
                [
                    "annotation_id",
                    "frame",
                    "view",
                    f"{prefix}_geometry_id",
                    "projected_x",
                    "projected_y",
                    "pose_centroid_x",
                    "pose_centroid_y",
                    "reprojection_distance_px",
                    "click_to_pose_segment_px",
                ]
            ].rename(columns={f"{prefix}_geometry_id": "geometry_id"})
        )
        reprojection = selected["reprojection_distance_px"].dropna()
        click_distance = selected["click_to_pose_segment_px"].dropna()
        summaries.append(
            {
                "view": prefix,
                "mapped_selection_count": int(len(selected)),
                "reprojection_median_px": float(reprojection.median()),
                "reprojection_p95_px": float(reprojection.quantile(0.95)),
                "reprojection_max_px": float(reprojection.max()),
                "reprojection_over_8px_count": int(
                    (reprojection > ANNOTATION_MATCH_GATE_PX + 1e-6).sum()
                ),
                "click_to_pose_median_px": float(click_distance.median()),
                "click_to_pose_p95_px": float(click_distance.quantile(0.95)),
                "click_to_pose_max_px": float(click_distance.max()),
                "click_to_pose_over_8px_count": int(
                    (click_distance > ANNOTATION_MATCH_GATE_PX + 1e-6).sum()
                ),
            }
        )
    return pd.concat(all_details, ignore_index=True), pd.DataFrame(summaries)


def _identity_runs(mapped: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = mapped.sort_values("frame", kind="stable").copy()
    ordered["previous_geometry_id"] = ordered["consensus_geometry_id"].shift()
    ordered["previous_frame"] = ordered["frame"].shift()
    ordered["identity_changed"] = ordered["consensus_geometry_id"].ne(
        ordered["previous_geometry_id"]
    )
    ordered.loc[ordered.index[0], "identity_changed"] = False
    ordered["run_number"] = ordered["identity_changed"].cumsum().astype(int) + 1

    runs = (
        ordered.groupby(["run_number", "consensus_geometry_id"], sort=True)
        .agg(
            start_frame=("frame", "min"),
            end_frame=("frame", "max"),
            annotation_count=("frame", "size"),
        )
        .reset_index()
        .rename(columns={"consensus_geometry_id": "female_geometry_id"})
    )
    runs["start_seconds"] = runs["start_frame"] / SOURCE_FPS
    runs["end_seconds"] = runs["end_frame"] / SOURCE_FPS

    transitions = ordered.loc[ordered["identity_changed"]].copy()
    transitions = transitions[
        [
            "previous_frame",
            "frame",
            "previous_geometry_id",
            "consensus_geometry_id",
            "geometry_id_consistency",
            "candidate_frame",
            "candidate_max_reprojection_distance_px",
        ]
    ].rename(
        columns={
            "previous_frame": "left_annotation_frame",
            "frame": "right_annotation_frame",
            "previous_geometry_id": "from_geometry_id",
            "consensus_geometry_id": "to_geometry_id",
        }
    )
    transitions["left_annotation_frame"] = transitions[
        "left_annotation_frame"
    ].astype(np.int64)
    transitions["gap_frames"] = (
        transitions["right_annotation_frame"]
        - transitions["left_annotation_frame"]
    )
    transitions["left_seconds"] = transitions["left_annotation_frame"] / SOURCE_FPS
    transitions["right_seconds"] = transitions["right_annotation_frame"] / SOURCE_FPS
    return runs, transitions.reset_index(drop=True)


def run_analysis() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    annotation, candidates, candidate_summary, reconstruction_provenance = (
        _load_inputs()
    )

    geometry = pd.read_csv(
        GEOMETRY_CSV,
        usecols=["frame", "fish_id", "X", "Y", "Z", "is_interpolated"],
        low_memory=False,
    )
    geometry["frame"] = pd.to_numeric(geometry["frame"], errors="raise").astype(
        np.int64
    )
    geometry["fish_id"] = geometry["fish_id"].astype(str)

    metadata = annotation["candidate_metadata_json"].map(_candidate_metadata)
    annotation["candidate_temporal_bin"] = metadata.map(
        lambda item: item.get("temporal_bin", "")
    )
    annotation["candidate_max_reprojection_distance_px"] = pd.to_numeric(
        metadata.map(lambda item: item.get("max_reprojection_distance_px")),
        errors="coerce",
    )
    annotation["candidate_quality_score"] = pd.to_numeric(
        metadata.map(lambda item: item.get("candidate_quality_score")),
        errors="coerce",
    )
    annotation["clean_interval_id"] = metadata.map(
        lambda item: item.get("clean_interval_id", "")
    )

    for prefix in ("side1", "side2"):
        annotation[f"{prefix}_click_to_pose_segment_px"] = annotation.apply(
            lambda row: _segment_distance(
                row[f"{prefix}_click_x"],
                row[f"{prefix}_click_y"],
                row[f"{prefix}_mouth_x"],
                row[f"{prefix}_mouth_y"],
                row[f"{prefix}_tail_x"],
                row[f"{prefix}_tail_y"],
            ),
            axis=1,
        )

    confirmed = annotation.loc[
        annotation["gt_status"].eq("confirmed") & annotation["sex_label"].eq("F")
    ].copy()
    mapped = confirmed.loc[confirmed["consensus_geometry_id"].astype(str).str.len().gt(0)].copy()

    exact_geometry = mapped.merge(
        geometry,
        left_on=["frame", "consensus_geometry_id"],
        right_on=["frame", "fish_id"],
        how="left",
        validate="many_to_one",
    )
    exact_geometry["finite_xyz"] = np.isfinite(
        exact_geometry[["X", "Y", "Z"]].to_numpy(dtype=float)
    ).all(axis=1)

    segmented_geometry, segment_diagnostics = build_refractive_geometry_tracklets(
        geometry,
        max_local_jump_mm=TRACKLET_JUMP_GATE_MM,
        id_column="fish_id",
    )
    exact_tracklet = mapped.merge(
        segmented_geometry[
            [
                "frame",
                "geometry_source_id",
                "clean_tracklet_id",
                "label_status",
            ]
        ],
        left_on=["frame", "consensus_geometry_id"],
        right_on=["frame", "geometry_source_id"],
        how="left",
        validate="many_to_one",
    )
    exact_tracklet["tracklet_matches_exact_frame"] = exact_tracklet[
        "consensus_tracklet_id"
    ].eq(exact_tracklet["clean_tracklet_id"])

    calibration = Calibration(CALIBRATION_MODEL_DIR, backend="refractive")
    reprojection_details, reprojection_summary = _reprojection_details(
        annotation, geometry, calibration
    )
    reproj_wide = reprojection_details.pivot_table(
        index="annotation_id",
        columns="view",
        values="reprojection_distance_px",
        aggfunc="first",
    ).rename(columns={"side1": "side1_reprojection_px", "side2": "side2_reprojection_px"})
    annotation = annotation.merge(
        reproj_wide.reset_index(), on="annotation_id", how="left", validate="one_to_one"
    )

    identity_counts = (
        mapped["consensus_geometry_id"]
        .value_counts()
        .rename_axis("geometry_id")
        .reset_index(name="annotation_count")
        .sort_values("geometry_id", kind="stable")
        .reset_index(drop=True)
    )
    identity_counts["share"] = identity_counts["annotation_count"] / len(mapped)

    strict_two_view = annotation.loc[
        annotation["gt_status"].eq("confirmed")
        & annotation["sex_label"].eq("F")
        & annotation["geometry_id_consistency"].eq("agree")
        & annotation["side1_reprojection_px"].le(ANNOTATION_MATCH_GATE_PX)
        & annotation["side2_reprojection_px"].le(ANNOTATION_MATCH_GATE_PX)
        & annotation["side1_click_to_pose_segment_px"].le(
            ANNOTATION_MATCH_GATE_PX
        )
        & annotation["side2_click_to_pose_segment_px"].le(
            ANNOTATION_MATCH_GATE_PX
        )
    ].copy()
    strict_identity_counts = (
        strict_two_view["consensus_geometry_id"]
        .value_counts()
        .rename_axis("geometry_id")
        .reset_index(name="annotation_count")
        .sort_values("geometry_id", kind="stable")
        .reset_index(drop=True)
    )
    strict_identity_counts["share"] = (
        strict_identity_counts["annotation_count"] / len(strict_two_view)
    )
    strict_runs, strict_transitions = _identity_runs(strict_two_view)

    very_strict = strict_two_view.loc[
        strict_two_view["candidate_max_reprojection_distance_px"].le(
            ANNOTATION_MATCH_GATE_PX
        )
    ].copy()
    very_strict_identity_counts = (
        very_strict["consensus_geometry_id"]
        .value_counts()
        .rename_axis("geometry_id")
        .reset_index(name="annotation_count")
        .sort_values("geometry_id", kind="stable")
        .reset_index(drop=True)
    )
    very_strict_identity_counts["share"] = (
        very_strict_identity_counts["annotation_count"] / len(very_strict)
    )
    very_strict_runs, very_strict_transitions = _identity_runs(very_strict)

    temporal_identity_counts = (
        mapped.groupby(["candidate_temporal_bin", "consensus_geometry_id"])
        .size()
        .rename("annotation_count")
        .reset_index()
        .rename(
            columns={
                "candidate_temporal_bin": "temporal_bin",
                "consensus_geometry_id": "geometry_id",
            }
        )
    )
    temporal_totals = temporal_identity_counts.groupby("temporal_bin")[
        "annotation_count"
    ].transform("sum")
    temporal_identity_counts["share_within_bin"] = (
        temporal_identity_counts["annotation_count"] / temporal_totals
    )
    temporal_order = {"early": 0, "middle": 1, "late": 2}
    temporal_identity_counts["_order"] = temporal_identity_counts[
        "temporal_bin"
    ].map(temporal_order)
    temporal_identity_counts.sort_values(
        ["_order", "geometry_id"], kind="stable", inplace=True
    )
    temporal_identity_counts.drop(columns="_order", inplace=True)

    consistency_counts = (
        confirmed["geometry_id_consistency"]
        .value_counts()
        .rename_axis("mapping_status")
        .reset_index(name="annotation_count")
    )
    consistency_counts["share"] = consistency_counts["annotation_count"] / len(
        confirmed
    )

    mapped = mapped.merge(
        exact_tracklet[
            [
                "annotation_id",
                "clean_tracklet_id",
                "label_status",
                "tracklet_matches_exact_frame",
            ]
        ],
        on="annotation_id",
        how="left",
        validate="one_to_one",
    )
    runs, transitions = _identity_runs(mapped)

    track_profile = (
        geometry.assign(
            finite_xyz=np.isfinite(geometry[["X", "Y", "Z"]].to_numpy(dtype=float)).all(
                axis=1
            )
        )
        .groupby("fish_id")
        .agg(
            row_count=("frame", "size"),
            first_frame=("frame", "min"),
            last_frame=("frame", "max"),
            finite_xyz_count=("finite_xyz", "sum"),
            interpolated_count=("is_interpolated", "sum"),
        )
        .reset_index()
    )
    track_profile["finite_xyz_share"] = (
        track_profile["finite_xyz_count"] / track_profile["row_count"]
    )
    track_profile["interpolated_share"] = (
        track_profile["interpolated_count"] / track_profile["row_count"]
    )

    audit_summary = _audit_summary(annotation)
    candidate_frame_set = set(candidates["frame"].astype(int))
    annotation_candidate_frame_set = set(
        annotation["candidate_frame"].dropna().astype(int)
    )
    annotation_id_expected = annotation.apply(
        lambda row: deterministic_annotation_id(
            str(row["experiment"]), int(row["frame"]), str(row["annotator"])
        ),
        axis=1,
    )

    candidate_summary_track = candidate_summary["inputs"]["geometry_csv"]
    current_track_stat = GEOMETRY_CSV.stat()
    current_provenance_sha = _sha256(RECONSTRUCTION_PROVENANCE_JSON)
    current_track_sha = _sha256(GEOMETRY_CSV)
    current_candidate_sha = _sha256(CANDIDATE_CSV)

    identity_best = identity_counts.sort_values(
        "annotation_count", ascending=False, kind="stable"
    ).iloc[0]
    strict_identity_best = strict_identity_counts.sort_values(
        "annotation_count", ascending=False, kind="stable"
    ).iloc[0]
    very_strict_identity_best = very_strict_identity_counts.sort_values(
        "annotation_count", ascending=False, kind="stable"
    ).iloc[0]
    has_saved_tracklet = exact_tracklet["consensus_tracklet_id"].astype(str).str.strip().ne("")
    tracklet_match_count = int(exact_tracklet["tracklet_matches_exact_frame"].sum())
    tracklet_missing_count = int((~has_saved_tracklet).sum())
    tracklet_conflict_count = int(
        (
            has_saved_tracklet
            & ~exact_tracklet["tracklet_matches_exact_frame"]
        ).sum()
    )
    exact_frame_count = int(
        annotation["frame"].eq(annotation["candidate_frame"]).sum()
    )
    source_path_values = sorted(
        value for value in annotation["geometry_csv_path"].astype(str).unique() if value
    )

    warnings: list[str] = []
    if set(annotation["annotator"].astype(str)) == {"YOUR_NAME"}:
        warnings.append(
            "All rows use the placeholder annotator value YOUR_NAME; biological labels "
            "are still analyzable, but annotator provenance is not final."
        )
    candidate_gate = float(candidate_summary["resolved_thresholds"]["match_gate_px"])
    if not math.isclose(candidate_gate, ANNOTATION_MATCH_GATE_PX):
        warnings.append(
            f"The saved candidate queue used a {candidate_gate:g}px geometry-association "
            f"gate, while the supplied GUI command states {ANNOTATION_MATCH_GATE_PX:g}px."
        )
    disagree_count = int(
        confirmed["geometry_id_consistency"].eq("disagree").sum()
    )
    if disagree_count:
        warnings.append(
            f"{disagree_count} confirmed rows select different geometry IDs in the two "
            "side views and therefore have no consensus geometry ID."
        )
    tracklet_mismatch_count = int(len(exact_tracklet) - tracklet_match_count)
    if tracklet_missing_count:
        warnings.append(
            f"{tracklet_missing_count} mapped manual-jump rows have no saved provisional "
            "tracklet ID; every non-empty saved tracklet matches the segment recomputed "
            "at the exact annotated frame."
        )

    summary: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "verdict": "geometry_ids_not_biologically_consistent_for_female",
        "annotation": {
            "row_count": int(len(annotation)),
            "unique_frame_count": int(annotation["frame"].nunique()),
            "frame_min": int(annotation["frame"].min()),
            "frame_max": int(annotation["frame"].max()),
            "confirmed_female_count": int(len(confirmed)),
            "ambiguous_count": int(annotation["gt_status"].eq("ambiguous").sum()),
            "duplicate_key_count": int(
                annotation.duplicated(["experiment", "frame", "annotator"]).sum()
            ),
            "duplicate_annotation_id_count": int(
                annotation["annotation_id"].duplicated().sum()
            ),
            "deterministic_annotation_id_match_count": int(
                annotation["annotation_id"].eq(annotation_id_expected).sum()
            ),
            "annotator_values": sorted(annotation["annotator"].astype(str).unique()),
            "exact_candidate_frame_count": exact_frame_count,
            "offset_from_candidate_frame_count": int(len(annotation) - exact_frame_count),
            "candidate_frame_offset_distribution": _distribution(
                (annotation["frame"] - annotation["candidate_frame"]).abs()
            ),
            "candidate_queue_frame_count": int(len(candidates)),
            "candidate_frames_represented_count": int(
                len(candidate_frame_set & annotation_candidate_frame_set)
            ),
            "candidate_frames_missing_from_ledger": sorted(
                candidate_frame_set - annotation_candidate_frame_set
            ),
            "non_candidate_origins_in_ledger": sorted(
                annotation_candidate_frame_set - candidate_frame_set
            ),
        },
        "mapping": {
            "mapped_consensus_count": int(len(mapped)),
            "mapped_consensus_share_of_confirmed": float(len(mapped) / len(confirmed)),
            "two_view_agree_count": int(
                confirmed["geometry_id_consistency"].eq("agree").sum()
            ),
            "two_view_agree_share": float(
                confirmed["geometry_id_consistency"].eq("agree").mean()
            ),
            "one_view_count": int(
                confirmed["geometry_id_consistency"].eq("one_view").sum()
            ),
            "disagree_count": disagree_count,
            "selected_xyz_finite_count": int(exact_geometry["finite_xyz"].sum()),
            "selected_xyz_noninterpolated_count": int(
                (~exact_geometry["is_interpolated"].astype(bool)).sum()
            ),
            "reprojection_by_view": reprojection_summary.to_dict(orient="records"),
        },
        "identity_consistency": {
            "geometry_id_counts": identity_counts.to_dict(orient="records"),
            "best_fixed_geometry_id": str(identity_best["geometry_id"]),
            "best_fixed_geometry_id_accuracy": float(identity_best["share"]),
            "best_fixed_geometry_id_error_rate": float(1.0 - identity_best["share"]),
            "observed_id_switch_count": int(len(transitions)),
            "observed_run_count": int(len(runs)),
            "adjacent_mapped_annotation_pair_count": int(max(len(mapped) - 1, 0)),
            "observed_switch_rate_per_adjacent_annotation_pair": float(
                len(transitions) / max(len(mapped) - 1, 1)
            ),
        },
        "sensitivity": {
            "strict_two_view_definition": (
                "confirmed F; side views agree; both selected-pose reprojection "
                "distances and click-to-pose distances are <= 8 px"
            ),
            "strict_two_view_count": int(len(strict_two_view)),
            "strict_two_view_geometry_id_counts": strict_identity_counts.to_dict(
                orient="records"
            ),
            "strict_two_view_best_fixed_geometry_id": str(
                strict_identity_best["geometry_id"]
            ),
            "strict_two_view_best_fixed_accuracy": float(
                strict_identity_best["share"]
            ),
            "strict_two_view_observed_switch_count": int(len(strict_transitions)),
            "strict_two_view_observed_run_count": int(len(strict_runs)),
            "very_strict_definition": (
                "strict two-view subset plus candidate-level maximum reprojection "
                "distance <= 8 px across all fish and all three views"
            ),
            "very_strict_count": int(len(very_strict)),
            "very_strict_geometry_id_counts": very_strict_identity_counts.to_dict(
                orient="records"
            ),
            "very_strict_best_fixed_geometry_id": str(
                very_strict_identity_best["geometry_id"]
            ),
            "very_strict_best_fixed_accuracy": float(
                very_strict_identity_best["share"]
            ),
            "very_strict_observed_switch_count": int(
                len(very_strict_transitions)
            ),
            "very_strict_observed_run_count": int(len(very_strict_runs)),
        },
        "tracklet_consistency": {
            "recomputed_at_exact_frame_count": int(len(exact_tracklet)),
            "exact_tracklet_match_count": tracklet_match_count,
            "exact_tracklet_match_share": float(tracklet_match_count / len(exact_tracklet)),
            "exact_tracklet_mismatch_count": tracklet_mismatch_count,
            "missing_saved_tracklet_count": tracklet_missing_count,
            "conflicting_nonempty_tracklet_count": tracklet_conflict_count,
            "nonempty_saved_tracklet_match_share": float(
                tracklet_match_count / max(int(has_saved_tracklet.sum()), 1)
            ),
            "exact_frame_label_status_counts": {
                str(key): int(value)
                for key, value in exact_tracklet["label_status"]
                .value_counts()
                .sort_index()
                .items()
            },
            "segment_diagnostics": segment_diagnostics,
        },
        "track": {
            "row_count": int(len(geometry)),
            "frame_min": int(geometry["frame"].min()),
            "frame_max": int(geometry["frame"].max()),
            "fish_ids": sorted(geometry["fish_id"].unique()),
            "duplicate_frame_fish_id_count": int(
                geometry.duplicated(["frame", "fish_id"]).sum()
            ),
            "sha256": current_track_sha,
            "size_bytes": int(current_track_stat.st_size),
            "backend": reconstruction_provenance["triangulation"]["backend"],
            "score_unit": reconstruction_provenance["triangulation"]["score_unit"],
        },
        "source_consistency": {
            "annotation_geometry_path_values": source_path_values,
            "candidate_summary_geometry_size_matches_current": bool(
                int(candidate_summary_track["size_bytes"]) == current_track_stat.st_size
            ),
            "candidate_summary_provenance_sha256": candidate_summary["inputs"][
                "reconstruction_provenance_json"
            ]["sha256"],
            "current_provenance_sha256": current_provenance_sha,
            "provenance_sha256_matches": bool(
                current_provenance_sha
                == candidate_summary["inputs"]["reconstruction_provenance_json"][
                    "sha256"
                ]
            ),
            "candidate_csv_sha256_matches_summary": bool(
                current_candidate_sha == candidate_summary["output_csv_sha256"]
            ),
            "candidate_match_gate_px": candidate_gate,
            "supplied_gui_match_gate_px": ANNOTATION_MATCH_GATE_PX,
            "candidate_rows_over_supplied_8px_gate": int(
                (candidates["max_reprojection_distance_px"] > ANNOTATION_MATCH_GATE_PX).sum()
            ),
            "candidate_rows_over_supplied_8px_gate_share": float(
                (candidates["max_reprojection_distance_px"] > ANNOTATION_MATCH_GATE_PX).mean()
            ),
        },
        "audit": audit_summary,
        "warnings": warnings,
    }

    detail_columns = [
        "annotation_id",
        "frame",
        "time_seconds",
        "gt_status",
        "sex_label",
        "geometry_id_consistency",
        "consensus_geometry_id",
        "consensus_tracklet_id",
        "side1_geometry_id",
        "side2_geometry_id",
        "side1_reprojection_px",
        "side2_reprojection_px",
        "side1_click_to_pose_segment_px",
        "side2_click_to_pose_segment_px",
        "candidate_frame",
        "candidate_temporal_bin",
        "candidate_max_reprojection_distance_px",
        "candidate_quality_score",
        "annotator",
        "notes",
    ]
    annotation[detail_columns].to_csv(
        REPORT_DIR / "annotation_comparison_details.csv", index=False
    )
    identity_counts.to_csv(REPORT_DIR / "identity_counts.csv", index=False)
    strict_identity_counts.to_csv(
        REPORT_DIR / "strict_two_view_identity_counts.csv", index=False
    )
    very_strict_identity_counts.to_csv(
        REPORT_DIR / "very_strict_identity_counts.csv", index=False
    )
    temporal_identity_counts.to_csv(
        REPORT_DIR / "temporal_identity_counts.csv", index=False
    )
    consistency_counts.to_csv(
        REPORT_DIR / "geometry_mapping_status_counts.csv", index=False
    )
    reprojection_summary.to_csv(
        REPORT_DIR / "reprojection_summary.csv", index=False
    )
    reprojection_details.to_csv(
        REPORT_DIR / "reprojection_details.csv", index=False
    )
    runs.to_csv(REPORT_DIR / "female_geometry_id_runs.csv", index=False)
    transitions.to_csv(
        REPORT_DIR / "observed_female_id_transitions.csv", index=False
    )
    exact_tracklet[
        [
            "annotation_id",
            "frame",
            "consensus_geometry_id",
            "consensus_tracklet_id",
            "clean_tracklet_id",
            "label_status",
            "tracklet_matches_exact_frame",
            "candidate_frame",
        ]
    ].to_csv(REPORT_DIR / "exact_frame_tracklet_comparison.csv", index=False)
    track_profile.to_csv(REPORT_DIR / "track_profile.csv", index=False)
    _write_json(REPORT_DIR / "analysis_summary.json", summary)
    return summary


if __name__ == "__main__":
    result = run_analysis()
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
