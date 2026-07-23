"""Audit human evidence quality and multi-frame motion at 6/8 mm branches.

This report-side analysis deliberately does not decide which jump gate is
correct. It separates human-only identity evidence from automatic
SLEAP/geometry mapping evidence, then measures bidirectional robust tangent
continuity around each branch for both controlled outputs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.analysis.female_sex_gt import read_annotation_ledger


HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
LEDGER = (
    REPO_ROOT
    / "annotations/fish_sex_gt/20251225_K_1_2_FCMC/"
    "jump6_gate_branch_female_review.csv"
)
JUMP6 = HERE.parent / "mouth_only_jump6_3d_aligned_to_jump8.csv"
JUMP8 = (
    REPO_ROOT
    / "report/20260718/k12_jump8_strict_mouth_rerun/"
    "mouth_only_jump8_rerun_3d_aligned_to_historical_frame.csv"
)
TRANSITIONS = HERE / "gate_branch_transitions.csv"


def _parse_mapping(value: object) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in str(value).split(";"):
        if "->" not in item:
            continue
        left, right = item.split("->", maxsplit=1)
        result[left.strip()] = right.strip()
    return result


def _branch_from_annotation(row: pd.Series) -> str:
    match = re.search(r"\bB\d{3}\b", str(row.get("candidate_ids", "")))
    return match.group(0) if match else ""


def audit_review_quality(
    annotations: pd.DataFrame, transitions: pd.DataFrame
) -> pd.DataFrame:
    """Return one row per branch without promoting legacy rows to usable."""
    annotations = annotations.copy()
    annotations["branch_id"] = annotations.apply(_branch_from_annotation, axis=1)
    annotations["frame"] = pd.to_numeric(annotations["frame"]).astype(int)
    rows: list[dict[str, Any]] = []
    for branch in transitions.itertuples(index=False):
        candidates = annotations[annotations["branch_id"].eq(branch.branch_id)]
        if candidates.empty:
            candidates = annotations[
                annotations["frame"].eq(int(branch.trigger_frame))
            ]
        record = {
            "branch_id": str(branch.branch_id),
            "trigger_frame": int(branch.trigger_frame),
            "review_start_frame": int(branch.review_start_frame),
            "review_end_frame": int(branch.review_end_frame),
            "annotation_frame": pd.NA,
            "gt_status": "unannotated",
            "side1_selection_status": "",
            "side2_selection_status": "",
            "geometry_id_consistency": "",
            "consensus_geometry_id": "",
            "evidence_quality": "unreviewed",
            "observation_issue_flags": "",
            "notes": "",
        }
        if not candidates.empty:
            selected = candidates.sort_values("updated_at_utc").iloc[-1]
            for column in (
                "gt_status",
                "side1_selection_status",
                "side2_selection_status",
                "geometry_id_consistency",
                "consensus_geometry_id",
                "evidence_quality",
                "observation_issue_flags",
                "notes",
            ):
                record[column] = str(selected.get(column, ""))
            record["annotation_frame"] = int(selected["frame"])
        notes_bad = "bad" in str(record["notes"]).lower()
        record["automatic_mapping_candidate"] = bool(
            record["gt_status"] == "confirmed"
            and record["side1_selection_status"] == "sleap_linked"
            and record["side2_selection_status"] == "sleap_linked"
            and record["geometry_id_consistency"] == "agree"
            and not notes_bad
        )
        record["tracker_scoring_eligible"] = bool(
            record["automatic_mapping_candidate"]
            and record["evidence_quality"] == "usable"
        )
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def load_track_windows(
    path: Path, trigger_frames: np.ndarray, *, radius: int = 30
) -> pd.DataFrame:
    """Load only direct finite rows near branch triggers from a full CSV."""
    parts: list[pd.DataFrame] = []
    columns = ["frame", "fish_id", "X", "Y", "Z", "is_interpolated"]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=200_000):
        frames = chunk["frame"].to_numpy(dtype=np.int64)
        near = np.zeros(len(chunk), dtype=bool)
        for trigger in trigger_frames:
            near |= np.abs(frames - int(trigger)) <= radius
        if not near.any():
            continue
        part = chunk.loc[near].copy()
        interpolated = part["is_interpolated"].astype(str).str.lower().eq("true")
        finite = np.isfinite(part[["X", "Y", "Z"]].to_numpy(dtype=float)).all(axis=1)
        parts.append(part.loc[~interpolated & finite])
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["fish_id", "frame"], kind="stable", ignore_index=True
    )


def _robust_velocity(frames: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    if len(frames) < 3 or int(frames.max() - frames.min()) < 2:
        return np.full(3, np.nan)
    slopes: list[np.ndarray] = []
    for left in range(len(frames) - 1):
        for right in range(left + 1, len(frames)):
            gap = int(frames[right] - frames[left])
            if gap > 0:
                slopes.append((xyz[right] - xyz[left]) / gap)
    return (
        np.median(np.asarray(slopes, dtype=float), axis=0)
        if slopes
        else np.full(3, np.nan)
    )


def _angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if not np.isfinite(denominator) or denominator <= 1e-12:
        return np.nan
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def boundary_motion(
    table: pd.DataFrame,
    *,
    fish_id: str,
    trigger_frame: int,
    radius: int = 30,
    endpoint_points: int = 15,
) -> dict[str, Any]:
    track = table[table["fish_id"].eq(fish_id)].sort_values("frame")
    past = track[
        track["frame"].between(trigger_frame - radius, trigger_frame - 1)
    ].tail(endpoint_points)
    future = track[
        track["frame"].between(trigger_frame, trigger_frame + radius)
    ].head(endpoint_points)
    past_frames = past["frame"].to_numpy(dtype=np.int64)
    future_frames = future["frame"].to_numpy(dtype=np.int64)
    past_xyz = past[["X", "Y", "Z"]].to_numpy(dtype=float)
    future_xyz = future[["X", "Y", "Z"]].to_numpy(dtype=float)
    past_velocity = _robust_velocity(past_frames, past_xyz)
    future_velocity = _robust_velocity(future_frames, future_xyz)
    bridge_error = np.nan
    bridge_gap = np.nan
    if len(past) and len(future) and np.all(np.isfinite(past_velocity)):
        bridge_gap = int(future_frames[0] - past_frames[-1])
        prediction = past_xyz[-1] + past_velocity * bridge_gap
        bridge_error = float(np.linalg.norm(future_xyz[0] - prediction))
    return {
        "past_samples": int(len(past)),
        "future_samples": int(len(future)),
        "past_span_frames": (
            int(past_frames[-1] - past_frames[0]) if len(past) else 0
        ),
        "future_span_frames": (
            int(future_frames[-1] - future_frames[0]) if len(future) else 0
        ),
        "robust_tangent_angle_deg": _angle_degrees(
            past_velocity, future_velocity
        ),
        "forward_bridge_prediction_error_mm": bridge_error,
        "bridge_gap_frames": bridge_gap,
        "past_speed_mm_per_frame": float(np.linalg.norm(past_velocity)),
        "future_speed_mm_per_frame": float(np.linalg.norm(future_velocity)),
    }


def audit_branch_motion(
    transitions: pd.DataFrame,
    jump6: pd.DataFrame,
    jump8: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for branch in transitions.itertuples(index=False):
        old_mapping = _parse_mapping(branch.old_mapping_6_to_8)
        new_mapping = _parse_mapping(branch.new_mapping_6_to_8)
        for six_id in sorted(old_mapping):
            old_eight_id = old_mapping.get(six_id, "")
            new_eight_id = new_mapping.get(six_id, "")
            if not old_eight_id or old_eight_id == new_eight_id:
                continue
            six_metrics = boundary_motion(
                jump6,
                fish_id=six_id,
                trigger_frame=int(branch.trigger_frame),
            )
            eight_metrics = boundary_motion(
                jump8,
                fish_id=old_eight_id,
                trigger_frame=int(branch.trigger_frame),
            )
            record: dict[str, Any] = {
                "branch_id": str(branch.branch_id),
                "trigger_frame": int(branch.trigger_frame),
                "six_id": six_id,
                "old_eight_id": old_eight_id,
                "new_eight_id": new_eight_id,
            }
            record.update({f"jump6_{key}": value for key, value in six_metrics.items()})
            record.update({f"jump8_{key}": value for key, value in eight_metrics.items()})
            rows.append(record)
    return pd.DataFrame.from_records(rows)


def summarize(
    quality: pd.DataFrame, motion: pd.DataFrame
) -> dict[str, Any]:
    six_angle = pd.to_numeric(
        motion.get("jump6_robust_tangent_angle_deg"), errors="coerce"
    ).dropna()
    eight_angle = pd.to_numeric(
        motion.get("jump8_robust_tangent_angle_deg"), errors="coerce"
    ).dropna()
    return {
        "schema_version": 1,
        "decision": "6_vs_8_inconclusive_pending_quality_review",
        "review": {
            "branches": int(len(quality)),
            "annotated_branches": int(quality["annotation_frame"].notna().sum()),
            "two_view_consistent_candidates": int(
                quality["automatic_mapping_candidate"].sum()
            ),
            "explicitly_quality_reviewed": int(
                quality["evidence_quality"].ne("unreviewed").sum()
            ),
            "tracker_scoring_eligible": int(
                quality["tracker_scoring_eligible"].sum()
            ),
            "policy": (
                "Only evidence_quality=usable may score an automatic geometry "
                "mapping; human_only remains valid human identity evidence."
            ),
        },
        "motion": {
            "changed_track_boundaries": int(len(motion)),
            "history_policy": (
                "15 direct observations on each side within 30 frames; "
                "Theil-Sen-style median of all pairwise 3D slopes"
            ),
            "jump6_tangent_angle_median_deg": (
                float(six_angle.median()) if len(six_angle) else None
            ),
            "jump8_tangent_angle_median_deg": (
                float(eight_angle.median()) if len(eight_angle) else None
            ),
            "jump6_boundaries_over_90_deg": int((six_angle > 90).sum()),
            "jump8_boundaries_over_90_deg": int((eight_angle > 90).sum()),
            "warning": (
                "Motion continuity is a diagnostic, not identity truth, when "
                "SLEAP observations are false-negative, shared, or merged."
            ),
        },
    }


def run() -> dict[str, Any]:
    transitions = pd.read_csv(TRANSITIONS)
    annotations = read_annotation_ledger(LEDGER)
    quality = audit_review_quality(annotations, transitions)
    triggers = transitions["trigger_frame"].to_numpy(dtype=np.int64)
    jump6 = load_track_windows(JUMP6, triggers)
    jump8 = load_track_windows(JUMP8, triggers)
    motion = audit_branch_motion(transitions, jump6, jump8)
    branch_motion = (
        motion.groupby(["branch_id", "trigger_frame"], as_index=False)
        .agg(
            changed_tracks=("six_id", "size"),
            jump6_max_tangent_angle_deg=(
                "jump6_robust_tangent_angle_deg",
                "max",
            ),
            jump8_max_tangent_angle_deg=(
                "jump8_robust_tangent_angle_deg",
                "max",
            ),
            jump6_max_bridge_error_mm=(
                "jump6_forward_bridge_prediction_error_mm",
                "max",
            ),
            jump8_max_bridge_error_mm=(
                "jump8_forward_bridge_prediction_error_mm",
                "max",
            ),
        )
        .sort_values("trigger_frame")
    )
    quality.to_csv(HERE / "review_quality_audit.csv", index=False)
    motion.to_csv(HERE / "branch_motion_diagnostics.csv", index=False)
    branch_motion.to_csv(HERE / "branch_motion_summary.csv", index=False)
    result = summarize(quality, motion)
    (HERE / "motion_audit_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "summary": result,
        "quality": quality,
        "motion": motion,
        "branch_motion": branch_motion,
    }


if __name__ == "__main__":
    output = run()
    print(json.dumps(output["summary"], indent=2))
