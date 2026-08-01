from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("report/20260731/round8_mouth_tracking_ablation")
A5_ROOT = ROOT / "training_a5"
WINDOWS = {
    "reviewed_event_1": (72371, 73027),
    "reviewed_event_3": (174316, 180515),
    "control_for_event_1": (32264, 32920),
    "control_for_event_3": (21840, 28039),
}
EXPECTED_TRACKLET_RUNNER = (
    "1e75e10654b5d6c233ec3ea5ff2f8c969dac3dd41027031cd332d3c88096d4e0"
)
EXPECTED_MOUTH_RUNNER = (
    "d856585da1515a06067909f0b7aeead1eab2af81973a2cf097e17de33a7192e0"
)
EXPECTED_DEFERRED = (
    "c08ac9071b7ab218d338a2e1a2859811a2c322bf1d93f5740e1202c88a663add"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "n": int(len(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def authenticate(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash mismatch: {actual} != {expected}")


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, table: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    window_records = []
    all_direct = []
    all_hints = []
    all_hint_audits = []
    summary_manifest = []
    for window, (frame_start, frame_end) in WINDOWS.items():
        directory = A5_ROOT / window
        hint_summary_path = directory / "a5_hints_summary.json"
        a5_summary_path = directory / "a5_summary.json"
        hint_summary = json.loads(hint_summary_path.read_text(encoding="utf-8"))
        a5_summary = json.loads(a5_summary_path.read_text(encoding="utf-8"))
        hints_path = directory / "a5_hints.csv"
        resets_path = directory / "a5_forced_reset_frames.json"
        authenticate(
            hints_path,
            hint_summary["outputs"]["hints_sha256"],
            f"{window} hints",
        )
        authenticate(
            resets_path,
            hint_summary["outputs"]["forced_reset_frames_sha256"],
            f"{window} resets",
        )
        a4_path = (
            ROOT
            / "training_a4_cutoff_sweep"
            / window
            / "s035_m005_g0"
            / "a4_selected_observations.parquet"
        )
        a2_path = (
            ROOT
            / "training_relink"
            / window
            / "A2_v025_h5"
            / "a2_direct.csv"
        )
        authenticate(
            a4_path,
            hint_summary["input_sha256"]["a4_selected"],
            f"{window} selected A4",
        )
        authenticate(
            a2_path,
            hint_summary["input_sha256"]["a2_direct"],
            f"{window} frozen A2",
        )
        hint_code = hint_summary["code_sha256"]
        if (
            hint_code[
                "scripts\\reconstruction_3d\\mouth_tracklet_ablation.py"
            ]
            != EXPECTED_TRACKLET_RUNNER
        ):
            raise RuntimeError(f"{window} hint runner drift")
        for name, artifact in a5_summary["inputs"].items():
            authenticate(
                Path(artifact["path"]), artifact["sha256"], f"{window} input {name}"
            )
        for name, artifact in a5_summary["outputs"].items():
            authenticate(
                Path(artifact["path"]), artifact["sha256"], f"{window} output {name}"
            )
        code = a5_summary["code_sha256"]
        if (
            code["scripts\\reconstruction_3d\\mouth_tracklet_ablation.py"]
            != EXPECTED_TRACKLET_RUNNER
        ):
            raise RuntimeError(f"{window} A5 runner drift")
        if (
            code["scripts\\reconstruction_3d\\mouth_ablation.py"]
            != EXPECTED_MOUTH_RUNNER
        ):
            raise RuntimeError(f"{window} mouth runner drift")
        if (
            code["scripts\\reconstruction_3d\\deferred_identity_tracking.py"]
            != EXPECTED_DEFERRED
        ):
            raise RuntimeError(f"{window} deferred runner drift")
        if a5_summary["calibration"]["backend"] != "refractive":
            raise RuntimeError(f"{window} is not refractive")
        if a5_summary["calibration"]["score_unit"] != "mm^2":
            raise RuntimeError(f"{window} residual unit drift")

        hints = pd.read_csv(hints_path)
        direct = pd.read_csv(directory / "a5_direct.csv")
        hint_audit = pd.read_csv(directory / "a5_hint_application_audit.csv")
        frames = frame_end - frame_start + 1
        if len(direct) != 3 * frames:
            raise RuntimeError(f"{window} direct grid is incomplete")
        if direct[["frame", "fish_id"]].duplicated().any():
            raise RuntimeError(f"{window} direct keys are not unique")
        if direct.frame.min() != frame_start or direct.frame.max() != frame_end:
            raise RuntimeError(f"{window} direct frame bounds differ")
        if len(hint_audit) != len(hints):
            raise RuntimeError(f"{window} hint audit does not cover every hint")
        honored = (
            hint_audit.honored.astype(str).str.lower().map(
                {"true": True, "false": False}
            )
        )
        if honored.isna().any():
            raise RuntimeError(f"{window} hint honored field is malformed")
        if int(honored.sum()) != int(a5_summary["hints_honored"]):
            raise RuntimeError(f"{window} honored count differs from summary")

        finite = np.all(
            np.isfinite(direct[["X", "Y", "Z"]].to_numpy(float)), axis=1
        )
        direct_complete = (
            direct.assign(_finite=finite).groupby("frame")._finite.sum().eq(3)
        )
        jumps = []
        for (_fish, _segment), group in direct[finite].groupby(
            ["fish_id", "identity_segment"], sort=True
        ):
            group = group.sort_values("frame")
            adjacent = np.diff(group.frame.to_numpy(int)) == 1
            delta = np.diff(group[["X", "Y", "Z"]].to_numpy(float), axis=0)
            jumps.extend(np.linalg.norm(delta[adjacent], axis=1))
        counts = hint_summary["filter_counts"]
        vote_audit = hint_summary["mapping_vote_audit"]
        record = {
            "window": window,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frames": frames,
            "rows": int(len(direct)),
            "a4_selected_input_rows": int(counts["selected_input_rows"]),
            "hint_rows": int(len(hints)),
            "hint_frames": int(hints.frame.nunique()),
            "hint_row_coverage": float(len(hints) / (3 * frames)),
            "mapped_segments": int(hint_summary["mapped_segments"]),
            "vote_rejected_segments": int(
                sum(not bool(row["accepted_by_vote"]) for row in vote_audit)
            ),
            "overlap_conflict_segments": int(
                counts["rejected_overlap_identity_segments"]
            ),
            "nonbijective_hint_rows_rejected": int(
                counts["rejected_per_frame_non_bijective"]
            ),
            "forced_reset_frames": int(
                len(hint_summary["forced_reset_frames"])
            ),
            "hints_honored": int(honored.sum()),
            "hints_not_honored": int((~honored).sum()),
            "hint_honor_rate": float(honored.mean()) if len(honored) else None,
            "all_three_direct_frames": int(direct_complete.sum()),
            "all_three_direct_fraction": float(direct_complete.mean()),
            "finite_direct_rows": int(finite.sum()),
            "unresolved_rows": int(
                direct.identity_status.astype(str)
                .str.startswith("unresolved")
                .sum()
            ),
            "unresolved_row_fraction": float(
                direct.identity_status.astype(str)
                .str.startswith("unresolved")
                .mean()
            ),
            "jump_gt8_count": int((np.asarray(jumps) > 8).sum()),
            "jump_gt20_count": int((np.asarray(jumps) > 20).sum()),
            "residual_n": int(
                np.isfinite(direct.ray_residual_mm2.to_numpy(float)).sum()
            ),
            "wall_seconds_hints": float(hint_summary["wall_seconds"]),
            "peak_rss_bytes_hints": int(hint_summary["peak_rss_bytes"]),
            "wall_seconds_a5": float(a5_summary["wall_seconds"]),
            "peak_rss_bytes_a5": int(a5_summary["peak_rss_bytes"]),
            "female_full_eligible_accuracy": None,
            "female_metric_status": "pending_independent_quality_scorer",
        }
        window_records.append(record)
        direct = direct.copy()
        direct["window"] = window
        hints = hints.copy()
        hints["window"] = window
        hint_audit = hint_audit.copy()
        hint_audit["window"] = window
        all_direct.append(direct)
        all_hints.append(hints)
        all_hint_audits.append(hint_audit)
        summary_manifest.append(
            {
                "window": window,
                "hint_summary": {
                    "path": str(hint_summary_path.resolve()),
                    "sha256": sha256(hint_summary_path),
                },
                "a5_summary": {
                    "path": str(a5_summary_path.resolve()),
                    "sha256": sha256(a5_summary_path),
                },
            }
        )

    records = pd.DataFrame.from_records(window_records).sort_values("window")
    direct = pd.concat(all_direct, ignore_index=True)
    hints = pd.concat(all_hints, ignore_index=True)
    hint_audits = pd.concat(all_hint_audits, ignore_index=True)
    finite = np.all(
        np.isfinite(direct[["X", "Y", "Z"]].to_numpy(float)), axis=1
    )
    complete = (
        direct.assign(_finite=finite)
        .groupby(["window", "frame"])
        ._finite.sum()
        .eq(3)
    )
    combined_jumps = []
    for (_window, _fish, _segment), group in direct[finite].groupby(
        ["window", "fish_id", "identity_segment"], sort=True
    ):
        group = group.sort_values("frame")
        adjacent = np.diff(group.frame.to_numpy(int)) == 1
        delta = np.diff(group[["X", "Y", "Z"]].to_numpy(float), axis=0)
        combined_jumps.extend(np.linalg.norm(delta[adjacent], axis=1))
    combined_honored = (
        hint_audits.honored.astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )
    aggregate = {
        "schema_version": 1,
        "status": "training_a5_completed_and_authenticated",
        "scope": "authorized training events and length-matched training controls only",
        "forbidden_cohorts_opened": [],
        "frozen_dependencies": {
            "A2": {
                "configuration": "A2_v025_h5",
                "velocity_weight": 0.25,
                "history_frames": 5,
            },
            "A3": {
                "configuration": "A3_a005",
                "acceleration_weight": 0.05,
            },
            "A4": {
                "configuration": "s035_m005_g0",
                "separation_body_lengths": 0.35,
                "assignment_ambiguity_margin_body_lengths": 0.05,
                "maximum_missing_gap_frames": 0,
            },
            "A5_hint_thresholds": {
                "minimum_mouth_confidence": 0.5,
                "minimum_group_support_frames": 3,
                "maximum_group_median_ray_residual_mm2": 1.0,
                "maximum_group_mad_ray_residual_mm2": 0.25,
                "minimum_group_median_mouth_confidence": 0.5,
                "maximum_group_assignment_cost": 0.0,
                "minimum_mapping_votes": 3,
                "minimum_mapping_purity": 0.9,
                "minimum_vote_margin": 2,
                "xyz_tolerance_mm": 1e-9,
            },
        },
        "windows": window_records,
        "aggregate": {
            "frames": int(records.frames.sum()),
            "rows": int(len(direct)),
            "a4_selected_input_rows": int(
                records.a4_selected_input_rows.sum()
            ),
            "hint_rows": int(len(hints)),
            "hint_frames": int(
                hints[["window", "frame"]].drop_duplicates().shape[0]
            ),
            "hint_row_coverage": float(
                len(hints) / (3 * records.frames.sum())
            ),
            "mapped_segments": int(records.mapped_segments.sum()),
            "vote_rejected_segments": int(
                records.vote_rejected_segments.sum()
            ),
            "overlap_conflict_segments": int(
                records.overlap_conflict_segments.sum()
            ),
            "nonbijective_hint_rows_rejected": int(
                records.nonbijective_hint_rows_rejected.sum()
            ),
            "hints_honored": int(combined_honored.sum()),
            "hints_not_honored": int((~combined_honored).sum()),
            "hint_honor_rate": float(combined_honored.mean()),
            "all_three_direct_frames": int(complete.sum()),
            "all_three_direct_fraction": float(complete.mean()),
            "finite_direct_rows": int(finite.sum()),
            "unresolved_rows": int(
                direct.identity_status.astype(str)
                .str.startswith("unresolved")
                .sum()
            ),
            "unresolved_row_fraction": float(
                direct.identity_status.astype(str)
                .str.startswith("unresolved")
                .mean()
            ),
            "ray_residual_mm2": distribution(direct.ray_residual_mm2),
            "adjacent_jump_mm": distribution(combined_jumps),
            "jump_gt8_count": int(
                (np.asarray(combined_jumps, dtype=float) > 8).sum()
            ),
            "jump_gt20_count": int(
                (np.asarray(combined_jumps, dtype=float) > 20).sum()
            ),
            "female_full_eligible_accuracy": None,
            "female_metric_status": "pending_independent_quality_scorer",
        },
        "summary_manifest": summary_manifest,
        "code_sha256": {
            "mouth_tracklet_ablation": EXPECTED_TRACKLET_RUNNER,
            "mouth_ablation": EXPECTED_MOUTH_RUNNER,
            "deferred_identity_tracking": EXPECTED_DEFERRED,
            "finalizer": sha256(Path(__file__)),
            "launcher": sha256(ROOT / "run_training_a5.ps1"),
        },
    }
    csv_path = A5_ROOT / "training_a5_summary.csv"
    audit_path = A5_ROOT / "training_a5_audit.json"
    atomic_csv(csv_path, records)
    atomic_json(audit_path, aggregate)
    marker = {
        "schema_version": 1,
        "status": "completed_and_authenticated",
        "windows": 4,
        "hint_summaries": 4,
        "a5_summaries": 4,
        "all_inputs_outputs_authenticated": True,
        "training_a5_summary_csv": {
            "path": str(csv_path.resolve()),
            "sha256": sha256(csv_path),
        },
        "training_a5_audit_json": {
            "path": str(audit_path.resolve()),
            "sha256": sha256(audit_path),
        },
        "summary_manifest": summary_manifest,
    }
    atomic_json(A5_ROOT / "TRAINING_A5_COMPLETED.json", marker)
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
