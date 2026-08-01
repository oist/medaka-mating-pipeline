"""Independent end-to-end contract audit for the A5 Python runner."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from scripts.reconstruction_3d.deferred_identity_tracking import (
    DeferredTrackingConfig,
)
from scripts.reconstruction_3d.mouth_tracklet_ablation import (
    build_a5_hints,
    run_a5_window,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    review_dir = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(dir=review_dir) as temporary:
        root = Path(temporary)
        geometry_records: list[dict[str, object]] = []
        selected_records: list[dict[str, object]] = []
        a2_records: list[dict[str, object]] = []
        for frame in range(3):
            for source_index in range(3):
                x = float(source_index * 20 + frame)
                hypothesis_id = f"f{frame}_h{source_index}"
                geometry_records.append(
                    {
                        "frame": frame,
                        "view_count": 3,
                        "ray_residual_mm2": 0.1,
                        "hypothesis_id": hypothesis_id,
                        "X": x,
                        "Y": 1.0,
                        "Z": 2.0,
                        "slot_cam0": source_index,
                        "slot_cam1": source_index,
                        "slot_cam2": source_index,
                    }
                )
                selected_records.append(
                    {
                        "frame": frame,
                        "path_id": f"PATH_{source_index + 1}",
                        "path_segment": 0,
                        "hypothesis_id": hypothesis_id,
                        "X": x,
                        "Y": 1.0,
                        "Z": 2.0,
                        "mouth_confidence": 0.95,
                        "view_count": 3,
                        "group_eligible": True,
                        "group_support_frames": 3,
                        "group_median_ray_residual_mm2": 0.1,
                        "group_mad_ray_residual_mm2": 0.0,
                        "group_median_mouth_confidence": 0.95,
                        "group_assignment_cost": -0.5,
                        "group_retained_count_mismatch_fraction": 0.0,
                    }
                )
                a2_records.append(
                    {
                        "frame": frame,
                        "fish_id": f"ID_{source_index + 1}",
                        "identity_segment": 0,
                        "source_index": source_index,
                        "hypothesis_id": hypothesis_id,
                        "X": x,
                        "Y": 1.0,
                        "Z": 2.0,
                    }
                )

        geometry_path = root / "geometry.parquet"
        pd.DataFrame(geometry_records).to_parquet(geometry_path, index=False)
        geometry_summary_path = root / "geometry_summary.json"
        geometry_summary_path.write_text(
            json.dumps(
                {
                    "output_sha256": sha256(geometry_path),
                    "calibration": {
                        "backend": "refractive",
                        "score_unit": "mm^2",
                        "provenance": {
                            "resolved_backend": "refractive",
                            "score_unit": "mm^2",
                        },
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        hints, resets, hint_summary = build_a5_hints(
            pd.DataFrame(selected_records),
            pd.DataFrame(a2_records),
        )
        hints_path = root / "a5_hints.csv"
        reset_path = root / "a5_resets.json"
        hints.to_csv(hints_path, index=False)
        reset_path.write_text(json.dumps(resets, indent=2), encoding="utf-8")
        hint_summary["outputs"] = {
            "hints_sha256": sha256(hints_path),
            "forced_reset_frames_sha256": sha256(reset_path),
        }
        hints_summary_path = root / "a5_hints_summary.json"
        hints_summary_path.write_text(
            json.dumps(hint_summary, indent=2),
            encoding="utf-8",
        )

        output_dir = root / "a5"
        summary = run_a5_window(
            geometry_path=geometry_path,
            geometry_summary_path=geometry_summary_path,
            hints_path=hints_path,
            forced_reset_frames_path=reset_path,
            hints_summary_path=hints_summary_path,
            output_dir=output_dir,
            frame_start=0,
            frame_end=2,
            config=DeferredTrackingConfig(
                velocity_prediction_weight=0.25,
                acceleration_weight=0.05,
                motion_history_frames=5,
            ),
            command=["independent_synthetic_a5_audit"],
        )
        direct = pd.read_csv(output_dir / "a5_direct.csv")
        application = pd.read_csv(output_dir / "a5_hint_application_audit.csv")
        conditions = {
            "hint_rows_are_complete": len(hints) == 9,
            "direct_has_three_rows_per_frame": bool(
                direct.groupby("frame").size().eq(3).all()
            ),
            "all_direct_coordinates_finite": bool(
                direct[["X", "Y", "Z"]].notna().all().all()
            ),
            "all_hints_honored": bool(
                application.honored.astype(bool).all()
            ),
            "summary_hashes_outputs": all(
                sha256(Path(value["path"])) == value["sha256"]
                for value in summary["outputs"].values()
            ),
            "calibration_contract_preserved": (
                summary["calibration"]["backend"] == "refractive"
                and summary["calibration"]["score_unit"] == "mm^2"
            ),
        }
        result = {
            "audit": "synthetic end-to-end A5 runner contract",
            "mouth_tracklet_ablation_sha256": sha256(
                Path(__file__).resolve().parents[4]
                / "scripts"
                / "reconstruction_3d"
                / "mouth_tracklet_ablation.py"
            ),
            "conditions": conditions,
            "hint_rows": int(len(hints)),
            "summary_hint_rows": int(summary["hint_rows"]),
            "summary_hints_honored": int(summary["hints_honored"]),
            "configuration_hash": summary["configuration_hash"],
            "result": "PASS" if all(conditions.values()) else "FAIL",
        }
    (review_dir / "a5_end_to_end_audit.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
