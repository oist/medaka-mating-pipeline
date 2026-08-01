"""Independent regression audit for A1/A2 long-gap identity semantics."""

from __future__ import annotations

import json
import hashlib
import tempfile
from pathlib import Path

import pandas as pd

from scripts.reconstruction_3d.mouth_ablation import relink_geometry


def main() -> None:
    output_root = Path(__file__).resolve().parent
    rows: list[dict[str, object]] = []
    for frame, offset in ((0, 0.0), (102, 100.0)):
        for index in range(3):
            rows.append(
                {
                    "frame": frame,
                    "view_count": 3,
                    "ray_residual_mm2": 0.1,
                    "hypothesis_id": f"f{frame}_h{index}",
                    "X": offset + 10.0 * index,
                    "Y": 0.0,
                    "Z": 0.0,
                    "slot_cam0": index,
                    "slot_cam1": index,
                    "slot_cam2": index,
                }
            )

    with tempfile.TemporaryDirectory(dir=output_root) as temporary:
        temporary_path = Path(temporary)
        geometry_path = temporary_path / "selected.parquet"
        pd.DataFrame(rows).to_parquet(geometry_path, index=False)
        methods: dict[str, object] = {}
        for method in ("A1", "A2"):
            method_dir = temporary_path / method
            relink_geometry(
                geometry_path=geometry_path,
                output_dir=method_dir,
                method=method,
                frame_start=0,
                frame_end=102,
                max_jump_mm=8.0,
                max_jump_hard_cap_mm=80.0,
                max_gap_frames=100,
                velocity_weight=0.25,
                motion_history_frames=5,
            )
            direct = pd.read_csv(method_dir / f"{method.lower()}_direct.csv")
            last = direct.loc[direct.frame.eq(102)].copy()
            methods[method] = {
                "last_frame_rows": last[
                    [
                        "frame",
                        "fish_id",
                        "X",
                        "identity_status",
                        "identity_segment",
                    ]
                ].to_dict(orient="records"),
                "identity_segments": sorted(
                    int(value) for value in direct.identity_segment.unique()
                ),
                "status_values": sorted(
                    str(value) for value in direct.identity_status.unique()
                ),
            }

    safe = all(
        method["identity_segments"] == [0, 1]
        and all(
            row["identity_segment"] == 1
            and row["identity_status"] == "new_segment_after_max_gap_reseed"
            for row in method["last_frame_rows"]
        )
        for method in methods.values()
    )
    source_path = (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "reconstruction_3d"
        / "mouth_ablation.py"
    )
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    finding = {
        "audit": "A1/A2 long-gap state retirement and reseed segment semantics",
        "mouth_ablation_sha256": source_hash,
        "setup": {
            "frames": [0, 102],
            "max_gap_frames": 100,
            "meaning": (
                "At frame 102, all states are older than the allowed gap and "
                "the production linker retires then re-seeds them."
            ),
        },
        "methods": methods,
        "expected_safe_semantics": (
            "A reseed after exceeding max_gap_frames must start a new identity "
            "segment or remain explicitly unresolved."
        ),
        "result": "PASS" if safe else "FAIL",
        "finding": (
            "Both A1 and A2 expose the post-retirement reseed as "
            "identity_segment=1 and mark the boundary explicitly."
            if safe
            else (
                "At least one method silently asserted continuity across a "
                "gap that exceeded the configured identity memory."
            )
        ),
    }
    (output_root / "a1_a2_long_gap_segment_audit.json").write_text(
        json.dumps(finding, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(finding, indent=2))


if __name__ == "__main__":
    main()
