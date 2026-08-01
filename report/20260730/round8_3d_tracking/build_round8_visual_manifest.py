from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": str(path.relative_to(root)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-root", required=True, type=Path)
    parser.add_argument("--quality-json", required=True, type=Path)
    parser.add_argument("--windows-csv", required=True, type=Path)
    parser.add_argument("--windows-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    media = sorted(
        path
        for path in args.visual_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mp4", ".png"}
    )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "tag": "round8_260729_snn031_m6",
        "status": "review",
        "selection_policy": (
            "A stratified minimum set was rendered from the 61-window queue: "
            "K1 residual extreme, K2 ambiguous-extra/chunk-boundary interval, "
            "and K3 close-contact interval. Seven evenly spaced frames cover "
            "each 31-frame clip. The exact K1 residual-max frame was also "
            "rendered because frame-step 2 did not include that odd frame. "
            "Whole-run trajectory PNGs are context-only and are not treated as "
            "identity diagnostics."
        ),
        "source_artifacts": {
            "quality_json": {
                "path": str(args.quality_json),
                "sha256": sha256(args.quality_json),
            },
            "risk_windows_csv": {
                "path": str(args.windows_csv),
                "sha256": sha256(args.windows_csv),
            },
            "risk_windows_json": {
                "path": str(args.windows_json),
                "sha256": sha256(args.windows_json),
            },
        },
        "rendered_media": [artifact(path, args.visual_root) for path in media],
        "visual_decisions": {
            "20251225_K_1_1_FCMC": {
                "target": "high_prelink_residual",
                "center_frame": 198209,
                "residual_mm2": 9579.57515782122,
                "observation_mode": "three_view",
                "sampled_contact_sheet_frames": [
                    198194,
                    198198,
                    198204,
                    198208,
                    198214,
                    198220,
                    198224,
                ],
                "exact_frame_rendered": 198209,
                "decision": "review_observation_failure_not_obvious_id_swap",
                "evidence": (
                    "IDs 13/14 remain geometrically plausible through the "
                    "sample. At exact frame 198209, gray UNASSIGNED fish-like "
                    "detections occur in top and side2 and final ID 6 has no "
                    "direct rendered skeleton. Its saved centroid is finite "
                    "but is_interpolated=True; IDs 13/14 are direct. The "
                    "extreme was handled conservatively by interpolation. At "
                    "frame 198224 all three centroid rows are direct and "
                    "finite; orange 14 is hidden by green 13 in the chosen 3D "
                    "projection, not omitted."
                ),
            },
            "20251225_K_1_2_FCMC": {
                "target": "ambiguous_extra_candidate_and_chunk_boundary",
                "center_frame": 174305,
                "sampled_contact_sheet_frames": [
                    174290,
                    174294,
                    174300,
                    174304,
                    174310,
                    174316,
                    174320,
                ],
                "decision": "review_selected_geometry_plausible_extra_filtered",
                "evidence": (
                    "Gray UNASSIGNED extra predictions are visible early in "
                    "the window. Selected IDs 11/8 are very close in top, "
                    "side1, side2, and 3D while ID 9 remains isolated. Their "
                    "sampled ordering is continuous and the extra is filtered; "
                    "this does not prove biological identity through contact."
                ),
            },
            "20251225_K_1_3_FCMC": {
                "target": "low_top_pair_separation",
                "center_frame": 96446,
                "sampled_contact_sheet_frames": [
                    96431,
                    96435,
                    96441,
                    96445,
                    96451,
                    96457,
                    96461,
                ],
                "decision": "review_close_pair_continuous_in_sample",
                "evidence": (
                    "IDs 10/11 preserve relative ordering through the sampled "
                    "close contact in all three views and 3D with no visible "
                    "discontinuity or gray extra. This short sample is not "
                    "biological-identity proof for the whole recording."
                ),
            },
        },
        "limitations": [
            (
                "The visual set is stratified risk sampling, not exhaustive "
                "review of all 61 windows or all 348600 frames."
            ),
            (
                "Whole-run trajectory plots can show occupancy and gross "
                "bounds but cannot diagnose smooth identity swaps."
            ),
            (
                "No labeled ground-truth identity or pose annotations were "
                "available, so visual plausibility is not supervised accuracy."
            ),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256(args.output),
                "media_count": len(media),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
