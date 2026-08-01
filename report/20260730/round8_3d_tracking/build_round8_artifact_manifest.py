from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, str | int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--visual-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    names = (
        "round8_260729_snn031_m6_inference_manifest.json",
        "round8_3d_quality.json",
        "round8_3d_quality_summary.csv",
        "round8_3d_quality_report.md",
        "round8_visual_qc_windows.csv",
        "round8_visual_qc_windows.json",
        "evaluate_round8_quality.py",
        "select_round8_risk_windows.py",
        "summarize_round8_quality.py",
        "build_round8_visual_manifest.py",
    )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "tag": "round8_260729_snn031_m6",
        "overall_decision": "review",
        "interpretation": (
            "Execution and structural/integrity QC completed successfully. "
            "The result remains review rather than biological-identity pass "
            "because no labeled ground truth was available and smooth "
            "contacts/crossings cannot be excluded by bounded continuity."
        ),
        "core_artifacts": [
            record(args.report_dir / name) for name in names
        ],
        "source_inference_qc": [
            record(path)
            for path in sorted(
                (args.report_dir / "source_inference_qc").glob("*.json")
            )
        ],
        "visual_manifest": record(args.visual_manifest),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256(args.output),
                "core_artifact_count": len(payload["core_artifacts"]),
                "source_qc_count": len(payload["source_inference_qc"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
