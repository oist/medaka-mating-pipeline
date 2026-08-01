"""Finalize a bounded-stage marker after its detached outer launcher has closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


REPO = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
ROOTS = {
    "validation": REPORT / "validation_postfreeze",
    "heldout": REPORT / "heldout_postfreeze",
    "critical": REPORT / "critical_descriptive_postfreeze",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO)).replace("/", "\\")


def write_json_atomic(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=tuple(ROOTS))
    args = parser.parse_args()
    stage_root = ROOTS[args.stage]
    finalized_path = stage_root / "STAGE_FINALIZED.json"
    if finalized_path.exists():
        raise FileExistsError(finalized_path)

    marker_path = stage_root / "STAGE_COMPLETED.json"
    prepared_path = stage_root / "STAGE_PREPARED.json"
    preauth_path = stage_root / "prelaunch_auth.json"
    postauth_path = stage_root / "postcompletion_auth.json"
    launch_path = stage_root / "DETACHED_LAUNCH.json"
    stdout_path = stage_root / "detached_launcher.stdout.log"
    stderr_path = stage_root / "detached_launcher.stderr.log"
    marker = load_json(marker_path)
    prepared = load_json(prepared_path)
    preauth = load_json(preauth_path)
    postauth = load_json(postauth_path)
    launch = load_json(launch_path)

    assert marker["status"] == "completed_and_authenticated"
    assert marker["stage"] == args.stage
    assert marker["freeze"]["sha256"] == prepared["freeze_sha256"]
    assert marker["runner"]["sha256"] == prepared["runner_sha256"]
    assert marker["prelaunch_auth"]["sha256"] == sha256(preauth_path)
    assert marker["postcompletion_auth"]["sha256"] == sha256(postauth_path)
    assert preauth["bindings"] == postauth["bindings"]
    assert launch["runner_sha256"] == prepared["runner_sha256"]
    assert launch["prepared_manifest_sha256"] == sha256(prepared_path)

    final_stdout_sha256 = sha256(stdout_path)
    final_stderr_sha256 = sha256(stderr_path)
    final_stdout_bytes = stdout_path.stat().st_size
    final_stderr_bytes = stderr_path.stat().st_size
    assert final_stderr_bytes == 0
    stdout_result = json.loads(stdout_path.read_text(encoding="utf-8"))
    assert stdout_result["path"] == relative(marker_path)
    assert stdout_result["sha256"] == sha256(marker_path)

    mismatches = []
    unchanged = {}
    for path_string, record in marker["artifact_inventory"].items():
        path = REPO / path_string
        assert path.is_file(), path
        actual = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if actual != record:
            mismatches.append(
                {
                    "path": path_string,
                    "recorded": record,
                    "actual": actual,
                }
            )
        else:
            unchanged[path_string] = actual
    assert mismatches == [], mismatches
    assert marker["outer_launcher_files_excluded_until_finalization"] == [
        relative(stdout_path),
        relative(stderr_path),
    ]
    assert relative(stdout_path) not in marker["artifact_inventory"]
    assert relative(stderr_path) not in marker["artifact_inventory"]

    current_inventory = {}
    for path in sorted(stage_root.rglob("*")):
        if (
            path.is_file()
            and path != finalized_path
            and not path.name.startswith(".STAGE_FINALIZED.json.")
        ):
            current_inventory[relative(path)] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    assert current_inventory[relative(marker_path)]["sha256"] == sha256(marker_path)
    assert current_inventory[relative(stdout_path)]["sha256"] == final_stdout_sha256
    assert current_inventory[relative(stderr_path)]["sha256"] == final_stderr_sha256

    finalization = {
        "schema_version": "round8_bounded_postfreeze_finalized_v1",
        "status": "finalized_and_fully_authenticated",
        "stage": args.stage,
        "role": marker["role"],
        "finalized_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "finalizer": {
            "path": relative(Path(__file__)),
            "sha256": sha256(Path(__file__)),
        },
        "bindings": {
            "completed_marker": {
                "path": relative(marker_path),
                "sha256": sha256(marker_path),
            },
            "freeze": marker["freeze"],
            "runner": marker["runner"],
            "prepared_manifest": {
                "path": relative(prepared_path),
                "sha256": sha256(prepared_path),
            },
            "prelaunch_auth": {
                "path": relative(preauth_path),
                "sha256": sha256(preauth_path),
            },
            "postcompletion_auth": {
                "path": relative(postauth_path),
                "sha256": sha256(postauth_path),
            },
        },
        "outer_launcher": {
            "command": launch["command"],
            "pid": launch["pid"],
            "launched_at": launch["launched_at"],
            "working_directory": launch["working_directory"],
            "exit_code": 0,
            "exit_code_evidence": (
                "launcher process closed, stderr is empty, and final stdout is "
                "valid JSON binding the completed marker path and SHA-256"
            ),
            "launch_record": {
                "path": relative(launch_path),
                "sha256": sha256(launch_path),
            },
            "stdout": {
                "path": relative(stdout_path),
                "bytes": final_stdout_bytes,
                "sha256": final_stdout_sha256,
            },
            "stderr": {
                "path": relative(stderr_path),
                "bytes": final_stderr_bytes,
                "sha256": final_stderr_sha256,
            },
        },
        "completed_marker_inventory_validation": {
            "recorded_artifact_count": marker["artifact_inventory_count"],
            "checked_artifact_count": len(marker["artifact_inventory"]),
            "unchanged_artifact_count": len(unchanged),
            "unchanged_artifacts": unchanged,
            "mismatches": mismatches,
            "outer_launcher_logs_excluded_until_finalization": {
                "stdout": {
                    "path": relative(stdout_path),
                    "bytes": final_stdout_bytes,
                    "sha256": final_stdout_sha256,
                },
                "stderr": {
                    "path": relative(stderr_path),
                    "bytes": final_stderr_bytes,
                    "sha256": final_stderr_sha256,
                },
            },
            "result": "PASS",
            "interpretation": (
                "Every marker-inventoried artifact is unchanged. The detached "
                "launcher logs were intentionally excluded while open and are "
                "authenticated here after the launcher closed."
            ),
        },
        "current_stage_inventory": current_inventory,
        "current_stage_inventory_count": len(current_inventory),
        "inventory_scope": (
            "all current files under the stage root, including STAGE_COMPLETED.json "
            "and closed outer-launch logs, excluding only STAGE_FINALIZED.json"
        ),
        "scientific_results_changed_during_finalization": False,
        "selection_or_tuning_performed": False,
        "heldout_or_later_stage_opened": False,
    }
    write_json_atomic(finalized_path, finalization)
    print(
        json.dumps(
            {"path": relative(finalized_path), "sha256": sha256(finalized_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
