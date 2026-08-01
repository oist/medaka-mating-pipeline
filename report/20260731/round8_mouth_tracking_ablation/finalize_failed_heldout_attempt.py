"""Seal the failed first held-out attempt without changing any prior artifact."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


REPO = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
FAILED = REPORT / "quarantine_failed_heldout_attempt_runner1f737_20260731_0457"
OUTPUT = FAILED / "FAILED_ATTEMPT_FINALIZED.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO)).replace("/", "\\")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    prepared = FAILED / "STAGE_PREPARED.json"
    preauth = FAILED / "prelaunch_auth.json"
    launch = FAILED / "DETACHED_LAUNCH.json"
    exit_record = FAILED / "DETACHED_LAUNCH_EXIT.json"
    failed_record = (
        FAILED
        / "process_records"
        / "low_top_pair_separation_8_9_b.A5_hints.json"
    )
    failed_stderr = (
        FAILED
        / "low_top_pair_separation_8_9_b"
        / "A5"
        / "hints.stderr.log"
    )
    prior_manifest = FAILED / "FAILED_ATTEMPT_MANIFEST.json"
    prepared_data = load(prepared)
    exit_data = load(exit_record)
    failed_data = load(failed_record)
    assert prepared_data["runner_sha256"] == (
        "1f737fb484d0b81fb52bcd1dc2551aea35ec1137ff3010bb8bc82d487aef4a7d"
    )
    assert exit_data["exit_code"] == 1
    assert failed_data["status"] == "failed"
    assert failed_data["record_name"] == "low_top_pair_separation_8_9_b.A5_hints"
    assert "A4 selected table lacks" in failed_stderr.read_text(encoding="utf-8")
    assert not (FAILED / "STAGE_COMPLETED.json").exists()

    inventory = {}
    for path in sorted(FAILED.rglob("*")):
        if path.is_file() and path != OUTPUT:
            inventory[str(path.relative_to(FAILED)).replace("/", "\\")] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    record = {
        "schema_version": "round8_failed_heldout_attempt_finalized_v1",
        "status": "failed_attempt_preserved_and_authenticated",
        "finalized_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "finalizer": {
            "path": relative(Path(__file__)),
            "sha256": sha256(Path(__file__)),
        },
        "failure": (
            "Frozen A5 hint admission rejected an authenticated A4 scientific "
            "abstention table with zero rows and zero columns."
        ),
        "bindings": {
            "prepared_manifest": {
                "path": relative(prepared),
                "sha256": sha256(prepared),
            },
            "prelaunch_auth": {
                "path": relative(preauth),
                "sha256": sha256(preauth),
            },
            "outer_launch": {
                "path": relative(launch),
                "sha256": sha256(launch),
            },
            "outer_exit": {
                "path": relative(exit_record),
                "sha256": sha256(exit_record),
                "exit_code": 1,
            },
            "failed_process_record": {
                "path": relative(failed_record),
                "sha256": sha256(failed_record),
            },
            "failed_stderr": {
                "path": relative(failed_stderr),
                "sha256": sha256(failed_stderr),
            },
            "prior_inventory_manifest": {
                "path": relative(prior_manifest),
                "sha256": sha256(prior_manifest),
            },
        },
        "inventory": inventory,
        "inventory_count": len(inventory),
        "scientific_completion_marker_written": False,
        "selection_or_tuning_impact": "none",
        "critical_opened": False,
    }
    payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(
        prefix=".FAILED_ATTEMPT_FINALIZED.", suffix=".tmp", dir=FAILED
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, OUTPUT)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({"path": relative(OUTPUT), "sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
