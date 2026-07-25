from __future__ import annotations

from pathlib import Path

import json

import pytest

from scripts.sleap.run_phase4_ohkm_evaluation import (
    build_phase4_jobs,
    wait_for_training_queue,
)


def test_phase4_jobs_have_complete_two_by_three_grid(
    tmp_path: Path,
) -> None:
    candidates = {
        name: tmp_path / name
        for name in (
            "centroid_sigma15",
            "centroid_sigma20",
            "instance_control",
            "instance_ohkmoff",
            "instance_ohkmon",
        )
    }
    jobs = {
        job.name: job for job in build_phase4_jobs(candidates)
    }

    assert len(jobs["phase4_centroid_only"].models) == 2
    assert len(jobs["phase4_oracle_centered_instance"].models) == 3
    assert len(jobs["phase4_full_topdown_grid"].models) == 6
    assert jobs[
        "phase4_full_topdown_grid"
    ].models["sigma20__ohkmon"] == (
        tmp_path / "centroid_sigma20",
        tmp_path / "instance_ohkmon",
    )
    assert jobs["phase4_centroid_only"].strict_anchor_eligible_only
    assert jobs[
        "phase4_oracle_centered_instance"
    ].strict_anchor_eligible_only
    assert not jobs[
        "phase4_full_topdown_grid"
    ].strict_anchor_eligible_only


def test_training_queue_waiter_accepts_only_clean_completion(
    tmp_path: Path,
) -> None:
    status = tmp_path / "queue_status.json"
    payload = {
        "state": "complete",
        "active": None,
        "completed": ["off", "on"],
        "failed": [],
    }
    status.write_text(json.dumps(payload), encoding="utf-8")
    assert wait_for_training_queue(
        tmp_path,
        poll_seconds=0.001,
        max_wait_hours=0.001,
    ) == payload

    status.write_text(
        json.dumps(
            {
                "state": "complete_with_failures",
                "active": None,
                "completed": ["off"],
                "failed": [{"candidate": "on", "return_code": 1}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="did not complete cleanly"):
        wait_for_training_queue(
            tmp_path,
            poll_seconds=0.001,
            max_wait_hours=0.001,
        )
