from __future__ import annotations

from pathlib import Path

from scripts.sleap.run_topdown_optimization_evaluation import (
    build_command,
    build_jobs,
    summarize_benchmark,
)


def candidate_runs(tmp_path: Path) -> dict[str, Path]:
    return {
        name: tmp_path / name
        for name in (
            "centroid_sigma25",
            "centroid_sigma20",
            "centroid_sigma15",
            "instance_control",
            "instance_jitter8",
            "instance_jitter16",
        )
    }


def test_build_jobs_has_component_sets_and_complete_pairing_grid(tmp_path):
    jobs = {job.name: job for job in build_jobs(candidate_runs(tmp_path))}

    assert len(jobs["centroid_only"].models) == 3
    assert len(jobs["oracle_centered_instance"].models) == 3
    assert len(jobs["full_topdown_grid"].models) == 9
    assert jobs["centroid_only"].strict_anchor_eligible_only
    assert jobs["oracle_centered_instance"].strict_anchor_eligible_only
    assert not jobs["full_topdown_grid"].strict_anchor_eligible_only
    assert jobs["full_topdown_grid"].models["sigma20__jitter8"] == (
        tmp_path / "centroid_sigma20",
        tmp_path / "instance_jitter8",
    )


def test_build_command_uses_shared_cache_and_strict_component_filter(tmp_path):
    jobs = {job.name: job for job in build_jobs(candidate_runs(tmp_path))}
    command = build_command(
        job=jobs["centroid_only"],
        benchmark=tmp_path / "benchmark.csv",
        output_dir=tmp_path / "evaluation",
        batch_size=4,
        device="cuda",
    )

    assert "--reuse-existing-predictions" in command
    assert "--strict-centroid-eligible-only" in command
    assert command.count("--model") == 3
    assert command[command.index("--evaluation-mode") + 1] == "centroid_only"

    full_command = build_command(
        job=jobs["full_topdown_grid"],
        benchmark=tmp_path / "benchmark.csv",
        output_dir=tmp_path / "evaluation",
        batch_size=4,
        device="cuda",
    )
    assert "--strict-centroid-eligible-only" not in full_command
    assert full_command.count("--model") == 9


def test_summarize_benchmark_groups_both_side_views(tmp_path):
    benchmark = tmp_path / "benchmark.csv"
    benchmark.write_text(
        "view,strict_centroid_eligible,cohort,experiment,difficulty\n"
        "top_cam,True,pilot,k1,ordinary\n"
        "side1_cam,True,pilot,k1,ordinary\n"
        "side2_cam,False,stress,k2,close_contact\n",
        encoding="utf-8",
    )

    summary = summarize_benchmark(benchmark)
    assert summary["all_rows"] == 3
    assert summary["side_rows"] == 2
    assert summary["side_strict_centroid_eligible_rows"] == 1
    assert summary["side_by_cohort"] == {"pilot": 1, "stress": 1}
