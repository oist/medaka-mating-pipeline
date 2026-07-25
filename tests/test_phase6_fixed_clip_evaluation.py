from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.sleap.run_phase6_fixed_clip_evaluation import (
    DEFAULT_WINDOWS,
    ClipWindow,
    ModelPair,
    Phase6Paths,
    PlannedStep,
    _execute_step,
    build_inference_command,
    build_plan,
    main,
    parse_window,
)


def _pair(root: Path, name: str) -> ModelPair:
    return ModelPair(
        name=name,
        centroid=root / name / "centroid",
        centered_instance=root / name / "centered_instance",
    )


def test_default_windows_are_inclusive_and_preregistered() -> None:
    windows = [parse_window(value) for value in DEFAULT_WINDOWS]
    assert [
        (window.name, window.start, window.end)
        for window in windows
    ] == [
        ("unresolved_contact", 95000, 99000),
        ("non_event_control", 151050, 155050),
    ]


def test_inference_command_forwards_both_stage_thresholds() -> None:
    pair = ModelPair(
        name="candidate",
        centroid=Path("centroid"),
        centered_instance=Path("centered_instance"),
        centroid_threshold=0.15,
        keypoint_threshold=0.25,
    )
    command = build_inference_command(
        python="python",
        video=Path("video.mp4"),
        output_slp=Path("clip.slp"),
        pair=pair,
        window=ClipWindow("event", 95000, 99000),
        batch_size=12,
    )
    assert command[command.index("--centroid-threshold") + 1] == "0.15"
    assert command[command.index("--keypoint-threshold") + 1] == "0.25"
    assert command[command.index("--max-instances") + 1] == "6"
    assert command[command.index("--start-frame") + 1] == "95000"
    assert command[command.index("--end-frame") + 1] == "99000"
    assert "predict" not in command


def test_plan_reuses_top_view_and_forwards_explicit_roots(
    tmp_path: Path,
) -> None:
    paths = Phase6Paths(
        data_root=tmp_path / "data",
        sleap_output_root=tmp_path / "sleap",
        reconstruction_output_root=tmp_path / "reconstruction",
        report_dir=tmp_path / "report",
        calibration_model_dir=tmp_path / "calibration",
        paths_config=tmp_path / "paths.toml",
    )
    windows = [
        ClipWindow("event", 95000, 99000),
        ClipWindow("control", 151050, 155050),
    ]
    top = _pair(tmp_path / "models", "top")
    sides = [
        _pair(tmp_path / "models", name)
        for name in ("baseline", "sigma20", "sigma15")
    ]
    steps = build_plan(
        experiment="experiment",
        windows=windows,
        top_pair=top,
        side_pairs=sides,
        paths=paths,
        python="python",
        batch_size=12,
        device="cuda",
    )

    assert len(steps) == 36
    assert sum(step.name.endswith(":top_inference") for step in steps) == 2
    assert sum(step.name.endswith("_inference") for step in steps) == 14
    assert sum(step.name.endswith(":stage_top") for step in steps) == 6
    assert sum(step.name.endswith(":reconstruct") for step in steps) == 6
    assert sum(step.name.endswith(":render_4panel") for step in steps) == 6
    assert sum(step.name.endswith(":qc") for step in steps) == 2

    event_top = next(
        step
        for step in steps
        if step.name == "event:top_inference"
    )
    staged = [
        step for step in steps if step.name.startswith("event:")
        and step.name.endswith(":stage_top")
    ]
    assert all(
        step.copy_source == event_top.expected_output for step in staged
    )

    reconstruction = next(
        step for step in steps if step.name == "event:baseline:reconstruct"
    )
    command = reconstruction.command
    assert command[command.index("--sleap-output-dir") + 1] == str(
        paths.sleap_output_root
    )
    assert command[command.index("--model-folder") + 1] == str(
        paths.calibration_model_dir
    )
    assert command[command.index("--start-frame") + 1] == "95000"
    assert command[command.index("--end-frame") + 1] == "99000"
    assert command[command.index("--num-candidates") + 1] == "6"
    assert "--use-visible-node-centroid" in command

    render = next(
        step
        for step in steps
        if step.name == "event:baseline:render_4panel"
    )
    render_command = render.command
    assert render_command[
        render_command.index("--frame-step") + 1
    ] == "5"
    assert render_command[
        render_command.index("--display-id-col") + 1
    ] == "final_fish_id"
    assert "--show-unassigned-sleap" in render_command
    assert render_command[
        render_command.index("--start-frame") + 1
    ] == "95000"
    assert render_command[
        render_command.index("--end-frame") + 1
    ] == "99000"


def test_prepare_only_writes_manifest_without_running_commands(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import scripts.sleap.run_phase6_fixed_clip_evaluation as phase6

    def fail_if_called(*args, **kwargs):
        raise AssertionError("prepare-only must not launch a subprocess")

    monkeypatch.setattr(phase6.subprocess, "run", fail_if_called)
    report_dir = tmp_path / "report"
    result = main(
        [
            "--paths-config",
            str(tmp_path / "missing.toml"),
            "--data-root",
            str(tmp_path / "data"),
            "--sleap-output-root",
            str(tmp_path / "sleap"),
            "--reconstruction-output-root",
            str(tmp_path / "reconstruction"),
            "--model-root",
            str(tmp_path / "models"),
            "--calibration-model-dir",
            str(tmp_path / "calibration"),
            "--report-dir",
            str(report_dir),
            "--skip-input-checks",
            "--prepare-only",
        ]
    )
    assert result == 0
    manifest = json.loads(
        (report_dir / "phase6_execution_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["windows"] == [
        {"name": "unresolved_contact", "start": 95000, "end": 99000},
        {"name": "non_event_control", "start": 151050, "end": 155050},
    ]
    assert len(manifest["steps"]) == 36
    assert "centroid_artifacts" in manifest["top_pair"]
    assert "centered_instance_artifacts" in manifest["top_pair"]
    assert {
        artifact["path"] for artifact in manifest["source_artifacts"]
    } >= {
        "scripts/sleap/run_phase6_fixed_clip_evaluation.py",
        "scripts/sleap/run_fixed_clip_topdown_inference.py",
        "scripts/reconstruction_3d/compare_fixed_clip_candidates.py",
    }
    assert manifest["visual_qc"]["frame_step"] == 5
    assert manifest["visual_qc"]["show_unassigned_sleap"]


def test_restartable_step_requires_its_provenance_sidecar(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "clip.slp"
    provenance = tmp_path / "clip.inference_provenance.json"
    output.write_bytes(b"existing output without provenance")
    calls = []

    def fake_run(*_args, **_kwargs):
        calls.append(True)
        provenance.write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    import scripts.sleap.run_phase6_fixed_clip_evaluation as phase6

    monkeypatch.setattr(phase6.subprocess, "run", fake_run)
    step = PlannedStep(
        name="event:top_inference",
        kind="command",
        expected_output=output,
        command=("python", "-m", "inference"),
        provenance_output=provenance,
    )
    assert (
        _execute_step(
            step,
            repo_root=tmp_path,
            log_dir=tmp_path / "logs",
            force=False,
        )
        == 0
    )
    assert calls
