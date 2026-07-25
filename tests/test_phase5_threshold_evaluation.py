from __future__ import annotations

from pathlib import Path

from scripts.sleap.run_phase5_threshold_evaluation import (
    build_single_command,
    run_threshold_strategy,
    suffixed_model_specs,
    threshold_label,
)


def test_threshold_labels_and_model_names_are_cache_safe() -> None:
    assert threshold_label(0.1, 0.25) == "centroid0p1__keypoint0p25"
    assert suffixed_model_specs(
        ["candidate=C:/centroid,C:/instance"],
        centroid_threshold=0.1,
        keypoint_threshold=0.25,
    ) == [
        "candidate__centroid0p1__keypoint0p25="
        "C:/centroid,C:/instance"
    ]


def test_single_command_is_restartable_and_records_both_thresholds(
    tmp_path: Path,
) -> None:
    command = build_single_command(
        model_specs=["candidate=C:/centroid,C:/instance"],
        benchmark=tmp_path / "benchmark.csv",
        output_dir=tmp_path / "evaluation",
        centroid_threshold=0.15,
        keypoint_threshold=0.3,
        max_instances=6,
        batch_size=4,
        device="cuda",
        force=False,
    )
    assert "--run-one" in command
    assert command[command.index("--max-instances") + 1] == "6"
    assert command[command.index("--centroid-thresholds") + 1] == "0.15"
    assert command[command.index("--keypoint-thresholds") + 1] == "0.3"


def test_threshold_strategy_forwards_stage_specific_values(
    monkeypatch,
) -> None:
    import sleap_nn.inference
    import scripts.sleap.run_phase5_threshold_evaluation as phase5

    calls = {}
    sentinel = object()

    def fake_predict(**kwargs):
        calls.update(kwargs)
        return sentinel

    monkeypatch.setattr(sleap_nn.inference, "predict", fake_predict)
    monkeypatch.setattr(
        phase5.evaluator,
        "make_prediction_labels_self_contained",
        lambda labels: labels,
    )
    output = run_threshold_strategy(
        strategy="candidate",
        model_paths=[Path("centroid"), Path("instance")],
        subset_labels="labels",
        max_instances=6,
        batch_size=2,
        device="cpu",
        centroid_threshold=0.15,
        keypoint_threshold=0.25,
    )
    assert output is sentinel
    assert calls["centroid_threshold"] == 0.15
    assert calls["keypoint_threshold"] == 0.25
    assert calls["max_instances"] == 6
