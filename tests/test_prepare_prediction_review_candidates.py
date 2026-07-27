from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.sleap.prepare_prediction_review_candidates import (
    build_parser,
    candidate_overlay_provenance,
    merge_prediction_subsets,
    split_positions_by_model,
    validate_distinct_artifact_paths,
    validate_prediction_subset,
)


class _FakeLabels:
    def __init__(self, labeled_frames, *, skeletons=None, provenance=None):
        self.labeled_frames = labeled_frames
        self.skeletons = skeletons or []
        self.provenance = provenance or {}


def _lf(path: str, frame_idx: int):
    backend = SimpleNamespace(source_filename=path)
    video = SimpleNamespace(
        backend=backend,
        filename=path,
        source_video=None,
    )
    return SimpleNamespace(video=video, frame_idx=frame_idx)


def test_split_positions_routes_top_and_both_side_views() -> None:
    labels = _FakeLabels(
        [
            _lf(r"Z:\exp\top_cam\cam0_top.mp4", 1),
            _lf(r"Z:\exp\side1_cam\cam1_side.mp4", 1),
            _lf(r"Z:\exp\side2_cam\cam2_side.mp4", 1),
            _lf(r"Z:\exp\top_cam\cam0_top.mp4", 2),
        ]
    )
    assert split_positions_by_model(labels) == ([0, 3], [1, 2])


def test_split_positions_rejects_unknown_camera() -> None:
    labels = _FakeLabels([_lf(r"Z:\exp\front\camera.mp4", 1)])
    with pytest.raises(ValueError, match="could not classify"):
        split_positions_by_model(labels)


def test_prediction_subset_requires_exact_sparse_frame_order() -> None:
    seed = _FakeLabels(
        [
            _lf(r"Z:\exp\top_cam\cam0_top.mp4", 10),
            _lf(r"Z:\exp\top_cam\cam0_top.mp4", 20),
        ]
    )
    predictions = _FakeLabels(
        [
            _lf(r"Z:\exp\top_cam\cam0_top.mp4", 20),
            _lf(r"Z:\exp\top_cam\cam0_top.mp4", 10),
        ]
    )
    with pytest.raises(ValueError, match="identity/order mismatch"):
        validate_prediction_subset(
            name="top",
            seed_labels=seed,
            positions=[0, 1],
            predictions=predictions,
        )


def test_prediction_subset_accepts_sparse_noncontiguous_positions() -> None:
    seed = _FakeLabels(
        [
            _lf(r"Z:\exp\top_cam\cam0_top.mp4", 10),
            _lf(r"Z:\exp\side1_cam\cam1_side.mp4", 10),
            _lf(r"Z:\exp\top_cam\cam0_top.mp4", 20),
        ]
    )
    predictions = _FakeLabels(
        [
            _lf(r"Z:\exp\top_cam\cam0_top.mp4", 10),
            _lf(r"Z:\exp\top_cam\cam0_top.mp4", 20),
        ]
    )
    validate_prediction_subset(
        name="top",
        seed_labels=seed,
        positions=[0, 2],
        predictions=predictions,
    )


def test_artifact_paths_refuse_overwriting_seed(tmp_path) -> None:
    seed = tmp_path / "seed.slp"
    with pytest.raises(ValueError, match="must be distinct"):
        validate_distinct_artifact_paths(
            {
                "seed_slp": seed,
                "output_slp": tmp_path / "." / "seed.slp",
            }
        )


@pytest.mark.parametrize("argument", ["--max-instances", "--batch-size"])
def test_positive_inference_cli_values_are_validated_early(argument) -> None:
    parser = build_parser()
    required = [
        "--seed-slp", "seed.slp",
        "--top-centroid-model", "top-centroid",
        "--top-centered-instance-model", "top-instance",
        "--side-centroid-model", "side-centroid",
        "--side-centered-instance-model", "side-instance",
        "--output-slp", "out.slp",
        "--provenance-json", "out.json",
        "--cache-dir", "cache",
        argument, "0",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(required)


def test_candidate_provenance_overrides_blank_seed_without_mutating_it() -> None:
    seed = {
        "contains_model_predictions": False,
        "selection": {"seed": 20260724},
    }
    provenance = candidate_overlay_provenance(
        seed,
        max_instances=3,
        batch_size=4,
        device="cuda",
    )
    assert provenance["contains_model_predictions"] is True
    assert provenance["annotation_source"] == (
        "round8_model_candidates_for_human_review"
    )
    assert provenance["candidate_overlay"] == {
        "kind": "model_prediction_for_human_review",
        "authoritative_annotation": False,
        "seed_safety": "separate_overlay",
        "max_instances": 3,
        "batch_size": 4,
        "device": "cuda",
    }
    provenance["selection"]["seed"] = 0
    assert seed == {
        "contains_model_predictions": False,
        "selection": {"seed": 20260724},
    }


def test_merge_prediction_subsets_restores_seed_order(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.sleap.prepare_prediction_review_candidates."
        "make_prediction_labels_self_contained",
        lambda labels: labels,
    )

    class _FakeSioLabels:
        def __init__(self, labeled_frames, skeletons, provenance):
            self.labeled_frames = labeled_frames
            self.skeletons = skeletons
            self.provenance = provenance

    monkeypatch.setitem(
        __import__("sys").modules,
        "sleap_io",
        SimpleNamespace(Labels=_FakeSioLabels),
    )
    seed = _FakeLabels(
        ["seed0", "seed1", "seed2", "seed3"],
        skeletons=["seed-skeleton"],
        provenance={"selection": "locked"},
    )
    top = _FakeLabels(["pred0", "pred3"], skeletons=["prediction-skeleton"])
    side = _FakeLabels(["pred1", "pred2"], skeletons=["prediction-skeleton"])
    merged = merge_prediction_subsets(
        seed_labels=seed,
        top_positions=[0, 3],
        top_predictions=top,
        side_positions=[1, 2],
        side_predictions=side,
    )
    assert merged.labeled_frames == ["pred0", "pred1", "pred2", "pred3"]
    assert merged.skeletons == ["prediction-skeleton"]
    assert merged.provenance == {"selection": "locked"}


def test_merge_prediction_subsets_rejects_missing_positions(monkeypatch) -> None:
    monkeypatch.setitem(
        __import__("sys").modules,
        "sleap_io",
        SimpleNamespace(Labels=object),
    )
    seed = _FakeLabels(["seed0", "seed1"])
    top = _FakeLabels(["pred0"], skeletons=["skeleton"])
    side = _FakeLabels([])
    with pytest.raises(ValueError, match="missing=\\[1\\]"):
        merge_prediction_subsets(
            seed_labels=seed,
            top_positions=[0],
            top_predictions=top,
            side_positions=[],
            side_predictions=None,
        )


def test_merge_prediction_subsets_rejects_duplicate_positions(
    monkeypatch,
) -> None:
    monkeypatch.setitem(
        __import__("sys").modules,
        "sleap_io",
        SimpleNamespace(Labels=object),
    )
    with pytest.raises(ValueError, match="positions contain duplicates"):
        merge_prediction_subsets(
            seed_labels=_FakeLabels(["seed0", "seed1"]),
            top_positions=[0, 0],
            top_predictions=_FakeLabels(
                ["pred0a", "pred0b"],
                skeletons=["skeleton"],
            ),
            side_positions=[],
            side_predictions=None,
        )


def test_merge_prediction_subsets_caps_instances_by_score(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.sleap.prepare_prediction_review_candidates."
        "make_prediction_labels_self_contained",
        lambda labels: labels,
    )

    class _FakeLabeledFrame:
        def __init__(self, video, frame_idx, instances):
            self.video = video
            self.frame_idx = frame_idx
            self.instances = instances

    class _FakeSioLabels:
        def __init__(self, labeled_frames, skeletons, provenance):
            self.labeled_frames = labeled_frames
            self.skeletons = skeletons
            self.provenance = provenance

    monkeypatch.setitem(
        __import__("sys").modules,
        "sleap_io",
        SimpleNamespace(Labels=_FakeSioLabels, LabeledFrame=_FakeLabeledFrame),
    )
    predictions = _FakeLabels(
        [
            _FakeLabeledFrame(
                video="video",
                frame_idx=10,
                instances=[
                    SimpleNamespace(score=0.1),
                    SimpleNamespace(score=0.9),
                    SimpleNamespace(score=0.4),
                ],
            )
        ],
        skeletons=["skeleton"],
    )
    merged = merge_prediction_subsets(
        seed_labels=_FakeLabels(["seed"]),
        top_positions=[0],
        top_predictions=predictions,
        side_positions=[],
        side_predictions=None,
        max_instances=2,
    )
    assert [item.score for item in merged.labeled_frames[0].instances] == [0.9, 0.4]


def test_merge_prediction_subsets_supports_side_only_seed(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.sleap.prepare_prediction_review_candidates."
        "make_prediction_labels_self_contained",
        lambda labels: labels,
    )

    class _FakeSioLabels:
        def __init__(self, labeled_frames, skeletons, provenance):
            self.labeled_frames = labeled_frames
            self.skeletons = skeletons
            self.provenance = provenance

    monkeypatch.setitem(
        __import__("sys").modules,
        "sleap_io",
        SimpleNamespace(Labels=_FakeSioLabels),
    )
    seed = _FakeLabels(
        ["seed0", "seed1"],
        skeletons=["seed-skeleton"],
        provenance={"purpose": "side-only"},
    )
    side = _FakeLabels(["side0", "side1"], skeletons=["side-skeleton"])
    merged = merge_prediction_subsets(
        seed_labels=seed,
        top_positions=[],
        top_predictions=None,
        side_positions=[0, 1],
        side_predictions=side,
    )
    assert merged.labeled_frames == ["side0", "side1"]
    assert merged.skeletons == ["side-skeleton"]
