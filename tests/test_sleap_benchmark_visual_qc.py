from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scripts.sleap.benchmark_visual_qc import (
    FrameEvidence,
    PoseInstance,
    ReviewDecision,
    ReviewStore,
    draw_pose_overlay,
    filter_sort_rows,
    load_manifest,
    load_tabular_predictions,
    main,
)


def _decision(frame_key: str, label: str, notes: str = "") -> ReviewDecision:
    return ReviewDecision(
        frame_key=frame_key,
        experiment="K_1_2",
        view="side1_cam",
        frame_idx=100,
        review_label=label,
        notes=notes,
        annotator="tester",
        updated_at_utc="2026-07-24T00:00:00Z",
    )


def test_review_store_atomic_csv_json_resume_and_update(tmp_path: Path) -> None:
    source = tmp_path / "source.pkg.slp"
    source.write_bytes(b"source must remain untouched")
    before = source.read_bytes()
    csv_path = tmp_path / "manual_visual_qc.csv"
    json_path = tmp_path / "manual_visual_qc.json"

    store = ReviewStore(
        csv_path,
        json_path=json_path,
        source_provenance={"source_slp": str(source)},
    )
    store.set(_decision("frame-a", "model_error", "tail is wrong"))
    store.set(_decision("frame-b", "looks_correct"))
    store.set(_decision("frame-a", "annotation_error", "GT tail is wrong"))

    assert source.read_bytes() == before
    assert store.reviewed_count() == 2
    resumed = ReviewStore(csv_path, json_path=json_path)
    assert resumed.get("frame-a").review_label == "annotation_error"
    assert resumed.get("frame-a").notes == "GT tail is wrong"
    assert resumed.get("frame-b").review_label == "looks_correct"
    assert list(pd.read_csv(csv_path)["frame_key"]) == ["frame-a", "frame-b"]
    sidecar = json.loads(json_path.read_text(encoding="utf-8"))
    assert sidecar["counts"] == {
        "reviewed": 2,
        "saved_rows": 2,
        "with_notes": 1,
    }


def test_long_prediction_table_parses_partial_points_and_aggregates_flags(
    tmp_path: Path,
) -> None:
    path = tmp_path / "matches.csv"
    pd.DataFrame(
        [
            {
                "frame_key": "side2-97340",
                "round": "round7",
                "gt_index": 0,
                "pred_index": 2,
                "match_basis": "mouth",
                "is_partial": True,
                "mouth_error_px": 2.5,
                "tail_error_px": "",
                "gt_points": json.dumps([[10.0, 20.0], [None, None]]),
                "pred_points": json.dumps([[12.0, 21.5], [40.0, 50.0]]),
            },
            {
                "frame_key": "side2-97340",
                "round": "round7",
                "gt_index": 1,
                "pred_index": "",
                "status": "miss",
                "match_basis": "unmatchable",
                "gt_mouth_x": 60,
                "gt_mouth_y": 70,
            },
        ]
    ).to_csv(path, index=False)

    loaded = load_tabular_predictions(path)
    evidence = loaded["side2-97340"]
    assert len(evidence.gt) == 2
    assert len(evidence.predictions["round7"]) == 1
    assert evidence.gt[0].mouth == (10.0, 20.0)
    assert evidence.gt[0].tail is None
    assert evidence.predictions["round7"][0].tail == (40.0, 50.0)
    assert evidence.model_metrics["round7"]["is_partial"] is True
    assert evidence.model_metrics["round7"]["is_miss"] is True
    assert evidence.model_metrics["round7"]["mouth_error_px"] == 2.5
    assert evidence.model_metrics["round7"]["match_bases"] == [
        "mouth",
        "unmatchable",
    ]


def test_filter_sort_supports_failures_partial_and_largest_errors() -> None:
    manifest = pd.DataFrame(
        [
            {
                "frame_key": "a",
                "experiment": "K_1_2",
                "view": "side1_cam",
                "frame_idx": 10,
                "difficulty": "ordinary",
                "_manifest_order": 0,
            },
            {
                "frame_key": "b",
                "experiment": "K_1_2",
                "view": "side1_cam",
                "frame_idx": 20,
                "difficulty": "partial/occluded",
                "_manifest_order": 1,
            },
            {
                "frame_key": "c",
                "experiment": "K_1_2",
                "view": "side1_cam",
                "frame_idx": 30,
                "difficulty": "crossing",
                "_manifest_order": 2,
            },
        ]
    )
    evidence = {
        "a": FrameEvidence(
            gt=[],
            predictions={},
            model_metrics={"round8": {"tail_error_px": 9.0}},
        ),
        "b": FrameEvidence(
            gt=[],
            predictions={},
            model_metrics={"round8": {"tail_error_px": 2.0, "is_partial": True}},
        ),
        "c": FrameEvidence(
            gt=[],
            predictions={},
            model_metrics={
                "round8": {
                    "tail_error_px": 45.2,
                    "is_swap": True,
                    "is_shared_tail": True,
                }
            },
        ),
    }
    decisions = {"a": _decision("a", "looks_correct")}

    assert filter_sort_rows(
        manifest, evidence, decisions, filter_name="partial"
    ) == [1]
    assert filter_sort_rows(
        manifest, evidence, decisions, filter_name="swaps"
    ) == [2]
    assert filter_sort_rows(
        manifest, evidence, decisions, filter_name="shared_tails"
    ) == [2]
    assert filter_sort_rows(
        manifest, evidence, decisions, filter_name="unreviewed"
    ) == [1, 2]
    assert filter_sort_rows(
        manifest,
        evidence,
        decisions,
        sort_name="tail_error_desc",
    ) == [2, 0, 1]


def test_manifest_aliases_and_overlay_render_without_display(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "stable_frame_key": "K_1_2|side1|97219",
                "experiment": "K_1_2",
                "view": "side1_cam",
                "frame_index": 97219,
                "source_path": "raw.mp4",
            }
        ]
    ).to_csv(manifest_path, index=False)
    manifest = load_manifest(manifest_path)
    assert manifest.loc[0, "frame_key"] == "K_1_2|side1|97219"
    assert manifest.loc[0, "frame_idx"] == 97219
    assert manifest.loc[0, "video_path"] == "raw.mp4"

    image = np.zeros((80, 120, 3), dtype=np.uint8)
    rendered = draw_pose_overlay(
        image,
        gt_instances=[PoseInstance((20, 25), (50, 25), "0")],
        predictions=[PoseInstance((22, 25), (80, 50), "0")],
        model="round8",
    )
    assert rendered.shape == image.shape
    assert np.count_nonzero(rendered) > 0
    output = tmp_path / "headless_overlay.png"
    assert cv2.imwrite(str(output), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
    assert output.stat().st_size > 0


def test_preflight_accepts_evaluator_matches_schema_without_display(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "benchmark_manifest_v1.csv"
    pd.DataFrame(
        [
            {
                "frame_key": "pilot|side2|97340",
                "experiment": "K_1_2",
                "view": "side2_cam",
                "video_path": "not-opened-during-preflight.mp4",
                "frame_idx": 97340,
            }
        ]
    ).to_csv(manifest_path, index=False)
    matches_path = tmp_path / "comparison_matches.csv"
    pd.DataFrame(
        [
            {
                "round": "round8",
                "view": "side",
                "frame_key": "pilot|side2|97340",
                "gt_index": 0,
                "pred_index": 0,
                "status": "matched",
                "match_basis": "mouth",
                "match_distance_px": 1.2,
                "is_partial": True,
                "mouth_error_px": 1.2,
                "tail_error_px": "",
                "is_swap": "",
            }
        ]
    ).to_csv(matches_path, index=False)

    review_csv = tmp_path / "reviews.csv"
    assert main(
        [
            "--manifest",
            str(manifest_path),
            "--predictions",
            str(matches_path),
            "--review-csv",
            str(review_csv),
            "--preflight",
        ]
    ) == 0
    assert review_csv.exists()
    assert review_csv.with_suffix(".json").exists()
    provenance = json.loads(
        review_csv.with_suffix(".json").read_text(encoding="utf-8")
    )["source_provenance"]
    assert provenance["manifest"]["sha256"]
    assert provenance["predictions"][0]["sha256"]


def test_headless_frame_export_does_not_require_qt(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    raw = np.full((70, 100, 3), 35, dtype=np.uint8)
    cv2.imwrite(str(image_path), raw)
    frame_key = "pilot|side1|97219"
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "frame_key": frame_key,
                "experiment": "K_1_2",
                "view": "side1_cam",
                "image_path": str(image_path),
                "frame_idx": 97219,
                "cohort": "held_out",
                "difficulty": "partial",
                "node_visibility": "partial",
            }
        ]
    ).to_csv(manifest_path, index=False)
    predictions_path = tmp_path / "matches.csv"
    pd.DataFrame(
        [
            {
                "frame_key": frame_key,
                "round": model,
                "gt_index": 0,
                "pred_index": 0,
                "status": "matched",
                "match_basis": "both",
                "mouth_error_px": 1.0,
                "tail_error_px": tail_error,
                "gt_points": json.dumps([[25, 25], [65, 30]]),
                "pred_points": json.dumps([[26, 25], pred_tail]),
            }
            for model, tail_error, pred_tail in (
                ("round7", 4.0, [68, 32]),
                ("round8", 45.2, [45, 62]),
            )
        ]
    ).to_csv(predictions_path, index=False)
    export_path = tmp_path / "comparison.png"

    assert main(
        [
            "--manifest",
            str(manifest_path),
            "--predictions",
            str(predictions_path),
            "--review-csv",
            str(tmp_path / "review.csv"),
            "--export-frame-key",
            frame_key,
            "--export-path",
            str(export_path),
        ]
    ) == 0
    exported = cv2.imread(str(export_path))
    assert exported is not None
    assert exported.shape[0] > raw.shape[0]
    assert exported.shape[1] == 2 * raw.shape[1]
