from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.analysis.female_sex_gt_annotator as annotator_module
from scripts.analysis.female_sex_gt import (ANNOTATION_COLUMNS,
                                            LEGACY_ANNOTATION_COLUMNS,
                                            SCHEMA_VERSION,
                                            TRACKLET_ANNOTATION_COLUMNS,
                                            V11_ANNOTATION_COLUMNS,
                                            AnnotationLedger,
                                            candidate_tracklet_context,
                                            deterministic_annotation_id,
                                            geometry_consistency,
                                            hit_test_sleap_pose,
                                            load_and_merge_candidate_csvs,
                                            make_annotation_row,
                                            manual_jump_candidate,
                                            match_geometry_to_instances,
                                            normalize_candidate_table,
                                            queue_status,
                                            read_annotation_ledger,
                                            selection_from_click)
from scripts.common.local_paths import REPO_ROOT

EXPERIMENT = "20251225_K_1_2_FCMC"


def _free_selection(x: float = 12.0, y: float = 18.0) -> dict:
    return selection_from_click(
        experiment=EXPERIMENT,
        view="side1_cam",
        frame=5,
        click_xy=(x, y),
        instances=None,
        gate_px=20,
    )


def _row(
    *,
    frame: int = 5,
    annotator: str = "alice",
    status: str = "confirmed",
    notes: str = "",
    candidate: dict | None = None,
    selected: bool = True,
) -> dict:
    return make_annotation_row(
        experiment=EXPERIMENT,
        frame=frame,
        source_fps=166,
        annotator=annotator,
        gt_status=status,
        side1_selection=_free_selection() if selected else None,
        side2_selection=None,
        candidate=candidate or manual_jump_candidate(EXPERIMENT, frame),
        video_paths={
            "side1_cam": "side1.mp4",
            "side2_cam": "side2.mp4",
        },
        slp_paths={
            "side1_cam": "side1.slp",
            "side2_cam": "side2.slp",
        },
        notes=notes,
    )


def test_three_fish_focus_uses_geometry_matches_and_ignores_extra_detection():
    instances = np.array(
        [
            [[100, 100], [80, 100]],
            [[900, 450], [880, 450]],  # reflection/extra detection
            [[300, 180], [280, 180]],
            [[500, 260], [480, 260]],
        ],
        dtype=float,
    )
    points = annotator_module._three_fish_focus_points(
        instances,
        {0: "ID_1", 2: "ID_2", 3: "ID_3"},
        {
            "ID_1": np.array([90, 100]),
            "ID_2": np.array([290, 180]),
            "ID_3": np.array([490, 260]),
        },
    )
    assert points.shape == (9, 2)
    assert points[:, 0].max() == 500
    assert not np.any(np.all(points == [900, 450], axis=1))


def test_three_fish_focus_requires_exactly_three_without_geometry():
    three = np.array(
        [
            [[100, 100], [80, 100]],
            [[300, 180], [280, 180]],
            [[500, 260], [480, 260]],
        ],
        dtype=float,
    )
    assert annotator_module._three_fish_focus_points(three).shape == (6, 2)
    assert not len(annotator_module._three_fish_focus_points(three[:2]))


def test_focus_transform_adds_context_and_centers_fish_region():
    transform = annotator_module._focus_transform(
        image_size=(1000, 500),
        viewport_size=(500, 400),
        points=np.array([[400, 200], [600, 300]], dtype=float),
    )
    assert transform is not None
    zoom, center = transform
    assert center == pytest.approx((500, 250))
    assert 1.7 < zoom < 1.8


def test_anchored_scroll_keeps_source_pixel_under_pointer():
    scroll = annotator_module._anchored_scroll_value(
        source_coordinate=400,
        source_extent=1000,
        scaled_extent=2000,
        viewport_coordinate=175,
        viewport_extent=600,
    )
    assert scroll == 625
    assert 400 * 2000 / 1000 - scroll == 175


@pytest.mark.parametrize("alias", ["frame", "center_frame", "center", "jump_frame"])
def test_normalizes_each_candidate_frame_alias(alias):
    result = normalize_candidate_table(
        pd.DataFrame({alias: [7], "candidate_id": ["c1"]}),
        experiment=EXPERIMENT,
        max_frame=20,
    )
    assert result.loc[0, "frame"] == 7
    metadata = json.loads(result.loc[0, "candidate_metadata_json"])
    assert metadata[0]["frame_column"] == alias


def test_merges_candidate_files_and_retains_all_metadata(tmp_path):
    first = tmp_path / "motion.csv"
    second = tmp_path / "crossings.csv"
    pd.DataFrame(
        {
            "frame": [9],
            "candidate_id": ["motion-9"],
            "source": ["motion"],
            "priority": [2],
            "custom_metric": [0.25],
        }
    ).to_csv(first, index=False)
    pd.DataFrame(
        {
            "jump_frame": [9],
            "candidate_id": ["cross-9"],
            "source": ["crossing"],
            "reason": ["close"],
            "reviewer_hint": ["side1"],
        }
    ).to_csv(second, index=False)

    merged = load_and_merge_candidate_csvs(
        [first, second], experiment=EXPERIMENT, max_frame=20
    )
    assert len(merged) == 1
    assert merged.loc[0, "candidate_ids"] == "motion-9;cross-9"
    assert merged.loc[0, "candidate_sources"] == "motion;crossing"
    metadata = json.loads(merged.loc[0, "candidate_metadata_json"])
    assert metadata[0]["priority"] == 2
    assert metadata[0]["extra"]["custom_metric"] == 0.25
    assert metadata[1]["reason"] == "close"
    assert metadata[1]["extra"]["reviewer_hint"] == "side1"


@pytest.mark.parametrize(
    ("table", "message"),
    [
        (pd.DataFrame({"frame": [1], "experiment": ["wrong"]}), "expected"),
        (pd.DataFrame({"frame": [1.5]}), "integral"),
        (pd.DataFrame({"frame": [-1]}), "non-negative"),
        (pd.DataFrame({"frame": [21]}), "out of range"),
    ],
)
def test_rejects_invalid_candidate_rows(table, message):
    with pytest.raises(ValueError, match=message):
        normalize_candidate_table(table, experiment=EXPERIMENT, max_frame=20)


def test_annotation_ids_and_csv_sort_order_are_deterministic(tmp_path):
    assert deterministic_annotation_id(EXPERIMENT, 8, "alice") == deterministic_annotation_id(
        EXPERIMENT, 8, "alice"
    )
    assert deterministic_annotation_id(EXPERIMENT, 8, "alice") != deterministic_annotation_id(
        EXPERIMENT, 8, "bob"
    )
    ledger = AnnotationLedger(tmp_path / "gt.csv", tmp_path / "audit.jsonl")
    ledger.upsert(_row(frame=10, annotator="bob"))
    ledger.upsert(_row(frame=2, annotator="bob"))
    ledger.upsert(_row(frame=2, annotator="alice"))
    saved = read_annotation_ledger(tmp_path / "gt.csv")
    assert saved[["frame", "annotator"]].values.tolist() == [
        ["2", "alice"],
        ["2", "bob"],
        ["10", "bob"],
    ]
    assert list(saved.columns) == ANNOTATION_COLUMNS


def test_atomic_upsert_never_duplicates_canonical_key(tmp_path):
    csv_path = tmp_path / "gt.csv"
    ledger = AnnotationLedger(csv_path, tmp_path / "audit.jsonl")
    ledger.upsert(_row(notes="first"))
    ledger.upsert(_row(notes="edited"))
    saved = read_annotation_ledger(csv_path)
    assert len(saved) == 1
    assert saved.loc[0, "notes"] == "edited"
    assert not csv_path.with_suffix(".csv.tmp").exists()


def test_edit_and_undo_audit_preserve_old_and_new_values(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    ledger = AnnotationLedger(tmp_path / "gt.csv", audit_path)
    ledger.upsert(_row(status="confirmed", notes="positive"))
    ledger.upsert(_row(status="ambiguous", notes="unclear"))
    restored = ledger.undo()
    assert restored["gt_status"] == "confirmed"
    assert ledger.get(EXPERIMENT, 5, "alice")["notes"] == "positive"

    records = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert [record["action"] for record in records] == ["add", "edit", "undo"]
    assert records[1]["old_row"]["gt_status"] == "confirmed"
    assert records[1]["new_row"]["gt_status"] == "ambiguous"
    assert records[2]["old_row"]["gt_status"] == "ambiguous"
    assert records[2]["new_row"]["gt_status"] == "confirmed"


def test_sleap_pose_hit_testing_is_deterministic_and_keeps_free_click():
    instances = np.array(
        [
            [[0.0, 0.0], [10.0, 0.0]],
            [[0.0, 2.0], [10.0, 2.0]],
        ]
    )
    index, distance = hit_test_sleap_pose(instances, (5.0, 1.8), gate_px=3)
    assert index == 1
    assert distance == pytest.approx(0.2)
    tied_index, _ = hit_test_sleap_pose(instances, (5.0, 1.0), gate_px=3)
    assert tied_index == 0
    selection = selection_from_click(
        experiment=EXPERIMENT,
        view="side2_cam",
        frame=4,
        click_xy=(100.0, 100.0),
        instances=instances,
        gate_px=5,
    )
    assert selection["selection_status"] == "free_click"
    assert selection["click_x"] == 100.0
    assert selection["detection_key"] == ""


def test_geometry_matching_is_one_to_one_and_consistency_keeps_disagreement():
    instances = np.array(
        [
            [[0.0, 0.0], [2.0, 0.0]],
            [[8.0, 0.0], [10.0, 0.0]],
        ]
    )
    matches = match_geometry_to_instances(
        {"G1": (1.0, 0.1), "G2": (8.9, 0.0), "far": (100.0, 100.0)},
        instances,
        gate_px=3,
    )
    assert matches == {0: "G1", 1: "G2"}
    assert geometry_consistency("G1", "") == ("one_view", "G1")
    assert geometry_consistency("G1", "G1") == ("agree", "G1")
    assert geometry_consistency("G1", "G2") == ("disagree", "")
    assert geometry_consistency("", "") == ("unmapped", "")


def test_confirmed_decision_requires_one_side_view_selection():
    with pytest.raises(ValueError, match="at least one side-view selection"):
        _row(status="confirmed", selected=False)
    ambiguous = _row(status="ambiguous", selected=False)
    assert ambiguous["sex_label"] == ""


def test_structured_scene_quality_separates_human_gt_from_tracker_evidence():
    row = make_annotation_row(
        experiment=EXPERIMENT,
        frame=5,
        source_fps=166,
        annotator="alice",
        gt_status="confirmed",
        side1_selection=_free_selection(),
        side2_selection=None,
        candidate=manual_jump_candidate(EXPERIMENT, 5),
        video_paths={"side1_cam": "one.mp4", "side2_cam": "two.mp4"},
        slp_paths={"side1_cam": "one.slp", "side2_cam": "two.slp"},
        evidence_quality="human_only",
        observation_issue_flags=[
            "shared_tail_or_node",
            "close_contact",
            "shared_tail_or_node",
        ],
    )
    assert row["sex_label"] == "F"
    assert row["evidence_quality"] == "human_only"
    assert row["observation_issue_flags"] == "close_contact;shared_tail_or_node"


def test_resume_row_and_candidate_queue_status_are_restored(tmp_path):
    candidate = normalize_candidate_table(
        pd.DataFrame({"frame": [5], "candidate_id": ["c5"], "source": ["motion"]}),
        experiment=EXPERIMENT,
        max_frame=10,
    ).iloc[0].to_dict()
    ledger = AnnotationLedger(tmp_path / "gt.csv", tmp_path / "audit.jsonl")
    ledger.upsert(_row(candidate=candidate, status="ambiguous", notes="resume me"))

    reopened = AnnotationLedger(tmp_path / "gt.csv", tmp_path / "audit.jsonl")
    restored = reopened.get(EXPERIMENT, 5, "alice")
    assert restored["notes"] == "resume me"
    assert queue_status(
        candidate,
        reopened.rows,
        experiment=EXPERIMENT,
        annotator="alice",
    ) == "ambiguous"


def test_manual_frame_jump_records_manual_candidate_provenance():
    row = _row(frame=73, candidate=manual_jump_candidate(EXPERIMENT, 73))
    assert row["frame"] == 73
    assert row["candidate_frame"] == 73
    assert row["candidate_sources"] == "manual_jump"


def test_candidate_geometry_mapping_is_promoted_to_explicit_tracklet_gt():
    candidate = normalize_candidate_table(
        pd.DataFrame(
            {
                "frame": [5],
                "candidate_id": ["clean-5"],
                "source": ["clean_tracklet_three_view"],
                "geometry_tracklet_map_json": [
                    json.dumps({"G1": "R3D:G1:S1", "G2": "R3D:G2:S4", "G3": "R3D:G3:S2"})
                ],
                "geometry_source_mode": ["refractive_geometry_segments"],
            }
        ),
        experiment=EXPERIMENT,
        max_frame=20,
    ).iloc[0].to_dict()
    mapping, source_mode = candidate_tracklet_context(candidate)
    assert mapping["G2"] == "R3D:G2:S4"
    assert source_mode == "refractive_geometry_segments"
    instances = np.array([[[0.0, 0.0], [10.0, 0.0]]])
    selection = selection_from_click(
        experiment=EXPERIMENT,
        view="side1_cam",
        frame=5,
        click_xy=(5.0, 0.0),
        instances=instances,
        gate_px=10,
        instance_geometry_ids={0: "G2"},
    )
    row = make_annotation_row(
        experiment=EXPERIMENT,
        frame=5,
        source_fps=166,
        annotator="alice",
        gt_status="confirmed",
        side1_selection=selection,
        side2_selection=selection,
        candidate=candidate,
        video_paths={"side1_cam": "one.mp4", "side2_cam": "two.mp4"},
        slp_paths={"side1_cam": "one.slp", "side2_cam": "two.slp"},
    )
    assert row["side1_tracklet_id"] == "R3D:G2:S4"
    assert row["side2_tracklet_id"] == "R3D:G2:S4"
    assert row["tracklet_id_consistency"] == "agree"
    assert row["consensus_tracklet_id"] == "R3D:G2:S4"
    assert row["candidate_geometry_source_mode"] == "refractive_geometry_segments"


def test_legacy_1_0_ledger_is_upgraded_to_tracklet_schema(tmp_path):
    modern = _row()
    legacy = {column: modern[column] for column in LEGACY_ANNOTATION_COLUMNS}
    legacy["schema_version"] = "1.0"
    path = tmp_path / "legacy.csv"
    pd.DataFrame([legacy], columns=LEGACY_ANNOTATION_COLUMNS).to_csv(path, index=False)
    upgraded = read_annotation_ledger(path)
    assert list(upgraded.columns) == ANNOTATION_COLUMNS
    assert upgraded.loc[0, "schema_version"] == SCHEMA_VERSION
    assert upgraded.loc[0, "consensus_tracklet_id"] == ""
    assert TRACKLET_ANNOTATION_COLUMNS <= set(upgraded.columns)
    assert upgraded.loc[0, "evidence_quality"] == "unreviewed"


def test_previous_1_1_ledger_is_upgraded_to_scene_quality_schema(tmp_path):
    modern = _row()
    previous = {column: modern[column] for column in V11_ANNOTATION_COLUMNS}
    previous["schema_version"] = "1.1"
    path = tmp_path / "previous.csv"
    pd.DataFrame([previous], columns=V11_ANNOTATION_COLUMNS).to_csv(path, index=False)

    upgraded = read_annotation_ledger(path)

    assert list(upgraded.columns) == ANNOTATION_COLUMNS
    assert upgraded.loc[0, "schema_version"] == SCHEMA_VERSION
    assert upgraded.loc[0, "evidence_quality"] == "unreviewed"
    assert upgraded.loc[0, "observation_issue_flags"] == ""


def test_annotator_build_candidates_invokes_bound_selector(tmp_path, monkeypatch):
    captured = {}

    def fake_selector(**kwargs):
        captured.update(kwargs)
        return {"selected_candidate_count": 17, "eligible_frame_count": 42}

    monkeypatch.setattr(annotator_module, "run_candidate_selector", fake_selector)
    output = tmp_path / "candidates.csv"
    args = SimpleNamespace(
        build_candidates=True,
        manual_only=False,
        geometry_csv=tmp_path / "refractive_3d.csv",
        calibration_model_dir=tmp_path / "calibration",
        candidate_output_csv=output,
        candidate_summary_json=tmp_path / "summary.json",
        candidate_target_count=17,
        candidate_min_frame_spacing=25,
        match_gate_px=80.0,
        candidate_min_view_separation_px=40.0,
        candidate_border_margin_px=20.0,
        candidate_max_local_jump_mm=8.0,
        candidate_seed=3,
        experiment=EXPERIMENT,
        candidate_clean_tracklets_csv=None,
        reconstruction_provenance_json=tmp_path / "provenance.json",
        geometry_id_column="fish_id",
        slp_dir=tmp_path / "slp",
        video_dir=tmp_path / "video",
        calibration_backend="refractive",
        candidate_csv=[],
    )
    annotator_module._prepare_candidate_queue(args, annotator_module.build_parser())
    assert args.candidate_csv == [output]
    assert captured["clean_tracklets_csv"] is None
    assert captured["config"].target_count == 17
    assert captured["config"].max_local_jump_mm == 8.0


def test_annotator_reuses_default_candidate_queue(tmp_path):
    output = tmp_path / "candidates.csv"
    output.write_text("frame\n5\n", encoding="utf-8")
    args = SimpleNamespace(
        build_candidates=False,
        manual_only=False,
        candidate_csv=[],
        candidate_output_csv=output,
    )
    annotator_module._prepare_candidate_queue(args, annotator_module.build_parser())
    assert args.candidate_csv == [output]


def test_offscreen_cli_smoke_test_renders_navigates_and_saves(tmp_path):
    if importlib.util.find_spec("PyQt5") is None:
        pytest.skip("PyQt5 GUI extra is not installed")
    csv_path = tmp_path / "smoke_gt.csv"
    command = [
        sys.executable,
        "-m",
        "scripts.analysis.female_sex_gt_annotator",
        "synthetic_experiment",
        "--annotator",
        "qt-smoke",
        "--annotations-csv",
        str(csv_path),
        "--smoke-test",
    ]
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    saved = read_annotation_ledger(csv_path)
    assert len(saved) == 1
    assert saved.loc[0, "gt_status"] == "confirmed"
    assert saved.loc[0, "side1_selection_status"] == "sleap_linked"
    assert saved.loc[0, "candidate_sources"] == "synthetic"
