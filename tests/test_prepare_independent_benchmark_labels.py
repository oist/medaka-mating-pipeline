from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.sleap.multiview_annotation_gui import (
    FishKeypoints,
    clone_fish_keypoints,
    copy_labels_provenance,
    fish_have_annotations,
)
from scripts.sleap.prepare_independent_benchmark_labels import (
    add_round8_selection_exposures,
    balanced_quotas,
    expanded_interval,
    finalize_locked_package,
    intervals_overlap,
    reserve_new_output_dir,
    select_future_queue_events,
    select_uniform_unexposed_frames,
    sha256_file,
    validate_locked_annotation_labels,
)


def test_balanced_40_frame_quotas() -> None:
    assert balanced_quotas(40, ["K1", "K2", "K3"]) == {
        "K1": 14,
        "K2": 13,
        "K3": 13,
    }


def test_uniform_selection_is_deterministic_unexposed_and_disjoint() -> None:
    kwargs = {
        "experiment": "K1",
        "count": 6,
        "exposed_frames": {1000, 3000, 7000},
        "total_frames": 12000,
        "guard_frames": 100,
        "seed": 42,
    }
    first = select_uniform_unexposed_frames(**kwargs)
    second = select_uniform_unexposed_frames(**kwargs)
    assert first == second
    assert len(first) == 6
    for row in first:
        frame = int(row["frame_idx"])
        assert min(abs(frame - exposed) for exposed in kwargs["exposed_frames"]) > 100
    intervals = [expanded_interval(row, 100) for row in first]
    assert not any(
        intervals_overlap(intervals[i], intervals[j])
        for i in range(len(intervals))
        for j in range(i + 1, len(intervals))
    )


def test_future_queue_selects_one_row_per_nonoverlapping_event(
    tmp_path: Path,
) -> None:
    rows = []
    for event, start, end, frames in [
        ("event_a", 1000, 1010, [1001, 1005]),
        ("event_b", 2000, 2010, [2002]),
        ("event_c", 2075, 2080, [2077]),
        ("event_d", 3000, 3010, [3005]),
    ]:
        for frame in frames:
            rows.append(
                {
                    "experiment": "K2",
                    "view": "top_cam",
                    "frame_idx": frame,
                    "event_id": event,
                    "event_start_frame": start,
                    "event_end_frame": end,
                    "difficulty": "close_contact_or_crossing",
                    "source_category": "test",
                    "future_lock_eligible": True,
                }
            )
    queue = tmp_path / "queue.csv"
    pd.DataFrame(rows).to_csv(queue, index=False)
    selected = select_future_queue_events(
        queue,
        experiment="K2",
        count=3,
        exposed_frames={500},
        guard_frames=50,
        seed=7,
    )
    assert len(selected) == 3
    assert len({row["event_id"] for row in selected}) == 3
    intervals = [expanded_interval(row, 50) for row in selected]
    assert not any(
        intervals_overlap(intervals[i], intervals[j])
        for i in range(len(intervals))
        for j in range(i + 1, len(intervals))
    )


def test_annotation_cache_copy_is_independent() -> None:
    source = [FishKeypoints()]
    source[0].mouth["top_cam"] = (10.0, 20.0)
    copied = clone_fish_keypoints(source)
    assert fish_have_annotations(copied)
    copied[0].mouth["top_cam"] = (30.0, 40.0)
    assert source[0].mouth["top_cam"] == (10.0, 20.0)
    assert copied[0].mouth["top_cam"] == (30.0, 40.0)


def test_round8_exposure_uses_pilot_rows_only(tmp_path: Path) -> None:
    selection = tmp_path / "selection.csv"
    pd.DataFrame(
        [
            {
                "experiment": "K1",
                "frame": 100,
                "annotation_batch": "pilot",
            },
            {
                "experiment": "K1",
                "frame": 200,
                "annotation_batch": "expansion",
            },
        ]
    ).to_csv(selection, index=False)
    exposed = {"K1": {50}}
    counts = add_round8_selection_exposures(exposed, selection, ["K1"])
    assert counts == {"K1": 1}
    assert exposed == {"K1": {50, 100}}


def test_output_directory_reservation_refuses_existing_target(
    tmp_path: Path,
) -> None:
    output = tmp_path / "independent40-v1"
    reserve_new_output_dir(output)
    assert output.is_dir()
    with pytest.raises(FileExistsError, match="refusing existing output"):
        reserve_new_output_dir(output)


def test_package_provenance_copy_is_deep_and_independent() -> None:
    class Labels:
        provenance = {"selection": {"seed": 20260724}}

    copied = copy_labels_provenance(Labels())
    copied["selection"]["seed"] = 1
    assert Labels.provenance["selection"]["seed"] == 20260724


class _Node:
    def __init__(self, name: str) -> None:
        self.name = name


class _Skeleton:
    nodes = [_Node("mouth"), _Node("tail")]


class _Video:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.source_video = None


class _Instance:
    def __init__(self, points: np.ndarray) -> None:
        self._points = points

    def numpy(self) -> np.ndarray:
        return self._points.copy()


class _LabeledFrame:
    def __init__(
        self,
        *,
        experiment: str,
        frame_idx: int,
        view: str,
        instances: list[_Instance],
    ) -> None:
        filenames = {
            "top_cam": "cam0_top.mp4",
            "side1_cam": "cam1_side.mp4",
            "side2_cam": "cam2_side.mp4",
        }
        self.video = _Video(
            f"C:/raw/{experiment}/{view}/{filenames[view]}"
        )
        self.frame_idx = frame_idx
        self.instances = instances


class _Labels:
    def __init__(self, labeled_frames: list[_LabeledFrame]) -> None:
        self.skeletons = [_Skeleton()]
        self.labeled_frames = labeled_frames
        self.provenance = {"contains_model_predictions": False}


def _locked_manifest_and_labels() -> tuple[pd.DataFrame, _Labels]:
    manifest_rows = []
    labeled_frames = []
    for frame_idx in range(40):
        experiment = f"K{frame_idx % 3 + 1}"
        manifest_rows.append(
            {
                "frame_key": f"li1_{frame_idx:02d}",
                "experiment": experiment,
                "frame_idx": frame_idx + 1000,
            }
        )
        for view in ("top_cam", "side1_cam", "side2_cam"):
            instances = [
                _Instance(
                    np.array(
                        [
                            [10.0 + fish, 20.0 + fish],
                            [30.0 + fish, 40.0 + fish],
                        ]
                    )
                )
                for fish in range(3)
            ]
            labeled_frames.append(
                _LabeledFrame(
                    experiment=experiment,
                    frame_idx=frame_idx + 1000,
                    view=view,
                    instances=instances,
                )
            )
    return pd.DataFrame(manifest_rows), _Labels(labeled_frames)


def test_locked_validator_emits_complete_selection_ledger() -> None:
    manifest, labels = _locked_manifest_and_labels()
    ledger, counts = validate_locked_annotation_labels(labels, manifest)
    assert counts == {
        "source_frames": 40,
        "view_frames": 120,
        "fish_per_view": 3,
        "instances": 360,
        "finite_mouths": 360,
        "finite_tails": 360,
    }
    assert len(ledger) == 360
    assert ledger["frame_key"].nunique() == 40
    assert set(ledger["view"]) == {"top_cam", "side1_cam", "side2_cam"}


def test_locked_validator_rejects_missing_fish() -> None:
    manifest, labels = _locked_manifest_and_labels()
    labels.labeled_frames[0].instances.pop()
    with pytest.raises(ValueError, match="exactly 3 fish"):
        validate_locked_annotation_labels(labels, manifest)


def test_locked_validator_rejects_nonfinite_nodes() -> None:
    manifest, labels = _locked_manifest_and_labels()
    labels.labeled_frames[0].instances[0]._points[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite mouth/tail"):
        validate_locked_annotation_labels(labels, manifest)


def test_locked_finalizer_copies_input_and_writes_hashed_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, labels = _locked_manifest_and_labels()
    output_dir = tmp_path / "independent40-v1"
    output_dir.mkdir()
    manifest_path = output_dir / "independent_annotation_40_manifest.csv"
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    selection_provenance_path = (
        output_dir / "independent_annotation_40_provenance.json"
    )
    selection_provenance_path.write_text(
        json.dumps(
            {
                "manifest": {"sha256": sha256_file(manifest_path)},
                "contains_model_predictions": False,
                "contains_keypoint_coordinates": False,
            }
        ),
        encoding="utf-8",
    )
    annotated_path = output_dir / "independent_annotation_40_manual.pkg.slp"
    annotated_bytes = b"manual-package-remains-immutable"
    annotated_path.write_bytes(annotated_bytes)
    monkeypatch.setitem(
        sys.modules,
        "sleap_io",
        SimpleNamespace(load_slp=lambda *_args, **_kwargs: labels),
    )

    result = finalize_locked_package(
        annotated_path=annotated_path,
        manifest_path=manifest_path,
        selection_provenance_path=selection_provenance_path,
        output_dir=output_dir,
    )

    locked_path = Path(result["locked_package"])
    ledger_path = Path(result["selection_ledger"])
    frozen_provenance = json.loads(
        Path(result["provenance"]).read_text(encoding="utf-8")
    )
    assert annotated_path.read_bytes() == annotated_bytes
    assert locked_path.read_bytes() == annotated_bytes
    assert len(pd.read_csv(ledger_path)) == 360
    assert frozen_provenance["locked_package"]["sha256"] == sha256_file(
        locked_path
    )
    assert frozen_provenance["selection_ledger"]["sha256"] == sha256_file(
        ledger_path
    )
