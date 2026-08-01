from __future__ import annotations

from pathlib import Path

import pytest

from scripts.tracking import slp_merge


def _touch(path: Path, payload: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _raw_experiment(tmp_path: Path, total_frames: int = 10) -> Path:
    experiment = tmp_path / "raw" / "20251225_K_1_1_FCMC"
    camera = experiment / "top_cam"
    camera.mkdir(parents=True)
    _touch(camera / "cam0_top.mp4")
    (camera / "metadata.csv").write_text(
        "\n".join(
            [
                '"cameraName","cam0"',
                '"videoFilename","cam0_top.mp4"',
                f'"totalFrames","{total_frames}"',
            ]
        ),
        encoding="utf-8",
    )
    return experiment


def test_find_and_validate_chunk_ranges(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks"
    later = _touch(chunks / "top_cam.frames_5-9.slp")
    earlier = _touch(chunks / "top_cam.frames_0-4.slp")

    found = slp_merge.find_chunks(chunks, "top_cam")

    assert found == [earlier, later]
    assert slp_merge.validate_chunk_ranges(
        found, "top_cam", expected_total_frames=10
    ) == [(0, 4), (5, 9)]


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (["top_cam.frames_1-9.slp"], "starts at 1"),
        (
            ["top_cam.frames_0-4.slp", "top_cam.frames_6-9.slp"],
            "chunk gap",
        ),
        (
            ["top_cam.frames_0-5.slp", "top_cam.frames_5-9.slp"],
            "chunk overlap",
        ),
        (["top_cam.frames_0-8.slp"], "ends at 8"),
    ],
)
def test_validate_chunk_ranges_rejects_bad_coverage(
    tmp_path: Path, names: list[str], message: str
) -> None:
    paths = [_touch(tmp_path / name) for name in names]

    with pytest.raises(ValueError, match=message):
        slp_merge.validate_chunk_ranges(
            slp_merge.find_chunks(tmp_path, "top_cam"),
            "top_cam",
            expected_total_frames=10,
        )


def test_merge_experiment_uses_round_tag_and_separates_frame_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_experiment = _raw_experiment(tmp_path, total_frames=10)
    output_root = tmp_path / "sleap"
    tagged = output_root / raw_experiment.name / "round8_test"
    chunk = _touch(tagged / "top_cam.frames_0-9.slp")
    calls: list[tuple[list[Path], Path, str | None]] = []

    def fake_merge(
        chunk_paths: list[Path],
        output_path: Path,
        canonical_video_path: str | None = None,
    ) -> int:
        calls.append((chunk_paths, output_path, canonical_video_path))
        _touch(output_path, b"merged")
        return 7

    monkeypatch.setattr(slp_merge, "merge_view_chunks", fake_merge)

    manifest = slp_merge.merge_experiment(
        experiment_name=raw_experiment.name,
        sleap_output_dir=output_root,
        raw_data_dir=raw_experiment.parent,
        sleap_subdir="round8_test",
        views=("top_cam",),
    )

    assert calls == [
        (
            [chunk],
            tagged / "top_cam.slp",
            str(raw_experiment / "top_cam" / "cam0_top.mp4"),
        )
    ]
    status = manifest["sleap"]["views"]["top_cam"]
    assert status["total_frames"] == 10
    assert status["labeled_frames"] == 7
    assert manifest["sleap"]["sleap_subdir"] == "round8_test"


def test_fresh_tagged_merge_is_loaded_for_labeled_frame_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_experiment = _raw_experiment(tmp_path, total_frames=10)
    output_root = tmp_path / "sleap"
    tagged = output_root / raw_experiment.name / "round8_test"
    _touch(tagged / "top_cam.frames_0-9.slp")
    merged = _touch(tagged / "top_cam.slp", b"new merged")
    merged.touch()

    monkeypatch.setattr(slp_merge, "count_labeled_frames", lambda path: 6)
    monkeypatch.setattr(
        slp_merge,
        "merge_view_chunks",
        lambda *args, **kwargs: pytest.fail("fresh merge should be skipped"),
    )

    manifest = slp_merge.merge_experiment(
        experiment_name=raw_experiment.name,
        sleap_output_dir=output_root,
        raw_data_dir=raw_experiment.parent,
        sleap_subdir="round8_test",
        views=("top_cam",),
    )

    status = manifest["sleap"]["views"]["top_cam"]
    assert status["total_frames"] == 10
    assert status["labeled_frames"] == 6
    assert status["skipped_reason"] == "merged file already fresh"


@pytest.mark.parametrize("value", ["../round8", "round8/tag", ".", "..", ""])
def test_merge_experiment_rejects_invalid_round_tag(
    tmp_path: Path, value: str
) -> None:
    raw_experiment = _raw_experiment(tmp_path)

    with pytest.raises((ValueError, FileNotFoundError)):
        slp_merge.merge_experiment(
            experiment_name=raw_experiment.name,
            sleap_output_dir=tmp_path / "sleap",
            raw_data_dir=raw_experiment.parent,
            sleap_subdir=value,
            views=("top_cam",),
        )
