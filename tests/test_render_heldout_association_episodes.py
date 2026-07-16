from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.visualization.render_heldout_association_episodes import (
    Episode,
    MethodSpec,
    RenderSettings,
    build_render_command,
    load_episodes,
    parse_methods,
    render_episodes,
)


def test_episode_validation_sorts_and_rejects_overlap(tmp_path):
    valid = tmp_path / "valid.csv"
    pd.DataFrame(
        [
            {"episode_id": "late", "start_frame": 30, "end_frame": 35, "n_frames": 6},
            {"episode_id": "early", "start_frame": 10, "end_frame": 15, "n_frames": 6},
        ]
    ).to_csv(valid, index=False)
    episodes = load_episodes(valid)
    assert [(item.episode_id, item.start_frame, item.end_frame) for item in episodes] == [
        ("early", 10, 15), ("late", 30, 35)
    ]

    overlap = tmp_path / "overlap.csv"
    pd.DataFrame(
        [
            {"episode_id": "a", "start_frame": 10, "end_frame": 20},
            {"episode_id": "b", "start_frame": 20, "end_frame": 30},
        ]
    ).to_csv(overlap, index=False)
    with pytest.raises(ValueError, match="overlap"):
        load_episodes(overlap)


def test_render_command_preserves_source_range_and_overlay_policy():
    command = build_render_command(
        python_executable="python", experiment="K_1_2",
        episode=Episode("event", 11200, 11400),
        method=MethodSpec("optimized", Path("track.csv"), "id_opt", "optimized"),
        slp_dir=Path("slp"), video_dir=Path("video"), output_path=Path("clip.mp4"),
        ffmpeg="ffmpeg", settings=RenderSettings(arrow_thickness=1),
    )
    assert command[:4] == ["python", "-m", "scripts.visualization.render_qc_4panel_fast", "K_1_2"]
    assert command[command.index("--start-frame") + 1] == "11200"
    assert command[command.index("--end-frame") + 1] == "11400"
    assert command[command.index("--id-column") + 1] == "id_opt"
    assert command[command.index("--arrow-thickness") + 1] == "1"
    assert "--draw-raw-sleap" in command
    assert "--raw-by-index" not in command


def test_dry_run_needs_no_runtime_inputs_and_refuses_overwrite(tmp_path):
    episodes_csv = tmp_path / "episodes.csv"
    pd.DataFrame(
        [
            {"episode_id": "e1", "start_frame": 100, "end_frame": 110},
            {"episode_id": "e2", "start_frame": 300, "end_frame": 320},
        ]
    ).to_csv(episodes_csv, index=False)
    episodes = load_episodes(episodes_csv)
    methods = parse_methods(
        [["baseline", str(tmp_path / "missing_baseline.csv"), "id_final"],
         ["optimized", str(tmp_path / "missing_optimized.csv"), "id_optimized"]]
    )
    output = tmp_path / "renders"
    manifest = render_episodes(
        experiment="K_1_2", episodes=episodes, methods=methods,
        slp_dir=tmp_path / "missing_slp", video_dir=tmp_path / "missing_video",
        output_dir=output, ffmpeg="missing_ffmpeg", dry_run=True,
    )
    assert manifest["status"] == "dry_run"
    assert len(manifest["methods"]) == 2
    assert all(len(method["clips"]) == 2 for method in manifest["methods"])
    assert not list(output.rglob("*.mp4"))
    saved = json.loads((output / "heldout_association_render_manifest.json").read_text())
    assert saved["overlay_policy"]["raw_sleap"] == "all raw instances in gray"
    assert saved["methods"][0]["clips"][1]["source_start_frame"] == 300

    with pytest.raises(FileExistsError, match="--overwrite"):
        render_episodes(
            experiment="K_1_2", episodes=episodes, methods=methods,
            slp_dir=tmp_path / "missing_slp", video_dir=tmp_path / "missing_video",
            output_dir=output, ffmpeg="missing_ffmpeg", dry_run=True,
        )
