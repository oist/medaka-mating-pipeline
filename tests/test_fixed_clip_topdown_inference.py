from __future__ import annotations

import json
from pathlib import Path

from scripts.sleap.run_fixed_clip_topdown_inference import run_topdown_clip


def test_fixed_clip_wrapper_forwards_independent_thresholds(
    tmp_path: Path,
) -> None:
    output = tmp_path / "clip.slp"
    centroid = tmp_path / "centroid"
    centered_instance = tmp_path / "centered_instance"
    for model_dir in (centroid, centered_instance):
        model_dir.mkdir()
        (model_dir / "best.ckpt").write_bytes(b"checkpoint")
        (model_dir / "training_config.yaml").write_text(
            "model: test\n", encoding="utf-8"
        )
    calls = {}
    sentinel = object()

    def fake_predict(**kwargs):
        calls.update(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"slp")
        return sentinel

    result = run_topdown_clip(
        video=tmp_path / "video.mp4",
        centroid_model=centroid,
        centered_instance_model=centered_instance,
        output_slp=output,
        start_frame=95000,
        end_frame=95002,
        centroid_threshold=0.15,
        keypoint_threshold=0.25,
        max_instances=6,
        batch_size=12,
        device="cuda",
        predict_fn=fake_predict,
    )
    assert result is sentinel
    assert calls["source"] == str(tmp_path / "video.mp4")
    assert calls["frames"] == [95000, 95001, 95002]
    assert calls["centroid_threshold"] == 0.15
    assert calls["keypoint_threshold"] == 0.25
    assert calls["max_instances"] == 6
    assert calls["integral_refinement"] == "integral"
    assert "export_dir" not in calls

    provenance = json.loads(
        output.with_suffix(".inference_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance["checkpoint_only_torch_path"]
    assert not provenance["exported_runtime_used"]
    assert provenance["inclusive_frame_window"]["count"] == 3
    assert len(provenance["centroid_model"]["best.ckpt"]["sha256"]) == 64
    assert (
        provenance["centroid_model"]["best.ckpt"]["sha256"]
        == provenance["centered_instance_model"]["best.ckpt"]["sha256"]
    )
