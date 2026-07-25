from __future__ import annotations

import math
from pathlib import Path

import pytest

from scripts.sleap.run_topdown_optimization_training import (
    CANDIDATE_BY_NAME,
    PHASE4_CONTROLLED_NAMES,
    PRIMARY_CANDIDATES,
    _write_prepared_text,
    build_candidate_config,
    jitter_fraction,
)


def base_config(kind: str) -> dict:
    head_configs = {
        "centroid": {"confmaps": {"sigma": 2.5}},
        "centered_instance": {"confmaps": {"sigma": 2.5}},
    }
    return {
        "data_config": {
            "use_existing_imgs": False,
            "preprocessing": {"crop_size": 512},
            "augmentation_config": {
                "geometric": {
                    "rotation_min": -180.0 if kind == "centered_instance" else -15.0,
                    "rotation_max": 180.0 if kind == "centered_instance" else 15.0,
                    "rotation_p": 1.0,
                    "scale_min": 0.7 if kind == "centered_instance" else 0.9,
                    "scale_max": 1.3 if kind == "centered_instance" else 1.1,
                    "scale_p": 1.0,
                    "translate_width": 0.0,
                    "translate_height": 0.0,
                    "translate_p": None,
                }
            },
        },
        "model_config": {"head_configs": head_configs},
        "trainer_config": {
            "seed": None,
            "run_name": "base",
            "ckpt_dir": "runs",
            "resume_ckpt_path": None,
            "train_data_loader": {
                "batch_size": 16 if kind == "centered_instance" else 8
            },
            "train_steps_per_epoch": 999,
            "early_stopping": {
                "stop_training_on_plateau": False,
                "patience": 10,
            },
            "lr_scheduler": {
                "reduce_lr_on_plateau": {"patience": 5}
            },
            "model_ckpt": {"save_top_k": 1, "save_last": False},
            "online_hard_keypoint_mining": {"online_mining": True},
        },
    }


LABEL_SUMMARY = {
    "frames": 2764,
    "frames_all_anchor": 2287,
    "frames_any_anchor": 2764,
    "instances": 8293,
    "instances_nonempty": 8250,
    "instances_anchor": 7764,
    "instances_missing_anchor": 529,
}


def test_jitter_fraction_is_measured_on_oversized_crop() -> None:
    fraction = jitter_fraction(8, 448)

    assert fraction * int(448 * math.sqrt(2.0)) == pytest.approx(8.0)


def test_centroid_candidate_changes_sigma_and_plateau() -> None:
    candidate = CANDIDATE_BY_NAME["side_centroid_sigma20_strictframe_p40"]

    config, derived = build_candidate_config(
        base_config("centroid"),
        candidate,
        output_root=Path("out"),
        label_summary=LABEL_SUMMARY,
    )

    assert config["model_config"]["head_configs"]["centroid"]["confmaps"][
        "sigma"
    ] == 2.0
    assert config["trainer_config"]["early_stopping"] == {
        "stop_training_on_plateau": True,
        "patience": 40,
    }
    assert config["trainer_config"]["train_steps_per_epoch"] == math.ceil(
        2287 / 8
    )
    assert derived["centroid_anchor_policy"] == "frame"


def test_more_anchor_candidate_uses_all_frames_with_a_finite_mouth() -> None:
    candidate = CANDIDATE_BY_NAME[
        "side_centroid_sigma20_moreanchors_experimental_p40"
    ]

    config, derived = build_candidate_config(
        base_config("centroid"),
        candidate,
        output_root=Path("out"),
        label_summary=LABEL_SUMMARY,
    )

    assert config["trainer_config"]["train_steps_per_epoch"] == math.ceil(
        2764 / 8
    )
    assert derived["centroid_anchor_policy"] == "instance"


def test_centered_instance_control_changes_only_training_guardrails() -> None:
    candidate = CANDIDATE_BY_NAME["side_centered_instance_control_p40"]

    config, derived = build_candidate_config(
        base_config("centered_instance"),
        candidate,
        output_root=Path("out"),
        label_summary=LABEL_SUMMARY,
    )

    geometric = config["data_config"]["augmentation_config"]["geometric"]
    assert config["data_config"]["preprocessing"]["crop_size"] == 512
    assert (geometric["rotation_min"], geometric["rotation_max"]) == (
        -180.0,
        180.0,
    )
    assert config["trainer_config"]["online_hard_keypoint_mining"][
        "online_mining"
    ] is True
    assert config["trainer_config"]["train_steps_per_epoch"] == math.ceil(
        8250 / 16
    )
    assert derived["control"] is True


def test_centered_instance_primary_recipe() -> None:
    candidate = CANDIDATE_BY_NAME[
        "side_centered_instance_opt_jitter8_p40"
    ]

    config, derived = build_candidate_config(
        base_config("centered_instance"),
        candidate,
        output_root=Path("out"),
        label_summary=LABEL_SUMMARY,
    )

    geometric = config["data_config"]["augmentation_config"]["geometric"]
    assert config["data_config"]["preprocessing"]["crop_size"] == 448
    assert (geometric["rotation_min"], geometric["rotation_max"]) == (
        -90.0,
        90.0,
    )
    assert geometric["translate_p"] == 1.0
    assert geometric["translate_width"] * int(448 * math.sqrt(2.0)) == (
        pytest.approx(8.0)
    )
    assert config["trainer_config"]["online_hard_keypoint_mining"][
        "online_mining"
    ] is False
    assert config["trainer_config"]["train_steps_per_epoch"] == math.ceil(
        7764 / 16
    )
    assert derived["centered_instance_anchor_policy"] == "strict"
    assert derived["jitter_implementation"] == "post_affine_translation"
    assert "post-affine pure translation" in config["description"]


def test_phase4_pair_is_optional_and_differs_only_by_ohkm_and_name() -> None:
    primary_names = {candidate.name for candidate in PRIMARY_CANDIDATES}
    assert not (primary_names & PHASE4_CONTROLLED_NAMES)

    off = CANDIDATE_BY_NAME[
        "side_centered_instance_crop448_rot90_jitter0_"
        "strictmouth_ohkmoff_p40"
    ]
    on = CANDIDATE_BY_NAME[
        "side_centered_instance_crop448_rot90_jitter0_"
        "strictmouth_ohkmon_p40"
    ]
    off_config, off_derived = build_candidate_config(
        base_config("centered_instance"),
        off,
        output_root=Path("out"),
        label_summary=LABEL_SUMMARY,
    )
    on_config, on_derived = build_candidate_config(
        base_config("centered_instance"),
        on,
        output_root=Path("out"),
        label_summary=LABEL_SUMMARY,
    )

    for config, derived, expected_ohkm in (
        (off_config, off_derived, False),
        (on_config, on_derived, True),
    ):
        geometric = config["data_config"]["augmentation_config"]["geometric"]
        assert config["data_config"]["preprocessing"]["crop_size"] == 448
        assert (geometric["rotation_min"], geometric["rotation_max"]) == (
            -90.0,
            90.0,
        )
        assert geometric["translate_width"] == 0.0
        assert geometric["translate_height"] == 0.0
        assert geometric["translate_p"] == 0.0
        assert config["trainer_config"]["online_hard_keypoint_mining"][
            "online_mining"
        ] is expected_ohkm
        assert config["trainer_config"]["train_steps_per_epoch"] == math.ceil(
            7764 / 16
        )
        assert derived["jitter_implementation"] == "none"
        assert derived["online_hard_keypoint_mining"] is expected_ohkm

    def flattened(value, prefix=""):
        if isinstance(value, dict):
            output = {}
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else key
                output.update(flattened(child, child_prefix))
            return output
        return {prefix: value}

    off_flat = flattened(off_config)
    on_flat = flattened(on_config)
    differences = {
        key
        for key in off_flat
        if off_flat[key] != on_flat[key]
    }
    assert differences == {
        "trainer_config.run_name",
        "trainer_config.online_hard_keypoint_mining.online_mining",
    }


def test_prepared_artifacts_are_immutable_without_explicit_override(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    _write_prepared_text(path, "first\n", overwrite_prepared=False)
    _write_prepared_text(path, "first\n", overwrite_prepared=False)

    with pytest.raises(FileExistsError, match="differs"):
        _write_prepared_text(path, "second\n", overwrite_prepared=False)

    _write_prepared_text(path, "second\n", overwrite_prepared=True)
    assert path.read_text(encoding="utf-8") == "second\n"
