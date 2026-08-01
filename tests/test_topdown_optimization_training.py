from __future__ import annotations

import json
import math
import re
from pathlib import Path, PureWindowsPath

import pytest
import torch
import yaml

from scripts.sleap.run_topdown_optimization_training import (
    CANDIDATE_BY_NAME,
    PHASE4_CONTROLLED_NAMES,
    PRIMARY_CANDIDATES,
    _finalize_monitored_checkpoints,
    _preserved_checkpoint_name,
    _select_resume_checkpoint,
    _sha256,
    _snapshot_live_top_checkpoints,
    _write_prepared_text,
    build_candidate_config,
    jitter_fraction,
    prepare_candidates,
    run_candidate,
)


def base_config(kind: str) -> dict:
    head_configs = {
        "centroid": {"confmaps": {"sigma": 2.5}},
        "centered_instance": {"confmaps": {"sigma": 2.5}},
    }
    return {
        "data_config": {
            "use_existing_imgs": False,
            "train_labels_path": ["old_train.pkg.slp"],
            "val_labels_path": ["old_val.pkg.slp"],
            "validation_fraction": 0.0,
            "cache_img_path": "old_cache",
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
            "val_data_loader": {
                "batch_size": 16 if kind == "centered_instance" else 8
            },
            "train_steps_per_epoch": 999,
            "early_stopping": {
                "stop_training_on_plateau": False,
                "patience": 10,
                "min_delta": 1.0e-8,
            },
            "lr_scheduler": {
                "reduce_lr_on_plateau": {
                    "patience": 5,
                    "threshold": 1.0e-6,
                }
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
        "min_delta": 1.0e-8,
    }
    assert config["trainer_config"]["train_steps_per_epoch"] == math.ceil(
        2287 / 8
    )
    assert derived["centroid_anchor_policy"] == "frame"


def test_centroid_batch6_restart_recomputes_steps_per_epoch() -> None:
    candidate = CANDIDATE_BY_NAME[
        "side_centroid_sigma15_strictframe_batch6_restart_p40"
    ]

    config, derived = build_candidate_config(
        base_config("centroid"),
        candidate,
        output_root=Path("out"),
        label_summary=LABEL_SUMMARY,
    )

    trainer = config["trainer_config"]
    assert trainer["train_data_loader"]["batch_size"] == 6
    assert trainer["val_data_loader"]["batch_size"] == 6
    assert trainer["train_steps_per_epoch"] == math.ceil(2287 / 6)
    assert config["model_config"]["head_configs"]["centroid"]["confmaps"][
        "sigma"
    ] == 1.5
    assert derived["batch_size"] == 6
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


def test_centered_instance_strictmouth_control_preserves_geometry() -> None:
    candidate = CANDIDATE_BY_NAME[
        "side_centered_instance_control_strictmouth_p40"
    ]

    config, derived = build_candidate_config(
        base_config("centered_instance"),
        candidate,
        output_root=Path("out"),
        label_summary=LABEL_SUMMARY,
    )

    data = config["data_config"]
    geometric = data["augmentation_config"]["geometric"]
    trainer = config["trainer_config"]
    assert data["preprocessing"]["crop_size"] == 512
    assert (geometric["rotation_min"], geometric["rotation_max"]) == (
        -180.0,
        180.0,
    )
    assert (geometric["scale_min"], geometric["scale_max"]) == (0.7, 1.3)
    assert geometric["translate_width"] == 0.0
    assert geometric["translate_height"] == 0.0
    assert geometric["translate_p"] == 0.0
    assert trainer["train_data_loader"]["batch_size"] == 16
    assert trainer["val_data_loader"]["batch_size"] == 16
    assert trainer["online_hard_keypoint_mining"]["online_mining"] is True
    assert trainer["early_stopping"]["stop_training_on_plateau"] is True
    assert trainer["early_stopping"]["patience"] == 40
    assert trainer["early_stopping"]["min_delta"] == 1.0e-8
    assert trainer["lr_scheduler"]["reduce_lr_on_plateau"]["patience"] == 5
    assert (
        trainer["lr_scheduler"]["reduce_lr_on_plateau"]["threshold"]
        == 1.0e-6
    )
    assert trainer["train_steps_per_epoch"] == math.ceil(7764 / 16)
    assert derived["preserves_control_geometry"] is True
    assert derived["centered_instance_anchor_policy"] == "strict"
    assert derived["jitter_implementation"] == "none"


def test_round9b_uses_side_geometry_and_adds_exact_jitter() -> None:
    candidate = CANDIDATE_BY_NAME[
        "side_centered_instance_round9b_crop448_rot90_jitter8_"
        "strictmouth_ohkmon_p40"
    ]

    inherited = base_config("centered_instance")
    inherited_geometric = inherited["data_config"]["augmentation_config"][
        "geometric"
    ]
    inherited_geometric["scale_min"] = 0.9
    inherited_geometric["scale_max"] = 1.1
    inherited_geometric["scale_p"] = 0.25
    config, derived = build_candidate_config(
        inherited,
        candidate,
        output_root=Path("out"),
        label_summary=LABEL_SUMMARY,
    )

    data = config["data_config"]
    geometric = data["augmentation_config"]["geometric"]
    trainer = config["trainer_config"]
    expected_fraction = 8 / int(448 * math.sqrt(2.0))
    assert data["preprocessing"]["crop_size"] == 448
    assert (geometric["rotation_min"], geometric["rotation_max"]) == (
        -90.0,
        90.0,
    )
    assert (geometric["scale_min"], geometric["scale_max"]) == (0.7, 1.3)
    assert geometric["scale_p"] == 1.0
    assert geometric["translate_width"] == pytest.approx(expected_fraction)
    assert geometric["translate_height"] == pytest.approx(expected_fraction)
    assert geometric["translate_p"] == 1.0
    assert trainer["online_hard_keypoint_mining"]["online_mining"] is True
    assert trainer["model_ckpt"]["save_top_k"] == 3
    assert trainer["model_ckpt"]["save_last"] is True
    assert trainer["model_ckpt"]["monitor"] == "eval/val/mOKS"
    assert trainer["model_ckpt"]["mode"] == "max"
    assert trainer["eval"]["enabled"] is True
    assert trainer["eval"]["frequency"] == 1
    assert config["model_config"]["head_configs"]["centered_instance"][
        "confmaps"
    ]["anchor_part"] == "mouth"
    assert derived["preserves_control_geometry"] is False
    assert derived["jitter_px"] == 8
    assert derived["jitter_implementation"] == "post_affine_translation"
    assert derived["save_top_k"] == 3
    assert derived["save_last"] is True
    assert derived["checkpoint_monitor"] == "eval/val/mOKS"
    assert derived["checkpoint_mode"] == "max"
    assert derived["eval_frequency"] == 1
    assert derived["recovery_checkpoint_every_validation"] is True
    assert derived["centered_instance_anchor_part"] == "mouth"


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
    assert "monitor" not in config["trainer_config"]["model_ckpt"]
    assert "mode" not in config["trainer_config"]["model_ckpt"]
    assert "eval" not in config["trainer_config"]
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


def test_prepare_rewrites_labels_caches_and_both_checkpoint_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_round = tmp_path / "combined_round"
    side_dir = base_round / "side"
    side_dir.mkdir(parents=True)
    train_labels = side_dir / "train_labels.pkg.slp"
    val_labels = side_dir / "val_labels.pkg.slp"
    train_labels.write_bytes(b"train")
    val_labels.write_bytes(b"val")

    checkpoint_by_kind = {}
    for kind in ("centroid", "centered_instance"):
        run_dir = tmp_path / f"{kind}_initial"
        run_dir.mkdir()
        checkpoint = run_dir / "best.ckpt"
        checkpoint.write_bytes(kind.encode("ascii"))
        (run_dir / "training_config.yaml").write_text(
            yaml.safe_dump(base_config(kind), sort_keys=False),
            encoding="utf-8",
        )
        checkpoint_by_kind[kind] = checkpoint

    monkeypatch.setattr(
        "scripts.sleap.run_topdown_optimization_training."
        "summarize_anchor_labels",
        lambda _path: dict(LABEL_SUMMARY),
    )
    output_root = tmp_path / "new_side_round"
    cache_root = tmp_path / "cache"
    candidates = [
        CANDIDATE_BY_NAME["side_centroid_sigma15_strictframe_p40"],
        CANDIDATE_BY_NAME[
            "side_centered_instance_control_strictmouth_p40"
        ],
    ]

    manifest = prepare_candidates(
        base_round_dir=base_round,
        output_root=output_root,
        cache_root=cache_root,
        candidates=candidates,
        seed=42,
        early_stopping_patience=40,
        initial_checkpoint_overrides=checkpoint_by_kind,
    )

    assert manifest["cache_root"] == str(cache_root.resolve())
    assert manifest["software_versions"]["sleap-nn"]
    assert manifest["software_versions"]["sleap-io"]
    for candidate in candidates:
        config_path = (
            output_root / "side" / "configs" / f"{candidate.name}.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        data = config["data_config"]
        expected_cache = (
            cache_root
            / output_root.name
            / "side"
            / candidate.name
        ).resolve()
        assert data["train_labels_path"] == [str(train_labels.resolve())]
        assert data["val_labels_path"] == [str(val_labels.resolve())]
        assert data["validation_fraction"] == 0.0
        assert data["cache_img_path"] == str(expected_cache)
        assert data["use_existing_imgs"] is False
        checkpoint = checkpoint_by_kind[candidate.kind].resolve()
        assert (
            Path(config["model_config"]["pretrained_backbone_weights"])
            == checkpoint
        )
        assert (
            Path(config["model_config"]["pretrained_head_weights"])
            == checkpoint
        )

    entries = {
        entry["candidate"]: entry for entry in manifest["candidates"]
    }
    assert (
        entries["side_centroid_sigma15_strictframe_p40"][
            "initial_checkpoint"
        ]
        == str(checkpoint_by_kind["centroid"].resolve())
    )
    assert (
        entries["side_centered_instance_control_strictmouth_p40"][
            "initial_checkpoint"
        ]
        == str(checkpoint_by_kind["centered_instance"].resolve())
    )


def test_prepare_accepts_explicit_validation_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_round = tmp_path / "combined_round"
    side_dir = base_round / "side"
    side_dir.mkdir(parents=True)
    train_labels = side_dir / "train_labels.pkg.slp"
    default_val_labels = side_dir / "val_labels.pkg.slp"
    combined_val_labels = side_dir / "val_labels_combined.pkg.slp"
    train_labels.write_bytes(b"train")
    default_val_labels.write_bytes(b"default-validation")
    combined_val_labels.write_bytes(b"combined-validation")

    run_dir = tmp_path / "centered_instance_initial"
    run_dir.mkdir()
    checkpoint = run_dir / "best.ckpt"
    checkpoint.write_bytes(b"centered-instance")
    (run_dir / "training_config.yaml").write_text(
        yaml.safe_dump(base_config("centered_instance"), sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.sleap.run_topdown_optimization_training."
        "summarize_anchor_labels",
        lambda path: {
            **LABEL_SUMMARY,
            "frames": 60 if path == combined_val_labels else 100,
        },
    )
    candidate = CANDIDATE_BY_NAME[
        "side_centered_instance_control_strictmouth_p40"
    ]

    manifest = prepare_candidates(
        base_round_dir=base_round,
        output_root=tmp_path / "round9b",
        cache_root=tmp_path / "cache",
        candidates=[candidate],
        seed=42,
        early_stopping_patience=40,
        validation_labels=combined_val_labels,
        initial_checkpoint_overrides={"centered_instance": checkpoint},
    )

    config = yaml.safe_load(
        Path(manifest["candidates"][0]["config"]).read_text(encoding="utf-8")
    )
    assert manifest["val_labels"] == str(combined_val_labels.resolve())
    assert manifest["val_label_summary"]["frames"] == 60
    assert config["data_config"]["val_labels_path"] == [
        str(combined_val_labels.resolve())
    ]


def test_checkpoint_selection_promotes_true_moks_best_and_preserves_topk(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    scores = {
        "best.ckpt": (0.70, 2, 300),
        "best-v1.ckpt": (0.82, 4, 500),
        "best-v2.ckpt": (0.78, 3, 400),
    }
    for filename, (score, epoch, global_step) in scores.items():
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "callbacks": {
                    "ModelCheckpoint": {
                        "monitor": "eval/val/mOKS",
                        "current_score": torch.tensor(score),
                        "best_k_models": {},
                    }
                },
            },
            run_dir / filename,
        )
    original_hashes = {
        filename: _sha256(run_dir / filename) for filename in scores
    }

    result = _finalize_monitored_checkpoints(
        run_dir,
        monitor="eval/val/mOKS",
        mode="max",
    )

    assert result["candidate_count"] == 3
    assert result["selected"]["filename"] == "best-v1.ckpt"
    assert result["selected"]["score"] == pytest.approx(0.82)
    assert _sha256(run_dir / "best.ckpt") == original_hashes["best-v1.ckpt"]
    promoted = torch.load(
        run_dir / "best.ckpt", map_location="cpu", weights_only=False
    )
    assert promoted["epoch"] == 4
    for filename, original_hash in original_hashes.items():
        preserved = run_dir / "checkpoint_originals" / filename
        assert _sha256(preserved) == original_hash
    saved_metadata = json.loads(
        (run_dir / "checkpoint_selection.json").read_text(encoding="utf-8")
    )
    assert saved_metadata["selected"]["filename"] == "best-v1.ckpt"
    assert saved_metadata["canonical_best"]["sha256"] == original_hashes[
        "best-v1.ckpt"
    ]


def test_interrupted_resume_uses_latest_state_and_finalizes_attempt_union(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def save_checkpoint(
        filename: str, score: float, epoch: int, global_step: int
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "state_dict": {"weight": torch.tensor([score])},
                "optimizer_states": [{"state": {}}],
                "lr_schedulers": [{}],
                "callbacks": {
                    "ModelCheckpoint": {
                        "monitor": "eval/val/mOKS",
                        "current_score": torch.tensor(score),
                        "best_k_models": {},
                    }
                },
            },
            run_dir / filename,
        )

    save_checkpoint("last.ckpt", 0.60, 1, 200)
    save_checkpoint("best.ckpt", 0.70, 2, 300)
    save_checkpoint("best-v1.ckpt", 0.82, 4, 500)
    save_checkpoint("recovery-e00005-s000000550.ckpt", 0.79, 5, 550)
    selected, metadata, audit = _select_resume_checkpoint(run_dir)

    assert selected == run_dir / "recovery-e00005-s000000550.ckpt"
    assert metadata == {"epoch": 5, "global_step": 550}
    assert len(audit) == 4
    pre_resume_hashes = {
        name: _sha256(run_dir / name)
        for name in ("best.ckpt", "best-v1.ckpt")
    }
    snapshots = _snapshot_live_top_checkpoints(run_dir, "attempt1")
    assert len(snapshots) == 2
    assert {
        name: _sha256(run_dir / name)
        for name in ("best.ckpt", "best-v1.ckpt")
    } == pre_resume_hashes

    # Simulate the resumed callback overwriting one filename and versioning a
    # second. The immutable history keeps the pre-resume top-k in the union.
    save_checkpoint("best.ckpt", 0.78, 5, 600)
    save_checkpoint("best-v2.ckpt", 0.88, 6, 700)
    result = _finalize_monitored_checkpoints(
        run_dir,
        monitor="eval/val/mOKS",
        mode="max",
    )

    assert result["candidate_count"] == 4
    assert sorted(item["score"] for item in result["candidates"]) == pytest.approx(
        [0.70, 0.78, 0.82, 0.88]
    )
    assert result["selected"]["filename"] == "best-v2.ckpt"
    assert result["selected"]["score"] == pytest.approx(0.88)
    historical_preserved = {
        relative_path: Path(metadata["path"]).name
        for relative_path, metadata in result["preserved_originals"].items()
        if relative_path.startswith("resume_history/")
    }
    assert historical_preserved
    assert all(
        re.fullmatch(r"history_[0-9a-f]{12}\.ckpt", filename)
        for filename in historical_preserved.values()
    )


def test_historical_checkpoint_name_fits_production_unc_path_budget() -> None:
    production_run_dir = PureWindowsPath(
        r"\\bucket.oist.jp\bucket\StephensU\medaka\sleap_models"
        r"\retrain_20260728_round9b_ci_crop448_rot90_jitter8_ohkm"
        r"\side\runs"
        r"\side_centered_instance_round9b_crop448_rot90_jitter8_"
        r"strictmouth_ohkmon_p40"
    )
    metadata = {
        "relative_path": "resume_history/20260729_121300/best-v1.ckpt",
        "sha256": "a" * 64,
    }

    preserved_name = _preserved_checkpoint_name(metadata)
    preserved_path = (
        production_run_dir / "checkpoint_originals" / preserved_name
    )

    assert preserved_name == "history_aaaaaaaaaaaa.ckpt"
    # MAX_PATH is 260 including the terminating NUL when LongPathsEnabled=0.
    assert len(str(preserved_path)) <= 259
    assert len(str(preserved_path)) == 241


def test_moks_candidate_refuses_sleap_nn_030_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = CANDIDATE_BY_NAME[
        "side_centered_instance_round9b_crop448_rot90_jitter8_"
        "strictmouth_ohkmon_p40"
    ]
    config_path = tmp_path / "candidate.yaml"
    config_path.write_text(
        "data_config:\n  val_labels_path: [val.pkg.slp]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.sleap.run_topdown_optimization_training._software_versions",
        lambda: {
            "python": "3.13.11",
            "sleap-nn": "0.3.0",
            "sleap-io": "0.9.2",
            "torch": "2.10.0",
            "lightning": "2.6.1",
        },
    )

    with pytest.raises(RuntimeError, match=r"requires sleap-nn >=0\.3\.1"):
        run_candidate(
            candidate=candidate,
            config_path=config_path,
            output_root=tmp_path / "output",
            force=False,
            dry_run=True,
        )


def test_moks_candidate_requires_one_validation_label_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = CANDIDATE_BY_NAME[
        "side_centered_instance_round9b_crop448_rot90_jitter8_"
        "strictmouth_ohkmon_p40"
    ]
    config_path = tmp_path / "candidate.yaml"
    config_path.write_text(
        "data_config:\n"
        "  val_labels_path: [val_a.pkg.slp, val_b.pkg.slp]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.sleap.run_topdown_optimization_training._software_versions",
        lambda: {
            "python": "3.13.11",
            "sleap-nn": "0.3.1",
            "sleap-io": "0.9.2",
            "torch": "2.10.0",
            "lightning": "2.6.1",
        },
    )

    with pytest.raises(RuntimeError, match="exactly one validation label source"):
        run_candidate(
            candidate=candidate,
            config_path=config_path,
            output_root=tmp_path / "output",
            force=False,
            dry_run=True,
        )


def test_run_candidate_resume_keeps_callback_topk_live(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = CANDIDATE_BY_NAME[
        "side_centered_instance_round9b_crop448_rot90_jitter8_"
        "strictmouth_ohkmon_p40"
    ]
    output_root = tmp_path / "output"
    run_dir = output_root / "side" / "runs" / candidate.name
    run_dir.mkdir(parents=True)
    config_path = tmp_path / "candidate.yaml"
    config_path.write_text(
        "data_config:\n  val_labels_path: [val.pkg.slp]\n",
        encoding="utf-8",
    )
    versions = {
        "python": "3.13.11",
        "sleap-nn": "0.3.1",
        "sleap-io": "0.9.2",
        "torch": "2.10.0",
        "lightning": "2.6.1",
    }
    monkeypatch.setattr(
        "scripts.sleap.run_topdown_optimization_training._software_versions",
        lambda: versions,
    )

    def save_checkpoint(
        filename: str, score: float, epoch: int, global_step: int
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "state_dict": {"weight": torch.tensor([score])},
                "optimizer_states": [{"state": {}}],
                "lr_schedulers": [{}],
                "callbacks": {
                    "ModelCheckpoint": {
                        "monitor": "eval/val/mOKS",
                        "current_score": torch.tensor(score),
                        "best_k_models": {},
                    }
                },
            },
            run_dir / filename,
        )

    save_checkpoint("last.ckpt", 0.60, 1, 200)
    save_checkpoint("best.ckpt", 0.70, 2, 300)
    save_checkpoint("best-v1.ckpt", 0.82, 4, 500)
    live_hashes = {
        name: _sha256(run_dir / name)
        for name in ("best.ckpt", "best-v1.ckpt")
    }
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "state": "failed",
                "config_sha256": _sha256(config_path),
                "software_versions": versions,
            }
        ),
        encoding="utf-8",
    )
    observed: dict[str, bool] = {}

    def fake_run(*args, **kwargs):
        observed["topk_live"] = all(
            (run_dir / name).is_file()
            and _sha256(run_dir / name) == expected_hash
            for name, expected_hash in live_hashes.items()
        )
        observed["resume_from_best_v1"] = any(
            "resume_source_epoch00004_step000000500.ckpt" in str(part)
            for part in args[0]
        )
        observed["recovery_every_validation"] = (
            kwargs["env"]["MEDAKA_RECOVERY_CHECKPOINT_EVERY_VAL"] == "1"
        )
        (run_dir / "training_log.csv").write_text(
            "epoch,eval/val/mOKS\n6,0.88\n", encoding="utf-8"
        )
        save_checkpoint("best-v2.ckpt", 0.88, 6, 700)

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(
        "scripts.sleap.run_topdown_optimization_training.subprocess.run",
        fake_run,
    )

    return_code = run_candidate(
        candidate=candidate,
        config_path=config_path,
        output_root=output_root,
        force=False,
        dry_run=False,
    )

    assert return_code == 0
    assert observed == {
        "topk_live": True,
        "resume_from_best_v1": True,
        "recovery_every_validation": True,
    }
    assert not list(run_dir.glob("best*.interrupted_*.ckpt"))
    selection = json.loads(
        (run_dir / "checkpoint_selection.json").read_text(encoding="utf-8")
    )
    assert selection["candidate_count"] == 3
    assert selection["selected"]["filename"] == "best-v2.ckpt"
    state = json.loads(
        (run_dir / "run_state.json").read_text(encoding="utf-8")
    )
    assert len(state["pre_resume_top_k_snapshots"]) == 2


def test_run_candidate_refuses_cross_version_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = CANDIDATE_BY_NAME[
        "side_centered_instance_round9b_crop448_rot90_jitter8_"
        "strictmouth_ohkmon_p40"
    ]
    output_root = tmp_path / "output"
    run_dir = output_root / "side" / "runs" / candidate.name
    run_dir.mkdir(parents=True)
    config_path = tmp_path / "candidate.yaml"
    config_path.write_text("trainer_config: {}\n", encoding="utf-8")
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "state": "failed",
                "config_sha256": _sha256(config_path),
                "software_versions": {
                    "python": "3.13.0",
                    "sleap-nn": "0.3.0",
                },
            }
        ),
        encoding="utf-8",
    )
    current_versions = {
        "python": "3.13.11",
        "sleap-nn": "0.3.1",
        "sleap-io": "0.9.2",
        "torch": "2.10.0",
        "lightning": "2.6.1",
    }
    monkeypatch.setattr(
        "scripts.sleap.run_topdown_optimization_training._software_versions",
        lambda: current_versions,
    )

    with pytest.raises(RuntimeError, match="cross-version checkpoint resume"):
        run_candidate(
            candidate=candidate,
            config_path=config_path,
            output_root=output_root,
            force=False,
            dry_run=True,
        )
