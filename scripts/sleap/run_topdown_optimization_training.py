from __future__ import annotations

"""Prepare and run the side-view top-down optimization queue.

The default queue is intentionally small and ordered:

1. continued-control centroid sigma 2.5;
2. centroid sigma 2.0;
3. centroid sigma 1.5;
4. continued-control centered-instance training;
5. the primary centered-instance recipe with +/-8 px crop-center jitter;
6. the same recipe with +/-16 px jitter as a robustness ablation.

All candidates start from the frozen matching Round 8 best checkpoint, loading
backbone and head weights but not optimizer/scheduler state. Source labels and
installed SLEAP-NN files are never modified.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.common.local_paths import load_local_paths


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: str
    centroid_policy: str = "disabled"
    centered_instance_policy: str = "disabled"
    sigma: float | None = None
    jitter_px: int | None = None
    online_hard_keypoint_mining: bool | None = None
    required_initial_checkpoint_sha256: str | None = None


PRIMARY_CANDIDATES = (
    Candidate(
        name="side_centroid_sigma25_control_p40",
        kind="centroid",
        centroid_policy="frame",
        sigma=2.5,
    ),
    Candidate(
        name="side_centroid_sigma20_strictframe_p40",
        kind="centroid",
        centroid_policy="frame",
        sigma=2.0,
    ),
    Candidate(
        name="side_centroid_sigma15_strictframe_p40",
        kind="centroid",
        centroid_policy="frame",
        sigma=1.5,
    ),
    Candidate(name="side_centered_instance_control_p40", kind="centered_instance"),
    Candidate(
        name="side_centered_instance_opt_jitter8_p40",
        kind="centered_instance",
        centered_instance_policy="strict",
        jitter_px=8,
        online_hard_keypoint_mining=False,
    ),
    Candidate(
        name="side_centered_instance_opt_jitter16_p40",
        kind="centered_instance",
        centered_instance_policy="strict",
        jitter_px=16,
        online_hard_keypoint_mining=False,
    ),
)
EVALUATED_CONTROL_CI_SHA256 = (
    "18feb2e61c632ceed1e696e47ead14dd95a69522c99c3f6ad5d0fc570455fc5a"
)
PHASE4_CONTROLLED_CANDIDATES = (
    Candidate(
        name=(
            "side_centered_instance_crop448_rot90_jitter0_"
            "strictmouth_ohkmoff_p40"
        ),
        kind="centered_instance",
        centered_instance_policy="strict",
        jitter_px=0,
        online_hard_keypoint_mining=False,
        required_initial_checkpoint_sha256=EVALUATED_CONTROL_CI_SHA256,
    ),
    Candidate(
        name=(
            "side_centered_instance_crop448_rot90_jitter0_"
            "strictmouth_ohkmon_p40"
        ),
        kind="centered_instance",
        centered_instance_policy="strict",
        jitter_px=0,
        online_hard_keypoint_mining=True,
        required_initial_checkpoint_sha256=EVALUATED_CONTROL_CI_SHA256,
    ),
)
OPTIONAL_CANDIDATES = (
    # The missing-mouth fish remains in the image and becomes dense background,
    # so this is deliberately excluded from the default queue.
    Candidate(
        name="side_centroid_sigma20_moreanchors_experimental_p40",
        kind="centroid",
        centroid_policy="instance",
        sigma=2.0,
    ),
    *PHASE4_CONTROLLED_CANDIDATES,
)
ALL_CANDIDATES = PRIMARY_CANDIDATES + OPTIONAL_CANDIDATES
CANDIDATE_BY_NAME = {candidate.name: candidate for candidate in ALL_CANDIDATES}
PHASE4_CONTROLLED_NAMES = {
    candidate.name for candidate in PHASE4_CONTROLLED_CANDIDATES
}


def jitter_fraction(jitter_px: int, crop_size: int) -> float:
    """Convert desired pixels to a fraction of the oversized crop.

    SLEAP-NN first extracts ``floor(crop_size * sqrt(2))`` pixels for rotation
    headroom. The strict-anchor launcher applies this translation after
    rotation/scale so the requested pixel bound is not amplified by the affine
    composition.
    """
    oversized_crop = int(crop_size * math.sqrt(2.0))
    return float(jitter_px) / float(oversized_crop)


def summarize_anchor_labels(labels_path: Path, anchor_index: int = 0) -> dict[str, int]:
    """Count finite semantic anchors without decoding images."""
    import sleap_io as sio

    labels = sio.load_slp(str(labels_path), open_videos=False)
    frames = 0
    frames_all_anchor = 0
    frames_any_anchor = 0
    instances = 0
    instances_nonempty = 0
    instances_anchor = 0
    for labeled_frame in labels:
        frame_anchor_count = 0
        frame_instance_count = 0
        for instance in labeled_frame.user_instances:
            points = np.asarray(instance.numpy(), dtype=float)
            frame_instance_count += 1
            instances_nonempty += int(not instance.is_empty)
            if (
                points.ndim == 2
                and anchor_index < points.shape[0]
                and np.isfinite(points[anchor_index]).all()
            ):
                frame_anchor_count += 1
        frames += 1
        instances += frame_instance_count
        instances_anchor += frame_anchor_count
        frames_any_anchor += int(frame_anchor_count > 0)
        frames_all_anchor += int(
            frame_instance_count > 0 and frame_anchor_count == frame_instance_count
        )
    return {
        "frames": frames,
        "frames_all_anchor": frames_all_anchor,
        "frames_any_anchor": frames_any_anchor,
        "instances": instances,
        "instances_nonempty": instances_nonempty,
        "instances_anchor": instances_anchor,
        "instances_missing_anchor": instances - instances_anchor,
    }


def build_candidate_config(
    base_config: dict[str, Any],
    candidate: Candidate,
    *,
    output_root: Path,
    label_summary: dict[str, int],
    seed: int = 42,
    early_stopping_patience: int = 40,
    initial_checkpoint: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a candidate training config plus derived provenance."""
    config = copy.deepcopy(base_config)
    data = config["data_config"]
    trainer = config["trainer_config"]
    model = config["model_config"]
    batch_size = int(trainer["train_data_loader"]["batch_size"])

    trainer["seed"] = int(seed)
    trainer["run_name"] = candidate.name
    trainer["ckpt_dir"] = str((output_root / "side" / "runs").resolve())
    trainer["resume_ckpt_path"] = None
    trainer["early_stopping"]["stop_training_on_plateau"] = True
    trainer["early_stopping"]["patience"] = int(early_stopping_patience)
    # Keep the shorter LR plateau so several reductions can occur before the
    # 40-epoch early-stop window expires.
    trainer["lr_scheduler"]["reduce_lr_on_plateau"]["patience"] = 5
    trainer["model_ckpt"]["save_top_k"] = 1
    trainer["model_ckpt"]["save_last"] = True
    data["use_existing_imgs"] = True
    if initial_checkpoint is not None:
        model["pretrained_backbone_weights"] = str(initial_checkpoint.resolve())
        model["pretrained_head_weights"] = str(initial_checkpoint.resolve())

    derived: dict[str, Any] = {
        "candidate": candidate.name,
        "kind": candidate.kind,
        "seed": int(seed),
        "early_stopping_patience": int(early_stopping_patience),
        "lr_plateau_patience": 5,
        "label_summary": dict(label_summary),
        "initial_checkpoint": (
            str(initial_checkpoint.resolve())
            if initial_checkpoint is not None
            else None
        ),
    }

    if candidate.kind == "centroid":
        model["head_configs"]["centroid"]["confmaps"]["sigma"] = float(
            candidate.sigma
        )
        if candidate.centroid_policy == "frame":
            samples_per_epoch = label_summary["frames_all_anchor"]
        elif candidate.centroid_policy == "instance":
            samples_per_epoch = label_summary["frames_any_anchor"]
        else:
            raise ValueError(
                f"Unsupported centroid policy: {candidate.centroid_policy}"
            )
        trainer["train_steps_per_epoch"] = int(
            math.ceil(samples_per_epoch / batch_size)
        )
        derived.update(
            {
                "sigma": float(candidate.sigma),
                "centroid_anchor_policy": candidate.centroid_policy,
                "samples_per_epoch": int(samples_per_epoch),
            }
        )
    elif candidate.kind == "centered_instance" and candidate.jitter_px is None:
        trainer["train_steps_per_epoch"] = int(
            math.ceil(label_summary["instances_nonempty"] / batch_size)
        )
        geometric = data["augmentation_config"]["geometric"]
        derived.update(
            {
                "control": True,
                "crop_size": int(data["preprocessing"]["crop_size"]),
                "jitter_px": 0,
                "rotation_degrees": [
                    float(geometric["rotation_min"]),
                    float(geometric["rotation_max"]),
                ],
                "scale_range": [
                    float(geometric["scale_min"]),
                    float(geometric["scale_max"]),
                ],
                "centered_instance_anchor_policy": "disabled",
                "online_hard_keypoint_mining": bool(
                    trainer["online_hard_keypoint_mining"]["online_mining"]
                ),
                "samples_per_epoch": int(label_summary["instances_nonempty"]),
            }
        )
    elif candidate.kind == "centered_instance":
        crop_size = 448
        jitter_px = int(candidate.jitter_px)
        jitter_frac = jitter_fraction(jitter_px, crop_size)
        if candidate.online_hard_keypoint_mining is None:
            raise ValueError(
                f"{candidate.name} must set online_hard_keypoint_mining"
            )
        online_hard_keypoint_mining = bool(
            candidate.online_hard_keypoint_mining
        )
        data["preprocessing"]["crop_size"] = crop_size
        geometric = data["augmentation_config"]["geometric"]
        geometric["rotation_min"] = -90.0
        geometric["rotation_max"] = 90.0
        geometric["rotation_p"] = 1.0
        geometric["translate_width"] = jitter_frac
        geometric["translate_height"] = jitter_frac
        geometric["translate_p"] = 1.0 if jitter_px > 0 else 0.0
        trainer["online_hard_keypoint_mining"]["online_mining"] = (
            online_hard_keypoint_mining
        )
        # With two nodes, OHKM selected both and multiplied ordinary MSE by 5.
        # Keep identical monitor thresholds in the controlled OHKM pair so the
        # mining flag is the only substantive config difference.
        trainer["lr_scheduler"]["reduce_lr_on_plateau"]["threshold"] = 2.0e-7
        trainer["early_stopping"]["min_delta"] = 2.0e-9
        trainer["train_steps_per_epoch"] = int(
            math.ceil(label_summary["instances_anchor"] / batch_size)
        )
        derived.update(
            {
                "crop_size": crop_size,
                "jitter_px": jitter_px,
                "translation_fraction_of_oversized_crop": jitter_frac,
                "jitter_implementation": (
                    "post_affine_translation" if jitter_px > 0 else "none"
                ),
                "rotation_degrees": [-90.0, 90.0],
                "scale_range": [
                    float(geometric["scale_min"]),
                    float(geometric["scale_max"]),
                ],
                "centered_instance_anchor_policy": (
                    candidate.centered_instance_policy
                ),
                "online_hard_keypoint_mining": (
                    online_hard_keypoint_mining
                ),
                "samples_per_epoch": int(label_summary["instances_anchor"]),
            }
        )
    else:
        raise ValueError(f"Unsupported candidate kind: {candidate.kind}")

    config["description"] = (
        "Side-view top-down geometry optimization candidate generated by "
        "scripts.sleap.run_topdown_optimization_training; see "
        "experiment_manifest.json for exact derived settings."
    )
    if candidate.kind == "centered_instance" and candidate.jitter_px is not None:
        config["description"] += (
            " Crop-position jitter is applied as a post-affine pure translation "
            f"bounded to +/-{int(candidate.jitter_px)} input pixels per axis."
        )
    return config, derived


def _cache_is_complete(cache_root: Path, train_frames: int, val_frames: int) -> bool:
    train_count = len(list((cache_root / "train_imgs").glob("sample_*.jpg")))
    val_count = len(list((cache_root / "val_imgs").glob("sample_*.jpg")))
    return train_count == train_frames and val_count == val_frames


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_prepared_text(
    path: Path, text: str, *, overwrite_prepared: bool
) -> None:
    """Write a prepared artifact without silently changing prior provenance."""
    if path.is_file():
        current = path.read_text(encoding="utf-8")
        if current == text:
            return
        if not overwrite_prepared:
            raise FileExistsError(
                f"Prepared artifact differs from requested content: {path}. "
                "Use a new output root, or --overwrite-prepared before any "
                "training outputs exist."
            )
    path.write_text(text, encoding="utf-8")


def _training_outputs_exist(output_root: Path) -> bool:
    runs_dir = output_root / "side" / "runs"
    if not runs_dir.is_dir():
        return False
    material_names = {
        "best.ckpt",
        "last.ckpt",
        "training_log.csv",
        "training_console.log",
        "run_state.json",
    }
    return any(
        path.is_file() and path.name in material_names
        for path in runs_dir.rglob("*")
    )


def _load_checkpoint_resume_metadata(path: Path) -> dict[str, int]:
    """Validate a Lightning checkpoint and return its resume coordinates."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"state_dict", "optimizer_states", "lr_schedulers"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(
            f"Checkpoint is missing resume state {missing}: {path}"
        )
    if not checkpoint["optimizer_states"]:
        raise ValueError(f"Checkpoint has no optimizer state: {path}")
    return {
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
    }


def _archive_retry_artifact(path: Path, attempt_stamp: str) -> Path | None:
    """Preserve a prior attempt artifact before retrying the same run name."""
    if not path.is_file():
        return None
    archive = path.with_name(
        f"{path.stem}.interrupted_{attempt_stamp}{path.suffix}"
    )
    suffix = 1
    while archive.exists():
        archive = path.with_name(
            f"{path.stem}.interrupted_{attempt_stamp}_{suffix}{path.suffix}"
        )
        suffix += 1
    path.rename(archive)
    return archive


def resolve_round8_checkpoint(base_round_dir: Path, kind: str) -> Path:
    """Resolve the unique strict-mouth Round 8 side checkpoint."""
    candidates = sorted(
        (base_round_dir / "side" / "runs").glob(
            f"round8_260723_strictmouth.{kind}_*/best.ckpt"
        )
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one Round 8 side {kind} checkpoint, found {candidates}"
        )
    return candidates[0]


def prepare_candidates(
    *,
    base_round_dir: Path,
    output_root: Path,
    candidates: list[Candidate],
    seed: int,
    early_stopping_patience: int,
    initial_checkpoint_overrides: dict[str, Path] | None = None,
    overwrite_prepared: bool = False,
) -> dict[str, Any]:
    """Write candidate YAMLs and return a machine-readable manifest."""
    train_labels = base_round_dir / "side" / "train_labels.pkg.slp"
    val_labels = base_round_dir / "side" / "val_labels.pkg.slp"
    if not train_labels.is_file() or not val_labels.is_file():
        raise FileNotFoundError(
            f"Round 8 side labels not found under {base_round_dir}"
        )
    train_summary = summarize_anchor_labels(train_labels)
    val_summary = summarize_anchor_labels(val_labels)
    config_dir = output_root / "side" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "side" / "runs").mkdir(parents=True, exist_ok=True)

    entries = []
    baseline_artifacts: dict[str, dict[str, str]] = {}
    checkpoint_overrides = initial_checkpoint_overrides or {}
    for candidate in candidates:
        initial_checkpoint = checkpoint_overrides.get(
            candidate.kind
        ) or resolve_round8_checkpoint(base_round_dir, candidate.kind)
        if not initial_checkpoint.is_file():
            raise FileNotFoundError(
                f"Initial checkpoint not found: {initial_checkpoint}"
            )
        initial_checkpoint_sha256 = _sha256(initial_checkpoint)
        if (
            candidate.required_initial_checkpoint_sha256 is not None
            and initial_checkpoint_sha256
            != candidate.required_initial_checkpoint_sha256
        ):
            raise ValueError(
                f"{candidate.name} requires initial checkpoint SHA-256 "
                f"{candidate.required_initial_checkpoint_sha256}, got "
                f"{initial_checkpoint_sha256}: {initial_checkpoint}"
            )
        base_path = initial_checkpoint.parent / "training_config.yaml"
        if not base_path.is_file():
            raise FileNotFoundError(
                f"Initial training config not found: {base_path}"
            )
        base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        config, derived = build_candidate_config(
            base_config,
            candidate,
            output_root=output_root,
            label_summary=train_summary,
            seed=seed,
            early_stopping_patience=early_stopping_patience,
            initial_checkpoint=initial_checkpoint,
        )
        derived["initial_checkpoint_sha256"] = initial_checkpoint_sha256
        derived["initial_training_config_sha256"] = _sha256(base_path)
        cache_root = Path(config["data_config"]["cache_img_path"])
        cache_complete = _cache_is_complete(
            cache_root, train_summary["frames"], val_summary["frames"]
        )
        config["data_config"]["use_existing_imgs"] = bool(cache_complete)
        config_path = config_dir / f"{candidate.name}.yaml"
        config_text = yaml.safe_dump(config, sort_keys=False)
        _write_prepared_text(
            config_path,
            config_text,
            overwrite_prepared=overwrite_prepared,
        )
        entries.append(
            {
                **derived,
                "config": str(config_path.resolve()),
                "base_config": str(base_path.resolve()),
                "run_dir": str(
                    (
                        output_root
                        / "side"
                        / "runs"
                        / candidate.name
                    ).resolve()
                ),
                "cache_root": str(cache_root),
                "reuses_complete_cache": bool(cache_complete),
                # Hash the bytes actually written. On Windows, text-mode
                # newline translation may make these differ from config_text.
                "generated_config_sha256": _sha256(config_path),
            }
        )
        if candidate.kind not in baseline_artifacts:
            baseline_artifacts[candidate.kind] = {
                "checkpoint": str(initial_checkpoint.resolve()),
                "checkpoint_sha256": initial_checkpoint_sha256,
                "training_config": str(base_path.resolve()),
                "training_config_sha256": _sha256(base_path),
            }

    manifest = {
        "workflow": "side_topdown_geometry_optimization",
        "base_round_dir": str(base_round_dir.resolve()),
        "output_root": str(output_root.resolve()),
        "train_labels": str(train_labels.resolve()),
        "val_labels": str(val_labels.resolve()),
        "label_artifacts": {
            "train": {
                "path": str(train_labels.resolve()),
                "size_bytes": train_labels.stat().st_size,
                "mtime_ns": train_labels.stat().st_mtime_ns,
            },
            "validation": {
                "path": str(val_labels.resolve()),
                "size_bytes": val_labels.stat().st_size,
                "mtime_ns": val_labels.stat().st_mtime_ns,
            },
        },
        "train_label_summary": train_summary,
        "val_label_summary": val_summary,
        "baseline_artifacts": baseline_artifacts,
        "candidate_order": [candidate.name for candidate in candidates],
        "candidates": entries,
    }
    manifest_path = output_root / "experiment_manifest.json"
    _write_prepared_text(
        manifest_path,
        json.dumps(manifest, indent=2) + "\n",
        overwrite_prepared=overwrite_prepared,
    )
    return manifest


def run_candidate(
    *,
    candidate: Candidate,
    config_path: Path,
    output_root: Path,
    force: bool,
    dry_run: bool,
) -> int:
    """Run one candidate synchronously with durable file logging."""
    run_dir = output_root / "side" / "runs" / candidate.name
    best_checkpoint = run_dir / "best.ckpt"
    last_checkpoint = run_dir / "last.ckpt"
    training_log = run_dir / "training_log.csv"
    run_state_path = run_dir / "run_state.json"
    config_sha256 = _sha256(config_path)
    prior_state: dict[str, Any] = {}
    if run_state_path.is_file():
        try:
            prior_state = json.loads(
                run_state_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            prior_state = {}
    if (
        not force
        and prior_state
        and prior_state.get("config_sha256") not in {None, config_sha256}
    ):
        raise RuntimeError(
            f"Prior run has a different config hash: {candidate.name}"
        )
    if not force and prior_state.get("state") == "complete":
        if prior_state.get("config_sha256") != config_sha256:
            raise RuntimeError(
                f"Completed run has a different config hash: {candidate.name}"
            )
        if best_checkpoint.is_file() and training_log.is_file():
            print(f"[skip verified complete] {candidate.name}", flush=True)
            return 0
        raise RuntimeError(
            f"Completion marker exists but required outputs are missing: "
            f"{candidate.name}"
        )

    strict_policy_active = (
        candidate.centroid_policy != "disabled"
        or candidate.centered_instance_policy != "disabled"
    )
    if strict_policy_active:
        entrypoint = [str(Path(__file__).with_name("train_strict_anchor.py"))]
    else:
        entrypoint = ["-m", "sleap_nn.cli"]
    command = [
        sys.executable,
        *entrypoint,
        "train",
        "--config-name",
        config_path.name,
        "--config-dir",
        str(config_path.parent),
    ]
    resume_checkpoint: Path | None = None
    resume_metadata: dict[str, int] | None = None
    resume_candidate: Path | None = None
    if not force and prior_state.get("state") in {"running", "failed"}:
        if last_checkpoint.is_file():
            resume_candidate = last_checkpoint
        else:
            preserved_sources = sorted(
                run_dir.glob("resume_source_epoch*_step*.ckpt"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
            if preserved_sources:
                resume_candidate = preserved_sources[0]
    if resume_candidate is not None:
        resume_metadata = _load_checkpoint_resume_metadata(resume_candidate)
        if resume_candidate.name.startswith("resume_source_epoch"):
            resume_checkpoint = resume_candidate
        else:
            resume_checkpoint = run_dir / (
                "resume_source_"
                f"epoch{resume_metadata['epoch']:05d}_"
                f"step{resume_metadata['global_step']:09d}.ckpt"
            )
        command.append(
            "trainer_config.resume_ckpt_path="
            f"{resume_checkpoint.resolve().as_posix()}"
        )
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["MEDAKA_CENTROID_ANCHOR_POLICY"] = candidate.centroid_policy
    env["MEDAKA_CENTERED_INSTANCE_ANCHOR_POLICY"] = (
        candidate.centered_instance_policy
    )
    env["MEDAKA_CENTERED_INSTANCE_JITTER_ORDER"] = (
        "post_affine"
        if candidate.centered_instance_policy == "strict"
        and candidate.jitter_px is not None
        and candidate.jitter_px > 0
        else "native"
    )
    print(f"\n--- {candidate.name} ---", flush=True)
    print(f"config: {config_path}", flush=True)
    print(f"run:    {run_dir}", flush=True)
    print(f"command: {' '.join(command)}", flush=True)
    if dry_run:
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    if (
        resume_checkpoint is not None
        and not resume_checkpoint.is_file()
    ):
        shutil.copy2(resume_candidate, resume_checkpoint)
    console_log = run_dir / "training_console.log"
    started = time.time()
    attempt_stamp = time.strftime(
        "%Y%m%d_%H%M%S", time.localtime(started)
    )
    archived_artifacts = {}
    if prior_state.get("state") in {"running", "failed"}:
        retry_artifacts = [console_log, training_log, run_state_path]
        if resume_checkpoint is not None:
            # SLEAP-NN adds a "-1" suffix whenever best.ckpt already exists,
            # even for a genuine Lightning resume. The immutable resume source
            # above makes it safe to archive both live checkpoint names so the
            # resumed trainer writes back into the manifest-declared run dir.
            retry_artifacts = [
                best_checkpoint,
                last_checkpoint,
                *retry_artifacts,
            ]
        for artifact in retry_artifacts:
            archived = _archive_retry_artifact(artifact, attempt_stamp)
            if archived is not None:
                archived_artifacts[artifact.name] = str(archived.resolve())
    run_state_path.write_text(
        json.dumps(
            {
                "state": "running",
                "candidate": candidate.name,
                "config": str(config_path.resolve()),
                "config_sha256": config_sha256,
                "command": command,
                "resume_checkpoint": (
                    str(resume_checkpoint.resolve())
                    if resume_checkpoint is not None
                    else None
                ),
                "resume_metadata": resume_metadata,
                "archived_prior_artifacts": archived_artifacts,
                "started_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%S%z", time.localtime(started)
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with console_log.open("w", encoding="utf-8", errors="replace") as stream:
        stream.write(f"# {candidate.name}\n")
        stream.write(f"# command: {' '.join(command)}\n")
        stream.write(f"# started: {time.ctime(started)}\n")
        stream.write(
            "# policies: "
            f"centroid={candidate.centroid_policy}; "
            "centered_instance="
            f"{candidate.centered_instance_policy}\n\n"
        )
        stream.flush()
        process = subprocess.run(
            command,
            cwd=str(config_path.parent),
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - started
    effective_return_code = int(process.returncode)
    if (
        effective_return_code == 0
        and (not best_checkpoint.is_file() or not training_log.is_file())
    ):
        effective_return_code = 2
    run_state_path.write_text(
        json.dumps(
            {
                "state": (
                    "complete" if effective_return_code == 0 else "failed"
                ),
                "candidate": candidate.name,
                "config": str(config_path.resolve()),
                "config_sha256": config_sha256,
                "command": command,
                "resume_checkpoint": (
                    str(resume_checkpoint.resolve())
                    if resume_checkpoint is not None
                    else None
                ),
                "resume_metadata": resume_metadata,
                "archived_prior_artifacts": archived_artifacts,
                "started_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%S%z", time.localtime(started)
                ),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "elapsed_seconds": elapsed,
                "subprocess_return_code": int(process.returncode),
                "return_code": effective_return_code,
                "best_checkpoint_exists": best_checkpoint.is_file(),
                "training_log_exists": training_log.is_file(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"{candidate.name}: rc={effective_return_code}; "
        f"elapsed={elapsed / 60.0:.1f} min; log={console_log}",
        flush=True,
    )
    return effective_return_code


def _write_queue_status(
    output_root: Path,
    *,
    state: str,
    active: str | None,
    completed: list[str],
    failed: list[dict[str, Any]],
) -> None:
    path = output_root / "queue_status.json"
    path.write_text(
        json.dumps(
            {
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "state": state,
                "active": active,
                "completed": completed,
                "failed": failed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=Path("configs/paths.local.toml"),
    )
    parser.add_argument("--base-round-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--centered-instance-initial-checkpoint",
        type=Path,
        help=(
            "Override the initialization checkpoint and sibling training "
            "config for selected centered-instance candidates."
        ),
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=list(CANDIDATE_BY_NAME),
        default=None,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--overwrite-prepared",
        action="store_true",
        help=(
            "Replace differing prepared YAML/manifest files. Refused after "
            "material training outputs exist."
        ),
    )
    parser.add_argument("--continue-on-fail", action="store_true")
    args = parser.parse_args(argv)

    local_paths = load_local_paths(args.paths_config)
    if local_paths is None or local_paths.sleap_models_dir is None:
        parser.error("paths config must define paths.sleap_models_dir")
    base_round_dir = args.base_round_dir or (
        local_paths.sleap_models_dir / "retrain_20260722_round8"
    )
    output_root = args.output_root or (
        local_paths.sleap_models_dir / "retrain_20260724_sideopt"
    )
    selected = (
        [CANDIDATE_BY_NAME[name] for name in args.candidates]
        if args.candidates
        else list(PRIMARY_CANDIDATES)
    )
    selected_names = {candidate.name for candidate in selected}
    phase4_selected = bool(selected_names & PHASE4_CONTROLLED_NAMES)
    completed_root = (
        local_paths.sleap_models_dir / "retrain_20260724_sideopt"
    ).resolve()
    resolved_output_root = output_root.resolve()
    if phase4_selected:
        if args.output_root is None:
            parser.error(
                "controlled Phase-4 candidates require an explicit new "
                "--output-root"
            )
        if args.centered_instance_initial_checkpoint is None:
            parser.error(
                "controlled Phase-4 candidates require "
                "--centered-instance-initial-checkpoint"
            )
        if (
            resolved_output_root == completed_root
            or completed_root in resolved_output_root.parents
        ):
            parser.error(
                "controlled Phase-4 candidates cannot write into the "
                f"completed run root: {completed_root}"
            )
    if args.overwrite_prepared and _training_outputs_exist(output_root):
        parser.error(
            "--overwrite-prepared is refused because training outputs already "
            f"exist under {output_root / 'side' / 'runs'}"
        )
    manifest = prepare_candidates(
        base_round_dir=base_round_dir,
        output_root=output_root,
        candidates=selected,
        seed=args.seed,
        early_stopping_patience=40,
        initial_checkpoint_overrides=(
            {
                "centered_instance": (
                    args.centered_instance_initial_checkpoint
                )
            }
            if args.centered_instance_initial_checkpoint is not None
            else None
        ),
        overwrite_prepared=args.overwrite_prepared,
    )
    print(
        f"Prepared {len(selected)} candidate config(s): "
        f"{output_root / 'experiment_manifest.json'}",
        flush=True,
    )
    if args.prepare_only:
        return 0

    entries = {entry["candidate"]: entry for entry in manifest["candidates"]}
    if args.dry_run:
        for candidate in selected:
            run_candidate(
                candidate=candidate,
                config_path=Path(entries[candidate.name]["config"]),
                output_root=output_root,
                force=args.force,
                dry_run=True,
            )
        return 0

    completed: list[str] = []
    failed: list[dict[str, Any]] = []
    _write_queue_status(
        output_root,
        state="running",
        active=None,
        completed=completed,
        failed=failed,
    )
    for candidate in selected:
        _write_queue_status(
            output_root,
            state="running",
            active=candidate.name,
            completed=completed,
            failed=failed,
        )
        return_code = run_candidate(
            candidate=candidate,
            config_path=Path(entries[candidate.name]["config"]),
            output_root=output_root,
            force=args.force,
            dry_run=False,
        )
        if return_code == 0:
            completed.append(candidate.name)
        else:
            failed.append(
                {"candidate": candidate.name, "return_code": return_code}
            )
            if not args.continue_on_fail:
                _write_queue_status(
                    output_root,
                    state="failed",
                    active=None,
                    completed=completed,
                    failed=failed,
                )
                return return_code

    final_state = "complete" if not failed else "complete_with_failures"
    _write_queue_status(
        output_root,
        state=final_state,
        active=None,
        completed=completed,
        failed=failed,
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
