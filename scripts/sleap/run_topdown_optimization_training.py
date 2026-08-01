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
import importlib.metadata
import json
import math
import os
import platform
import re
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
    batch_size: int | None = None
    jitter_px: int | None = None
    online_hard_keypoint_mining: bool | None = None
    preserve_control_geometry: bool = False
    save_top_k: int = 1
    centered_instance_anchor_part: str | None = None
    scale_range: tuple[float, float] | None = None
    scale_p: float | None = None
    eval_frequency: int | None = None
    checkpoint_monitor: str | None = None
    checkpoint_mode: str | None = None
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
ROUND8_PRODUCTION_CI_SHA256 = (
    "0d4930a437a9cd61022c98b16ff9386af3782abea364da069edecf0ab139eab1"
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
    Candidate(
        name="side_centroid_sigma15_strictframe_batch6_restart_p40",
        kind="centroid",
        centroid_policy="frame",
        sigma=1.5,
        batch_size=6,
    ),
    Candidate(
        name="side_centered_instance_control_strictmouth_p40",
        kind="centered_instance",
        centered_instance_policy="strict",
        jitter_px=0,
        online_hard_keypoint_mining=True,
        preserve_control_geometry=True,
    ),
    Candidate(
        name=(
            "side_centered_instance_round9b_crop448_rot90_jitter8_"
            "strictmouth_ohkmon_p40"
        ),
        kind="centered_instance",
        centered_instance_policy="strict",
        jitter_px=8,
        online_hard_keypoint_mining=True,
        save_top_k=3,
        centered_instance_anchor_part="mouth",
        scale_range=(0.7, 1.3),
        scale_p=1.0,
        eval_frequency=1,
        checkpoint_monitor="eval/val/mOKS",
        checkpoint_mode="max",
        required_initial_checkpoint_sha256=ROUND8_PRODUCTION_CI_SHA256,
    ),
    *PHASE4_CONTROLLED_CANDIDATES,
)
ALL_CANDIDATES = PRIMARY_CANDIDATES + OPTIONAL_CANDIDATES
CANDIDATE_BY_NAME = {candidate.name: candidate for candidate in ALL_CANDIDATES}
PHASE4_CONTROLLED_NAMES = {
    candidate.name for candidate in PHASE4_CONTROLLED_CANDIDATES
}


def _software_versions() -> dict[str, str]:
    """Return the training software fingerprint without importing CUDA."""
    versions = {"python": platform.python_version()}
    for distribution in ("sleap-nn", "sleap-io", "torch", "lightning"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _validate_candidate_runtime(
    candidate: Candidate, software_versions: dict[str, str]
) -> None:
    """Fail before training when an mOKS candidate uses an unaudited runtime."""
    if candidate.checkpoint_monitor != "eval/val/mOKS":
        return
    raw_version = software_versions.get("sleap-nn", "not-installed")
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", raw_version)
    if match is None:
        raise RuntimeError(
            f"{candidate.name} requires sleap-nn >=0.3.1 for grouped mOKS "
            f"checkpointing; found {raw_version!r}."
        )
    version = tuple(int(part) for part in match.groups())
    if version < (0, 3, 1):
        raise RuntimeError(
            f"{candidate.name} requires sleap-nn >=0.3.1 for grouped mOKS "
            f"checkpointing; found {raw_version!r}."
        )


def _validate_single_validation_label_source(
    candidate: Candidate, config_path: Path
) -> None:
    """Enforce unambiguous ``video_idx, frame_idx`` grouping for mOKS."""
    if candidate.checkpoint_monitor != "eval/val/mOKS":
        return
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = (config.get("data_config") or {}).get("val_labels_path")
    if (
        not isinstance(paths, list)
        or len(paths) != 1
        or not isinstance(paths[0], str)
        or not paths[0].strip()
    ):
        raise RuntimeError(
            f"{candidate.name} grouped mOKS evaluation requires exactly one "
            "validation label source because epoch records do not expose "
            f"labels_idx; found val_labels_path={paths!r}."
        )


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
    if int(candidate.save_top_k) <= 0:
        raise ValueError(f"{candidate.name} must save at least one top checkpoint")
    trainer["model_ckpt"]["save_top_k"] = int(candidate.save_top_k)
    trainer["model_ckpt"]["save_last"] = True
    if candidate.checkpoint_monitor is not None:
        if candidate.checkpoint_mode not in {"min", "max"}:
            raise ValueError(
                f"{candidate.name} checkpoint_mode must be 'min' or 'max'"
            )
        if candidate.eval_frequency is None or candidate.eval_frequency <= 0:
            raise ValueError(
                f"{candidate.name} must enable positive-frequency evaluation "
                "when checkpointing on an eval metric"
            )
        trainer.setdefault("eval", {})
        trainer["eval"]["enabled"] = True
        trainer["eval"]["frequency"] = int(candidate.eval_frequency)
        trainer["model_ckpt"]["monitor"] = candidate.checkpoint_monitor
        trainer["model_ckpt"]["mode"] = candidate.checkpoint_mode
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
        "save_top_k": int(candidate.save_top_k),
        "save_last": True,
        "checkpoint_monitor": candidate.checkpoint_monitor or "val/loss",
        "checkpoint_mode": candidate.checkpoint_mode or "min",
        "eval_frequency": candidate.eval_frequency,
        "recovery_checkpoint_every_validation": (
            candidate.checkpoint_monitor == "eval/val/mOKS"
        ),
        "label_summary": dict(label_summary),
        "initial_checkpoint": (
            str(initial_checkpoint.resolve())
            if initial_checkpoint is not None
            else None
        ),
    }

    if candidate.kind == "centroid":
        if candidate.batch_size is not None:
            if int(candidate.batch_size) <= 0:
                raise ValueError(
                    f"{candidate.name} must use a positive batch size"
                )
            batch_size = int(candidate.batch_size)
            trainer["train_data_loader"]["batch_size"] = batch_size
            trainer["val_data_loader"]["batch_size"] = batch_size
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
                "batch_size": int(batch_size),
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
        jitter_px = int(candidate.jitter_px)
        crop_size = 512 if candidate.preserve_control_geometry else 448
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
        if candidate.centered_instance_anchor_part is not None:
            model["head_configs"]["centered_instance"]["confmaps"][
                "anchor_part"
            ] = candidate.centered_instance_anchor_part
        if candidate.scale_range is not None:
            scale_min, scale_max = map(float, candidate.scale_range)
            if scale_min <= 0 or scale_max < scale_min:
                raise ValueError(
                    f"{candidate.name} has invalid scale range "
                    f"{candidate.scale_range}"
                )
            geometric["scale_min"] = scale_min
            geometric["scale_max"] = scale_max
        if candidate.scale_p is not None:
            if not 0.0 <= float(candidate.scale_p) <= 1.0:
                raise ValueError(
                    f"{candidate.name} scale_p must be between 0 and 1"
                )
            geometric["scale_p"] = float(candidate.scale_p)
        if candidate.preserve_control_geometry:
            geometric["rotation_min"] = -180.0
            geometric["rotation_max"] = 180.0
            geometric["scale_min"] = 0.7
            geometric["scale_max"] = 1.3
            geometric["scale_p"] = 1.0
            trainer["train_data_loader"]["batch_size"] = 16
            trainer["val_data_loader"]["batch_size"] = 16
            batch_size = 16
        else:
            geometric["rotation_min"] = -90.0
            geometric["rotation_max"] = 90.0
        geometric["rotation_p"] = 1.0
        geometric["translate_width"] = jitter_frac
        geometric["translate_height"] = jitter_frac
        geometric["translate_p"] = 1.0 if jitter_px > 0 else 0.0
        trainer["online_hard_keypoint_mining"]["online_mining"] = (
            online_hard_keypoint_mining
        )
        if not candidate.preserve_control_geometry:
            # With two nodes, OHKM selected both and multiplied ordinary MSE by
            # 5. Keep identical monitor thresholds in the controlled OHKM pair
            # so the mining flag is the only substantive config difference.
            trainer["lr_scheduler"]["reduce_lr_on_plateau"]["threshold"] = (
                2.0e-7
            )
            trainer["early_stopping"]["min_delta"] = 2.0e-9
        trainer["train_steps_per_epoch"] = int(
            math.ceil(label_summary["instances_anchor"] / batch_size)
        )
        derived.update(
            {
                "preserves_control_geometry": bool(
                    candidate.preserve_control_geometry
                ),
                "crop_size": crop_size,
                "jitter_px": jitter_px,
                "translation_fraction_of_oversized_crop": jitter_frac,
                "jitter_implementation": (
                    "post_affine_translation" if jitter_px > 0 else "none"
                ),
                "rotation_degrees": [
                    float(geometric["rotation_min"]),
                    float(geometric["rotation_max"]),
                ],
                "scale_range": [
                    float(geometric["scale_min"]),
                    float(geometric["scale_max"]),
                ],
                "centered_instance_anchor_policy": (
                    candidate.centered_instance_policy
                ),
                "centered_instance_anchor_part": (
                    candidate.centered_instance_anchor_part
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
    if (
        candidate.kind == "centered_instance"
        and candidate.jitter_px is not None
        and int(candidate.jitter_px) > 0
    ):
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


_TOP_CHECKPOINT_PATTERN = re.compile(r"^best(?:-v\d+)?\.ckpt$")
_RECOVERY_CHECKPOINT_PATTERN = re.compile(
    r"^recovery-e-?\d+-s-?\d+(?:-v\d+)?\.ckpt$"
)
_RESUME_SOURCE_PATTERN = re.compile(
    r"^resume_source_epoch-?\d+_step-?\d+(?:_\d+)?\.ckpt$"
)


def _select_resume_checkpoint(
    run_dir: Path,
) -> tuple[Path | None, dict[str, int] | None, list[dict[str, Any]]]:
    """Select the most advanced valid checkpoint by saved training state."""
    paths = []
    for path in sorted(run_dir.glob("*.ckpt"), key=lambda item: item.name):
        if (
            path.name == "last.ckpt"
            or _TOP_CHECKPOINT_PATTERN.fullmatch(path.name)
            or _RECOVERY_CHECKPOINT_PATTERN.fullmatch(path.name)
            or _RESUME_SOURCE_PATTERN.fullmatch(path.name)
        ):
            paths.append(path)
    audit: list[dict[str, Any]] = []
    valid: list[tuple[Path, dict[str, int]]] = []
    for path in paths:
        try:
            metadata = _load_checkpoint_resume_metadata(path)
        except Exception as error:  # noqa: BLE001 - retain per-file audit.
            audit.append(
                {
                    "path": str(path.resolve()),
                    "valid": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        audit.append(
            {
                "path": str(path.resolve()),
                "valid": True,
                **metadata,
            }
        )
        valid.append((path, metadata))
    if not valid:
        if paths:
            details = "; ".join(
                f"{Path(item['path']).name}: {item.get('error')}"
                for item in audit
            )
            raise ValueError(
                f"No valid resume checkpoint under {run_dir}: {details}"
            )
        return None, None, audit
    selected_path, selected_metadata = max(
        valid,
        key=lambda item: (
            item[1]["global_step"],
            item[1]["epoch"],
            item[0].name,
        ),
    )
    return selected_path, selected_metadata, audit


def _resume_history_dir(run_dir: Path) -> Path:
    """Return a short candidate-isolated history path for Windows safety."""
    candidate_key = hashlib.sha256(
        run_dir.name.encode("utf-8")
    ).hexdigest()[:12]
    return run_dir.parent / ".resume_topk" / candidate_key


def _snapshot_live_top_checkpoints(
    run_dir: Path, attempt_stamp: str
) -> list[dict[str, str]]:
    """Copy pre-resume top-k aside without moving callback-owned live paths."""
    checkpoints = sorted(
        (
            path
            for path in run_dir.glob("best*.ckpt")
            if _TOP_CHECKPOINT_PATTERN.fullmatch(path.name)
        ),
        key=lambda path: path.name,
    )
    if not checkpoints:
        return []
    history_dir = _resume_history_dir(run_dir)
    snapshot_dir = history_dir / attempt_stamp
    suffix = 1
    while snapshot_dir.exists():
        snapshot_dir = history_dir / f"{attempt_stamp}_{suffix}"
        suffix += 1
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    snapshots = []
    for checkpoint in checkpoints:
        snapshot_path = snapshot_dir / checkpoint.name
        shutil.copy2(checkpoint, snapshot_path)
        snapshots.append(
            {
                "live_path": str(checkpoint.resolve()),
                "snapshot_path": str(snapshot_path.resolve()),
                "sha256": _sha256(snapshot_path),
            }
        )
    return snapshots


def _as_finite_float(value: Any) -> float | None:
    """Convert a scalar/tensor-like value to a finite float."""
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _checkpoint_monitor_metadata(path: Path, monitor: str) -> dict[str, Any]:
    """Read the monitored score saved in one Lightning checkpoint."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    matching_states = []
    for state in (checkpoint.get("callbacks") or {}).values():
        if isinstance(state, dict) and state.get("monitor") == monitor:
            matching_states.append(state)
    if len(matching_states) != 1:
        raise ValueError(
            f"Expected one ModelCheckpoint state monitoring {monitor!r}, "
            f"found {len(matching_states)} in {path}"
        )
    state = matching_states[0]
    score = _as_finite_float(state.get("current_score"))
    if score is None:
        score_by_path = state.get("best_k_models") or {}
        resolved = path.resolve()
        for saved_path, saved_score in score_by_path.items():
            candidate_path = Path(saved_path)
            if (
                candidate_path.name == path.name
                or candidate_path.resolve() == resolved
            ):
                score = _as_finite_float(saved_score)
                break
    if score is None:
        raise ValueError(
            f"Checkpoint has no finite {monitor!r} score: {path}"
        )
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "score": score,
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "sha256": _sha256(path),
    }


def _preserved_checkpoint_name(metadata: dict[str, Any]) -> str:
    """Return a short deterministic filename while provenance stays metadata."""
    relative_path = Path(metadata["relative_path"])
    if relative_path.parts and relative_path.parts[0] == "resume_history":
        return f"history_{metadata['sha256'][:12]}.ckpt"
    return (
        relative_path.name
        if len(relative_path.parts) == 1
        else f"checkpoint_{metadata['sha256'][:12]}.ckpt"
    )


def _finalize_monitored_checkpoints(
    run_dir: Path,
    *,
    monitor: str,
    mode: str,
) -> dict[str, Any]:
    """Preserve top-k files and promote the actual monitored best checkpoint."""
    if mode not in {"min", "max"}:
        raise ValueError(f"Unsupported checkpoint mode: {mode!r}")
    live_checkpoints = [
        path
        for path in run_dir.glob("best*.ckpt")
        if _TOP_CHECKPOINT_PATTERN.fullmatch(path.name)
    ]
    history_dir = _resume_history_dir(run_dir)
    historical_checkpoints = [
        path
        for path in history_dir.glob("*/*")
        if path.is_file() and _TOP_CHECKPOINT_PATTERN.fullmatch(path.name)
    ]

    def source_key(path: Path) -> str:
        if path.parent == run_dir:
            return path.name
        return (
            Path("resume_history")
            / path.relative_to(history_dir)
        ).as_posix()

    checkpoints = sorted(
        [*live_checkpoints, *historical_checkpoints],
        key=source_key,
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"No monitored best*.ckpt files found under {run_dir}"
        )

    candidates_by_hash: dict[str, dict[str, Any]] = {}
    checkpoint_by_hash: dict[str, Path] = {}
    for checkpoint in checkpoints:
        metadata = _checkpoint_monitor_metadata(checkpoint, monitor)
        relative_path = source_key(checkpoint)
        existing = candidates_by_hash.get(metadata["sha256"])
        if existing is not None:
            existing.setdefault("aliases", []).append(relative_path)
            continue
        metadata["relative_path"] = relative_path
        metadata["aliases"] = []
        candidates_by_hash[metadata["sha256"]] = metadata
        checkpoint_by_hash[metadata["sha256"]] = checkpoint
    candidates = list(candidates_by_hash.values())
    score_direction = 1.0 if mode == "max" else -1.0
    selected = max(
        candidates,
        key=lambda item: (
            score_direction * item["score"],
            item["global_step"],
            item["epoch"],
            item["filename"],
        ),
    )

    originals_dir = run_dir / "checkpoint_originals"
    originals_dir.mkdir(parents=True, exist_ok=True)
    preserved = {}
    for metadata in candidates:
        checkpoint = checkpoint_by_hash[metadata["sha256"]]
        preserved_name = _preserved_checkpoint_name(metadata)
        preserved_path = originals_dir / preserved_name
        if preserved_path.exists():
            if _sha256(preserved_path) != metadata["sha256"]:
                preserved_path = originals_dir / (
                    f"{Path(preserved_name).stem}__"
                    f"{metadata['sha256'][:12]}"
                    f"{Path(preserved_name).suffix}"
                )
                if (
                    preserved_path.exists()
                    and _sha256(preserved_path) != metadata["sha256"]
                ):
                    raise FileExistsError(
                        "Content-addressed preserved checkpoint differs from "
                        f"current candidate: {preserved_path}"
                    )
                if not preserved_path.exists():
                    shutil.copy2(checkpoint, preserved_path)
        else:
            shutil.copy2(checkpoint, preserved_path)
        preserved[metadata["relative_path"]] = {
            "path": str(preserved_path.resolve()),
            "sha256": metadata["sha256"],
        }

    selected_path = Path(selected["path"])
    canonical_path = run_dir / "best.ckpt"
    if selected_path.resolve() != canonical_path.resolve():
        temporary_path = run_dir / "best.promoting.tmp"
        shutil.copy2(selected_path, temporary_path)
        os.replace(temporary_path, canonical_path)
    canonical_sha256 = _sha256(canonical_path)
    if canonical_sha256 != selected["sha256"]:
        raise RuntimeError(
            "Canonical best checkpoint hash does not match selected checkpoint"
        )

    result = {
        "schema_version": "monitored_checkpoint_selection_v1",
        "monitor": monitor,
        "mode": mode,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected": selected,
        "canonical_best": {
            "path": str(canonical_path.resolve()),
            "sha256": canonical_sha256,
        },
        "preserved_originals": preserved,
    }
    selection_path = run_dir / "checkpoint_selection.json"
    selection_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    result["metadata_path"] = str(selection_path.resolve())
    return result


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
    cache_root: Path,
    candidates: list[Candidate],
    seed: int,
    early_stopping_patience: int,
    validation_labels: Path | None = None,
    initial_checkpoint_overrides: dict[str, Path] | None = None,
    overwrite_prepared: bool = False,
) -> dict[str, Any]:
    """Write candidate YAMLs and return a machine-readable manifest."""
    train_labels = base_round_dir / "side" / "train_labels.pkg.slp"
    val_labels = (
        validation_labels
        if validation_labels is not None
        else base_round_dir / "side" / "val_labels.pkg.slp"
    )
    if not train_labels.is_file() or not val_labels.is_file():
        raise FileNotFoundError(
            "Side training or validation labels not found: "
            f"train={train_labels}, validation={val_labels}"
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
        data_config = config["data_config"]
        data_config["train_labels_path"] = [str(train_labels.resolve())]
        data_config["val_labels_path"] = [str(val_labels.resolve())]
        data_config["validation_fraction"] = 0.0
        candidate_cache_root = (
            cache_root
            / output_root.resolve().name
            / "side"
            / candidate.name
        ).resolve()
        data_config["cache_img_path"] = str(candidate_cache_root)
        cache_complete = _cache_is_complete(
            candidate_cache_root,
            train_summary["frames"],
            val_summary["frames"],
        )
        data_config["use_existing_imgs"] = bool(cache_complete)
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
                "cache_root": str(candidate_cache_root),
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
        "software_versions": _software_versions(),
        "base_round_dir": str(base_round_dir.resolve()),
        "output_root": str(output_root.resolve()),
        "cache_root": str(cache_root.resolve()),
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
    software_versions = _software_versions()
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
    if (
        not force
        and prior_state.get("state") in {"running", "failed"}
        and prior_state.get("software_versions") != software_versions
    ):
        raise RuntimeError(
            "Refusing cross-version checkpoint resume for "
            f"{candidate.name}: prior={prior_state.get('software_versions')}; "
            f"current={software_versions}. Use a fresh output root."
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
    _validate_candidate_runtime(candidate, software_versions)
    _validate_single_validation_label_source(candidate, config_path)

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
    resume_candidates: list[dict[str, Any]] = []
    if not force and prior_state.get("state") in {"running", "failed"}:
        (
            resume_candidate,
            resume_metadata,
            resume_candidates,
        ) = _select_resume_checkpoint(run_dir)
    if resume_candidate is not None:
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
    env["MEDAKA_RECOVERY_CHECKPOINT_EVERY_VAL"] = (
        "1"
        if candidate.checkpoint_monitor == "eval/val/mOKS"
        else "0"
    )
    print(f"\n--- {candidate.name} ---", flush=True)
    print(f"config: {config_path}", flush=True)
    print(f"run:    {run_dir}", flush=True)
    print(f"command: {' '.join(command)}", flush=True)
    if dry_run:
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    if resume_checkpoint is not None and not resume_checkpoint.is_file():
        shutil.copy2(resume_candidate, resume_checkpoint)
    elif (
        resume_checkpoint is not None
        and resume_candidate is not None
        and _sha256(resume_checkpoint) != _sha256(resume_candidate)
    ):
        raise FileExistsError(
            "Immutable resume source name collides with different checkpoint: "
            f"{resume_checkpoint}"
        )
    console_log = run_dir / "training_console.log"
    started = time.time()
    attempt_stamp = time.strftime(
        "%Y%m%d_%H%M%S", time.localtime(started)
    )
    archived_artifacts = {}
    pre_resume_top_k_snapshots: list[dict[str, str]] = []
    if prior_state.get("state") in {"running", "failed"}:
        if resume_checkpoint is not None:
            pre_resume_top_k_snapshots = _snapshot_live_top_checkpoints(
                run_dir, attempt_stamp
            )
        retry_artifacts = [console_log, training_log, run_state_path]
        if resume_checkpoint is not None:
            # Keep callback-owned best*.ckpt paths live. The process-local
            # strict-anchor wrapper suppresses upstream run-name suffixing only
            # for a genuine same-directory resume.
            retry_artifacts = [
                last_checkpoint,
                run_dir / "checkpoint_selection.json",
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
                "software_versions": software_versions,
                "command": command,
                "resume_checkpoint": (
                    str(resume_checkpoint.resolve())
                    if resume_checkpoint is not None
                    else None
                ),
                "resume_metadata": resume_metadata,
                "resume_candidates": resume_candidates,
                "pre_resume_top_k_snapshots": pre_resume_top_k_snapshots,
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
            "# software_versions: "
            f"{json.dumps(software_versions, sort_keys=True)}\n"
        )
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
    checkpoint_selection = None
    checkpoint_selection_error = None
    if effective_return_code == 0 and candidate.checkpoint_monitor is not None:
        try:
            checkpoint_selection = _finalize_monitored_checkpoints(
                run_dir,
                monitor=candidate.checkpoint_monitor,
                mode=candidate.checkpoint_mode or "min",
            )
        except Exception as error:  # noqa: BLE001 - persist finalization failure.
            checkpoint_selection_error = f"{type(error).__name__}: {error}"
            effective_return_code = 3
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
                "software_versions": software_versions,
                "command": command,
                "resume_checkpoint": (
                    str(resume_checkpoint.resolve())
                    if resume_checkpoint is not None
                    else None
                ),
                "resume_metadata": resume_metadata,
                "resume_candidates": resume_candidates,
                "pre_resume_top_k_snapshots": pre_resume_top_k_snapshots,
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
                "checkpoint_selection": checkpoint_selection,
                "checkpoint_selection_error": checkpoint_selection_error,
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
        "--validation-labels",
        type=Path,
        help=(
            "Optional explicit validation SLP. Use a frozen combined "
            "new-data plus regression set when selecting offset-robust "
            "centered-instance checkpoints."
        ),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("C:/sleap_cache"),
        help=(
            "Local cache base. Each candidate receives a fresh cache directory "
            "namespaced by output root, view, and candidate name."
        ),
    )
    parser.add_argument(
        "--centroid-initial-checkpoint",
        type=Path,
        help=(
            "Override the initialization checkpoint and sibling training "
            "config for selected centroid candidates."
        ),
    )
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
    initial_checkpoint_overrides: dict[str, Path] = {}
    if args.centroid_initial_checkpoint is not None:
        initial_checkpoint_overrides["centroid"] = (
            args.centroid_initial_checkpoint
        )
    if args.centered_instance_initial_checkpoint is not None:
        initial_checkpoint_overrides["centered_instance"] = (
            args.centered_instance_initial_checkpoint
        )
    manifest = prepare_candidates(
        base_round_dir=base_round_dir,
        output_root=output_root,
        cache_root=args.cache_root,
        validation_labels=args.validation_labels,
        candidates=selected,
        seed=args.seed,
        early_stopping_patience=40,
        initial_checkpoint_overrides=initial_checkpoint_overrides or None,
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
