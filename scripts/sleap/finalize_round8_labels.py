"""Validate corrected Round 8 labels and build leakage-safe train/holdout sets.

The corrected multi-view package is validated against the authoritative
selection manifest. Corrected training frames are split into top and side
packages, then merged over (never into) the Round 7 cumulative packages with
new labels winning exact collisions. Locked holdout labels and every existing
training label inside a holdout event's +/- guard band are excluded from
training. Explicit validation packages and configs are written; no model is
trained and Round 7 inputs are never modified.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from scripts.analysis.contact_episode_gt import VIEWS, utc_now
from scripts.common.local_paths import DEFAULT_PATHS_CONFIG, load_local_paths
from scripts.sleap.pyqt_multiview_annotator import (
    _video_source_path,
    classify_view,
    experiment_from_video,
)


BUCKET_FOR_VIEW = {"top_cam": "top", "side1_cam": "side", "side2_cam": "side"}
BUCKETS = ("top", "side")
MODEL_KINDS = ("centroid", "centered_instance")


def label_key(lf: Any) -> tuple[str, str, int]:
    source = _video_source_path(lf.video)
    experiment = experiment_from_video(source)
    view = classify_view(source)
    if experiment is None or view is None:
        raise ValueError(f"cannot recover experiment/view from label video: {source}")
    return experiment, view, int(lf.frame_idx)


def blocked_training_ranges(
    manifest: pd.DataFrame,
    *,
    guard_band_frames: int = 166,
) -> list[tuple[int, int, str]]:
    """Return merged source-frame ranges unavailable to any training package."""
    holdout = manifest.loc[manifest["split"].astype(str).eq("holdout")]
    ranges = sorted(
        {
            (
                int(row.event_start_frame) - int(guard_band_frames),
                int(row.event_end_frame) + int(guard_band_frames),
                str(row.event_group_id),
            )
            for row in holdout.itertuples()
        }
    )
    return ranges


def frame_is_blocked(frame: int, ranges: Sequence[tuple[int, int, str]]) -> bool:
    return any(start <= int(frame) <= end for start, end, _event in ranges)


def _visible_node_count(inst: Any) -> int:
    points = np.asarray(inst.numpy(), dtype=float)
    return int(np.isfinite(points).all(axis=1).sum())


def validate_corrected_labels(
    corrected_paths: Sequence[Path],
    manifest: pd.DataFrame,
    *,
    experiment: str,
    annotation_batch: str,
    instances_per_view: int = 3,
    exclude_invalid_training_source_frames: bool = False,
) -> tuple[dict[tuple[str, str, int], Any], Any, dict[str, Any]]:
    """Validate exact expected coverage and three real fish per selected view."""
    import sleap_io as sio

    if annotation_batch not in {"pilot", "expansion", "all"}:
        raise ValueError("annotation_batch must be pilot, expansion, or all")
    expected_manifest = (
        manifest
        if annotation_batch == "all"
        else manifest.loc[manifest["annotation_batch"].astype(str).eq(annotation_batch)]
    )
    expected = {
        (experiment, view, int(frame))
        for frame in expected_manifest["frame"]
        for view in VIEWS
    }
    observed: dict[tuple[str, str, int], Any] = {}
    canonical_skeleton = None
    partial_by_key: dict[tuple[str, str, int], int] = {}
    hidden_by_key: dict[tuple[str, str, int], int] = {}
    invalid: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in corrected_paths:
        labels = sio.load_slp(str(path))
        if len(labels.skeletons) != 1:
            raise ValueError(f"{path} must contain one skeleton")
        names = [node.name for node in labels.skeletons[0].nodes]
        if names != ["mouth", "tail"]:
            raise ValueError(f"{path} node order must be ['mouth', 'tail'], got {names}")
        if canonical_skeleton is None:
            canonical_skeleton = labels.skeletons[0]
        for lf in labels.labeled_frames:
            key = label_key(lf)
            if key in observed:
                raise ValueError(f"duplicate corrected labeled frame: {key}")
            finite_nodes_per_instance: list[int] = []
            partial_count = 0
            hidden_count = 0
            for instance_index, inst in enumerate(lf.instances):
                visible_count = _visible_node_count(inst)
                finite_nodes_per_instance.append(visible_count)
                if visible_count == 1:
                    partial_count += 1
                points = getattr(inst, "points", None)
                if points is not None:
                    hidden_count += sum(
                        bool(np.isfinite(point["xy"]).all()) and not bool(point["visible"])
                        for point in points
                    )
            reasons: list[str] = []
            if len(lf.instances) != int(instances_per_view):
                reasons.append(
                    f"instance_count={len(lf.instances)} (expected {instances_per_view})"
                )
            empty_indices = [
                index
                for index, finite_count in enumerate(finite_nodes_per_instance)
                if finite_count < 1
            ]
            if empty_indices:
                reasons.append(f"instances_without_finite_nodes={empty_indices}")
            if reasons:
                invalid[key] = {
                    "reasons": reasons,
                    "finite_nodes_per_instance": finite_nodes_per_instance,
                }
            partial_by_key[key] = partial_count
            hidden_by_key[key] = hidden_count
            observed[key] = lf
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    if missing or extra:
        raise ValueError(
            f"corrected package coverage mismatch; missing={missing[:12]}, extra={extra[:12]}"
        )
    if canonical_skeleton is None:
        raise ValueError("no corrected labels were loaded")

    excluded_source_frames: list[int] = []
    excluded_view_images: list[dict[str, Any]] = []
    if invalid:
        if not exclude_invalid_training_source_frames:
            preview = [
                {"key": list(key), **details}
                for key, details in list(sorted(invalid.items()))[:12]
            ]
            raise ValueError(
                "corrected package has invalid instance structure; "
                f"invalid_view_images={len(invalid)}, preview={preview}"
            )
        split_by_frame = {
            int(row.frame): str(row.split) for row in expected_manifest.itertuples()
        }
        invalid_frames = sorted({key[2] for key in invalid})
        unsafe_holdout = [
            frame for frame in invalid_frames if split_by_frame.get(frame) != "train"
        ]
        if unsafe_holdout:
            raise ValueError(
                "invalid corrected labels occur outside the training split and "
                f"cannot be excluded automatically: {unsafe_holdout}"
            )
        excluded_source_frames = invalid_frames
        excluded_view_images = [
            {"key": list(key), **details} for key, details in sorted(invalid.items())
        ]
        observed = {
            key: lf for key, lf in observed.items()
            if key[2] not in set(excluded_source_frames)
        }

    partial_instances = sum(partial_by_key[key] for key in observed)
    hidden_nodes = sum(hidden_by_key[key] for key in observed)
    return observed, canonical_skeleton, {
        "expected_source_frames": int(len(expected_manifest)),
        "expected_view_images": int(len(expected)),
        "validated_view_images": int(len(observed)),
        "instances_per_view": int(instances_per_view),
        "partial_instances": int(partial_instances),
        "hidden_finite_nodes": int(hidden_nodes),
        "excluded_invalid_training_source_frames": excluded_source_frames,
        "excluded_invalid_view_images": excluded_view_images,
    }


def _clone_instance(inst: Any, skeleton: Any) -> Any:
    import sleap_io as sio

    points = np.asarray(inst.numpy(), dtype=float)
    source_names = [node.name for node in inst.skeleton.nodes]
    data = np.full((2, 3), np.nan, dtype=float)
    data[:, 2] = 0.0
    structured = getattr(inst, "points", None)
    for target_index, name in enumerate(("mouth", "tail")):
        if name not in source_names:
            continue
        source_index = source_names.index(name)
        if source_index >= len(points) or not np.isfinite(points[source_index]).all():
            continue
        data[target_index, :2] = points[source_index]
        visible = True
        if structured is not None and source_index < len(structured):
            visible = bool(structured[source_index]["visible"])
        data[target_index, 2] = float(visible)
    return sio.Instance.from_numpy(points_data=data, skeleton=skeleton)


def _clone_lf(lf: Any, skeleton: Any) -> Any:
    import sleap_io as sio

    return sio.LabeledFrame(
        video=lf.video,
        frame_idx=int(lf.frame_idx),
        instances=[_clone_instance(inst, skeleton) for inst in lf.instances],
    )


def _write_package(
    lfs: Sequence[Any],
    *,
    skeleton: Any,
    path: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    import sleap_io as sio

    cloned = [_clone_lf(lf, skeleton) for lf in lfs]
    videos = list({id(lf.video): lf.video for lf in cloned}.values())
    suggestions = [
        sio.SuggestionFrame(video=lf.video, frame_idx=int(lf.frame_idx)) for lf in cloned
    ]
    labels = sio.Labels(
        skeletons=[skeleton],
        videos=videos,
        labeled_frames=cloned,
        suggestions=suggestions,
        provenance=dict(provenance),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.save_slp(labels, str(path), embed="all")
    return {
        "path": str(path.resolve()),
        "labeled_frames": len(cloned),
        "instances": sum(len(lf.instances) for lf in cloned),
        "size_bytes": int(path.stat().st_size),
    }


def merge_corrected_training_over_base(
    base_path: Path,
    corrected_train: Mapping[tuple[str, str, int], Any],
    *,
    bucket: str,
    experiment: str,
    blocked_ranges: Sequence[tuple[int, int, str]],
) -> tuple[list[Any], Any, dict[str, Any]]:
    """Union base/new train labels with corrected collisions winning."""
    import sleap_io as sio

    base = sio.load_slp(str(base_path))
    if not base.skeletons:
        raise ValueError(f"{base_path} contains no skeleton")
    skeleton = base.skeletons[0]
    for skeleton_index, candidate_skeleton in enumerate(base.skeletons):
        names = [node.name for node in candidate_skeleton.nodes]
        edges = [
            (edge.source.name, edge.destination.name)
            for edge in candidate_skeleton.edges
        ]
        if names != ["mouth", "tail"] or edges != [("mouth", "tail")]:
            raise ValueError(
                f"base skeleton {skeleton_index} is not canonical mouth->tail: "
                f"nodes={names}, edges={edges}"
            )
    merged: dict[tuple[str, str, int], Any] = {}
    blocked_base = 0
    for lf in base.labeled_frames:
        key = label_key(lf)
        if BUCKET_FOR_VIEW[key[1]] != bucket:
            continue
        if key[0] == experiment and frame_is_blocked(key[2], blocked_ranges):
            blocked_base += 1
            continue
        merged[key] = lf
    collisions = 0
    added = 0
    for key, lf in corrected_train.items():
        if BUCKET_FOR_VIEW[key[1]] != bucket:
            continue
        if frame_is_blocked(key[2], blocked_ranges):
            raise ValueError(f"corrected training label lies in holdout guard band: {key}")
        if key in merged:
            collisions += 1
        else:
            added += 1
        merged[key] = lf
    lfs = [merged[key] for key in sorted(merged)]
    return lfs, skeleton, {
        "base_input_frames": int(len(base.labeled_frames)),
        "blocked_base_frames_removed": int(blocked_base),
        "corrected_collisions_won": int(collisions),
        "corrected_new_frames_added": int(added),
        "merged_frames": int(len(lfs)),
        "compatible_base_skeleton_objects_canonicalized": int(len(base.skeletons)),
    }


def _write_training_configs(
    *,
    bucket: str,
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    cache_root: Path,
    pretrained_checkpoints: Mapping[str, Path] | None = None,
) -> list[str]:
    template_dir = Path(__file__).resolve().parents[2] / "configs" / "sleap" / "training"
    outputs: list[str] = []
    for kind in MODEL_KINDS:
        config = yaml.safe_load((template_dir / f"{kind}.yaml").read_text(encoding="utf-8"))
        data = config["data_config"]
        data["train_labels_path"] = [str(train_path.resolve())]
        data["val_labels_path"] = [str(val_path.resolve())]
        data["validation_fraction"] = 0.0
        data["data_pipeline_fw"] = "torch_dataset_cache_img_disk"
        data["cache_img_path"] = str(
            (cache_root / f"{output_dir.name}_{bucket}_{kind}").resolve()
        )
        data["use_existing_imgs"] = False
        data["delete_cache_imgs_after_training"] = False
        trainer = config["trainer_config"]
        trainer["ckpt_dir"] = str((output_dir / bucket / "runs").resolve())
        trainer["run_name"] = f"round8_finetune.{kind}"
        trainer["resume_ckpt_path"] = None
        checkpoint = (
            pretrained_checkpoints.get(kind)
            if pretrained_checkpoints is not None
            else None
        )
        if checkpoint is not None:
            config["model_config"]["pretrained_backbone_weights"] = str(
                checkpoint.resolve()
            )
            config["model_config"]["pretrained_head_weights"] = str(
                checkpoint.resolve()
            )
        # Keep the training YAML strictly within TrainingJobConfig's schema.
        # Detailed Round 8 provenance lives in finalize_report.json and
        # training_paths.json beside the configs.
        config["sleap_nn_version"] = "0.3.0"
        config["description"] = (
            "Round 8 cumulative fine-tune with event-level holdout; see "
            "finalize_report.json for source, cache, and warm-start provenance."
        )
        out = output_dir / bucket / "configs" / f"{kind}.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        outputs.append(str(out.resolve()))
    return outputs


def find_pretrained_checkpoint(
    round_dir: Path,
    *,
    bucket: str,
    kind: str,
) -> Path:
    """Resolve one prior-round checkpoint for a top-down model component."""
    direct = round_dir / bucket / "runs" / kind / "best.ckpt"
    if direct.is_file():
        return direct
    token = f".{kind}_"
    candidates = sorted(
        path
        for path in (round_dir / bucket / "runs").glob("*/best.ckpt")
        if token in path.parent.name
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one {bucket}/{kind} best.ckpt under {round_dir}, "
            f"found {candidates}"
        )
    return candidates[0]


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    local = load_local_paths(args.paths_config)
    if local is None or local.sleap_models_dir is None or local.runs_dir is None:
        raise ValueError("paths config must define sleap_models_dir and runs_dir")
    experiment = str(args.experiment)
    manifest_path = Path(args.manifest)
    corrected_paths = [Path(path) for path in args.corrected]
    base_top = Path(
        args.round7_top_labels
        or (local.sleap_models_dir / "retrain_20260704" / "top" / "labels.pkg.slp")
    )
    base_side = Path(
        args.round7_side_labels
        or (local.sleap_models_dir / "retrain_20260704" / "side" / "labels.pkg.slp")
    )
    output_dir = Path(
        args.output_dir
        or (local.runs_dir / "round8_active_learning" / experiment / f"finalized_{args.annotation_batch}")
    )
    cache_root = Path(getattr(args, "cache_root", None) or "C:/sleap_cache")
    pretrained_round_dir_arg = getattr(args, "pretrained_round_dir", None)
    pretrained_round_dir = (
        Path(pretrained_round_dir_arg) if pretrained_round_dir_arg else None
    )
    pretrained_by_bucket: dict[str, dict[str, Path]] = {}
    if pretrained_round_dir is not None:
        pretrained_by_bucket = {
            bucket: {
                kind: find_pretrained_checkpoint(
                    pretrained_round_dir, bucket=bucket, kind=kind
                )
                for kind in MODEL_KINDS
            }
            for bucket in BUCKETS
        }
    for path in [manifest_path, base_top, base_side, *corrected_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    observed, corrected_skeleton, validation = validate_corrected_labels(
        corrected_paths,
        manifest,
        experiment=experiment,
        annotation_batch=args.annotation_batch,
        instances_per_view=args.instances_per_view,
        exclude_invalid_training_source_frames=bool(
            getattr(args, "exclude_invalid_training_source_frames", False)
        ),
    )
    selected_manifest = (
        manifest
        if args.annotation_batch == "all"
        else manifest.loc[manifest["annotation_batch"].eq(args.annotation_batch)]
    )
    split_by_frame = {
        int(row.frame): str(row.split) for row in selected_manifest.itertuples()
    }
    corrected_train = {
        key: lf for key, lf in observed.items() if split_by_frame[key[2]] == "train"
    }
    corrected_holdout = {
        key: lf for key, lf in observed.items() if split_by_frame[key[2]] == "holdout"
    }
    blocked = blocked_training_ranges(
        manifest, guard_band_frames=args.guard_band_frames
    )
    provenance = {
        "workflow": "round8_event_level_finalize",
        "generated_at_utc": utc_now(),
        "manifest": str(manifest_path.resolve()),
        "corrected_inputs": [str(path.resolve()) for path in corrected_paths],
        "annotation_batch": args.annotation_batch,
        "blocked_training_ranges": blocked,
    }
    outputs: dict[str, Any] = {}
    merge_report: dict[str, Any] = {}
    training_paths: dict[str, Any] = {}
    combined_train_lfs: list[Any] = []
    combined_train_skeleton = None
    for bucket, base_path in (("top", base_top), ("side", base_side)):
        train_lfs, train_skeleton, stats = merge_corrected_training_over_base(
            base_path,
            corrected_train,
            bucket=bucket,
            experiment=experiment,
            blocked_ranges=blocked,
        )
        val_lfs = [
            lf for key, lf in sorted(corrected_holdout.items())
            if BUCKET_FOR_VIEW[key[1]] == bucket
        ]
        train_path = output_dir / bucket / "train_labels.pkg.slp"
        val_path = output_dir / bucket / "val_labels.pkg.slp"
        outputs[f"{bucket}_train"] = _write_package(
            train_lfs,
            skeleton=train_skeleton,
            path=train_path,
            provenance={**provenance, "bucket": bucket, "split": "train"},
        )
        outputs[f"{bucket}_val"] = _write_package(
            val_lfs,
            skeleton=corrected_skeleton,
            path=val_path,
            provenance={**provenance, "bucket": bucket, "split": "holdout"},
        )
        configs = _write_training_configs(
            bucket=bucket,
            train_path=train_path,
            val_path=val_path,
            output_dir=output_dir,
            cache_root=cache_root,
            pretrained_checkpoints=pretrained_by_bucket.get(bucket),
        )
        combined_train_lfs.extend(train_lfs)
        if combined_train_skeleton is None:
            combined_train_skeleton = train_skeleton
        merge_report[bucket] = stats
        training_paths[bucket] = {
            "train_labels_path": str(train_path.resolve()),
            "val_labels_path": str(val_path.resolve()),
            "configs": configs,
            "validation_fraction": 0.0,
            "disk_cache_root": str(cache_root.resolve()),
            "pretrained_checkpoints": {
                kind: str(path.resolve())
                for kind, path in pretrained_by_bucket.get(bucket, {}).items()
            },
        }

    if combined_train_skeleton is None:
        raise ValueError("no cumulative training labels were assembled")
    combined_train_path = output_dir / "combined_labels.pkg.slp"
    outputs["combined_train"] = _write_package(
        combined_train_lfs,
        skeleton=combined_train_skeleton,
        path=combined_train_path,
        provenance={
            **provenance,
            "bucket": "top_and_side",
            "split": "train",
            "dedup_key": "experiment|view|source_frame",
        },
    )

    train_keys = set(corrected_train)
    holdout_keys = set(corrected_holdout)
    leakage = sorted(train_keys & holdout_keys)
    if leakage:
        raise ValueError(f"corrected train/holdout key leakage: {leakage[:10]}")
    report = {
        "workflow": "round8_event_level_finalize",
        "generated_at_utc": utc_now(),
        "experiment": experiment,
        "annotation_batch": args.annotation_batch,
        "validation": validation,
        "corrected_counts": {
            "train_view_images": len(corrected_train),
            "holdout_view_images": len(corrected_holdout),
        },
        "blocked_training_ranges": blocked,
        "merge": merge_report,
        "outputs": outputs,
        "training_paths": training_paths,
        "random_frame_validation_disabled": True,
        "train_holdout_key_leakage": leakage,
        "round7_inputs_modified": False,
        "cache_root": str(cache_root.resolve()),
        "pretrained_round_dir": (
            str(pretrained_round_dir.resolve())
            if pretrained_round_dir is not None
            else None
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "finalize_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "training_paths.json").write_text(
        json.dumps(training_paths, indent=2) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrected", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotation-batch", choices=("pilot", "expansion", "all"), required=True)
    parser.add_argument("--experiment", default="20251225_K_1_2_FCMC")
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--round7-top-labels", type=Path)
    parser.add_argument("--round7-side-labels", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--instances-per-view", type=int, default=3)
    parser.add_argument("--guard-band-frames", type=int, default=166)
    parser.add_argument(
        "--exclude-invalid-training-source-frames",
        action="store_true",
        help=(
            "Exclude an entire synchronized training source frame when any "
            "view has an unexpected instance count or an empty instance. "
            "Invalid holdout frames still fail validation."
        ),
    )
    parser.add_argument(
        "--pretrained-round-dir",
        type=Path,
        help=(
            "Prior round root containing per-view centroid and "
            "centered-instance best.ckpt files. Both backbone and compatible "
            "head weights are loaded for fine-tuning."
        ),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("C:/sleap_cache"),
        help="Local disk root for persistent SLEAP-NN image caches.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = finalize(args)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
