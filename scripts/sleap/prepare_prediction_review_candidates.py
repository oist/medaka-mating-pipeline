from __future__ import annotations

"""Generate sparse SLEAP candidates for human-in-the-loop annotation review.

The output of this command is a *candidate overlay*, not an annotation file.
Use it with ``pyqt_multiview_annotator --candidate-slp`` while keeping a blank
or manually edited package as the authoritative ``--in`` file.

Only the labeled-frame positions already present in the seed package are sent
through SLEAP-NN. Top-camera frames use the top model pair; both side-camera
frames use the side model pair. This avoids full-video inference for a small
review set.
"""

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Iterable

from scripts.sleap.eval_topdown_vs_bottomup import (
    build_subset_labels,
    make_prediction_labels_self_contained,
    run_strategy,
    save_prediction_slp,
)
from scripts.sleap.pyqt_multiview_annotator import (
    _video_source_path,
    classify_view,
    experiment_from_video,
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of ``path`` without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_positions_by_model(labels) -> tuple[list[int], list[int]]:
    """Return seed LF positions for the top and side model pairs."""
    top_positions: list[int] = []
    side_positions: list[int] = []
    unknown: list[tuple[int, str]] = []
    for position, labeled_frame in enumerate(labels.labeled_frames):
        source = _video_source_path(labeled_frame.video)
        view = classify_view(source)
        if view == "top_cam":
            top_positions.append(position)
        elif view in {"side1_cam", "side2_cam"}:
            side_positions.append(position)
        else:
            unknown.append((position, source))
    if unknown:
        preview = ", ".join(f"{position}:{source}" for position, source in unknown[:5])
        raise ValueError(
            f"could not classify {len(unknown)} seed labeled frame(s): {preview}"
        )
    return top_positions, side_positions


def validate_model_dirs(paths: Iterable[Path]) -> None:
    """Require every model directory and its selected checkpoint."""
    for path in paths:
        if not path.is_dir():
            raise FileNotFoundError(f"model directory not found: {path}")
        checkpoint = path / "best.ckpt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"model checkpoint not found: {checkpoint}")


def positive_int(value: str) -> int:
    """Parse a strictly positive CLI integer before starting inference."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def validate_distinct_artifact_paths(paths: dict[str, Path | None]) -> None:
    """Refuse input/output aliases that could overwrite review evidence."""
    seen: dict[Path, str] = {}
    for name, path in paths.items():
        if path is None:
            continue
        resolved = path.resolve()
        previous = seen.get(resolved)
        if previous is not None:
            raise ValueError(
                "artifact paths must be distinct: "
                f"{previous} and {name} both resolve to {resolved}"
            )
        seen[resolved] = name


def _labeled_frame_key(labeled_frame) -> tuple[str | None, str | None, int]:
    """Return the stable source identity needed to verify sparse ordering."""
    source = _video_source_path(labeled_frame.video)
    return (
        experiment_from_video(source),
        classify_view(source),
        int(labeled_frame.frame_idx),
    )


def validate_prediction_subset(
    *,
    name: str,
    seed_labels,
    positions: list[int],
    predictions,
) -> None:
    """Require predictions to retain the requested sparse LF order."""
    if len(predictions.labeled_frames) != len(positions):
        raise ValueError(
            f"{name} prediction LF count mismatch: "
            f"{len(predictions.labeled_frames)} != {len(positions)}"
        )
    for subset_index, (seed_position, predicted_frame) in enumerate(
        zip(positions, predictions.labeled_frames)
    ):
        expected = _labeled_frame_key(seed_labels.labeled_frames[seed_position])
        observed = _labeled_frame_key(predicted_frame)
        if observed != expected:
            raise ValueError(
                f"{name} prediction LF identity/order mismatch at subset "
                f"position {subset_index}: expected {expected}, got {observed}"
            )


def candidate_overlay_provenance(
    seed_provenance: dict,
    *,
    max_instances: int,
    batch_size: int,
    device: str,
) -> dict:
    """Return detached provenance that identifies the derived model overlay."""
    provenance = deepcopy(dict(seed_provenance or {}))
    provenance.update(
        {
            # The blank seed deliberately declares this false. Override it in
            # the derived overlay so generic provenance readers cannot mistake
            # model candidates for authoritative human annotations.
            "contains_model_predictions": True,
            "annotation_source": "round8_model_candidates_for_human_review",
            "candidate_overlay": {
                "kind": "model_prediction_for_human_review",
                "authoritative_annotation": False,
                "seed_safety": "separate_overlay",
                "max_instances": max_instances,
                "batch_size": batch_size,
                "device": device,
            },
        }
    )
    return provenance


def _prediction_frames_for_positions(
    *,
    positions: list[int],
    predictions,
    expected_count: int,
) -> dict[int, object]:
    if len(positions) != expected_count:
        raise ValueError(
            f"internal position count mismatch: {len(positions)} != {expected_count}"
        )
    if len(set(positions)) != len(positions):
        raise ValueError("prediction subset positions contain duplicates")
    if not positions:
        if predictions is not None:
            raise ValueError("empty prediction subset must not supply predictions")
        return {}
    if predictions is None:
        raise ValueError("non-empty prediction subset is missing predictions")
    if len(predictions.labeled_frames) != len(positions):
        raise ValueError(
            "prediction LF count mismatch: "
            f"{len(predictions.labeled_frames)} != {len(positions)}"
        )
    return dict(zip(positions, predictions.labeled_frames))


def merge_prediction_subsets(
    *,
    seed_labels,
    top_positions: list[int],
    top_predictions,
    side_positions: list[int],
    side_predictions,
    max_instances: int | None = None,
):
    """Merge top/side prediction subsets back into the seed LF order.

    SLEAP-NN versions differ in whether ``max_instances`` is enforced after
    peak grouping. Apply a final score-ranked cap here as well so the review
    overlay cannot contain more fish slots than the annotator requested.
    """
    import sleap_io as sio

    by_position = _prediction_frames_for_positions(
        positions=top_positions,
        predictions=top_predictions,
        expected_count=len(top_positions),
    )
    overlap = set(by_position).intersection(side_positions)
    if overlap:
        raise ValueError(f"top/side seed positions overlap: {sorted(overlap)[:5]}")
    by_position.update(
        _prediction_frames_for_positions(
            positions=side_positions,
            predictions=side_predictions,
            expected_count=len(side_positions),
        )
    )
    expected_positions = set(range(len(seed_labels.labeled_frames)))
    actual_positions = set(by_position)
    missing = sorted(expected_positions - actual_positions)
    extra = sorted(actual_positions - expected_positions)
    if missing or extra:
        raise ValueError(
            "prediction positions do not exactly cover seed LFs: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    ordered = [
        by_position[position]
        for position in range(len(seed_labels.labeled_frames))
    ]
    if max_instances is not None:
        if max_instances < 1:
            raise ValueError("max_instances must be at least 1")
        capped = []
        for labeled_frame in ordered:
            ranked = sorted(
                labeled_frame.instances,
                key=lambda instance: -float(getattr(instance, "score", 0.0) or 0.0),
            )
            capped.append(
                sio.LabeledFrame(
                    video=labeled_frame.video,
                    frame_idx=int(labeled_frame.frame_idx),
                    instances=ranked[:max_instances],
                )
            )
        ordered = capped
    prediction_sets = [
        predictions
        for predictions in (top_predictions, side_predictions)
        if predictions is not None
    ]
    skeletons = next(
        (
            list(predictions.skeletons)
            for predictions in prediction_sets
            if predictions.skeletons
        ),
        [],
    )
    if not skeletons:
        skeletons = list(seed_labels.skeletons)
    merged = sio.Labels(
        labeled_frames=ordered,
        skeletons=skeletons,
        provenance=dict(getattr(seed_labels, "provenance", {}) or {}),
    )
    return make_prediction_labels_self_contained(merged)


def _load_or_predict(
    *,
    name: str,
    seed_labels,
    positions: list[int],
    model_paths: list[Path],
    prediction_cache: Path,
    supplied_predictions: Path | None,
    max_instances: int,
    batch_size: int,
    device: str,
):
    import sleap_io as sio

    if supplied_predictions is not None:
        print(f"Loading supplied {name} predictions: {supplied_predictions}")
        predictions = sio.load_slp(str(supplied_predictions))
    else:
        predictions = run_strategy(
            strategy=f"prediction_review::{name}",
            model_paths=model_paths,
            subset_labels=build_subset_labels(seed_labels, positions),
            max_instances=max_instances,
            batch_size=batch_size,
            device=device,
            output_path=prediction_cache,
            prediction_mode="full_topdown",
        )
    validate_prediction_subset(
        name=name,
        seed_labels=seed_labels,
        positions=positions,
        predictions=predictions,
    )
    return predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-slp", type=Path, required=True)
    parser.add_argument(
        "--top-centroid-model",
        type=Path,
        help="Required only when the seed contains top-camera frames.",
    )
    parser.add_argument(
        "--top-centered-instance-model",
        type=Path,
        help="Required only when the seed contains top-camera frames.",
    )
    parser.add_argument(
        "--side-centroid-model",
        type=Path,
        help="Required only when the seed contains side-camera frames.",
    )
    parser.add_argument(
        "--side-centered-instance-model",
        type=Path,
        help="Required only when the seed contains side-camera frames.",
    )
    parser.add_argument("--output-slp", type=Path, required=True)
    parser.add_argument("--provenance-json", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory for the separate top and side native prediction caches.",
    )
    parser.add_argument(
        "--top-predictions",
        type=Path,
        help="Use an existing ordered top prediction SLP instead of inference.",
    )
    parser.add_argument(
        "--side-predictions",
        type=Path,
        help="Use an existing ordered side prediction SLP instead of inference.",
    )
    parser.add_argument("--max-instances", type=positive_int, default=3)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import sleap_io as sio

    top_cache = args.cache_dir / "round8_top.predictions.slp"
    side_cache = args.cache_dir / "round8_side.predictions.slp"
    validate_distinct_artifact_paths(
        {
            "seed_slp": args.seed_slp,
            "output_slp": args.output_slp,
            "provenance_json": args.provenance_json,
            "top_predictions": args.top_predictions,
            "side_predictions": args.side_predictions,
            "top_cache": top_cache if args.top_predictions is None else None,
            "side_cache": side_cache if args.side_predictions is None else None,
        }
    )
    if not args.seed_slp.is_file():
        raise FileNotFoundError(f"seed package not found: {args.seed_slp}")
    for supplied in (args.top_predictions, args.side_predictions):
        if supplied is not None and not supplied.is_file():
            raise FileNotFoundError(f"supplied prediction package not found: {supplied}")

    print(f"Loading review seed: {args.seed_slp}")
    seed_labels = sio.load_slp(str(args.seed_slp))
    top_positions, side_positions = split_positions_by_model(seed_labels)
    print(
        f"Sparse selection: {len(top_positions)} top + "
        f"{len(side_positions)} side = {len(seed_labels.labeled_frames)} LFs"
    )
    model_pairs = {
        "top": [args.top_centroid_model, args.top_centered_instance_model],
        "side": [args.side_centroid_model, args.side_centered_instance_model],
    }
    positions_by_name = {"top": top_positions, "side": side_positions}
    for name, positions in positions_by_name.items():
        pair = model_pairs[name]
        if positions and any(path is None for path in pair):
            raise ValueError(
                f"{name} seed frames require both the centroid and "
                "centered-instance model directories"
            )
        if not positions and any(path is not None for path in pair):
            print(f"Ignoring unused {name} model pair: seed has no {name} frames")
    required_model_dirs = [
        path
        for name, pair in model_pairs.items()
        if positions_by_name[name]
        for path in pair
        if path is not None
    ]
    validate_model_dirs(required_model_dirs)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    predictions_by_name = {}
    for name, positions, supplied, cache in (
        ("top", top_positions, args.top_predictions, top_cache),
        ("side", side_positions, args.side_predictions, side_cache),
    ):
        if not positions:
            if supplied is not None:
                raise ValueError(
                    f"--{name}-predictions was supplied but the seed has no "
                    f"{name} frames"
                )
            predictions_by_name[name] = None
            continue
        predictions_by_name[name] = _load_or_predict(
            name=name,
            seed_labels=seed_labels,
            positions=positions,
            model_paths=[path for path in model_pairs[name] if path is not None],
            prediction_cache=cache,
            supplied_predictions=supplied,
            max_instances=args.max_instances,
            batch_size=args.batch_size,
            device=args.device,
        )
    top_predictions = predictions_by_name["top"]
    side_predictions = predictions_by_name["side"]
    merged = merge_prediction_subsets(
        seed_labels=seed_labels,
        top_positions=top_positions,
        top_predictions=top_predictions,
        side_positions=side_positions,
        side_predictions=side_predictions,
        max_instances=args.max_instances,
    )
    merged.provenance = candidate_overlay_provenance(
        getattr(merged, "provenance", {}) or {},
        max_instances=args.max_instances,
        batch_size=args.batch_size,
        device=args.device,
    )

    args.output_slp.parent.mkdir(parents=True, exist_ok=True)
    save_prediction_slp(merged, args.output_slp)
    provenance = {
        "schema_version": "sleap_prediction_review_candidates_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "editable_model_candidates_for_human_in_the_loop_review",
        "authoritative_annotation": False,
        "seed": {
            "path": str(args.seed_slp.resolve()),
            "sha256": sha256_file(args.seed_slp),
            "labeled_frames": len(seed_labels.labeled_frames),
        },
        "selection": {
            "top_labeled_frames": len(top_positions),
            "side_labeled_frames": len(side_positions),
            "total_labeled_frames": len(seed_labels.labeled_frames),
        },
        "inference": {
            "max_instances": args.max_instances,
            "batch_size": args.batch_size,
            "device": args.device,
        },
        "models": {
            name: {
                "path": str(path),
                "best_checkpoint_sha256": sha256_file(path / "best.ckpt"),
            }
            for name, path in {
                "top_centroid": args.top_centroid_model,
                "top_centered_instance": args.top_centered_instance_model,
                "side_centroid": args.side_centroid_model,
                "side_centered_instance": args.side_centered_instance_model,
            }.items()
            if path is not None
        },
        "candidate_overlay": {
            "path": str(args.output_slp.resolve()),
            "sha256": sha256_file(args.output_slp),
        },
        "component_prediction_caches": {
            name: (
                str((supplied or cache).resolve())
                if positions_by_name[name]
                else None
            )
            for name, supplied, cache in (
                ("top", args.top_predictions, top_cache),
                ("side", args.side_predictions, side_cache),
            )
        },
    }
    args.provenance_json.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_json.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Candidate overlay: {args.output_slp}")
    print(f"Provenance:        {args.provenance_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
