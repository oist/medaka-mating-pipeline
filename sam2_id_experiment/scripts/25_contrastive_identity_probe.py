from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sam2_id_experiment.src.evaluation.contrastive_identity_probe import (
    apply_label_remap,
    best_label_remap,
    evaluate_crop_predictions,
    evaluate_tracklet_predictions,
    identity_centroids,
    majority_vote_by_tracklet,
    predict_by_centroid,
    transfer_nearest_identity,
)
from sam2_id_experiment.src.id_dataset.pair_sampler import sample_positive_negative_pairs
from sam2_id_experiment.src.id_model.infer import embed_crops
from sam2_id_experiment.src.id_model.train import train_embedding_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/evaluate contrastive fish-ID embeddings on segmentation crop datasets."
    )
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--train-reference", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-name", default="train")
    parser.add_argument("--train-max-distance-px", type=float, default=40.0)
    parser.add_argument("--within-split-frame", type=int, default=None)
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help=(
            "Target spec as name|dataset_dir|reference_csv_or_parquet|max_distance_px. "
            "May be repeated."
        ),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--positive-pairs-per-tracklet", type=int, default=1000)
    parser.add_argument("--negative-pairs-per-tracklet-pair", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--min-margin", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = Path(args.train_dataset)
    train_reference = _read_table(args.train_reference)
    train_meta = pd.read_parquet(train_dataset / "crop_metadata.parquet")
    labeled_train = transfer_nearest_identity(
        train_meta,
        train_reference,
        max_distance_px=float(args.train_max_distance_px),
    )
    labeled_train.to_parquet(output_dir / f"{args.train_name}_crop_metadata_labeled.parquet", index=False)

    if args.within_split_frame is not None:
        train_mask = labeled_train["frame"].astype(int) <= int(args.within_split_frame)
        train_for_model = _materialize_subset_dataset(
            source_dataset=train_dataset,
            labeled_metadata=labeled_train[train_mask].copy(),
            output_dir=output_dir / f"{args.train_name}_within_train_dataset",
            config=_pair_config(args),
        )
        implicit_targets = [
            {
                "name": f"{args.train_name}_within_after_{int(args.within_split_frame)}",
                "dataset": train_dataset,
                "reference": Path(args.train_reference),
                "max_distance_px": float(args.train_max_distance_px),
                "metadata_filter": labeled_train["frame"].astype(int) > int(args.within_split_frame),
            }
        ]
    else:
        train_for_model = _materialize_subset_dataset(
            source_dataset=train_dataset,
            labeled_metadata=labeled_train.copy(),
            output_dir=output_dir / f"{args.train_name}_full_train_dataset",
            config=_pair_config(args),
        )
        implicit_targets = []

    train_result = train_embedding_model(
        dataset_dir=train_for_model,
        output_dir=output_dir / f"{args.train_name}_contrastive_model",
        embedding_dim=int(args.embedding_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        device=str(args.device),
    )
    train_embed_dir = output_dir / f"{args.train_name}_train_embeddings"
    embed_crops(
        dataset_dir=train_for_model,
        model_path=train_result["model_path"],
        output_dir=train_embed_dir,
        batch_size=int(args.batch_size),
        device=str(args.device),
    )

    train_eval_meta = pd.read_parquet(train_for_model / "crop_metadata.parquet")
    train_embeddings = np.load(train_embed_dir / "crop_embeddings.npy")
    centroid_labels, centroids = identity_centroids(
        train_embeddings,
        train_eval_meta["reference_identity"].fillna("").astype(str).to_numpy(),
    )

    target_specs = implicit_targets + [_parse_target_spec(spec) for spec in args.target]
    target_reports = []
    for target in target_specs:
        report = _evaluate_target(
            target=target,
            model_path=Path(train_result["model_path"]),
            centroid_labels=centroid_labels,
            centroids=centroids,
            output_dir=output_dir,
            batch_size=int(args.batch_size),
            device=str(args.device),
            min_margin=float(args.min_margin),
        )
        target_reports.append(report)

    report = {
        "train_dataset": str(train_dataset),
        "train_reference": str(args.train_reference),
        "train_for_model": str(train_for_model),
        "train_labeled_crops": int(labeled_train["reference_label_status"].eq("labeled").sum()),
        "train_label_status_counts": _counts(labeled_train["reference_label_status"]),
        "train_identity_counts": _counts(labeled_train.loc[labeled_train["reference_label_status"].eq("labeled"), "reference_identity"]),
        "train_result": train_result,
        "centroid_labels": list(centroid_labels),
        "targets": target_reports,
    }
    report_path = output_dir / "contrastive_identity_probe_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def _pair_config(args: argparse.Namespace) -> dict:
    return {
        "positive_pairs_per_tracklet": int(args.positive_pairs_per_tracklet),
        "negative_pairs_per_tracklet_pair": int(args.negative_pairs_per_tracklet_pair),
        "random_seed": int(args.random_seed),
    }


def _materialize_subset_dataset(
    *,
    source_dataset: Path,
    labeled_metadata: pd.DataFrame,
    output_dir: Path,
    config: dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    labeled = labeled_metadata[labeled_metadata["reference_label_status"].eq("labeled")].copy()
    if labeled.empty:
        raise RuntimeError(f"No labeled crops available for {output_dir}")
    old_ids = labeled["crop_id"].astype(int).to_numpy()
    with h5py.File(source_dataset / "crops.h5", "r") as src:
        crops = src["crops"][old_ids]
    labeled = labeled.reset_index(drop=True)
    labeled["source_crop_id"] = old_ids
    labeled["crop_id"] = np.arange(len(labeled), dtype=int)
    with h5py.File(output_dir / "crops.h5", "w") as dst:
        dst.create_dataset("crops", data=crops, compression="gzip")
    labeled.to_parquet(output_dir / "crop_metadata.parquet", index=False)
    pairs = sample_positive_negative_pairs(labeled, config)
    if pairs.empty:
        raise RuntimeError(f"No contrastive pairs could be sampled for {output_dir}")
    pairs.to_parquet(output_dir / "pairs.parquet", index=False)
    debug_dir = source_dataset / "debug_crops"
    if debug_dir.exists() and not (output_dir / "debug_crops").exists():
        try:
            shutil.copytree(debug_dir, output_dir / "debug_crops", dirs_exist_ok=True)
        except OSError:
            pass
    return output_dir


def _evaluate_target(
    *,
    target: dict,
    model_path: Path,
    centroid_labels: list[str],
    centroids: np.ndarray,
    output_dir: Path,
    batch_size: int,
    device: str,
    min_margin: float,
) -> dict:
    name = str(target["name"])
    dataset = Path(target["dataset"])
    reference = _read_table(target["reference"])
    meta = pd.read_parquet(dataset / "crop_metadata.parquet")
    metadata_filter = target.get("metadata_filter")
    if metadata_filter is not None:
        meta = meta[pd.Series(metadata_filter).to_numpy()].copy()
    labeled = transfer_nearest_identity(
        meta.reset_index(drop=True),
        reference,
        max_distance_px=float(target["max_distance_px"]),
    )
    target_dir = output_dir / f"{name}_eval_dataset"
    target_dataset = _materialize_crops_without_pairs(
        source_dataset=dataset,
        labeled_metadata=labeled,
        output_dir=target_dir,
    )
    embed_dir = output_dir / f"{name}_embeddings"
    embed_crops(
        dataset_dir=target_dataset,
        model_path=model_path,
        output_dir=embed_dir,
        batch_size=batch_size,
        device=device,
    )
    eval_meta = pd.read_parquet(target_dataset / "crop_metadata.parquet")
    embeddings = np.load(embed_dir / "crop_embeddings.npy")
    pred, distance, margin = predict_by_centroid(embeddings, centroid_labels, centroids)
    remap = best_label_remap(truth := eval_meta["reference_identity"].fillna("").astype(str).to_numpy(), pred)
    remapped_pred = apply_label_remap(pred, remap)
    accepted_pred = pred.astype(object).copy()
    accepted_pred[margin < float(min_margin)] = "unassigned"
    remapped_accepted_pred = apply_label_remap(accepted_pred, remap)

    crop_report = evaluate_crop_predictions(truth, pred, margins=margin)
    remapped_crop_report = evaluate_crop_predictions(truth, remapped_pred, margins=margin)
    accepted_crop_report = evaluate_crop_predictions(truth, accepted_pred, margins=margin)
    remapped_accepted_crop_report = evaluate_crop_predictions(truth, remapped_accepted_pred, margins=margin)
    votes = majority_vote_by_tracklet(eval_meta, truth, pred)
    remapped_votes = majority_vote_by_tracklet(eval_meta, truth, remapped_pred)
    accepted_votes = majority_vote_by_tracklet(eval_meta, truth, accepted_pred)
    remapped_accepted_votes = majority_vote_by_tracklet(eval_meta, truth, remapped_accepted_pred)
    votes.to_csv(output_dir / f"{name}_tracklet_predictions.csv", index=False)
    accepted_votes.to_csv(output_dir / f"{name}_tracklet_predictions_margin{min_margin:g}.csv", index=False)

    pred_table = eval_meta.copy()
    pred_table["predicted_identity"] = pred
    pred_table["accepted_predicted_identity"] = accepted_pred
    pred_table["embedding_distance"] = distance
    pred_table["embedding_margin"] = margin
    pred_table.to_parquet(output_dir / f"{name}_crop_predictions.parquet", index=False)

    return {
        "name": name,
        "dataset": str(dataset),
        "reference": str(target["reference"]),
        "max_distance_px": float(target["max_distance_px"]),
        "n_labeled_crops": int(len(eval_meta)),
        "reference_identity_counts": _counts(eval_meta["reference_identity"]),
        "crop": crop_report,
        "crop_best_label_remap": remapped_crop_report,
        "crop_with_margin_reject": accepted_crop_report,
        "crop_with_margin_reject_best_label_remap": remapped_accepted_crop_report,
        "tracklet": evaluate_tracklet_predictions(votes),
        "tracklet_best_label_remap": evaluate_tracklet_predictions(remapped_votes),
        "tracklet_with_margin_reject": evaluate_tracklet_predictions(accepted_votes),
        "tracklet_with_margin_reject_best_label_remap": evaluate_tracklet_predictions(remapped_accepted_votes),
        "best_label_remap": {str(k): str(v) for k, v in remap.items()},
        "mean_embedding_distance": float(np.nanmean(distance)) if len(distance) else None,
        "median_embedding_margin": float(np.nanmedian(margin)) if len(margin) else None,
        "tracklet_predictions": str(output_dir / f"{name}_tracklet_predictions.csv"),
        "crop_predictions": str(output_dir / f"{name}_crop_predictions.parquet"),
    }


def _materialize_crops_without_pairs(
    *,
    source_dataset: Path,
    labeled_metadata: pd.DataFrame,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    labeled = labeled_metadata[labeled_metadata["reference_label_status"].eq("labeled")].copy()
    if labeled.empty:
        raise RuntimeError(f"No labeled target crops available for {output_dir}")
    old_ids = labeled["crop_id"].astype(int).to_numpy()
    with h5py.File(source_dataset / "crops.h5", "r") as src:
        crops = src["crops"][old_ids]
    labeled = labeled.reset_index(drop=True)
    labeled["source_crop_id"] = old_ids
    labeled["crop_id"] = np.arange(len(labeled), dtype=int)
    with h5py.File(output_dir / "crops.h5", "w") as dst:
        dst.create_dataset("crops", data=crops, compression="gzip")
    labeled.to_parquet(output_dir / "crop_metadata.parquet", index=False)
    pd.DataFrame(columns=["crop_a_id", "crop_b_id", "label", "reason"]).to_parquet(
        output_dir / "pairs.parquet", index=False
    )
    return output_dir


def _parse_target_spec(spec: str) -> dict:
    parts = spec.split("|")
    if len(parts) != 4:
        raise ValueError("Target specs must be name|dataset_dir|reference_csv_or_parquet|max_distance_px")
    return {
        "name": parts[0],
        "dataset": Path(parts[1]),
        "reference": Path(parts[2]),
        "max_distance_px": float(parts[3]),
    }


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)


def _counts(values: pd.Series | np.ndarray | list[str]) -> dict[str, int]:
    series = pd.Series(values).fillna("").astype(str)
    return {str(k): int(v) for k, v in series.value_counts().sort_index().to_dict().items()}


if __name__ == "__main__":
    main()
