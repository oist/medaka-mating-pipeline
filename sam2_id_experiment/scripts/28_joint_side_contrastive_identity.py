from __future__ import annotations

import argparse
import json
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
    predict_by_centroid,
    transfer_nearest_identity,
)
from sam2_id_experiment.src.evaluation.joint_side_identity import (
    cross_view_tracklet_correspondence,
    sample_supervised_identity_pairs,
    tracklet_majority_table,
)
from sam2_id_experiment.src.id_model.infer import embed_crops
from sam2_id_experiment.src.id_model.train import train_embedding_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a joint side-camera contrastive ID model with explicit cross-view identity pairs."
    )
    parser.add_argument(
        "--side",
        action="append",
        required=True,
        help="Side spec as name|dataset_dir|reference_csv_or_parquet|max_distance_px. Repeat for side1/side2.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-frame", type=int, default=None, help="Train on frames <= split, evaluate > split.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--positive-pairs-per-identity", type=int, default=1500)
    parser.add_argument("--cross-view-positive-pairs-per-identity", type=int, default=1500)
    parser.add_argument("--negative-pairs-per-identity-pair", type=int, default=1500)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument(
        "--min-tracklet-crops",
        type=int,
        default=1,
        help="Keep only labeled (view, tracklet, identity) groups with at least this many crops.",
    )
    parser.add_argument(
        "--max-train-crops-per-identity-view",
        type=int,
        default=None,
        help="Optional deterministic cap for train crops per reference identity and side view.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    side_specs = [_parse_side_spec(spec) for spec in args.side]
    labeled_dataset = _materialize_labeled_sides(
        side_specs=side_specs,
        output_dir=output_dir / "joint_labeled_dataset",
    )
    raw_meta = pd.read_parquet(labeled_dataset / "crop_metadata.parquet")
    all_meta = _filter_labeled_metadata(raw_meta, min_tracklet_crops=int(args.min_tracklet_crops))

    if args.split_frame is None:
        train_meta = all_meta.copy()
        eval_meta = all_meta.copy()
        train_name = "joint_all_train_dataset"
        eval_name = "joint_all_eval_dataset"
    else:
        train_meta = all_meta[all_meta["frame"].astype(int) <= int(args.split_frame)].copy()
        eval_meta = all_meta[all_meta["frame"].astype(int) > int(args.split_frame)].copy()
        train_name = f"joint_train_to_f{int(args.split_frame)}"
        eval_name = f"joint_eval_after_f{int(args.split_frame)}"
    train_meta = _cap_train_crops(
        train_meta,
        max_crops_per_identity_view=args.max_train_crops_per_identity_view,
        random_seed=int(args.random_seed),
    )
    if train_meta.empty:
        raise RuntimeError("No crops in the train split.")
    if eval_meta.empty:
        raise RuntimeError("No crops in the eval split.")

    train_dataset = _materialize_subset(
        source_dataset=labeled_dataset,
        metadata=train_meta,
        output_dir=output_dir / train_name,
        pair_config=_pair_config(args),
    )
    eval_dataset = _materialize_subset(
        source_dataset=labeled_dataset,
        metadata=eval_meta,
        output_dir=output_dir / eval_name,
        pair_config=None,
    )

    train_result = train_embedding_model(
        dataset_dir=train_dataset,
        output_dir=output_dir / "joint_side_contrastive_model",
        embedding_dim=int(args.embedding_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        device=str(args.device),
    )

    train_embed_dir = output_dir / "train_embeddings"
    embed_crops(
        dataset_dir=train_dataset,
        model_path=train_result["model_path"],
        output_dir=train_embed_dir,
        batch_size=int(args.batch_size),
        device=str(args.device),
    )
    train_eval_meta = pd.read_parquet(train_dataset / "crop_metadata.parquet")
    train_embeddings = np.load(train_embed_dir / "crop_embeddings.npy")
    centroid_labels, centroids = identity_centroids(
        train_embeddings,
        train_eval_meta["reference_identity"].fillna("").astype(str).to_numpy(),
    )

    eval_embed_dir = output_dir / "eval_embeddings"
    embed_crops(
        dataset_dir=eval_dataset,
        model_path=train_result["model_path"],
        output_dir=eval_embed_dir,
        batch_size=int(args.batch_size),
        device=str(args.device),
    )
    eval_meta = pd.read_parquet(eval_dataset / "crop_metadata.parquet")
    eval_embeddings = np.load(eval_embed_dir / "crop_embeddings.npy")
    pred, distance, margin = predict_by_centroid(eval_embeddings, centroid_labels, centroids)
    truth = eval_meta["reference_identity"].fillna("").astype(str).to_numpy()
    remap = best_label_remap(truth, pred)
    remapped_pred = apply_label_remap(pred, remap)
    accepted_pred = pred.astype(object).copy()
    accepted_pred[margin < float(args.min_margin)] = "unassigned"
    remapped_accepted_pred = apply_label_remap(accepted_pred, remap)

    votes = tracklet_majority_table(eval_meta, truth, pred)
    remapped_votes = tracklet_majority_table(eval_meta, truth, remapped_pred)
    accepted_votes = tracklet_majority_table(eval_meta, truth, accepted_pred)
    remapped_accepted_votes = tracklet_majority_table(eval_meta, truth, remapped_accepted_pred)
    votes.to_csv(output_dir / "eval_tracklet_predictions.csv", index=False)
    accepted_votes.to_csv(output_dir / f"eval_tracklet_predictions_margin{float(args.min_margin):g}.csv", index=False)

    pred_table = eval_meta.copy()
    pred_table["predicted_identity"] = pred
    pred_table["remapped_predicted_identity"] = remapped_pred
    pred_table["accepted_predicted_identity"] = accepted_pred
    pred_table["remapped_accepted_predicted_identity"] = remapped_accepted_pred
    pred_table["embedding_distance"] = distance
    pred_table["embedding_margin"] = margin
    pred_table.to_parquet(output_dir / "eval_crop_predictions.parquet", index=False)

    correspond = cross_view_tracklet_correspondence(eval_meta, eval_embeddings, predicted=pred)
    correspond.to_csv(output_dir / "cross_view_tracklet_correspondence.csv", index=False)
    best_correspond = correspond[correspond["rank_for_left_tracklet"].eq(1)].copy() if not correspond.empty else correspond
    best_correspond.to_csv(output_dir / "cross_view_tracklet_best_matches.csv", index=False)
    remapped_correspond = cross_view_tracklet_correspondence(eval_meta, eval_embeddings, predicted=remapped_pred)
    remapped_correspond.to_csv(output_dir / "cross_view_tracklet_correspondence_remapped.csv", index=False)
    best_remapped_correspond = (
        remapped_correspond[remapped_correspond["rank_for_left_tracklet"].eq(1)].copy()
        if not remapped_correspond.empty
        else remapped_correspond
    )
    best_remapped_correspond.to_csv(output_dir / "cross_view_tracklet_best_matches_remapped.csv", index=False)

    report = {
        "side_specs": [
            {
                "name": spec["name"],
                "dataset": str(spec["dataset"]),
                "reference": str(spec["reference"]),
                "max_distance_px": float(spec["max_distance_px"]),
            }
            for spec in side_specs
        ],
        "labeled_dataset": str(labeled_dataset),
        "train_dataset": str(train_dataset),
        "eval_dataset": str(eval_dataset),
        "split_frame": int(args.split_frame) if args.split_frame is not None else None,
        "min_tracklet_crops": int(args.min_tracklet_crops),
        "max_train_crops_per_identity_view": (
            int(args.max_train_crops_per_identity_view)
            if args.max_train_crops_per_identity_view is not None
            else None
        ),
        "n_raw_labeled_crops": int(len(raw_meta)),
        "n_filtered_labeled_crops": int(len(all_meta)),
        "n_labeled_crops": int(len(all_meta)),
        "n_train_crops": int(len(train_eval_meta)),
        "n_eval_crops": int(len(eval_meta)),
        "train_identity_counts": _counts(train_eval_meta["reference_identity"]),
        "eval_identity_counts": _counts(eval_meta["reference_identity"]),
        "train_source_counts": _counts(train_eval_meta["source_view"]),
        "eval_source_counts": _counts(eval_meta["source_view"]),
        "train_pair_counts": _counts(pd.read_parquet(train_dataset / "pairs.parquet")["label"]),
        "train_pair_reason_prefix_counts": _pair_reason_prefix_counts(train_dataset / "pairs.parquet"),
        "train_result": train_result,
        "centroid_labels": list(centroid_labels),
        "crop": evaluate_crop_predictions(truth, pred, margins=margin),
        "crop_best_label_remap": evaluate_crop_predictions(truth, remapped_pred, margins=margin),
        "crop_with_margin_reject": evaluate_crop_predictions(truth, accepted_pred, margins=margin),
        "crop_with_margin_reject_best_label_remap": evaluate_crop_predictions(
            truth, remapped_accepted_pred, margins=margin
        ),
        "tracklet": evaluate_tracklet_predictions(votes),
        "tracklet_best_label_remap": evaluate_tracklet_predictions(remapped_votes),
        "tracklet_with_margin_reject": evaluate_tracklet_predictions(accepted_votes),
        "tracklet_with_margin_reject_best_label_remap": evaluate_tracklet_predictions(remapped_accepted_votes),
        "best_label_remap": {str(k): str(v) for k, v in remap.items()},
        "cross_view_best_match_accuracy": (
            float(best_correspond["same_truth_identity"].mean()) if not best_correspond.empty else None
        ),
        "cross_view_best_predicted_match_accuracy": (
            float(best_correspond["same_predicted_identity"].mean()) if not best_correspond.empty else None
        ),
        "cross_view_best_remapped_predicted_match_accuracy": (
            float(best_remapped_correspond["same_predicted_identity"].mean())
            if not best_remapped_correspond.empty
            else None
        ),
        "cross_view_tracklet_correspondence": str(output_dir / "cross_view_tracklet_correspondence.csv"),
        "cross_view_tracklet_best_matches": str(output_dir / "cross_view_tracklet_best_matches.csv"),
        "cross_view_tracklet_correspondence_remapped": str(
            output_dir / "cross_view_tracklet_correspondence_remapped.csv"
        ),
        "cross_view_tracklet_best_matches_remapped": str(
            output_dir / "cross_view_tracklet_best_matches_remapped.csv"
        ),
        "eval_crop_predictions": str(output_dir / "eval_crop_predictions.parquet"),
    }
    (output_dir / "joint_side_contrastive_identity_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


def _parse_side_spec(spec: str) -> dict:
    parts = spec.split("|")
    if len(parts) != 4:
        raise ValueError("Side specs must be name|dataset_dir|reference_csv_or_parquet|max_distance_px")
    return {
        "name": parts[0],
        "dataset": Path(parts[1]),
        "reference": Path(parts[2]),
        "max_distance_px": float(parts[3]),
    }


def _materialize_labeled_sides(*, side_specs: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_chunks: list[np.ndarray] = []
    meta_chunks: list[pd.DataFrame] = []
    next_crop_id = 0
    for spec in side_specs:
        dataset = Path(spec["dataset"])
        meta = pd.read_parquet(dataset / "crop_metadata.parquet")
        ref = _read_table(spec["reference"])
        labeled = transfer_nearest_identity(meta, ref, max_distance_px=float(spec["max_distance_px"]))
        labeled = labeled[labeled["reference_label_status"].eq("labeled")].copy()
        if labeled.empty:
            continue
        old_ids = labeled["crop_id"].astype(int).to_numpy()
        with h5py.File(dataset / "crops.h5", "r") as h5:
            source_crops = h5["crops"][...]
        crops = source_crops[old_ids]
        labeled = labeled.reset_index(drop=True)
        labeled["source_view"] = str(spec["name"])
        labeled["source_dataset"] = str(dataset)
        labeled["source_crop_id"] = old_ids
        labeled["source_tracklet_id"] = labeled["tracklet_id"].astype(int)
        labeled["crop_id"] = np.arange(next_crop_id, next_crop_id + len(labeled), dtype=int)
        next_crop_id += len(labeled)
        crop_chunks.append(crops)
        meta_chunks.append(labeled)

    if not crop_chunks:
        raise RuntimeError("No labeled crops found across side specs.")
    with h5py.File(output_dir / "crops.h5", "w") as h5:
        h5.create_dataset("crops", data=np.concatenate(crop_chunks, axis=0), compression="gzip")
    pd.concat(meta_chunks, ignore_index=True).to_parquet(output_dir / "crop_metadata.parquet", index=False)
    pd.DataFrame(columns=["crop_a_id", "crop_b_id", "label", "reason"]).to_parquet(output_dir / "pairs.parquet", index=False)
    return output_dir


def _filter_labeled_metadata(metadata: pd.DataFrame, *, min_tracklet_crops: int) -> pd.DataFrame:
    if min_tracklet_crops <= 1:
        return metadata.copy()
    group_cols = ["source_view", "source_tracklet_id", "reference_identity"]
    missing = [col for col in group_cols if col not in metadata.columns]
    if missing:
        raise ValueError(f"Cannot filter tracklets because metadata is missing columns: {missing}")
    sizes = metadata.groupby(group_cols, dropna=False)["crop_id"].transform("size")
    filtered = metadata[sizes >= int(min_tracklet_crops)].copy()
    if filtered.empty:
        raise RuntimeError(f"No labeled crops remain after --min-tracklet-crops {min_tracklet_crops}.")
    return filtered


def _cap_train_crops(
    metadata: pd.DataFrame,
    *,
    max_crops_per_identity_view: int | None,
    random_seed: int,
) -> pd.DataFrame:
    if max_crops_per_identity_view is None:
        return metadata.copy()
    cap = int(max_crops_per_identity_view)
    if cap <= 0:
        raise ValueError("--max-train-crops-per-identity-view must be positive when provided.")
    group_cols = ["source_view", "reference_identity"]
    missing = [col for col in group_cols if col not in metadata.columns]
    if missing:
        raise ValueError(f"Cannot cap train crops because metadata is missing columns: {missing}")

    chunks: list[pd.DataFrame] = []
    for key_index, (_, group) in enumerate(metadata.groupby(group_cols, sort=True, dropna=False)):
        if len(group) <= cap:
            chunks.append(group)
            continue
        chunks.append(group.sample(n=cap, random_state=int(random_seed) + key_index))
    if not chunks:
        return metadata.iloc[0:0].copy()
    return pd.concat(chunks, ignore_index=True).sort_values("frame").reset_index(drop=True)


def _materialize_subset(
    *,
    source_dataset: Path,
    metadata: pd.DataFrame,
    output_dir: Path,
    pair_config: dict | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = metadata.reset_index(drop=True).copy()
    old_ids = selected["crop_id"].astype(int).to_numpy()
    with h5py.File(source_dataset / "crops.h5", "r") as src:
        crops = src["crops"][...][old_ids]
    selected["joint_crop_id"] = old_ids
    selected["crop_id"] = np.arange(len(selected), dtype=int)
    with h5py.File(output_dir / "crops.h5", "w") as dst:
        dst.create_dataset("crops", data=crops, compression="gzip")
    selected.to_parquet(output_dir / "crop_metadata.parquet", index=False)
    if pair_config is None:
        pairs = pd.DataFrame(columns=["crop_a_id", "crop_b_id", "label", "reason"])
    else:
        pairs = sample_supervised_identity_pairs(selected, **pair_config)
        if pairs.empty:
            raise RuntimeError(f"No supervised pairs could be sampled for {output_dir}")
    pairs.to_parquet(output_dir / "pairs.parquet", index=False)
    return output_dir


def _pair_config(args: argparse.Namespace) -> dict:
    return {
        "positive_pairs_per_identity": int(args.positive_pairs_per_identity),
        "cross_view_positive_pairs_per_identity": int(args.cross_view_positive_pairs_per_identity),
        "negative_pairs_per_identity_pair": int(args.negative_pairs_per_identity_pair),
        "random_seed": int(args.random_seed),
    }


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)


def _counts(values: pd.Series | np.ndarray | list) -> dict[str, int]:
    series = pd.Series(values).fillna("").astype(str)
    return {str(k): int(v) for k, v in series.value_counts().sort_index().to_dict().items()}


def _pair_reason_prefix_counts(path: Path) -> dict[str, int]:
    pairs = pd.read_parquet(path)
    if pairs.empty:
        return {}
    prefixes = pairs["reason"].fillna("").astype(str).str.split(":").str[0]
    return _counts(prefixes)


if __name__ == "__main__":
    main()
