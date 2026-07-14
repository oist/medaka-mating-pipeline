from __future__ import annotations

"""Evaluate an explicit-constraint fish embedding without clustering.

The evaluator scores must-link/cannot-link pairs, calibrates similarity on a dedicated
calibration split, freezes all thresholds before touching test metrics, and fails
closed on sample/window/frame/fragment leakage.  A model checkpoint may be evaluated
with bounded crop inference, or precomputed row-aligned embeddings may be supplied.

The optional prototype export is deliberately long-form: it writes every trusted-
prototype similarity for every query fragment and never emits an identity assignment.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sam2_id_experiment.src.evaluation.geometry_contrastive_embedding import (
    GEOMETRY_ID_COLUMNS,
    SPLITS,
    binary_pair_metrics,
    cross_camera_retrieval,
    fit_logistic_calibrator,
    load_membership_ids,
    membership_from_pairs,
    normalize_embeddings,
    score_pairs,
    select_threshold,
    stratified_pair_metrics,
    target_far_threshold,
    temporal_block_bootstrap_metrics,
    trusted_prototype_similarities,
    validate_split_integrity,
)


BLUE = "#3979A8"
GOLD = "#C99A3D"
INK = "#24313A"
GRID = "#D9E0E4"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=None, help="dataset directory with crops/metadata/pairs")
    p.add_argument("--metadata", default=None, help="crop metadata parquet/csv; overrides --dataset")
    p.add_argument("--pairs", default=None, help="pair parquet/csv; overrides --dataset")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--embeddings",
        help="trainer embeddings parquet/csv joined by sample ID",
    )
    source.add_argument("--checkpoint", help="SmallMedaka or ResNet18 embedding checkpoint")
    fit = p.add_mutually_exclusive_group(required=True)
    fit.add_argument("--fit-membership", help="csv/parquet/json of crop IDs used to fit the checkpoint")
    fit.add_argument("--fit-pairs", help="training-pair table; endpoint union is the fit membership")
    p.add_argument(
        "--training-report",
        default=None,
        help="trainer report.json containing the expected fit_membership SHA256; inferred beside embeddings/checkpoint when present",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-inference-crops", type=int, default=50000)

    p.add_argument("--split-col", default=None, help="existing metadata split column")
    p.add_argument("--train-value", default="train")
    p.add_argument("--calibration-value", default="calibration")
    p.add_argument("--test-value", default="test")
    p.add_argument("--train-windows", nargs="*", default=None)
    p.add_argument("--calibration-windows", nargs="*", default=None)
    p.add_argument("--test-windows", nargs="*", default=None)

    p.add_argument("--crop-id-col", default="sample_id")
    p.add_argument("--pair-a-col", default="sample_a_id")
    p.add_argument("--pair-b-col", default="sample_b_id")
    p.add_argument("--pair-label-col", default="label")
    p.add_argument("--pair-reason-col", default="pair_type")
    p.add_argument("--window-col", default="split_block_id")
    p.add_argument("--frame-col", default="frame")
    p.add_argument("--camera-col", default="view")
    p.add_argument("--fragment-col", default="local_fragment_id")
    p.add_argument("--identity-col", default="reference_identity")
    p.add_argument("--allow-geometry-eval-labels", action="store_true")

    p.add_argument("--threshold-rule", choices=["eer", "youden"], default="eer")
    p.add_argument("--target-far", type=float, default=0.01)
    p.add_argument("--ece-bins", type=int, default=10)
    p.add_argument("--bootstrap-replicates", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=13)
    p.add_argument("--bootstrap-confidence", type=float, default=0.95)
    p.add_argument("--bootstrap-block-col", default="window_pair")
    p.add_argument("--stratify-cols", nargs="*", default=[])
    p.add_argument("--max-pca-points", type=int, default=10000)

    p.add_argument("--prototype-key-col", default=None)
    p.add_argument("--prototype-state-col", default="review_state")
    p.add_argument("--trusted-state-values", nargs="+", default=["clean", "trusted"])
    p.add_argument("--prototype-splits", nargs="+", default=["train", "calibration"])
    p.add_argument("--prototype-query-splits", nargs="+", default=["test"])
    return p


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported table format: {path}")


def _first_table(root: Path, stems: list[str]) -> Path | None:
    for stem in stems:
        for suffix in (".parquet", ".csv"):
            candidate = root / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path | tuple[Path, Path], Path | None]:
    dataset = Path(args.dataset) if args.dataset else None
    metadata = (
        Path(args.metadata)
        if args.metadata
        else (_first_table(dataset, ["samples", "crop_metadata"]) if dataset else None)
    )
    if args.pairs:
        pairs: Path | tuple[Path, Path] | None = Path(args.pairs)
    elif dataset:
        combined = _first_table(dataset, ["pairs"])
        positive = _first_table(dataset, ["positive_pairs"])
        negative = _first_table(dataset, ["negative_pairs"])
        pairs = combined if combined else ((positive, negative) if positive and negative else None)
    else:
        pairs = None
    crops = dataset / "crops.h5" if dataset else None
    if metadata is None or not metadata.exists():
        raise ValueError("metadata path is required and must exist")
    if pairs is None:
        raise ValueError(
            "pairs are required: provide --pairs or a dataset with pairs, or positive_pairs and negative_pairs"
        )
    if isinstance(pairs, Path) and not pairs.exists():
        raise ValueError("pairs path is required and must exist")
    if args.checkpoint and (crops is None or not crops.exists()):
        raise ValueError("--checkpoint inference requires --dataset with crops.h5")
    return metadata, pairs, crops


def _read_pairs(source: Path | tuple[Path, Path], label_col: str) -> pd.DataFrame:
    if isinstance(source, Path):
        return _read_table(source)
    positive, negative = source
    pos = _read_table(positive).copy()
    neg = _read_table(negative).copy()
    pos[label_col] = 1
    neg[label_col] = 0
    return pd.concat([pos, neg], ignore_index=True, sort=False)


def _resolve_builder_aliases(
    metadata: pd.DataFrame, pairs: pd.DataFrame, args: argparse.Namespace
) -> None:
    """Use script-48 contract names while retaining historical explicit overrides."""
    if args.window_col not in metadata.columns:
        for alias in ("split_block_id", "split_block"):
            if alias in metadata.columns:
                args.window_col = alias
                break
    if args.fragment_col not in metadata.columns:
        for alias in ("local_fragment_id", "fragment_id", "view_tracklet"):
            if alias in metadata.columns:
                args.fragment_col = alias
                break
    if args.crop_id_col not in metadata.columns and "crop_id" in metadata.columns:
        args.crop_id_col = "crop_id"
    endpoint_aliases = [
        ("sample_a_id", "sample_b_id"),
        ("crop_a_id", "crop_b_id"),
        ("sample_id_a", "sample_id_b"),
        ("crop_id_a", "crop_id_b"),
    ]
    if args.pair_a_col not in pairs.columns or args.pair_b_col not in pairs.columns:
        found = next(((a, b) for a, b in endpoint_aliases if a in pairs and b in pairs), None)
        if found:
            args.pair_a_col, args.pair_b_col = found
    if args.pair_reason_col not in pairs.columns:
        for alias in ("pair_type", "reason", "pair_subtype"):
            if alias in pairs.columns:
                args.pair_reason_col = alias
                break
    if not args.split_col and "split" in metadata.columns:
        args.split_col = "split"


def _enrich_metadata_from_embedding_export(
    metadata: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    """Recover the trainer-assigned split from its keyed parquet/csv export."""
    if not args.embeddings or Path(args.embeddings).suffix.lower() not in {".parquet", ".csv"}:
        return metadata
    table = _read_table(args.embeddings)
    if args.crop_id_col not in metadata.columns or args.crop_id_col not in table.columns:
        return metadata
    provenance = [
        column
        for column in ("split", "split_block_id", "split_block")
        if column in table.columns and column not in metadata.columns
    ]
    if not provenance:
        return metadata
    if table[args.crop_id_col].isna().any() or table[args.crop_id_col].duplicated().any():
        raise ValueError("trainer embedding export has invalid sample IDs")
    joined = metadata.merge(
        table[[args.crop_id_col, *provenance]],
        on=args.crop_id_col,
        how="left",
        validate="one_to_one",
    )
    if joined[provenance].isna().any().any():
        raise ValueError("trainer embedding export is missing split provenance for metadata samples")
    return joined


def _assign_splits(metadata: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, str, dict]:
    result = metadata.copy()
    if args.split_col:
        if args.split_col not in result.columns:
            raise ValueError(f"metadata lacks --split-col {args.split_col!r}")
        mapping = {
            str(args.train_value): "train",
            str(args.calibration_value): "calibration",
            str(args.test_value): "test",
        }
        if len(mapping) != 3:
            raise ValueError("train/calibration/test split values must be distinct")
        split_col = "evaluation_split"
        result[split_col] = result[args.split_col].astype(str).map(mapping).fillna("excluded")
        spec = {"kind": "column", "source_column": args.split_col, "mapping": mapping}
    else:
        window_lists = {
            "train": list(args.train_windows or []),
            "calibration": list(args.calibration_windows or []),
            "test": list(args.test_windows or []),
        }
        if any(not values for values in window_lists.values()):
            raise ValueError(
                "provide --split-col or non-empty --train-windows/--calibration-windows/--test-windows"
            )
        flattened = [value for values in window_lists.values() for value in values]
        if len(flattened) != len(set(flattened)):
            raise ValueError("window lists overlap")
        if args.window_col not in result.columns:
            raise ValueError(f"metadata lacks window column {args.window_col!r}")
        inverse = {str(window): split for split, values in window_lists.items() for window in values}
        split_col = "evaluation_split"
        result[split_col] = result[args.window_col].astype(str).map(inverse).fillna("excluded")
        spec = {"kind": "window", "windows": window_lists}
    return result, split_col, spec


def _load_embeddings(
    path: str | Path, metadata: pd.DataFrame, *, crop_id_col: str
) -> tuple[np.ndarray, dict]:
    path = Path(path)
    if path.suffix.lower() in {".npy", ".npz"}:
        raise ValueError(
            "row-order embedding arrays are not accepted; provide parquet/csv with a unique sample_id join key"
        )
    if path.suffix.lower() in {".parquet", ".csv"}:
        table = _read_table(path)
        if crop_id_col not in table.columns:
            raise ValueError(f"embedding table lacks join key {crop_id_col!r}")
        if table[crop_id_col].isna().any() or table[crop_id_col].duplicated().any():
            raise ValueError(f"embedding table {crop_id_col!r} must be non-null and unique")
        columns = sorted(
            (column for column in table.columns if str(column).startswith("embedding_")),
            key=str,
        )
        if not columns:
            raise ValueError("embedding table has no embedding_* columns")
        indexed = table.set_index(crop_id_col)
        missing = set(metadata[crop_id_col]) - set(indexed.index)
        extra = set(indexed.index) - set(metadata[crop_id_col])
        if missing or extra:
            raise ValueError(
                f"embedding table/sample ID mismatch: missing={len(missing)}, extra={len(extra)}"
            )
        aligned = indexed.loc[metadata[crop_id_col], columns]
        values = aligned.to_numpy(dtype=np.float32)
        return values, {
            "alignment": f"joined_by_{crop_id_col}",
            "embedding_columns": len(columns),
        }
    raise ValueError("precomputed embeddings must be keyed .parquet or .csv")


def _infer_embeddings(
    checkpoint_path: str | Path,
    crops_path: Path,
    *,
    n_metadata: int,
    max_crops: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, dict]:
    if n_metadata > int(max_crops):
        raise ValueError(
            f"bounded inference refused: {n_metadata} crops exceeds --max-inference-crops={max_crops}"
        )
    import torch

    from sam2_id_experiment.src.id_model.geometry_contrastive import GeometryContrastiveEmbedder
    from sam2_id_experiment.src.id_model.model import SmallMedakaEmbeddingNet
    from sam2_id_experiment.src.id_model.resnet_embedder import ResNet18Embedder

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state", checkpoint.get("state_dict"))
    if not isinstance(state, dict):
        raise ValueError("checkpoint lacks model_state/state_dict")
    architecture = str(checkpoint.get("architecture", checkpoint.get("arch", "")))
    classifier_state = any(key.startswith("embedder.") for key in state)
    if architecture == "resnet18_pair_embedding" or any(
        key.startswith("features.") or key.startswith("embedder.features.") for key in state
    ):
        model = GeometryContrastiveEmbedder(
            embedding_dim=int(checkpoint.get("embedding_dim", 128)),
            pretrained=False,
            train_backbone=bool(checkpoint.get("train_backbone", False)),
            normalize=bool(checkpoint.get("normalize_embeddings", True)),
        ).to(device)
        if classifier_state:
            state = {
                key.removeprefix("embedder."): value
                for key, value in state.items()
                if key.startswith("embedder.")
            }
        architecture = "resnet18_pair_embedding"
    elif architecture.startswith("resnet18") or any("net.layer4" in key for key in state):
        model = ResNet18Embedder(
            embedding_dim=int(checkpoint.get("embedding_dim", 128)), pretrained=False
        ).to(device)
        if classifier_state:
            state = {key.removeprefix("embedder."): value for key, value in state.items() if key.startswith("embedder.")}
        architecture = "resnet18_embedder"
    else:
        model = SmallMedakaEmbeddingNet(
            embedding_dim=int(checkpoint.get("embedding_dim", 64))
        ).to(device)
        if classifier_state:
            state = {
                key.removeprefix("embedder."): value
                for key, value in state.items()
                if key.startswith("embedder.")
            }
        architecture = "small_medaka_embedding_net"
    model.load_state_dict(state)
    model.eval()

    output = []
    with h5py.File(crops_path, "r") as h5:
        key = "crops" if "crops" in h5 else "images"
        crops = h5[key]
        if len(crops) != n_metadata:
            raise ValueError("crops and metadata row counts differ")
        with torch.no_grad():
            for start in range(0, len(crops), int(batch_size)):
                batch = np.asarray(crops[start : start + int(batch_size)], dtype=np.float32) / 255.0
                tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
                output.append(model(tensor).cpu().numpy())
    return np.concatenate(output, axis=0), {
        "kind": "checkpoint",
        "path": str(checkpoint_path),
        "architecture": architecture,
        "n_inferred": n_metadata,
        "device": device,
    }


def _fit_ids(args: argparse.Namespace) -> set:
    if args.fit_membership:
        return load_membership_ids(args.fit_membership, crop_id_col=args.crop_id_col)
    table = _read_table(args.fit_pairs)
    return membership_from_pairs(table, args.pair_a_col, args.pair_b_col)


def _canonical_table_sha256(path: str | Path) -> str:
    table = _read_table(path)
    canonical = table.fillna("<NA>").astype(str).to_dict("records")
    payload = "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) for row in canonical
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _verify_fit_membership_hash(args: argparse.Namespace) -> dict:
    if not args.fit_membership:
        return {
            "available": False,
            "verified": False,
            "reason": "--fit-pairs does not provide the trainer's exact fit-membership artifact",
        }
    report_path = Path(args.training_report) if args.training_report else None
    if report_path is None:
        model_source = Path(args.embeddings or args.checkpoint)
        candidate = model_source.parent / "report.json"
        report_path = candidate if candidate.exists() else None
    if report_path is None or not report_path.exists():
        raise ValueError(
            "exact fit-membership hash verification requires --training-report or report.json beside the embeddings/checkpoint"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = str(report.get("fit_membership", {}).get("sha256", ""))
    if not expected:
        raise ValueError("training report lacks fit_membership.sha256")
    actual = _canonical_table_sha256(args.fit_membership)
    if actual != expected:
        raise ValueError(
            f"fit-membership SHA256 mismatch: expected={expected}, actual={actual}"
        )
    return {
        "available": True,
        "verified": True,
        "expected_sha256": expected,
        "actual_sha256": actual,
        "training_report": str(report_path),
    }


def _calibration_metrics(
    pair_scores: pd.DataFrame,
    *,
    label_col: str,
    threshold_rule: str,
    target_far: float,
    ece_bins: int,
) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calibration = pair_scores[pair_scores["split"] == "calibration"].copy()
    test = pair_scores[pair_scores["split"] == "test"].copy()
    for name, table in [("calibration", calibration), ("test", test)]:
        if set(table[label_col].astype(int).unique()) != {0, 1}:
            raise ValueError(f"{name} pair split must contain both pair classes")
    calibrator = fit_logistic_calibrator(
        calibration["cosine_similarity"], calibration[label_col]
    )
    pair_scores = pair_scores.copy()
    pair_scores["same_probability"] = calibrator.predict(pair_scores["cosine_similarity"])
    calibration = pair_scores[pair_scores["split"] == "calibration"].copy()
    test = pair_scores[pair_scores["split"] == "test"].copy()
    threshold = select_threshold(
        calibration["cosine_similarity"], calibration[label_col], threshold_rule
    )
    far_threshold = target_far_threshold(
        calibration["cosine_similarity"], calibration[label_col], target_far
    )
    probability_threshold = calibrator.probability_at_score(threshold)
    calibration_metrics, _, _ = binary_pair_metrics(
        calibration["cosine_similarity"],
        calibration[label_col],
        calibration["same_probability"],
        decision_threshold_score=threshold,
        target_far_score=far_threshold,
        probability_threshold=probability_threshold,
        ece_bins=ece_bins,
    )
    test_metrics, reliability, risk = binary_pair_metrics(
        test["cosine_similarity"],
        test[label_col],
        test["same_probability"],
        decision_threshold_score=threshold,
        target_far_score=far_threshold,
        probability_threshold=probability_threshold,
        ece_bins=ece_bins,
    )
    frozen = {
        "rule": threshold_rule,
        "decision_threshold_cosine": threshold,
        "decision_threshold_probability": probability_threshold,
        "target_far": float(target_far),
        "target_far_threshold_cosine": far_threshold,
        "target_far_threshold_probability": calibrator.probability_at_score(far_threshold),
        "calibrator": {
            "kind": "logistic",
            "coefficient": calibrator.coefficient,
            "intercept": calibrator.intercept,
        },
    }
    return frozen, {"calibration": calibration_metrics, "test": test_metrics}, pair_scores, reliability, risk


def _pca_projection(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    split_col: str,
    max_points: int,
) -> tuple[pd.DataFrame, dict]:
    normalized = normalize_embeddings(embeddings)
    train_mask = metadata[split_col].eq("train").to_numpy()
    if int(train_mask.sum()) < 3:
        raise ValueError("PCA requires at least three training samples")
    pca = PCA(n_components=2, random_state=0)
    pca.fit(normalized[train_mask])
    transformed = pca.transform(normalized)
    table = metadata.copy()
    table["pc1"] = transformed[:, 0]
    table["pc2"] = transformed[:, 1]
    if len(table) > int(max_points):
        selected = []
        for _, group in table.groupby(split_col, sort=False):
            n = max(1, round(int(max_points) * len(group) / len(table)))
            selected.extend(group.sample(min(n, len(group)), random_state=23).index.tolist())
        table = table.loc[sorted(set(selected))].copy()
    return table, {
        "fit_split": "train",
        "n_fit": int(train_mask.sum()),
        "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        "mean_checksum": float(pca.mean_.sum()),
    }


def _style_axis(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=18)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color="#5B6870", fontsize=9, va="bottom")
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8B979E")


def _plot_score_distributions(pair_scores: pd.DataFrame, label_col: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
    for ax, split in zip(axes, ["calibration", "test"]):
        table = pair_scores[pair_scores["split"] == split]
        for label, color, name in [(0, GOLD, "different"), (1, BLUE, "same")]:
            values = table.loc[table[label_col].astype(int) == label, "cosine_similarity"]
            ax.hist(values, bins=35, density=True, alpha=0.45, color=color, edgecolor=color, label=name)
        _style_axis(ax, f"{split.title()} pair scores", "Cosine similarity; explicit pair constraints")
        ax.set_xlabel("Cosine similarity")
        ax.legend(frameon=False)
    axes[0].set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_reliability(reliability: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.7, 5.0))
    ax.plot([0, 1], [0, 1], color=INK, linestyle="--", linewidth=1.2, label="ideal")
    ax.plot(
        reliability["mean_probability"], reliability["observed_same_rate"],
        color=BLUE, marker="o", linewidth=2, label="test bins",
    )
    _style_axis(ax, "Pair-probability reliability", "Equal-mass bins on the test split")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean calibrated probability", ylabel="Observed same-pair rate")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion(metrics: dict, out: Path) -> None:
    matrix = np.asarray([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5.3, 4.8))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center", color=INK, fontsize=12)
    ax.set_xticks([0, 1], ["different", "same"])
    ax.set_yticks([0, 1], ["different", "same"])
    ax.set_xlabel("Predicted at calibration-frozen threshold")
    ax.set_ylabel("Explicit constraint")
    ax.set_title("Test pair confusion matrix", loc="left", color=INK, fontsize=12, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_risk_coverage(risk: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.plot(risk["coverage"], risk["risk"], color=BLUE, linewidth=2)
    _style_axis(ax, "Selective pair-classification risk", "Test pairs ordered by distance from the frozen decision threshold")
    ax.set(xlim=(0, 1), xlabel="Coverage", ylabel="Cumulative error rate")
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_pca(table: pd.DataFrame, split_col: str, camera_col: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 5.4))
    markers = {"train": "o", "calibration": "s", "test": "^", "excluded": "."}
    colors = {"train": BLUE, "calibration": GOLD, "test": INK, "excluded": "#AAB2B7"}
    for split, group in table.groupby(split_col, sort=False):
        ax.scatter(group["pc1"], group["pc2"], s=9, alpha=0.45, marker=markers.get(split, "."),
                   color=colors.get(split, "#AAB2B7"), label=str(split))
    camera_note = f"; cameras={table[camera_col].nunique()}" if camera_col in table.columns else ""
    _style_axis(ax, "Train-fitted PCA projection", f"PCA fit only on train embeddings{camera_note}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    metadata_path, pairs_source, crops_path = _resolve_inputs(args)
    metadata = _read_table(metadata_path).reset_index(drop=True)
    pairs = _read_pairs(pairs_source, args.pair_label_col)
    metadata = _enrich_metadata_from_embedding_export(metadata, args)
    _resolve_builder_aliases(metadata, pairs, args)
    metadata, split_col, split_spec = _assign_splits(metadata, args)
    fit_ids = _fit_ids(args)
    fit_membership_hash = _verify_fit_membership_hash(args)
    integrity = validate_split_integrity(
        metadata,
        split_col=split_col,
        crop_id_col=args.crop_id_col,
        window_col=args.window_col or None,
        frame_col=args.frame_col or None,
        camera_col=args.camera_col or None,
        fragment_col=args.fragment_col or None,
        fit_ids=fit_ids,
    )

    if args.embeddings:
        embeddings, alignment = _load_embeddings(
            args.embeddings, metadata, crop_id_col=args.crop_id_col
        )
        embedding_source = {
            "kind": "precomputed",
            "path": str(args.embeddings),
            **alignment,
        }
    else:
        if "crop_index" in metadata.columns and not np.array_equal(
            pd.to_numeric(metadata["crop_index"], errors="raise").to_numpy(),
            np.arange(len(metadata)),
        ):
            raise ValueError(
                "checkpoint inference requires metadata rows to match crops.h5 crop_index order"
            )
        embeddings, embedding_source = _infer_embeddings(
            args.checkpoint,
            crops_path,
            n_metadata=len(metadata),
            max_crops=args.max_inference_crops,
            batch_size=args.batch_size,
            device=args.device,
        )
        np.save(out / "crop_embeddings.npy", embeddings)
    if len(embeddings) != len(metadata):
        raise ValueError("metadata and embeddings row counts differ")

    pair_scores = score_pairs(
        metadata,
        pairs,
        embeddings,
        split_col=split_col,
        crop_id_col=args.crop_id_col,
        pair_a_col=args.pair_a_col,
        pair_b_col=args.pair_b_col,
        label_col=args.pair_label_col,
        reason_col=args.pair_reason_col,
        camera_col=args.camera_col or None,
        window_col=args.window_col or None,
        frame_col=args.frame_col or None,
    )
    frozen, metrics, pair_scores, reliability, risk = _calibration_metrics(
        pair_scores,
        label_col=args.pair_label_col,
        threshold_rule=args.threshold_rule,
        target_far=args.target_far,
        ece_bins=args.ece_bins,
    )
    pair_scores["predicted_same_frozen_threshold"] = (
        pair_scores["cosine_similarity"] >= frozen["decision_threshold_cosine"]
    )

    bootstrap_table = pd.DataFrame(
        columns=[
            "metric",
            "estimate_mean",
            "ci_lower",
            "ci_upper",
            "confidence",
            "valid_replicates",
        ]
    )
    bootstrap_info = {
        "available": False,
        "block_column": args.bootstrap_block_col,
        "reason": "not attempted",
    }
    try:
        bootstrap_table, bootstrap_info = temporal_block_bootstrap_metrics(
            pair_scores[pair_scores["split"] == "test"],
            block_col=args.bootstrap_block_col,
            label_col=args.pair_label_col,
            probability_col="same_probability",
            decision_threshold_score=frozen["decision_threshold_cosine"],
            target_far_score=frozen["target_far_threshold_cosine"],
            probability_threshold=frozen["decision_threshold_probability"],
            ece_bins=args.ece_bins,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
            confidence=args.bootstrap_confidence,
        )
    except ValueError as error:
        bootstrap_info = {
            "available": False,
            "block_column": args.bootstrap_block_col,
            "reason": str(error),
            "requested_replicates": int(args.bootstrap_replicates),
            "confidence": float(args.bootstrap_confidence),
            "seed": int(args.bootstrap_seed),
        }

    # Camera and window pairs are the explicit cross-view/time-block slices.  Frame
    # pairs are intentionally excluded because they are usually one row per frame.
    automatic_strata = ["pair_kind", "camera_pair", "window_pair"]
    automatic_strata.extend(
        column for column in pair_scores if column.endswith("_any") or column.endswith("_pair")
        if column not in {"frame_pair", "camera_pair", "window_pair"}
    )
    strata = list(dict.fromkeys(automatic_strata + list(args.stratify_cols)))
    stratified = stratified_pair_metrics(
        pair_scores[pair_scores["split"] == "test"],
        probability_col="same_probability",
        label_col=args.pair_label_col,
        threshold_score=frozen["decision_threshold_cosine"],
        group_columns=strata,
    )

    included_mask = metadata[split_col].isin(SPLITS)
    pca_table, pca_info = _pca_projection(
        metadata,
        embeddings,
        split_col=split_col,
        max_points=args.max_pca_points,
    )

    retrieval_metrics = {
        "cross_camera": {"available": False, "reason": "identity column unavailable"},
        "same_frame_cross_camera": {
            "available": False,
            "reason": "identity, camera, or frame column unavailable",
        },
    }
    retrieval = pd.DataFrame()
    same_frame_retrieval = pd.DataFrame()
    identity_col = args.identity_col
    if identity_col.lower() in GEOMETRY_ID_COLUMNS and not args.allow_geometry_eval_labels:
        raise ValueError(
            f"refusing geometry-derived identity evaluation column {identity_col!r}; "
            "use an independent reference label or explicitly pass --allow-geometry-eval-labels"
        )
    test_mask = metadata[split_col].eq("test") & metadata.get(identity_col, pd.Series(index=metadata.index, dtype=object)).notna()
    if identity_col in metadata.columns and args.camera_col in metadata.columns and metadata.loc[test_mask, args.camera_col].nunique() >= 2:
        retrieval_metadata = metadata.loc[test_mask].reset_index(drop=True)
        retrieval_embeddings = embeddings[test_mask.to_numpy()]
        retrieval, retrieval_metrics["cross_camera"] = cross_camera_retrieval(
            retrieval_metadata,
            retrieval_embeddings,
            identity_col=identity_col,
            camera_col=args.camera_col,
            unit_col=args.fragment_col if args.fragment_col in metadata.columns else None,
        )
        if args.frame_col in retrieval_metadata.columns:
            same_frame_retrieval, retrieval_metrics["same_frame_cross_camera"] = cross_camera_retrieval(
                retrieval_metadata,
                retrieval_embeddings,
                identity_col=identity_col,
                camera_col=args.camera_col,
                unit_col=args.fragment_col if args.fragment_col in metadata.columns else None,
                frame_col=args.frame_col,
                same_frame_only=True,
            )

    prototype_status = {"available": False, "reason": "--prototype-key-col not supplied"}
    prototype_table = pd.DataFrame()
    if args.prototype_key_col:
        if args.prototype_state_col not in metadata.columns:
            raise ValueError(f"metadata lacks prototype state column {args.prototype_state_col!r}")
        trusted = (
            metadata[split_col].isin(args.prototype_splits)
            & metadata[args.prototype_state_col].astype(str).isin(args.trusted_state_values)
        )
        query = metadata[split_col].isin(args.prototype_query_splits)
        prototype_table = trusted_prototype_similarities(
            metadata.reset_index(drop=True),
            embeddings,
            prototype_key_col=args.prototype_key_col,
            fragment_col=args.fragment_col,
            trusted_mask=trusted,
            query_mask=query,
        )
        prototype_status = {
            "available": True,
            "n_rows": int(len(prototype_table)),
            "n_fragments": int(prototype_table["fragment_id"].nunique()),
            "n_prototypes": int(prototype_table["prototype_key"].nunique()),
            "assignment_emitted": False,
        }

    pair_scores.to_csv(out / "pair_scores.csv", index=False)
    reliability.to_csv(out / "reliability_equal_mass.csv", index=False)
    risk.to_csv(out / "risk_coverage.csv", index=False)
    stratified.to_csv(out / "stratified_pair_metrics.csv", index=False)
    bootstrap_table.to_csv(out / "temporal_block_bootstrap_ci.csv", index=False)
    pca_columns = [args.crop_id_col, split_col, "pc1", "pc2"]
    pca_columns.extend(column for column in [args.camera_col, args.window_col, identity_col] if column in pca_table.columns)
    pca_table[pca_columns].to_csv(out / "train_fitted_pca_projection.csv", index=False)
    if len(retrieval):
        retrieval.to_csv(out / "cross_camera_retrieval.csv", index=False)
    if len(same_frame_retrieval):
        same_frame_retrieval.to_csv(out / "same_frame_cross_camera_retrieval.csv", index=False)
    if len(prototype_table):
        prototype_table.to_csv(out / "trusted_prototype_fragment_similarities.csv", index=False)

    _plot_score_distributions(pair_scores, args.pair_label_col, out / "pair_score_distributions.png")
    _plot_reliability(reliability, out / "reliability_diagram.png")
    _plot_confusion(metrics["test"], out / "test_confusion_matrix.png")
    _plot_risk_coverage(risk, out / "risk_coverage.png")
    _plot_pca(pca_table, split_col, args.camera_col, out / "train_fitted_embedding_pca.png")

    report = {
        "inputs": {
            "metadata": str(metadata_path),
            "pairs": (
                [str(path) for path in pairs_source]
                if isinstance(pairs_source, tuple)
                else str(pairs_source)
            ),
            "embedding_source": embedding_source,
            "fit_membership": str(args.fit_membership or args.fit_pairs),
        },
        "split": split_spec,
        "leakage_checks": integrity,
        "fit_membership_hash": fit_membership_hash,
        "thresholds_frozen_on_calibration": frozen,
        "pair_metrics": metrics,
        "temporal_block_bootstrap": bootstrap_info,
        "retrieval": retrieval_metrics,
        "prototype_export": prototype_status,
        "pca": pca_info,
        "notes": [
            "No K-means or other clustering is used.",
            "Test labels never select a calibration model or operating threshold.",
            "Test EER threshold is descriptive only; all confusion/TAR/FAR decisions use calibration-frozen thresholds.",
            "Cross-camera and synchronized same-frame retrieval require an independent identity reference column and at least two test cameras.",
            "Camera-pair and window-pair rows in stratified_pair_metrics.csv are the cross-view and time-block slices.",
            "Prototype export contains similarities only and makes no identity assignment.",
        ],
    }
    (out / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
