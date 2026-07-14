from __future__ import annotations

"""Leakage-safe evaluation helpers for explicit-constraint embeddings.

The functions in this module evaluate pair similarity and retrieval.  They never
cluster embeddings and never turn reconstruction-local geometry IDs into predicted
biological identities.  Any identity column is reference data used only for optional
retrieval evaluation or for explicitly trusted prototype keys.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)


SPLITS = ("train", "calibration", "test")
GEOMETRY_ID_COLUMNS = {
    "global_id",
    "raw_global_id",
    "fish_id",
    "final_fish_id",
    "geometry_id",
    "geometry_source_idx_local",
    "selected_source_idx",
    "component_id",
    "component_id_provenance",
    "local_fragment_id",
    "fragment_id",
    "tracklet_id",
    "node_id",
}


@dataclass(frozen=True)
class LogisticPairCalibrator:
    coefficient: float
    intercept: float

    def predict(self, scores: Sequence[float]) -> np.ndarray:
        values = self.coefficient * np.asarray(scores, dtype=float) + self.intercept
        values = np.clip(values, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-values))

    def probability_at_score(self, score: float) -> float:
        return float(self.predict([score])[0])


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("embeddings must be a non-empty 2D array")
    if not np.isfinite(values).all():
        raise ValueError("embeddings contain NaN or infinite values")
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("embeddings contain zero-norm rows")
    return values / norm


def load_membership_ids(path: str | Path, crop_id_col: str = "crop_id") -> set:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        table = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        payload = pd.read_json(path)
        table = payload if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload)
    else:
        raise ValueError(f"unsupported fit-membership format: {path.suffix}")
    if crop_id_col not in table.columns:
        raise ValueError(f"fit membership lacks {crop_id_col!r}")
    result = set(table[crop_id_col].dropna().tolist())
    if not result:
        raise ValueError("fit membership is empty")
    return result


def membership_from_pairs(pairs: pd.DataFrame, a_col: str, b_col: str) -> set:
    missing = {a_col, b_col} - set(pairs.columns)
    if missing:
        raise ValueError(f"fit-pairs table missing columns: {sorted(missing)}")
    result = set(pairs[a_col].dropna()) | set(pairs[b_col].dropna())
    if not result:
        raise ValueError("fit-pairs membership is empty")
    return result


def validate_split_integrity(
    metadata: pd.DataFrame,
    *,
    split_col: str,
    crop_id_col: str,
    window_col: str | None,
    frame_col: str | None,
    camera_col: str | None,
    fragment_col: str | None,
    fit_ids: Iterable,
) -> dict:
    required = {split_col, crop_id_col}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata missing split columns: {sorted(missing)}")
    if metadata[crop_id_col].isna().any() or metadata[crop_id_col].duplicated().any():
        raise ValueError(f"{crop_id_col} must be non-null and unique")

    included = metadata[metadata[split_col].isin(SPLITS)].copy()
    counts = included[split_col].value_counts()
    absent = [split for split in SPLITS if int(counts.get(split, 0)) == 0]
    if absent:
        raise ValueError(f"empty required splits: {absent}")

    def overlapping_values(column: str, label: str) -> None:
        if column not in included.columns:
            raise ValueError(f"metadata lacks configured {label} column {column!r}")
        overlap = included.groupby(column, dropna=False)[split_col].nunique()
        bad = overlap[overlap > 1]
        if len(bad):
            examples = bad.index.astype(str).tolist()[:5]
            raise ValueError(f"{label} leakage across splits: {len(bad)} values; examples={examples}")

    if window_col:
        overlapping_values(window_col, "window")
    if fragment_col:
        overlapping_values(fragment_col, "fragment")
    if frame_col:
        if camera_col and camera_col not in included.columns:
            raise ValueError(f"metadata lacks configured camera column {camera_col!r}")
        # Every synchronized camera observation at one experiment/frame belongs
        # to the same split. Camera must not be part of this key: including it
        # would allow top and side views of the same instant to leak across
        # train/calibration/test.
        frame_key = [
            column
            for column in ("experiment", "session", "recording")
            if column in included.columns
        ] + [frame_col]
        overlap = included.groupby(frame_key, dropna=False)[split_col].nunique()
        bad = overlap[overlap > 1]
        if len(bad):
            raise ValueError(f"frame leakage across splits: {len(bad)} frame keys")

    all_ids = set(included[crop_id_col])
    fit = set(fit_ids)
    missing_fit = fit - all_ids
    if missing_fit:
        raise ValueError(f"fit membership contains {len(missing_fit)} unknown or excluded crop IDs")
    train_ids = set(included.loc[included[split_col] == "train", crop_id_col])
    outside_train = fit - train_ids
    if outside_train:
        raise ValueError(
            f"model-fit leakage: {len(outside_train)} fit crop IDs are outside the train split"
        )
    if not fit:
        raise ValueError("fit membership is empty")

    return {
        "split_counts": {split: int(counts.get(split, 0)) for split in SPLITS},
        "n_fit_ids": int(len(fit)),
        "fit_is_train_subset": True,
        "window_disjoint": bool(window_col),
        "frame_disjoint": bool(frame_col),
        "synchronized_camera_frames_kept_together": bool(frame_col),
        "fragment_disjoint": bool(fragment_col),
    }


def score_pairs(
    metadata: pd.DataFrame,
    pairs: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    split_col: str,
    crop_id_col: str = "crop_id",
    pair_a_col: str = "crop_a_id",
    pair_b_col: str = "crop_b_id",
    label_col: str = "label",
    reason_col: str = "reason",
    camera_col: str | None = "view",
    window_col: str | None = "window",
    frame_col: str | None = "frame",
) -> pd.DataFrame:
    missing = {pair_a_col, pair_b_col, label_col} - set(pairs.columns)
    if missing:
        raise ValueError(f"pairs missing columns: {sorted(missing)}")
    if len(metadata) != len(embeddings):
        raise ValueError("metadata and embeddings row counts differ")
    normalized = normalize_embeddings(embeddings)
    row_for_id = pd.Series(np.arange(len(metadata)), index=metadata[crop_id_col]).to_dict()
    unknown = (set(pairs[pair_a_col]) | set(pairs[pair_b_col])) - set(row_for_id)
    if unknown:
        raise ValueError(f"pairs reference {len(unknown)} unknown crop IDs")
    if not set(pd.to_numeric(pairs[label_col], errors="raise").astype(int).unique()) <= {0, 1}:
        raise ValueError("pair labels must be binary 0/1")

    a_rows = np.asarray([row_for_id[value] for value in pairs[pair_a_col]], dtype=int)
    b_rows = np.asarray([row_for_id[value] for value in pairs[pair_b_col]], dtype=int)
    split_a = metadata.iloc[a_rows][split_col].to_numpy()
    split_b = metadata.iloc[b_rows][split_col].to_numpy()
    relevant_a = np.isin(split_a, SPLITS)
    relevant_b = np.isin(split_b, SPLITS)
    boundary = relevant_a ^ relevant_b
    if bool(boundary.any()):
        raise ValueError(f"{int(boundary.sum())} pairs cross the included/excluded split boundary")
    cross_split = relevant_a & relevant_b & (split_a != split_b)
    if bool(cross_split.any()):
        raise ValueError(f"pair leakage: {int(cross_split.sum())} pairs cross train/calibration/test")
    keep = relevant_a & relevant_b

    result = pairs.loc[keep].copy().reset_index(drop=True)
    a_rows = a_rows[keep]
    b_rows = b_rows[keep]
    result["split"] = split_a[keep]
    result["cosine_similarity"] = np.sum(normalized[a_rows] * normalized[b_rows], axis=1)
    result["pair_kind"] = (
        result[reason_col].fillna("unspecified").astype(str).str.split(":", n=1).str[0]
        if reason_col in result.columns
        else "unspecified"
    )

    def add_endpoint_context(column: str | None, name: str) -> None:
        if not column or column not in metadata.columns:
            return
        left = metadata.iloc[a_rows][column].to_numpy()
        right = metadata.iloc[b_rows][column].to_numpy()
        result[f"{name}_a"] = left
        result[f"{name}_b"] = right
        result[f"{name}_pair"] = [str(a) if a == b else f"{a}|{b}" for a, b in zip(left, right)]

    add_endpoint_context(camera_col, "camera")
    add_endpoint_context(window_col, "window")
    add_endpoint_context(frame_col, "frame")

    hard_columns = [
        column
        for column in metadata.columns
        if any(
            token in column.lower()
            for token in (
                "hard",
                "occl",
                "merge",
                "crossing",
                "trunc",
                "contact",
                "partial",
            )
        )
    ]
    for column in hard_columns:
        left = metadata.iloc[a_rows][column].to_numpy()
        right = metadata.iloc[b_rows][column].to_numpy()
        if pd.api.types.is_bool_dtype(metadata[column]) or set(pd.Series(metadata[column]).dropna().unique()) <= {0, 1, True, False}:
            result[f"{column}_any"] = np.asarray(left, dtype=bool) | np.asarray(right, dtype=bool)
        else:
            result[f"{column}_pair"] = [str(a) if a == b else f"{a}|{b}" for a, b in zip(left, right)]
    return result


def fit_logistic_calibrator(scores: Sequence[float], labels: Sequence[int]) -> LogisticPairCalibrator:
    x = np.asarray(scores, dtype=float).reshape(-1, 1)
    y = np.asarray(labels, dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("calibration pairs must contain both classes")
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=0)
    model.fit(x, y)
    coefficient = float(model.coef_[0, 0])
    if coefficient <= 0:
        raise ValueError("calibration slope is non-positive; higher similarity is not same-fish evidence")
    return LogisticPairCalibrator(coefficient, float(model.intercept_[0]))


def select_threshold(scores: Sequence[float], labels: Sequence[int], rule: str = "eer") -> float:
    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("threshold selection requires both classes")
    fpr, tpr, thresholds = roc_curve(y, values)
    finite = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
    if rule == "eer":
        index = int(np.argmin(np.abs((1.0 - tpr) - fpr)))
    elif rule == "youden":
        index = int(np.argmax(tpr - fpr))
    else:
        raise ValueError(f"unknown threshold rule: {rule}")
    return float(thresholds[index])


def target_far_threshold(scores: Sequence[float], labels: Sequence[int], target_far: float) -> float:
    if not 0.0 <= float(target_far) < 1.0:
        raise ValueError("target FAR must be in [0, 1)")
    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    negatives = np.sort(values[y == 0])
    if not len(negatives):
        raise ValueError("target FAR threshold requires negative calibration pairs")
    allowed_fp = int(np.floor(float(target_far) * len(negatives)))
    if allowed_fp <= 0:
        return float(np.nextafter(negatives[-1], np.inf))
    return float(np.nextafter(negatives[-allowed_fp], np.inf))


def equal_mass_reliability(
    probabilities: Sequence[float], labels: Sequence[int], bins: int = 10
) -> tuple[pd.DataFrame, float]:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    if len(p) != len(y) or not len(p):
        raise ValueError("probabilities and labels must be non-empty and aligned")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    order = np.argsort(p, kind="mergesort")
    chunks = [chunk for chunk in np.array_split(order, min(int(bins), len(order))) if len(chunk)]
    rows = []
    ece = 0.0
    for bin_index, idx in enumerate(chunks):
        confidence = float(p[idx].mean())
        observed = float(y[idx].mean())
        weight = len(idx) / len(p)
        ece += weight * abs(observed - confidence)
        rows.append(
            {
                "bin": int(bin_index),
                "n": int(len(idx)),
                "probability_min": float(p[idx].min()),
                "probability_max": float(p[idx].max()),
                "mean_probability": confidence,
                "observed_same_rate": observed,
                "absolute_gap": abs(observed - confidence),
            }
        )
    return pd.DataFrame(rows), float(ece)


def risk_coverage_curve(
    probabilities: Sequence[float], labels: Sequence[int], decision_threshold: float
) -> tuple[pd.DataFrame, float]:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    prediction = (p >= float(decision_threshold)).astype(int)
    confidence = np.abs(p - float(decision_threshold))
    order = np.argsort(-confidence, kind="mergesort")
    error = (prediction[order] != y[order]).astype(float)
    cumulative_risk = np.cumsum(error) / np.arange(1, len(error) + 1)
    coverage = np.arange(1, len(error) + 1) / len(error)
    table = pd.DataFrame(
        {
            "coverage": coverage,
            "risk": cumulative_risk,
            "n_retained": np.arange(1, len(error) + 1),
        }
    )
    aurc = float(np.trapezoid(cumulative_risk, coverage)) if len(table) > 1 else float(cumulative_risk[0])
    return table, aurc


def binary_pair_metrics(
    scores: Sequence[float],
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    decision_threshold_score: float,
    target_far_score: float,
    probability_threshold: float,
    ece_bins: int = 10,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    score = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("overall pair metrics require both classes")
    fpr, tpr, roc_thresholds = roc_curve(y, score)
    finite_threshold = np.isfinite(roc_thresholds)
    finite_indices = np.flatnonzero(finite_threshold)
    eer_index = int(
        finite_indices[
            np.argmin(np.abs((1.0 - tpr[finite_threshold]) - fpr[finite_threshold]))
        ]
    )
    eer = float((fpr[eer_index] + (1.0 - tpr[eer_index])) / 2.0)
    prediction = (score >= float(decision_threshold_score)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    far_prediction = score >= float(target_far_score)
    neg = y == 0
    pos = y == 1
    reliability, ece = equal_mass_reliability(p, y, bins=ece_bins)
    risk, aurc = risk_coverage_curve(p, y, probability_threshold)
    metrics = {
        "n_pairs": int(len(y)),
        "n_same": int(pos.sum()),
        "n_different": int(neg.sum()),
        "auroc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "eer": eer,
        "eer_threshold_test_descriptive_only": float(roc_thresholds[eer_index]),
        "brier": float(brier_score_loss(y, p)),
        "nll": float(log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1])),
        "ece_equal_mass": ece,
        "aurc": aurc,
        "decision_threshold_score_frozen_from_calibration": float(decision_threshold_score),
        "decision_threshold_probability_frozen_from_calibration": float(probability_threshold),
        "accuracy_at_frozen_threshold": float((prediction == y).mean()),
        "balanced_accuracy_at_frozen_threshold": float(balanced_accuracy_score(y, prediction)),
        "precision_at_frozen_threshold": float(tp / max(tp + fp, 1)),
        "recall_at_frozen_threshold": float(tp / max(tp + fn, 1)),
        "far_at_frozen_threshold": float(fp / max(fp + tn, 1)),
        "tar_at_frozen_threshold": float(tp / max(tp + fn, 1)),
        "target_far_threshold_score_frozen_from_calibration": float(target_far_score),
        "far_at_calibration_target_far_threshold": float(far_prediction[neg].mean()),
        "tar_at_calibration_target_far_threshold": float(far_prediction[pos].mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics, reliability, risk


def stratified_pair_metrics(
    pair_scores: pd.DataFrame,
    *,
    probability_col: str,
    label_col: str,
    threshold_score: float,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    rows = []
    total_pairs = max(int(len(pair_scores)), 1)
    for column in group_columns:
        if column not in pair_scores.columns:
            continue
        for value, group in pair_scores.groupby(column, dropna=False):
            y = group[label_col].astype(int).to_numpy()
            score = group["cosine_similarity"].to_numpy()
            pred = score >= float(threshold_score)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            row = {
                "stratifier": column,
                "stratum": str(value),
                "n_pairs": int(len(group)),
                "n_same": int((y == 1).sum()),
                "n_different": int((y == 0).sum()),
                "accuracy": float((pred == y).mean()),
                "precision_at_frozen_threshold": float(tp / max(tp + fp, 1)),
                "recall_at_frozen_threshold": float(tp / max(tp + fn, 1)),
                "far_at_frozen_threshold": float(fp / max(fp + tn, 1)),
                "tar_at_frozen_threshold": float(tp / max(tp + fn, 1)),
                "coverage_fraction_of_test_pairs": float(len(group) / total_pairs),
                "mean_probability": float(group[probability_col].mean()),
                "auroc": np.nan,
                "average_precision": np.nan,
            }
            if set(np.unique(y)) == {0, 1}:
                row["auroc"] = float(roc_auc_score(y, score))
                row["average_precision"] = float(average_precision_score(y, score))
            rows.append(row)
    return pd.DataFrame(rows)


def temporal_block_bootstrap_metrics(
    pair_scores: pd.DataFrame,
    *,
    block_col: str,
    label_col: str,
    probability_col: str,
    decision_threshold_score: float,
    target_far_score: float,
    probability_threshold: float,
    ece_bins: int = 10,
    replicates: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, dict]:
    """Bootstrap pair metrics by resampling complete temporal blocks.

    Pair rows are never sampled independently. Each replicate samples block IDs
    with replacement and includes every pair from each sampled block, preserving
    within-block dependence. Replicates lacking either class are skipped rather
    than receiving an invented metric.
    """
    if block_col not in pair_scores.columns:
        raise ValueError(f"bootstrap block column {block_col!r} is unavailable")
    if int(replicates) <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("bootstrap confidence must be in (0, 1)")

    table = pair_scores.dropna(subset=[block_col]).copy()
    blocks = list(pd.unique(table[block_col]))
    if len(blocks) < 2:
        raise ValueError("temporal-block bootstrap requires at least two non-empty blocks")
    groups = {block: group for block, group in table.groupby(block_col, sort=False)}
    metric_names = (
        "auroc",
        "average_precision",
        "eer",
        "precision_at_frozen_threshold",
        "recall_at_frozen_threshold",
        "far_at_frozen_threshold",
        "tar_at_frozen_threshold",
        "brier",
        "nll",
        "ece_equal_mass",
        "aurc",
    )
    draws: dict[str, list[float]] = {name: [] for name in metric_names}
    rng = np.random.default_rng(int(seed))
    invalid_replicates = 0
    for _ in range(int(replicates)):
        sampled = rng.choice(blocks, size=len(blocks), replace=True)
        replicate = pd.concat([groups[block] for block in sampled], ignore_index=True)
        labels = replicate[label_col].astype(int).to_numpy()
        if set(np.unique(labels)) != {0, 1}:
            invalid_replicates += 1
            continue
        metrics, _, _ = binary_pair_metrics(
            replicate["cosine_similarity"],
            labels,
            replicate[probability_col],
            decision_threshold_score=decision_threshold_score,
            target_far_score=target_far_score,
            probability_threshold=probability_threshold,
            ece_bins=ece_bins,
        )
        for name in metric_names:
            draws[name].append(float(metrics[name]))

    valid_replicates = int(replicates) - invalid_replicates
    if valid_replicates == 0:
        raise ValueError("no valid temporal-block bootstrap replicate contained both classes")
    alpha = (1.0 - float(confidence)) / 2.0
    rows = []
    for name in metric_names:
        values = np.asarray(draws[name], dtype=float)
        rows.append(
            {
                "metric": name,
                "estimate_mean": float(values.mean()),
                "ci_lower": float(np.quantile(values, alpha)),
                "ci_upper": float(np.quantile(values, 1.0 - alpha)),
                "confidence": float(confidence),
                "valid_replicates": int(len(values)),
            }
        )
    info = {
        "available": True,
        "method": "nonparametric bootstrap over complete temporal blocks",
        "block_column": block_col,
        "n_blocks": int(len(blocks)),
        "requested_replicates": int(replicates),
        "valid_replicates": int(valid_replicates),
        "invalid_single_class_replicates": int(invalid_replicates),
        "confidence": float(confidence),
        "seed": int(seed),
    }
    return pd.DataFrame(rows), info


def average_precision_from_ranking(relevance: np.ndarray) -> float:
    relevant = np.asarray(relevance, dtype=bool)
    if not relevant.any():
        return float("nan")
    precision = np.cumsum(relevant) / np.arange(1, len(relevant) + 1)
    return float(precision[relevant].mean())


def cross_camera_retrieval(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    identity_col: str,
    camera_col: str,
    unit_col: str | None = None,
    frame_col: str | None = None,
    same_frame_only: bool = False,
) -> tuple[pd.DataFrame, dict]:
    for column in [identity_col, camera_col]:
        if column not in metadata.columns:
            raise ValueError(f"retrieval metadata lacks {column!r}")
    normalized = normalize_embeddings(embeddings)
    table = metadata.reset_index(drop=True).copy()
    if same_frame_only:
        if not frame_col or frame_col not in table.columns:
            raise ValueError("same-frame retrieval requires a valid frame column")
        grouping = [camera_col, identity_col, frame_col]
        if unit_col:
            if unit_col not in table.columns:
                raise ValueError(f"retrieval metadata lacks unit column {unit_col!r}")
            grouping.insert(0, unit_col)
        records = []
        vectors = []
        for keys, group in table.groupby(grouping, dropna=False, sort=False):
            if unit_col:
                unit, camera, identity, frame = keys
            else:
                camera, identity, frame = keys
                unit = int(group.index[0])
            vector = normalized[group.index.to_numpy()].mean(axis=0)
            vector /= max(np.linalg.norm(vector), 1e-12)
            records.append(
                {
                    "unit": unit,
                    "camera": camera,
                    "identity": identity,
                    "frame": frame,
                    "n_samples": len(group),
                }
            )
            vectors.append(vector)
        items = pd.DataFrame(records)
        values = np.asarray(vectors)
    elif unit_col:
        if unit_col not in table.columns:
            raise ValueError(f"retrieval metadata lacks unit column {unit_col!r}")
        records = []
        vectors = []
        for keys, group in table.groupby([unit_col, camera_col, identity_col], dropna=False, sort=False):
            unit, camera, identity = keys
            vector = normalized[group.index.to_numpy()].mean(axis=0)
            vector /= max(np.linalg.norm(vector), 1e-12)
            records.append({"unit": unit, "camera": camera, "identity": identity, "n_samples": len(group)})
            vectors.append(vector)
        items = pd.DataFrame(records)
        values = np.asarray(vectors)
    else:
        items = pd.DataFrame(
            {
                "unit": table.index,
                "camera": table[camera_col].to_numpy(),
                "identity": table[identity_col].to_numpy(),
                "n_samples": 1,
            }
        )
        values = normalized

    rows = []
    for query_index, query in items.iterrows():
        gallery_mask = items["camera"].to_numpy() != query["camera"]
        if same_frame_only:
            gallery_mask &= items["frame"].to_numpy() == query["frame"]
        gallery_indices = np.flatnonzero(gallery_mask)
        if not len(gallery_indices):
            continue
        relevance = items.iloc[gallery_indices]["identity"].to_numpy() == query["identity"]
        if not relevance.any():
            continue
        similarity = values[gallery_indices] @ values[query_index]
        order = np.argsort(-similarity, kind="mergesort")
        ranked_relevance = relevance[order]
        rows.append(
            {
                "query_unit": query["unit"],
                "query_camera": query["camera"],
                "query_identity": query["identity"],
                "query_frame": query.get("frame", np.nan),
                "n_gallery": int(len(order)),
                "rank1_correct": bool(ranked_relevance[0]),
                "average_precision": average_precision_from_ranking(ranked_relevance),
                "top_similarity": float(similarity[order[0]]),
                "top_gallery_unit": items.iloc[gallery_indices[order[0]]]["unit"],
                "top_gallery_camera": items.iloc[gallery_indices[order[0]]]["camera"],
                "top_gallery_frame": items.iloc[gallery_indices[order[0]]].get("frame", np.nan),
            }
        )
    result = pd.DataFrame(rows)
    metrics = {
        "available": bool(len(result)),
        "n_queries": int(len(result)),
        "rank1": float(result["rank1_correct"].mean()) if len(result) else None,
        "map": float(result["average_precision"].mean()) if len(result) else None,
        "unit": unit_col or "crop",
        "same_frame_only": bool(same_frame_only),
    }
    return result, metrics


def trusted_prototype_similarities(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    prototype_key_col: str,
    fragment_col: str,
    trusted_mask: Sequence[bool],
    query_mask: Sequence[bool],
) -> pd.DataFrame:
    if prototype_key_col.lower() in GEOMETRY_ID_COLUMNS:
        raise ValueError("geometry IDs cannot be used as trusted biological prototype keys")
    for column in [prototype_key_col, fragment_col]:
        if column not in metadata.columns:
            raise ValueError(f"prototype export metadata lacks {column!r}")
    normalized = normalize_embeddings(embeddings)
    trusted = np.array(trusted_mask, dtype=bool, copy=True)
    query = np.array(query_mask, dtype=bool, copy=True)
    if len(trusted) != len(metadata) or len(query) != len(metadata):
        raise ValueError("prototype masks do not align with metadata")
    trusted &= metadata[prototype_key_col].notna().to_numpy()
    if not trusted.any():
        raise ValueError("no trusted prototype samples")
    if not query.any():
        raise ValueError("no query fragment samples")

    prototypes = []
    for key, group in metadata[trusted].groupby(prototype_key_col, sort=True):
        idx = group.index.to_numpy()
        vector = normalized[idx].mean(axis=0)
        vector /= max(np.linalg.norm(vector), 1e-12)
        prototypes.append((key, vector, int(len(idx))))
    rows = []
    for fragment, group in metadata[query].groupby(fragment_col, sort=True):
        idx = group.index.to_numpy()
        vector = normalized[idx].mean(axis=0)
        vector /= max(np.linalg.norm(vector), 1e-12)
        for key, prototype, n_prototype in prototypes:
            rows.append(
                {
                    "fragment_id": fragment,
                    "prototype_key": key,
                    "similarity": float(vector @ prototype),
                    "n_fragment_samples": int(len(idx)),
                    "n_prototype_samples": n_prototype,
                }
            )
    return pd.DataFrame(rows)
