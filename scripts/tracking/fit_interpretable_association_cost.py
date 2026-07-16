from __future__ import annotations

"""Fit an interpretable motion-association cost on reviewed mouth-3D tracks.

Each sample pairs one persistent previous identity state with one current 3D
candidate.  The target is one when their reviewed ``H`` identities match and
zero otherwise.  Evaluation is restricted to supplied contiguous held-out
episodes; training excludes those episodes, their guard bands, a configurable
context margin, and any pair whose motion history touches an excluded frame.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


FEATURE_BUILDER_VERSION = "reviewed_mouth_association_pairs_v1"
MODEL_SCHEMA_VERSION = "interpretable_association_cost_v1"
EPSILON = 1e-9

FEATURES = (
    "distance_mm",
    "prediction_error_mm",
    "tangent_discontinuity",
    "acceleration_discontinuity_mm_per_frame2",
    "frame_gap",
    "mean_reprojection_error_px",
    "max_reprojection_error_px",
    "top_cam_match_px",
    "side1_cam_match_px",
    "side2_cam_match_px",
    "all_views_agree",
    "top_cam_missing",
    "side1_cam_missing",
    "side2_cam_missing",
    "any_view_missing",
    "motion_available",
    "acceleration_available",
)

SOURCE_COLUMNS_BY_FEATURE = {
    "mean_reprojection_error_px": ("mean_reprojection_error_px",),
    "max_reprojection_error_px": ("max_reprojection_error_px",),
    "top_cam_match_px": ("top_cam_match_px",),
    "side1_cam_match_px": ("side1_cam_match_px",),
    "side2_cam_match_px": ("side2_cam_match_px",),
    "all_views_agree": ("all_views_agree",),
    "top_cam_missing": ("top_mouth_x", "top_mouth_y", "top_cam_match_px"),
    "side1_cam_missing": ("side1_mouth_x", "side1_mouth_y", "side1_cam_match_px"),
    "side2_cam_missing": ("side2_mouth_x", "side2_mouth_y", "side2_cam_match_px"),
}


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _as_bool_float(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return 1.0
        if normalized in {"false", "0", "no", "n"}:
            return 0.0
        return float("nan")
    return float(bool(value))


def _view_missing(candidate: pd.Series, view: str) -> float:
    required = (f"{view}_mouth_x", f"{view}_mouth_y", f"{view}_cam_match_px")
    if any(column not in candidate.index for column in required):
        return 1.0
    return float(any(not np.isfinite(_as_float(candidate[column])) for column in required))


def _kinematic_features(
    history: list[tuple[int, np.ndarray]],
    *,
    candidate_frame: int,
    candidate_position: np.ndarray,
) -> dict[str, float]:
    previous_frame, previous_position = history[-1]
    gap = int(candidate_frame - previous_frame)
    displacement = candidate_position - previous_position
    distance = float(np.linalg.norm(displacement))
    candidate_velocity = displacement / max(1, gap)
    features = {
        "distance_mm": distance,
        "prediction_error_mm": float("nan"),
        "tangent_discontinuity": float("nan"),
        "acceleration_discontinuity_mm_per_frame2": float("nan"),
        "frame_gap": float(gap),
        "motion_available": 0.0,
        "acceleration_available": 0.0,
    }
    if len(history) < 2:
        return features

    older_frame, older_position = history[-2]
    previous_gap = max(1, int(previous_frame - older_frame))
    previous_velocity = (previous_position - older_position) / previous_gap
    predicted_position = previous_position + previous_velocity * gap
    features["prediction_error_mm"] = float(
        np.linalg.norm(candidate_position - predicted_position)
    )
    features["motion_available"] = 1.0
    previous_speed = float(np.linalg.norm(previous_velocity))
    candidate_speed = float(np.linalg.norm(candidate_velocity))
    if previous_speed > EPSILON and candidate_speed > EPSILON:
        cosine = float(
            np.dot(previous_velocity, candidate_velocity)
            / (previous_speed * candidate_speed)
        )
        features["tangent_discontinuity"] = 1.0 - float(np.clip(cosine, -1.0, 1.0))

    if len(history) < 3:
        return features
    oldest_frame, oldest_position = history[-3]
    older_gap = max(1, int(older_frame - oldest_frame))
    older_velocity = (older_position - oldest_position) / older_gap
    previous_midpoint_gap = max(EPSILON, 0.5 * (older_gap + previous_gap))
    previous_acceleration = (previous_velocity - older_velocity) / previous_midpoint_gap
    candidate_midpoint_gap = max(EPSILON, 0.5 * (previous_gap + gap))
    candidate_acceleration = (
        candidate_velocity - previous_velocity
    ) / candidate_midpoint_gap
    features["acceleration_discontinuity_mm_per_frame2"] = float(
        np.linalg.norm(candidate_acceleration - previous_acceleration)
    )
    features["acceleration_available"] = 1.0
    return features


def build_association_pairs(
    clean_track: pd.DataFrame,
    *,
    identity_column: str = "human_identity",
    max_frame_gap: int | None = 100,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build all previous-track/current-candidate pairs in deterministic order."""
    required = {"frame", identity_column, "X", "Y", "Z"}
    missing = required - set(clean_track.columns)
    if missing:
        raise ValueError("clean track is missing columns: " + ", ".join(sorted(missing)))
    if max_frame_gap is not None and max_frame_gap < 1:
        raise ValueError("max_frame_gap must be positive or None")

    data = clean_track.copy()
    data["frame"] = pd.to_numeric(data["frame"], errors="coerce")
    for coordinate in ("X", "Y", "Z"):
        data[coordinate] = pd.to_numeric(data[coordinate], errors="coerce")
    valid = (
        data["frame"].notna()
        & data[identity_column].notna()
        & np.isfinite(data[["X", "Y", "Z"]]).all(axis=1)
    )
    dropped_rows = int((~valid).sum())
    data = data.loc[valid].copy()
    data["frame"] = data["frame"].astype(np.int64)
    data[identity_column] = data[identity_column].astype(str)
    if data.duplicated(["frame", identity_column]).any():
        raise ValueError("clean track has duplicate frame/identity rows")
    identities = sorted(data[identity_column].unique())
    if len(identities) < 2:
        raise ValueError("at least two reviewed identities are required")
    data.sort_values(["frame", identity_column], kind="stable", inplace=True)

    histories: dict[str, list[tuple[int, np.ndarray]]] = {identity: [] for identity in identities}
    pair_rows: list[dict[str, Any]] = []
    for frame, candidates in data.groupby("frame", sort=True, observed=True):
        current_frame = int(frame)
        for previous_identity in identities:
            history = histories[previous_identity]
            if not history:
                continue
            previous_frame = history[-1][0]
            if max_frame_gap is not None and current_frame - previous_frame > max_frame_gap:
                continue
            for candidate in candidates.itertuples(index=False, name=None):
                candidate_series = pd.Series(candidate, index=candidates.columns)
                candidate_identity = str(candidate_series[identity_column])
                candidate_position = candidate_series[["X", "Y", "Z"]].to_numpy(dtype=float)
                motion = _kinematic_features(
                    history,
                    candidate_frame=current_frame,
                    candidate_position=candidate_position,
                )
                top_missing = _view_missing(candidate_series, "top")
                side1_missing = _view_missing(candidate_series, "side1")
                side2_missing = _view_missing(candidate_series, "side2")
                row: dict[str, Any] = {
                    "frame": current_frame,
                    "previous_identity": previous_identity,
                    "candidate_identity": candidate_identity,
                    "label_same_identity": int(previous_identity == candidate_identity),
                    "history_frame_1": int(history[-1][0]),
                    "history_frame_2": int(history[-2][0]) if len(history) >= 2 else np.nan,
                    "history_frame_3": int(history[-3][0]) if len(history) >= 3 else np.nan,
                    **motion,
                    "mean_reprojection_error_px": _as_float(
                        candidate_series.get("mean_reprojection_error_px")
                    ),
                    "max_reprojection_error_px": _as_float(
                        candidate_series.get("max_reprojection_error_px")
                    ),
                    "top_cam_match_px": _as_float(candidate_series.get("top_cam_match_px")),
                    "side1_cam_match_px": _as_float(candidate_series.get("side1_cam_match_px")),
                    "side2_cam_match_px": _as_float(candidate_series.get("side2_cam_match_px")),
                    "all_views_agree": _as_bool_float(candidate_series.get("all_views_agree")),
                    "top_cam_missing": top_missing,
                    "side1_cam_missing": side1_missing,
                    "side2_cam_missing": side2_missing,
                    "any_view_missing": float(bool(top_missing or side1_missing or side2_missing)),
                }
                pair_rows.append(row)
        # Update all persistent states only after every pairing for this frame is built.
        for candidate in candidates.itertuples(index=False, name=None):
            candidate_series = pd.Series(candidate, index=candidates.columns)
            identity = str(candidate_series[identity_column])
            position = candidate_series[["X", "Y", "Z"]].to_numpy(dtype=float)
            histories[identity].append((current_frame, position))
            histories[identity] = histories[identity][-3:]

    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        raise ValueError("no association pairs could be built from the clean track")
    pairs.sort_values(
        ["frame", "previous_identity", "candidate_identity"],
        kind="stable",
        inplace=True,
        ignore_index=True,
    )
    source_support = {
        feature: {
            "required_columns": list(columns),
            "missing_columns": [column for column in columns if column not in clean_track],
        }
        for feature, columns in SOURCE_COLUMNS_BY_FEATURE.items()
    }
    return pairs, {
        "input_rows": int(len(clean_track)),
        "usable_rows": int(len(data)),
        "dropped_nonfinite_or_unlabelled_rows": dropped_rows,
        "identities": identities,
        "pair_rows": int(len(pairs)),
        "positive_pairs": int(pairs["label_same_identity"].sum()),
        "negative_pairs": int((pairs["label_same_identity"] == 0).sum()),
        "source_feature_support": source_support,
    }


def load_heldout_episodes(path: Path) -> pd.DataFrame:
    episodes = pd.read_csv(path, low_memory=False)
    required = {"start_frame", "end_frame"}
    missing = required - set(episodes.columns)
    if missing:
        raise ValueError("episode CSV is missing columns: " + ", ".join(sorted(missing)))
    if episodes.empty:
        raise ValueError("episode CSV is empty")
    result = episodes.copy()
    for column in ("start_frame", "end_frame"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(np.int64)
    for column, fallback in (
        ("guard_start_frame", "start_frame"),
        ("guard_end_frame", "end_frame"),
    ):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
            result[column] = result[column].fillna(result[fallback]).astype(np.int64)
    if (result["end_frame"] < result["start_frame"]).any():
        raise ValueError("episode end_frame precedes start_frame")
    result.sort_values(["start_frame", "end_frame"], kind="stable", inplace=True)
    result.reset_index(drop=True, inplace=True)
    if len(result) > 1 and (
        result["start_frame"].to_numpy()[1:] <= result["end_frame"].to_numpy()[:-1]
    ).any():
        raise ValueError("held-out episodes overlap")
    if "episode_id" not in result:
        result["episode_id"] = [f"episode_{index:03d}" for index in range(1, len(result) + 1)]
    return result


def _inside_any(values: np.ndarray, intervals: Iterable[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(len(values), dtype=bool)
    finite = np.isfinite(values)
    for start, end in intervals:
        mask |= finite & (values >= start) & (values <= end)
    return mask


def split_pairs_by_episodes(
    pairs: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    context_margin_frames: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Label pair rows as train, heldout, or excluded_context without leakage."""
    if context_margin_frames < 0:
        raise ValueError("context_margin_frames must be non-negative")
    heldout_intervals = [
        (int(row.start_frame), int(row.end_frame))
        for row in episodes.itertuples(index=False)
    ]
    excluded_intervals: list[tuple[int, int]] = []
    for row in episodes.itertuples(index=False):
        guard_start = getattr(row, "guard_start_frame", row.start_frame)
        guard_end = getattr(row, "guard_end_frame", row.end_frame)
        start = min(int(row.start_frame), int(guard_start)) - context_margin_frames
        end = max(int(row.end_frame), int(guard_end)) + context_margin_frames
        excluded_intervals.append((start, end))

    result = pairs.copy()
    current = result["frame"].to_numpy(dtype=float)
    heldout = _inside_any(current, heldout_intervals)
    touches_exclusion = _inside_any(current, excluded_intervals)
    history_columns = ["history_frame_1", "history_frame_2", "history_frame_3"]
    history_touches = np.zeros(len(result), dtype=bool)
    for column in history_columns:
        history_touches |= _inside_any(
            pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=float),
            excluded_intervals,
        )
    train = ~heldout & ~touches_exclusion & ~history_touches
    split = np.full(len(result), "excluded_context", dtype=object)
    split[train] = "train"
    split[heldout] = "heldout"
    result["split"] = split
    if np.any(train & heldout):
        raise AssertionError("held-out pairs leaked into training")
    heldout_frames = set(result.loc[heldout, "frame"].astype(int))
    train_current_frames = set(result.loc[train, "frame"].astype(int))
    return result, {
        "policy": "contiguous held-out episodes; no random row split",
        "context_margin_frames": int(context_margin_frames),
        "heldout_intervals": [[start, end] for start, end in heldout_intervals],
        "training_exclusion_intervals": [[start, end] for start, end in excluded_intervals],
        "pair_counts": {
            key: int(value)
            for key, value in result["split"].value_counts().sort_index().items()
        },
        "heldout_training_current_frame_intersection": int(
            len(heldout_frames & train_current_frames)
        ),
        "training_pairs_with_excluded_history": int((train & history_touches).sum()),
    }


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = probabilities >= 0.5
    metrics: dict[str, Any] = {
        "n_pairs": int(len(labels)),
        "positive_pairs": int(labels.sum()),
        "negative_pairs": int((labels == 0).sum()),
    }
    if not len(labels):
        metrics.update(
            {"accuracy": None, "balanced_accuracy": None, "roc_auc": None, "log_loss": None}
        )
        return metrics
    metrics["accuracy"] = float(accuracy_score(labels, predictions))
    metrics["balanced_accuracy"] = (
        float(balanced_accuracy_score(labels, predictions))
        if len(np.unique(labels)) == 2
        else None
    )
    metrics["roc_auc"] = (
        float(roc_auc_score(labels, probabilities))
        if len(np.unique(labels)) == 2
        else None
    )
    metrics["log_loss"] = float(log_loss(labels, probabilities, labels=[0, 1]))
    return metrics


def _ranking_accuracy(rows: pd.DataFrame, probabilities: np.ndarray) -> dict[str, Any]:
    scored = rows[["frame", "previous_identity", "candidate_identity", "label_same_identity"]].copy()
    scored["probability_same_identity"] = probabilities
    correct = []
    for _, group in scored.groupby(["frame", "previous_identity"], sort=False, observed=True):
        if not (group["label_same_identity"] == 1).any() or len(group) < 2:
            continue
        best = group.sort_values(
            ["probability_same_identity", "candidate_identity"],
            ascending=[False, True],
            kind="stable",
        ).iloc[0]
        correct.append(int(best["label_same_identity"] == 1))
    return {
        "eligible_previous_track_frames": int(len(correct)),
        "top1_candidate_accuracy": float(np.mean(correct)) if correct else None,
    }


def fit_interpretable_model(
    split_pairs: pd.DataFrame,
    *,
    regularization_c: float = 1.0,
    max_iter: int = 2000,
) -> dict[str, Any]:
    if regularization_c <= 0:
        raise ValueError("regularization_c must be positive")
    train = split_pairs[split_pairs["split"] == "train"].copy()
    heldout = split_pairs[split_pairs["split"] == "heldout"].copy()
    if train.empty:
        raise ValueError("no training pairs remain after held-out/context exclusion")
    if heldout.empty:
        raise ValueError("no held-out pairs overlap the supplied episodes")
    train_labels = train["label_same_identity"].to_numpy(dtype=np.int64)
    if len(np.unique(train_labels)) != 2:
        raise ValueError("training pairs must contain both positive and negative labels")

    support: dict[str, Any] = {}
    fitted_features: list[str] = []
    for feature in FEATURES:
        values = pd.to_numeric(train[feature], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        record: dict[str, Any] = {
            "training_finite_values": int(len(finite)),
            "training_missing_values": int(len(values) - len(finite)),
            "heldout_missing_values": int(
                pd.to_numeric(heldout[feature], errors="coerce").isna().sum()
            ),
        }
        if not len(finite):
            record.update({"fitted": False, "reason": "no finite training values"})
        elif len(np.unique(finite)) < 2:
            record.update({"fitted": False, "reason": "constant in training data"})
        else:
            record["fitted"] = True
            fitted_features.append(feature)
        support[feature] = record
    if not fitted_features:
        raise ValueError("no non-constant association features are available for fitting")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = train[fitted_features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    x_heldout = heldout[fitted_features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    train_imputed = imputer.fit_transform(x_train)
    heldout_imputed = imputer.transform(x_heldout)
    train_scaled = scaler.fit_transform(train_imputed)
    heldout_scaled = scaler.transform(heldout_imputed)
    model = LogisticRegression(
        C=regularization_c,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=0,
        solver="lbfgs",
    )
    model.fit(train_scaled, train_labels)
    heldout_labels = heldout["label_same_identity"].to_numpy(dtype=np.int64)
    train_probability = model.predict_proba(train_scaled)[:, 1]
    heldout_probability = model.predict_proba(heldout_scaled)[:, 1]

    standardized_coefficients = {
        feature: float(value)
        for feature, value in zip(fitted_features, model.coef_[0])
    }
    scaling = {
        feature: {
            "imputation_median": float(imputer.statistics_[index]),
            "mean": float(scaler.mean_[index]),
            "scale": float(scaler.scale_[index]),
        }
        for index, feature in enumerate(fitted_features)
    }
    # Association minimizes cost, whereas logistic regression maximizes the
    # same-identity logit. Thus raw cost weights are -coefficient / scale.
    raw_cost_weights = {
        feature: float(-model.coef_[0, index] / scaler.scale_[index])
        for index, feature in enumerate(fitted_features)
    }
    raw_cost_intercept = float(
        -model.intercept_[0]
        + sum(
            model.coef_[0, index] * scaler.mean_[index] / scaler.scale_[index]
            for index in range(len(fitted_features))
        )
    )
    expected_logit_sign = {
        "distance_mm": "negative",
        "prediction_error_mm": "negative",
        "tangent_discontinuity": "negative",
        "acceleration_discontinuity_mm_per_frame2": "negative",
        "frame_gap": "negative",
        "mean_reprojection_error_px": "negative",
        "max_reprojection_error_px": "negative",
        "top_cam_match_px": "negative",
        "side1_cam_match_px": "negative",
        "side2_cam_match_px": "negative",
        "all_views_agree": "positive",
        "top_cam_missing": "negative",
        "side1_cam_missing": "negative",
        "side2_cam_missing": "negative",
        "any_view_missing": "negative",
    }
    sign_audit = {}
    for feature, expected in expected_logit_sign.items():
        coefficient = standardized_coefficients.get(feature)
        sign_audit[feature] = {
            "expected_same_identity_logit_sign": expected,
            "observed_standardized_coefficient": coefficient,
            "matches_expectation": None
            if coefficient is None
            else bool(coefficient < 0 if expected == "negative" else coefficient > 0),
        }

    train_metrics = _metrics(train_labels, train_probability)
    train_metrics.update(_ranking_accuracy(train, train_probability))
    heldout_metrics = _metrics(heldout_labels, heldout_probability)
    heldout_metrics.update(_ranking_accuracy(heldout, heldout_probability))
    return {
        "model_type": "standardized_logistic_regression",
        "target": "label_same_identity",
        "interpretation": (
            "Higher model logit means a more likely same-identity pair; the serialized "
            "raw association cost is the negative logit and should be minimized."
        ),
        "hyperparameters": {
            "C": float(regularization_c),
            "class_weight": "balanced",
            "max_iter": int(max_iter),
            "solver": "lbfgs",
            "random_state": 0,
        },
        "fitted_features": fitted_features,
        "unsupported_or_constant_features": {
            feature: record["reason"]
            for feature, record in support.items()
            if not record["fitted"]
        },
        "feature_support": support,
        "standardized_logit": {
            "intercept": float(model.intercept_[0]),
            "coefficients": standardized_coefficients,
        },
        "scaling": scaling,
        "raw_association_cost": {
            "intercept": raw_cost_intercept,
            "weights": raw_cost_weights,
            "formula": "cost = intercept + sum(weight[feature] * imputed_raw_feature)",
        },
        "coefficient_sign_audit": sign_audit,
        "metrics": {"train": train_metrics, "heldout": heldout_metrics},
    }


def _table_provenance(path: Path, table: pd.DataFrame) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "byte_size": int(path.stat().st_size),
        "row_count": int(len(table)),
        "schema": {
            "columns": [str(column) for column in table.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in table.dtypes.items()},
        },
    }


def run(
    clean_track_csv: Path,
    heldout_episodes_csv: Path,
    *,
    identity_column: str = "human_identity",
    max_frame_gap: int | None = 100,
    context_margin_frames: int = 100,
    regularization_c: float = 1.0,
    max_iter: int = 2000,
) -> tuple[dict[str, Any], pd.DataFrame]:
    clean_track = pd.read_csv(clean_track_csv, low_memory=False)
    episode_source_table = pd.read_csv(heldout_episodes_csv, low_memory=False)
    episodes = load_heldout_episodes(heldout_episodes_csv)
    pairs, pair_summary = build_association_pairs(
        clean_track,
        identity_column=identity_column,
        max_frame_gap=max_frame_gap,
    )
    split_pairs, split_audit = split_pairs_by_episodes(
        pairs,
        episodes,
        context_margin_frames=context_margin_frames,
    )
    model = fit_interpretable_model(
        split_pairs,
        regularization_c=regularization_c,
        max_iter=max_iter,
    )
    report = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "feature_builder_version": FEATURE_BUILDER_VERSION,
        "coordinate_definition": "reviewed multiview SLEAP mouth-derived 3D point",
        "pair_definition": {
            "positive": "previous track and current candidate have the same reviewed H identity",
            "negative": "previous track and current candidate have different reviewed H identities",
        },
        "provenance": {
            "clean_track": _table_provenance(clean_track_csv, clean_track),
            "heldout_episodes": _table_provenance(
                heldout_episodes_csv, episode_source_table
            ),
        },
        "configuration": {
            "identity_column": identity_column,
            "max_frame_gap": max_frame_gap,
            "context_margin_frames": int(context_margin_frames),
        },
        "pair_summary": pair_summary,
        "split_audit": split_audit,
        "episodes": episodes.to_dict(orient="records"),
        "model": model,
    }
    return report, split_pairs


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clean_track_csv", type=Path)
    parser.add_argument("--heldout-episodes", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--pairs-output", type=Path, default=None)
    parser.add_argument("--identity-column", default="human_identity")
    parser.add_argument("--max-frame-gap", type=int, default=100)
    parser.add_argument("--context-margin-frames", type=int, default=100)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outputs = [args.output_json]
    if args.pairs_output is not None:
        outputs.append(args.pairs_output)
    for path in outputs:
        if path.exists() and not args.force:
            raise FileExistsError(f"output exists (pass --force): {path}")
    report, pairs = run(
        args.clean_track_csv,
        args.heldout_episodes,
        identity_column=args.identity_column,
        max_frame_gap=args.max_frame_gap,
        context_margin_frames=args.context_margin_frames,
        regularization_c=args.regularization_c,
        max_iter=args.max_iter,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.pairs_output is not None:
        args.pairs_output.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(args.pairs_output, index=False)
    summary = {
        "output_json": str(args.output_json.resolve()),
        "pairs": int(len(pairs)),
        "train_pairs": report["split_audit"]["pair_counts"].get("train", 0),
        "heldout_pairs": report["split_audit"]["pair_counts"].get("heldout", 0),
        "heldout_roc_auc": report["model"]["metrics"]["heldout"]["roc_auc"],
        "heldout_top1_accuracy": report["model"]["metrics"]["heldout"][
            "top1_candidate_accuracy"
        ],
    }
    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
