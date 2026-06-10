from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


ID_COLUMNS = ("identity", "assigned_identity", "track_label")


def normalize_reference_tracks(reference: pd.DataFrame) -> pd.DataFrame:
    """Return reference rows with frame, x, y, and identity columns."""
    out = reference.copy()
    if "frame" not in out.columns:
        raise ValueError("Reference tracks missing columns: frame")
    if "x" not in out.columns or "y" not in out.columns:
        if "centroid_x" in out.columns and "centroid_y" in out.columns:
            out["x"] = out["centroid_x"]
            out["y"] = out["centroid_y"]
        else:
            raise ValueError("Reference tracks need x/y or centroid_x/centroid_y columns.")
    identity_col = next((col for col in ID_COLUMNS if col in out.columns), None)
    if identity_col is None:
        if "track_id" not in out.columns:
            raise ValueError("Reference tracks need identity/assigned_identity/track_label or track_id.")
        out["identity"] = out["track_id"].map(lambda value: f"track_{int(value)}")
    else:
        out["identity"] = out[identity_col].astype(str)
    return out[["frame", "x", "y", "identity"]].copy()


def transfer_nearest_identity(
    crop_metadata: pd.DataFrame,
    reference_tracks: pd.DataFrame,
    *,
    max_distance_px: float,
    allowed_identities: set[str] | None = None,
) -> pd.DataFrame:
    """Assign each crop the nearest same-frame reference identity."""
    required = {"frame", "centroid_x", "centroid_y"}
    missing = required - set(crop_metadata.columns)
    if missing:
        raise ValueError(f"Crop metadata missing columns: {', '.join(sorted(missing))}")

    ref = normalize_reference_tracks(reference_tracks)
    ref_by_frame = {int(frame): group for frame, group in ref.groupby("frame", sort=False)}
    rows: list[dict] = []
    for row in crop_metadata.to_dict("records"):
        frame = int(row["frame"])
        candidates = ref_by_frame.get(frame)
        label = ""
        distance = np.nan
        status = "no_reference_frame"
        if candidates is not None and not candidates.empty:
            xy = candidates[["x", "y"]].to_numpy(dtype=float)
            distances = np.hypot(
                xy[:, 0] - float(row["centroid_x"]),
                xy[:, 1] - float(row["centroid_y"]),
            )
            best_idx = int(np.argmin(distances))
            distance = float(distances[best_idx])
            candidate_label = str(candidates.iloc[best_idx]["identity"])
            if distance > float(max_distance_px):
                status = "too_far"
            elif allowed_identities is not None and candidate_label not in allowed_identities:
                status = "identity_not_allowed"
            else:
                label = candidate_label
                status = "labeled"
        out = dict(row)
        out.update(
            {
                "reference_identity": label,
                "reference_distance_px": distance,
                "reference_label_status": status,
            }
        )
        rows.append(out)
    return pd.DataFrame(rows)


def label_counts(labels: Sequence[str]) -> dict[str, int]:
    series = pd.Series(list(labels), dtype=str)
    return {str(k): int(v) for k, v in series.value_counts().sort_index().to_dict().items()}


def evaluate_crop_predictions(
    truth: Sequence[str],
    predicted: Sequence[str],
    *,
    margins: Sequence[float] | None = None,
) -> dict:
    truth_arr = np.asarray(list(truth), dtype=str)
    pred_arr = np.asarray(list(predicted), dtype=str)
    if len(truth_arr) != len(pred_arr):
        raise ValueError("truth and predicted must have the same length.")
    valid = truth_arr != ""
    n_valid = int(valid.sum())
    correct = (truth_arr == pred_arr) & valid
    out = {
        "n_crops": int(len(truth_arr)),
        "n_labeled_crops": n_valid,
        "crop_accuracy": float(correct.sum() / n_valid) if n_valid else None,
        "truth_counts": label_counts(truth_arr[valid]),
        "predicted_counts": label_counts(pred_arr[valid]),
    }
    if margins is not None:
        margin_arr = np.asarray(list(margins), dtype=float)
        if len(margin_arr) != len(truth_arr):
            raise ValueError("margins must have the same length as truth.")
        out["mean_margin"] = float(np.nanmean(margin_arr[valid])) if n_valid else None
        out["median_margin"] = float(np.nanmedian(margin_arr[valid])) if n_valid else None
    return out


def best_label_remap(truth: Sequence[str], predicted: Sequence[str]) -> dict[str, str]:
    truth_arr = np.asarray(list(truth), dtype=str)
    pred_arr = np.asarray(list(predicted), dtype=str)
    valid = (truth_arr != "") & (pred_arr != "")
    truth_labels = sorted(set(truth_arr[valid]))
    pred_labels = sorted(set(pred_arr[valid]))
    if not truth_labels or not pred_labels:
        return {}
    matrix = np.zeros((len(truth_labels), len(pred_labels)), dtype=int)
    truth_index = {label: idx for idx, label in enumerate(truth_labels)}
    pred_index = {label: idx for idx, label in enumerate(pred_labels)}
    for truth_label, pred_label in zip(truth_arr[valid], pred_arr[valid]):
        matrix[truth_index[truth_label], pred_index[pred_label]] += 1
    row_idx, col_idx = linear_sum_assignment(-matrix)
    return {pred_labels[col]: truth_labels[row] for row, col in zip(row_idx, col_idx)}


def apply_label_remap(predicted: Sequence[str], remap: dict[str, str]) -> np.ndarray:
    return np.asarray([remap.get(str(label), str(label)) for label in predicted], dtype=str)


def majority_vote_by_tracklet(
    crop_metadata: pd.DataFrame,
    truth: Sequence[str],
    predicted: Sequence[str],
) -> pd.DataFrame:
    if "tracklet_id" not in crop_metadata.columns or "frame" not in crop_metadata.columns:
        raise ValueError("crop_metadata needs tracklet_id and frame columns.")
    truth_arr = np.asarray(list(truth), dtype=str)
    pred_arr = np.asarray(list(predicted), dtype=str)
    if len(crop_metadata) != len(truth_arr) or len(crop_metadata) != len(pred_arr):
        raise ValueError("metadata, truth, and predicted lengths must match.")

    table = crop_metadata[["frame", "tracklet_id"]].copy()
    table["truth"] = truth_arr
    table["predicted"] = pred_arr
    table = table[table["truth"] != ""].copy()
    rows: list[dict] = []
    for tracklet_id, group in table.groupby("tracklet_id", sort=True):
        truth_major = str(group["truth"].mode().iloc[0])
        pred_major = str(group["predicted"].mode().iloc[0])
        rows.append(
            {
                "tracklet_id": int(tracklet_id),
                "n_crops": int(len(group)),
                "frame_min": int(group["frame"].min()),
                "frame_max": int(group["frame"].max()),
                "truth": truth_major,
                "predicted": pred_major,
                "correct": bool(truth_major == pred_major),
                "truth_purity": float((group["truth"] == truth_major).mean()),
                "prediction_purity": float((group["predicted"] == pred_major).mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_tracklet_predictions(tracklet_votes: pd.DataFrame) -> dict:
    if tracklet_votes.empty:
        return {
            "n_tracklets": 0,
            "tracklet_majority_accuracy": None,
            "mean_prediction_purity": None,
        }
    return {
        "n_tracklets": int(len(tracklet_votes)),
        "tracklet_majority_accuracy": float(tracklet_votes["correct"].mean()),
        "mean_prediction_purity": float(tracklet_votes["prediction_purity"].mean()),
        "mean_truth_purity": float(tracklet_votes["truth_purity"].mean()),
    }


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    arr = np.asarray(embeddings, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norm, 1e-9)


def identity_centroids(embeddings: np.ndarray, labels: Sequence[str]) -> tuple[list[str], np.ndarray]:
    labels_arr = np.asarray(list(labels), dtype=str)
    valid = labels_arr != ""
    names = sorted(set(labels_arr[valid]))
    if not names:
        return [], np.zeros((0, embeddings.shape[1]), dtype=np.float32)
    emb = normalize_embeddings(embeddings)
    centroids = np.stack([emb[labels_arr == name].mean(axis=0) for name in names], axis=0)
    return names, normalize_embeddings(centroids)


def predict_by_centroid(
    embeddings: np.ndarray,
    centroid_labels: Sequence[str],
    centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(centroid_labels) == 0 or len(centroids) == 0:
        return (
            np.asarray([""] * len(embeddings), dtype=str),
            np.full((len(embeddings),), np.nan, dtype=float),
            np.full((len(embeddings),), np.nan, dtype=float),
        )
    emb = normalize_embeddings(embeddings)
    sim = emb @ normalize_embeddings(centroids).T
    order = np.argsort(-sim, axis=1)
    best = order[:, 0]
    second = order[:, 1] if sim.shape[1] > 1 else order[:, 0]
    labels = np.asarray(list(centroid_labels), dtype=str)[best]
    margin = sim[np.arange(len(sim)), best] - sim[np.arange(len(sim)), second]
    distance = 1.0 - sim[np.arange(len(sim)), best]
    return labels, distance.astype(float), margin.astype(float)
