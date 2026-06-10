from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd

from sam2_id_experiment.src.evaluation.contrastive_identity_probe import normalize_embeddings


def sample_supervised_identity_pairs(
    metadata: pd.DataFrame,
    *,
    positive_pairs_per_identity: int = 1500,
    cross_view_positive_pairs_per_identity: int = 1500,
    negative_pairs_per_identity_pair: int = 1500,
    random_seed: int = 13,
) -> pd.DataFrame:
    """Sample contrastive pairs from explicit biological identity labels.

    Positive pairs come from the same reference_identity, with an extra quota for
    cross-view pairs so side1/side2 crops of the same fish are pulled together.
    Negative pairs are sampled across different reference_identity labels.
    """

    required = {"crop_id", "reference_identity"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {', '.join(sorted(missing))}")

    labeled = metadata[metadata["reference_identity"].fillna("").astype(str).ne("")].copy()
    if labeled.empty:
        return pd.DataFrame(columns=["crop_a_id", "crop_b_id", "label", "reason"])

    rng = np.random.default_rng(int(random_seed))
    rows: list[dict] = []
    seen: set[tuple[int, int, int]] = set()

    for identity, group in labeled.groupby("reference_identity", sort=True):
        if "source_view" in group.columns and group["source_view"].nunique() > 1:
            view_groups = [
                grp["crop_id"].astype(int).tolist()
                for _view, grp in group.groupby("source_view", sort=True)
            ]
            cross_pairs = (
                (a, b)
                for left, right in itertools.combinations(view_groups, 2)
                for a in left
                for b in right
            )
            for a, b in _sample_from_pairs(cross_pairs, cross_view_positive_pairs_per_identity, rng):
                _append_pair(rows, seen, a, b, 1, f"same_identity_cross_view:{identity}")

        crop_ids = group["crop_id"].astype(int).tolist()
        for a, b in _sample_from_pairs(itertools.combinations(crop_ids, 2), positive_pairs_per_identity, rng):
            _append_pair(rows, seen, a, b, 1, f"same_identity:{identity}")

    by_identity = {
        str(identity): group["crop_id"].astype(int).tolist()
        for identity, group in labeled.groupby("reference_identity", sort=True)
    }
    for left_identity, right_identity in itertools.combinations(sorted(by_identity), 2):
        negative_pairs = (
            (a, b)
            for a in by_identity[left_identity]
            for b in by_identity[right_identity]
        )
        reason = f"different_identity:{left_identity}:{right_identity}"
        for a, b in _sample_from_pairs(negative_pairs, negative_pairs_per_identity_pair, rng):
            _append_pair(rows, seen, a, b, 0, reason)

    pairs = pd.DataFrame(rows, columns=["crop_a_id", "crop_b_id", "label", "reason"])
    if not pairs.empty:
        pairs = pairs.sample(frac=1.0, random_state=int(random_seed)).reset_index(drop=True)
    return pairs


def tracklet_majority_table(
    metadata: pd.DataFrame,
    truth: Sequence[str],
    predicted: Sequence[str],
    *,
    group_cols: Sequence[str] = ("source_view", "tracklet_id"),
) -> pd.DataFrame:
    truth_arr = np.asarray(list(truth), dtype=str)
    pred_arr = np.asarray(list(predicted), dtype=str)
    if len(metadata) != len(truth_arr) or len(metadata) != len(pred_arr):
        raise ValueError("metadata, truth, and predicted lengths must match.")
    missing = set(group_cols) - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {', '.join(sorted(missing))}")

    table = metadata[list(group_cols) + ["frame"]].copy()
    table["truth"] = truth_arr
    table["predicted"] = pred_arr
    table = table[table["truth"] != ""].copy()

    rows: list[dict] = []
    for group_key, group in table.groupby(list(group_cols), sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        truth_major = str(group["truth"].mode().iloc[0])
        pred_major = str(group["predicted"].mode().iloc[0])
        row = {col: value for col, value in zip(group_cols, group_key)}
        row.update(
            {
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
        rows.append(row)
    return pd.DataFrame(rows)


def cross_view_tracklet_correspondence(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    predicted: Sequence[str] | None = None,
) -> pd.DataFrame:
    required = {"source_view", "tracklet_id", "reference_identity"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {', '.join(sorted(missing))}")
    if len(metadata) != len(embeddings):
        raise ValueError("metadata and embeddings lengths must match.")

    meta = metadata.reset_index(drop=True).copy()
    meta["predicted_identity"] = list(predicted) if predicted is not None else ""
    emb = normalize_embeddings(embeddings)

    summaries: list[dict] = []
    for (view, tracklet_id), group in meta.groupby(["source_view", "tracklet_id"], sort=True):
        idx = group.index.to_numpy(dtype=int)
        centroid = normalize_embeddings(emb[idx].mean(axis=0, keepdims=True))[0]
        summaries.append(
            {
                "source_view": str(view),
                "tracklet_id": int(tracklet_id),
                "n_crops": int(len(group)),
                "frame_min": int(group["frame"].min()) if "frame" in group else -1,
                "frame_max": int(group["frame"].max()) if "frame" in group else -1,
                "truth": str(group["reference_identity"].mode().iloc[0]),
                "truth_purity": float((group["reference_identity"] == group["reference_identity"].mode().iloc[0]).mean()),
                "predicted": str(group["predicted_identity"].mode().iloc[0]) if predicted is not None else "",
                "embedding": centroid,
            }
        )

    if not summaries:
        return pd.DataFrame()

    views = sorted({row["source_view"] for row in summaries})
    if len(views) < 2:
        return pd.DataFrame()
    left_view, right_view = views[:2]
    left = [row for row in summaries if row["source_view"] == left_view]
    right = [row for row in summaries if row["source_view"] == right_view]

    rows: list[dict] = []
    for left_row in left:
        distances = []
        for right_row in right:
            distance = float(1.0 - np.dot(left_row["embedding"], right_row["embedding"]))
            distances.append((distance, right_row))
        distances.sort(key=lambda item: item[0])
        for rank, (distance, right_row) in enumerate(distances, start=1):
            rows.append(
                {
                    "left_view": left_view,
                    "left_tracklet_id": int(left_row["tracklet_id"]),
                    "left_n_crops": int(left_row["n_crops"]),
                    "left_frame_min": int(left_row["frame_min"]),
                    "left_frame_max": int(left_row["frame_max"]),
                    "left_truth": str(left_row["truth"]),
                    "left_predicted": str(left_row["predicted"]),
                    "right_view": right_view,
                    "right_tracklet_id": int(right_row["tracklet_id"]),
                    "right_n_crops": int(right_row["n_crops"]),
                    "right_frame_min": int(right_row["frame_min"]),
                    "right_frame_max": int(right_row["frame_max"]),
                    "right_truth": str(right_row["truth"]),
                    "right_predicted": str(right_row["predicted"]),
                    "embedding_distance": distance,
                    "rank_for_left_tracklet": int(rank),
                    "same_truth_identity": bool(left_row["truth"] == right_row["truth"]),
                    "same_predicted_identity": bool(
                        predicted is not None and str(left_row["predicted"]) == str(right_row["predicted"])
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["left_tracklet_id", "rank_for_left_tracklet"]).reset_index(drop=True)


def _sample_from_pairs(
    pairs: itertools.combinations | object,
    max_pairs: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    items = list(pairs)
    if len(items) <= int(max_pairs):
        return [(int(a), int(b)) for a, b in items]
    idx = rng.choice(len(items), size=int(max_pairs), replace=False)
    return [(int(items[int(i)][0]), int(items[int(i)][1])) for i in idx]


def _append_pair(
    rows: list[dict],
    seen: set[tuple[int, int, int]],
    a: int,
    b: int,
    label: int,
    reason: str,
) -> None:
    left, right = sorted((int(a), int(b)))
    key = (left, right, int(label))
    if key in seen or left == right:
        return
    seen.add(key)
    rows.append({"crop_a_id": left, "crop_b_id": right, "label": int(label), "reason": reason})
