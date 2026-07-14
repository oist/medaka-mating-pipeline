from __future__ import annotations

"""Appearance-embedding utilities for geometry-preserving identity stitching.

The metric-learning dataset may use reconstruction-local source indices to mine
same/different constraints.  Those indices intentionally do not enter this
module.  Biological labels are accepted only for explicitly trusted anchor
embeddings, while ordinary fragments contribute label-free pairwise distances.
"""

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from scripts.tracking.global_fragment_stitching import EdgeFeatures


@dataclass(frozen=True)
class PooledEmbeddings:
    keys: tuple[str, ...]
    embeddings: np.ndarray
    sample_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional matrix")
        if len(self.keys) != len(matrix) or len(self.sample_counts) != len(matrix):
            raise ValueError("keys, embeddings, and sample_counts must have equal length")
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("pooled embedding keys must be unique")
        if any(int(count) <= 0 for count in self.sample_counts):
            raise ValueError("sample_counts must be positive")
        object.__setattr__(self, "embeddings", _normalize_rows(matrix))

    def as_mapping(self) -> dict[str, np.ndarray]:
        return {
            key: self.embeddings[index]
            for index, key in enumerate(self.keys)
        }

    def count_mapping(self) -> dict[str, int]:
        return dict(zip(self.keys, self.sample_counts, strict=True))


@dataclass(frozen=True)
class PrototypeBank:
    identities: tuple[str, ...]
    embeddings: np.ndarray
    sample_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        pooled = PooledEmbeddings(
            self.identities,
            self.embeddings,
            self.sample_counts,
        )
        if any(not identity for identity in self.identities):
            raise ValueError("prototype identities must be non-empty")
        object.__setattr__(self, "embeddings", pooled.embeddings)


@dataclass(frozen=True)
class PrototypeAssignment:
    fragment_id: str
    candidate_identity: str | None
    cosine_distance: float | None
    margin: float | None
    status: str


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("embedding input must be a two-dimensional matrix")
    if values.shape[0] == 0:
        return values.copy()
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.all(np.isfinite(values)) or np.any(norms <= 1e-12):
        raise ValueError("embeddings must contain finite, non-zero rows")
    return values / norms


def cosine_distance_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return cosine distances in [0, 2] after strict row normalization."""
    left_norm = _normalize_rows(left)
    right_norm = _normalize_rows(right)
    if left_norm.shape[1] != right_norm.shape[1]:
        raise ValueError("embedding dimensions do not match")
    return np.clip(1.0 - left_norm @ right_norm.T, 0.0, 2.0)


def pool_embeddings(
    embeddings: np.ndarray,
    keys: Sequence[str],
    *,
    views: Sequence[str] | None = None,
    min_samples: int = 1,
    balance_views: bool = True,
) -> PooledEmbeddings:
    """Pool crop embeddings by fragment or trusted identity.

    When view labels are supplied, each camera contributes one unit-normalized
    centroid before the final mean.  This prevents a camera with many sampled
    frames from dominating the pooled representation.
    """
    values = _normalize_rows(embeddings)
    key_array = np.asarray([str(key) for key in keys], dtype=object)
    if len(key_array) != len(values):
        raise ValueError("keys and embeddings must have equal length")
    if int(min_samples) <= 0:
        raise ValueError("min_samples must be positive")
    view_array: np.ndarray | None = None
    if views is not None:
        view_array = np.asarray([str(view) for view in views], dtype=object)
        if len(view_array) != len(values):
            raise ValueError("views and embeddings must have equal length")

    output_keys: list[str] = []
    output_vectors: list[np.ndarray] = []
    counts: list[int] = []
    for key in sorted(set(key_array)):
        if not key:
            continue
        selection = key_array == key
        count = int(selection.sum())
        if count < int(min_samples):
            continue
        selected = values[selection]
        if view_array is not None and balance_views:
            selected_views = view_array[selection]
            view_centroids = [
                _normalize_rows(selected[selected_views == view].mean(axis=0, keepdims=True))[0]
                for view in sorted(set(selected_views))
            ]
            vector = np.mean(view_centroids, axis=0, keepdims=True)
        else:
            vector = selected.mean(axis=0, keepdims=True)
        output_keys.append(key)
        output_vectors.append(_normalize_rows(vector)[0])
        counts.append(count)

    dimension = int(values.shape[1])
    matrix = (
        np.stack(output_vectors).astype(np.float32, copy=False)
        if output_vectors
        else np.zeros((0, dimension), dtype=np.float32)
    )
    return PooledEmbeddings(tuple(output_keys), matrix, tuple(counts))


def build_trusted_prototypes(
    embeddings: np.ndarray,
    trusted_identities: Sequence[str],
    *,
    views: Sequence[str] | None = None,
    min_samples_per_identity: int = 3,
    balance_views: bool = True,
) -> PrototypeBank:
    """Build prototypes only from externally validated biological anchors."""
    pooled = pool_embeddings(
        embeddings,
        trusted_identities,
        views=views,
        min_samples=min_samples_per_identity,
        balance_views=balance_views,
    )
    return PrototypeBank(pooled.keys, pooled.embeddings, pooled.sample_counts)


def assign_simultaneous_fragments(
    fragments: PooledEmbeddings,
    prototypes: PrototypeBank,
    *,
    max_distance: float = 0.35,
    min_margin: float = 0.05,
) -> tuple[PrototypeAssignment, ...]:
    """Match one synchronized fish batch to prototypes with abstention.

    Hungarian assignment enforces that two simultaneous fragments cannot claim
    the same fish.  Dummy columns allow any row to remain unknown.  A real match
    must also pass a row-local distance and nearest-vs-second-nearest margin,
    so a globally forced second choice is never promoted as an anchor.
    """
    if float(max_distance) < 0.0 or float(min_margin) < 0.0:
        raise ValueError("max_distance and min_margin must be non-negative")
    if len(fragments.keys) == 0:
        return ()
    if len(prototypes.identities) == 0:
        return tuple(
            PrototypeAssignment(key, None, None, None, "no_prototypes")
            for key in fragments.keys
        )

    distance = cosine_distance_matrix(fragments.embeddings, prototypes.embeddings)
    n_fragments, n_prototypes = distance.shape
    # One dummy identity per row makes abstention independent across fragments.
    dummy_cost = np.full((n_fragments, n_fragments), float(max_distance), dtype=float)
    assignment_cost = np.concatenate((distance, dummy_cost), axis=1)
    rows, columns = linear_sum_assignment(assignment_cost)
    selected = dict(zip(rows.tolist(), columns.tolist(), strict=True))

    output: list[PrototypeAssignment] = []
    for row, fragment_id in enumerate(fragments.keys):
        column = selected[row]
        if column >= n_prototypes:
            best = float(np.min(distance[row]))
            output.append(
                PrototypeAssignment(fragment_id, None, best, None, "distance_reject")
            )
            continue
        chosen_distance = float(distance[row, column])
        alternatives = np.delete(distance[row], column)
        margin = (
            float(np.min(alternatives) - chosen_distance)
            if len(alternatives)
            else float("inf")
        )
        if chosen_distance > float(max_distance):
            status = "distance_reject"
            identity = None
        elif margin < float(min_margin):
            status = "margin_reject"
            identity = None
        else:
            status = "assigned"
            identity = prototypes.identities[column]
        output.append(
            PrototypeAssignment(
                fragment_id,
                identity,
                chosen_distance,
                margin,
                status,
            )
        )
    return tuple(output)


def attach_fragment_appearance(
    edges: Sequence[EdgeFeatures],
    fragment_embeddings: PooledEmbeddings,
    *,
    overwrite: bool = False,
    calibrator_coefficient: float | None = None,
    calibrator_intercept: float | None = None,
    same_probability_min: float | None = None,
    different_probability_max: float | None = None,
) -> tuple[EdgeFeatures, ...]:
    """Attach label-free fragment distances to global-stitching edges.

    Missing embeddings and scores inside the calibration-frozen uncertainty
    band remain neutral. Existing appearance evidence is protected unless the
    caller opts into replacement explicitly.
    """
    vector_by_id = fragment_embeddings.as_mapping()
    count_by_id = fragment_embeddings.count_mapping()
    output: list[EdgeFeatures] = []
    for edge in edges:
        if edge.appearance_cosine_distance is not None and not overwrite:
            output.append(edge)
            continue
        source = vector_by_id.get(str(edge.source_id))
        target = vector_by_id.get(str(edge.target_id))
        if source is None or target is None:
            output.append(edge)
            continue
        distance = float(cosine_distance_matrix(source[None, :], target[None, :])[0, 0])
        similarity = 1.0 - distance
        state = "unknown"
        probability = None
        calibration_values = (
            calibrator_coefficient,
            calibrator_intercept,
            same_probability_min,
            different_probability_max,
        )
        if all(value is not None for value in calibration_values):
            logit = float(calibrator_coefficient) * similarity + float(calibrator_intercept)
            probability = float(1.0 / (1.0 + np.exp(-np.clip(logit, -50.0, 50.0))))
            if probability >= float(same_probability_min):
                state = "support"
            elif probability <= float(different_probability_max):
                state = "contradiction"
        metadata: Mapping[str, object] = {
            **dict(edge.metadata),
            "appearance_source": "pooled_metric_embedding",
            "appearance_source_samples": count_by_id[str(edge.source_id)],
            "appearance_target_samples": count_by_id[str(edge.target_id)],
            "appearance_same_probability": probability,
            "appearance_state": state,
        }
        output.append(
            replace(
                edge,
                appearance_cosine_distance=distance,
                appearance_state=state,
                metadata=metadata,
            )
        )
    return tuple(output)
