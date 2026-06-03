from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def assign_tracklet_identities(
    *,
    reference_embeddings_path: str | Path,
    reference_assignments_path: str | Path,
    target_embeddings_path: str | Path,
    target_assignments_path: str | Path,
    output_assignments_path: str | Path,
    distance_matrix_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    max_distance: float = 0.35,
    min_margin: float = 0.05,
    identity_prefix: str = "pseudo_ID_",
    one_to_one: bool = True,
) -> dict:
    reference_embeddings = _load_and_normalize(reference_embeddings_path)
    target_embeddings = _load_and_normalize(target_embeddings_path)
    reference = pd.read_parquet(reference_assignments_path)
    target = pd.read_parquet(target_assignments_path)

    if len(reference) != len(reference_embeddings):
        raise ValueError("Reference assignment rows do not match reference embeddings.")
    if len(target) != len(target_embeddings):
        raise ValueError("Target assignment rows do not match target embeddings.")

    distance = cosine_distance_matrix(target_embeddings, reference_embeddings)
    reference_labels = _reference_identity_labels(reference, identity_prefix=identity_prefix)
    matches = one_to_one_min_distance_assignment(distance) if one_to_one else nearest_reference_assignment(distance)

    rows = []
    for target_idx, target_row in target.reset_index(drop=True).iterrows():
        ref_idx = matches.get(int(target_idx))
        sorted_dist = np.sort(distance[int(target_idx)]) if distance.shape[1] else np.asarray([])
        second_best = float(sorted_dist[1]) if len(sorted_dist) > 1 else np.nan
        if ref_idx is None:
            best_distance = np.nan
            margin = np.nan
            assigned = "unassigned"
            status = "no_reference"
            reference_tracklet = -1
        else:
            best_distance = float(distance[int(target_idx), ref_idx])
            margin = float(second_best - best_distance) if np.isfinite(second_best) else np.inf
            reference_tracklet = int(reference.iloc[ref_idx]["tracklet_id"])
            if best_distance <= max_distance and margin >= min_margin:
                assigned = reference_labels[ref_idx]
                status = "assigned"
            else:
                assigned = "unassigned"
                status = "low_confidence"
        row = dict(target_row)
        row.update({
            "assigned_identity": assigned,
            "reference_tracklet_id": reference_tracklet,
            "match_distance": best_distance,
            "match_margin": margin,
            "assignment_status": status,
        })
        rows.append(row)

    output = pd.DataFrame(rows)
    output_path = Path(output_assignments_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)

    matrix = _distance_dataframe(distance, target=target, reference=reference)
    if distance_matrix_path is not None:
        matrix_path = Path(distance_matrix_path)
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(matrix_path)

    summary = assignment_summary(output, matrix)
    summary["max_distance"] = float(max_distance)
    summary["min_margin"] = float(min_margin)
    summary["one_to_one"] = bool(one_to_one)
    if summary_path is not None:
        path = Path(summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return {
        "assignments": str(output_path),
        "distance_matrix": str(distance_matrix_path) if distance_matrix_path else "",
        "summary": str(summary_path) if summary_path else "",
        "n_assigned": int((output["assignment_status"] == "assigned").sum()),
        "n_tracklets": int(len(output)),
    }


def cosine_distance_matrix(target_embeddings: np.ndarray, reference_embeddings: np.ndarray) -> np.ndarray:
    if target_embeddings.ndim != 2 or reference_embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays.")
    if target_embeddings.shape[1] != reference_embeddings.shape[1]:
        raise ValueError("Embedding dimensions do not match.")
    return 1.0 - target_embeddings @ reference_embeddings.T


def one_to_one_min_distance_assignment(distance: np.ndarray) -> dict[int, int]:
    if distance.size == 0:
        return {}
    n_target, n_reference = distance.shape
    n_match = min(n_target, n_reference)
    if n_target <= n_reference and n_reference <= 8:
        best_score = float("inf")
        best_assignment: dict[int, int] = {}
        for refs in itertools.permutations(range(n_reference), n_match):
            score = float(sum(distance[target_idx, ref_idx] for target_idx, ref_idx in enumerate(refs)))
            if score < best_score:
                best_score = score
                best_assignment = {target_idx: ref_idx for target_idx, ref_idx in enumerate(refs)}
        return best_assignment

    pairs = [
        (float(distance[target_idx, ref_idx]), target_idx, ref_idx)
        for target_idx in range(n_target)
        for ref_idx in range(n_reference)
    ]
    pairs.sort()
    assigned_targets: set[int] = set()
    assigned_refs: set[int] = set()
    out: dict[int, int] = {}
    for _dist, target_idx, ref_idx in pairs:
        if target_idx in assigned_targets or ref_idx in assigned_refs:
            continue
        out[target_idx] = ref_idx
        assigned_targets.add(target_idx)
        assigned_refs.add(ref_idx)
        if len(out) == n_match:
            break
    return out


def nearest_reference_assignment(distance: np.ndarray) -> dict[int, int]:
    if distance.size == 0:
        return {}
    return {
        int(target_idx): int(np.argmin(distance[target_idx]))
        for target_idx in range(distance.shape[0])
    }


def assignment_summary(assignments: pd.DataFrame, distance_matrix: pd.DataFrame) -> dict:
    status_counts = assignments["assignment_status"].value_counts().to_dict() if "assignment_status" in assignments else {}
    assigned = assignments[assignments.get("assignment_status", "") == "assigned"]
    return {
        "n_tracklets": int(len(assignments)),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "assigned_identity_counts": {
            str(k): int(v) for k, v in assignments["assigned_identity"].value_counts().to_dict().items()
        } if "assigned_identity" in assignments else {},
        "mean_assigned_distance": float(assigned["match_distance"].mean()) if not assigned.empty else None,
        "max_assigned_distance": float(assigned["match_distance"].max()) if not assigned.empty else None,
        "distance_matrix_rows": list(distance_matrix.index.astype(str)),
        "distance_matrix_columns": list(distance_matrix.columns.astype(str)),
    }


def _load_and_normalize(path: str | Path) -> np.ndarray:
    embeddings = np.asarray(np.load(path), dtype=np.float32)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norm, 1e-9)


def _reference_identity_labels(reference: pd.DataFrame, *, identity_prefix: str) -> list[str]:
    labels: list[str] = []
    for row in reference.to_dict("records"):
        assigned = str(row.get("assigned_identity", "") or "")
        if assigned and assigned != "unassigned":
            labels.append(assigned)
        else:
            labels.append(f"{identity_prefix}{int(row['tracklet_id'])}")
    return labels


def _distance_dataframe(distance: np.ndarray, *, target: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    rows = [f"target_t{int(tracklet_id)}" for tracklet_id in target["tracklet_id"]]
    cols = [f"reference_t{int(tracklet_id)}" for tracklet_id in reference["tracklet_id"]]
    return pd.DataFrame(distance, index=rows, columns=cols)
