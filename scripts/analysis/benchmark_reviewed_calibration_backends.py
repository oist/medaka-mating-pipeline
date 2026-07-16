"""Independent reviewed-data benchmark for 3D calibration backends.

This benchmark uses the frozen, human-reviewed *mouth* coordinates. It keeps
four questions separate: cross-view assignment, XYZ with assignments frozen,
cross-frame geometry identity linking, and their end-to-end combination. The
reviewed XYZ values were produced by the production polynomial calibration and
are therefore a consistency reference, not absolute fish-position ground truth.

The script validates the frozen manifest hashes and coordinate provenance before
loading a backend. It performs no threshold, affine, or shift fitting. Backend
scores are always reported with the backend-declared unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from scripts.calibration.refractive_geometry import FlatPortCamera
from scripts.reconstruction_3d.calibration import Calibration
from scripts.reconstruction_3d.cross_frame_tracking import apply_cross_frame_tracking
from scripts.reconstruction_3d.triangulate_extra_candidates import (
    compute_best_assignments_per_frame,
)


VIEWS = ("top_cam", "side1_cam", "side2_cam")
VIEW_INDEX = {name: index for index, name in enumerate(VIEWS)}
MOUTH_COLUMNS = {
    "top_cam": ("top_mouth_x", "top_mouth_y"),
    "side1_cam": ("side1_mouth_x", "side1_mouth_y"),
    "side2_cam": ("side2_mouth_x", "side2_mouth_y"),
}
PAIR_LABELS = ((0, 1), (0, 2), (1, 2))
IDENTITIES = ("H1", "H2", "H3")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_artifact(manifest: dict[str, Any], filename: str) -> dict[str, Any]:
    for artifact in manifest.get("artifacts", []):
        if artifact.get("path") == filename:
            return artifact
    raise ValueError(f"review manifest does not list required artifact {filename!r}")


def verify_manifest_artifact(
    review_dir: Path, manifest: dict[str, Any], filename: str
) -> dict[str, Any]:
    artifact = manifest_artifact(manifest, filename)
    path = review_dir / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = sha256_file(path)
    if actual_hash != artifact.get("sha256"):
        raise ValueError(
            f"SHA256 mismatch for {filename}: {actual_hash} != {artifact.get('sha256')}"
        )
    if path.stat().st_size != int(artifact.get("byte_size", -1)):
        raise ValueError(f"byte-size mismatch for {filename}")
    return {
        "path": str(path),
        "sha256": actual_hash,
        "byte_size": int(path.stat().st_size),
        "row_count": int(artifact.get("row_count", -1)),
    }


def validate_reviewed_data(
    review_dir: Path,
    *,
    per_view_name: str = "corrected_per_view_sleap_mouth_tracks_f0_15000.csv",
    clean_name: str = "clean_reviewed_multiview_mouth_3d_f0_15000.csv",
    manifest_name: str = "reviewed_consensus_manifest_v1.json",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest_path = review_dir / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "reviewed_consensus_dataset_manifest_v1":
        raise ValueError("unsupported reviewed consensus manifest schema")
    if manifest.get("coordinate_definition", {}).get("point") != "mouth":
        raise ValueError("review manifest does not declare mouth coordinates")

    fingerprints = {
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "schema_version": manifest["schema_version"],
        },
        "per_view": verify_manifest_artifact(review_dir, manifest, per_view_name),
        "clean_3d": verify_manifest_artifact(review_dir, manifest, clean_name),
    }
    per_view = pd.read_csv(review_dir / per_view_name)
    clean = pd.read_csv(review_dir / clean_name)
    for key, table in (("per_view", per_view), ("clean_3d", clean)):
        expected = fingerprints[key]["row_count"]
        if expected >= 0 and len(table) != expected:
            raise ValueError(f"row-count mismatch for {key}: {len(table)} != {expected}")

    required_per_view = {
        "frame", "view", "geometry_slot", "human_identity", "mouth_x", "mouth_y",
        "centroid_x", "centroid_y", "geometry_match_distance_px",
    }
    missing = required_per_view - set(per_view.columns)
    if missing:
        raise ValueError("per-view review data is missing: " + ", ".join(sorted(missing)))
    if per_view.duplicated(["frame", "view", "geometry_slot"]).any():
        raise ValueError("duplicate frame/view/geometry_slot reviewed observations")
    if per_view.duplicated(["frame", "view", "human_identity"]).any():
        raise ValueError("duplicate frame/view/human_identity reviewed observations")
    if set(per_view["view"].astype(str)) - set(VIEWS):
        raise ValueError("reviewed per-view table contains unknown cameras")
    finite_mouth = np.all(
        np.isfinite(per_view[["mouth_x", "mouth_y"]].to_numpy(float)), axis=1
    )
    nonfinite_by_view = (
        per_view.loc[~finite_mouth, "view"].value_counts().sort_index().astype(int).to_dict()
    )
    fingerprints["per_view_mouth_filter"] = {
        "input_rows": int(len(per_view)),
        "finite_mouth_rows": int(finite_mouth.sum()),
        "dropped_nonfinite_mouth_rows": int((~finite_mouth).sum()),
        "dropped_by_view": nonfinite_by_view,
        "policy": "Nonfinite mouth rows are missing observations, never replaced by centroid coordinates.",
    }
    per_view = per_view.loc[finite_mouth].copy()

    required_clean = {
        "frame", "human_identity", "X", "Y", "Z", "coordinate_source",
        "all_views_agree", "label_status", "filter_all_view_mapping_agreement",
        "filter_max_geometry_match_px", "filter_max_reprojection_px",
        "max_reprojection_error_px", "top_cam_match_px", "side1_cam_match_px",
        "side2_cam_match_px", *{
            column for columns in MOUTH_COLUMNS.values() for column in columns
        },
    }
    missing = required_clean - set(clean.columns)
    if missing:
        raise ValueError("clean reviewed table is missing: " + ", ".join(sorted(missing)))
    if clean.duplicated(["frame", "human_identity"]).any():
        raise ValueError("duplicate clean frame/human_identity rows")
    if not clean["all_views_agree"].astype(bool).all():
        raise ValueError("clean reference includes rows without all-view mapping agreement")
    if set(clean["coordinate_source"].astype(str)) != {"three_view_reviewed_sleap_mouth"}:
        raise ValueError("clean reference coordinate source is not reviewed SLEAP mouth")
    if set(clean["label_status"].astype(str)) != {"clean_reviewed_three_view"}:
        raise ValueError("clean reference contains a non-clean label status")
    if not clean["filter_all_view_mapping_agreement"].astype(bool).all():
        raise ValueError("clean reference includes rows that failed the mapping-agreement filter")
    if (clean["max_reprojection_error_px"] > clean["filter_max_reprojection_px"] + 1e-9).any():
        raise ValueError("clean reference includes rows above its reprojection gate")
    for column in ("top_cam_match_px", "side1_cam_match_px", "side2_cam_match_px"):
        if (clean[column] > clean["filter_max_geometry_match_px"] + 1e-9).any():
            raise ValueError(f"clean reference includes rows above its geometry-match gate: {column}")
    numeric = ["X", "Y", "Z", *{
        column for columns in MOUTH_COLUMNS.values() for column in columns
    }]
    if not np.isfinite(clean[numeric].to_numpy(float)).all():
        raise ValueError("clean reviewed coordinates must be finite")

    # Prove that benchmark coordinates are the frozen mouth points. The
    # geometry-match gate was computed from centroids and remains QC provenance.
    long_rows = []
    for view, (x_col, y_col) in MOUTH_COLUMNS.items():
        part = clean[["frame", "human_identity", x_col, y_col]].copy()
        part.columns = ["frame", "human_identity", "mouth_x_ref", "mouth_y_ref"]
        part["view"] = view
        long_rows.append(part)
    clean_long = pd.concat(long_rows, ignore_index=True)
    joined = clean_long.merge(
        per_view[["frame", "view", "human_identity", "mouth_x", "mouth_y"]],
        on=["frame", "view", "human_identity"], how="left", validate="one_to_one",
    )
    if joined[["mouth_x", "mouth_y"]].isna().any().any():
        raise ValueError("clean mouth coordinates cannot be matched to reviewed per-view rows")
    parity = np.linalg.norm(
        joined[["mouth_x_ref", "mouth_y_ref"]].to_numpy(float)
        - joined[["mouth_x", "mouth_y"]].to_numpy(float), axis=1,
    )
    if float(np.max(parity, initial=0.0)) > 1e-9:
        raise ValueError("clean/per-view mouth coordinate provenance mismatch")
    fingerprints["coordinate_provenance"] = {
        "point": "mouth",
        "rows_compared": int(len(joined)),
        "maximum_pixel_difference": float(np.max(parity, initial=0.0)),
        "geometry_match_gate_point": "visible-node centroid (QC provenance only)",
    }
    return per_view, clean, fingerprints


def build_episode_split(
    start_frame: int,
    end_frame: int,
    *,
    episode_frames: int = 1500,
    cycle: int = 5,
    holdout_index: int = 4,
) -> pd.DataFrame:
    if episode_frames <= 0 or cycle <= 1 or not 0 <= holdout_index < cycle:
        raise ValueError("invalid episode split parameters")
    rows = []
    episode = 0
    for start in range(start_frame, end_frame + 1, episode_frames):
        end = min(end_frame, start + episode_frames - 1)
        split = "heldout" if episode % cycle == holdout_index else "tuning"
        rows.append({
            "episode_id": f"episode_{episode:03d}", "start_frame": start,
            "end_frame": end, "n_frames": end - start + 1, "split": split,
        })
        episode += 1
    return pd.DataFrame(rows)


def split_for_frame(frames: np.ndarray, episodes: pd.DataFrame) -> np.ndarray:
    values = np.full(len(frames), "outside", dtype=object)
    for row in episodes.itertuples(index=False):
        mask = (frames >= row.start_frame) & (frames <= row.end_frame)
        values[mask] = row.split
    return values


class DLTBackend:
    """Small evaluation-only direct-ray adapter for checked-in DLT cameras."""

    backend_name = "projective_dlt"
    score_unit = "mm^2"

    def __init__(self, model_dir: Path) -> None:
        paths = [model_dir / f"cam{index}_pinhole_dlt.json" for index in range(3)]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError("DLT camera JSONs are incomplete")
        self.cameras = tuple(FlatPortCamera.from_json(path) for path in paths)
        self.provenance = {
            "backend": self.backend_name,
            "model_files": [
                {"path": str(path), "sha256": sha256_file(path)} for path in paths
            ],
        }
        self.capabilities = {
            "triangulation": True, "projection": True,
            "two_view_triangulation": True,
        }

    def triangulate(self, observations: np.ndarray) -> np.ndarray:
        values = np.asarray(observations, dtype=float)
        leading = values.shape[:-2]
        flat = values.reshape(-1, 3, 2)
        output = np.full((len(flat), 3), np.nan)
        finite = np.all(np.isfinite(flat), axis=2)
        for mask_value in range(8):
            views = tuple(index for index in range(3) if mask_value & (1 << index))
            if len(views) < 2:
                continue
            rows = np.where(np.all(finite == np.array(
                [index in views for index in range(3)]
            ), axis=1))[0]
            if not len(rows):
                continue
            origins, directions = [], []
            for view in views:
                origin, direction = self.cameras[view].backproject(flat[rows, view])
                origins.append(origin)
                directions.append(direction)
            origins_array = np.stack(origins, axis=1)
            directions_array = np.stack(directions, axis=1)
            identity = np.eye(3)
            projectors = (
                identity[None, None]
                - directions_array[..., :, None] * directions_array[..., None, :]
            )
            system = np.sum(projectors, axis=1)
            rhs = np.sum(projectors @ origins_array[..., None], axis=1).squeeze(-1)
            try:
                output[rows] = np.linalg.solve(system, rhs[..., None]).squeeze(-1)
            except np.linalg.LinAlgError:
                output[rows] = np.einsum("nij,nj->ni", np.linalg.pinv(system), rhs)
        return output.reshape(*leading, 3)

    def score_hypotheses(
        self,
        observations: np.ndarray,
        xyz: np.ndarray | None = None,
        *,
        profile: str = "production_fast",
    ) -> np.ndarray:
        del profile
        values = np.asarray(observations, dtype=float)
        leading = values.shape[:-2]
        flat = values.reshape(-1, 3, 2)
        points = self.triangulate(values).reshape(-1, 3) if xyz is None else np.asarray(
            xyz, dtype=float
        ).reshape(-1, 3)
        squared = np.full((len(flat), 3), np.nan)
        for view, camera in enumerate(self.cameras):
            valid = np.all(np.isfinite(flat[:, view]), axis=1) & np.all(
                np.isfinite(points), axis=1
            )
            if valid.any():
                residual = camera.point_to_ray_residuals(flat[valid, view], points[valid])
                squared[valid, view] = np.sum(residual**2, axis=1)
        count = np.sum(np.isfinite(squared), axis=1)
        score = np.full(len(flat), np.inf)
        valid = count >= 2
        score[valid] = np.nansum(squared[valid], axis=1) / count[valid]
        return score.reshape(leading)

    def project_world(self, xyz: np.ndarray) -> np.ndarray:
        values = np.asarray(xyz, dtype=float)
        leading = values.shape[:-1]
        flat = values.reshape(-1, 3)
        return np.stack([camera.project(flat) for camera in self.cameras], axis=1).reshape(
            *leading, 3, 2
        )


@dataclass
class BackendRun:
    label: str
    backend: Any


def _score_batches(backend: Any, batches: list[np.ndarray]) -> list[np.ndarray]:
    if not batches:
        return []
    lengths = [len(batch) for batch in batches]
    scores = np.asarray(backend.score_hypotheses(np.concatenate(batches)), dtype=float)
    cuts = np.cumsum(lengths)[:-1]
    return list(np.split(scores, cuts))


def pair_association_metrics(
    backend: Any, per_view: pd.DataFrame, episodes: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for left, right in PAIR_LABELS:
        left_name, right_name = VIEWS[left], VIEWS[right]
        left_groups = {frame: table.sort_values("geometry_slot", kind="stable") for frame, table
                       in per_view.loc[per_view["view"].eq(left_name)].groupby("frame")}
        right_groups = {frame: table.sort_values("geometry_slot", kind="stable") for frame, table
                        in per_view.loc[per_view["view"].eq(right_name)].groupby("frame")}
        tasks: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []
        batches = []
        for frame in sorted(set(left_groups) & set(right_groups)):
            a, b = left_groups[frame], right_groups[frame]
            obs = np.full((len(a) * len(b), 3, 2), np.nan)
            index = 0
            for a_row in a.itertuples(index=False):
                for b_row in b.itertuples(index=False):
                    obs[index, left] = (a_row.mouth_x, a_row.mouth_y)
                    obs[index, right] = (b_row.mouth_x, b_row.mouth_y)
                    index += 1
            tasks.append((int(frame), a, b))
            batches.append(obs)
        score_batches = _score_batches(backend, batches)
        details = []
        for (frame, a, b), scores in zip(tasks, score_batches, strict=True):
            cost = scores.reshape(len(a), len(b))
            finite_cost = np.where(np.isfinite(cost), cost, 1e15)
            raw_ai, raw_bi = linear_sum_assignment(finite_cost)
            finite_pairs = [
                (int(i), int(j)) for i, j in zip(raw_ai, raw_bi, strict=True)
                if np.isfinite(cost[i, j])
            ]
            ai = np.array([pair[0] for pair in finite_pairs], dtype=int)
            bi = np.array([pair[1] for pair in finite_pairs], dtype=int)
            matched_a, matched_b = set(ai), set(bi)
            correct = sum(
                str(a.iloc[i]["human_identity"]) == str(b.iloc[j]["human_identity"])
                for i, j in zip(ai, bi, strict=True)
            )
            unmatched_a = [i for i in range(len(a)) if i not in matched_a]
            unmatched_b = [j for j in range(len(b)) if j not in matched_b]
            ids_a, ids_b = set(a["human_identity"].astype(str)), set(b["human_identity"].astype(str))
            unmatched_correct = sum(str(a.iloc[i]["human_identity"]) not in ids_b for i in unmatched_a)
            unmatched_correct += sum(str(b.iloc[j]["human_identity"]) not in ids_a for j in unmatched_b)
            details.append({
                "frame": frame, "matched": len(ai), "correct": correct,
                "unmatched": len(unmatched_a) + len(unmatched_b),
                "unmatched_correct": unmatched_correct,
            })
        detail = pd.DataFrame(details)
        if not detail.empty:
            detail["split"] = split_for_frame(detail["frame"].to_numpy(), episodes)
        for split in ("tuning", "heldout", "all"):
            selected = detail if split == "all" else detail.loc[detail["split"].eq(split)]
            matched = int(selected["matched"].sum()) if not selected.empty else 0
            unmatched = int(selected["unmatched"].sum()) if not selected.empty else 0
            rows.append({
                "backend": backend.backend_name, "score_unit": backend.score_unit,
                "camera_pair": f"{left_name}+{right_name}", "split": split,
                "frames": int(len(selected)), "matched_pairs": matched,
                "correct_pairs": int(selected["correct"].sum()) if matched else 0,
                "pair_association_accuracy": (
                    float(selected["correct"].sum() / matched) if matched else np.nan
                ),
                "predicted_unmatched": unmatched,
                "correct_unmatched": int(selected["unmatched_correct"].sum()) if unmatched else 0,
                "unmatched_accuracy": (
                    float(selected["unmatched_correct"].sum() / unmatched)
                    if unmatched else np.nan
                ),
                "annotated_false_positive_candidates": 0,
                "false_positive_rate": np.nan,
            })
    return pd.DataFrame(rows)


def full_triplet_assignments(
    backend: Any,
    per_view: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    xyz_bounds: tuple[float, float],
) -> tuple[pd.DataFrame, dict[int, dict[str, Any]], float]:
    grouped = {
        view: {frame: table.sort_values("geometry_slot", kind="stable") for frame, table
               in per_view.loc[per_view["view"].eq(view)].groupby("frame")}
        for view in VIEWS
    }
    frames = [frame for frame in sorted(set.intersection(*(set(grouped[v]) for v in VIEWS)))
              if all(len(grouped[view][frame]) == 3 for view in VIEWS)]
    idx = np.array([(i, j, k) for i in range(3) for j in range(3) for k in range(3)])
    all_obs = np.empty((len(frames), 27, 3, 2), dtype=float)
    identities: dict[int, list[list[str]]] = {}
    slots: dict[int, list[list[str]]] = {}
    for frame_index, frame in enumerate(frames):
        tables = [grouped[view][frame] for view in VIEWS]
        points = [table[["mouth_x", "mouth_y"]].to_numpy(float) for table in tables]
        all_obs[frame_index, :, 0] = points[0][idx[:, 0]]
        all_obs[frame_index, :, 1] = points[1][idx[:, 1]]
        all_obs[frame_index, :, 2] = points[2][idx[:, 2]]
        identities[frame] = [table["human_identity"].astype(str).tolist() for table in tables]
        slots[frame] = [table["geometry_slot"].astype(str).tolist() for table in tables]

    started = time.perf_counter()
    flat = all_obs.reshape(-1, 3, 2)
    xyz = np.asarray(backend.triangulate(flat), dtype=float).reshape(len(frames), 27, 3)
    scores = np.asarray(backend.score_hypotheses(flat, xyz.reshape(-1, 3)), dtype=float).reshape(
        len(frames), 27
    )
    elapsed = time.perf_counter() - started
    low, high = xyz_bounds
    valid = np.all(np.isfinite(xyz), axis=2) & np.all((xyz >= low) & (xyz <= high), axis=2)
    scores = np.where(valid, scores, np.inf)

    rows = []
    selected_by_frame: dict[int, dict[str, Any]] = {}
    side_permutations = list(permutations(range(3)))
    for frame_index, frame in enumerate(frames):
        best: tuple[float, tuple[int, ...], tuple[int, ...]] | None = None
        for perm1 in side_permutations:
            for perm2 in side_permutations:
                indices = tuple(i * 9 + perm1[i] * 3 + perm2[i] for i in range(3))
                total = float(np.sum(scores[frame_index, list(indices)]))
                if np.isfinite(total) and (best is None or total < best[0]):
                    best = (total, perm1, perm2)
        split = str(split_for_frame(np.array([frame]), episodes)[0])
        if best is None:
            rows.append({
                "frame": frame, "split": split, "selected_groups": 0,
                "correct_groups": 0, "all_three_correct": False,
            })
            continue
        _, perm1, perm2 = best
        points, truth, group_correct, selected_indices = [], [], [], []
        for i in range(3):
            candidate_index = i * 9 + perm1[i] * 3 + perm2[i]
            selected_indices.append(candidate_index)
            points.append(xyz[frame_index, candidate_index].tolist())
            group_ids = (
                identities[frame][0][i], identities[frame][1][perm1[i]],
                identities[frame][2][perm2[i]],
            )
            truth.append(group_ids[0])
            group_correct.append(len(set(group_ids)) == 1)
        selected_by_frame[frame] = {
            "points": points, "truth": truth, "correct": group_correct,
            "cam0_slots": slots[frame][0],
            "scores": scores[frame_index, selected_indices].tolist(),
        }
        rows.append({
            "frame": frame, "split": split, "selected_groups": 3,
            "correct_groups": int(sum(group_correct)),
            "all_three_correct": bool(all(group_correct)),
        })
    return pd.DataFrame(rows), selected_by_frame, elapsed


def summarize_triplets(
    backend: Any, details: pd.DataFrame, total_frames: int, elapsed: float
) -> pd.DataFrame:
    rows = []
    for split in ("tuning", "heldout", "all"):
        selected = details if split == "all" else details.loc[details["split"].eq(split)]
        groups = int(selected["selected_groups"].sum()) if not selected.empty else 0
        rows.append({
            "backend": backend.backend_name, "score_unit": backend.score_unit,
            "split": split, "eligible_full_view_frames": int(len(selected)),
            "frames_with_three_selected": int((selected["selected_groups"] == 3).sum()),
            "all_three_correct_frames": int(selected["all_three_correct"].sum()),
            "correct_groups": int(selected["correct_groups"].sum()),
            "correct_cross_view_grouping_rate": (
                float(selected["correct_groups"].sum() / groups) if groups else np.nan
            ),
            "three_fish_valid_coverage_of_full_timeline": (
                float((selected["selected_groups"] == 3).sum() / total_frames)
                if split == "all" else np.nan
            ),
            "assignment_runtime_s": elapsed if split == "all" else np.nan,
            "assignment_throughput_frames_s": (
                float(len(details) / elapsed) if split == "all" and elapsed else np.nan
            ),
        })
    return pd.DataFrame(rows)


def production_selector_assignments(
    backend: Any,
    per_view: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    xyz_bounds: tuple[float, float],
) -> tuple[pd.DataFrame, dict[int, dict[str, Any]], float]:
    """Run the production mutually-exclusive selector on all reviewed frames.

    Candidate rows retain their frozen ``geometry_slot`` order, so the saved
    ``perm_cam*`` indices can be mapped back to reviewed human identities.
    Frames may have fewer than three observations in any camera; direct
    two-view hypotheses are therefore exercised rather than restricted to the
    exactly-three-view clean cohort.
    """

    tables: dict[int, list[pd.DataFrame]] = {}
    points_by_frame: dict[int, dict[str, list[tuple[float, float]]]] = {}
    empty = per_view.iloc[:0].copy()
    for frame, frame_table in per_view.groupby("frame", sort=True):
        view_tables = [
            frame_table.loc[frame_table["view"].eq(view)].sort_values(
                "geometry_slot", kind="stable"
            ).reset_index(drop=True)
            for view in VIEWS
        ]
        if sum(not table.empty for table in view_tables) < 2:
            continue
        frame_int = int(frame)
        tables[frame_int] = view_tables
        points_by_frame[frame_int] = {
            f"cam{view_index}": [
                (float(row.mouth_x), float(row.mouth_y))
                for row in (view_tables[view_index] if view_tables else empty).itertuples(index=False)
            ]
            for view_index in range(3)
        }
    started = time.perf_counter()
    selected_raw = compute_best_assignments_per_frame(
        backend,
        points_by_frame,
        xyz_bounds=xyz_bounds,
        num_individuals=3,
        num_candidates=3,
        recover_missing_views=True,
        max_residual_per_fish=None,
        global_max_states=None,
    )
    elapsed = time.perf_counter() - started
    details, selected_by_frame = [], {}
    for frame in sorted(points_by_frame):
        info = selected_raw.get(frame, {})
        groups = info.get("groups", {})
        perms = [list(info.get(f"perm_cam{view}", [])) for view in range(3)]
        synthetic = list(info.get("synthetic_views", []))
        points, truth, correct, missing_views = [], [], [], []
        for group_index, label in enumerate(("a", "b", "c")):
            point = np.asarray(groups.get(label, [np.nan, np.nan, np.nan]), float)
            points.append(point.tolist())
            identities = []
            observed_views = []
            for view_index in range(3):
                slot = perms[view_index][group_index] if group_index < len(perms[view_index]) else -1
                if 0 <= slot < len(tables[frame][view_index]):
                    identities.append(str(tables[frame][view_index].iloc[slot]["human_identity"]))
                    observed_views.append(view_index)
            truth.append(identities[0] if identities else "")
            is_valid = bool(np.all(np.isfinite(point)))
            correct.append(is_valid and len(identities) >= 2 and len(set(identities)) == 1)
            if group_index < len(synthetic) and int(synthetic[group_index]) >= 0:
                missing_views.append(VIEWS[int(synthetic[group_index])])
            elif len(observed_views) == 2:
                missing_views.append(VIEWS[({0, 1, 2} - set(observed_views)).pop()])
            else:
                missing_views.append("")
        selected_by_frame[frame] = {
            "points": points, "truth": truth, "correct": correct,
            "missing_views": missing_views,
        }
        valid = np.array([np.all(np.isfinite(point)) for point in points], bool)
        details.append({
            "frame": frame,
            "split": str(split_for_frame(np.array([frame]), episodes)[0]),
            "valid_groups": int(valid.sum()),
            "correct_groups": int(sum(correct)),
            "all_three_valid": bool(valid.all()),
            "all_three_correct": bool(all(correct)),
            "two_view_groups": int(sum(bool(value) for value in missing_views)),
        })
    return pd.DataFrame(details), selected_by_frame, elapsed


def summarize_production_selector(
    backend: Any, details: pd.DataFrame, total_frames: int, elapsed: float
) -> pd.DataFrame:
    rows = []
    for split in ("tuning", "heldout", "all"):
        selected = details if split == "all" else details.loc[details["split"].eq(split)]
        valid_groups = int(selected["valid_groups"].sum()) if not selected.empty else 0
        rows.append({
            "backend": backend.backend_name, "score_unit": backend.score_unit,
            "split": split, "input_frames_with_at_least_two_cameras": int(len(selected)),
            "valid_groups": valid_groups,
            "correct_groups": int(selected["correct_groups"].sum()) if valid_groups else 0,
            "correct_grouping_rate": (
                float(selected["correct_groups"].sum() / valid_groups) if valid_groups else np.nan
            ),
            "frames_with_all_three_valid": int(selected["all_three_valid"].sum()),
            "frames_with_all_three_correct": int(selected["all_three_correct"].sum()),
            "three_fish_valid_coverage_full_timeline": (
                float(selected["all_three_valid"].sum() / total_frames)
                if split == "all" else np.nan
            ),
            "direct_two_view_groups": int(selected["two_view_groups"].sum()),
            "runtime_s": elapsed if split == "all" else np.nan,
            "throughput_frames_s": (
                float(len(details) / elapsed) if split == "all" and elapsed else np.nan
            ),
        })
    return pd.DataFrame(rows)


def _distribution(values: Iterable[float], prefix: str) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {f"{prefix}_n": 0, f"{prefix}_median": np.nan, f"{prefix}_p95": np.nan,
                f"{prefix}_max": np.nan}
    return {
        f"{prefix}_n": int(len(array)), f"{prefix}_median": float(np.median(array)),
        f"{prefix}_p95": float(np.percentile(array, 95)), f"{prefix}_max": float(np.max(array)),
    }


def frozen_xyz_metrics(
    backend: Any, clean: pd.DataFrame, episodes: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, float]:
    observations = np.stack([
        clean[list(MOUTH_COLUMNS[view])].to_numpy(float) for view in VIEWS
    ], axis=1)
    started = time.perf_counter()
    xyz = np.asarray(backend.triangulate(observations), dtype=float)
    scores = np.asarray(backend.score_hypotheses(observations, xyz), dtype=float)
    elapsed = time.perf_counter() - started
    reference = clean[["X", "Y", "Z"]].to_numpy(float)
    delta = np.linalg.norm(xyz - reference, axis=1)
    split_values = split_for_frame(clean["frame"].to_numpy(int), episodes)
    rows = []
    for split in ("tuning", "heldout", "all"):
        mask = np.ones(len(clean), dtype=bool) if split == "all" else split_values == split
        row = {
            "backend": backend.backend_name, "score_unit": backend.score_unit,
            "split": split,
            **_distribution(scores[mask], "hypothesis_score"),
            **_distribution(delta[mask], "difference_to_reviewed_polynomial_reference_mm"),
            "runtime_s": elapsed if split == "all" else np.nan,
            "throughput_observations_s": (
                float(len(clean) / elapsed) if split == "all" and elapsed else np.nan
            ),
        }
        rows.append(row)
    for missing, missing_view in enumerate(VIEWS):
        masked = observations.copy()
        masked[:, missing] = np.nan
        two_xyz = np.asarray(backend.triangulate(masked), dtype=float)
        error = np.linalg.norm(two_xyz - xyz, axis=1)
        for split in ("tuning", "heldout", "all"):
            mask = np.ones(len(clean), dtype=bool) if split == "all" else split_values == split
            rows.append({
                "backend": backend.backend_name, "score_unit": backend.score_unit,
                "split": split, "missing_camera": missing_view,
                **_distribution(error[mask], "two_vs_three_view_error_mm"),
            })
    return pd.DataFrame(rows), xyz, elapsed


def natural_transition_metrics(
    backend: Any,
    per_view: pd.DataFrame,
    clean: pd.DataFrame,
    episodes: pd.DataFrame,
    event_frames: set[int],
) -> pd.DataFrame:
    clean_lookup = {
        (int(row.frame), str(row.human_identity)): np.array([row.X, row.Y, row.Z], float)
        for row in clean.itertuples(index=False)
    }
    observations: dict[tuple[int, str], np.ndarray] = {}
    for (frame, identity), table in per_view.groupby(["frame", "human_identity"], sort=True):
        obs = np.full((3, 2), np.nan)
        for row in table.itertuples(index=False):
            obs[VIEW_INDEX[str(row.view)]] = (row.mouth_x, row.mouth_y)
        observations[(int(frame), str(identity))] = obs
    candidates: list[dict[str, Any]] = []
    for (frame, identity), obs in observations.items():
        finite = np.all(np.isfinite(obs), axis=1)
        if int(finite.sum()) != 2 or any(abs(frame - event) <= 5 for event in event_frames):
            continue
        needed = [(frame + offset, identity) for offset in (-2, -1, 1, 2)]
        if any(key not in clean_lookup for key in needed):
            continue
        p_m2, p_m1, p_p1, p_p2 = (clean_lookup[key] for key in needed)
        v_before, v_after = p_m1 - p_m2, p_p2 - p_p1
        speed_ok = max(np.linalg.norm(v_before), np.linalg.norm(v_after)) <= 3.0
        acceleration_ok = np.linalg.norm(v_after - v_before) <= 2.0
        denom = np.linalg.norm(v_before) * np.linalg.norm(v_after)
        turn_cosine = float(np.dot(v_before, v_after) / denom) if denom > 1e-9 else 1.0
        if not speed_ok or not acceleration_ok or turn_cosine < 0.5:
            continue
        candidates.append({
            "frame": frame, "identity": identity, "observation": obs,
            "missing_camera": VIEWS[int(np.where(~finite)[0][0])],
            "reference": 0.5 * (p_m1 + p_p1),
        })
    if not candidates:
        return pd.DataFrame(columns=[
            "backend", "score_unit", "split", "missing_camera", "events",
            "trajectory_error_mm_n", "trajectory_error_mm_median",
            "trajectory_error_mm_p95", "trajectory_error_mm_max",
        ])
    xyz = np.asarray(backend.triangulate(np.stack([row["observation"] for row in candidates])))
    errors = np.array([
        np.linalg.norm(point - row["reference"]) for point, row in zip(xyz, candidates, strict=True)
    ])
    frames = np.array([row["frame"] for row in candidates])
    splits = split_for_frame(frames, episodes)
    rows = []
    for missing_camera in VIEWS:
        camera_mask = np.array([row["missing_camera"] == missing_camera for row in candidates])
        for split in ("tuning", "heldout", "all"):
            mask = camera_mask & (np.ones(len(candidates), bool) if split == "all" else splits == split)
            rows.append({
                "backend": backend.backend_name, "score_unit": backend.score_unit,
                "split": split, "missing_camera": missing_camera,
                "events": int(mask.sum()), **_distribution(errors[mask], "trajectory_error_mm"),
            })
    return pd.DataFrame(rows)


def tracking_metrics(
    backend: Any,
    selected_by_frame: dict[int, dict[str, Any]],
    episodes: pd.DataFrame,
    event_frames: set[int],
) -> pd.DataFrame:
    def _run(mask_incorrect_groups: bool) -> tuple[pd.DataFrame, list[float]]:
        best = {}
        for frame, selected in selected_by_frame.items():
            groups = {}
            for index, label in enumerate(("a", "b", "c")):
                point = selected["points"][index]
                if mask_incorrect_groups and not selected["correct"][index]:
                    point = [np.nan, np.nan, np.nan]
                groups[label] = point
            best[frame] = {
                "groups": groups, "recovered": [False, False, False],
                "synthetic_views": [-1, -1, -1],
            }
        tracked = apply_cross_frame_tracking(
            best, max_jump_mm=8.0, max_jump_hard_cap_mm=80.0,
            max_gap_frames=100, num_individuals=3,
        )
        rows, speeds = [], []
        last_point: dict[str, tuple[int, np.ndarray]] = {}
        for frame in sorted(tracked):
            info = tracked[frame]
            mapping = info.get("id_mapping", [-1, -1, -1])
            for id_index, tracker_id in enumerate(("ID_1", "ID_2", "ID_3")):
                source = int(mapping[id_index]) if id_index < len(mapping) else -1
                if source >= 0 and source < len(selected_by_frame[frame]["truth"]):
                    rows.append({
                        "frame": frame, "tracker_id": tracker_id,
                        "human_identity": selected_by_frame[frame]["truth"][source],
                        "group_correct": bool(selected_by_frame[frame]["correct"][source]),
                    })
                point = np.asarray(info.get("groups", {}).get(tracker_id, [np.nan] * 3), float)
                if np.all(np.isfinite(point)):
                    if tracker_id in last_point and frame - last_point[tracker_id][0] == 1:
                        speeds.append(float(np.linalg.norm(point - last_point[tracker_id][1])))
                    last_point[tracker_id] = (frame, point)
        return pd.DataFrame(rows), speeds

    def _evaluate(association: pd.DataFrame, *, require_group_correct: bool) -> dict[str, Any]:
        if association.empty:
            return {"mapping": {}, "tuning_rows": 0, "heldout_rows": 0,
                    "accuracy": np.nan, "switches": 0, "event_rows": 0,
                    "event_accuracy": np.nan}
        association = association.copy()
        association["split"] = split_for_frame(association["frame"].to_numpy(), episodes)
        tuning = association.loc[association["split"].eq("tuning")]
        best_mapping, best_correct = None, -1
        for perm in permutations(IDENTITIES):
            mapping = dict(zip(("ID_1", "ID_2", "ID_3"), perm, strict=True))
            correct = int((tuning["tracker_id"].map(mapping) == tuning["human_identity"]).sum())
            if correct > best_correct:
                best_mapping, best_correct = mapping, correct
        assert best_mapping is not None
        association["identity_correct"] = association["tracker_id"].map(best_mapping).eq(
            association["human_identity"]
        )
        association["correct"] = association["identity_correct"] & (
            association["group_correct"] if require_group_correct else True
        )
        heldout = association.loc[association["split"].eq("heldout")]
        switches = 0
        for _, table in heldout.sort_values("frame").groupby("human_identity"):
            switches += int((table["frame"].diff().eq(1) & table["tracker_id"].ne(
                table["tracker_id"].shift()
            )).sum())
        near_event = heldout["frame"].map(
            lambda frame: any(abs(frame - event) <= 10 for event in event_frames)
        )
        return {
            "mapping": best_mapping, "tuning_rows": int(len(tuning)),
            "heldout_rows": int(len(heldout)),
            "accuracy": float(heldout["correct"].mean()) if len(heldout) else np.nan,
            "switches": switches, "event_rows": int(near_event.sum()),
            "event_accuracy": (
                float(heldout.loc[near_event, "correct"].mean()) if near_event.any() else np.nan
            ),
        }

    linker_association, linker_speeds = _run(mask_incorrect_groups=True)
    end_association, end_speeds = _run(mask_incorrect_groups=False)
    linker = _evaluate(linker_association, require_group_correct=False)
    end_to_end = _evaluate(end_association, require_group_correct=True)
    return pd.DataFrame([{
        "backend": backend.backend_name,
        "linker_input_scope": "correctly_grouped candidates only; full timeline",
        "linker_tuning_id_mapping": json.dumps(linker["mapping"], sort_keys=True),
        "linker_tuning_rows": linker["tuning_rows"],
        "linker_heldout_rows": linker["heldout_rows"],
        "linker_heldout_identity_accuracy": linker["accuracy"],
        "linker_heldout_id_switch_evidence": linker["switches"],
        "linker_speed_scope": "correctly_grouped input, consecutive full-timeline steps",
        "linker_consecutive_steps": int(len(linker_speeds)),
        "linker_speed_spikes_gt8_mm": int(np.sum(np.asarray(linker_speeds) > 8.0)),
        "linker_speed_spikes_gt20_mm": int(np.sum(np.asarray(linker_speeds) > 20.0)),
        "linker_speed_step_p95_mm": (
            float(np.percentile(linker_speeds, 95)) if linker_speeds else np.nan
        ),
        "end_to_end_definition": "tracker identity correct AND cross-view group correct",
        "end_to_end_tuning_id_mapping": json.dumps(end_to_end["mapping"], sort_keys=True),
        "end_to_end_heldout_rows": end_to_end["heldout_rows"],
        "end_to_end_heldout_accuracy": end_to_end["accuracy"],
        "end_to_end_heldout_id_switch_evidence": end_to_end["switches"],
        "end_to_end_heldout_review_event_rows": end_to_end["event_rows"],
        "end_to_end_heldout_review_event_accuracy": end_to_end["event_accuracy"],
        "end_to_end_speed_scope": "all selected groups, consecutive full-timeline steps",
        "end_to_end_consecutive_steps": int(len(end_speeds)),
        "end_to_end_speed_spikes_gt8_mm": int(np.sum(np.asarray(end_speeds) > 8.0)),
        "end_to_end_speed_spikes_gt20_mm": int(np.sum(np.asarray(end_speeds) > 20.0)),
    }])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--refractive-model-dir", type=Path, required=True)
    parser.add_argument("--polynomial-model-dir", type=Path, required=True)
    parser.add_argument("--dlt-model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-frames", type=int, default=1500)
    parser.add_argument("--xyz-bounds", type=float, nargs=2, default=(-20.0, 220.0))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    per_view, clean, fingerprints = validate_reviewed_data(args.review_dir)
    low, high = (float(value) for value in args.xyz_bounds)
    clean_xyz = clean[["X", "Y", "Z"]].to_numpy(float)
    if not np.all((clean_xyz >= low) & (clean_xyz <= high)):
        raise ValueError(
            f"clean reviewed XYZ falls outside requested benchmark bounds [{low}, {high}] mm"
        )
    start_frame = int(min(per_view["frame"].min(), clean["frame"].min()))
    end_frame = int(max(per_view["frame"].max(), clean["frame"].max()))
    episodes = build_episode_split(start_frame, end_frame, episode_frames=args.episode_frames)
    events_path = args.review_dir / "suggested_3d_mapping_events.csv"
    event_frames = set(pd.read_csv(events_path)["frame"].astype(int)) if events_path.exists() else set()

    backends = [
        BackendRun("fixed_refractive", Calibration(args.refractive_model_dir, backend="refractive")),
        BackendRun("production_polynomial", Calibration(args.polynomial_model_dir, backend="polynomial")),
    ]
    if args.dlt_model_dir is not None:
        backends.append(BackendRun("projective_dlt", DLTBackend(args.dlt_model_dir)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.output_dir / "episode_split.csv", index=False)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pair_tables, triplet_tables, production_tables = [], [], []
    xyz_tables, transition_tables, tracking_tables = [], [], []
    backend_records = []
    total_frames = end_frame - start_frame + 1
    for run in backends:
        backend = run.backend
        if backend.backend_name != run.label and run.label not in backend.backend_name:
            # Labels are descriptive; persisted metadata always uses the resolved backend name.
            pass
        started = time.perf_counter()
        pair_result = pair_association_metrics(backend, per_view, episodes)
        pair_tables.append(pair_result)
        triplet_detail, _, assignment_elapsed = full_triplet_assignments(
            backend, per_view, episodes, xyz_bounds=tuple(args.xyz_bounds)
        )
        triplet_result = summarize_triplets(
            backend, triplet_detail, total_frames, assignment_elapsed
        )
        triplet_tables.append(triplet_result)
        production_detail, production_selected, production_elapsed = (
            production_selector_assignments(
                backend, per_view, episodes, xyz_bounds=tuple(args.xyz_bounds)
            )
        )
        production_result = summarize_production_selector(
            backend, production_detail, total_frames, production_elapsed
        )
        production_tables.append(production_result)
        frozen, _, frozen_elapsed = frozen_xyz_metrics(backend, clean, episodes)
        xyz_tables.append(frozen)
        transition_result = natural_transition_metrics(
            backend, per_view, clean, episodes, event_frames
        )
        transition_tables.append(transition_result)
        tracking_result = tracking_metrics(
            backend, production_selected, episodes, event_frames
        )
        tracking_tables.append(tracking_result)
        backend_record = {
            "requested_label": run.label, "resolved_backend": backend.backend_name,
            "score_unit": backend.score_unit,
            "provenance": _json_safe(backend.provenance),
            "capabilities": _json_safe(backend.capabilities),
            "total_runtime_s": float(time.perf_counter() - started),
            "frozen_xyz_runtime_s": frozen_elapsed,
            "assignment_runtime_s": assignment_elapsed,
            "production_selector_runtime_s": production_elapsed,
        }
        backend_records.append(backend_record)

        # Durable per-backend checkpoints make late failures inspectable and
        # prevent a completed backend from disappearing merely because a later
        # comparison backend failed. A checkpoint is complete only after all
        # six stage tables and its metadata have been written.
        checkpoint_tables = {
            "pair_association": pair_result,
            "triplet_association": triplet_result,
            "production_selector": production_result,
            "frozen_xyz": frozen,
            "natural_transitions": transition_result,
            "tracking": tracking_result,
        }
        checkpoint_artifacts = {}
        for stage_name, stage_table in checkpoint_tables.items():
            filename = f"{backend.backend_name}_{stage_name}.csv"
            stage_table.to_csv(checkpoint_dir / filename, index=False)
            checkpoint_artifacts[stage_name] = filename
        checkpoint_payload = {
            "schema_version": "reviewed_backend_checkpoint_v1",
            "status": "complete",
            "backend_run": backend_record,
            "artifacts": checkpoint_artifacts,
        }
        (checkpoint_dir / f"{backend.backend_name}_checkpoint.json").write_text(
            json.dumps(_json_safe(checkpoint_payload), indent=2) + "\n",
            encoding="utf-8",
        )

    outputs = {
        "pair_association": pd.concat(pair_tables, ignore_index=True),
        "triplet_association": pd.concat(triplet_tables, ignore_index=True),
        "production_selector": pd.concat(production_tables, ignore_index=True),
        "frozen_xyz": pd.concat(xyz_tables, ignore_index=True),
        "natural_transitions": pd.concat(transition_tables, ignore_index=True),
        "tracking": pd.concat(tracking_tables, ignore_index=True),
    }
    artifacts = {"episode_split": "episode_split.csv", "checkpoints": "checkpoints/"}
    for name, table in outputs.items():
        filename = f"{name}.csv"
        table.to_csv(args.output_dir / filename, index=False)
        artifacts[name] = filename
    summary = {
        "schema_version": "reviewed_calibration_backend_benchmark_v1",
        "frame_range": {"start": start_frame, "end": end_frame, "count": total_frames},
        "coordinate_definition": {
            "benchmark_point": "SLEAP mouth",
            "geometry_match_qc_point": "visible-node centroid",
            "warning": "Centroid match distances are provenance/QC only; no centroid was triangulated.",
        },
        "reviewed_data": fingerprints,
        "review_qc": {
            "clean_rows": int(len(clean)),
            "frames_with_all_three_clean_identities": int(
                clean.groupby("frame")["human_identity"].nunique().eq(3).sum()
            ),
            "clean_all_three_frame_coverage": float(
                clean.groupby("frame")["human_identity"].nunique().eq(3).sum() / total_frames
            ),
            "clean_filters": {
                "all_views_agree": True,
                "max_geometry_to_centroid_match_px": float(clean["filter_max_geometry_match_px"].iloc[0]),
                "max_polynomial_reprojection_px": float(clean["filter_max_reprojection_px"].iloc[0]),
                "interpolation": "none",
            },
        },
        "episode_policy": {
            "episode_frames": args.episode_frames, "cycle": 5, "heldout_index": 4,
            "fitting": "No thresholds, affine corrections, or shift weights were fit. Only the arbitrary tracker-ID permutation was learned from tuning episodes.",
        },
        "backend_runs": backend_records,
        "metric_boundaries": {
            "absolute_calibration": "Owned by independent held-out rigid-object and wet-board validation, not fish tracks.",
            "assignment": "Human-identity agreement of reviewed mouth observations.",
            "frozen_xyz": "Backend consistency and two-view versus own-three-view error; reviewed XYZ is a polynomial reference, not ground truth.",
            "tracking": "Continuity and reviewed identity agreement after geometry-only cross-frame linking.",
        },
        "false_positive_limit": "The corrected per-view table contains only reviewed matched observations. It tests missing/unmatched handling but has no retained annotated false-positive candidates, so FP rate is not estimable.",
        "artifacts": artifacts,
    }
    (args.output_dir / "benchmark_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "backends": [record["resolved_backend"] for record in backend_records],
        "clean_rows": len(clean), "total_runtime_s": sum(
            record["total_runtime_s"] for record in backend_records
        ),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
