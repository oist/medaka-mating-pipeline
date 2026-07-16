from __future__ import annotations

"""Benchmark 2-view <-> 3-view reconstruction discontinuities.

The benchmark freezes the cross-camera assignments from an existing
``*_best.json`` file and compares three geometry backends:

* the production polynomial model with a regressed synthetic missing view;
* projective DLT with direct two- or three-ray least-squares intersection;
* the fitted flat refractive model with the same variable-view ray objective.

Two complementary cohorts are measured:

1. Clean three-view observations are counterfactually masked one camera at a
   time.  This gives a same-frame three-view reference and is used to fit a
   robust affine correction for each missing camera.
2. Naturally recovered two-view runs bracketed by three-view frames are
   compared with constant-velocity interpolation between their two anchors.

The affine corrections are trained only on time-blocked counterfactual rows;
held-out counterfactual rows and natural transitions remain evaluation data.
"""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from scripts.calibration.refractive_geometry import FlatPortCamera
from scripts.reconstruction_3d.calibration import Calibration
from scripts.reconstruction_3d.load_2d_slp import (
    VIEW_TO_CAM,
    _instance_point,
    _instances_in_slot_order,
)


VIEW_NAMES = ("top_cam", "side1_cam", "side2_cam")
CAMERA_NAMES = ("cam0", "cam1", "cam2")


@dataclass(frozen=True)
class TransitionRun:
    fish_index: int
    missing_view: int
    start_frame: int
    end_frame: int

    @property
    def run_length(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass
class AffineShiftCorrection:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    ridge_alpha: float
    huber_iterations: int

    def predict_delta(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
        standardized = (xyz - self.feature_mean) / self.feature_scale
        design = np.column_stack((np.ones(len(xyz)), standardized))
        return design @ self.coefficients

    def correct(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
        return xyz + self.predict_delta(xyz)

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "ridge_alpha": float(self.ridge_alpha),
            "huber_iterations": int(self.huber_iterations),
            "formula": (
                "delta_xyz = [1, (xyz-feature_mean)/feature_scale] @ coefficients; "
                "corrected_xyz = xyz + delta_xyz"
            ),
        }


class GeometryBackend:
    name: str

    def reconstruct(self, image_coordinates: np.ndarray, views: Sequence[int]) -> np.ndarray:
        raise NotImplementedError


class ProductionPolynomialBackend(GeometryBackend):
    name = "production_polynomial"

    def __init__(self, model_dir: Path) -> None:
        self.calibration = Calibration(model_dir)

    def reconstruct(self, image_coordinates: np.ndarray, views: Sequence[int]) -> np.ndarray:
        values = np.asarray(image_coordinates, dtype=float).reshape(-1, 3, 2)
        views = tuple(int(view) for view in views)
        if len(views) == 3:
            return np.asarray(
                self.calibration._model_im2xyz.predict(values.reshape(-1, 6)),
                dtype=float,
            )
        if len(views) != 2:
            raise ValueError("production reconstruction requires two or three views")
        missing = ({0, 1, 2} - set(views)).pop()
        completed = values.copy()
        if missing == 0:
            pair = np.hstack((values[:, 1], values[:, 2]))
            model = self.calibration._model_xy_from_xz_yz
        elif missing == 1:
            pair = np.hstack((values[:, 0], values[:, 2]))
            model = self.calibration._model_xz_from_xy_yz
        else:
            # The helper filename encodes its non-camera-order feature contract.
            pair = np.hstack((values[:, 1], values[:, 0]))
            model = self.calibration._model_yz_from_xz_xy
        if model is None:
            raise RuntimeError(f"production calibration lacks helper for missing cam{missing}")
        completed[:, missing] = np.asarray(model.predict(pair), dtype=float).reshape(-1, 2)
        return np.asarray(
            self.calibration._model_im2xyz.predict(completed.reshape(-1, 6)),
            dtype=float,
        )


class RayBackend(GeometryBackend):
    def __init__(self, name: str, cameras: Sequence[FlatPortCamera]) -> None:
        if len(cameras) != 3:
            raise ValueError("ray backend requires exactly three cameras")
        self.name = name
        self.cameras = tuple(cameras)

    def reconstruct(self, image_coordinates: np.ndarray, views: Sequence[int]) -> np.ndarray:
        values = np.asarray(image_coordinates, dtype=float).reshape(-1, 3, 2)
        views = tuple(int(view) for view in views)
        if len(views) < 2:
            raise ValueError("ray reconstruction requires at least two views")
        origins = []
        directions = []
        for view in views:
            origin, direction = self.cameras[view].backproject(values[:, view])
            origins.append(origin)
            directions.append(direction)
        origins_array = np.stack(origins, axis=1)
        directions_array = np.stack(directions, axis=1)
        identity = np.eye(3, dtype=float)
        projectors = (
            identity[None, None]
            - directions_array[..., :, None] * directions_array[..., None, :]
        )
        system = np.sum(projectors, axis=1)
        rhs = np.sum(projectors @ origins_array[..., None], axis=1)
        try:
            return np.linalg.solve(system, rhs).squeeze(-1)
        except np.linalg.LinAlgError:
            return np.einsum(
                "nij,nj->ni", np.linalg.pinv(system), rhs.squeeze(-1)
            )


def load_dlt_cameras(model_dir: Path) -> tuple[FlatPortCamera, ...]:
    calibration = Calibration(model_dir)
    model = calibration._model_im2xyz
    cameras = getattr(model, "cameras", None)
    if cameras is None:
        cameras = tuple(
            FlatPortCamera.from_json(model_dir / f"cam{camera}_pinhole_dlt.json")
            for camera in range(3)
        )
    return tuple(cameras)


def load_refractive_cameras(camera_dir: Path) -> tuple[FlatPortCamera, ...]:
    paths = [camera_dir / f"cam{camera}_bundle_refractive.json" for camera in range(3)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing refractive camera JSON: " + ", ".join(missing))
    return tuple(FlatPortCamera.from_json(path) for path in paths)


def extract_assignment_arrays(
    best_payload: dict[str, object],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Return identities, raw slots, synthetic view, and saved linked XYZ."""
    frames = best_payload.get("frames")
    identities = best_payload.get("id_labels")
    if not isinstance(frames, dict) or not isinstance(identities, list):
        raise ValueError("best JSON must contain frames and id_labels")
    identity_names = [str(identity) for identity in identities]
    identity_index = {identity: index for index, identity in enumerate(identity_names)}
    max_frame = max(int(frame) for frame in frames)
    slots = np.full((max_frame + 1, len(identity_names), 3), -1, dtype=np.int16)
    synthetic = np.full((max_frame + 1, len(identity_names)), -2, dtype=np.int8)
    saved_xyz = np.full((max_frame + 1, len(identity_names), 3), np.nan, dtype=float)
    for frame_key, info_value in frames.items():
        if not isinstance(info_value, dict):
            continue
        frame = int(frame_key)
        for view in range(3):
            mapping = info_value.get(f"cam{view}_to_id") or []
            for slot, identity in enumerate(mapping):
                if identity in identity_index:
                    slots[frame, identity_index[str(identity)], view] = int(slot)
        if "synthetic_views" not in info_value:
            raise ValueError(
                "best JSON lacks ID-ordered synthetic_views; regenerate it with the "
                "current reconstruction code before transition benchmarking"
            )
        synthetic_views = info_value.get("synthetic_views") or []
        for fish_index in range(min(len(identity_names), len(synthetic_views))):
            synthetic[frame, fish_index] = int(synthetic_views[fish_index])
        groups = info_value.get("groups") or {}
        for identity, fish_index in identity_index.items():
            point = np.asarray(groups.get(identity, []), dtype=float)
            if point.shape == (3,):
                saved_xyz[frame, fish_index] = point
    return identity_names, slots, synthetic, saved_xyz


def load_assigned_image_coordinates(
    slp_dir: Path,
    slots: np.ndarray,
    *,
    num_candidates: int,
    filter_fp: bool = True,
    min_span_px: float = 15.0,
    dup_dist_px: float = 12.0,
) -> np.ndarray:
    """Load only the raw candidate slots selected by the frozen assignment."""
    try:
        import sleap_io as sio
    except ImportError as exc:
        raise RuntimeError("sleap-io is required; install the sleap extra") from exc

    n_frames, n_fish, _ = slots.shape
    output = np.full((n_frames, n_fish, 3, 2), np.nan, dtype=float)
    for view_index, view_name in enumerate(VIEW_NAMES):
        path = slp_dir / f"{view_name}.slp"
        if not path.exists():
            raise FileNotFoundError(path)
        labels = sio.load_slp(str(path))
        node_names = (
            [node.name for node in labels.skeletons[0].nodes]
            if labels.skeletons
            else []
        )
        for labeled_frame in labels.labeled_frames:
            frame = int(labeled_frame.frame_idx)
            if frame < 0 or frame >= n_frames:
                continue
            ordered = _instances_in_slot_order(
                labeled_frame.instances,
                node_names=node_names,
                anchor_node=None,
                filter_fp=filter_fp,
                min_span_px=min_span_px,
                dup_dist_px=dup_dist_px,
            )[:num_candidates]
            for fish_index in range(n_fish):
                slot = int(slots[frame, fish_index, view_index])
                if slot < 0 or slot >= len(ordered):
                    continue
                x, y, _score = _instance_point(
                    ordered[slot], node_names=node_names, anchor_node=None
                )
                if np.isfinite(x) and np.isfinite(y):
                    output[frame, fish_index, view_index] = (x, y)
        del labels
    return output


def observation_state(
    coordinates: np.ndarray, synthetic_views: np.ndarray
) -> np.ndarray:
    """Encode 3-view as 3, two-view as missing camera 0..2, else -1."""
    finite = np.isfinite(coordinates).all(axis=-1)
    count = finite.sum(axis=-1)
    state = np.full(count.shape, -1, dtype=np.int8)
    state[count == 3] = 3
    for missing_view in range(3):
        expected = finite.copy()
        valid_pair = (
            (count == 2)
            & (~expected[..., missing_view])
            & (synthetic_views == missing_view)
        )
        state[valid_pair] = missing_view
    return state


def find_bounded_transition_runs(state: np.ndarray) -> list[TransitionRun]:
    """Find constant-missing-view runs immediately bracketed by 3-view frames."""
    values = np.asarray(state, dtype=np.int8)
    if values.ndim != 2:
        raise ValueError("state must have shape (frames, fish)")
    runs: list[TransitionRun] = []
    n_frames, n_fish = values.shape
    for fish_index in range(n_fish):
        frame = 1
        while frame < n_frames - 1:
            missing_view = int(values[frame, fish_index])
            if missing_view not in (0, 1, 2) or values[frame - 1, fish_index] != 3:
                frame += 1
                continue
            end = frame
            while end + 1 < n_frames and values[end + 1, fish_index] == missing_view:
                end += 1
            if end + 1 < n_frames and values[end + 1, fish_index] == 3:
                runs.append(
                    TransitionRun(
                        fish_index=fish_index,
                        missing_view=missing_view,
                        start_frame=frame,
                        end_frame=end,
                    )
                )
            frame = end + 1
    return runs


def fit_affine_shift_correction(
    two_view_xyz: np.ndarray,
    three_view_xyz: np.ndarray,
    *,
    ridge_alpha: float = 1.0,
    huber_iterations: int = 8,
    huber_delta: float = 1.5,
) -> AffineShiftCorrection:
    """Robustly fit a position-dependent 2-view -> 3-view shift correction."""
    two = np.asarray(two_view_xyz, dtype=float).reshape(-1, 3)
    three = np.asarray(three_view_xyz, dtype=float).reshape(-1, 3)
    valid = np.isfinite(two).all(axis=1) & np.isfinite(three).all(axis=1)
    two, three = two[valid], three[valid]
    if len(two) < 20:
        raise ValueError("at least 20 finite rows are required to fit shift weights")
    mean = np.mean(two, axis=0)
    scale = np.std(two, axis=0)
    scale[scale < 1e-6] = 1.0
    design = np.column_stack((np.ones(len(two)), (two - mean) / scale))
    target = three - two
    weights = np.ones(len(two), dtype=float)
    coefficients = np.zeros((4, 3), dtype=float)
    regularizer = np.diag([0.0, ridge_alpha, ridge_alpha, ridge_alpha])
    iterations_used = 0
    for iteration in range(max(1, int(huber_iterations))):
        root_weight = np.sqrt(weights)[:, None]
        weighted_design = design * root_weight
        weighted_target = target * root_weight
        coefficients = np.linalg.solve(
            weighted_design.T @ weighted_design + regularizer,
            weighted_design.T @ weighted_target,
        )
        residual_norm = np.linalg.norm(target - design @ coefficients, axis=1)
        median = float(np.median(residual_norm))
        robust_scale = 1.4826 * float(np.median(np.abs(residual_norm - median)))
        iterations_used = iteration + 1
        if robust_scale < 1e-9:
            break
        cutoff = huber_delta * robust_scale
        new_weights = np.minimum(1.0, cutoff / np.maximum(residual_norm, 1e-12))
        if np.max(np.abs(new_weights - weights)) < 1e-4:
            weights = new_weights
            break
        weights = new_weights
    return AffineShiftCorrection(
        feature_mean=mean,
        feature_scale=scale,
        coefficients=coefficients,
        ridge_alpha=float(ridge_alpha),
        huber_iterations=iterations_used,
    )


def summarize_errors(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "mean": np.nan, "median": np.nan, "p95": np.nan, "p99": np.nan}
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def reconstruct_state_positions(
    backend: GeometryBackend,
    coordinates: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    shape = coordinates.shape[:2] + (3,)
    output = np.full(shape, np.nan, dtype=float)
    for missing_view in (3, 0, 1, 2):
        mask = state == missing_view
        if not mask.any():
            continue
        views = (0, 1, 2) if missing_view == 3 else tuple(
            view for view in range(3) if view != missing_view
        )
        output[mask] = backend.reconstruct(coordinates[mask], views)
    return output


def transition_event_rows(
    backend_name: str,
    identities: Sequence[str],
    runs: Sequence[TransitionRun],
    raw_positions: np.ndarray,
    corrected_positions: np.ndarray,
    clean_three_view: np.ndarray,
    holdout_frames: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frame_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for event_index, run in enumerate(runs):
        before = run.start_frame - 1
        after = run.end_frame + 1
        fish = run.fish_index
        raw_path = raw_positions[before : after + 1, fish]
        corrected_path = corrected_positions[before : after + 1, fish]
        if not (
            np.isfinite(raw_path).all()
            and np.isfinite(corrected_path).all()
        ):
            continue
        start_xyz = raw_path[0]
        end_xyz = raw_path[-1]
        ideal_step = float(np.linalg.norm(end_xyz - start_xyz) / (run.run_length + 1))
        raw_steps = np.linalg.norm(np.diff(raw_path, axis=0), axis=1)
        corrected_steps = np.linalg.norm(np.diff(corrected_path, axis=0), axis=1)
        raw_residuals = []
        corrected_residuals = []
        event_id = f"{backend_name}:{identities[fish]}:{run.start_frame}-{run.end_frame}"
        split = (
            "holdout"
            if bool(np.all(holdout_frames[before : after + 1]))
            else "fit_or_boundary"
        )
        has_velocity_context = (
            before - 1 >= 0
            and after + 1 < len(clean_three_view)
            and bool(clean_three_view[before - 1, fish])
            and bool(clean_three_view[after + 1, fish])
            and np.isfinite(raw_positions[before - 1, fish]).all()
            and np.isfinite(raw_positions[after + 1, fish]).all()
        )
        for offset, frame in enumerate(range(run.start_frame, run.end_frame + 1), start=1):
            alpha = offset / (run.run_length + 1)
            ideal = (1.0 - alpha) * start_xyz + alpha * end_xyz
            raw_xyz = raw_positions[frame, fish]
            corrected_xyz = corrected_positions[frame, fish]
            raw_residual = float(np.linalg.norm(raw_xyz - ideal))
            corrected_residual = float(np.linalg.norm(corrected_xyz - ideal))
            raw_residuals.append(raw_residual)
            corrected_residuals.append(corrected_residual)
            frame_rows.append(
                {
                    "event_id": event_id,
                    "model": backend_name,
                    "split": split,
                    "fish_id": identities[fish],
                    "frame": frame,
                    "missing_view": CAMERA_NAMES[run.missing_view],
                    "run_length": run.run_length,
                    "run_offset": offset,
                    "ideal_X": float(ideal[0]),
                    "ideal_Y": float(ideal[1]),
                    "ideal_Z": float(ideal[2]),
                    "raw_X": float(raw_xyz[0]),
                    "raw_Y": float(raw_xyz[1]),
                    "raw_Z": float(raw_xyz[2]),
                    "corrected_X": float(corrected_xyz[0]),
                    "corrected_Y": float(corrected_xyz[1]),
                    "corrected_Z": float(corrected_xyz[2]),
                    "raw_trajectory_residual_mm": raw_residual,
                    "corrected_trajectory_residual_mm": corrected_residual,
                    "ideal_step_mm": ideal_step,
                }
            )
        event_rows.append(
            {
                "event_id": event_id,
                "model": backend_name,
                "split": split,
                "fish_id": identities[fish],
                "missing_view": CAMERA_NAMES[run.missing_view],
                "start_frame": run.start_frame,
                "end_frame": run.end_frame,
                "run_length": run.run_length,
                "isolated": run.run_length == 1,
                "ideal_step_mm": ideal_step,
                "raw_peak_step_mm": float(np.max(raw_steps)),
                "corrected_peak_step_mm": float(np.max(corrected_steps)),
                "raw_peak_step_excess_mm": float(np.max(raw_steps) - ideal_step),
                "corrected_peak_step_excess_mm": float(
                    np.max(corrected_steps) - ideal_step
                ),
                "raw_peak_trajectory_residual_mm": float(np.max(raw_residuals)),
                "corrected_peak_trajectory_residual_mm": float(
                    np.max(corrected_residuals)
                ),
                "raw_entry_velocity_discontinuity_mm": (
                    float(
                        np.linalg.norm(
                            (raw_positions[run.start_frame, fish] - raw_positions[before, fish])
                            - (raw_positions[before, fish] - raw_positions[before - 1, fish])
                        )
                    )
                    if has_velocity_context
                    else np.nan
                ),
                "corrected_entry_velocity_discontinuity_mm": (
                    float(
                        np.linalg.norm(
                            (
                                corrected_positions[run.start_frame, fish]
                                - corrected_positions[before, fish]
                            )
                            - (
                                corrected_positions[before, fish]
                                - corrected_positions[before - 1, fish]
                            )
                        )
                    )
                    if has_velocity_context
                    else np.nan
                ),
                "raw_exit_velocity_discontinuity_mm": (
                    float(
                        np.linalg.norm(
                            (raw_positions[after, fish] - raw_positions[run.end_frame, fish])
                            - (raw_positions[after + 1, fish] - raw_positions[after, fish])
                        )
                    )
                    if has_velocity_context
                    else np.nan
                ),
                "corrected_exit_velocity_discontinuity_mm": (
                    float(
                        np.linalg.norm(
                            (
                                corrected_positions[after, fish]
                                - corrected_positions[run.end_frame, fish]
                            )
                            - (
                                corrected_positions[after + 1, fish]
                                - corrected_positions[after, fish]
                            )
                        )
                    )
                    if has_velocity_context
                    else np.nan
                ),
            }
        )
    return frame_rows, event_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--slp-dir", type=Path, required=True)
    parser.add_argument("--best-json", type=Path, required=True)
    parser.add_argument("--production-model-dir", type=Path, required=True)
    parser.add_argument("--dlt-model-dir", type=Path, required=True)
    parser.add_argument("--refractive-camera-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-candidates", type=int, default=6)
    parser.add_argument("--block-frames", type=int, default=6000)
    parser.add_argument("--holdout-block", type=int, default=4)
    parser.add_argument("--block-cycle", type=int, default=5)
    parser.add_argument("--fit-stride", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--max-transition-run", type=int, default=3)
    parser.add_argument("--max-production-residual", type=float, default=5000.0)
    parser.add_argument("--parity-tolerance-mm", type=float, default=1e-4)
    return parser


def file_fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.block_frames <= 0 or args.block_cycle <= 0:
        raise ValueError("block-frames and block-cycle must be positive")
    if not 0 <= args.holdout_block < args.block_cycle:
        raise ValueError("holdout-block must be in [0, block-cycle)")
    if args.fit_stride <= 0 or args.max_transition_run <= 0:
        raise ValueError("fit-stride and max-transition-run must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("loading frozen assignments ...", flush=True)
    payload = json.loads(args.best_json.read_text(encoding="utf-8"))
    identities, slots, synthetic_views, saved_xyz = extract_assignment_arrays(payload)
    del payload

    print("loading assigned SLEAP centroids ...", flush=True)
    coordinates = load_assigned_image_coordinates(
        args.slp_dir,
        slots,
        num_candidates=args.num_candidates,
    )
    state = observation_state(coordinates, synthetic_views)
    all_runs = find_bounded_transition_runs(state)
    runs = [
        run
        for run in all_runs
        if run.run_length <= args.max_transition_run
    ]
    print(f"bounded transition runs: {len(runs):,}", flush=True)

    production_backend = ProductionPolynomialBackend(args.production_model_dir)
    backends: list[GeometryBackend] = [
        production_backend,
        RayBackend("projective_dlt", load_dlt_cameras(args.dlt_model_dir)),
        RayBackend("flat_refractive", load_refractive_cameras(args.refractive_camera_dir)),
    ]

    frame_grid = np.broadcast_to(
        np.arange(len(coordinates), dtype=int)[:, None], state.shape
    )
    observed = state >= 0
    production_reconstructed = reconstruct_state_positions(
        production_backend, coordinates, state
    )
    parity_error = np.linalg.norm(production_reconstructed - saved_xyz, axis=2)
    parity_valid = observed & np.isfinite(saved_xyz).all(axis=2)
    parity_values = parity_error[parity_valid]
    parity_mismatch = parity_valid & (parity_error > args.parity_tolerance_mm)
    if parity_values.size == 0:
        raise RuntimeError("no finite saved groups were available for slot parity validation")
    if np.any(parity_mismatch):
        raise RuntimeError(
            "reloaded SLEAP slots do not reproduce saved production XYZ: "
            f"{int(parity_mismatch.sum()):,}/{int(parity_valid.sum()):,} rows exceed "
            f"{args.parity_tolerance_mm:g} mm; max={float(np.max(parity_values)):.6g} mm"
        )

    three_view = state == 3
    production_full = production_reconstructed[three_view]
    production_reprojection = np.asarray(
        production_backend.calibration._model_xyz2im.predict(production_full),
        dtype=float,
    )
    production_input = coordinates[three_view].reshape(-1, 6)
    residual_values = np.sum((production_input - production_reprojection) ** 2, axis=1)
    residual = np.full(state.shape, np.inf, dtype=float)
    residual[three_view] = residual_values
    in_bounds = np.all((saved_xyz >= -20.0) & (saved_xyz <= 220.0), axis=2)
    clean = (
        three_view
        & parity_valid
        & (residual <= args.max_production_residual)
        & in_bounds
    )
    runs = [
        run
        for run in runs
        if bool(clean[run.start_frame - 1, run.fish_index])
        and bool(clean[run.end_frame + 1, run.fish_index])
    ]
    print(f"QC-qualified transition runs: {len(runs):,}", flush=True)
    test = (
        (frame_grid // args.block_frames) % args.block_cycle == args.holdout_block
    )
    fit_sample = clean & (~test) & (frame_grid % max(1, args.fit_stride) == 0)

    weights_payload: dict[str, object] = {
        "definition": (
            "Robust affine delta weights fitted on counterfactually masked clean "
            "three-view rows from non-holdout time blocks."
        ),
        "models": {},
    }
    ablation_rows: list[dict[str, object]] = []
    transition_frame_rows: list[dict[str, object]] = []
    transition_event_output: list[dict[str, object]] = []

    for backend in backends:
        print(f"benchmarking {backend.name} ...", flush=True)
        full_positions = np.full(state.shape + (3,), np.nan, dtype=float)
        full_positions[clean] = backend.reconstruct(coordinates[clean], (0, 1, 2))
        model_corrections: dict[int, AffineShiftCorrection] = {}
        weights_payload["models"][backend.name] = {}

        for missing_view in range(3):
            views = tuple(view for view in range(3) if view != missing_view)
            two_all = np.full(state.shape + (3,), np.nan, dtype=float)
            two_all[clean] = backend.reconstruct(coordinates[clean], views)
            correction = fit_affine_shift_correction(
                two_all[fit_sample],
                full_positions[fit_sample],
                ridge_alpha=args.ridge_alpha,
            )
            model_corrections[missing_view] = correction
            weights_payload["models"][backend.name][CAMERA_NAMES[missing_view]] = (
                correction.to_dict()
            )
            eval_mask = clean & test
            raw_error = np.linalg.norm(
                two_all[eval_mask] - full_positions[eval_mask], axis=1
            )
            corrected_error = np.linalg.norm(
                correction.correct(two_all[eval_mask]) - full_positions[eval_mask],
                axis=1,
            )
            for variant, errors in (
                ("raw", raw_error),
                ("affine_corrected", corrected_error),
            ):
                summary = summarize_errors(errors)
                ablation_rows.append(
                    {
                        "model": backend.name,
                        "missing_view": CAMERA_NAMES[missing_view],
                        "variant": variant,
                        **summary,
                    }
                )

        raw_positions = reconstruct_state_positions(backend, coordinates, state)
        corrected_positions = raw_positions.copy()
        for missing_view, correction in model_corrections.items():
            mask = state == missing_view
            corrected_positions[mask] = correction.correct(raw_positions[mask])
        frame_rows, event_rows = transition_event_rows(
            backend.name,
            identities,
            runs,
            raw_positions,
            corrected_positions,
            clean,
            test[:, 0],
        )
        transition_frame_rows.extend(frame_rows)
        transition_event_output.extend(event_rows)

    ablation = pd.DataFrame(ablation_rows)
    transition_frames = pd.DataFrame(
        transition_frame_rows,
        columns=[
            "event_id", "model", "split", "fish_id", "frame", "missing_view",
            "run_length", "run_offset", "ideal_X", "ideal_Y", "ideal_Z",
            "raw_X", "raw_Y", "raw_Z", "corrected_X", "corrected_Y",
            "corrected_Z", "raw_trajectory_residual_mm",
            "corrected_trajectory_residual_mm", "ideal_step_mm",
        ],
    )
    transition_events = pd.DataFrame(
        transition_event_output,
        columns=[
            "event_id", "model", "split", "fish_id", "missing_view",
            "start_frame", "end_frame", "run_length", "isolated",
            "ideal_step_mm", "raw_peak_step_mm", "corrected_peak_step_mm",
            "raw_peak_step_excess_mm", "corrected_peak_step_excess_mm",
            "raw_peak_trajectory_residual_mm",
            "corrected_peak_trajectory_residual_mm",
            "raw_entry_velocity_discontinuity_mm",
            "corrected_entry_velocity_discontinuity_mm",
            "raw_exit_velocity_discontinuity_mm",
            "corrected_exit_velocity_discontinuity_mm",
        ],
    )
    ablation.to_csv(args.output_dir / "heldout_three_view_ablation.csv", index=False)
    transition_frames.to_csv(args.output_dir / "natural_transition_frames.csv", index=False)
    transition_events.to_csv(args.output_dir / "natural_transition_events.csv", index=False)
    (args.output_dir / "camera_shift_weights.json").write_text(
        json.dumps(weights_payload, indent=2) + "\n", encoding="utf-8"
    )

    natural_summary_rows: list[dict[str, object]] = []
    heldout_events = (
        transition_events.loc[transition_events["split"] == "holdout"]
        if not transition_events.empty
        else transition_events
    )
    for keys, group in heldout_events.groupby(
        ["model", "missing_view", "isolated"], dropna=False
    ):
        model, missing_view, isolated = keys
        for variant in ("raw", "corrected"):
            residual = summarize_errors(
                group[f"{variant}_peak_trajectory_residual_mm"]
            )
            excess = summarize_errors(group[f"{variant}_peak_step_excess_mm"])
            natural_summary_rows.append(
                {
                    "model": model,
                    "missing_view": missing_view,
                    "cohort": "isolated" if isolated else "multi_frame",
                    "variant": variant,
                    "event_n": int(len(group)),
                    "trajectory_residual_median_mm": residual["median"],
                    "trajectory_residual_p95_mm": residual["p95"],
                    "trajectory_residual_p99_mm": residual["p99"],
                    "step_excess_median_mm": excess["median"],
                    "step_excess_p95_mm": excess["p95"],
                    "step_excess_p99_mm": excess["p99"],
                }
            )
    natural_summary = pd.DataFrame(natural_summary_rows)
    natural_summary.to_csv(args.output_dir / "natural_transition_summary.csv", index=False)

    summary = {
        "experiment": args.experiment,
        "frame_count": int(len(coordinates)),
        "fish_ids": identities,
        "clean_three_view_rows": int(clean.sum()),
        "raw_three_view_rows": int(three_view.sum()),
        "dropped_three_view_rows_by_qc": int(three_view.sum() - clean.sum()),
        "bounded_transition_runs_all_lengths": int(len(all_runs)),
        "bounded_transition_runs": int(len(runs)),
        "isolated_transition_runs": int(sum(run.run_length == 1 for run in runs)),
        "heldout_transition_runs": int(
            sum(
                bool(np.all(test[run.start_frame - 1 : run.end_frame + 2, 0]))
                for run in runs
            )
        ),
        "max_transition_run": int(args.max_transition_run),
        "slot_parity": {
            "validated_rows": int(parity_valid.sum()),
            "tolerance_mm": float(args.parity_tolerance_mm),
            "maximum_error_mm": float(np.max(parity_values)),
        },
        "three_view_qc": {
            "max_production_reprojection_sum_squared_px": float(
                args.max_production_residual
            ),
            "xyz_bounds_mm": [-20.0, 220.0],
        },
        "time_block_split": {
            "block_frames": int(args.block_frames),
            "block_cycle": int(args.block_cycle),
            "holdout_block": int(args.holdout_block),
            "fit_stride": int(args.fit_stride),
        },
        "metric_definitions": {
            "trajectory_residual_mm": (
                "Euclidean distance from a two-view result to constant-velocity "
                "interpolation between bracketing three-view anchors."
            ),
            "peak_step_excess_mm": (
                "Largest reconstructed step across the transition minus the "
                "anchor-to-anchor chord length divided by run_length+1."
            ),
            "heldout_ablation_error_mm": (
                "Same-frame Euclidean difference between a masked two-view result "
                "and its full three-view result for the same backend."
            ),
        },
        "interpretation_boundary": (
            "Round7 metrics compare two-view/three-view internal consistency and "
            "trajectory continuity under frozen production assignments. They do not "
            "measure absolute 3D accuracy; held-out rigid-object metrics own that claim."
        ),
        "source_fingerprints": {
            "best_json": file_fingerprint(args.best_json),
            **{
                view: file_fingerprint(args.slp_dir / f"{view}.slp")
                for view in VIEW_NAMES
            },
        },
        "artifacts": {
            "ablation": "heldout_three_view_ablation.csv",
            "transition_frames": "natural_transition_frames.csv",
            "transition_events": "natural_transition_events.csv",
            "transition_summary": "natural_transition_summary.csv",
            "weights": "camera_shift_weights.json",
        },
    }
    (args.output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"outputs: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
