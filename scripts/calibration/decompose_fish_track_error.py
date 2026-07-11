"""Decompose a fish track's 3D precision into calibration and detection terms.

The decomposition uses a real, already-selected multiview fish track:

1. Per-camera 2D localization jitter is estimated from robust second
   differences of continuous per-view tracks. At 166 fps this is a practical
   upper bound because true acceleration also contributes.
2. The 2D jitter covariance is propagated through the local Jacobian of the
   learned 2D-to-3D calibration at the selected fish image coordinates.
3. The independent held-out rigid-object RMS is the calibration term.
4. The two independent terms are combined in quadrature.

This quantifies clean-track precision. Gross cross-view association errors are
reported separately; they are discrete tracking failures, not Gaussian
localization noise and should not be folded into a sub-millimetre error budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi

from scripts.reconstruction_3d.calibration import Calibration


VIEW_NAMES = ("top_cam", "side1_cam", "side2_cam")
VIEW_COLS = ("cam0", "cam1", "cam2")


def _consecutive_second_differences(
    frame: np.ndarray, values: np.ndarray
) -> np.ndarray:
    frame = np.asarray(frame, dtype=int)
    values = np.asarray(values, dtype=float)
    if len(frame) < 3:
        return np.empty((0, values.shape[1]), dtype=float)
    consecutive = (frame[1:-1] - frame[:-2] == 1) & (frame[2:] - frame[1:-1] == 1)
    second = values[2:] - 2.0 * values[1:-1] + values[:-2]
    return second[consecutive]


def estimate_isotropic_2d_jitter(
    tracks: pd.DataFrame, *, score_threshold: float
) -> tuple[float, int, np.ndarray]:
    """Estimate per-coordinate pixel sigma from radial second differences.

    For iid isotropic Gaussian coordinate noise with standard deviation sigma,
    a second difference has coordinate sigma ``sqrt(6)*sigma`` and its 2D norm
    follows a Rayleigh distribution. Dividing the median radius by
    ``sqrt(6) * sqrt(2 log 2)`` recovers sigma. Real acceleration makes this a
    conservative localization-noise estimate.
    """
    required = {"frame", "track_id", "x", "y", "score"}
    missing = sorted(required.difference(tracks.columns))
    if missing:
        raise ValueError(f"track CSV is missing required columns: {missing}")
    pieces = []
    usable = tracks[np.isfinite(tracks["score"]) & (tracks["score"] >= score_threshold)]
    for _, group in usable.groupby("track_id"):
        group = group.sort_values("frame")
        pieces.append(
            _consecutive_second_differences(
                group["frame"].to_numpy(), group[["x", "y"]].to_numpy()
            )
        )
    pieces = [piece for piece in pieces if len(piece)]
    if not pieces:
        raise ValueError("no continuous high-score track segments were available")
    second = np.vstack(pieces)
    median_radius = float(np.median(np.linalg.norm(second, axis=1)))
    sigma = median_radius / (np.sqrt(6.0) * np.sqrt(2.0 * np.log(2.0)))
    return sigma, int(len(second)), second


def _candidate_lookup(tracks: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Reproduce load_2d_slp's score-descending candidate ordering."""
    return {
        int(frame): group.sort_values(
            ["score", "x"], ascending=[False, True], kind="stable"
        ).reset_index(drop=True)
        for frame, group in tracks.groupby("frame")
    }


def extract_selected_triplets(
    best_pre_link_json: Path,
    track_tables: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    payload = json.loads(best_pre_link_json.read_text(encoding="utf-8"))
    frames = payload.get("frames", payload)
    lookups = [_candidate_lookup(table) for table in track_tables]
    rows = []
    for frame_text, info in frames.items():
        frame = int(frame_text)
        groups = info.get("groups", {})
        labels = list(groups) if groups else ["a", "b", "c"]
        permutations = [info.get(f"perm_cam{index}", []) for index in range(3)]
        errors = info.get("errors", {})
        for slot, label in enumerate(labels):
            indices = [
                int(permutation[slot]) if slot < len(permutation) else -1
                for permutation in permutations
            ]
            if min(indices) < 0:
                continue
            if any(frame not in lookup or indices[index] >= len(lookup[frame])
                   for index, lookup in enumerate(lookups)):
                continue
            selected = [lookups[index][frame].iloc[indices[index]] for index in range(3)]
            error = float(errors.get(label, np.nan))
            row: dict[str, object] = {
                "frame": frame,
                "slot": label,
                "registration_error_sum_sq_px": error,
                "registration_rms_per_coordinate_px": (
                    float(np.sqrt(error / 6.0)) if np.isfinite(error) else np.nan
                ),
            }
            for index, detection in enumerate(selected):
                row[f"cam{index}_x"] = float(detection["x"])
                row[f"cam{index}_y"] = float(detection["y"])
                row[f"cam{index}_score"] = float(detection["score"])
                row[f"cam{index}_track_id"] = int(detection["track_id"])
            rows.append(row)
    if not rows:
        raise ValueError("no complete, non-recovered selected triplets were found")
    return pd.DataFrame(rows).sort_values(["frame", "slot"]).reset_index(drop=True)


def finite_difference_jacobians(
    calibration: Calibration,
    image_coordinates: np.ndarray,
    *,
    step_px: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    image_coordinates = np.asarray(image_coordinates, dtype=float).reshape(-1, 6)
    xyz = np.asarray(calibration._model_im2xyz.predict(image_coordinates), dtype=float)
    jacobian = np.empty((len(image_coordinates), 3, 6), dtype=float)
    for coordinate in range(6):
        plus = image_coordinates.copy()
        minus = image_coordinates.copy()
        plus[:, coordinate] += step_px
        minus[:, coordinate] -= step_px
        predicted_plus = np.asarray(calibration._model_im2xyz.predict(plus), dtype=float)
        predicted_minus = np.asarray(calibration._model_im2xyz.predict(minus), dtype=float)
        jacobian[:, :, coordinate] = (predicted_plus - predicted_minus) / (2.0 * step_px)
    return xyz, jacobian


def propagated_radial_sigma(jacobian: np.ndarray, coordinate_sigma_px: np.ndarray) -> np.ndarray:
    coordinate_variance = np.asarray(coordinate_sigma_px, dtype=float) ** 2
    # trace(J diag(sigma^2) J.T) = sum over squared Jacobian elements.
    variance_3d = np.sum(jacobian**2 * coordinate_variance[None, None, :], axis=(1, 2))
    return np.sqrt(variance_3d)


def estimate_observed_3d_high_frequency_sigma(selected: pd.DataFrame) -> dict[str, float | int]:
    pieces = []
    for _, group in selected.groupby("cam0_track_id"):
        group = group.sort_values("frame")
        pieces.append(
            _consecutive_second_differences(
                group["frame"].to_numpy(), group[["X", "Y", "Z"]].to_numpy()
            )
        )
    pieces = [piece for piece in pieces if len(piece)]
    if not pieces:
        return {"n_second_differences": 0, "per_axis_sigma_mm": np.nan, "radial_rms_sigma_mm": np.nan}
    second = np.vstack(pieces)
    median_radius = float(np.median(np.linalg.norm(second, axis=1)))
    per_axis_sigma = median_radius / (np.sqrt(6.0) * float(chi.ppf(0.5, df=3)))
    return {
        "n_second_differences": int(len(second)),
        "median_second_difference_norm_mm": median_radius,
        "per_axis_sigma_mm": float(per_axis_sigma),
        "radial_rms_sigma_mm": float(np.sqrt(3.0) * per_axis_sigma),
    }


def registration_floor_on_cube(
    image_coords_csv: Path,
    heldout_pose_list: Path,
    forward_calibration: Calibration,
    reverse_calibration: Calibration,
) -> np.ndarray:
    coords = pd.read_csv(image_coords_csv)
    coords["pose_id"] = coords["pose_id"].astype(str)
    pose_ids = {
        line.strip()
        for line in heldout_pose_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    coords = coords[coords["pose_id"].isin(pose_ids)]
    columns = [f"cam{camera}_{axis}" for camera in range(3) for axis in ("x", "y")]
    image = coords[columns].to_numpy(dtype=float)
    xyz = forward_calibration._model_im2xyz.predict(image)
    reverse = reverse_calibration._model_xyz2im.predict(xyz)
    return np.sqrt(np.mean((image - reverse) ** 2, axis=1))


def _percentiles(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def plot_error_budget(
    selected: pd.DataFrame,
    jitter_by_view: dict[str, float],
    calibration_rms_mm: float,
    combined_rms_mm: float,
    observed_high_frequency: dict[str, float | int],
    cube_registration: np.ndarray | None,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)

    detection_median = float(np.median(selected["detection_radial_sigma_mm"]))
    axes[0, 0].bar(
        ["Calibration\nheld-out RMS", "Detection\npropagated", "Combined\nquadrature"],
        [calibration_rms_mm, detection_median, combined_rms_mm],
        color=["#D55E00", "#0072B2", "#5B4B8A"],
    )
    axes[0, 0].set_ylabel("3D RMS-equivalent error (mm)")
    axes[0, 0].set_title("Clean-track error budget")
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(
        list(jitter_by_view), list(jitter_by_view.values()),
        color=["#56B4E9", "#009E73", "#009E73"],
    )
    axes[0, 1].set_ylabel("Estimated per-coordinate jitter sigma (px)")
    axes[0, 1].set_title("Robust 2D second-difference estimate")
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[0, 2].hist(
        selected["detection_radial_sigma_mm"], bins=45, color="#0072B2", alpha=0.85
    )
    axes[0, 2].axvline(detection_median, color="black", linestyle="--", label=f"median {detection_median:.3f} mm")
    axes[0, 2].set_xlabel("Propagated detection radial sigma (mm)")
    axes[0, 2].set_ylabel("Selected triplets")
    axes[0, 2].set_title("Detection sensitivity along the fish track")
    axes[0, 2].legend()

    scatter = axes[1, 0].scatter(
        selected["X"], selected["Y"],
        c=selected["detection_radial_sigma_mm"], cmap="viridis", s=7, alpha=0.45,
    )
    axes[1, 0].set_xlim(0, 200)
    axes[1, 0].set_ylim(200, 0)
    axes[1, 0].set_aspect("equal", adjustable="box")
    axes[1, 0].set_xlabel("X (mm)")
    axes[1, 0].set_ylabel("Y (mm)")
    axes[1, 0].set_title("Where detection uncertainty is amplified")
    fig.colorbar(scatter, ax=axes[1, 0], label="3D sigma (mm)")

    fish_registration = selected["registration_rms_per_coordinate_px"].dropna().to_numpy()
    axes[1, 1].hist(
        fish_registration, bins=np.linspace(0, min(20, np.percentile(fish_registration, 99)), 50),
        density=True, histtype="step", linewidth=2.2, color="#0072B2", label="Fish triplets",
    )
    if cube_registration is not None:
        axes[1, 1].hist(
            cube_registration, bins=np.linspace(0, min(20, np.percentile(fish_registration, 99)), 50),
            density=True, histtype="step", linewidth=2.2, color="#D55E00", label="Held-out cube",
        )
    axes[1, 1].set_xlabel("Forward/reverse registration RMS per coordinate (px)")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Reprojection residual includes a large calibration floor")
    axes[1, 1].legend()

    axes[1, 2].axis("off")
    calibration_fraction = calibration_rms_mm**2 / combined_rms_mm**2
    text = (
        f"Calibration held-out RMS:       {calibration_rms_mm:.3f} mm\n"
        f"Detection propagated median:    {detection_median:.3f} mm\n"
        f"Detection propagated P90:       {np.percentile(selected['detection_radial_sigma_mm'], 90):.3f} mm\n"
        f"Combined clean-track RMS:       {combined_rms_mm:.3f} mm\n"
        f"Calibration share of variance:  {100*calibration_fraction:.1f}%\n\n"
        f"Observed high-frequency radial: {float(observed_high_frequency['radial_rms_sigma_mm']):.3f} mm\n\n"
        "Interpretation:\n"
        "- Calibration limits sub-mm precision on clean associations.\n"
        "- Candidate/identity failures dominate multi-mm tail errors.\n"
        "- Reprojection RMS is not detector sigma: even accurate cube\n"
        "  centroids have multi-pixel forward/reverse inconsistency."
    )
    axes[1, 2].text(
        0.03, 0.97, text, va="top", family="monospace", fontsize=11,
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#F6F4EF", "edgecolor": "#C7C0B2"},
    )

    fig.suptitle(
        "Fish-track 3D error decomposition: calibration vs detection localization\n"
        "Real selected triplets; detection estimate is conservative because true acceleration remains",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-pre-link-json", type=Path, required=True)
    parser.add_argument(
        "--track-csvs", type=Path, nargs=3, required=True,
        metavar=("TOP", "SIDE1", "SIDE2"),
    )
    parser.add_argument("--calibration-model-folder", type=Path, required=True)
    parser.add_argument("--calibration-eval-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.8)
    parser.add_argument("--jacobian-step-px", type=float, default=0.25)
    parser.add_argument("--jacobian-stride", type=int, default=1)
    parser.add_argument("--cube-image-coords-csv", type=Path, default=None)
    parser.add_argument("--cube-heldout-pose-list", type=Path, default=None)
    parser.add_argument("--reverse-model-folder", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    track_tables = [pd.read_csv(path) for path in args.track_csvs]
    selected = extract_selected_triplets(args.best_pre_link_json, track_tables)
    selected = selected[
        np.all(np.isfinite(selected[[f"cam{i}_{axis}" for i in range(3) for axis in ("x", "y")]]), axis=1)
    ].copy()

    jitter_by_view: dict[str, float] = {}
    jitter_details: dict[str, dict[str, float | int]] = {}
    coordinate_sigma = []
    for view_name, table in zip(VIEW_NAMES, track_tables, strict=True):
        sigma, count, second = estimate_isotropic_2d_jitter(
            table, score_threshold=args.score_threshold
        )
        jitter_by_view[view_name] = sigma
        coordinate_sigma.extend([sigma, sigma])
        jitter_details[view_name] = {
            "per_coordinate_sigma_px": sigma,
            "n_second_differences": count,
            "median_second_difference_norm_px": float(np.median(np.linalg.norm(second, axis=1))),
        }

    calibration = Calibration(args.calibration_model_folder)
    image_columns = [f"cam{i}_{axis}" for i in range(3) for axis in ("x", "y")]
    sampled = selected.iloc[::max(1, args.jacobian_stride)].copy()
    xyz, jacobian = finite_difference_jacobians(
        calibration,
        sampled[image_columns].to_numpy(),
        step_px=args.jacobian_step_px,
    )
    sampled[["X", "Y", "Z"]] = xyz
    sampled["detection_radial_sigma_mm"] = propagated_radial_sigma(
        jacobian, np.asarray(coordinate_sigma)
    )
    sampled["isotropic_noise_gain_radial_mm_per_px"] = np.sqrt(
        np.sum(jacobian**2, axis=(1, 2))
    )

    calibration_evaluation = json.loads(args.calibration_eval_json.read_text(encoding="utf-8"))
    calibration_rms = float(calibration_evaluation["kabsch_resid_mm"]["rms"])
    detection_median = float(np.median(sampled["detection_radial_sigma_mm"]))
    combined_rms = float(np.sqrt(calibration_rms**2 + detection_median**2))
    calibration_variance_fraction = float(calibration_rms**2 / combined_rms**2)

    # Use every selected triplet for the observed 3D high-frequency check.
    all_xyz = calibration._model_im2xyz.predict(selected[image_columns].to_numpy())
    selected[["X", "Y", "Z"]] = all_xyz
    observed_high_frequency = estimate_observed_3d_high_frequency_sigma(selected)

    cube_registration = None
    if args.cube_image_coords_csv and args.cube_heldout_pose_list and args.reverse_model_folder:
        reverse_calibration = Calibration(args.reverse_model_folder)
        cube_registration = registration_floor_on_cube(
            args.cube_image_coords_csv,
            args.cube_heldout_pose_list,
            calibration,
            reverse_calibration,
        )

    selected_path = args.output_dir / "selected_fish_triplets.csv"
    sampled_path = args.output_dir / "fish_detection_propagation_samples.csv"
    selected.to_csv(selected_path, index=False)
    sampled.to_csv(sampled_path, index=False)
    figure_path = args.output_dir / "fish_error_budget.png"
    plot_error_budget(
        sampled,
        jitter_by_view,
        calibration_rms,
        combined_rms,
        observed_high_frequency,
        cube_registration,
        figure_path,
    )

    summary = {
        "method": {
            "calibration_component": (
                "independent held-out rigid-object Kabsch RMS; validates shape/scale and is a "
                "lower bound on absolute tank-coordinate calibration error"
            ),
            "detection_component": (
                "robust per-view 2D second-difference jitter propagated through the local "
                "2D-to-3D model Jacobian; conservative because true acceleration remains"
            ),
            "combination": "root-sum-square of calibration RMS and median propagated detection sigma",
        },
        "n_complete_selected_triplets": int(len(selected)),
        "n_jacobian_samples": int(len(sampled)),
        "localization_jitter": jitter_details,
        "calibration_heldout_rms_mm": calibration_rms,
        "propagated_detection_radial_sigma_mm": _percentiles(
            sampled["detection_radial_sigma_mm"].to_numpy()
        ),
        "isotropic_noise_gain_radial_mm_per_px": _percentiles(
            sampled["isotropic_noise_gain_radial_mm_per_px"].to_numpy()
        ),
        "combined_clean_track_rms_mm": combined_rms,
        "variance_fraction": {
            "calibration": calibration_variance_fraction,
            "detection_localization": 1.0 - calibration_variance_fraction,
        },
        "observed_3d_high_frequency": observed_high_frequency,
        "fish_registration_rms_per_coordinate_px": _percentiles(
            selected["registration_rms_per_coordinate_px"].dropna().to_numpy()
        ),
        "heldout_cube_forward_reverse_rms_per_coordinate_px": (
            _percentiles(cube_registration) if cube_registration is not None else None
        ),
        "important_scope": (
            "This is the clean-association relative-precision budget. The rigid object's unknown "
            "absolute pose means the calibration term validates shape/scale, not absolute tank "
            "localization. Multi-mm candidate/identity swaps are discrete tracking failures and "
            "are not included in the Gaussian decomposition."
        ),
        "outputs": {
            "figure": str(figure_path),
            "selected_triplets": str(selected_path),
            "propagation_samples": str(sampled_path),
        },
    }
    summary_path = args.output_dir / "fish_error_budget_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
