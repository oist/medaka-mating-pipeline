from __future__ import annotations

"""Validate fish-derived missing-camera shift weights on held-out rigid poses."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.calibration.benchmark_view_transition_shifts import (
    AffineShiftCorrection,
    CAMERA_NAMES,
    ProductionPolynomialBackend,
    RayBackend,
    load_dlt_cameras,
    load_refractive_cameras,
)
from scripts.calibration.fit_refractive_rigid_object import (
    _geometry_lookup,
    _heldout_coordinates,
    evaluate_predictor,
    summarize_model,
)


def load_correction(payload: dict[str, object]) -> AffineShiftCorrection:
    return AffineShiftCorrection(
        feature_mean=np.asarray(payload["feature_mean"], dtype=float),
        feature_scale=np.asarray(payload["feature_scale"], dtype=float),
        coefficients=np.asarray(payload["coefficients"], dtype=float),
        ridge_alpha=float(payload["ridge_alpha"]),
        huber_iterations=int(payload["huber_iterations"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-coordinates", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--heldout-pose-list", type=Path, required=True)
    parser.add_argument("--production-model-dir", type=Path, required=True)
    parser.add_argument("--dlt-model-dir", type=Path, required=True)
    parser.add_argument("--refractive-camera-dir", type=Path, required=True)
    parser.add_argument("--weights-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heldout = _heldout_coordinates(args.image_coordinates, args.heldout_pose_list)
    geometry = _geometry_lookup(args.geometry)
    weight_payload = json.loads(args.weights_json.read_text(encoding="utf-8"))["models"]

    backends = [
        ProductionPolynomialBackend(args.production_model_dir),
        RayBackend("projective_dlt", load_dlt_cameras(args.dlt_model_dir)),
        RayBackend("flat_refractive", load_refractive_cameras(args.refractive_camera_dir)),
    ]
    marker_parts = []
    pair_parts = []
    summary_rows = []
    summary_payload: dict[str, object] = {
        "scope": (
            "13 untouched rigid-object poses. Kabsch residual validates shape; "
            "pairwise error validates scale without pose fitting."
        ),
        "models": {},
    }

    for backend in backends:
        variants: list[tuple[str, tuple[int, ...], AffineShiftCorrection | None]] = [
            ("three_view", (0, 1, 2), None)
        ]
        for missing_view in range(3):
            correction = load_correction(
                weight_payload[backend.name][CAMERA_NAMES[missing_view]]
            )
            views = tuple(view for view in range(3) if view != missing_view)
            variants.extend(
                [
                    (f"missing_{CAMERA_NAMES[missing_view]}_raw", views, None),
                    (
                        f"missing_{CAMERA_NAMES[missing_view]}_affine_corrected",
                        views,
                        correction,
                    ),
                ]
            )

        summary_payload["models"][backend.name] = {}
        for variant, views, correction in variants:
            model_name = f"{backend.name}:{variant}"

            def predictor(image_points, *, _views=views, _correction=correction):
                point = backend.reconstruct(
                    np.asarray(image_points, dtype=float).reshape(1, 3, 2), _views
                )[0]
                if _correction is not None:
                    point = _correction.correct(point.reshape(1, 3))[0]
                return point, np.nan

            markers, pairs = evaluate_predictor(
                heldout, geometry, predictor, model_name=model_name
            )
            marker_parts.append(markers)
            pair_parts.append(pairs)
            summary = summarize_model(markers, pairs)
            summary_payload["models"][backend.name][variant] = summary
            missing_view = "none"
            correction_variant = "three_view"
            if variant.startswith("missing_"):
                missing_view = variant.split("_", 2)[1]
                correction_variant = (
                    "affine_corrected" if variant.endswith("affine_corrected") else "raw"
                )
            summary_rows.append(
                {
                    "model": backend.name,
                    "missing_view": missing_view,
                    "variant": correction_variant,
                    "n_poses": summary["n_poses"],
                    "n_markers": summary["n_markers"],
                    "kabsch_rms_mm": summary["kabsch_residual_mm"]["rms"],
                    "kabsch_p95_mm": summary["kabsch_residual_mm"]["p95"],
                    "pairwise_mae_mm": summary["pairwise_distance_error_mm"]["mae"],
                    "pairwise_p95_mm": summary["pairwise_distance_error_mm"]["p95"],
                    "signed_scale_bias_mm": summary["pairwise_distance_error_mm"][
                        "signed_bias"
                    ],
                }
            )

    pd.concat(marker_parts, ignore_index=True).to_csv(
        args.output_dir / "heldout_rigid_marker_errors.csv", index=False
    )
    pd.concat(pair_parts, ignore_index=True).to_csv(
        args.output_dir / "heldout_rigid_pairwise_errors.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "heldout_rigid_summary.csv", index=False
    )
    (args.output_dir / "heldout_rigid_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
