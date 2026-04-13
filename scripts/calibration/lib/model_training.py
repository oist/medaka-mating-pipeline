"""
Model-training helpers for the calibration pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


def _mean_point_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.mean(np.linalg.norm(diff, axis=1)))


def prepare_model_data(
    image_coords: np.ndarray,
    measured_coords: np.ndarray,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Build all calibration training datasets from one aligned point set."""
    image_coords = np.asarray(image_coords, dtype=float)
    measured_coords = np.asarray(measured_coords, dtype=float)
    if image_coords.ndim != 3 or image_coords.shape[0] != 3 or image_coords.shape[2] != 2:
        raise ValueError(f"image_coords must have shape (3, N, 2), got {image_coords.shape}")
    if measured_coords.ndim != 2 or measured_coords.shape[1] != 3:
        raise ValueError(f"measured_coords must have shape (N, 3), got {measured_coords.shape}")
    if image_coords.shape[1] != measured_coords.shape[0]:
        raise ValueError("image_coords and measured_coords must describe the same number of points")

    cam0 = image_coords[0]
    cam1 = image_coords[1]
    cam2 = image_coords[2]
    imcoords_flat = image_coords.swapaxes(0, 1).reshape(measured_coords.shape[0], -1)

    return {
        "imCoords_to_XYZ": (imcoords_flat, measured_coords),
        "xy_yz_to_xz": (np.hstack([cam0, cam2]), cam1),
        "xz_yz_to_xy": (np.hstack([cam1, cam2]), cam0),
        "xz_xy_to_yz": (np.hstack([cam1, cam0]), cam2),
    }


def _make_estimator(
    model_type: str,
    *,
    degree: int = 2,
    alpha: float = 1e-3,
) -> object:
    model_type = model_type.lower()
    if model_type == "polynomial":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
                ("reg", MultiOutputRegressor(Ridge(alpha=alpha))),
            ]
        )
    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            random_state=0,
            min_samples_leaf=1,
        )
    if model_type == "svr":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", MultiOutputRegressor(SVR(C=10.0, epsilon=0.1, kernel="rbf"))),
            ]
        )
    if model_type == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "reg",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        learning_rate_init=1e-3,
                        max_iter=5000,
                        random_state=0,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def _iter_candidates(
    alpha_range: Iterable[float],
    poly_deg_range: Iterable[int],
    model_types: Iterable[str] | None,
):
    chosen_model_types = list(model_types) if model_types is not None else ["polynomial", "random_forest", "svr", "mlp"]
    for model_type in chosen_model_types:
        if model_type == "polynomial":
            for degree in poly_deg_range:
                for alpha in alpha_range:
                    yield {"model_type": model_type, "degree": int(degree), "alpha": float(alpha)}
        else:
            yield {"model_type": model_type}


def _evaluate_candidate(
    X: np.ndarray,
    Y: np.ndarray,
    params: dict,
    *,
    train_fraction: float,
    num_permutations: int,
) -> dict:
    train_errors = []
    test_errors = []
    for seed in range(max(1, int(num_permutations))):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            Y,
            train_size=train_fraction,
            random_state=seed,
        )
        estimator = _make_estimator(**params)
        estimator.fit(X_train, y_train)
        train_pred = estimator.predict(X_train)
        test_pred = estimator.predict(X_test)
        train_errors.append(_mean_point_error(y_train, train_pred))
        test_errors.append(_mean_point_error(y_test, test_pred))

    return {
        **params,
        "cv_mean_error": float(np.mean(test_errors)),
        "cv_std_error": float(np.std(test_errors)),
        "mean_train_error": float(np.mean(train_errors)),
        "std_train_error": float(np.std(train_errors)),
        "mean_test_error": float(np.mean(test_errors)),
        "std_test_error": float(np.std(test_errors)),
    }


def train_calibration_model(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    alpha_range: Iterable[float],
    poly_deg_range: Iterable[int],
    cv_folds: int = 3,
    train_fraction: float = 0.9,
    num_permutations: int = 100,
    model_types: Iterable[str] | None = None,
):
    """
    Search a small model family and return the best fitted estimator.

    `cv_folds` is accepted for CLI compatibility; the current implementation
    uses repeated random train/test splits controlled by `num_permutations`.
    """
    del cv_folds

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    all_models_results = [
        _evaluate_candidate(
            X,
            Y,
            params,
            train_fraction=train_fraction,
            num_permutations=num_permutations,
        )
        for params in _iter_candidates(alpha_range, poly_deg_range, model_types)
    ]
    all_models_results.sort(key=lambda item: item["cv_mean_error"])
    if not all_models_results:
        raise ValueError("No calibration model candidates were evaluated")

    best = all_models_results[0]
    best_params = {
        key: value
        for key, value in best.items()
        if key in {"model_type", "degree", "alpha"}
    }
    model = _make_estimator(**best_params)
    model.fit(X, Y)
    val_results = {
        "mean_train_error": best["mean_train_error"],
        "std_train_error": best["std_train_error"],
        "mean_test_error": best["mean_test_error"],
        "std_test_error": best["std_test_error"],
    }
    return model, best_params, all_models_results, val_results


def save_all_models(trained_models: Dict[str, object], output_dir: str | Path) -> None:
    """Save a dict of trained estimators as joblib files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name, model in trained_models.items():
        dump(model, output_dir / f"{model_name}.joblib")


__all__ = ["prepare_model_data", "train_calibration_model", "save_all_models"]
