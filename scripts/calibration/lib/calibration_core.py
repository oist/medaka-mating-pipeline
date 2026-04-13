"""
Shared calibration pipeline helpers used by the CLI entry points.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

try:
    from .model_training import prepare_model_data, save_all_models, train_calibration_model
    from .visualize_coords_from_template import parse_template
except ImportError:
    from model_training import prepare_model_data, save_all_models, train_calibration_model  # type: ignore
    from visualize_coords_from_template import parse_template  # type: ignore


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize_calib_name(path_like: str | Path) -> str:
    name = Path(path_like).name
    name = re.sub(r"(_pic|_images?)$", "", name, flags=re.IGNORECASE)
    return name


def _find_template_file(calibration_folder: str, way_name: str | None = None) -> Path | None:
    calibration_folder = os.path.normpath(calibration_folder)
    search_dirs = [Path(calibration_folder), Path(calibration_folder).parent]
    way_name = (way_name or "").lower()

    preferred_names = []
    if way_name:
        preferred_names.extend(
            [
                f"cali_coords_template_{way_name}.txt",
                f"Calibration_Coords_Template_{way_name.upper()}.txt",
            ]
        )

    for search_dir in search_dirs:
        for name in preferred_names:
            candidate = search_dir / name
            if candidate.exists():
                return candidate

        patterns = []
        if way_name:
            patterns.extend(
                [
                    f"*coords*template*{way_name}*.txt",
                    f"*{way_name}*template*.txt",
                ]
            )
        patterns.append("*template*.txt")
        for pattern in patterns:
            matches = sorted(search_dir.glob(pattern))
            if matches:
                return matches[0]

    return None


def _find_image_coord_files(
    image_coords_folder: str,
    filename_keyword: str | None = None,
    way_name: str | None = None,
) -> Dict[str, Path]:
    folder = Path(image_coords_folder)
    if not folder.exists():
        raise FileNotFoundError(f"image_coords_folder not found: {folder}")

    keyword = (filename_keyword or way_name or "").lower()
    candidates = sorted(folder.rglob("*_image_coords.txt"))
    if keyword:
        filtered = [path for path in candidates if keyword in path.name.lower()]
        candidates = filtered or candidates

    out: Dict[str, Path] = {}
    for path in candidates:
        name = path.name.lower()
        cam_name = None
        if "cam0" in name or "top" in name:
            cam_name = "cam0"
        elif "cam1" in name or "side1" in name:
            cam_name = "cam1"
        elif "cam2" in name or "side2" in name:
            cam_name = "cam2"
        if cam_name and cam_name not in out:
            out[cam_name] = path

    missing = [cam for cam in ("cam0", "cam1", "cam2") if cam not in out]
    if missing:
        raise FileNotFoundError(
            f"Could not find image coordinate files for {missing} in {folder} (keyword={keyword!r})"
        )
    return out


def _load_image_coords_file(path: str | Path, cam_name: str) -> pd.DataFrame:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            rows.append(
                {
                    "index": int(parts[0]),
                    f"{cam_name}_x": float(parts[1]),
                    f"{cam_name}_y": float(parts[2]),
                    "X": float(parts[3]),
                    "Y": float(parts[4]),
                    "Z": float(parts[5]),
                    "label": " ".join(parts[6:]).strip(),
                }
            )
    if not rows:
        raise ValueError(f"No valid coordinate rows found in {path}")
    return pd.DataFrame(rows).sort_values("index").reset_index(drop=True)


def _flatten_template_sections(
    sections: Dict[str, tuple[np.ndarray, List[str]]],
    *,
    beads_only: bool,
) -> pd.DataFrame:
    rows = []
    point_index = 0

    if "CORNERS" in sections:
        coords, names = sections["CORNERS"]
        for coord, name in zip(coords, names):
            if not beads_only:
                rows.append(
                    {
                        "index": point_index,
                        "X": float(coord[0]),
                        "Y": float(coord[1]),
                        "Z": float(coord[2]),
                        "label": f"CORNER: {name}",
                    }
                )
            point_index += 1

    for section_name in sorted(sections.keys()):
        if not section_name.startswith("BEADS_WAY"):
            continue
        coords, names = sections[section_name]
        for coord, name in zip(coords, names):
            rows.append(
                {
                    "index": point_index,
                    "X": float(coord[0]),
                    "Y": float(coord[1]),
                    "Z": float(coord[2]),
                    "label": f"BEAD: {name}",
                }
            )
            point_index += 1

    if not rows and "POINTS" in sections:
        coords, names = sections["POINTS"]
        for coord, name in zip(coords, names):
            rows.append(
                {
                    "index": point_index,
                    "X": float(coord[0]),
                    "Y": float(coord[1]),
                    "Z": float(coord[2]),
                    "label": name,
                }
            )
            point_index += 1

    return pd.DataFrame(rows)


def load_calibration_data(
    calibration_folder: str,
    image_coords_folder: str,
    filename_keyword: str | None = None,
    way_name: str | None = None,
    beads_only: bool = False,
):
    """
    Load one aligned calibration set.

    Returns:
        `(template_3d, image_coords, point_labels, metadata)`
    """
    cam_files = _find_image_coord_files(image_coords_folder, filename_keyword=filename_keyword, way_name=way_name)
    cam_dfs = {cam: _load_image_coords_file(path, cam) for cam, path in cam_files.items()}

    template_path = _find_template_file(calibration_folder, way_name=way_name)
    if template_path is not None:
        template_df = _flatten_template_sections(parse_template(template_path), beads_only=beads_only)
    else:
        template_df = cam_dfs["cam0"][["index", "X", "Y", "Z", "label"]].copy()
        if beads_only and "label" in template_df.columns:
            template_df = template_df[template_df["label"].str.startswith("BEAD:", na=False)]

    template_indices = set(template_df["index"].tolist())
    extra_rows = cam_dfs["cam0"][["index", "X", "Y", "Z", "label"]].copy()
    extra_rows = extra_rows[~extra_rows["index"].isin(template_indices)]
    if not extra_rows.empty:
        valid_extra = extra_rows.copy()
        for cam in ("cam1", "cam2"):
            valid_extra = valid_extra.merge(
                cam_dfs[cam][["index", "X", "Y", "Z"]],
                on=["index", "X", "Y", "Z"],
                how="inner",
            )
        if not valid_extra.empty:
            template_df = pd.concat(
                [template_df, valid_extra[["index", "X", "Y", "Z", "label"]]],
                ignore_index=True,
            )

    merged = template_df.sort_values("index").reset_index(drop=True)
    for cam in ("cam0", "cam1", "cam2"):
        merged = merged.merge(
            cam_dfs[cam][["index", f"{cam}_x", f"{cam}_y"]],
            on="index",
            how="inner",
        )

    # Always prefer the annotation-file XYZ (from cam0) over template XYZ.
    # The annotation files carry the correct per-way coordinates: corners
    # are in tank frame (assigned by pixel-position matching) and beads are
    # in the per-way cube frame (from the template's BEADS_WAY* section).
    cam0_xyz = cam_dfs["cam0"].set_index("index")[["X", "Y", "Z"]]
    merged_idx = merged.set_index("index")
    merged_idx.update(cam0_xyz)
    merged = merged_idx.reset_index()

    # For even ways (way2, way4), bead XYZ is in the rotated cube frame.
    # Convert beads to tank frame so all points share the same coordinate
    # system.  Corners are already in tank frame and must be skipped.
    import re as _re
    _way_match = _re.search(r"way(\d+)", way_name or "", _re.IGNORECASE)
    _way_num = int(_way_match.group(1)) if _way_match else 1
    if _way_num % 2 == 0:
        for i in range(len(merged)):
            label = merged.iloc[i].get("label", "")
            if isinstance(label, str) and label.startswith(("CORNER:", "CORNER_BOTTOM:")):
                continue  # already in tank frame
            merged.iloc[i, merged.columns.get_loc("X")] = 190.0 - merged.iloc[i]["X"]
            merged.iloc[i, merged.columns.get_loc("Y")] = 190.0 - merged.iloc[i]["Y"]

    if beads_only:
        if "label" in merged.columns and merged["label"].notna().any():
            merged = merged[merged["label"].str.startswith("BEAD:", na=False)]
        else:
            merged = merged[merged["index"] >= 4]

    merged = merged.sort_values("index").reset_index(drop=True)
    if merged.empty:
        raise ValueError("No calibration points remained after alignment/filtering")

    template_3d = merged[["X", "Y", "Z"]].to_numpy(float)
    image_coords = np.stack(
        [
            merged[["cam0_x", "cam0_y"]].to_numpy(float),
            merged[["cam1_x", "cam1_y"]].to_numpy(float),
            merged[["cam2_x", "cam2_y"]].to_numpy(float),
        ],
        axis=0,
    )
    point_labels = merged["label"].fillna("").tolist() if "label" in merged.columns else [""] * len(merged)
    metadata = {
        "template_path": str(template_path) if template_path is not None else None,
        "cam_files": {cam: str(path) for cam, path in cam_files.items()},
        "n_points": int(len(merged)),
    }
    return template_3d, image_coords, point_labels, metadata


def _ensure_project_paths(
    calibration_folder: str,
    output_folder: str | None = None,
    results_root_override: str | None = None,
) -> Dict[str, str]:
    """Infer stable output paths for calibration artifacts."""
    project_root = _project_root()
    calib_name = _normalize_calib_name(calibration_folder)
    model_output_folder = output_folder or str(project_root / "models" / "calibration" / calib_name)
    results_root = results_root_override or str(project_root / "calibration_results" / calib_name)
    return {
        "project_root": str(project_root),
        "calib_name": calib_name,
        "model_output_folder": model_output_folder,
        "results_root": results_root,
    }


def _save_point_mapping_and_figures(
    template_3d: np.ndarray,
    image_coords: np.ndarray,
    pred_3d: np.ndarray,
    results_root: str,
    calib_name: str,
    point_labels: Iterable[str] | None = None,
) -> str:
    """
    Save the point_mapping CSV expected by the calibration QC scripts.
    """
    results_root_path = Path(results_root)
    results_root_path.mkdir(parents=True, exist_ok=True)

    labels = list(point_labels) if point_labels is not None else [""] * len(template_3d)
    df = pd.DataFrame(
        {
            "X": template_3d[:, 0],
            "Y": template_3d[:, 1],
            "Z": template_3d[:, 2],
            "pred_X": pred_3d[:, 0],
            "pred_Y": pred_3d[:, 1],
            "pred_Z": pred_3d[:, 2],
            "cam0_x": image_coords[0, :, 0],
            "cam0_y": image_coords[0, :, 1],
            "cam1_x": image_coords[1, :, 0],
            "cam1_y": image_coords[1, :, 1],
            "cam2_x": image_coords[2, :, 0],
            "cam2_y": image_coords[2, :, 1],
            "label": labels,
        }
    )
    csv_path = results_root_path / f"{calib_name}_point_mapping.csv"
    df.to_csv(csv_path, index=False)

    err = pred_3d - template_3d
    summary = {
        "n_points": int(len(df)),
        "mean_error_mm": float(np.mean(np.linalg.norm(err, axis=1))),
        "rmse_mm": float(np.sqrt(np.mean(np.sum(err ** 2, axis=1)))),
        "max_error_mm": float(np.max(np.linalg.norm(err, axis=1))),
    }
    with (results_root_path / f"{calib_name}_point_mapping_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return str(csv_path)


def _save_training_summary(summary_rows: List[dict], summary_path: str) -> None:
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)


def run_calibration_pipeline(
    *,
    calibration_folder: str,
    alpha_range: Iterable[float],
    poly_deg_range: Iterable[int],
    cv_folds: int,
    train_fraction: float,
    num_permutations: int,
    output_folder: str | None = None,
    image_coords_folder: str | None = None,
    filename_keyword: str | None = None,
    beads_only: bool = False,
    model_types: Iterable[str] | None = None,
    results_root: str | None = None,
) -> Dict[str, str]:
    """Run the calibration training pipeline for one calibration set."""
    image_coords_folder = image_coords_folder or calibration_folder
    template_3d, image_coords, point_labels, metadata = load_calibration_data(
        calibration_folder=calibration_folder,
        image_coords_folder=image_coords_folder,
        filename_keyword=filename_keyword,
        way_name=filename_keyword,
        beads_only=beads_only,
    )

    model_data = prepare_model_data(image_coords, template_3d)
    trained_models = {}
    summary_rows = []
    for model_name, (X_data, Y_data) in model_data.items():
        valid_mask = (~np.isnan(X_data).any(axis=1)) & (~np.isnan(Y_data).any(axis=1))
        X_clean = X_data[valid_mask]
        Y_clean = Y_data[valid_mask]
        if len(X_clean) == 0:
            continue

        model, best_params, all_models_results, val_results = train_calibration_model(
            X_clean,
            Y_clean,
            alpha_range=alpha_range,
            poly_deg_range=poly_deg_range,
            cv_folds=cv_folds,
            train_fraction=train_fraction,
            num_permutations=num_permutations,
            model_types=model_types,
        )
        trained_models[model_name] = model
        best_result = min(all_models_results, key=lambda item: item["cv_mean_error"])
        row = {
            "model_name": model_name,
            "num_samples": int(len(X_clean)),
            "num_features": int(X_clean.shape[1]),
            "output_dim": int(Y_clean.shape[1]),
            "best_model_type": best_params.get("model_type", "polynomial"),
            "cv_mean_error": float(best_result["cv_mean_error"]),
            "cv_std_error": float(best_result["cv_std_error"]),
            "mean_train_error": float(val_results["mean_train_error"]),
            "std_train_error": float(val_results["std_train_error"]),
            "mean_test_error": float(val_results["mean_test_error"]),
            "std_test_error": float(val_results["std_test_error"]),
        }
        if "alpha" in best_params:
            row["best_alpha"] = float(best_params["alpha"])
        if "degree" in best_params:
            row["best_degree"] = int(best_params["degree"])
        summary_rows.append(row)

    paths = _ensure_project_paths(
        calibration_folder,
        output_folder=output_folder,
        results_root_override=results_root,
    )
    save_all_models(trained_models, paths["model_output_folder"])
    summary_path = os.path.join(paths["results_root"], f"{paths['calib_name']}_training_summary.csv")
    _save_training_summary(summary_rows, summary_path)

    mapping_csv = None
    if "imCoords_to_XYZ" in trained_models:
        X_feat = image_coords.swapaxes(0, 1).reshape(template_3d.shape[0], -1)
        pred_3d = np.asarray(trained_models["imCoords_to_XYZ"].predict(X_feat), dtype=float)
        mapping_csv = _save_point_mapping_and_figures(
            template_3d,
            image_coords,
            pred_3d,
            paths["results_root"],
            paths["calib_name"],
            point_labels=point_labels,
        )

    return {
        **paths,
        "summary_path": summary_path,
        "mapping_csv": mapping_csv or "",
        "n_points": str(metadata["n_points"]),
    }


__all__ = [
    "load_calibration_data",
    "_ensure_project_paths",
    "_save_point_mapping_and_figures",
    "_save_training_summary",
    "run_calibration_pipeline",
]
