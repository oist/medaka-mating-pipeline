from __future__ import annotations

"""
Reprojection-NN bind: attach top-camera 2D track IDs to the 3D tracks produced
by `reconstruct_from_slp` (geometry-only IDs).

Per frame:
  1. Project each 3D point onto the top_cam image plane via
     `Calibration.project_world_to_view`.
  2. Hungarian-match the projected points against the top_cam tracks.csv rows
     for that frame, gated by `--bind-distance` pixels.
  3. Geometry slots with no top match (top occluded, top track filtered out by
     --require-min-frames, or projection outside the gate) get
     `source_top_id = NaN`, `bind_source = "geometry_only"` and KEEP the
     geometry stream's own 3D-temporal-link `fish_id`.

After the per-frame pass, each geometry `fish_id` is mapped to its modal
`source_top_id` over the full run and written to `final_fish_id` so that
downstream behavioral analysis sees a stable identity.

Output columns include frame, fish_id (geometry), X, Y, Z, source_top_id,
source_top_label, source_identity_trajectory_id, bind_score, bind_source, and
final_fish_id.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.local_paths import choose_path, load_local_paths
from scripts.reconstruction_3d.calibration import Calibration


def bind(
    exp_3d_csv: Path,
    top_tracks_csv: Path,
    *,
    calib: Calibration,
    out_csv: Path,
    bind_distance: float = 60.0,
    require_min_frames: int = 500,
) -> pd.DataFrame:
    calib.require_capability("projection")
    geom = pd.read_csv(exp_3d_csv)
    top = pd.read_csv(top_tracks_csv)

    required_geom = {"frame", "fish_id", "X", "Y", "Z"}
    required_top = {"frame", "track_id", "x", "y"}
    missing_g = required_geom - set(geom.columns)
    missing_t = required_top - set(top.columns)
    if missing_g:
        raise ValueError(f"{exp_3d_csv} missing columns: {missing_g}")
    if missing_t:
        raise ValueError(f"{top_tracks_csv} missing columns: {missing_t}")

    # The forward projector is vectorized. Project the geometry table in large
    # chunks once, rather than paying Python/solver overhead once per frame.
    geometry_xyz = geom[["X", "Y", "Z"]].to_numpy(dtype=float)
    geometry_valid = np.isfinite(geometry_xyz).all(axis=1)
    geometry_top_px = np.full((len(geom), 2), np.nan, dtype=float)
    valid_rows = np.flatnonzero(geometry_valid)
    projection_chunk_size = 100_000
    for start in range(0, len(valid_rows), projection_chunk_size):
        rows_chunk = valid_rows[start : start + projection_chunk_size]
        projected = np.asarray(calib.project_world(geometry_xyz[rows_chunk]), dtype=float)
        geometry_top_px[rows_chunk] = projected[:, 0, :]
    geom = geom.copy()
    geom["_projected_top_x"] = geometry_top_px[:, 0]
    geom["_projected_top_y"] = geometry_top_px[:, 1]

    top = top.copy()
    top["bind_identity_key"] = _bind_identity_keys(top)

    sizes = top.groupby("bind_identity_key").size()
    keep = sizes[sizes >= require_min_frames].index
    n_dropped = len(sizes) - len(keep)
    if n_dropped:
        print(f"[bind] dropping {n_dropped} top identity keys below {require_min_frames} frames")
    top = top[top["bind_identity_key"].isin(keep)].copy()

    top_by_frame = {int(fi): g for fi, g in top.groupby("frame")}

    rows: list[dict] = []
    for fi, gf in geom.groupby("frame"):
        gf = gf.reset_index(drop=True)
        proj = gf[["_projected_top_x", "_projected_top_y"]].to_numpy(dtype=float)
        projection_valid_mask = np.isfinite(proj).all(axis=1)

        tf = top_by_frame.get(int(fi))
        if tf is None or len(tf) == 0:
            for i in range(len(gf)):
                r = gf.iloc[i]
                rows.append({
                    "frame": int(fi),
                    "fish_id": r["fish_id"],
                    "X": float(r["X"]), "Y": float(r["Y"]), "Z": float(r["Z"]),
                    "source_top_id": np.nan,
                    "source_top_label": "",
                    "source_identity_trajectory_id": "",
                    "source_bind_key": "",
                    "source_qc_status": "",
                    "source_bind_weight": 0.0,
                    "bind_score": np.nan,
                    "bind_source": "geometry_only",
                })
            continue

        top_xy = tf[["x", "y"]].to_numpy(dtype=float)
        top_ids = tf["track_id"].to_numpy()
        top_labels = _optional_column(tf, "track_label", default="")
        top_trajectory_ids = _optional_column(tf, "identity_trajectory_id", default="")
        top_qc_status = _optional_column(tf, "qc_status", default="")
        top_weights = _numeric_optional_column(tf, "bind_weight", default=1.0)
        top_bind_keys = _optional_column(tf, "bind_identity_key", default="")

        cost = np.full((len(gf), len(tf)), 1e9, dtype=float)
        if projection_valid_mask.any():
            cost[projection_valid_mask] = cdist(
                proj[projection_valid_mask], top_xy
            )
        cost_for_lap = cost.copy()
        cost_for_lap[cost_for_lap > bind_distance] = 1e9
        row_ind, col_ind = linear_sum_assignment(cost_for_lap)

        used_geom: set[int] = set()
        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] > bind_distance:
                continue
            r = gf.iloc[ri]
            rows.append({
                "frame": int(fi),
                "fish_id": r["fish_id"],
                "X": float(r["X"]), "Y": float(r["Y"]), "Z": float(r["Z"]),
                "source_top_id": int(top_ids[ci]),
                "source_top_label": str(top_labels[ci]),
                "source_identity_trajectory_id": str(top_trajectory_ids[ci]),
                "source_bind_key": str(top_bind_keys[ci]),
                "source_qc_status": str(top_qc_status[ci]),
                "source_bind_weight": float(top_weights[ci]),
                "bind_score": float(cost[ri, ci]),
                "bind_source": "top_cam_2d",
            })
            used_geom.add(int(ri))

        for i in range(len(gf)):
            if i in used_geom:
                continue
            r = gf.iloc[i]
            rows.append({
                "frame": int(fi),
                "fish_id": r["fish_id"],
                "X": float(r["X"]), "Y": float(r["Y"]), "Z": float(r["Z"]),
                "source_top_id": np.nan,
                "source_top_label": "",
                "source_identity_trajectory_id": "",
                "source_bind_key": "",
                "source_qc_status": "",
                "source_bind_weight": 0.0,
                "bind_score": np.nan,
                "bind_source": "geometry_only",
            })

    df = pd.DataFrame(rows).sort_values(["frame", "fish_id"]).reset_index(drop=True)

    # Global Hungarian assignment on the (geometry fish_id) x stable top identity
    # key tally matrix. This prevents two geometry IDs from collapsing onto one
    # top identity and lets short segmentation fragments with the same
    # biological label vote together.
    bound_only = df.dropna(subset=["source_top_id"])
    if not bound_only.empty:
        if "source_bind_weight" not in bound_only.columns:
            bound_only = bound_only.copy()
            bound_only["source_bind_weight"] = 1.0
        tally = bound_only.pivot_table(
            index="fish_id", columns="source_bind_key",
            values="source_bind_weight", aggfunc="sum", fill_value=0,
        )
        geom_ids = tally.index.tolist()
        top_keys_unique = tally.columns.tolist()
        cost = -tally.to_numpy(dtype=float)  # maximize tally => minimize negative
        if cost.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost)
            mapping: dict = {}
            for ri, ci in zip(row_ind, col_ind):
                if -cost[ri, ci] > 0:  # only accept positive-tally pairs
                    mapping[geom_ids[ri]] = str(top_keys_unique[ci])
        else:
            mapping = {}
    else:
        mapping = {}
        geom_ids = []

    label_by_key = _top_label_lookup_by_key(bound_only) if not bound_only.empty else {}
    print(f"[bind] global 1-to-1 assignment geometry fish_id -> top identity key:")
    for fid in df["fish_id"].unique():
        key = mapping.get(fid)
        label = label_by_key.get(key, "") if key is not None else ""
        suffix = f" ({label})" if label and label != key else ""
        print(f"  {fid}: {key if key is not None else 'unbound (fallback to geometry id)'}{suffix}")

    fallback_str = df["fish_id"].astype(str)
    label_mapping = {fid: (label_by_key.get(key) or str(key)) for fid, key in mapping.items()}
    df["final_fish_id"] = df["fish_id"].map(label_mapping)
    df["final_fish_id"] = df["final_fish_id"].fillna(fallback_str)

    bound = int((df["bind_source"] == "top_cam_2d").sum())
    total = len(df)
    print(f"[bind] {bound}/{total} ({100*bound/max(total,1):.1f}%) rows bound via top_cam_2d")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    projection_provenance_path = out_csv.with_name(
        f"{out_csv.stem}_projection_provenance.json"
    )
    projection_provenance = {
        "schema_version": 1,
        "projection_backend": calib.backend_name,
        "camera": "cam0",
        "view": "top_cam",
        "camera_order": ["cam0", "cam1", "cam2"],
        "projection_unit": "px",
        "bind_gate": {"value": float(bind_distance), "unit": "px"},
        "calibration_provenance": calib.provenance,
        "legacy_projection_fallback_configured": False,
        "silent_cross_backend_fallback": False,
        "projection_chunk_size": projection_chunk_size,
    }
    projection_provenance_path.write_text(
        json.dumps(projection_provenance, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    df.attrs["projection_provenance_path"] = str(projection_provenance_path)
    print(f"[bind] wrote {out_csv}")
    print(f"[bind] projection provenance: {projection_provenance_path}")
    return df


def _optional_column(df: pd.DataFrame, name: str, *, default: str) -> np.ndarray:
    if name not in df.columns:
        return np.asarray([default] * len(df), dtype=object)
    return df[name].fillna(default).astype(str).to_numpy()


def _numeric_optional_column(df: pd.DataFrame, name: str, *, default: float) -> np.ndarray:
    if name not in df.columns:
        return np.full((len(df),), float(default), dtype=float)
    return pd.to_numeric(df[name], errors="coerce").fillna(float(default)).to_numpy(dtype=float)


def _bind_identity_keys(df: pd.DataFrame) -> pd.Series:
    candidates = []
    for col in ("identity", "track_label", "assigned_identity"):
        if col in df.columns:
            value = df[col].fillna("").astype(str).str.strip()
            value = value.mask(value.isin(["", "unknown", "unassigned", "nan", "None"]))
            candidates.append(value)
    if candidates:
        key = candidates[0]
        for value in candidates[1:]:
            key = key.fillna(value)
        key = key.fillna(df["track_id"].astype(str))
    else:
        key = df["track_id"].astype(str)
    return key.astype(str)


def _top_label_lookup(bound_only: pd.DataFrame) -> dict:
    if "source_top_label" not in bound_only.columns:
        return {}
    lookup: dict = {}
    for top_id, group in bound_only.groupby("source_top_id"):
        if pd.isna(top_id):
            continue
        labels = group["source_top_label"].fillna("").astype(str)
        labels = labels[labels != ""]
        if not labels.empty:
            lookup[int(top_id)] = labels.mode().iloc[0]
    return lookup


def _top_label_lookup_by_key(bound_only: pd.DataFrame) -> dict:
    if "source_bind_key" not in bound_only.columns:
        return {}
    lookup: dict = {}
    for key, group in bound_only.groupby("source_bind_key"):
        if pd.isna(key) or str(key) == "":
            continue
        labels = group.get("source_top_label", pd.Series([], dtype=str)).fillna("").astype(str)
        labels = labels[labels != ""]
        lookup[str(key)] = labels.mode().iloc[0] if not labels.empty else str(key)
    return lookup


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bind top-cam 2D track IDs to a 3D geometry CSV via reprojection."
    )
    p.add_argument("--exp-3d-csv", required=True, type=Path)
    p.add_argument("--top-tracks-csv", required=True, type=Path)
    p.add_argument("--out-csv", required=True, type=Path)
    p.add_argument("--bind-distance", type=float, default=60.0,
                   help="Max projection-NN cost in px to accept a bind (default 60).")
    p.add_argument("--require-min-frames", type=int, default=500,
                   help="Drop top tracks with fewer frames than this from the anchor pool.")
    p.add_argument("--paths-config", default=None)
    p.add_argument("--calibration-model-dir", default=None,
                   help="Calibration model folder; defaults to paths.local.toml entry.")
    p.add_argument(
        "--calibration-backend",
        choices=("auto", "refractive", "polynomial"),
        default="auto",
        help=(
            "Projection backend policy. Auto prefers valid refractive assets "
            "and otherwise recognizes legacy polynomial/joblib folders."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    local = load_local_paths(args.paths_config)
    model_dir = choose_path(
        args.calibration_model_dir,
        local.calibration_model_dir if local else None,
    )
    if model_dir is None:
        raise SystemExit("calibration_model_dir required (paths.local.toml or --calibration-model-dir)")
    calib = Calibration(model_dir, backend=args.calibration_backend)
    print(
        f"[calibration] projection backend={calib.backend_name} unit=px "
        f"provenance={json.dumps(calib.provenance, default=str, sort_keys=True)}"
    )
    bind(
        exp_3d_csv=args.exp_3d_csv,
        top_tracks_csv=args.top_tracks_csv,
        calib=calib,
        out_csv=args.out_csv,
        bind_distance=args.bind_distance,
        require_min_frames=args.require_min_frames,
    )


if __name__ == "__main__":
    main()
