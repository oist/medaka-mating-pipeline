from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml


DEFAULT_3D_SEED_CONFIG = {
    "num_individuals": 3,
    "min_separation_mm": 20.0,
    "max_speed_mm_per_frame": 8.0,
    "max_residual_px2": 5000.0,
    "max_bind_score_px": 60.0,
    "require_top_bind": True,
    "min_window_frames": 10,
    "max_windows": 25,
    "negative_points_per_other_fish": 1,
}


@dataclass(frozen=True)
class SeedMiningResult:
    windows: pd.DataFrame
    seed_points: pd.DataFrame
    prompts_payload: dict
    summary: dict


def mine_3d_seed_prompts(
    *,
    identified_3d: str | Path | pd.DataFrame,
    best_pre_link: str | Path | Mapping | None,
    calib,
    config: Mapping | None = None,
    start_frame: int | None = None,
) -> SeedMiningResult:
    """Select clean 3D frames and turn their top-view reprojections into SAM2 prompts.

    The selector is intentionally conservative: pseudo-ID seeds must be finite,
    separated in 3D, low residual, low bind error, and not moving implausibly
    fast. This keeps 3D-derived labels useful without feeding crossing/occlusion
    uncertainty back into the segmentation identity model.
    """
    cfg = dict(DEFAULT_3D_SEED_CONFIG)
    cfg.update(dict(config or {}))

    geom = _load_identified_3d(identified_3d)
    pre_link_qc = _load_pre_link_qc(best_pre_link)
    frame_table = _frame_cleanliness_table(geom, pre_link_qc, cfg)
    clean_frames = frame_table[frame_table["clean_seed_frame"]].copy()
    windows = _clean_windows(
        clean_frames,
        min_window_frames=int(cfg["min_window_frames"]),
        max_windows=cfg.get("max_windows"),
    )
    seed_points = _seed_points_for_windows(
        geom,
        windows,
        calib=calib,
        num_individuals=int(cfg["num_individuals"]),
    )
    if start_frame is None and not seed_points.empty:
        start_frame = int(seed_points["prompt_frame"].min())
    prompts_payload = sam2_prompts_from_seed_points(
        seed_points,
        start_frame=int(start_frame or 0),
        negative_points_per_other_fish=int(cfg["negative_points_per_other_fish"]),
    )
    summary = {
        "n_input_rows": int(len(geom)),
        "n_candidate_frames": int(frame_table["frame"].nunique()),
        "n_clean_frames": int(clean_frames["frame"].nunique()),
        "n_windows": int(len(windows)),
        "n_seed_points": int(len(seed_points)),
        "start_frame": int(start_frame or 0),
        "config": _jsonable_config(cfg),
        "reject_reason_counts": _reject_reason_counts(frame_table),
    }
    return SeedMiningResult(
        windows=windows,
        seed_points=seed_points,
        prompts_payload=prompts_payload,
        summary=summary,
    )


def write_seed_mining_outputs(
    result: SeedMiningResult,
    *,
    output_dir: str | Path,
    windows_csv: str = "clean_seed_windows.csv",
    seed_points_csv: str = "clean_seed_points.csv",
    prompts_yaml: str = "sam2_seed_prompts_from_3d.yaml",
    summary_json: str = "sam2_seed_prompts_from_3d_summary.json",
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows_path = out_dir / windows_csv
    seed_points_path = out_dir / seed_points_csv
    prompts_path = out_dir / prompts_yaml
    summary_path = out_dir / summary_json

    result.windows.to_csv(windows_path, index=False)
    result.seed_points.to_csv(seed_points_path, index=False)
    prompts_path.write_text(
        yaml.safe_dump(result.prompts_payload, sort_keys=False),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(result.summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "windows_csv": str(windows_path),
        "seed_points_csv": str(seed_points_path),
        "prompts_yaml": str(prompts_path),
        "summary_json": str(summary_path),
    }


def sam2_prompts_from_seed_points(
    seed_points: pd.DataFrame,
    *,
    start_frame: int,
    negative_points_per_other_fish: int = 1,
) -> dict:
    prompts: list[dict] = []
    if seed_points.empty:
        return {
            "sam2": {
                "runner": "sam2_video_predictor",
                "prompt_mode": "seeded_from_3d",
                "prompts": prompts,
            }
        }

    required = {"window_id", "prompt_frame", "final_fish_id", "obj_id", "top_x", "top_y"}
    missing = required - set(seed_points.columns)
    if missing:
        raise ValueError(f"seed_points missing columns: {', '.join(sorted(missing))}")

    ordered = seed_points.sort_values(["window_id", "obj_id"]).reset_index(drop=True)
    for (_window_id, frame), group in ordered.groupby(["window_id", "prompt_frame"], sort=True):
        point_rows = group.to_dict("records")
        for row in point_rows:
            own_point = [_round_px(row["top_x"]), _round_px(row["top_y"])]
            points = [own_point]
            labels = [1]
            negatives = [
                [_round_px(other["top_x"]), _round_px(other["top_y"])]
                for other in point_rows
                if int(other["obj_id"]) != int(row["obj_id"])
            ][: max(0, int(negative_points_per_other_fish))]
            points.extend(negatives)
            labels.extend([0] * len(negatives))
            prompts.append({
                "obj_id": int(row["obj_id"]),
                "identity": str(row["final_fish_id"]),
                "frame": int(frame) - int(start_frame),
                "absolute_frame": int(frame),
                "points": points,
                "labels": labels,
            })

    return {
        "sam2": {
            "runner": "sam2_video_predictor",
            "prompt_mode": "seeded_from_3d",
            "prompts": prompts,
        }
    }


def _load_identified_3d(path_or_df: str | Path | pd.DataFrame) -> pd.DataFrame:
    df = path_or_df.copy() if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df)
    required = {"frame", "fish_id", "X", "Y", "Z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"identified 3D rows missing columns: {', '.join(sorted(missing))}")
    out = df.copy()
    out["frame"] = out["frame"].astype(int)
    for col in ["X", "Y", "Z", "bind_score"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "final_fish_id" not in out.columns:
        out["final_fish_id"] = out["fish_id"].astype(str)
    out["final_fish_id"] = out["final_fish_id"].fillna("").astype(str)
    return out


def _load_pre_link_qc(best_pre_link: str | Path | Mapping | None) -> pd.DataFrame:
    if best_pre_link is None:
        return pd.DataFrame(columns=["frame", "max_residual_px2", "n_dropped_high_residual"])
    payload = (
        dict(best_pre_link)
        if isinstance(best_pre_link, Mapping)
        else json.loads(Path(best_pre_link).read_text(encoding="utf-8"))
    )
    frames = payload.get("frames", payload)
    rows: list[dict] = []
    for raw_frame, info in frames.items():
        errors = info.get("errors", {}) if isinstance(info, Mapping) else {}
        finite_errors = [
            float(value)
            for value in errors.values()
            if value is not None and np.isfinite(float(value))
        ]
        dropped = info.get("dropped_high_residual", []) if isinstance(info, Mapping) else []
        rows.append({
            "frame": int(raw_frame),
            "max_residual_px2": max(finite_errors) if finite_errors else np.nan,
            "n_dropped_high_residual": int(len(dropped or [])),
        })
    return pd.DataFrame(rows)


def _frame_cleanliness_table(
    geom: pd.DataFrame,
    pre_link_qc: pd.DataFrame,
    cfg: Mapping,
) -> pd.DataFrame:
    work = _add_speed_columns(geom)
    if not pre_link_qc.empty:
        work = work.merge(pre_link_qc, on="frame", how="left")
    else:
        work["max_residual_px2"] = np.nan
        work["n_dropped_high_residual"] = 0
    work["n_dropped_high_residual"] = (
        pd.to_numeric(work["n_dropped_high_residual"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    rows: list[dict] = []
    for frame, group in work.groupby("frame", sort=True):
        reasons: list[str] = []
        labels = group["final_fish_id"].fillna("").astype(str)
        xyz = group[["X", "Y", "Z"]].to_numpy(dtype=float)
        finite_xyz = bool(np.isfinite(xyz).all())
        n_individuals = int(cfg["num_individuals"])
        if len(group) != n_individuals:
            reasons.append("wrong_row_count")
        if labels.eq("").any() or labels.nunique() != len(group):
            reasons.append("missing_or_duplicate_final_fish_id")
        if not finite_xyz:
            reasons.append("nonfinite_xyz")
        if bool(cfg.get("require_top_bind", True)) and "bind_source" in group.columns:
            if not group["bind_source"].fillna("").astype(str).eq("top_cam_2d").all():
                reasons.append("not_all_top_bound")
        if "bind_score" in group.columns:
            bind_scores = pd.to_numeric(group["bind_score"], errors="coerce")
            if bind_scores.isna().any() or float(bind_scores.max()) > float(cfg["max_bind_score_px"]):
                reasons.append("bind_score_high_or_missing")
        max_residual = float(group["max_residual_px2"].max(skipna=True))
        if np.isfinite(max_residual) and max_residual > float(cfg["max_residual_px2"]):
            reasons.append("high_pre_link_residual")
        dropped = int(group["n_dropped_high_residual"].max(skipna=True))
        if dropped > 0:
            reasons.append("dropped_high_residual")
        min_sep = _min_pairwise_distance(xyz) if finite_xyz and len(group) >= 2 else np.nan
        if not np.isfinite(min_sep) or min_sep < float(cfg["min_separation_mm"]):
            reasons.append("fish_too_close_3d")
        max_speed = float(group["speed_mm_per_frame"].max(skipna=True))
        if np.isfinite(max_speed) and max_speed > float(cfg["max_speed_mm_per_frame"]):
            reasons.append("speed_spike")

        rows.append({
            "frame": int(frame),
            "n_rows": int(len(group)),
            "n_final_fish_id": int(labels.nunique()),
            "min_separation_mm": float(min_sep),
            "max_speed_mm_per_frame": max_speed,
            "max_residual_px2": max_residual if np.isfinite(max_residual) else np.nan,
            "max_bind_score_px": float(group["bind_score"].max(skipna=True)) if "bind_score" in group.columns else np.nan,
            "clean_seed_frame": not reasons,
            "reject_reason": ";".join(reasons),
        })
    return pd.DataFrame(rows)


def _add_speed_columns(geom: pd.DataFrame) -> pd.DataFrame:
    out = geom.sort_values(["final_fish_id", "frame"]).copy()
    speeds = pd.Series(np.nan, index=out.index, dtype=float)
    for _label, group in out.groupby("final_fish_id", sort=False):
        xyz = group[["X", "Y", "Z"]].to_numpy(dtype=float)
        frames = group["frame"].to_numpy(dtype=int)
        idxs = group.index.to_list()
        for i in range(len(group)):
            if i == 0:
                speeds.loc[idxs[i]] = 0.0
                continue
            gap = max(1, int(frames[i] - frames[i - 1]))
            if np.isfinite(xyz[i]).all() and np.isfinite(xyz[i - 1]).all():
                speeds.loc[idxs[i]] = float(np.linalg.norm(xyz[i] - xyz[i - 1]) / gap)
    out["speed_mm_per_frame"] = speeds
    return out


def _clean_windows(
    clean_frames: pd.DataFrame,
    *,
    min_window_frames: int,
    max_windows,
) -> pd.DataFrame:
    if clean_frames.empty:
        return pd.DataFrame(columns=[
            "window_id", "frame_start", "frame_end", "n_frames",
            "prompt_frame", "frame_step", "min_separation_mm",
            "max_speed_mm_per_frame", "max_residual_px2",
        ])
    frames = clean_frames["frame"].astype(int).sort_values().to_list()
    frame_step = _infer_frame_step(pd.Series(frames))
    windows: list[list[int]] = []
    current = [frames[0]]
    for frame in frames[1:]:
        if frame == current[-1] + frame_step:
            current.append(frame)
        else:
            windows.append(current)
            current = [frame]
    windows.append(current)

    table = clean_frames.set_index("frame")
    rows: list[dict] = []
    for window_frames in windows:
        if len(window_frames) < int(min_window_frames):
            continue
        prompt_frame = window_frames[len(window_frames) // 2]
        w = table.loc[window_frames]
        rows.append({
            "window_id": len(rows),
            "frame_start": int(window_frames[0]),
            "frame_end": int(window_frames[-1]),
            "n_frames": int(len(window_frames)),
            "prompt_frame": int(prompt_frame),
            "frame_step": int(frame_step),
            "min_separation_mm": float(w["min_separation_mm"].min()),
            "max_speed_mm_per_frame": float(w["max_speed_mm_per_frame"].max()),
            "max_residual_px2": float(w["max_residual_px2"].max(skipna=True)),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if max_windows is not None:
        out = out.sort_values(["n_frames", "frame_start"], ascending=[False, True]).head(int(max_windows))
        out = out.sort_values("frame_start").reset_index(drop=True)
        out["window_id"] = range(len(out))
    return out


def _seed_points_for_windows(
    geom: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    calib,
    num_individuals: int,
) -> pd.DataFrame:
    if windows.empty:
        return pd.DataFrame(columns=[
            "window_id", "frame_start", "frame_end", "prompt_frame",
            "final_fish_id", "obj_id", "X", "Y", "Z", "top_x", "top_y",
        ])
    labels = sorted(str(v) for v in geom["final_fish_id"].dropna().astype(str).unique())
    obj_ids = {label: idx for idx, label in enumerate(labels[:num_individuals])}
    geom_by_frame = {int(frame): group.copy() for frame, group in geom.groupby("frame", sort=False)}
    rows: list[dict] = []
    for window in windows.to_dict("records"):
        frame = int(window["prompt_frame"])
        group = geom_by_frame.get(frame)
        if group is None or group.empty:
            continue
        group = group[group["final_fish_id"].astype(str).isin(obj_ids)].copy()
        group.sort_values("final_fish_id", inplace=True)
        xyz = group[["X", "Y", "Z"]].to_numpy(dtype=float)
        top_xy = calib.project_world_to_view(xyz, "top_cam") if len(group) else np.empty((0, 2))
        for i, row in enumerate(group.to_dict("records")):
            label = str(row["final_fish_id"])
            rows.append({
                "window_id": int(window["window_id"]),
                "frame_start": int(window["frame_start"]),
                "frame_end": int(window["frame_end"]),
                "prompt_frame": frame,
                "final_fish_id": label,
                "obj_id": int(obj_ids[label]),
                "X": float(row["X"]),
                "Y": float(row["Y"]),
                "Z": float(row["Z"]),
                "top_x": float(top_xy[i, 0]),
                "top_y": float(top_xy[i, 1]),
                "min_separation_mm": float(window["min_separation_mm"]),
                "max_speed_mm_per_frame": float(window["max_speed_mm_per_frame"]),
                "max_residual_px2": float(window["max_residual_px2"]),
            })
    return pd.DataFrame(rows)


def _min_pairwise_distance(xyz: np.ndarray) -> float:
    distances: list[float] = []
    for i in range(len(xyz)):
        for j in range(i + 1, len(xyz)):
            distances.append(float(np.linalg.norm(xyz[i] - xyz[j])))
    return min(distances) if distances else np.nan


def _infer_frame_step(frames: pd.Series) -> int:
    unique_frames = pd.Series(frames.astype(int).unique()).sort_values()
    diffs = unique_frames.diff().dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 1
    return int(diffs.value_counts().sort_values(ascending=False).index[0])


def _round_px(value) -> int:
    return int(round(float(value)))


def _jsonable_config(cfg: Mapping) -> dict:
    return {str(k): (v.item() if hasattr(v, "item") else v) for k, v in cfg.items()}


def _reject_reason_counts(frame_table: pd.DataFrame) -> dict:
    counts: dict[str, int] = {}
    if frame_table.empty or "reject_reason" not in frame_table:
        return counts
    for reason_text in frame_table["reject_reason"].fillna("").astype(str):
        for reason in reason_text.split(";"):
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    return counts
