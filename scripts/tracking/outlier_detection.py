"""
2D 轨迹异常值检测：找到明显不可能的跳变，并将其置为 NaN。

设计目标
--------
- **简单**：只实现一种、可解释的规则（按帧间隔缩放的最大速度），避免过度复杂的多种算法和迭代逻辑。
- **分工清晰**：本模块只做「标记异常 → 置 NaN」，不做插值和平滑；
  插值和平滑由 `missing_value_filling.py` 负责。
- **可审计**：可选地在指定目录下输出一份 CSV，列出每个异常点所在的 player / fish_id / frame / 原始坐标。

核心规则
--------
对同一条轨迹（同一 `fish_id`）按 frame 排序后：

- 计算「上一有效点 → 当前点」的位移距离 `dist`；
- 计算帧间隔 `gap = frame_curr - frame_prev`；
- 若 `dist > max_jump_per_frame * gap`，则认为 **当前点不可能是同一条鱼**，
  将该点标记为异常（x,y 置为 NaN），供后续插值阶段补洞。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Hashable, List

import numpy as np
import pandas as pd


def detect_outliers_frame_gap_scaled(
    positions: np.ndarray,
    frame_indices: np.ndarray,
    max_jump_per_frame: float = 100.0,
) -> np.ndarray:
    """
    按帧间隔缩放的异常值检测。

    对同一条轨迹，若「上一有效点 → 当前点」的位移距离 dist 满足
        dist > max_jump_per_frame * gap
    （gap 为两点间的帧数差），则认为当前点是异常点。

    Args:
        positions: (n_points, 2) 位置数组，按 frame 排序；允许包含 NaN。
        frame_indices: (n_points,) 对应帧号，已按升序排列。
        max_jump_per_frame: 每帧允许的最大位移（像素），默认 100。

    Returns:
        outlier_mask: (n_points,) 布尔数组，True 表示该行应视为异常点。
    """
    n = len(positions)
    if n < 2:
        return np.zeros(n, dtype=bool)

    outlier_mask = np.zeros(n, dtype=bool)
    frame_indices = np.asarray(frame_indices, dtype=int)

    # 只在「前后都有有效坐标」的点上做检测
    valid = ~(np.isnan(positions[:, 0]) | np.isnan(positions[:, 1]))
    if valid.sum() < 2:
        return outlier_mask

    valid_idx = np.where(valid)[0]
    prev_i = valid_idx[0]
    prev_pos = positions[prev_i]
    prev_frame = frame_indices[prev_i]

    for curr_i in valid_idx[1:]:
        curr_pos = positions[curr_i]
        curr_frame = frame_indices[curr_i]
        gap = curr_frame - prev_frame
        if gap > 0:
            dist = float(np.linalg.norm(curr_pos - prev_pos))
            threshold = max_jump_per_frame * gap
            if dist > threshold:
                outlier_mask[curr_i] = True
            else:
                # 只有在当前点“合理”时，才更新上一有效点
                prev_i = curr_i
                prev_pos = curr_pos
                prev_frame = curr_frame
        else:
            # 非递增帧号直接跳过，不更新基准
            continue

    return outlier_mask


def detect_loop_segment_outliers(
    positions: np.ndarray,
    frame_indices: np.ndarray,
    *,
    max_loop_frames: int = 10,
    return_dist_thresh: float = 30.0,
    excursion_dist_thresh: float = 200.0,
) -> np.ndarray:
    """
    检测「短时间内离开又回到原地」但中间整段明显跑到远处的异常片段。

    适用于你截图里这种：423 帧和 429 帧几乎在同一位置，中间 424–428 却整段在另一个地方。

    逻辑：
    - 在同一条轨迹上找一对端点 (t0, t1)，要求：
      - 帧间隔 <= max_loop_frames；
      - 端点距离 <= return_dist_thresh（回到原地）；
    - 若在 (t0, t1) 之间的所有中间点都远离两端点（> excursion_dist_thresh），
      则认为这一段中间点是「错误段」，标为 outlier。
    """
    n = len(positions)
    out = np.zeros(n, dtype=bool)
    if n < 3:
        return out

    frames = np.asarray(frame_indices, dtype=int)
    valid = ~(np.isnan(positions[:, 0]) | np.isnan(positions[:, 1]))
    idx = np.where(valid)[0]
    if idx.size < 3:
        return out

    for ii, i0 in enumerate(idx):
        p0 = positions[i0]
        f0 = frames[i0]
        # 只考虑向前、且帧间隔不超过 max_loop_frames 的端点
        for jj in range(ii + 2, idx.size):
            i1 = idx[jj]
            f1 = frames[i1]
            if f1 - f0 > max_loop_frames:
                break
            p1 = positions[i1]
            # 端点距离要足够近，认为是「回到原地」
            if np.linalg.norm(p1 - p0) > return_dist_thresh:
                continue
            mid_idx = idx[ii + 1 : jj]
            if mid_idx.size == 0:
                continue
            mid_pos = positions[mid_idx]
            # 中间点到任一端点的最小距离都要「很远」才认为是错误段
            dist_to_ends = np.minimum(
                np.linalg.norm(mid_pos - p0, axis=1),
                np.linalg.norm(mid_pos - p1, axis=1),
            )
            if np.all(dist_to_ends > excursion_dist_thresh):
                out[mid_idx] = True

    return out


def _is_mirror_pair(
    x1: float, y1: float,
    x2: float, y2: float,
    y_tol: float = 20.0,
    sym_tol: float = 25.0,
) -> bool:
    """
    判断两点是否呈「镜像」关系：y 相近、x 关于中点对称（玻璃/水面反射）。
    """
    if abs(y1 - y2) > y_tol:
        return False
    axis = (x1 + x2) / 2.0
    d1 = x1 - axis
    d2 = x2 - axis
    if d1 * d2 >= 0:
        return False  # 同侧不算镜像
    if abs(abs(d1) - abs(d2)) > sym_tol:
        return False
    return True


def filter_mirror_in_view(
    df: pd.DataFrame,
    entity_col: str = "fish_id",
    *,
    y_tol: float = 20.0,
    sym_tol: float = 25.0,
    player: str | None = None,
    summary: List[dict] | None = None,
) -> pd.DataFrame:
    """
    单视角内按帧检测「个体与镜像」成对点，将更可能是镜像的那点置为 NaN。

    规则：同一帧内若两点 y 相近且 x 关于某轴对称，视为镜像对；
    保留与上一帧该 fish_id 位置更接近的点，另一者置为 NaN。
    若某鱼无上一帧，则保留与另一鱼距离更近的点（更可能是真实聚集侧）。
    """
    if entity_col not in df.columns or "frame" not in df.columns or "x" not in df.columns or "y" not in df.columns:
        return df

    out = df.copy()
    out = out.sort_values("frame")
    last_pos: Dict[Hashable, np.ndarray] = {}

    frames = out["frame"].unique()
    to_nan = set()  # (index,) 要置 NaN 的行 index

    for frame in frames:
        sub = out[(out["frame"] == frame)].dropna(subset=["x", "y"])
        if len(sub) < 2:
            for _, row in sub.iterrows():
                last_pos[row[entity_col]] = np.array([float(row["x"]), float(row["y"])])
            continue

        rows = sub.index.tolist()
        ids = sub[entity_col].values
        pos = sub[["x", "y"]].values.astype(float)

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if not _is_mirror_pair(
                    pos[i, 0], pos[i, 1],
                    pos[j, 0], pos[j, 1],
                    y_tol=y_tol,
                    sym_tol=sym_tol,
                ):
                    continue

                idi, idj = ids[i], ids[j]
                pi, pj = pos[i], pos[j]

                dist_i = np.inf
                dist_j = np.inf
                if idi in last_pos:
                    dist_i = float(np.linalg.norm(pi - last_pos[idi]))
                if idj in last_pos:
                    dist_j = float(np.linalg.norm(pj - last_pos[idj]))

                if dist_i < dist_j:
                    to_nan.add(rows[j])
                    if summary is not None:
                        row_j = sub.loc[rows[j]]
                        summary.append({
                            "player": player or "",
                            "fish_id": idj,
                            "frame": int(frame),
                            "x": float(row_j["x"]),
                            "y": float(row_j["y"]),
                            "reason": "mirror",
                        })
                elif dist_j < dist_i:
                    to_nan.add(rows[i])
                    if summary is not None:
                        row_i = sub.loc[rows[i]]
                        summary.append({
                            "player": player or "",
                            "fish_id": idi,
                            "frame": int(frame),
                            "x": float(row_i["x"]),
                            "y": float(row_i["y"]),
                            "reason": "mirror",
                        })
                else:
                    pass  # 都无上一帧或距离相等，不删除

        for idx in sub.index:
            if idx in to_nan:
                continue
            row = out.loc[idx]
            last_pos[row[entity_col]] = np.array([float(row["x"]), float(row["y"])])

    if to_nan:
        out.loc[list(to_nan), ["x", "y"]] = np.nan

    return out


def process_mirror_filter(
    df_dict: Dict[str, pd.DataFrame],
    *,
    y_tol: float = 20.0,
    sym_tol: float = 25.0,
    log_dir: str | None = None,
    set_key: str | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    对所有视角的 2D 轨迹做镜像过滤：同一帧内「个体+镜像」成对时，将镜像点置为 NaN。

    若提供 log_dir 与 set_key，会写出 {set_key}_mirror_outliers.csv 记录被删点。
    """
    summary: List[dict] = []
    processed: Dict[str, pd.DataFrame] = {}

    for player, df in df_dict.items():
        if "fish_id" not in df.columns:
            processed[player] = df.copy()
            continue
        out_df = filter_mirror_in_view(
            df,
            entity_col="fish_id",
            y_tol=y_tol,
            sym_tol=sym_tol,
            player=player,
            summary=summary,
        )
        processed[player] = out_df

    if log_dir and summary:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        name = f"{set_key}_mirror_outliers.csv" if set_key else "mirror_outliers.csv"
        out_csv = log_path / name
        pd.DataFrame(summary).sort_values(["player", "fish_id", "frame"]).to_csv(out_csv, index=False)
        print(f"  已写出镜像过滤汇总: {out_csv}")

    return processed


def _remove_outliers_for_entity(
    df: pd.DataFrame,
    entity_col: str,
    entity_id: Hashable,
    *,
    max_jump_per_frame: float,
    player: str | None = None,
    summary: List[dict] | None = None,
) -> pd.DataFrame:
    """
    对单个实体（如某一条鱼）移除异常点：仅将异常点的 x,y 置为 NaN。
    如提供 summary，则将每个异常点的 (player, fish_id, frame, x, y) 记录进去，便于后续写 CSV。
    """
    mask_entity = df[entity_col] == entity_id
    entity_df = df.loc[mask_entity].copy()
    entity_df = entity_df.sort_values("frame")

    if len(entity_df) < 2:
        return df

    positions = entity_df[["x", "y"]].to_numpy(dtype=float)
    frames = entity_df["frame"].to_numpy()

    outlier_mask = detect_outliers_frame_gap_scaled(
        positions, frames, max_jump_per_frame=max_jump_per_frame
    )
    # 进一步检测「短时间离开又回到原地」的整段错误（比如 423/429 正常，中间整段跑到另一处）
    loop_mask = detect_loop_segment_outliers(
        positions,
        frames,
        max_loop_frames=10,
        return_dist_thresh=30.0,
        excursion_dist_thresh=max_jump_per_frame * 2.0,
    )
    combined_mask = outlier_mask | loop_mask

    result_df = df.copy()
    outlier_indices = entity_df.index[combined_mask]
    if len(outlier_indices) > 0:
        # 记录异常点信息（包含「单点大跳」和「整段错误」两种）
        if summary is not None:
            for idx in outlier_indices:
                row = entity_df.loc[idx]
                summary.append(
                    {
                        "player": player if player is not None else "",
                        "fish_id": entity_id,
                        "frame": int(row["frame"]),
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                    }
                )
        result_df.loc[outlier_indices, ["x", "y"]] = np.nan

    return result_df


def process_all_fish_outliers(
    df_dict: Dict[str, pd.DataFrame],
    max_jump_distance: float = 200.0,
    iterative: bool = True,
    max_iterations: int = 10,
    use_frame_gap_scaling: bool = True,
    max_jump_per_frame: float = 100.0,
    *,
    log_dir: str | None = None,
    set_key: str | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    对所有视角、所有鱼的 2D 轨迹做一次简单的异常值检测。

    为兼容旧接口，保留原有参数，但内部逻辑已简化为 **单次扫描**：
    - 若 use_frame_gap_scaling=True（默认），使用 max_jump_per_frame；
    - 否则退化为使用 max_jump_distance 作为「1 帧位移」阈值。

    若提供 log_dir 和 set_key，则在 log_dir 下写出一份
        {set_key}_outliers.csv
    列出所有被判为异常的点（player, fish_id, frame, x, y）。
    """
    # 选择实际使用的阈值
    if use_frame_gap_scaling:
        jump_thr = float(max_jump_per_frame)
    else:
        jump_thr = float(max_jump_distance)

    processed: Dict[str, pd.DataFrame] = {}
    summary: List[dict] = []

    for player, df in df_dict.items():
        if "fish_id" not in df.columns:
            # 没有 fish_id 列就原样返回
            processed[player] = df.copy()
            continue

        out_df = df.copy()
        fish_ids = [fid for fid in df["fish_id"].unique() if fid != "unassigned"]

        for fid in fish_ids:
            out_df = _remove_outliers_for_entity(
                out_df,
                entity_col="fish_id",
                entity_id=fid,
                max_jump_per_frame=jump_thr,
                player=player,
                summary=summary,
            )

        processed[player] = out_df

    # 写出异常值汇总 CSV（可选）
    if log_dir is not None and summary:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # 明细：每个被判为异常的点
        name = f"{set_key}_outliers.csv" if set_key else "outliers.csv"
        out_csv = log_path / name
        df_summary = pd.DataFrame(summary)
        df_summary.sort_values(["player", "fish_id", "frame"]).to_csv(out_csv, index=False)
        print(f"  已写出异常值汇总: {out_csv}")

        # 统计：每个 player / fish_id 被删除了多少点，便于快速查看「有没有删」「删了几个」
        counts = (
            df_summary.groupby(["player", "fish_id"], as_index=False)["frame"]
            .count()
            .rename(columns={"frame": "n_outliers"})
        )
        summary_name = f"{set_key}_outliers_summary.csv" if set_key else "outliers_summary.csv"
        summary_csv = log_path / summary_name
        counts.to_csv(summary_csv, index=False)
        print(f"  已写出异常值数量统计: {summary_csv}")

    return processed


__all__ = [
    "detect_outliers_frame_gap_scaled",
    "process_all_fish_outliers",
    "filter_mirror_in_view",
    "process_mirror_filter",
]

