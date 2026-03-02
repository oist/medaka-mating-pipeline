"""
多视角 3D 多目标跟踪：ID 绑定 3D Track，全在 3D 空间关联
========================================================================

目标：三视角 3D 重建 → 跨帧/跨视频一致 ID，ID 永不交换。

核心原则：
- ID 绑定在 3D Track 上；所有关联在 3D 空间完成；2D ID 仅作视角内参考，无全局意义。
- 第一帧在 3D 上赋 ID；后续帧仅做「上一帧 3D Track → 当前帧 3D 点」的一对一匹配，不交换 ID。

数学定义
--------
1) 3D Track
   - 定义：Track_k = {(t, X_t) : t in frames, X_t in R^3 或 NaN}，同一 track 对应同一物理个体。
   - ID(track) 为全局唯一且不变；一旦分配不再重新分配或交换。

2) 关联代价（几何 + 运动）
   - 设上一帧 track i 的位置为 P_i，可选速度 v_i = (P_i - P_i_prev) / dt。
   - 当前帧 3D 点 Q_j。
   - 几何代价：d_ij = || P_i - Q_j ||_2
   - 运动代价（可选）：若 v_i 存在，m_ij = 1 - cos(angle(v_i, Q_j - P_i))，即方向不一致惩罚。
   - 总代价：c_ij = d_ij + λ * m_ij（λ ≥ 0）。若 P_i 或 Q_j 为缺失（NaN），c_ij = +∞。

3) 分配
   - 每帧：上一帧 K 条 track（本流程固定 K=3）与当前帧 M 个 3D 点（M ≤ 3）。
   - 求排列 π 使 sum_i c_{i, π(i)} 最小，约束为一对一（每条 track 至多匹配一个点，每个点至多匹配一条 track）。
   - 实现：匈牙利算法 linear_sum_assignment(C)，C 为 K×M，缺位用 inf 填充。
   - 匹配结果：matched (i,j) → track i 在本帧观测为 Q_j，保持 ID_i；未匹配的 track 本帧为缺失（NaN）；未匹配的点不赋予已有 ID（本流程中 M≤K，通常无未匹配点）。

4) ID 永不交换
   - 只做「track → 当前点」的分配，不做「点 ↔ 点」或「ID ↔ ID」的互换；因此同一 track 的 ID 在时间上恒定。

流程概览
--------
  2D (三视角) → 每帧 3D 重建（排列选最优）→ 3D Tracking（首帧赋 ID，之后匈牙利）→ ID 管理（空缺插值）→ 导出 CSV。

失败情况分析（见 FAILURE_ANALYSIS 常量）
-----------------------------------------
- 重建失败：缺视角/超范围 → 该帧 slot 填 NaN，插值补全。
- 关联歧义：两目标过近 → 加大 motion_weight 或后处理平滑。
- 遮挡/消失：该 track 该帧 NaN，ID 保留，插值补全。
- 目标数≠3：固定 3 track，多出的点不赋 ID，少则对应 track 为 NaN。
- 跨视频 ID：需外部规则（如首段为参考，后续段按 3D 匹配）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from scripts.reconstruction_3d.load_2d import run_load_2d
from scripts.reconstruction_3d.reconstruct_no_calibration import (
    compute_candidates_no_calib,
)
from scripts.reconstruction_3d.select_best import select_best_all_frames
from scripts.reconstruction_3d.calibration import Calibration
from scripts.reconstruction_3d.triangulate import compute_candidates
from scripts.reconstruction_3d.fill_missing_3d import (
    best_by_frame_to_dfs,
    dfs_to_single_csv,
    ensure_all_frames_and_fill_spline,
)
from scripts.reconstruction_3d.visualize_3d import run_plot

# 固定 3 个个体，ID 与 2D 无关
TRACK_IDS = ["ID_1", "ID_2", "ID_3"]
_SRC = ["a", "b", "c"]


def _groups_to_points(groups: Dict) -> np.ndarray:
    """groups (a,b,c 或 ID_1,2,3) → (3,3) 数组，无效为 NaN。"""
    out = np.full((3, 3), np.nan)
    labels = TRACK_IDS if "ID_1" in groups else _SRC
    for i, lb in enumerate(labels):
        if lb in groups and groups[lb] is not None:
            pt = groups[lb]
            if len(pt) == 3 and np.any(np.isfinite(pt)):
                out[i, :] = pt
    return out


def _cost_matrix(
    prev_pts: np.ndarray,
    curr_pts: np.ndarray,
    prev_vel: Optional[np.ndarray] = None,
    motion_weight: float = 0.0,
) -> np.ndarray:
    """
    几何 + 运动一致性代价矩阵 C[i,j] = d_ij + λ*m_ij。
    prev_pts (3,3), curr_pts (3,3)；prev_vel (3,3) 可选。
    """
    C = np.full((3, 3), np.inf)
    for i in range(3):
        if np.any(np.isnan(prev_pts[i])) or not np.any(np.isfinite(prev_pts[i])):
            continue
        for j in range(3):
            if np.any(np.isnan(curr_pts[j])) or not np.any(np.isfinite(curr_pts[j])):
                continue
            d = float(np.linalg.norm(prev_pts[i] - curr_pts[j]))
            m = 0.0
            if motion_weight > 0 and prev_vel is not None and np.any(np.isfinite(prev_vel[i])):
                delta = curr_pts[j] - prev_pts[i]
                v = prev_vel[i]
                nv = np.linalg.norm(v)
                nd = np.linalg.norm(delta)
                if nv > 1e-6 and nd > 1e-6:
                    cos_a = np.dot(v, delta) / (nv * nd)
                    cos_a = np.clip(cos_a, -1.0, 1.0)
                    m = 1.0 - cos_a  # 0 同向，2 反向
            C[i, j] = d + motion_weight * m
    return C


def assign_3d_tracks(
    best_by_frame: Dict[int, Dict],
    *,
    motion_weight: float = 10.0,
) -> Dict[int, Dict]:
    """
    3D Tracking：ID 绑定 3D Track，全在 3D 空间做关联，ID 永不交换。

    - 第一帧：3 个 3D 点按 X 升序赋 ID_1, ID_2, ID_3。
    - 后续帧：上一帧 3 条 track 与当前帧 3 个点做匈牙利（几何+运动代价），匹配后保持 ID。
    - 空缺 slot 填 NaN，不改变已有 ID。
    """
    frames = sorted(best_by_frame.keys())
    if not frames:
        return {}

    result: Dict[int, Dict] = {}
    # 上一帧位置与速度（用于运动一致性）
    prev_pts = np.full((3, 3), np.nan)
    prev_pts_2 = np.full((3, 3), np.nan)  # 再上一帧，用于速度

    for idx, f in enumerate(frames):
        info = best_by_frame[f]
        groups = info.get("groups", {})
        curr_pts = _groups_to_points(groups)

        if idx == 0:
            # 第一帧：按 X 赋 ID_1, ID_2, ID_3
            valid = [i for i in range(3) if np.any(np.isfinite(curr_pts[i]))]
            valid_sorted = sorted(valid, key=lambda i: curr_pts[i, 0])
            new_groups = {}
            for i, tid in enumerate(TRACK_IDS):
                if i < len(valid_sorted):
                    new_groups[tid] = list(curr_pts[valid_sorted[i]])
                else:
                    new_groups[tid] = [np.nan, np.nan, np.nan]
            result[f] = {
                "groups": new_groups,
                "perm_cam1": list(info.get("perm_cam1", [])),
                "perm_cam2": list(info.get("perm_cam2", [])),
            }
            prev_pts = _groups_to_points(new_groups)
            prev_pts_2 = prev_pts.copy()
            continue

        # 速度：上一帧 - 再上一帧（仅用于运动代价）
        dt = float(f - frames[idx - 1])
        if dt > 0 and idx >= 2:
            prev_vel = (prev_pts - prev_pts_2) / dt
        else:
            prev_vel = None

        C = _cost_matrix(prev_pts, curr_pts, prev_vel=prev_vel, motion_weight=motion_weight)
        # 匈牙利：行=track(ID)，列=当前点；若代价矩阵全为 inf 则不可行，本帧全部置 NaN
        n_track, n_pt = C.shape
        n_pt_orig = n_pt
        if n_pt < n_track:
            C = np.hstack([C, np.full((n_track, n_track - n_pt), np.inf)])
        feasible = np.isfinite(C) & (C < 1e9)
        if not np.any(feasible):
            new_groups = {tid: [np.nan, np.nan, np.nan] for tid in TRACK_IDS}
        else:
            # 用有限大值替代 inf，保证 linear_sum_assignment 有解
            C_safe = np.where(np.isfinite(C), C, 1e10)
            row_ind, col_ind = linear_sum_assignment(C_safe)
            new_groups = {}
            for i, tid in enumerate(TRACK_IDS):
                j = int(col_ind[np.where(row_ind == i)[0][0]])
                if j < n_pt_orig and np.isfinite(C[i, j]) and C[i, j] < 1e9:
                    new_groups[tid] = list(curr_pts[j])
                else:
                    new_groups[tid] = [np.nan, np.nan, np.nan]
        result[f] = {
            "groups": new_groups,
            "perm_cam1": list(info.get("perm_cam1", [])),
            "perm_cam2": list(info.get("perm_cam2", [])),
        }
        prev_pts_2 = prev_pts.copy()
        prev_pts = _groups_to_points(new_groups)

    return result


def run_full_pipeline(
    processed_dir: str,
    set_key: str,
    *,
    model_folder: Optional[str] = None,
    output_dir: Optional[str] = None,
    xyz_bounds: Optional[Tuple[float, float]] = (0.0, 200.0),
    fill_missing_3d: bool = True,
    motion_weight: float = 10.0,
    plot: bool = True,
) -> Dict:
    """
    完整流程：2D → 三视角 3D 重建（有校准用校准，无则用 top=XY/side1=XZ/side2=YZ）→ 3D tracking → 空缺线性插值 → 导出 CSV 与轨迹图。

    - model_folder：校准模型目录（含 imCoords_to_XYZ.joblib）。若提供则用校准三角化；若 None 则用无校准映射。
    - 2D 仅用于得到每帧每视角的点集，不依赖 2D ID 的全局一致性。
    - 关联与 ID 全部在 3D 空间完成，ID 永不交换。
    """
    points_by_frame = run_load_2d(processed_dir, set_key, use_fish_id=False)
    if model_folder:
        cal = Calibration(model_folder)
        candidates_by_frame = compute_candidates(
            cal, points_by_frame, xyz_bounds=xyz_bounds
        )
    else:
        candidates_by_frame = compute_candidates_no_calib(
            points_by_frame, xyz_bounds=xyz_bounds
        )
    best = select_best_all_frames(candidates_by_frame, xyz_bounds=xyz_bounds)
    best_tracked = assign_3d_tracks(best, motion_weight=motion_weight)

    out = Path(output_dir or processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = None
    png_path = None

    dfs_3d = best_by_frame_to_dfs(
        best_tracked, shift_to_positive=True, ensure_non_negative=False
    )
    if dfs_3d:
        if fill_missing_3d and points_by_frame:
            all_frames = sorted(points_by_frame.keys())
            dfs_3d = ensure_all_frames_and_fill_spline(dfs_3d, all_frames)
        csv_path = dfs_to_single_csv(dfs_3d, set_key=set_key, output_dir=str(out))
        if plot:
            png_path = run_plot(str(out), set_key, xyz_lim=xyz_bounds)

    return {
        "points_by_frame": points_by_frame,
        "best_tracked": best_tracked,
        "csv_path": csv_path,
        "png_path": png_path,
    }


# ---------------------------------------------------------------------------
# 失败情况分析（Failure case analysis）
# ---------------------------------------------------------------------------
FAILURE_ANALYSIS = r"""
1) 重建失败（某帧无有效 3D）
   - 原因：三视角 2D 缺一视角过多、或排列组合均超出 xyz_bounds、或几何不一致。
   - 表现：该帧 best 中 groups 全为 NaN 或部分 NaN。
   - 处理：该帧 3D tracking 仍进行，缺失 slot 填 NaN；后续用线性插值补全；若整段缺失则插值可能外推不准。

2) 关联歧义（两目标距离很近）
   - 原因：两 3D 点与上一帧两 track 的代价接近，匈牙利仍给出唯一解但可能偶发错配。
   - 表现：短时 ID 跳变或轨迹交叉。
   - 处理：加大 motion_weight 利用运动一致性；或引入轨迹平滑/后处理。

3) 遮挡/短暂消失
   - 原因：某个体在某帧三视角均未检测到或重建失败。
   - 表现：该 track 在该帧为 NaN。
   - 处理：ID 保留，该帧不匹配；插值阶段用前后帧补全；若消失过长可考虑 re-id（本流程不自动 re-id）。

4) 新目标出现 / 目标数 ≠ 3
   - 本流程假设固定 3 条 track；若某帧只有 2 个 3D 点，则 2 条 track 有观测、1 条为 NaN。
   - 若某帧出现多于 3 个点（例如多相机多检测），当前实现仍只取 3 点（来自 select_best），多余点不赋 ID。

5) 跨视频一致 ID
   - 当前单次 run 仅保证单视频内 ID 不交换；跨视频需在流程外定义规则（例如以第一段为参考，第二段按 3D 形状/轨迹匹配到参考 ID）。
"""


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="多视角 3D 多目标跟踪：2D→3D→3D tracking→ID 管理，ID 绑定 3D Track。"
    )
    p.add_argument("--processed_dir", required=True, help="*_processed.csv 所在目录")
    p.add_argument("--set_key", required=True, help="片段 set key")
    p.add_argument("--model_folder", default=None, help="校准模型目录（必填则用校准 3D 重建）")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--xyz_bounds", type=float, nargs=2, default=[0.0, 200.0], metavar=("LOW", "HIGH"))
    p.add_argument("--motion_weight", type=float, default=10.0, help="运动一致性代价权重")
    p.add_argument("--no_fill", action="store_true", help="不插值空缺")
    p.add_argument("--no_plot", action="store_true")
    args = p.parse_args()
    xyz_bounds = (float(args.xyz_bounds[0]), float(args.xyz_bounds[1])) if args.xyz_bounds else None
    res = run_full_pipeline(
        processed_dir=args.processed_dir,
        set_key=args.set_key,
        model_folder=args.model_folder,
        output_dir=args.output_dir,
        xyz_bounds=xyz_bounds,
        fill_missing_3d=not args.no_fill,
        motion_weight=args.motion_weight,
        plot=not args.no_plot,
    )
    if res.get("csv_path"):
        print("3D CSV:", res["csv_path"])
    if res.get("png_path"):
        print("轨迹图:", res["png_path"])


if __name__ == "__main__":
    main()
