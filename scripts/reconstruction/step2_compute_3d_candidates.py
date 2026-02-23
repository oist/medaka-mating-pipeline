"""
Step 2: 3次元候補の計算（参考 Dynamics_of_dominance 逻辑）
========================================================

对每帧的 9 个 2D 点（3 相机 × 最多 3 点），
枚举所有 cam0×cam1×cam2 组合，用 imCoords_to_XYZ 计算 3D 候选点，
并计算每个候选的配准误差（registration cost）。

配准误差用于 Step 3 选择最优对应，而非 KMeans 聚类。

为何会出现超过物理范围（如 >200mm）的 3D 坐标？
- 三视角 2D 点对应错误时：错误的 (cam0_i, cam1_j, cam2_k) 组合会喂给校准模型，
  模型仍会输出一个 XYZ，但该点往往在有效水槽外（或负或很大）。
- 校准模型是回归器，在「异常输入」下会外推，不保证落在标定物范围内。
- 当前仅排除了负坐标；未排除「正值但超出物理上界」的候选。
可通过 xyz_bounds=(0, 200) 或 (0, 1200) 等限定有效范围，超出范围的候选视为无效。

用法:
    python -m scripts.reconstruction.step2_compute_3d_candidates --model_folder H:\\calibration_output\\cali_way1 --points points.json --output candidates.json [--xyz_bounds 0 200]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from .calibration import Calibration


def compute_candidates_for_frame(
    cal: Calibration,
    points: Dict[str, List[Tuple[float, float]]],
    xyz_bounds: Tuple[float, float] | None = None,
) -> List[Dict]:
    """
    对单帧计算所有 cam0×cam1×cam2 组合的 3D 候选及配准误差。

    points:
        {"cam0": [(x,y),...], "cam1": [...], "cam2":[...]}
    xyz_bounds:
        若给定 (low, high)，则 X/Y/Z 任一超出 [low, high] 的候选视为无效（cost=inf）。
        例如 (0, 200) 或 (0, 1200)，与校准/水槽物理范围一致。

    返回:
        [
          {"cam0_idx": i0, "cam1_idx": i1, "cam2_idx": i2, "xyz": [X,Y,Z], "cost": float},
          ...
        ]
    """
    cam0 = points.get("cam0", [])
    cam1 = points.get("cam1", [])
    cam2 = points.get("cam2", [])
    n0, n1, n2 = len(cam0), len(cam1), len(cam2)
    if n0 == 0 or n1 == 0 or n2 == 0:
        return []

    out: List[Dict] = []
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                p0 = cam0[i0]
                p1 = cam1[i1]
                p2 = cam2[i2]

                imCoords = np.array([
                    [p0[0], p0[1]],
                    [p1[0], p1[1]],
                    [p2[0], p2[1]],
                ], dtype=float)

                xyz = cal.compute_XYZ_from_imageCoord_triplet(imCoords)
                # 负坐标或 NaN 表示错误对应；若设了 xyz_bounds，超出范围也视为无效
                invalid = np.any(xyz < 0) or np.any(np.isnan(xyz))
                if not invalid and xyz_bounds is not None:
                    low, high = xyz_bounds
                    invalid = np.any(xyz < low) or np.any(xyz > high)
                if invalid:
                    cost = np.inf
                else:
                    cost = cal.compute_triplet_registration_error(imCoords)

                out.append({
                    "cam0_idx": int(i0),
                    "cam1_idx": int(i1),
                    "cam2_idx": int(i2),
                    "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                    "cost": float(cost) if not np.isnan(cost) else None,
                })
    return out


def compute_candidates(
    cal: Calibration,
    points_by_frame: Dict[int, Dict[str, List[Tuple[float, float]]]],
    xyz_bounds: Tuple[float, float] | None = None,
) -> Dict[int, List[Dict]]:
    """对每帧计算 3D 候选及配准误差。xyz_bounds 超出范围的候选 cost=inf。"""
    items = sorted(points_by_frame.items())
    return {
        frame: compute_candidates_for_frame(cal, pts, xyz_bounds)
        for frame, pts in tqdm(items, desc="Step2 3D候选", unit="frame")
    }


def run(
    model_folder: str,
    points_path: str,
    output_path: str | None = None,
    xyz_bounds: Tuple[float, float] | None = None,
) -> Dict:
    """
    执行 Step 2。
    xyz_bounds: 例如 (0, 200) 或 (0, 1200)，与校准/水槽物理范围一致；超出范围的候选视为无效。
    """
    cal = Calibration(model_folder)

    with open(points_path, encoding="utf-8") as f:
        raw = json.load(f)
    points_by_frame = {int(k): v for k, v in raw.items()}
    n_frames = len(points_by_frame)
    print(f"共 {n_frames} 帧待处理")

    candidates = compute_candidates(cal, points_by_frame, xyz_bounds)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in candidates.items()}, f, indent=2, ensure_ascii=False)
        print(f"已保存: {output_path}")
    return candidates


def main():
    ap = argparse.ArgumentParser(description="Step 2: 计算 3D 候选及配准误差（参考 Dynamics_of_dominance）")
    ap.add_argument("--model_folder", required=True, help="校准模型目录")
    ap.add_argument("--points", required=True, help="Step 1 输出的 JSON")
    ap.add_argument("--output", default=None, help="输出 JSON 路径")
    ap.add_argument(
        "--xyz_bounds",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="3D 有效范围 [LOW, HIGH]（X/Y/Z 任一超出则候选无效），例如 --xyz_bounds 0 200 或 0 1200",
    )
    args = ap.parse_args()
    bounds = tuple(args.xyz_bounds) if args.xyz_bounds is not None else None
    run(args.model_folder, args.points, args.output, xyz_bounds=bounds)


if __name__ == "__main__":
    main()
