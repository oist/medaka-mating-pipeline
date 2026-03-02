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


def _triplets_with_filled_view(
    cal: Calibration,
    cam0: list,
    cam1: list,
    cam2: list,
    missing_view: str,
) -> List[tuple]:
    """
    当某一视角无点时，用另两视角预测该视角 2D，生成 (i0, i1, i2, p0, p1, p2) 列表。
    missing_view 为 "cam0" | "cam1" | "cam2"。
    """
    out: List[tuple] = []
    if missing_view == "cam2":
        # 缺 cam2(YZ)：用 cam0(XY) + cam1(XZ) 预测 YZ
        for i0 in range(len(cam0)):
            for i1 in range(len(cam1)):
                p0 = np.array(cam0[i0], dtype=float)
                p1 = np.array(cam1[i1], dtype=float)
                p2 = cal.compute_YZ_imcoords_from_XZ_XY(p1, p0)
                out.append((i0, i1, 0, p0, p1, p2))
    elif missing_view == "cam1":
        # 缺 cam1(XZ)：用 cam0(XY) + cam2(YZ) 预测 XZ
        for i0 in range(len(cam0)):
            for i2 in range(len(cam2)):
                p0 = np.array(cam0[i0], dtype=float)
                p2 = np.array(cam2[i2], dtype=float)
                p1 = cal.compute_XZ_imcoords_from_XY_YZ(p0, p2)
                out.append((i0, 0, i2, p0, p1, p2))
    else:
        # 缺 cam0(XY)：用 cam1(XZ) + cam2(YZ) 预测 XY
        for i1 in range(len(cam1)):
            for i2 in range(len(cam2)):
                p1 = np.array(cam1[i1], dtype=float)
                p2 = np.array(cam2[i2], dtype=float)
                p0 = cal.compute_XY_imcoords_from_XZ_YZ(p1, p2)
                out.append((0, i1, i2, p0, p1, p2))
    return out


def compute_candidates_for_frame(
    cal: Calibration,
    points: Dict[str, List[Tuple[float, float]]],
    xyz_bounds: Tuple[float, float] | None = None,
    fill_missing_view: bool = True,
) -> List[Dict]:
    """
    对单帧计算所有 cam0×cam1×cam2 组合的 3D 候选及配准误差。

    若某视角无点且 fill_missing_view=True，则用另两视角预测该视角 2D 再算 3D（最多补一个视角）。
    """
    cam0 = list(points.get("cam0", []))
    cam1 = list(points.get("cam1", []))
    cam2 = list(points.get("cam2", []))
    n0, n1, n2 = len(cam0), len(cam1), len(cam2)

    triplets: List[tuple] = []  # (i0, i1, i2, p0, p1, p2)
    if n0 > 0 and n1 > 0 and n2 > 0:
        for i0 in range(n0):
            for i1 in range(n1):
                for i2 in range(n2):
                    triplets.append((i0, i1, i2, np.array(cam0[i0]), np.array(cam1[i1]), np.array(cam2[i2])))
    elif fill_missing_view:
        if n0 == 0 and n1 > 0 and n2 > 0:
            triplets = _triplets_with_filled_view(cal, cam0, cam1, cam2, "cam0")
        elif n1 == 0 and n0 > 0 and n2 > 0:
            triplets = _triplets_with_filled_view(cal, cam0, cam1, cam2, "cam1")
        elif n2 == 0 and n0 > 0 and n1 > 0:
            triplets = _triplets_with_filled_view(cal, cam0, cam1, cam2, "cam2")

    out: List[Dict] = []
    for i0, i1, i2, p0, p1, p2 in triplets:
        imCoords = np.array(
            [
                [float(p0[0]), float(p0[1])],
                [float(p1[0]), float(p1[1])],
                [float(p2[0]), float(p2[1])],
            ],
            dtype=float,
        )
        xyz = cal.compute_XYZ_from_imageCoord_triplet(imCoords)
        # 与 Dynamics_of_dominance 一致：只在坐标为 NaN 时视为无效；
        # 不再用 xyz_bounds 直接丢弃候选，而是保留由配准误差决定优劣。
        invalid = np.any(np.isnan(xyz))
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
    fill_missing_view: bool = True,
) -> Dict[int, List[Dict]]:
    """对每帧计算 3D 候选及配准误差；缺一视角时用两视角预测该视角再算 3D。"""
    items = sorted(points_by_frame.items())
    return {
        frame: compute_candidates_for_frame(cal, pts, xyz_bounds, fill_missing_view=fill_missing_view)
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
