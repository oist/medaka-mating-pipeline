"""
3D 重建：从三视角 3 个个体的 2D 坐标，每帧做排列组合得到 3D 候选。

逻辑（与用户约定一致）：
- 提取跨三视角的 3 个个体的 2D 坐标，每帧枚举排列组合得到多组「3 个 3D 点」候选。
- 若某组合中任一视角的 2D 缺失（NaN），则该点不三角化，该 3D 留空（NaN），绝不填 0。
- 若三角化结果超出 xyz_bounds 或含 nan/inf，同样该点记为 NaN，不写入无效坐标。
- 得到每帧的 3D 候选后，由 select_best 选最优对应，再在 3D 上做跨帧 ID 匹配，空缺处最后统一插值。
"""

from itertools import permutations
from typing import Dict, List, Tuple

import numpy as np

from scripts.reconstruction_3d.calibration import Calibration
from scripts.reconstruction_3d.load_2d import NUM_INDIVIDUALS

CAM_ORDER = ["cam0", "cam1", "cam2"]
LABELS = ["a", "b", "c"]


def compute_candidates(
    cal: Calibration,
    points_by_frame: Dict[int, Dict[str, List[Tuple[float, float]]]],
    *,
    xyz_bounds: Tuple[float, float] | None = (0.0, 200.0),
) -> Dict[int, List[Dict]]:
    """
    对每一帧枚举 cam1/cam2 的索引排列，对每种排列计算 3 个 3D 点及三视图配准误差。
    返回 {frame: [{"groups": {a:[X,Y,Z],...}, "perm_cam1": [...], "perm_cam2": [...], "cost": float}, ...]}
    """
    low, high = (xyz_bounds or (None, None))[0], (xyz_bounds or (None, None))[1]
    candidates_by_frame: Dict[int, List[Dict]] = {}

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x

    for frame in tqdm(sorted(points_by_frame.keys()), desc="3D候选", unit="frame"):
        cams = points_by_frame[frame]
        pts0 = list(cams.get("cam0", []))
        pts1 = list(cams.get("cam1", []))
        pts2 = list(cams.get("cam2", []))
        while len(pts0) < NUM_INDIVIDUALS:
            pts0.append((np.nan, np.nan))
        while len(pts1) < NUM_INDIVIDUALS:
            pts1.append((np.nan, np.nan))
        while len(pts2) < NUM_INDIVIDUALS:
            pts2.append((np.nan, np.nan))
        pts0, pts1, pts2 = pts0[:NUM_INDIVIDUALS], pts1[:NUM_INDIVIDUALS], pts2[:NUM_INDIVIDUALS]

        perms1 = list(permutations(range(NUM_INDIVIDUALS)))
        perms2 = list(permutations(range(NUM_INDIVIDUALS)))
        frame_candidates = []

        for perm1 in perms1:
            for perm2 in perms2:
                groups = {}
                total_cost = 0.0
                valid = True
                for i, label in enumerate(LABELS):
                    j1, j2 = perm1[i], perm2[i]
                    im0 = np.array(pts0[i], dtype=float)
                    im1 = np.array(pts1[j1], dtype=float)
                    im2 = np.array(pts2[j2], dtype=float)
                    if np.any(np.isnan(im0)) or np.any(np.isnan(im1)) or np.any(np.isnan(im2)):
                        total_cost += np.inf
                        groups[label] = [np.nan, np.nan, np.nan]
                        continue
                    triplet = np.vstack([im0, im1, im2])
                    xyz = cal.compute_XYZ_from_imageCoord_triplet(triplet)
                    # 无效或超出范围一律留空，不写 0
                    if np.any(np.isnan(xyz)) or np.any(np.isinf(xyz)):
                        total_cost += np.inf
                        groups[label] = [np.nan, np.nan, np.nan]
                        continue
                    if low is not None and high is not None:
                        if np.any(xyz < low) or np.any(xyz > high):
                            total_cost += np.inf
                            groups[label] = [np.nan, np.nan, np.nan]
                            continue
                    err = cal.compute_triplet_registration_error(triplet)
                    if np.isfinite(err):
                        total_cost += err
                    else:
                        total_cost += np.inf
                    groups[label] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
                frame_candidates.append({
                    "groups": groups,
                    "perm_cam1": list(perm1),
                    "perm_cam2": list(perm2),
                    "cost": float(total_cost),
                })
        candidates_by_frame[frame] = frame_candidates

    return candidates_by_frame
