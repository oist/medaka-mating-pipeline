# 三视角 3D 重建与跨帧 ID 统一

从三视角 `*_processed.csv` 2D 坐标与相机校准模型，完成 3D 重建、跨帧 ID 统一与轨迹可视化。

## 数据约定

- **2D 坐标**：`{set_key}_player1_processed.csv`（top/cam0）、`player2`（side1/cam1）、`player3`（side2/cam2）。列需含 `frame`, `x`, `y`，可选 `fish_id`, `score`。允许少量视角缺失（NaN）。
- **校准模型目录**：须含 `imCoords_to_XYZ.joblib`（6 个图像坐标 → 3D）；可选 `XYZ_to_imCoords.joblib`（用于配准误差）。

## 流程概览

1. **load_2d**：加载三视角 CSV，组织为 `{frame: {cam0/cam1/cam2: [(x,y), ...]}}`。
2. **triangulate**：每帧枚举 cam1/cam2 的排列，用校准模型计算 3D 候选与配准误差。
3. **select_best**：每帧选代价最小且在 xyz_bounds 内的 a/b/c 对应。
4. **cross_frame_tracking**：匈牙利算法按 3D 距离对相邻帧做标签对应，保证同一物理个体时间上保持同一 a/b/c。
5. **fill_missing_3d**：样条插值补全缺失帧。
6. **export**：导出 `{set_key}_3d.csv`（frame, fish_id, X, Y, Z）与 `{set_key}_best.json`。
7. **visualize_3d**：绘制 3D 轨迹 PNG。

## 使用方式

**Notebook**：`notebooks/05_reconstruction_3d.ipynb`，配置 `PROCESSED_DIR`、`MODEL_FOLDER`、`OUTPUT_DIR` 后运行。

**代码**：

```python
from scripts.reconstruction_3d import run_pipeline, discover_sets

sets = discover_sets(r"D:\sleap_output\processed")
run_pipeline(
    processed_dir=r"D:\sleap_output\processed",
    set_key=sets[0],
    model_folder=r"D:\calibration_model\cali_251217_way1",
    output_dir=r"D:\sleap_output\reconstruction_3d",
    xyz_bounds=(0.0, 200.0),
    cross_frame_tracking=True,
    fill_missing_3d=True,
)
```

**命令行**（可自行在 `pipeline.py` 或本目录下加 `__main__`）：通过 notebook 或上述 API 调用即可。

## 模块说明

| 模块 | 作用 |
|------|------|
| `calibration.py` | 加载 joblib，提供 `compute_XYZ_from_imageCoord_triplet`、`compute_triplet_registration_error` |
| `load_2d.py` | 加载三视角 CSV，`get_points_per_frame`、`run_load_2d` |
| `triangulate.py` | 每帧 3D 候选与代价 `compute_candidates` |
| `select_best.py` | 每帧选最优 a/b/c `select_best_all_frames` |
| `cross_frame_tracking.py` | 跨帧 ID 统一 `apply_cross_frame_tracking` |
| `fill_missing_3d.py` | `best_by_frame_to_dfs`、样条补全、`dfs_to_single_csv` |
| `visualize_3d.py` | `plot_3d_trajectories`、`run_plot` |
| `pipeline.py` | `run_pipeline`、`discover_sets` |
