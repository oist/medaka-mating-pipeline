# Notebooks 说明

本目录包含 medaka 交配行为分析流水线的各步骤 Notebook，按**流水线顺序**编号。

| 编号 | 文件名 | 用途 | 对应脚本 |
|------|--------|------|----------|
| 01 | `01_acquisition_qc.ipynb` | **采集与预览 QC**：多相机同步录制、预览检查 | `scripts/acquisition/` |
| 02 | `02_calibration.ipynb` | **相机校准**：抽帧 → 手动标注 → 训练模型 → 重投影 QC → 3D 三角化 | `scripts/calibration/` |
| 03 | `03_sleap.ipynb` | **SLEAP 跟踪**：视频推理、.slp 导出 CSV/H5、多视角 QC | `scripts/sleap/` |
| 04 | `04_tracking.ipynb` | **跟踪后处理**：ID 分配、异常值检测、缺失值填充、轨迹可视化 | `scripts/tracking/` |
| 05 | `05_behavior_metrics.ipynb` | **行为指标**：求偶/攻击等行为事件与指标计算 | `scripts/analysis/` |
| 06 | `06_stats.ipynb` | **统计分析**：不同视频/个体组合的统计 | `scripts/analysis/` |
| 07 | `07_trajectory_3d.ipynb` | **3D 鱼轨迹**：从 2D 合并 → 三角化 → 3D 轨迹可视化 | `scripts/calibration/` |

## 流水线顺序

```
采集 (01) → 校准 (02) → SLEAP 推理 (03) → 跟踪后处理 (04) → 3D 轨迹 (07) → 行为指标 (05) → 统计 (06)
```

## 备注

- `notebook`：早期流水线笔记（纯文本），可作参考。
- 部分 Notebook（如 05、06）可能仍为空或占位，待后续补充。
