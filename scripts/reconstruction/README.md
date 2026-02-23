# 2D→3D 重建模块

从各相机 2D 轨迹到水槽内 3D 坐标的完整流程，支持**对应不明**时自动选择最佳点对应。

## 流程

| 步骤 | 脚本 | 说明 |
|------|------|------|
| Step 1 | `step1_get_points.py` | 点の取得：从 processed CSV 提取每帧 2D 点 |
| Step 2 | `step2_compute_3d_candidates.py` | 3次元候補の計算：用 2D→3D 模型计算 3D 候选及配准误差（参考 Dynamics_of_dominance） |
| Step 3 | `step3_select_best_correspondence.py` | 最適な対応の選択：基于配准误差选择最优 a/b/c 对应（非 KMeans） |
| Step 4 | `step4_cross_frame_tracking.py` | 跨帧个体追踪：匈牙利算法 + 运动方向 flow，保证同一物理个体时间上标签一致 |
| 合并 | `merge_with_mapping.py` | 合并到 2Dto3D_matching 表，添加 CHAIN 列 |
| 主流程 | `pipeline.py` | 串联 Step 1→2→3→4，输出 3D CSV 和合并表 |
| 可视化 | `visualize_3d.py` | 从 3D CSV 生成 MP4 视频 |

## 用法

```bash
# 完整流程（推荐）
python -m scripts.reconstruction.pipeline ^
  --model_folder "H:\calibration_output\cali_way1" ^
  --processed_dir "H:\sleap_output\processed" ^
  --set_key "20251220_E_1_2_FCMC_male1_Courtship_1905.506-1913.422" ^
  --output_dir "H:\3d_trajectories" ^
  --matching_csv "H:\3d_trajectories\2Dto3D_matching.csv"

# 分步执行
python -m scripts.reconstruction.step1_get_points --processed_dir ... --set_key ... --output points.json
python -m scripts.reconstruction.step2_compute_3d_candidates --model_folder ... --points points.json --output candidates.json
python -m scripts.reconstruction.step3_select_best_correspondence --candidates candidates.json --output best.json
python -m scripts.reconstruction.step4_cross_frame_tracking --best_json best.json --output best.json

# 可视化
python -m scripts.reconstruction.visualize_3d --output_dir ... --set_key ... --format mp4
```

## 输出

- `{set_key}_a_3d.csv` 等：每条鱼的 3D 轨迹
- `2Dto3D_matching_merged.csv`：合并表，含 CHAIN、ALG_CHAIN_0/1/2 列
- 链格式：`PLAYER1-FISH_1-PLAYER2-FISH_0-PLAYER3-FISH_0` 表示该 3D 点在各相机的 fish 索引
