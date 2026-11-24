# 20260106_E_2_1_FCMC — 3D 重建工作流程（当前方法：3-node bottom-up 模型）

本流程把原始三机位视频 `20260106_E_2_1_FCMC` 处理成每条鱼的 3D 轨迹
（带稳定身份 `final_fish_id`）。

**当前使用的模型**：`Z:\medaka\sleap_models\retrain_20260810_3node_reviewed`
里的 **bottom-up 3-node 模型**（`s14.bottomup_3node.reviewed_*_f24_ms64_paf8`，
骨架 `mouth – pectoral – tail`，只用人工复核过的标注训练）。
它不能用 `run_round_inference.py` 启动（那个只支持 top-down 的
centroid + centered_instance），因此 Stage 1 直接调用 `sleap-nn track`。
Stage 2 之后的 2D 追踪 / 3D 重建沿用 round8 已验证的参数
（见 `report/20260730/round8_3d_tracking/round8_3d_quality_report.md`）。

```
原始视频 (3 views)
  │  Stage 1  SLEAP 推理：sleap-nn track + 3-node bottom-up 模型, max_instances 6（本机 GPU）
  ▼
Z:\medaka\sleap_output\<EXP>\<TAG>\<view>.slp ×3
  │  Stage 2  top_cam 2D 身份追踪（track_simple_nn）
  ▼
top_cam.tracks.csv
  │  Stage 3  3D 重建 + 绑定 top_cam ID（reconstruct_from_slp）
  ▼
<tag>_3d.csv / <tag>_3d_identified.csv
  │  Stage 4  QC（速度尖峰、绑定比例、Rerun 目视检查）
  ▼
（可选）Stage 5  分割辅助 ID 修复 → 重新绑定
```

---

## 0. 现状与前提（2026-10-26 更新）

| 项目 | 位置 | 状态 |
|---|---|---|
| 原始视频 | `Z:\medaka\medaka_202512\20260106_E_2_1_FCMC\{top_cam\cam0_top.mp4, side1_cam\cam1_side.mp4, side2_cam\cam2_side.mp4}` | ✅ 三视角齐全，每个 348600 帧 @166 fps，1200×1200 |
| 实验元数据 | `...\20260106_E_2_1_FCMC\experiment_metadata.json` | ✅ 已存在 |
| SLEAP 模型（当前） | `Z:\medaka\sleap_models\retrain_20260810_3node_reviewed\{top,side}\s14.bottomup_3node.reviewed_{top,side}_f24_ms64_paf8` | ✅ 训练完成（2026-08-10）。side 验证集：平均误差 2.8 px，p90 5.3 px，PCK@10px 0.92 |
| 已有 SLEAP 输出 | `Z:\medaka\sleap_output\20260106_E_2_1_FCMC\round7_260704\` | ⚠️ 旧模型，只有 top_cam chunk，不用 |
| 标定 | `calibration_results\fixed_refractive_192x192x186\`（仓库内） | ✅ `--calibration-backend auto` 默认使用 |
| Python 环境 | 仓库 `.venv\` | ✅ 已建立，`sleap_nn` 可用 |
| GPU | 本机 RTX 5070 Ti 16 GB | ✅ 用 **batch 4**（~33 FPS）；batch 12 会显存溢出、慢 25 倍 → **一次只跑一个视角** |

**进度（2026-10-26）**：side1_cam 正在做 3000 帧测速
（`--frames 171800-174799`，输出到 Claude scratchpad 的 `bench_side1.slp`），
测完后估算全长时间，再跑 side1 全长。

> 另有 `20260106_E_2_1_FCMC_P2`（同组第二段录像，166000 帧）。如需一起分析，
> 把 `$EXP` 换成 `20260106_E_2_1_FCMC_P2` 再跑一遍。

### 0.1 本机路径配置（只需一次）

本机 `Z:` 映射到 `\\bucket.oist.jp\bucket\StephensU`，所以数据在 `Z:\medaka\...`；
而仓库里的 `configs/paths.local.toml`（被 git 追踪）写的是 `Z:\StephensU\medaka\...`。
**不要改那个共享文件**，另建一个本机配置（Stage 2 起要用）：

```powershell
@'
[paths]
data_root                 = "Z:\\medaka"
raw_data_dir              = "Z:\\medaka\\medaka_202512"
medaka_data_dir           = "Z:\\medaka\\medaka_202512"
sleap_output_dir          = "Z:\\medaka\\sleap_output"
sleap_models_dir          = "Z:\\medaka\\sleap_models"
processed_dir             = "Z:\\medaka\\processed"
tracking_output_dir       = "Z:\\medaka\\processed"
reconstruction_output_dir = "Z:\\medaka\\reconstruction_3d"
runs_dir                  = "Z:\\medaka\\runs"
calibration_model_dir     = "calibration_results\\fixed_refractive_192x192x186"
'@ | Set-Content -Encoding utf8 configs\paths.ReiterU.toml
```

### 0.2 本流程使用的变量（每次开新 PowerShell 窗口先粘贴这段）

```powershell
cd C:\Users\ReiterU\Documents\GitHub\medaka-mating-pipeline
$PY   = ".venv\Scripts\python.exe"
$CFG  = "configs\paths.ReiterU.toml"
$EXP  = "20260106_E_2_1_FCMC"

# 轮次标签 = SLEAP 输出子目录名，也是 3D 输出文件前缀
$TAG  = "s14_bottomup_3node_reviewed"

$M = "Z:\medaka\sleap_models\retrain_20260810_3node_reviewed"
$MODEL = @{
    top_cam   = "$M\top\s14.bottomup_3node.reviewed_top_f24_ms64_paf8"
    side1_cam = "$M\side\s14.bottomup_3node.reviewed_side_f24_ms64_paf8"
    side2_cam = "$M\side\s14.bottomup_3node.reviewed_side_f24_ms64_paf8"   # 两个侧视角共用 side 模型
}
$VIDEO = @{
    top_cam   = "Z:\medaka\medaka_202512\$EXP\top_cam\cam0_top.mp4"
    side1_cam = "Z:\medaka\medaka_202512\$EXP\side1_cam\cam1_side.mp4"
    side2_cam = "Z:\medaka\medaka_202512\$EXP\side2_cam\cam2_side.mp4"
}

$SLP_DIR = "Z:\medaka\sleap_output\$EXP\$TAG"
$OUT_DIR = "Z:\medaka\reconstruction_3d\$EXP\$TAG"
New-Item -ItemType Directory -Force $SLP_DIR, $OUT_DIR | Out-Null
```

---

## Stage 1 — SLEAP 推理（3-node bottom-up，三视角全长）

目标产物：`$SLP_DIR\{top_cam,side1_cam,side2_cam}.slp`
（文件名必须正好是 `<view>.slp`，`reconstruct_from_slp --sleap-subdir $TAG` 按这个名字找）。

**`--max_instances 6` 是故意的**（多于 3 条鱼），多出的候选让 3D 阶段能救回
被反光干扰的鱼；Stage 3 的 `--num-candidates 6` 依赖这一点。

### 1.1 测速（每个视角各测一次，约 3000 帧）

```powershell
$view = "side1_cam"     # 依次换成 top_cam / side2_cam
$t = Measure-Command {
    & $PY -m sleap_nn.cli track `
        --data_path   $VIDEO[$view] `
        --model_paths $MODEL[$view] `
        --output_path "$SLP_DIR\bench_$view.slp" `
        --device cuda --batch_size 4 --max_instances 6 `
        --frames 171800-174799
}
"{0}: {1:N1} s / 3000 帧 → 全长约 {2:N1} h" -f $view, $t.TotalSeconds, ($t.TotalSeconds / 3000 * 348600 / 3600)
```

- 估算值偏高一点（包含模型加载时间），作为上限即可。
- 测速得到的 `bench_<view>.slp` 也可以拿来做 1.3 的目视检查，确认无误后删掉。
- 实测记录（测完填写）：

  | 视角 | 3000 帧耗时 | 全长估计 |
  |---|---|---|
  | side1_cam | batch 4：1200 帧 42 s（33 FPS）；batch 12：3000 帧 37 min（1.35 FPS，显存溢出） | batch 4 约 2.9 h |
  | top_cam | | |
  | side2_cam | | |

### 1.2 全长推理（顺序跑三个视角）

```powershell
foreach ($view in "side1_cam", "top_cam", "side2_cam") {
    if (Test-Path "$SLP_DIR\$view.slp") { "skip $view (已存在)"; continue }
    & $PY -m sleap_nn.cli track `
        --data_path   $VIDEO[$view] `
        --model_paths $MODEL[$view] `
        --output_path "$SLP_DIR\$view.slp" `
        --device cuda --batch_size 4 --max_instances 6 `
        *> "$SLP_DIR\$view.inference.log"
}
```

注意事项：

- **一次只跑一个视角，batch 用 4**。2026-10-26 side1 实测：batch 2/4/6/8 = 31/33/33/34 FPS，
  batch 8 显存峰值 13.9 GB；**batch 12 显存溢出，只有 1.35 FPS（全长约 72 h）**。
  全长每视角约 2.9 h。
- 也可以用 notebook 跑（路径自动解析、后台运行、可查进度）：
  [batch_pipeline.ipynb](batch_pipeline.ipynb)
- 任务管理器里会看到**两个** python 进程跑同一条命令（`.venv\Scripts\python.exe`
  和 uv 的 `cpython-3.13\python.exe`）——这是 Windows 上 venv 启动器的正常现象，
  **不是重复任务，不要杀掉其中一个**。
- 不加 `--tracking`：身份追踪由 Stage 2 的 `track_simple_nn` 做。
- 不要用 TensorRT / `--mode predict` 路线——在这块 Blackwell 卡上导出会失败。
- sleap-nn 会缓冲 stdout，log 在跑完前可能是空的；判断进度看 `nvidia-smi`
  的 GPU 占用和 `<view>.slp` 是否出现。
- 中途失败时，循环会跳过已完成的视角，直接重跑即可。

### 1.3 Stage 1 检查

```powershell
Get-ChildItem $SLP_DIR\*.slp | Select-Object Name, Length, LastWriteTime

# 骨架和每帧实例数分布（应为 mouth/pectoral/tail；大多数帧 ≥3 个实例）
& $PY -c "import sys, collections, sleap_io as sio; l = sio.load_slp(sys.argv[1]); n = [len(f.instances) for f in l.labeled_frames]; print(l.skeletons[0].node_names, len(n), 'frames', sorted(collections.Counter(n).items()))" $SLP_DIR\side1_cam.slp

# 叠加预测到视频上，目测 5 秒（每个视角各跑一次）
& $PY -m scripts.sleap.visualize_predictions $EXP --paths-config $CFG `
    --view side1_cam --slp $SLP_DIR\side1_cam.slp --seconds 5 `
    --output $OUT_DIR\qc\side1_cam.pred_5s.mp4
```

重点看：两鱼贴近/交叉时是否把两条鱼的点连成一条（bottom-up 的 PAF 连接错误），
以及 mouth / tail 是否左右颠倒。

---

## Stage 2 — top_cam 2D 身份追踪

为 Stage 3 的绑定步骤提供稳定的 2D 身份锚点。使用默认的「可见节点质心」
（**不要**加 `--anchor-node`；用 mouth 锚定实测差约 2 倍）。
3-node 模型下质心 = mouth / pectoral / tail 中可见点的平均，不需要改代码。

```powershell
& $PY -m scripts.tracking.track_simple_nn $EXP `
    --paths-config $CFG `
    --view top_cam `
    --slp    $SLP_DIR\top_cam.slp `
    --output $SLP_DIR\top_cam.tracks.csv `
    --max-distance 100 --max-tracks 3 --max-gap 500
```

- 产物：`$SLP_DIR\top_cam.tracks.csv`，约 1–2 分钟。
- 参考值（round8，K_1_2）：15 个 track 片段，其中 3 个长片段覆盖大部分时长；
  片段多一些是正常的，Stage 3 会按片段做全局 1 对 1 投票。

---

## Stage 3 — 3D 重建 + 绑定 top_cam 身份

```powershell
& $PY -m scripts.reconstruction_3d.reconstruct_from_slp $EXP `
    --paths-config $CFG `
    --calibration-backend auto `
    --sleap-subdir $TAG `
    --num-candidates 6 `
    --use-visible-node-centroid `
    --xyz-bounds -20 220 --no-shift-to-positive `
    --max-jump-mm 8 --max-jump-hard-cap-mm 80 --max-gap-frames 100 `
    --num-individuals 3 `
    --bind-top-id `
    --top-tracks-csv $SLP_DIR\top_cam.tracks.csv `
    --output-dir $OUT_DIR `
    --output-tag $TAG
```

要点：

- **必须带 `--num-candidates 6`**。进度条应显示 **`3D best-assign`**（约 400–900 帧/s，
  全长端到端约 10–30 分钟）。如果看到 **`3D candidates`**，说明走了慢且不同的旧算法
  （约 9.7 h）——立刻停止，检查参数。
- `--calibration-backend auto` → 折射模型 `fixed_refractive_192x192x186`，残差单位 mm²。
  **不要**带旧的 `--max-residual-per-fish 5000`（那是多项式模型的 px² 阈值）。
- `--max-jump-hard-cap-mm 80` 很关键：防止长 NaN 间隙之后把另一条鱼接成同一个 ID
  （间隙 N 帧时有效门限 = `min(8·N, 80)` mm）。
- `--use-visible-node-centroid` 对 3-node 骨架同样适用（3 个点的可见质心，
  对 mouth/tail 互换不敏感）。

产物（在 `$OUT_DIR`）：

| 文件 | 内容 |
|---|---|
| `<tag>_3d.csv` | `frame, fish_id, X, Y, Z`（几何 ID，已样条补帧） |
| `<tag>_3d_identified.csv` | 增加 `final_fish_id`, `bind_source`, `bind_score`, `source_bind_key` … **下游分析用这个** |
| `<tag>_best_pre_link.json` | 链接前的逐帧候选与残差（QC 用） |
| `<tag>_best.json` | 链接后的逐帧 slot 表（渲染视频用） |
| `<tag>_reconstruction_provenance.json` | 标定后端、模型文件哈希 |
| `<tag>_3d_trajectories.png` | 轨迹总览图 |

---

## Stage 4 — QC

> ⚠️ 下面的参考数值来自 **round8 top-down 模型**在 K_1_x 上的结果。
> 3-node bottom-up 模型还没有在 3D 层面做过同样的 QC，
> 这些数字只能作为大致基准；本实验的结果可以作为第一份 3-node 3D QC 记录。

### 4.1 看终端日志

Stage 3 结束时会打印：

```
[bind] global 1-to-1 assignment geometry fish_id -> top identity key: ...
[bind] xxx/yyy (zz%) rows bound via top_cam_2d
```

参考（K_1_x round8）：top_cam_2d 绑定比例 95–97%，
链接前三鱼同时有效 ≈ 99%，最终三鱼同时有限 ≈ 99%。明显低于此需要排查。

### 4.2 3D 速度尖峰 / ID 交换检查

```powershell
& $PY -m scripts.tracking.evaluate_3d_id_consistency `
    --csv $OUT_DIR\${TAG}_3d_identified.csv `
    --id-col final_fish_id `
    --plot-dir $OUT_DIR\qc `
    --out-csv  $OUT_DIR\qc\id_consistency_3d.csv

& $PY -m scripts.visualization.plot_xyz_speed_qc $OUT_DIR\${TAG}_3d_identified.csv `
    --id-column final_fish_id `
    --output-dir $OUT_DIR\qc
```

- `final_fish_id` 是 top_cam track 的编号，只是稳定标签，**不代表**雌/雄。
  F / M1 / M2 需要人工对照视频确认；确认后可以用
  `--identity <id>=F --identity <id>=M1 --identity <id>=M2` 给图表加标签。
- 由于链接器有 8 mm/帧的相邻帧门限，>8 mm 的跳变应为 0；
  **≥7.5 mm 的近门限跳变**数量才是有用的风险信号（round8 K_1_x 约 600–860 次）。
- 开头几千帧出现非有限值（鱼尚未全部被检测到）属正常。

### 4.3 Rerun 交互式目视检查

挑选 4.2 中尖峰附近、或两鱼接近/交叉的时间窗：

```powershell
& $PY -m scripts.visualization.rerun_viewer $EXP --paths-config $CFG `
    --csv           $OUT_DIR\${TAG}_3d_identified.csv `
    --best-json     $OUT_DIR\${TAG}_best.json `
    --pre-link-json $OUT_DIR\${TAG}_best_pre_link.json `
    --slp-dir       $SLP_DIR `
    --start-frame 171800 --end-frame 176799
# 加 --save $OUT_DIR\qc\window.rrd 可保存离线查看
```

判断标准：三视角 2D 点与 3D 点颜色一致、交叉前后 ID 不互换、无跳变。

---

## Stage 5（可选）— 分割辅助 ID 修复

仅当 Stage 4 发现 top_cam 追踪在遮挡/交叉处碎片化或交换 ID 时使用。
不改变 3D 几何，只替换 Stage 3 绑定所用的 top_cam 身份 CSV。

1. 从现有 3D 结果挖掘干净的 SAM2 提示：

   ```powershell
   & $PY -m sam2_id_experiment.scripts.24_seed_prompts_from_3d `
       --identified-3d-csv  $OUT_DIR\${TAG}_3d_identified.csv `
       --best-pre-link-json $OUT_DIR\${TAG}_best_pre_link.json `
       --paths-config $CFG `
       --output-dir Z:\medaka\runs\seg_id\$EXP\$TAG\seed_from_3d
   ```

2. 用 `sam2_seed_prompts_from_3d.yaml` 运行 `sam2_id_experiment/scripts/01_*`–`23_*`
   的分割 ID 流程（SAM2/Detectron2 掩膜 → SLEAP 门控 → QC → ID 分配/拼接）。
3. 用 `20_export_segmentation_top_tracks.py` 导出 `top_cam.seg_identity_tracks.csv`。
4. 用新的 CSV 重跑 Stage 3（命令同上，只把 `--top-tracks-csv` 换成
   `top_cam.seg_identity_tracks.csv`，并换一个 `--output-tag`，例如 `${TAG}_segid`，
   避免覆盖原结果）。只有在 4.2 / 4.3 的指标不变差时才采用新结果。

---

## 输出位置汇总

```
Z:\medaka\sleap_output\20260106_E_2_1_FCMC\s14_bottomup_3node_reviewed\
    top_cam.slp  side1_cam.slp  side2_cam.slp       # Stage 1
    <view>.inference.log                             # Stage 1 日志
    top_cam.tracks.csv                               # Stage 2
Z:\medaka\reconstruction_3d\20260106_E_2_1_FCMC\s14_bottomup_3node_reviewed\
    <TAG>_3d.csv  <TAG>_3d_identified.csv  ...       # Stage 3
    qc\                                              # Stage 4
```

## 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `run_round_inference` 报 `no run found ... centroid_n*` | 3-node bottom-up 模型不能用这个启动器，改用 Stage 1 的 `sleap_nn.cli track` |
| CUDA out of memory | 同时跑了两个视角，或 batch 太大 → 一次一个视角，`--batch_size 8` |
| `reconstruct_from_slp` 报 `<view> .slp not found` | `.slp` 不在 `$SLP_DIR` 或文件名不是 `<view>.slp`；检查 `--sleap-subdir $TAG` |
| 找不到 `.slp` / 路径以 `Z:\StephensU` 开头 | 没有带 `--paths-config configs\paths.ReiterU.toml` |
| 进度条显示 `3D candidates` | 漏了 `--num-candidates 6`，停止重跑 |
| 绑定比例很低 | 检查 `top_cam.tracks.csv` 是否和 `.slp` 属于同一 `$TAG`；必要时做 Stage 5 |
| 想检查行为指标 | `scripts/analysis/compute_*_metrics.py` 目前仍是占位脚本（`SystemExit("TODO")`），尚不能直接使用 |

---

## 附录：备选模型 round8（top-down，已在 K_1_x 上验证）

如果 3-node bottom-up 的结果在 Stage 1.3 / Stage 4 明显不如预期，可退回 round8
（`retrain_20260722_round8`，top-down mouth+tail），Stage 2–5 命令完全相同，只换 `$TAG`：

```powershell
$TAG = "round8_260723_strictmouth"
$SLP_DIR = "Z:\medaka\sleap_output\$EXP\$TAG"
$OUT_DIR = "Z:\medaka\reconstruction_3d\$EXP\$TAG"

& $PY -m scripts.sleap.run_round_inference `
    --round-dir   Z:\medaka\sleap_models\retrain_20260722_round8 `
    --round-tag   $TAG `
    --experiment  $EXP `
    --data-root   Z:\medaka\medaka_202512 `
    --output-root Z:\medaka\sleap_output `
    --predict-batch-size 12 --max-instances 6 --continue-on-fail
```

（本机约 3.2 h/视角；直接输出 `<view>.slp`，不需要 merge。HPC 路线见
`scripts/hpc/README.md`，标签 `round8_260729_snn031_m6`，完成后需
`scripts.tracking.main --merge-chunks`。）两套结果放在不同 `$TAG` 目录下，
可以并排对比。
