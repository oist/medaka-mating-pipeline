# 20260106_E_2_1_FCMC — 3D 重建工作流程

本流程把原始三机位视频 `20260106_E_2_1_FCMC` 处理成每条鱼的 3D 轨迹
（带稳定身份 `final_fish_id`）。它照搬目前已验证的 **round8** 生产方案
（`round8_260729_snn031_m6`，在 `20251225_K_1_{1,2,3}_FCMC` 上跑通并做过 QC，
见 `report/20260730/round8_3d_tracking/round8_3d_quality_report.md`）。

```
原始视频 (3 views)
  │  Stage 1  SLEAP 推理（round8 模型, max-instances 6）  ← HPC 或本机 GPU 二选一
  ▼
<view>.slp ×3（HPC 路线需先合并 chunks）
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

## 0. 现状与前提（2026-10-26 检查结果）

| 项目 | 位置 | 状态 |
|---|---|---|
| 原始视频 | `Z:\medaka\medaka_202512\20260106_E_2_1_FCMC\{top_cam,side1_cam,side2_cam}\cam*.mp4` | ✅ 三个视角齐全，每个 348600 帧 @166 fps，1200×1200 |
| 实验元数据 | `...\20260106_E_2_1_FCMC\experiment_metadata.json` | ✅ 已存在（里面的路径写的是 `Z:\StephensU\...`，merge 步骤会刷新） |
| SLEAP 模型 | `Z:\medaka\sleap_models\retrain_20260722_round8\{top,side}\runs\round8_260723_strictmouth.{centroid,centered_instance}_n*` | ✅ round8 是目前生产模型 |
| 已有 SLEAP 输出 | `Z:\medaka\sleap_output\20260106_E_2_1_FCMC\round7_260704\` | ⚠️ **只有 top_cam 的 8 个 chunk，没有 side1/side2** → 不能直接用于 3D，需要重新跑 Stage 1 |
| 标定 | `calibration_results\fixed_refractive_192x192x186\`（仓库内） | ✅ `--calibration-backend auto` 默认使用它 |
| Python 环境 | 仓库 `.venv\` | ❌ 本机尚未创建 → 见 0.1 |
| GPU | 本机 RTX 5070 Ti 16 GB | ✅ 可用于本机推理（较慢） |

> 另有 `20260106_E_2_1_FCMC_P2`（同组的第二段录像，166000 帧）。如需一起分析，
> 把下面所有命令里的实验名换成 `20260106_E_2_1_FCMC_P2` 再跑一遍即可。

### 0.1 建立 Python 环境（只需一次）

```powershell
cd C:\Users\ReiterU\Documents\GitHub\medaka-mating-pipeline
uv sync --python 3.11          # 会装 torch(cu130)、sleap-io、rerun-sdk 等
.venv\Scripts\python.exe -c "import sleap_io, torch; print(torch.cuda.is_available())"
```

### 0.2 本机路径配置（只需一次）

本机 `Z:` 映射到 `\\bucket.oist.jp\bucket\StephensU`，所以数据在 `Z:\medaka\...`；
而仓库里的 `configs/paths.local.toml` 写的是另一台机器的 `Z:\StephensU\medaka\...`，
**不要改那个共享文件**，另建一个本机配置：

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

### 0.3 本流程使用的变量（每次开新 PowerShell 窗口先粘贴这段）

```powershell
cd C:\Users\ReiterU\Documents\GitHub\medaka-mating-pipeline
$PY   = ".venv\Scripts\python.exe"
$CFG  = "configs\paths.ReiterU.toml"
$EXP  = "20260106_E_2_1_FCMC"

# 轮次标签 = SLEAP 输出子目录名，也是 3D 输出文件前缀。
#   HPC 路线（推荐，与 K_1_x 完全同配置）：round8_260729_snn031_m6
#   本机路线（run_round_inference 用模型名作子目录）：round8_260723_strictmouth
$TAG  = "round8_260729_snn031_m6"

$SLP_DIR = "Z:\medaka\sleap_output\$EXP\$TAG"
$OUT_DIR = "Z:\medaka\reconstruction_3d\$EXP\$TAG"
```

---

## Stage 1 — SLEAP 推理（三个视角、全长）

目标产物：`$SLP_DIR\{top_cam,side1_cam,side2_cam}.slp`

两条路线二选一。**`max-instances 6` 是故意的**（多于 3 条鱼），多出的候选让 3D
阶段能救回被反光干扰的鱼；Stage 3 的 `--num-candidates 6` 依赖这一点。

### 路线 A：OIST saion HPC（推荐，~数小时，配置与已验证的 round8 完全一致）

在 saion 登录节点上（`Z:\medaka` = `/bucket/StephensU/medaka`）：

```bash
ssh saion.oist.jp
cd ~/medaka-mating-pipeline        # 需是与本仓库同步的版本

M=/bucket/StephensU/medaka/sleap_models/retrain_20260722_round8
export ROUND_TAG=round8_260729_snn031_m6
export TOP_CENTROID=$M/top/runs/round8_260723_strictmouth.centroid_n1382
export TOP_INSTANCE=$M/top/runs/round8_260723_strictmouth.centered_instance_n1382
export SIDE_CENTROID=$M/side/runs/round8_260723_strictmouth.centroid_n2764
export SIDE_INSTANCE=$M/side/runs/round8_260723_strictmouth.centered_instance_n2764
export SLEAP_MODULE=sleap-nn/0.3.1
export MODULEPATH_EXTRA=/apps/unit/StephensU/.modulefiles
export MAX_INSTANCES=6 BATCH_SIZE=12 NUM_CHUNKS=4
export PARTITION=largegpu TIME_LIMIT=12:00:00
export REQUIRE_ALL_VIEWS=1
export MEDAKA_WORK_ROOT=/work/StephensU/$USER/medaka_sleap
# MEDAKA_BUCKET_ROOT / MEDAKA_BUCKET_OUT 用默认值即可
#   (/bucket/StephensU/medaka/medaka_202512, /bucket/StephensU/medaka/sleap_output)

DRY_RUN=1 bash scripts/hpc/submit_experiment.sh 20260106_E_2_1_FCMC   # 先预览
bash scripts/hpc/submit_experiment.sh 20260106_E_2_1_FCMC             # 正式提交
squeue -u $USER
```

完成后 chunks 会被复制到
`/bucket/StephensU/medaka/sleap_output/20260106_E_2_1_FCMC/round8_260729_snn031_m6/`
（3 视角 × 4 个 `<view>.frames_<S>-<E>.slp`）。回到本机 Windows 合并：

```powershell
& $PY -m scripts.tracking.main $EXP --merge-chunks --sleap-subdir $TAG --paths-config $CFG
```

产物：`$SLP_DIR\<view>.slp` ×3，并更新原始数据目录下的 `experiment_metadata.json`。

### 路线 B：本机 GPU（RTX 5070 Ti，约 3.2 h/视角，三视角共约 10 h）

本机只用 torch 路径（`--mode track`，默认）。**不要用 `--mode predict`/TensorRT**——
在这块 Blackwell 卡上 TRT 导出会失败。

```powershell
$TAG = "round8_260723_strictmouth"            # 本机路线的标签，重设变量
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

- 直接输出合并好的 `<view>.slp`，**不需要** merge 步骤。
- sleap-nn 会缓冲 stdout，`<view>.inference.log` 在该视角跑完前只有表头；
  判断进度看 `<view>.slp` 是否出现，而不是看 log。
- 想先小试：加 `--frames 171800-176799`（只跑约 30 s），确认无误再跑全长。

### Stage 1 检查

```powershell
Get-ChildItem $SLP_DIR\*.slp | Select-Object Name, Length, LastWriteTime
# 叠加预测到视频上，目测 5 秒（每个视角各跑一次 --view top_cam / side1_cam / side2_cam）
& $PY -m scripts.sleap.visualize_predictions $EXP --paths-config $CFG `
    --view top_cam --slp $SLP_DIR\top_cam.slp --seconds 5 `
    --output $OUT_DIR\qc\top_cam.pred_5s.mp4
```

---

## Stage 2 — top_cam 2D 身份追踪

为 Stage 3 的绑定步骤提供稳定的 2D 身份锚点。使用默认的「可见节点质心」
（**不要**加 `--anchor-node`；用 mouth 锚定实测差约 2 倍）。

```powershell
& $PY -m scripts.tracking.track_simple_nn $EXP `
    --paths-config $CFG `
    --view top_cam `
    --slp    $SLP_DIR\top_cam.slp `
    --output $SLP_DIR\top_cam.tracks.csv `
    --max-distance 100 --max-tracks 3 --max-gap 500
```

- 产物：`$SLP_DIR\top_cam.tracks.csv`，约 1–2 分钟。
- 参考值：K_1_2 产生 15 个 track 片段，其中 3 个长片段覆盖大部分时长；
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
  round8 生产运行没有设置残差门限。
- `--max-jump-hard-cap-mm 80` 很关键：防止长 NaN 间隙之后把另一条鱼接成同一个 ID
  （间隙 N 帧时有效门限 = `min(8·N, 80)` mm）。

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

- `final_fish_id` 是 top_cam track 的编号（例如 K_1_2 为 11 / 9 / 8），只是稳定标签，
  **不代表** 雌/雄。F / M1 / M2 需要人工对照视频确认；确认后可以用
  `--identity 11=F --identity 9=M1 --identity 8=M2` 给图表加标签。

- 由于链接器有 8 mm/帧的相邻帧门限，>8 mm 的跳变应为 0；
  **≥7.5 mm 的近门限跳变**数量才是有用的风险信号（K_1_x 约 600–860 次）。
- 开头几千帧出现非有限值（鱼尚未全部被检测到）在 K_1_x 中也有，属正常。

### 4.3 Rerun 交互式目视检查

挑选 4.2 中尖峰附近、或两鱼接近/交叉的时间窗：

```powershell
& $PY -m scripts.visualization.rerun_viewer $EXP --paths-config $CFG `
    --csv          $OUT_DIR\${TAG}_3d_identified.csv `
    --best-json    $OUT_DIR\${TAG}_best.json `
    --pre-link-json $OUT_DIR\${TAG}_best_pre_link.json `
    --slp-dir      $SLP_DIR `
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
Z:\medaka\sleap_output\20260106_E_2_1_FCMC\<TAG>\
    top_cam.slp  side1_cam.slp  side2_cam.slp      # Stage 1
    top_cam.tracks.csv                              # Stage 2
Z:\medaka\reconstruction_3d\20260106_E_2_1_FCMC\<TAG>\
    <TAG>_3d.csv  <TAG>_3d_identified.csv  ...      # Stage 3
    qc\                                              # Stage 4
```

## 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 找不到 `.slp` / 路径以 `Z:\StephensU` 开头 | 没有带 `--paths-config configs\paths.ReiterU.toml` |
| 进度条显示 `3D candidates` | 漏了 `--num-candidates 6`，停止重跑 |
| `no run found ... matching '<tag>.centroid_n*'`（路线 B） | `--round-tag` 必须与模型目录名一致：`round8_260723_strictmouth` |
| 绑定比例很低 | 检查 `top_cam.tracks.csv` 是否和 `.slp` 属于同一 `$TAG`；必要时做 Stage 5 |
| 想复现旧的多项式标定结果 | `--calibration-backend polynomial --model-folder calibration_results\cali_20251217_manual\merged_rule_reassigned_beads_only\models`，残差门限改用 `--max-residual-px2` |
| 想检查行为指标 | `scripts/analysis/compute_*_metrics.py` 目前仍是占位脚本（`SystemExit("TODO")`），尚不能直接使用 |
