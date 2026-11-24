# Medaka 交配实验流程 — 命令合集（按流程顺序）

> 适用范围：1 雌 + 2 雄、三机位（top / side1 / side2，166 fps）实验。
> 本文件只收录仓库里**真实存在**的脚本命令；单个实验的详细记录请另写
> `<日期>_<实验>_*.md / .ipynb`（参考
> [20260106_E_2_1_FCMC_3d_reconstruction_workflow.md](20260106_E_2_1_FCMC_3d_reconstruction_workflow.md)）。
> 所有命令均为 **PowerShell**，在仓库根目录执行。

```
0 环境/路径 → 1 实验登记 → 2 标定 → 3 SLEAP 训练（主动学习循环）
→ 4 SLEAP 推理 → 5 top_cam 2D 追踪 → 6 3D 重建 + 绑定 ID → 7 QC
→ 8 外貌识别 F/M1/M2 → 9 人工 ID 审核 → 10 BORIS 行为标注与对照分析
→ 11 可视化/渲染
          ↑                                          │
          └──── 7/9 发现的失败帧 → 回到 3 重新标注训练 ┘
```

---

## 0. 环境与公共变量

### 0.1 环境（只需一次）

```powershell
uv sync --python 3.11          # 基础环境（含 sleap-io、rerun-sdk）
uv sync --extra gui            # 人工审核 / 标注 GUI（napari + PyQt5）
.venv\Scripts\python.exe -m pytest -q   # 确认环境正常
```

### 0.2 本机路径配置（每台电脑一次）

从 `configs\paths.example.toml` 复制出本机配置（如 `configs\paths.<用户名>.toml`），
填写 `data_root, medaka_data_dir, sleap_output_dir, sleap_models_dir,
reconstruction_output_dir, runs_dir, calibration_model_dir, boris_events_dir` 等。
**不要在脚本里写死 Z: / G: 路径。**

### 0.3 每次开新 PowerShell 窗口先粘贴

```powershell
$PY  = ".venv\Scripts\python.exe"
$CFG = "configs\paths.local.toml"            # 换成本机配置
$EXP = "20260106_E_2_1_FCMC"                  # 实验名
$TAG = "s14_bottomup_3node_reviewed"          # 模型轮次标签 = SLEAP 输出子目录 = 3D 输出前缀
$R   = "Z:\medaka"                            # 数据根目录（按本机修改）

$VID_DIR = "$R\medaka_202512\$EXP"
$SLP_DIR = "$R\sleap_output\$EXP\$TAG"
$OUT_DIR = "$R\reconstruction_3d\$EXP\$TAG"
$CALIB   = "calibration_results\fixed_refractive_192x192x186"   # 当前默认（--calibration-backend auto）
New-Item -ItemType Directory -Force $SLP_DIR, $OUT_DIR | Out-Null
```

---

## 1. 实验登记 / 元数据

录像目录结构：`<medaka_data_dir>\<EXP>\{top_cam\cam0_top.mp4, side1_cam\cam1_side.mp4, side2_cam\cam2_side.mp4}`
（采集脚本见 [scripts/acquisition/README.md](../../../scripts/acquisition/README.md)）。

```powershell
# 写入/更新每个实验目录下的 experiment_metadata.json
& $PY -m scripts.common.experiment_metadata --all --paths-config $CFG

# 汇总所有实验 → catalog.jsonl / catalog.csv（+ HTML），会附带 BORIS 事件统计
& $PY -m scripts.common.build_catalog --paths-config $CFG
```

---

## 2. 相机标定（换装置 / 移动相机后才需要）

当前默认标定：`calibration_results\fixed_refractive_192x192x186`（折射模型）；
旧多项式模型：`calibration_results\cali_20251217_manual\merged_rule_reassigned_beads_only\models`。

珠子标定流程（脚本均在 `scripts\calibration\`，参数见各脚本 `--help`）：

| 步骤 | 脚本 |
|---|---|
| 2.1 自动检测珠子中心 | `preannotate_colored_beads.py` |
| 2.2 GUI 人工修正 | `annotate_calibration_gui.py` |
| 2.3 按规则统一珠子 ID | `reassign_bead_ids_by_rule.py` |
| 2.4 训练标定模型（可多 way 合并） | `solve_intrinsics.py` |
| 2.5 QC | `qc_3d_error.py`, `qc_reprojection_2d.py`, `qc_volume_error.py` |

刚体物体精修（第二次标定拍摄时用）：

```powershell
& $PY -m scripts.calibration.refine_calibration_rigid_object `
    --rigid-object-geometry-csv    <rigid_object_geometry.csv> `
    --rigid-object-image-coords-csv <rigid_object_image_coords.csv> `
    --base-model-folder <当前模型目录> `
    --output-folder     <新模型目录> `
    --results-root      <QC 输出目录>
```

只有当留出刚体误差和下游 3D 残差 / 速度尖峰指标**都不变差**时才采用新模型。

---

## 3. SLEAP 训练（主动学习循环）

第一次可跳过（直接用现有模型）；之后每当 7/9 发现大量失败帧时走一轮。

### 3.1 挖掘难帧

```powershell
# A. 基于 2D 预测信号（反光静止、mouth/tail 翻转、视角间实例数不一致…）
& $PY -m scripts.sleap.mine_training_frames `
    --paths-config $CFG --experiments $EXP --target-count 300

# B. 基于 3D 失败（速度尖峰、近距离+尖峰 = 可能的 ID 交换），与已有标注合并
& $PY -m scripts.sleap.mine_3d_bad_frames `
    --existing-seed <旧 annotation_seed.pkg.slp> `
    --csv-3d  $OUT_DIR\${TAG}_3d.csv `
    --slp-dir $SLP_DIR `
    --experiment $EXP `
    --out <新 annotation_seed.pkg.slp>
```

### 3.2 人工标注

```powershell
# 方式 1：SLEAP 自带 GUI（按 suggestion 顺序逐帧改）
sleap-label <annotation_seed.pkg.slp>

# 方式 2：三视角同步标注 GUI（标定联动缩放、自动建议跨视角 fish ID）
& $PY -m scripts.sleap.pyqt_multiview_annotator `
    --in  <annotation_seed.pkg.slp> `
    --candidate-slp <prediction_candidates.slp> `
    --out <annotation_seed_v2.pkg.slp> `
    --paths-config $CFG

# 打包给没有 Z: 盘的同事标注（含标定、run.bat）
& $PY -m scripts.sleap.make_annotator_bundle `
    --seed <annotation_seed_full.pkg.slp> --paths-config $CFG `
    --out-dir <bundle 目录> --expand-missing-views
```

### 3.3 整理训练集（按视角拆分 + 生成配置）

```powershell
$ROUND = "$R\sleap_models\retrain_YYYYMMDD"

# 与历史各轮标注合并去重（新标注优先）
& $PY -m scripts.sleap.merge_cumulative_labels --help   # 查看 --round5-dir / --round6b-seed / --out-dir

# 拆成 top / side 两份 labels.pkg.slp，并按 configs\sleap\training\ 模板生成配置
& $PY -m scripts.sleap.prepare_retraining `
    --seed <annotation_seed.pkg.slp> `
    --out-dir $ROUND `
    --pretrained-from <上一轮 round 目录>     # 可选：微调
```

### 3.4 训练

```powershell
# 顺序训练（单 GPU）；top-down 需 centroid + centered_instance，bottom-up 用 bottomup
& $PY -m scripts.sleap.run_round3_training `
    --round-dir $ROUND --round-tag roundN_YYMMDD `
    --kinds centroid centered_instance --views top side `
    --centroid-anchor-policy strict
#   --kinds bottomup     → 3-node bottom-up 模型（当前生产模型的类型）
#   --dry-run            → 只打印命令

# 或者直接调 sleap-nn（缓存放本地 SSD，不要放 Z:）
sleap-nn train configs/sleap/training/centroid.yaml `
    "data_config.train_labels_path=[$ROUND/top/labels.pkg.slp]" `
    data_config.cache_img_path=C:/sleap_cache/roundN_top_centroid `
    trainer_config.ckpt_dir=$ROUND/top/runs `
    trainer_config.run_name=centroid
#   中断续训：追加 trainer_config.resume_ckpt_path=<run>/last.ckpt
```

> 判断新模型好坏要看**下游 3D 指标**（7.2 的速度尖峰、绑定比例），不只看 SLEAP loss。

---

## 4. SLEAP 推理

产物必须是 `$SLP_DIR\{top_cam,side1_cam,side2_cam}.slp`。
**`max_instances 6` 是故意的**（多于 3 条鱼，供 3D 阶段挑选）。

### 4.1 本机 GPU（当前 3-node bottom-up 模型）

```powershell
$M = "$R\sleap_models\retrain_20260810_3node_reviewed"
$MODEL = @{
    top_cam   = "$M\top\s14.bottomup_3node.reviewed_top_f24_ms64_paf8"
    side1_cam = "$M\side\s14.bottomup_3node.reviewed_side_f24_ms64_paf8"
    side2_cam = "$M\side\s14.bottomup_3node.reviewed_side_f24_ms64_paf8"
}
$VIDEO = @{
    top_cam   = "$VID_DIR\top_cam\cam0_top.mp4"
    side1_cam = "$VID_DIR\side1_cam\cam1_side.mp4"
    side2_cam = "$VID_DIR\side2_cam\cam2_side.mp4"
}
foreach ($view in "top_cam", "side1_cam", "side2_cam") {
    if (Test-Path "$SLP_DIR\$view.slp") { "skip $view"; continue }
    & $PY -m sleap_nn.cli track `
        --data_path $VIDEO[$view] --model_paths $MODEL[$view] `
        --output_path "$SLP_DIR\$view.slp" `
        --device cuda --batch_size 4 --max_instances 6 `
        *> "$SLP_DIR\$view.inference.log"
}
# 测速：追加 --frames 171800-174799，输出到 bench_<view>.slp
```

- 一次只跑一个视角；RTX 5070 Ti 上 batch 4 ≈ 33 FPS（约 2.9 h/视角），**batch 12 会显存溢出**。
- 不要用 TensorRT / `--mode predict`（Blackwell 卡导出失败）。

### 4.2 本机 GPU（top-down 模型，如 round8）

```powershell
& $PY -m scripts.sleap.run_round_inference `
    --round-dir   $R\sleap_models\retrain_20260722_round8 `
    --round-tag   round8_260723_strictmouth `
    --experiment  $EXP `
    --data-root   $R\medaka_202512 `
    --output-root $R\sleap_output `
    --predict-batch-size 12 --max-instances 6 --continue-on-fail
```

### 4.3 HPC（OIST saion）

```bash
# 在 saion 登录节点（bash），详见 scripts/hpc/README.md
DRY_RUN=1 bash scripts/hpc/submit_experiment.sh <EXP>    # 预览
bash scripts/hpc/submit_experiment.sh <EXP>              # 单个实验
bash scripts/hpc/submit_batch.sh --skip-existing         # 批量
```

回到本机合并 chunk：

```powershell
& $PY -m scripts.tracking.main --merge-chunks $EXP --paths-config $CFG
```

### 4.4 推理 QC

```powershell
# 实例数分布 / 骨架
& $PY -c "import sys, collections, sleap_io as sio; l = sio.load_slp(sys.argv[1]); n = [len(f.instances) for f in l.labeled_frames]; print(l.skeletons[0].node_names, len(n), sorted(collections.Counter(n).items()))" $SLP_DIR\top_cam.slp

# 叠加预测到视频（每个视角看 5 秒）
& $PY -m scripts.sleap.visualize_predictions $EXP --paths-config $CFG `
    --view top_cam --slp $SLP_DIR\top_cam.slp --seconds 5 `
    --output $OUT_DIR\qc\top_cam.pred_5s.mp4
```

---

## 5. top_cam 2D 身份追踪（3D 绑定用的 ID 锚点）

```powershell
& $PY -m scripts.tracking.track_simple_nn $EXP `
    --paths-config $CFG --view top_cam `
    --slp    $SLP_DIR\top_cam.slp `
    --output $SLP_DIR\top_cam.tracks.csv `
    --max-distance 100 --max-tracks 3 --max-gap 500
```

不要加 `--anchor-node`（默认可见节点质心更好）。

---

## 6. 3D 重建 + 绑定 top_cam ID

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
    --bind-top-id --top-tracks-csv $SLP_DIR\top_cam.tracks.csv `
    --output-dir $OUT_DIR --output-tag $TAG
```

- 进度条必须是 **`3D best-assign`**；看到 `3D candidates` = 漏了 `--num-candidates 6`，立刻停。
- 折射标定（auto）下**不要**加旧的 `--max-residual-per-fish 5000`（那是多项式模型的 px² 阈值）。
- 主要产物：`${TAG}_3d_identified.csv`（含 `final_fish_id`，下游都用它）、
  `${TAG}_best.json`、`${TAG}_best_pre_link.json`。

---

## 7. QC

```powershell
# 7.1 看终端 [bind] 日志：top_cam_2d 绑定比例应 ~95% 以上

# 7.2 速度尖峰 / ID 一致性
& $PY -m scripts.tracking.evaluate_3d_id_consistency `
    --csv $OUT_DIR\${TAG}_3d_identified.csv --id-col final_fish_id `
    --plot-dir $OUT_DIR\qc --out-csv $OUT_DIR\qc\id_consistency_3d.csv

& $PY -m scripts.visualization.plot_xyz_speed_qc $OUT_DIR\${TAG}_3d_identified.csv `
    --id-column final_fish_id --output-dir $OUT_DIR\qc

# 7.3 Rerun 交互式 3D + 三视角目视检查（挑尖峰 / 交叉时间窗）
& $PY -m scripts.visualization.rerun_viewer $EXP --paths-config $CFG `
    --csv $OUT_DIR\${TAG}_3d_identified.csv `
    --best-json $OUT_DIR\${TAG}_best.json `
    --pre-link-json $OUT_DIR\${TAG}_best_pre_link.json `
    --slp-dir $SLP_DIR `
    --start-frame 171800 --end-frame 176799
```

`final_fish_id` 只是稳定编号，**不代表雌/雄** → 进入第 8 步。

---

## 8. 外貌识别 ID（F / M1 / M2）

### 8.1 雌鱼锚定（整段 3D 轨迹标 F/M）

阶段：slot-points → detectron2 掩膜 → link → pool → 雌鱼分类器，
每阶段已有产物会自动跳过（`--force` 全部重跑，`--stop-after <stage>` 中途停）。

```powershell
& $PY -m sam2_id_experiment.scripts.40_female_anchor_pipeline $EXP `
    --paths-config $CFG `
    --round-tag $TAG `
    --identified-3d-csv $OUT_DIR\${TAG}_3d_identified.csv `
    --work-dir runs\female_anchor\$EXP\$TAG `
    --calibration-model-dir $CALIB `
    --views side1_cam side2_cam top_cam
# 产物：<tag>_3d_identified_sexed.csv（每行 sex / female_prob）
```

### 8.2 逐帧分辨 F / M1 / M2（需要雌、雄分类器）

```powershell
$SEGID = "$R\runs\udmt_validation\segid\K_1_2"     # 已训练好的分类器位置（按实际修改）
& $PY -m sam2_id_experiment.scripts.45_per_frame_resolve $EXP `
    --paths-config $CFG `
    --identified-3d-csv $OUT_DIR\${TAG}_3d_identified.csv `
    --masks-dir $R\detectron2_output\$EXP `
    --slot-points-dir runs\female_anchor\$EXP\$TAG\slot_points `
    --views side1_cam side2_cam `
    --female-model $SEGID\model_female_clean\model.pt `
    --male-model   $SEGID\model_male_clean\model.pt `
    --top-model    $SEGID\model_top_male\model.pt --top-view top_cam `
    --stride 10 `
    --out-csv $OUT_DIR\${TAG}_3d_individual_id.csv
# 产物：individual_id ∈ {F,M1,M2} + 置信度；.report.json 里看 frames_one_of_each
# 视频在 SMB 上解码慢 → 先拷到本地，用 --video top_cam=D:\...\cam0_top.mp4 等覆盖
```

### 8.3 去除瞬时 ID 跳变（运动连续性 debounce）

```powershell
& $PY runs\dual_line\debounce_track.py `
    --in-csv  <..._individual_id_interp.csv> `
    --out-csv <..._individual_id_interp_debounced.csv>
```

> ⚠️ 该脚本和第 10 步分析脚本读取的是带 `id_final` 列的插值轨迹
> （`frame, id_final, X, Y, Z`）。45 的输出列名是 `individual_id`；
> 目前仓库里没有正式的「45 输出 → interp 轨迹」脚本，需要先改列名/插值。

### 8.4 （需要时）训练新的外貌分类器

新实验的鱼与已有分类器不匹配时（数据集来自 8.1 的 pooled crops）：

```powershell
# 雌鱼分类器
& $PY -m sam2_id_experiment.scripts.37_train_female_classifier `
    --dataset <pooled_sides_full> --female-gid 1 --out-dir <model_female> --epochs 20
# 三鱼 / 仅雄鱼分类器
& $PY -m sam2_id_experiment.scripts.39_train_identity_classifier `
    --dataset <pooled_sides_full> --mode male --female-gid 1 --split window --out-dir <id_classifier_males>
# 用嵌入纠正标签后的干净数据集
& $PY -m sam2_id_experiment.scripts.43_build_corrected_segid_dataset `
    --dataset <pooled_sides_full> --embedder <model_resnet_full\model.pt> `
    --out-dir <pooled_sides_corrected> --anchor-female-gid 1
# 把干净雌鱼模型应用到目标实验
& $PY -m sam2_id_experiment.scripts.44_apply_clean_female_anchor `
    --target-dataset runs\female_anchor\$EXP\$TAG\pooled `
    --model <model_female_clean\model.pt> --embedder <model_resnet_full\model.pt> `
    --out-dir runs\female_anchor\$EXP\$TAG\female_anchor_clean
```

### 8.5 （可选）分割辅助修复 top_cam ID 碎片 → 重新绑定 3D

```powershell
& $PY -m sam2_id_experiment.scripts.24_seed_prompts_from_3d `
    --identified-3d-csv  $OUT_DIR\${TAG}_3d_identified.csv `
    --best-pre-link-json $OUT_DIR\${TAG}_best_pre_link.json `
    --paths-config $CFG `
    --output-dir runs\seg_id\$EXP\$TAG\seed_from_3d
# → 用 sam2_seed_prompts_from_3d.yaml 跑 sam2_id_experiment\scripts\01_*–23_*
# → 20_export_segmentation_top_tracks.py 导出 top_cam.seg_identity_tracks.csv
# → 重跑第 6 步，--top-tracks-csv 换成该文件，--output-tag 换成 "${TAG}_segid"
```

---

## 9. 人工 ID 审核

所有 GUI 结果写入可续写的 CSV（关掉重开会接着上次）。需要 `uv sync --extra gui`。

### 9.1 交叉事件审核（最常用：确认交叉前后 ID 是否交换）

```powershell
& $PY -m scripts.analysis.crossing_review_gui `
    --csv-3d $OUT_DIR\${TAG}_3d_identified.csv `
    --slp-dir $SLP_DIR `
    --model-folder $CALIB `
    --video-top   $VID_DIR\top_cam\cam0_top.mp4 `
    --video-side1 $VID_DIR\side1_cam\cam1_side.mp4 `
    --video-side2 $VID_DIR\side2_cam\cam2_side.mp4 `
    --out-csv $OUT_DIR\${TAG}_crossing_review.csv `
    --n-events 40
# 键盘：←/→ 上/下一个交叉，空格 播放，[ / ] 逐帧；记录雌鱼 ID / 是否交换 / 严重度
```

### 9.2 几何交换审核 → 导出干净 tracklet

```powershell
$GSR = "runs\qc\$EXP\geometry_swap_review"
& $PY -m scripts.analysis.geometry_swap_review $EXP `
    --geometry-csv $OUT_DIR\${TAG}_3d.csv `
    --video-dir $VID_DIR --slp-dir $SLP_DIR `
    --calibration-model-dir $CALIB `
    --events-csv    $GSR\geometry_review_events.csv `
    --candidate-csv $GSR\geometry_review_candidates.csv `
    --audit-jsonl   $GSR\geometry_review_audit.jsonl

& $PY -m scripts.tracking.export_geometry_swap_tracklets `
    --geometry-csv $OUT_DIR\${TAG}_3d.csv `
    --events-csv $GSR\geometry_review_events.csv `
    --output-dir $GSR\clean_tracklets `
    --calibration-model-dir $CALIB --write-constraints
```

### 9.3 雌鱼性别 GT 标注（外貌分类器的训练/验证真值）

```powershell
& $PY -m scripts.analysis.female_sex_gt_annotator $EXP `
    --paths-config $CFG `
    --video-dir $VID_DIR --slp-dir $SLP_DIR `
    --geometry-csv $OUT_DIR\${TAG}_3d.csv `
    --reconstruction-provenance-json $OUT_DIR\${TAG}_reconstruction_provenance.json `
    --calibration-model-dir $CALIB --calibration-backend refractive `
    --build-candidates --candidate-target-count 300
# 第二次起去掉 --build-candidates，自动续用候选队列和已有标注
```

### 9.4 多方法对比审核 / 接触事件标注

```powershell
# 多种 ID 方法结果叠加对比（manifest 格式见 configs\review\example.toml）
& $PY -m scripts.analysis.review_viewer --manifest <trial.toml>

# 接触事件（雌-雄接触片段）标注；先 --preflight 校验输入
& $PY -m scripts.analysis.contact_episode_annotator $EXP `
    --paths-config $CFG --sleap-subdir $TAG `
    --tracking-csv main=$OUT_DIR\${TAG}_3d_identified.csv --detector-method main `
    --anchor-mode visible-node-centroid --annotator <你的名字> --preflight
```

---

## 10. BORIS 行为标注与对照分析

### 10.1 在 BORIS 中标注

- 媒体：top_cam 视频（`Image index` 列 = top_cam 帧号，与 3D CSV 的 `frame` 直接对齐）。
- Subject 词表：`f, m1, m2`（单鱼）、`m1tf, m2tf`（雄→雌）、`m1tm2, m2tm1`（雄→雄）、`m1m2`（双雄互动）。
- 行为类别：Courtship / Competitive / Mating / Other。
- 导出：**Export events → tabular CSV**，文件名 `<EXP>.csv`，放到 `paths.toml` 里的
  `boris_events_dir`（之后重跑第 1 步的 `build_catalog` 会统计事件数）。

### 10.2 三视角视频 + 行为时间轴

```powershell
$BORIS = "<boris_events_dir>\$EXP.csv"
& $PY -m scripts.analysis.render_behavior_timeline_video `
    --video-top $VID_DIR\top_cam\cam0_top.mp4 `
    --video-side1 $VID_DIR\side1_cam\cam1_side.mp4 `
    --video-side2 $VID_DIR\side2_cam\cam2_side.mp4 `
    --boris-csv $BORIS `
    --output $OUT_DIR\qc\behavior_timeline.mp4 `
    --start-frame 130000 --end-frame 140000
```

### 10.3 BORIS 与 3D 轨迹对照（距离/速度是否支持标注的行为）

```powershell
# 默认参数是 K_1_2 的，务必全部显式传入
& $PY scripts\analysis\compare_boris_trajectory.py `
    --boris $BORIS `
    --tracks $OUT_DIR\${TAG}_3d_identified.csv `
    --female-slot ID_2 --male-slots ID_1 ID_3 `
    --out-dir $OUT_DIR\behavior_comparison --label $EXP
```

### 10.4 配偶选择 / 求偶 / 攻击指标（需要 8.3 的 F/M1/M2 轨迹）

```powershell
$TRACK = "<..._individual_id_interp_debounced.csv>"   # 列：frame, id_final, X, Y, Z

& $PY scripts\analysis\analyze_mate_choice.py `
    --track $TRACK --boris $BORIS --spawn-frame <产卵帧> `
    --out-dir $OUT_DIR\analysis

& $PY -m scripts.analysis.compute_courtship_metrics `
    --tracks $TRACK --output $OUT_DIR\analysis\courtship_metrics.json --spawn-frame <产卵帧>
& $PY -m scripts.analysis.compute_aggression_metrics `
    --tracks $TRACK --output $OUT_DIR\analysis\aggression_metrics.json
```

产卵帧可从 BORIS 的 Mating 事件读出（帧号 = 秒 × 166）。

---

## 11. 可视化 / 渲染（汇报用）

```powershell
# 三视角 + 3D 四宫格视频
& $PY -m scripts.visualization.render_4panel_video $EXP `
    --slp-dir $SLP_DIR --best-json $OUT_DIR\${TAG}_best.json `
    --model-folder $CALIB --video-dir $VID_DIR `
    --out $OUT_DIR\qc\4panel.mp4 --start-frame 171800 --end-frame 176799

# Rerun 窗口保存为离线文件（7.3 的命令后加）
#   --save $OUT_DIR\qc\window.rrd
```

其他渲染脚本：`scripts\analysis\render_*_video.py`、`scripts\visualization\render_*`（参数见 `--help`）。

---

## 附录：常见问题

| 现象 | 处理 |
|---|---|
| 路径以 `Z:\StephensU` 开头 / 找不到文件 | 没传本机 `--paths-config` |
| `run_round_inference` 报 `no run found ... centroid_n*` | bottom-up 模型不能用它，改用 4.1 的 `sleap_nn.cli track` |
| CUDA OOM / 推理极慢 | 同时跑了多个视角或 batch 太大 → 一次一个视角，batch 4 |
| 进度条 `3D candidates` | 漏了 `--num-candidates 6` |
| 绑定比例低 | `top_cam.tracks.csv` 与 `.slp` 不是同一个 `$TAG`；或做 8.5 |
| 任务管理器里两个 python 进程 | Windows venv 启动器的正常现象，不要杀 |
| 某脚本参数不确定 | `& $PY -m <module> --help` |
