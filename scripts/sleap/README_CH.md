# SLEAP 命令行工具使用指南

本目录包含用于 SLEAP 模型推理、质量控制和训练的辅助脚本。本文档说明如何在 **Windows/Powershell** 环境下使用这些工具。

---

## 目录

- [环境准备](#环境准备)
- [1. 模型训练（Training）](#1-)
- [2. 质量控制（QC）](#2-质量控制qc)
- [3. 导出跟踪数据](#3-导出跟踪数据)
- [4. 训练模型](#4-训练模型)
---

## 环境准备

### 安装 SLEAP

确保你的环境中已安装 SLEAP。可使用 conda 或 pip：

```powershell
conda install -c conda-forge sleap
# 或
pip install "sleap[nn]"
```

### 验证安装

```powershell
sleap-track --help
```

任何参数名称（如-i， -o）和含义可以使用以下命令查看
```powershell
sleap-convert --help
```
查看
------
## 0. 标记和训练模型

```powershell
# 激活虚拟环境
c:/Users/Molec/Documents/GitHub/medaka-mating-pipeline/.venv/Scripts/Activate.ps1

# 启动 SLEAP 标注工具
sleap-label

# 从GUI界面开始训练
参考官方文档

```
------
## 1. 视频推理（Instance）

使用训练好的SLEAP 模型对视频进行预测。

### 基本命令（命令行）
#### 使用双模型（Top-down 流程：Centroid + Instance）预测

```powershell
sleap track -i "D:\videos_raw\20251225_K_1_2_FCMC\side1_cam\cam1_side.mp4" -m "D:\sleap_side_mouth\models\260103_174517.centroid.n=437" -m "D:\sleap_side_mouth\models\260103_233217.centered_instance.n=437" -o "D:\sleap_output\side_mouth_437_20251225_K_1_2_FCMC_side1.slp" -n 3 --frames 19920-29919
```

运行后会在 `<指定的输出路径>`生成 `<指定的文件名>.slp` 文件。

### 批量跟踪（自动匹配 Top 和 Side 视频）

使用 `batch_track.py` 脚本可以自动处理多个 top 视频，并自动查找对应的 side 视频进行跟踪。

**功能特点：**
- 自动使用 top 模型对 top 视频（PLAYER1）进行跟踪
- 自动检索对应的 side 视频（PLAYER2 和 PLAYER3）
- 自动使用 side 模型对 side 视频进行跟踪
- 支持同时处理多个 top 视频
- 支持 GPU 加速（默认使用 GPU）

**命名规则：**
- Top 视频包含 `PLAYER1`
- Side 视频包含 `PLAYER2` 和 `PLAYER3`（将 `PLAYER1` 替换为 `PLAYER2` 或 `PLAYER3`）
- 其他部分（时间戳、行为类型等）相同

**使用示例：**

```powershell
python C:\Users\Molec\Documents\GitHub\medaka-mating-pipeline\scripts\sleap\batch_track.py `
  --top-videos "D:\behavior_clips\20251225_K_1_2_FCMC_clips\20251225_K_1_2_FCMC_PLAYER1_f_Spawn_816.572-820.572.mp4" `
  --top-videos "D:\behavior_clips\20251225_K_1_2_FCMC_clips\20251225_K_1_2_FCMC_PLAYER1_m1_Sneak_804.705-812.873.mp4" `

  --top-models "D:\sleap_top_mouth\models\260104_142312.centroid.n=180" `
  --top-models "D:\sleap_top_mouth\models\260104_183212.centered_instance.n=180" `
  --side-models "D:\sleap_side_mouth\models\260103_174517.centroid.n=437" `
  --side-models "D:\sleap_side_mouth\models\260103_233217.centered_instance.n=437" `
  --output-dir "D:\sleap_output" `
  --max-instances 3 
```

**参数说明：**
- `--top-videos`: Top 视角视频路径（可以指定多个，每个视频用 `--top-videos` 指定一次）
- `--top-models`: Top 模型路径（**重要**：对于 top-down 跟踪流程，通常需要两个模型：centroid 和 instance，每个模型用 `--top-models` 指定一次）
- `--side-models`: Side 模型路径（**重要**：对于 top-down 跟踪流程，通常需要两个模型：centroid 和 instance，每个模型用 `--side-models` 指定一次）
- `--output-dir`: 输出目录，所有 .slp 文件将保存在此目录
- `--max-instances`: 最大实例数（默认：3）
- `--frames`: 可选的帧范围，格式如 `19920-29919`
- `--skip-missing-side`: 如果 side 视频不存在，跳过而不报错
- `--device`: 计算设备（默认：`gpu`）。可选值：`cpu`, `gpu`, `first-gpu`, `last-gpu`

**重要提示：模型数量要求**
- 对于 **top-down 跟踪流程**（推荐），需要提供 **两个模型**：
  1. **Centroid 模型**：用于检测个体位置（文件名通常包含 `centroid`）
  2. **Instance 模型**：用于预测关键点（文件名通常包含 `instance` 或 `centered_instance`）
- 如果只提供一个模型，请确保它是 **single instance 模型**（可以同时检测和预测）
- 如果只提供一个模型但遇到错误 "Ground truth data was not detected... Please load both models"，说明需要提供两个模型

**输出文件命名：**
- Top 视频输出：`<原视频名>_top.slp`
- Side1 视频输出（PLAYER2）：`<原视频名>_side1.slp`
- Side2 视频输出（PLAYER3）：`<原视频名>_side2.slp`

例如，对于 top 视频 `20251225_K_1_2_FCMC_PLAYER1_f_Spawn_816.572-820.572.mp4`，脚本会：
1. 使用 top 模型生成 `20251225_K_1_2_FCMC_PLAYER1_f_Spawn_816.572-820.572_top.slp`
2. 自动查找 `20251225_K_1_2_FCMC_PLAYER2_f_Spawn_816.572-820.572.mp4`，使用 side 模型生成 `20251225_K_1_2_FCMC_PLAYER2_f_Spawn_816.572-820.572_side1.slp`
3. 自动查找 `20251225_K_1_2_FCMC_PLAYER3_f_Spawn_816.572-820.572.mp4`，使用 side 模型生成 `20251225_K_1_2_FCMC_PLAYER3_f_Spawn_816.572-820.572_side2.slp`

**GPU 支持：**
- 默认使用 GPU 加速（`--device gpu`）
- 如果遇到 GPU 相关问题，可以指定 `--device cpu` 使用 CPU
- 其他选项：`first-gpu`（使用第一个 GPU）、`last-gpu`（使用最后一个 GPU）

**重要：安装支持 CUDA 的 PyTorch**

如果你的 PyTorch 是 CPU 版本（`torch.__version__` 显示 `+cpu`），需要安装 CUDA 版本：

```powershell
# 1. 激活虚拟环境
.venv\Scripts\Activate.ps1

# 2. 卸载 CPU 版本
pip uninstall torch torchvision torchaudio -y

# 3. 安装 CUDA 版本（根据你的 CUDA 版本选择）
# CUDA 12.x（推荐，适用于 CUDA 12.1-12.9）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 或者 CUDA 11.8（如果上面的不行）
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

如果 `CUDA available: True`，说明安装成功，可以使用 GPU 了。
---
## 2. 标签视频生成
使用 `.slp` 数据生成带有身体部位标签的视频

```powershell
sleap-render -o "D:\sleap_output\20251225_K_1_2_side1_mouth_437_over_1000.mp4" --fps 166 --frames 19920-19980  "D:\sleap_output\side_mouth_437_20251225_K_1_2_FCMC_side2.slp"
```
运行后会在 `<指定的输出路径>`生成 `<指定的视频名>.slp` 文件。


## 2. 数据导出
使用 `.slp` 数据导出为各种格式（CSV、H5、JSON 等）

**重要说明**：
- **h5 格式（`--format analysis` 或 `--format h5`）包含所有 tracks**，推荐用于后续分析
- **CSV 格式（`--format analysis.csv`）可能只导出第一个 track**，如果视频中有多个个体，请使用 `--all-tracks` 选项

### 使用 export_tracks.py 导出数据
#### 导出为 CSV 格式（包含所有 tracks）

```powershell
# 导出所有 tracks 到 CSV（使用 --all-tracks 选项）
python C:\Users\Molec\Documents\GitHub\medaka-mating-pipeline\scripts\sleap\export_tracks.py --input D:\sleap_output\side_mouth_437_20251225_K_1_2_FCMC_side1.slp --format analysis.csv --output D:\sleap_output\side_mouth_437_20251225_K_1_2_FCMC_side1.analysis.csv --all-tracks
```

#### 批量导出文件夹中所有 .slp 文件为 CSV

如果需要批量处理一个文件夹中的所有 .slp 文件，可以使用以下 PowerShell 命令：

**单个文件夹批量导出：**

```powershell
cd c:\Users\Molec\Documents\GitHub\medaka-mating-pipeline
Get-ChildItem "D:\sleap_output\side_mouth_593\*.slp" | ForEach-Object {
    $csvName = $_.BaseName + ".csv"
    $csvPath = Join-Path "D:\sleap_output\side_mouth_593" $csvName
    Write-Host "处理: $($_.Name) -> $csvName"
    python scripts/sleap/export_tracks.py --input $_.FullName --format analysis.csv --all-tracks --output $csvPath
}
```

**同时处理多个文件夹：**

```powershell
cd c:\Users\Molec\Documents\GitHub\medaka-mating-pipeline
@("D:\sleap_output\side_mouth_593", "D:\sleap_output\top_mouth_180") | ForEach-Object {
    $folder = $_
    Write-Host "`n处理文件夹: $folder"
    Get-ChildItem "$folder\*.slp" | ForEach-Object {
        $csvName = $_.BaseName + ".csv"
        $csvPath = Join-Path $folder $csvName
        Write-Host "  处理: $($_.Name) -> $csvName"
        python scripts/sleap/export_tracks.py --input $_.FullName --format analysis.csv --all-tracks --output $csvPath
    }
}
```

**说明：**
- 命令会遍历指定文件夹中的所有 .slp 文件
- 为每个 .slp 文件生成同名的 .csv 文件（扩展名改为 .csv）
- 使用 `--all-tracks` 选项确保导出所有轨道数据
- 所有 CSV 文件保存在与 .slp 文件相同的文件夹中
- 每个 CSV 文件包含列：`frame`, `track`, `node`, `x`, `y`, `score`

#### 导出为 H5 格式（推荐，包含所有 tracks）

**两种 H5 格式的区别：**
- **`--format analysis`**：生成 `analysis.h5` 文件，这是 SLEAP 的分析格式，包含结构化的分析数据（tracks, nodes, scores等），专门用于后续分析
- **`--format h5`**：生成 SLEAP 原生数据集的 H5 格式，这是 SLEAP 的原始数据格式

```powershell
# 导出为分析格式 h5
python C:\Users\Molec\Documents\GitHub\medaka-mating-pipeline\scripts\sleap\export_tracks.py --input D:\sleap_output\side_mouth_437_20251225_K_1_2_FCMC_side1.slp --format analysis --output D:\sleap_output\side_mouth_437_20251225_K_1_2_FCMC_side1.analysis.h5
```

### 支持的格式

`export_tracks.py` 支持以下格式：

| 格式 | 参数值 | 说明 |
|------|--------|------|
| **Analysis H5** | `analysis` | SLEAP 分析格式的 HDF5 文件（推荐用于分析） |
| **H5** | `h5` | SLEAP 原生数据集的 HDF5 格式 |
| **JSON** | `json` | JSON 格式 |
| **CSV** | `analysis.csv` | 默认可能只导出第一个 track。如需导出所有 tracks，必须使用 --all-tracks 选项 |
| **NIX** | `analysis.nix` | NIX 格式 |
| **SLEAP** | `slp` | SLEAP 原生格式 |


## 3. 质量控制（QC）

### 3.1 多视角缺失 vs 遮挡（推荐）
```powershell
python scripts/sleap/qc_tracks.py `
  --top-slp "D:\sleap_output\20251225_K_1_2_FCMC_PLAYER1_m2tf_Dash_496.253-500.434_top.slp" `
  --top-slp "D:\sleap_output\20251225_K_1_2_FCMC_PLAYER1_f_Spawn_816.572-820.572_top.slp" `
  --top-slp "D:\sleap_output\20251225_K_1_2_FCMC_PLAYER1_m1_Sneak_804.705-812.873_top.slp" `
  --top-slp "D:\sleap_output\20251225_K_1_2_FCMC_PLAYER1_m1tf_Courtship_290.729-296.331_top.slp" `
  --expected-count 3 `
  --output-dir "D:\sleap_qc" `
  --fps 166 `
  --events-csv "D:\boris_events\20251225_K_1_2_FCMC_events.csv"
  ```

**重要功能：**
- **自动时间对齐**：脚本会自动从文件名中提取视频片段的起始时间（例如 `496.253-500.434` 中的 `496.253`），用于将片段内的相对时间转换为原始视频的绝对时间。这样 BORIS 事件时间（如果是原始视频的绝对时间）就能正确对齐。
- **BORIS 事件标注**：使用 `--events-csv` 指定 BORIS 导出的行为事件 CSV 文件，脚本会在可视化图中高亮事件区间，并在逐帧结果中标记事件。

实验里通常会同时录制 top / side1 / side2 三个视角。  
如果只看单个视角，很难区分：
- 是**跟踪完全失败**（三个视角都找不到鱼），还是
- 只是某个视角被遮挡（比如鱼贴到玻璃边，只在另一个视角能看到）。

`qc_tracks.py` 会同时读取三个 `.slp`，在逐帧尺度上判断：
- **`true_missing`**：这一帧在三个视角里都少于期望个体数（真数据缺失）
- **`*_occlusion`**：某个视角少于期望个体数，但至少一个其他视角是完整的（更像"视角遮挡"）

用法示例（假设已经有三个 `.slp`：top/side1/side2）：

```powershell
python C:\Users\Molec\Documents\GitHub\medaka-mating-pipeline\scripts\sleap\qc_tracks.py `
  --top   D:\sleap_output\20251225_K_1_2_FCMC_top.slp `
  --side1 D:\sleap_output\20251225_K_1_2_FCMC_side1.slp `
  --side2 D:\sleap_output\20251225_K_1_2_FCMC_side2.slp `
  --expected-count 3 `
  --output-dir D:\sleap_qc_multiview


  python scripts/sleap/qc_tracks.py `  --top-slp "D:\sleap_output\20251225_K_1_2_FCMC_PLAYER1_m2tf_Dash_496.253-500.434_top.slp" `
  --top-slp "D:\sleap_output\20251225_K_1_2_FCMC_PLAYER1_f_Spawn_816.572-820.572_top.slp" `
  --top-slp "D:\sleap_output\20251225_K_1_2_FCMC_PLAYER1_m1_Sneak_804.705-812.873_top.slp" `
  --top-slp "D:\sleap_output\20251225_K_1_2_FCMC_PLAYER1_m1tf_Courtship_290.729-296.331_top.slp" `  --expected-count 3 `  --output-dir "D:\sleap_qc" `  --fps 166 `  --events-csv "D:\boris_events\20251225_K_1_2_FCMC.csv" `  --events-time-col "Time" `  --events-label-col "Behavior"
```

主要输出：
- `multiview_missing_occlusion.csv`：每帧一行，包含：
  - `frame`
  - `top_count, side1_count, side2_count`：三个视角各自检测到的个体数量
  - `top_is_missing, side1_is_missing, side2_is_missing`：该视角是否"少于期望个体数"
  - `true_missing`：三视角都缺 → 真正的数据缺失
  - `top_occlusion, side1_occlusion, side2_occlusion`：该视角缺，但至少一个其他视角完整 → 更合理解释为"遮挡"
  - `event_name, events`：如果提供了 `--events-csv`，会标记每一帧是否在行为事件区间内

可视化输出（同一目录下）：
- `multiview_missing_occlusion_overview.png`：  
  - Y 轴为视角（top/side1/side2），X 轴为时间（如果提供了 `--fps`）或帧号  
  - 颜色编码：  
    - 浅灰：这一视角在该帧个体数正常  
    - 橙色：该视角缺失，且其他视角也趋势性缺失（更像整体 tracking 较难的时间段）  
    - 蓝色：该视角缺失，但至少一个其他视角是满的（更像"视角遮挡"）
  - 如果提供了 `--events-csv` 和对应的列名/帧率，图中会：  
    - 用深红色高亮核心行为区间 `[start_s, end_s]`  
    - 用浅红色高亮缓冲区 `[start_s-margin, start_s)` 和 `(end_s, end_s+margin]`（margin 由 `--events-margin-s` 控制）
- `multiview_summary.csv`：汇总统计表，包含整体缺失帧数、各视角缺失帧数、事件内缺失帧数等

这个表可以帮助你回答：
- 真正 tracking 崩掉的时间段有多少帧？
- 某些视角是否经常遮挡，而另一个视角仍然可靠？  
  （例如：sideA 容易被水面反光干扰，但 top/sideB 还正常）

### 3.2 行为事件标注（BORIS 事件）

`qc_tracks.py` 支持通过 `--events-csv` 参数加载 BORIS 行为事件标注，在多视角分析中高亮行为事件区间。

脚本会自动尝试识别 BORIS 常见列名（例如 `Start (s)` / `Stop (s)` / `Time (s)` / `Image index` / `Behavior`）。如列名不同，可以用以下可选参数指定：
- `--events-time-col`：点事件时间列名（只有一个时间列时用，例如 `Image index` 表示帧号）
- `--events-start-col` / `--events-end-col`：区间事件起止时间列名
- `--events-label-col`：行为名称列名（用于在 `multiview_missing_occlusion.csv` 中标记事件名称）

**重要**：如果使用 `Image index` 列（帧号）作为事件时间，脚本会直接用帧号对齐，`--fps` 仅用于可视化时的时间轴标签转换。

### 3.3 进一步的模型评估

如果需要更细粒度、针对特定 body part 的量化评估，可以结合单独的 `evaluate_tracking.py` 脚本（例如基于人工标注的 GT 计算误差、召回率等），此处不展开。
---


---

## 相关文件

- `run_inference_slurm.sh`：集群环境下的推理脚本（Linux/Bash）
- `train_model.py`：训练模型脚本
- `qc_tracks.py`：多视角缺失 vs 遮挡质量控制脚本（支持 BORIS 行为事件标注）
- `export_tracks.py`：数据导出脚本

---

## 参考资源

- [SLEAP 官方文档](https://sleap.ai/)
- [SLEAP-NN 训练文档](https://nn.sleap.ai/dev/training/)
- [SLEAP 命令行工具](https://sleap.ai/guides/command-line.html)

