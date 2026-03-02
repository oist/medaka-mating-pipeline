# 从 SLP 导出 CSV 之后的代码逻辑说明（含 3D 重建）

本文档详细描述：在「生成 / 使用 SLEAP 导出的 CSV」之后，到「3D 轨迹重建」为止的完整代码逻辑；并说明**异常值的定义与处理**、以及**为何可能重建不了 3D 轨迹**。  
**中日双语。**

---

## 一、整体流程概览

### 1.1 数据流（从 CSV 到 3D）

```
SLEAP 导出 CSV (top_mouth_180 / side_mouth_593 下的 *_PLAYER1/2/3.csv)
        ↓
[04_tracking / scripts/tracking]
  → 加载、解析 → ID 分配 → 异常值处理 → 镜像过滤 → （可选）2D 插值
        ↓
*_processed.csv（三视角：player1/2/3）保存到 processed/
        ↓
[05/06/07/08 与 scripts/reconstruction_3d]
  → load_2d 读取 *_processed.csv → points_by_frame
  → 三角化（有校准 compute_candidates / 无校准 compute_candidates_no_calib）
  → select_best_all_frames 每帧选最优排列
  → 3D 观测 / Track 关联 / N 帧确认（08）→ 最终 3D 轨迹
```

- **04 之前**：SLEAP 从 `.slp` 导出每视角的 CSV（含 frame, x, y, track 等）。  
- **04**：读入这些 CSV，做**跨帧 ID 分配**、**异常值处理**、**镜像过滤**，输出 `*_player1/2/3_processed.csv`。  
- **05/06/07/08**：只使用 **processed CSV**，不再直接读 SLP；通过 `run_load_2d` 得到 `points_by_frame`，再三角化、选最优排列、做 3D 跟踪。

---

## 二、异常值（Outlier）是什么？如何检测与处理？

### 2.1 异常值的定义（中文）

在本 pipeline 里，**异常值**指「在单条 2D 轨迹（同一相机、同一 `fish_id`）上，明显不可能由同一条鱼在相邻帧间产生的点」。  
具体有两类：

1. **单点大跳（按帧间隔缩放）**  
   - 对同一条轨迹按 `frame` 排序，取「上一有效点」到「当前点」的位移 `dist` 和帧间隔 `gap = frame_curr - frame_prev`。  
   - 若 **dist > max_jump_per_frame × gap**，则判定**当前点**为异常。  
   - 例如：`max_jump_per_frame=100` 时，相邻 1 帧最多允许 100 像素，2 帧 200 像素，依此类推。  
   - 含义：同一条鱼在短时间内不可能出现这么大的位移，多半是 ID 错配或检测错误。

2. **整段“绕一圈又回到原地”的错误段（loop segment）**  
   - 若存在两个端点 (t0, t1)，帧间隔 ≤ max_loop_frames（如 10 帧），且两端点距离 ≤ return_dist_thresh（如 30 像素），即「回到原地」。  
   - 若 (t0, t1) 之间**所有中间点**到任一端点的距离都 > excursion_dist_thresh（如 200 像素），则这段中间点被视为「错误段」，全部标为异常。  
   - 典型情况：423 帧和 429 帧几乎同位置，中间 424–428 却整段在另一处，说明中间段是错跟到别的目标或镜像。

此外还有**镜像过滤**（不算“异常值”模块，但会再删一批点）：

- 同一帧、同一视角内，若两点 y 相近且 x 关于某轴对称（个体与玻璃/水面镜像），保留与**上一帧该 fish_id 位置更近**的点，另一者置为 NaN，并写入 `*_mirror_outliers.csv`。

### 2.2 異常値の定義（日本語）

本パイプラインでは**異常値**とは、「同一の 2D 軌跡（同一カメラ・同一 `fish_id`）上で、隣接フレーム間で同一の魚ではあり得ないとみなされる点」です。  
具体的に次の 2 種類があります。

1. **単点の大きな飛び（フレーム間隔に比例）**  
   - 同一軌跡を `frame` でソートし、「直前の有効点」から「現在点」までの移動距離 `dist` とフレーム間隔 `gap = frame_curr - frame_prev` を計算。  
   - **dist > max_jump_per_frame × gap** のとき、**現在点**を異常と判定。  
   - 例：`max_jump_per_frame=100` なら、1 フレーム差で 100 ピクセル、2 フレームで 200 ピクセルまで許容。  
   - 意味：同一個体が短時間にこれだけ動くのは不可能であり、ID の取り違えや検出誤りと考えられる。

2. **「一度離れてまた戻る」誤った区間（ループ区間）**  
   - 二つの端点 (t0, t1) で、フレーム間隔 ≤ max_loop_frames（例 10）、端点間距離 ≤ return_dist_thresh（例 30 ピクセル）なら「元の位置に戻った」とみなす。  
   - (t0, t1) の**全ての中間点**がどちらの端点からも excursion_dist_thresh（例 200 ピクセル）より遠い場合、その区間全体を異常とする。  
   - 例：423 と 429 フレームはほぼ同位置なのに、424–428 が別の場所にまとまっているようなケース。

加えて**ミラー除去**（異常値モジュールとは別だが、さらに点を NaN にする処理）：

- 同一フレーム・同一視点で、y が近く x がある軸で対称な 2 点（実体とガラス/水面の鏡像）のうち、**前フレームのその fish_id の位置に近い方**を残し、もう一方を NaN にし、`*_mirror_outliers.csv` に記録する。

### 2.3 代码位置与调用关系

- **异常值检测与置 NaN**：`scripts/tracking/outlier_detection.py`  
  - `detect_outliers_frame_gap_scaled()`：单点大跳（按帧间隔缩放）。  
  - `detect_loop_segment_outliers()`：整段错误段。  
  - `_remove_outliers_for_entity()`：对单条鱼先做上述两种检测，再将该实体上的异常点 (x,y) 置为 NaN。  
  - `process_all_fish_outliers()`：对所有视角、所有 fish_id 调用上述逻辑，并可选写出 `*_outliers.csv`、`*_outliers_summary.csv`。  
- **镜像过滤**：同文件中的 `filter_mirror_in_view()`、`process_mirror_filter()`。  
- **主流程**：`scripts/tracking/process_tracking.py` 的 `process_single_set()` 中，先 ID 分配，再调用 `process_all_fish_outliers()`，再 `process_mirror_filter()`，最后（可选）缺失值填充，并保存 `*_processed.csv`。

---

## 三、从 processed CSV 到 3D 的逐步逻辑

### 3.1 加载 2D：run_load_2d（load_2d.py）

- **输入**：`processed_dir`、`set_key`；可选 `use_fish_id`。  
- **行为**：  
  - 读取三视角 `{set_key}_player1_processed.csv`、`player2`、`player3`。  
  - 每帧、每相机得到最多 3 个点，组成 `points_by_frame[frame][cam]`，cam ∈ {cam0, cam1, cam2}，对应 player1/2/3。  
- **取点顺序**：  
  - **use_fish_id=True**：按 `fish_id` 的 a, b, c 顺序取，保证三视角间「第 1/2/3 个点」对应同一 ID。  
  - **use_fish_id=False**：按 score 降序、无 score 则按 x 升序取前 3 个点，**不保证跨视角 ID 对应**。  
- **重要**：08 中 `get_observations_per_frame` 调用的是 `run_load_2d(..., use_fish_id=False)`，因此三视角之间点的对应关系仅依赖后续的「排列枚举 + 三角化代价」来隐式确定，若 2D 缺失多或排列歧义大，容易导致每帧选出的 a/b/c 大量为 NaN。

### 3.2 三角化候选：compute_candidates / compute_candidates_no_calib

- **有校准**（05/07，提供 `model_folder`）：  
  - 对每帧枚举 cam1、cam2 的索引排列，对每种排列用三视角 2D 点做线性三角化（DLT），得到 3 个 3D 点 (a,b,c)，并计算三视图配准误差作为 cost。  
  - 若某组合中任一视角的 2D 为 NaN，则该 3D 点为 NaN，且 cost 贡献为 inf；若 3D 超出 xyz_bounds 或含 nan/inf，同样该点记为 NaN、cost 设为 inf。  
- **无校准**（06/08 中 MODEL_FOLDER=None）：  
  - `compute_candidates_no_calib`：约定 top=(X,Y)、side1=(X,Z)、side2=(Y,Z)，用对应坐标取平均（缺视角则该分量为 NaN），得到 3D；cost 为同一坐标在两视角间不一致时的平方和。  
  - 同样对每帧枚举排列，得到多组 {a,b,c} 及 cost。

两种方式都会产生 **candidates_by_frame[frame]**：每帧一组候选（每组含 groups={a,b,c} 与 cost）。

### 3.3 每帧选最优排列：select_best_all_frames（select_best.py）

- 对每帧在所有候选中，优先选「groups 内 a,b,c 均在 xyz_bounds 内」且 **cost 最小**的候选；若无在范围内的，则只按 cost 最小选。  
- 得到 **best_by_frame[frame]**：每帧一个最优 {groups, perm_cam1, perm_cam2}。  
- 若某帧所有候选的 a,b,c 都含 NaN 或越界，则「最优」里 groups 仍可能全是 NaN。

### 3.4 08 专用：从 best_by_frame 到「无 ID 的 3D 观测」

- `get_observations_per_frame()`：  
  - 对每帧从 `best_by_frame[frame]["groups"]` 取 a, b, c 三个 3D 点。  
  - **仅当**某点 `len(p)==3` 且 `np.any(np.isfinite(p))` 时才加入该帧的观测列表。  
  - 若某帧的 a,b,c 全部为 NaN 或不全为有限值，则该帧 **observations_by_frame[frame]** 为形状 **(0, 3)**，即**该帧 0 个观测**。  
- 因此：若多数帧的 best 候选里 a,b,c 都是 NaN（例如 2D 缺失多、无校准下几何约束弱），就会出现「有观测的帧数」很少甚至为 0。

### 3.5 08：Track 与 N 帧确认

- **Step 2–5**：用常速度模型预测当前帧 3D 位置 → 构建 Track–Observation 代价矩阵（含 gating：距离 > GATE_MAX_DIST 则禁止匹配）→ Hungarian 一对一匹配 → 用观测更新 Track 状态；仅当**连续 N 帧**都匹配成功才将 Track 标记为 confirmed。  
- **首帧初始化**：在「尚未初始化」时，遇到**第一个观测数 M>0 的帧**，按 3D 点的 X 坐标排序，前 max_tracks 个观测分别赋给 ID_1, ID_2, ID_3，并从这里开始做预测与匹配。  
- 若**从未出现 M>0 的帧**（即所有帧的 observations 都是 (0,3)），则 `any_initialized` 一直为 False，所有帧的 ID_1/2/3 都会被填成 NaN，**最终 3D 轨迹完全没有有效点**（你当前看到的 ID_1/2/3 有效点数均为 0 正是这种情形）。

---

## 四、为什么重建不了 3D 轨迹？可能原因归纳

### 4.1 中文

1. **2D 缺失过多**  
   - 04 中 `skip_2d_interpolation=True` 时，异常值直接置 NaN、不做 2D 插值。  
   - 若异常值或镜像过滤删掉很多点，processed CSV 里大量 (frame, fish_id) 的 x,y 为 NaN。  
   - 三角化时「任一视角为 NaN → 该 3D 点为 NaN」，导致 best_by_frame 里很多帧的 a,b,c 全是或多为 NaN，进而 observations_by_frame 大量为 (0,3)。

2. **use_fish_id=False 导致对应关系弱**  
   - 08 用 score/x 顺序取点，三视角间未必是同一只鱼。  
   - 无校准下仅靠 top/side 坐标平均，对应错误会直接变成 3D 噪声或 NaN，select_best 选出的「最优」仍可能经常无效。

3. **首帧或多数帧无有效观测**  
   - 若第一帧或绝大多数帧的 observations 都是空的，则要么无法初始化 Track，要么初始化后几乎没有观测可匹配，Track 状态几乎全是 NaN，输出 3D 轨迹全空。

4. **无校准 + 坐标范围**  
   - 无校准时坐标为像素量级，若 xyz_bounds 设得与数据尺度不符（如 0–200 mm），可能大量点被判越界，best 候选里有效点更少。

5. **SET_KEY / 文件不一致**  
   - 若 08 用的 SET_KEY 与 04 生成 processed CSV 的 set_key 不一致，或 processed 目录/文件名不对，会读到错误或空数据，同样得不到 3D。

### 4.2 日本語

1. **2D の欠損が多すぎる**  
   - 04 で `skip_2d_interpolation=True` の場合、異常値を NaN にするだけで 2D 補間をしない。  
   - 異常値・ミラー除去で点が多く消えると、processed CSV で (frame, fish_id) の x,y が NaN だらけになる。  
   - 三角化では「いずれかの視点が NaN → その 3D 点は NaN」となり、best_by_frame で多くのフレームの a,b,c がすべて NaN に近くなり、observations_by_frame が多数 (0,3) になる。

2. **use_fish_id=False による対応のずれ**  
   - 08 では score/x 順で点を取るため、3 視点で「同じ個体」が揃わない。  
   - 無校正では top/side の座標平均だけなので、対応がずれると 3D がノイズや NaN になり、select_best で選んだ「最良」でも有効でないことが多い。

3. **最初のフレームや多くのフレームで有効な観測がない**  
   - 最初のフレームやほとんどのフレームで observations が空だと、Track の初期化ができなかったり、初期化後もマッチする観測がほとんどなく、出力 3D 軌跡がすべて NaN になる。

4. **無校正と座標範囲**  
   - 無校正では座標がピクセルオーダーなのに、xyz_bounds を mm など別スケールで設定していると、多くの点が範囲外となり、有効な 3D 点がさらに減る。

5. **SET_KEY / ファイルの不一致**  
   - 08 で使う SET_KEY が 04 で processed CSV を出した set_key と違う、あるいは processed のディレクトリ・ファイル名が一致していないと、誤ったデータや空データを読んで 3D が得られない。

---

## 五、建议排查与改进（简短）

- **确认 04 已跑且 SET_KEY 一致**：同一 set 的 `*_player1/2/3_processed.csv` 存在且 08 的 SET_KEY 与文件名一致。  
- **看 2D 是否过空**：打开某条 `*_processed.csv`，看 frame/x/y 中 NaN 比例；若极高，可考虑暂时 `skip_2d_interpolation=False` 或放宽异常值阈值（如 `max_jump_per_frame`）。  
- **08 中尝试 use_fish_id=True**：若 processed CSV 含可靠 fish_id，用 `use_fish_id=True` 调用 `run_load_2d`（并在 `get_observations_per_frame` 中传入），保证三视角按 a/b/c 对齐再三角化。  
- **无校准时放宽或关闭 xyz_bounds**：无校准时先用 `xyz_bounds=None` 或足够大的范围，确认能出 3D 再逐步收紧。  
- **先跑 06 无校准单流程**：用 `run_pipeline_no_calibration()` 看是否能得到非空的 3D CSV/图；若 06 也没有，问题更可能在 2D 加载或三角化输入，而不是 08 的 Track 逻辑。

---

以上即为「从生成 SLP 的 CSV 之后」到 3D 轨迹的代码逻辑、异常值的详细定义（中日双语）、以及重建不了 3D 轨迹时的可能原因与排查方向。
