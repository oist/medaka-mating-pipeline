# 跟踪数据处理算法说明文档

## 概述

本系统用于处理SLEAP输出的三条鱼跟踪数据，主要功能包括：
1. 自动生成file_sets.json文件，将同一实验的3个视角文件分组
2. ID分配：为每条鱼分配唯一且一致的ID
3. 异常值处理：检测并移除异常的位置数据
4. 缺失值填充：使用插值方法填充缺失的位置数据
5. 可视化：生成轨迹可视化图表

## 系统架构

```
scripts/tracking/
├── __init__.py
├── file_set_generator.py    # Set组分配
├── data_loader.py            # 数据加载
├── id_assignment.py         # ID分配算法
├── outlier_detection.py     # 异常值检测
├── missing_value_filling.py # 缺失值填充
├── visualization.py         # 可视化
├── process_tracking.py      # 主处理流程
├── main.py                  # 命令行接口
└── ALGORITHM_DESCRIPTION.md # 本文档
```

## 第一步：Set组分配

### 功能
自动扫描`sleap_output`目录，识别属于同一实验的3个视角文件（PLAYER1, PLAYER2, PLAYER3），并生成`file_sets.json`文件。

### 算法逻辑

1. **文件扫描**
   - 扫描`top_mouth_180/`目录，查找PLAYER1文件
   - 扫描`side_mouth_593/`目录，查找PLAYER2和PLAYER3文件

2. **Set Key提取**
   - 从文件名中提取set key
   - 文件名格式：`{view}_{date}_{...}_{PLAYER{num}}_{subject}_{behavior}_{time}.csv`
   - 提取规则：移除view前缀和PLAYER信息，保留实验标识和行为信息

3. **分组**
   - 将具有相同set key的文件归为一组
   - 只保留包含3个文件的完整set

4. **输出**
   - 生成`file_sets.json`文件
   - 格式：
     ```json
     {
       "set_key": {
         "key": "set_key",
         "player1_file": "path/to/player1.csv",
         "player2_file": "path/to/player2.csv",
         "player3_file": "path/to/player3.csv"
       }
     }
     ```

### 代码实现
- `file_set_generator.py::extract_set_key_from_filename()`: 提取set key
- `file_set_generator.py::generate_file_sets()`: 生成file_sets字典
- `file_set_generator.py::save_file_sets()`: 保存为JSON文件

## 第二步：ID分配算法

### 功能
为每条鱼分配唯一且一致的ID，确保在整个时间序列中同一条鱼的ID保持不变。

### 算法逻辑

#### 1. 基于Hungarian算法的帧间匹配

**核心思想**：使用匈牙利算法（Hungarian Algorithm）解决最优分配问题，将当前帧的检测结果与上一帧的已知ID进行匹配。

**步骤**：

1. **距离矩阵计算**
   - 对于上一帧的每个已知ID（fish_id），计算其位置与当前帧每个检测（track_id）的欧氏距离
   - 构建距离矩阵：`cost_matrix[i, j] = distance(prev_fish[i], curr_track[j])`

2. **阈值过滤**
   - 应用最大距离阈值（默认100像素）
   - 超过阈值的距离设为无穷大，表示不可匹配

3. **最优分配**
   - 使用`scipy.optimize.linear_sum_assignment`求解最优分配
   - 最小化总距离成本

4. **ID传递**
   - 将上一帧的fish_id分配给匹配的track_id
   - 对于未匹配的新检测，分配新的fish_id

#### 2. 时间序列传播

**策略**：
- 从第一帧开始，按时间顺序逐帧处理
- 每帧使用上一帧的ID分配结果作为参考
- 维护每个视角的ID历史记录

**处理流程**：
```
Frame 0: 初始化ID分配（按顺序：a, b, c）
Frame 1: 使用Hungarian算法匹配Frame 0的ID
Frame 2: 使用Hungarian算法匹配Frame 1的ID
...
Frame N: 使用Hungarian算法匹配Frame N-1的ID
```

#### 3. 多视角独立处理

- 每个视角（PLAYER1, PLAYER2, PLAYER3）独立进行ID分配
- 不进行跨视角的ID关联（因为这是2D跟踪，不是3D重建）

### 参数说明

- `max_distance`: 最大允许匹配距离（像素）
  - 默认值：100
  - 超过此距离的检测不会被匹配到已知ID

### 代码实现
- `id_assignment.py::compute_distance_matrix()`: 计算距离矩阵
- `id_assignment.py::assign_ids_hungarian()`: Hungarian算法分配
- `id_assignment.py::assign_ids_temporal()`: 时间序列ID分配

### 算法优势

1. **全局最优**：Hungarian算法保证找到最小总成本的分配方案
2. **鲁棒性**：通过距离阈值过滤，避免错误匹配
3. **处理新检测**：自动为新出现的检测分配新ID
4. **处理消失**：当某条鱼暂时消失时，ID保留，等待重新出现

## 第三步：异常值处理算法

### 功能
检测并移除异常的位置数据，包括：
- 异常大的速度
- 异常大的位置跳跃
- 基于统计的异常值（IQR、Z-score）

### 算法方法

#### 1. 基于速度的异常值检测

**原理**：计算相邻帧之间的速度，如果速度超过阈值，则认为是异常值。

**步骤**：
1. 计算速度：`v = ||p(t+1) - p(t)|| / dt`
2. 应用阈值：`v > max_velocity` → 异常值
3. 标记异常帧的位置为NaN

**参数**：
- `max_velocity`: 最大允许速度（像素/帧），默认500

#### 2. 基于位置跳跃的异常值检测

**原理**：检测相邻帧之间的位置跳跃，如果跳跃距离超过阈值，则认为是异常值。

**步骤**：
1. 计算位移：`d = ||p(t+1) - p(t)||`
2. 应用阈值：`d > max_jump_distance` → 异常值
3. 标记异常帧的位置为NaN

**参数**：
- `max_jump_distance`: 最大允许跳跃距离（像素），默认200

#### 3. 基于IQR的异常值检测

**原理**：使用四分位距（Interquartile Range）方法检测异常值。

**步骤**：
1. 计算Q1（25%分位数）和Q3（75%分位数）
2. 计算IQR = Q3 - Q1
3. 定义异常值范围：`[Q1 - factor*IQR, Q3 + factor*IQR]`
4. 超出范围的值标记为异常值

**参数**：
- `factor`: IQR倍数因子，默认1.5

#### 4. 基于Z-score的异常值检测

**原理**：使用Z-score（标准化分数）检测异常值。

**步骤**：
1. 计算均值和标准差
2. 计算Z-score：`z = |x - mean| / std`
3. 应用阈值：`z > threshold` → 异常值

**参数**：
- `threshold`: Z-score阈值，默认3.0

### 处理策略

- 对x和y坐标分别检测
- 任一坐标异常则标记整个位置为异常
- 异常值设为NaN，等待后续插值填充

### 代码实现
- `outlier_detection.py::detect_velocity_outliers()`: 速度异常检测
- `outlier_detection.py::detect_position_jumps()`: 位置跳跃检测
- `outlier_detection.py::detect_outliers_iqr()`: IQR异常检测
- `outlier_detection.py::detect_outliers_zscore()`: Z-score异常检测
- `outlier_detection.py::process_all_fish_outliers()`: 批量处理

## 第四步：缺失值填充算法

### 功能
使用插值方法填充缺失的位置数据（包括异常值处理后的NaN）。

### 算法方法

#### 1. 线性插值

**原理**：在已知点之间进行线性插值。

**步骤**：
1. 识别连续NaN区域
2. 对于长度≤limit的NaN区域，使用线性插值
3. 对于长度>limit的NaN区域，保持NaN

**参数**：
- `limit`: 最大连续NaN数量，默认5

#### 2. 多项式插值

**原理**：使用低阶多项式（通常1阶）进行插值。

**步骤**：
1. 识别连续NaN区域
2. 对于长度≤limit的NaN区域，使用多项式插值
3. 对于长度>limit的NaN区域，保持NaN

**参数**：
- `limit`: 最大连续NaN数量，默认5
- `order`: 多项式阶数，默认1

#### 3. 三次样条插值

**原理**：使用三次样条函数进行平滑插值。

**步骤**：
1. 识别连续NaN区域
2. 对于长度≤limit的NaN区域，使用三次样条插值
3. 对于长度>limit的NaN区域，保持NaN

**参数**：
- `limit`: 最大连续NaN数量，默认5

#### 4. Savitzky-Golay平滑

**原理**：在插值后应用Savitzky-Golay滤波器进行平滑处理。

**步骤**：
1. 对插值后的数据进行Savitzky-Golay滤波
2. 使用滑动窗口拟合多项式
3. 计算平滑后的值

**参数**：
- `window_length`: 窗口大小，默认9
- `polyorder`: 多项式阶数，默认2

### 处理策略

1. **小间隙填充**：对于短时间缺失（≤limit帧），使用插值填充
2. **大间隙保留**：对于长时间缺失（>limit帧），保持NaN，不进行填充
3. **平滑处理**：可选地应用Savitzky-Golay滤波器，减少噪声

### 代码实现
- `missing_value_filling.py::interpolate_1d_timeseries()`: 1D时间序列插值
- `missing_value_filling.py::fill_missing_values_trajectory()`: 轨迹缺失值填充
- `missing_value_filling.py::process_all_fish_missing_values()`: 批量处理

## 可视化功能

### 功能
生成轨迹可视化图表，包括：
1. ID分配后的轨迹可视化
2. 最终处理后的轨迹可视化

### 可视化内容

#### 单视角轨迹图
- X轴：X坐标（像素）
- Y轴：Y坐标（像素）
- 不同颜色的线表示不同的鱼
- 起点标记为圆圈（○）
- 终点标记为方块（□）

#### 三视角组合图
- 一张大图包含3个小图
- 每个小图对应一个视角（PLAYER1, PLAYER2, PLAYER3）
- 统一的颜色映射，便于跨视角对比

### 代码实现
- `visualization.py::plot_trajectories_single_view()`: 单视角可视化
- `visualization.py::plot_trajectories_three_views()`: 三视角组合可视化

## 主处理流程

### 处理步骤

1. **加载数据**
   - 读取3个视角的CSV文件
   - 解析为按帧组织的格式

2. **ID分配**
   - 使用Hungarian算法进行时间序列ID分配
   - 生成带ID的DataFrame
   - 生成ID分配后的可视化

3. **异常值处理**
   - 检测异常值（速度、跳跃、统计方法）
   - 将异常值标记为NaN

4. **缺失值填充**
   - 使用插值方法填充缺失值
   - 可选应用平滑处理
   - 生成最终处理后的可视化

5. **保存结果**
   - 保存处理后的CSV文件
   - 保存可视化图片

### 代码实现
- `process_tracking.py::process_single_set()`: 处理单个set
- `process_tracking.py::process_from_csv_path()`: 从CSV路径处理
- `process_tracking.py::process_from_file_set_key()`: 从set key处理

## 命令行使用

### 基本用法

```bash
python scripts/tracking/main.py <input> [options]
```

### 参数说明

- `input`: 输入CSV文件路径或file_set_key
- `--sleap-output-dir`: SLEAP输出目录（默认: d:\\sleap_output）
- `--output-dir`: 输出目录（默认: sleap_output_dir/processed）
- `--max-distance`: ID分配最大距离阈值（默认: 100）
- `--outlier-method`: 异常值检测方法（velocity/jump/iqr/zscore，默认: velocity）
- `--interp-method`: 插值方法（linear/polynomial/cubic，默认: linear）
- `--interp-limit`: 最大连续缺失值数量（默认: 5）
- `--no-smooth`: 禁用平滑处理
- `--no-viz`: 禁用可视化
- `--generate-file-sets`: 生成file_sets.json文件

### 示例

```bash
# 处理单个CSV文件
python scripts/tracking/main.py "d:\sleap_output\top_mouth_180\top_mouth_180_20251220_E_1_2_FCMC_PLAYER1_male1_Courtship_1905.506-1913.422.csv"

# 使用自定义参数
python scripts/tracking/main.py "d:\sleap_output\top_mouth_180\top_mouth_180_20251220_E_1_2_FCMC_PLAYER1_male1_Courtship_1905.506-1913.422.csv" --max-distance 150 --outlier-method jump --interp-method cubic

# 处理整个set
python scripts/tracking/main.py "20251220_E_1_2_FCMC_male1_Courtship_1905.506-1913.422"
```

## 算法特点

### 优势

1. **自动化**：自动生成file_sets.json，无需手动配置
2. **鲁棒性**：多种异常值检测方法，适应不同场景
3. **灵活性**：可配置的参数，适应不同实验条件
4. **可视化**：直观的轨迹可视化，便于质量检查
5. **可扩展性**：模块化设计，易于扩展新功能

### 局限性

1. **2D跟踪**：每个视角独立处理，不进行3D重建
2. **参数依赖**：需要根据实验条件调整参数
3. **大间隙处理**：长时间缺失的数据无法完全恢复

## 未来改进方向

1. **3D重建**：结合多个视角进行3D位置重建
2. **跨视角ID关联**：在3D空间中统一ID分配
3. **自适应参数**：根据数据特征自动调整参数
4. **深度学习**：使用深度学习模型进行异常值检测和缺失值填充
5. **实时处理**：支持实时数据流处理

## 参考文献

1. Hungarian Algorithm: Kuhn, H. W. (1955). The Hungarian method for the assignment problem. Naval Research Logistics Quarterly, 2(1‐2), 83-97.
2. Savitzky-Golay Filter: Savitzky, A., & Golay, M. J. (1964). Smoothing and differentiation of data by simplified least squares procedures. Analytical chemistry, 36(8), 1627-1639.
3. IQR Method: Tukey, J. W. (1977). Exploratory data analysis. Reading, Mass.
