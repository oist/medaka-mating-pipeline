# Camera Calibration Theory: Polynomial Regression Approach

相机校准原理详解：多项式回归方法

---

## 1. 问题定义 (Problem Definition)

### 1.1 目标

相机校准的核心目标是建立**2D图像坐标**与**3D世界坐标**之间的映射关系。

给定：
- 多个相机拍摄的2D像素坐标 $(u, v)$
- 对应点的3D世界坐标 $(X, Y, Z)$

求解：
- 一个函数 $f$，使得 $(X, Y, Z) = f(u_1, v_1, u_2, v_2, u_3, v_3)$

### 1.2 为什么不用传统方法？

传统相机校准（如OpenCV的`calibrateCamera`）需要：
1. 求解相机内参矩阵 $K$（焦距、主点、畸变系数）
2. 求解相机外参矩阵 $[R|t]$（旋转、平移）
3. 使用针孔模型 + 畸变校正

**问题**：
- 水下成像存在**折射效应**（空气-玻璃-水界面）
- 针孔模型假设光线直线传播，无法准确描述折射
- 需要复杂的物理建模

**我们的方法**：
- 直接学习2D→3D的映射，**跳过物理建模**
- 多项式回归可以**隐式学习**折射、畸变等非线性效应
- 数据驱动，不依赖相机参数

---

## 2. 数学原理 (Mathematical Principles)

### 2.1 多项式回归基础

多项式回归是线性回归的扩展。对于输入 $x$，我们不仅使用 $x$，还使用 $x^2, x^3, ...$ 作为特征。

**一维情况**：

$$y = w_0 + w_1 x + w_2 x^2 + w_3 x^3 + ... + w_n x^n$$

**多维情况（我们的场景）**：

对于6个输入变量 $(u_1, v_1, u_2, v_2, u_3, v_3)$，2阶多项式特征包括：
- 常数项：$1$
- 一次项：$u_1, v_1, u_2, v_2, u_3, v_3$
- 二次项：$u_1^2, u_1 v_1, u_1 u_2, ..., v_3^2$
- 交叉项：$u_1 u_2, u_1 v_2, v_1 u_3, ...$

### 2.2 特征展开 (Feature Expansion)

设原始输入向量为：

$$\mathbf{x} = [u_1, v_1, u_2, v_2, u_3, v_3]^T \in \mathbb{R}^6$$

多项式特征展开 $\phi(\mathbf{x})$ 将其映射到高维空间：

$$\phi: \mathbb{R}^6 \rightarrow \mathbb{R}^d$$

其中 $d$ 取决于多项式阶数：

| 阶数 (degree) | 特征数量 $d$ | 说明 |
|--------------|-------------|------|
| 1 | 7 | 6个原始特征 + 1个偏置 |
| 2 | 28 | 包含所有二次项和交叉项 |
| 3 | 84 | 包含所有三次项 |
| 4 | 210 | 包含所有四次项 |

特征数量公式：$d = \binom{n + k}{k}$，其中 $n=6$ 是输入维度，$k$ 是多项式阶数。

### 2.3 Ridge回归 (L2正则化线性回归)

在高维特征空间中，我们使用Ridge回归来防止过拟合：

$$\min_{\mathbf{W}} \|\mathbf{Y} - \mathbf{\Phi} \mathbf{W}\|_2^2 + \alpha \|\mathbf{W}\|_2^2$$

其中：
- $\mathbf{Y} \in \mathbb{R}^{N \times 3}$：$N$ 个校准点的3D坐标
- $\mathbf{\Phi} \in \mathbb{R}^{N \times d}$：展开后的特征矩阵
- $\mathbf{W} \in \mathbb{R}^{d \times 3}$：待学习的权重矩阵
- $\alpha$：正则化参数，控制模型复杂度

**闭式解**：

$$\mathbf{W}^* = (\mathbf{\Phi}^T \mathbf{\Phi} + \alpha \mathbf{I})^{-1} \mathbf{\Phi}^T \mathbf{Y}$$

### 2.4 为什么这是"行列对应"？

你老师说的"行列对应"本质上是指：

**训练数据的组织形式**：

```
特征矩阵 Φ (N×d)              目标矩阵 Y (N×3)
┌─────────────────────┐      ┌─────────────┐
│ φ(x₁)               │  →   │ X₁  Y₁  Z₁  │   第1个点
│ φ(x₂)               │  →   │ X₂  Y₂  Z₂  │   第2个点
│ φ(x₃)               │  →   │ X₃  Y₃  Z₃  │   第3个点
│  ...                │  →   │  ...        │
│ φ(xₙ)               │  →   │ Xₙ  Yₙ  Zₙ  │   第N个点
└─────────────────────┘      └─────────────┘
```

每一**行**是一个校准点，包含：
- 左边：该点在3个相机图像中的2D坐标（展开为多项式特征）
- 右边：该点的真实3D世界坐标

**学习的本质**：找到权重矩阵 $\mathbf{W}$，使得 $\mathbf{\Phi} \mathbf{W} \approx \mathbf{Y}$

---

## 3. 完整流水线 (Complete Pipeline)

### 3.1 数据预处理

```
输入: 
  - 3个相机的2D坐标: (u₁,v₁), (u₂,v₂), (u₃,v₃)
  - 对应的3D坐标: (X, Y, Z)

Step 1: 标准化 (StandardScaler)
  x̃ᵢ = (xᵢ - μᵢ) / σᵢ
  
  原因: 不同相机的像素范围可能不同，标准化使所有特征在同一尺度

Step 2: 多项式展开 (PolynomialFeatures)
  [u₁, v₁, u₂, v₂, u₃, v₃] → [1, u₁, v₁, ..., u₁², u₁v₁, ..., u₃v₃²]
  
  原因: 捕捉非线性关系和特征间的交互作用

Step 3: Ridge回归
  学习权重矩阵 W
```

### 3.2 模型架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        Model Pipeline                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐ │
│  │ Input (6D)  │ →  │ StandardScaler   │ →  │ PolynomialFeatures│ │
│  │ u₁,v₁,u₂,   │    │ 标准化           │    │ 多项式展开 (d维)  │ │
│  │ v₂,u₃,v₃    │    │                  │    │                   │ │
│  └─────────────┘    └──────────────────┘    └─────────────────┘ │
│                                                    ↓             │
│                                             ┌─────────────────┐ │
│                          Output (3D)   ←    │ Ridge Regression│ │
│                          X, Y, Z            │ 线性回归+L2正则 │ │
│                                             └─────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 3.3 超参数选择

通过**网格搜索 + 交叉验证**选择最优参数：

| 参数 | 搜索范围 | 作用 |
|------|---------|------|
| `alpha` | $[10^{-10}, 10^{-7}, ..., 1]$ | 正则化强度。越大越简单，防止过拟合 |
| `degree` | $[1, 2, 3, 4]$ | 多项式阶数。越高越复杂，能拟合更复杂的非线性关系 |

**选择标准**：最小化交叉验证误差（负均方误差）

---

## 4. 四个模型的作用 (Four Models Explained)

### 4.1 主模型：imCoords_to_XYZ

**功能**：从6个2D坐标直接预测3D位置

$$f: \mathbb{R}^6 \rightarrow \mathbb{R}^3$$
$$(u_1, v_1, u_2, v_2, u_3, v_3) \mapsto (X, Y, Z)$$

**输入**：3个相机的像素坐标（6维）
**输出**：世界坐标（3维）

这是**核心模型**，用于3D重建。

### 4.2 辅助模型：用于补全缺失视角

当某个相机看不到目标点时（遮挡、出界），可以用其他两个相机的坐标**预测**缺失相机的坐标。

| 模型名称 | 输入 | 输出 | 用途 |
|---------|------|------|------|
| `xy_yz_to_xz` | cam1 + cam2 (4D) | cam0 (2D) | 补全 cam0 |
| `xz_xy_to_yz` | cam0 + cam1 (4D) | cam2 (2D) | 补全 cam2 |
| `xz_yz_to_xy` | cam0 + cam2 (4D) | cam1 (2D) | 补全 cam1 |

**工作流程**：

```
情况: cam1 被遮挡，只有 cam0 和 cam2 的坐标

Step 1: 使用 xz_yz_to_xy 预测 cam1 坐标
        (u₀, v₀, u₂, v₂) → (û₁, v̂₁)

Step 2: 使用 imCoords_to_XYZ 预测 3D 坐标
        (u₀, v₀, û₁, v̂₁, u₂, v₂) → (X, Y, Z)
```

---

## 5. 为什么这种方法有效？(Why It Works)

### 5.1 理论基础：万能逼近定理

**Stone-Weierstrass 定理**：任何连续函数都可以用多项式任意精度逼近。

虽然真实的2D→3D映射可能非常复杂（涉及透视投影、折射、畸变），但只要这个映射是连续的，多项式回归就能学习它。

### 5.2 隐式学习非线性效应

多项式特征的交叉项（如 $u_1 \cdot u_2$）可以捕捉：
- **透视效应**：远近物体的尺度变化
- **镜头畸变**：径向和切向畸变
- **折射效应**：水下成像的光线弯曲
- **相机间关系**：不同视角的几何约束

### 5.3 优势与局限

**优势**：
1. ✅ 不需要相机内参/外参标定
2. ✅ 自动处理折射、畸变等非线性效应
3. ✅ 实现简单，无需复杂的物理建模
4. ✅ 端到端学习，减少误差累积

**局限**：
1. ⚠️ 需要足够多的校准点（建议 > 50 个）
2. ⚠️ 只在校准点覆盖的空间范围内准确
3. ⚠️ 外推（超出校准范围）误差可能较大
4. ⚠️ 高阶多项式可能过拟合

---

## 6. 误差分析 (Error Analysis)

### 6.1 误差来源

1. **标注误差**：手动标注2D点时的像素误差
2. **模板误差**：3D坐标测量的误差
3. **模型误差**：多项式逼近的近似误差
4. **泛化误差**：新数据上的预测误差

### 6.2 验证策略

```
训练数据 (90%)          测试数据 (10%)
┌─────────────────┐    ┌─────────────────┐
│ 用于训练模型     │    │ 用于评估泛化能力 │
│                 │    │                 │
│ 重复100次随机分割，统计误差分布          │
└─────────────────┘    └─────────────────┘
```

### 6.3 误差指标

- **训练误差**：模型在训练数据上的预测误差
- **测试误差**：模型在未见数据上的预测误差
- **过拟合程度**：测试误差 / 训练误差

如果测试误差远大于训练误差，说明过拟合，需要：
- 增加正则化 (增大 alpha)
- 降低多项式阶数 (减小 degree)
- 增加训练数据

---

## 7. 与传统方法的对比 (Comparison with Traditional Methods)

| 方面 | 传统方法 (OpenCV) | 多项式回归方法 |
|------|------------------|----------------|
| **相机模型** | 针孔模型 + 畸变参数 | 无显式模型 |
| **折射处理** | 需要额外建模 | 隐式学习 |
| **标定流程** | 内参→外参→三角化 | 直接学习2D→3D |
| **所需数据** | 棋盘格图像 | 已知3D位置的点 |
| **误差来源** | 多步骤误差累积 | 单步端到端 |
| **适用场景** | 空气中成像 | 任意介质（包括水下） |
| **可解释性** | 物理意义明确 | 黑盒模型 |

---

## 8. 数学推导 (Mathematical Derivation)

### 8.1 Ridge回归推导

目标函数：

$$J(\mathbf{W}) = \sum_{i=1}^{N} \|\mathbf{y}_i - \phi(\mathbf{x}_i)^T \mathbf{W}\|^2 + \alpha \|\mathbf{W}\|_F^2$$

矩阵形式：

$$J(\mathbf{W}) = \|\mathbf{Y} - \mathbf{\Phi} \mathbf{W}\|_F^2 + \alpha \|\mathbf{W}\|_F^2$$

对 $\mathbf{W}$ 求导并令其为零：

$$\frac{\partial J}{\partial \mathbf{W}} = -2\mathbf{\Phi}^T(\mathbf{Y} - \mathbf{\Phi}\mathbf{W}) + 2\alpha\mathbf{W} = 0$$

$$\mathbf{\Phi}^T\mathbf{Y} = (\mathbf{\Phi}^T\mathbf{\Phi} + \alpha\mathbf{I})\mathbf{W}$$

$$\mathbf{W}^* = (\mathbf{\Phi}^T\mathbf{\Phi} + \alpha\mathbf{I})^{-1}\mathbf{\Phi}^T\mathbf{Y}$$

### 8.2 预测

对于新的2D坐标 $\mathbf{x}_{new}$：

$$\hat{\mathbf{y}} = \phi(\mathbf{x}_{new})^T \mathbf{W}^*$$

---

## 9. 实际应用注意事项 (Practical Considerations)

### 9.1 校准点的选择

- **数量**：建议至少50个点，越多越好
- **分布**：均匀覆盖整个目标空间
- **深度**：包含不同深度（Z方向）的点

### 9.2 标注精度

- 2D标注误差直接影响最终精度
- 建议使用亚像素级标注（放大图像标注）
- 可以使用多人标注取平均

### 9.3 模型验证

- 检查训练误差和测试误差
- 可视化预测结果与真实值的对比
- 在新数据上验证泛化能力

---

## 10. 参考文献 (References)

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. (多项式回归、正则化)

2. Hartley, R., & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision*. Cambridge University Press. (多视角几何、三角化)

3. Zhang, Z. (2000). A flexible new technique for camera calibration. *IEEE Transactions on Pattern Analysis and Machine Intelligence*. (传统相机标定方法)

4. Sedlazeck, A., & Koch, R. (2012). Perspective and non-perspective camera models in underwater imaging. *OCEANS*. (水下成像折射问题)

5. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. (Ridge回归、交叉验证)

---

## 附录 A: 代码示例

### A.1 特征展开示例

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# 输入: 3个相机的2D坐标
x = np.array([[100, 200, 150, 250, 120, 220]])  # shape: (1, 6)

# 多项式展开 (degree=2)
poly = PolynomialFeatures(degree=2, include_bias=True)
x_poly = poly.fit_transform(x)

print(f"Original features: {x.shape[1]}")  # 6
print(f"Polynomial features: {x_poly.shape[1]}")  # 28
print(f"Feature names: {poly.get_feature_names_out()}")
```

### A.2 完整训练示例

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge

# 创建模型
model = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('ridge', Ridge(alpha=0.001))
])

# 训练
# X: (N, 6) - 6个2D坐标
# Y: (N, 3) - 3D世界坐标
model.fit(X_train, Y_train)

# 预测
Y_pred = model.predict(X_test)

# 计算误差
errors = np.linalg.norm(Y_test - Y_pred, axis=1)
print(f"Mean error: {errors.mean():.2f} mm")
```

---

*文档版本: 1.0*
*最后更新: 2024年12月*
