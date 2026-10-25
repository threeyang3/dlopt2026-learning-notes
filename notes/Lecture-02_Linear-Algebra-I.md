# 第 2 讲：线性代数 I——梯度、Hessian、线性变换

> **课程定位**：2026 年课程核心数学基础。
>
> 配套文件：`materials/Lecture-02_Linear-Algebra-I/01_gradient-and-hessian_notes.pdf`、`02_linear-transform-eigenvalues_notes.pdf`

## 学习目标

掌握标量函数的梯度、Hessian 与二阶近似，理解线性变换、相似矩阵、特征值和特征向量的几何意义，并能用曲率解释条件数。比较矩阵函数可进入 [[Lecture-03_Linear-Algebra-II]]；将梯度搬到随机目标后进入 [[Lecture-05_Gradient-Descent]]。

## 1. 偏导、梯度与 Hessian

**核心掌握。** 对 $f:\mathbb R^d\to\mathbb R$，梯度收集所有一阶偏导：

$$
\nabla f(x)=\left[\frac{\partial f(x)}{\partial x_1},\ldots,\frac{\partial f(x)}{\partial x_d}\right]^\top.
$$

Hessian 收集所有二阶偏导：

$$
[H_f(x)]_{ij}=\frac{\partial^2 f(x)}{\partial x_i\partial x_j}.
$$

当二阶偏导连续时 $H_f(x)$ 对称。梯度的列向量形式直接用于更新，行向量形式用于线性化；不要混用维度。

二阶 Taylor 近似是理解几乎所有优化器的入口：

$$
f(y)\approx f(x)+\nabla f(x)^\top(y-x)+\frac12(y-x)^\top H_f(x)(y-x).
$$

若 $H_f\succ0$，局部呈碗形；若沿某个方向曲率为负，则该方向可能是不稳定或下降方向；若 Hessian 有正、有负，附近可能像鞍点。

## 2. 常见函数与计算套路

**核心掌握。**

- 二次型： $f(x)=\frac12x^\top Ax+b^\top x$ 时， $\nabla f(x)=Ax+b$， $H_f=A$（ $A$ 对称）。
- 最小二乘： $f(x)=\frac12\|Ax-b\|_2^2$ 时， $\nabla f=A^\top(Ax-b)$， $H_f=A^\top A$。
- log-sum-exp： $\mathrm{LSE}(x)=\log\sum_j e^{x_j}$ 是 max 的光滑近似；其梯度是 softmax。
- 复合标量函数通常先求外层导数，再乘内层 Jacobian 或梯度。

Hessian 可以被看作“梯度变化率”。在 [[Lecture-05_Gradient-Descent#2. 凸性、平滑性与曲率|第 5 讲]] 中， $L$ 控制梯度最多变化多快， $\mu$ 控制最弱方向有多弯。

## 3. 线性映射与矩阵表示

**核心掌握。** 选定基以后，线性映射 $T:\mathbb R^n\to\mathbb R^m$ 能写为 $T(x)=Ax$。矩阵不是抽象对象，而是某个坐标基下的线性变换；换基后矩阵一般改变。

**熟悉。** 同一线性变换在不同基下的矩阵虽然不同，但秩、特征值和行列式等不变量保持不变；计算具体换基矩阵时，要分清 $P$ 是把新坐标映到旧坐标还是反过来。

若基变换用可逆矩阵 $P$，同一个线性变换的新矩阵常写成

$$
A'=P^{-1}AP.
$$

这就是**相似变换**。相似矩阵表示同一线性算子在不同基下的样子，因此具有相同的特征多项式、特征值和代数重数等不变量。

## 4. 特征值与特征向量

**核心掌握。** 若

$$
Ap=\lambda p,\qquad p\ne0,
$$

则 $p$ 是特征向量， $\lambda$ 是对应特征值。几何上，矩阵只在 $p$ 方向做缩放，不改变方向（负特征值视为反向）。特征值由特征方程

$$
\det(\lambda I-A)=0
$$

得到。

若 $A$ 可对角化，

$$
A=P\Lambda P^{-1},
$$

则重复作用 $A^k=P\Lambda^kP^{-1}$。这直接解释了为什么特征值决定迭代误差的衰减或爆炸。

对对称实矩阵有更强的 spectral theorem：

$$
A=Q\Lambda Q^\top,\qquad Q^\top Q=I.
$$

特征向量可选为正交基。正定矩阵的全部特征值严格为正；半正定允许有零特征值。

## 5. 条件数与“病态”

**核心掌握直觉。** 对 Hessian 满足 $\mu I\preceq H_f(x)\preceq LI$ 的二次目标，最强的参数方向曲率约为 $L$，最弱方向约为 $\mu$，条件数为

$$
\kappa=\frac{L}{\mu}.
$$

$\kappa\gg1$ 时，等高线是细长椭圆，梯度下降在陡方向振荡、在缓方向爬得很慢。这种“锯齿”会在 [[Lecture-05_Gradient-Descent#5. 病态问题与动量|加速方法]] 中被缓解；自适应方法从另一角度处理它，见 [[Lecture-06_Stochastic-Gradient-Descent#5. 自适应 SGD|AdaGrad/Adam]]。

这里用 $L/\mu$ 是一种便于优化分析的谱条件数，前提是下界 $\mu>0$。更一般地，条件数可由最大、最小非零奇异值定义；非方阵或奇异矩阵不能直接套用 $L/\mu$。

## 公式速查

| 对象 | 公式或性质 |
|---|---|
| Taylor 近似 | $f(y)\approx f(x)+\nabla f^\top(y-x)+\frac12(y-x)^\top H(y-x)$ |
| 相似矩阵 | $A'=P^{-1}AP$，特征值不变 |
| 对称矩阵谱分解 | $A=Q\Lambda Q^\top$ |
| 正定 | $v^\top Av>0,\ \forall v\ne0$ |
| 条件数 | $\kappa=L/\mu$ |

## 仅作了解

**了解。** 一般域上相似矩阵的完整分类、最小多项式和非对角化情形下的 Jordan 标准形；知道“可能需广义特征向量”即可。讲义里的具体矩阵数值例题用于验算，不必逐题默写。

## 资料定位

| 主题 | 原始资料位置 |
|---|---|
| 多元函数、梯度、Hessian 与可微性 | `01_gradient-and-hessian_notes.pdf`，PDF p. 1-10 |
| 线性映射、相似矩阵、特征值 | `02_linear-transform-eigenvalues_notes.pdf`，PDF p. 1-7 |

## 自测

1. 为什么连续二阶偏导下 Hessian 对称？
2. 写出最小二乘目标的梯度和 Hessian。
3. 相似矩阵为什么有相同特征值？
4. 条件数很大时，GD 的轨迹通常呈现什么形状？

下一讲继续讨论向量值函数的 Jacobian、链式法则、矩阵函数和 SVD：[[Lecture-03_Linear-Algebra-II]]。

