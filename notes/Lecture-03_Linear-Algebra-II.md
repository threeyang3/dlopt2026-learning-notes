# 第 3 讲：线性代数 II——Jacobian、矩阵函数、范数与 SVD

> **课程定位**：2026 年课程核心数学基础。
>
> 配套文件：`materials/Lecture-03_Linear-Algebra-II/01_jacobian-hessian-chain-rule_notes.pdf`、`02_inner-product-matrix-functions-backprop_notes.pdf`、`03_norms-svd_notes.pdf`

## 学习目标

把标量梯度扩展到向量值函数，掌握链式法则和反向传播的矩阵形式；理解正定、内积、范数、对偶范数与 SVD；能据此判断低秩、谱约束和权重衰减为何常有优化意义。

## 1. Jacobian 与链式法则

**核心掌握。** 对 $f:\mathbb R^n\to\mathbb R^m$，Jacobian 是 $m\times n$ 矩阵：

$$
[J_f(x)]_{ij}=\frac{\partial f_i(x)}{\partial x_j}.
$$

若 $z=g(x)$、$y=f(z)$，则局部线性化是

$$
dy=J_f(z)\,dz=J_f(g(x))J_g(x)\,dx.
$$

因此链式法则为

$$
J_{f\circ g}(x)=J_f(g(x))J_g(x).
$$

梯度是标量输出时的 Jacobian 转置（列向量约定），Hessian 是梯度的 Jacobian：

$$
H_f(x)=J_{\nabla f}(x).
$$

## 2. 向量化、Softmax 与反向传播

**核心掌握反向传播的维度与复用。** 对一个 Transformer 层或任意计算图，训练需要“损失对各层中间量的导数”。反向模式自动微分的核心是：

1. 前向保存或重算中间量；
2. 从标量损失开始，按逆拓扑顺序传递梯度；
3. 每步使用局部 Jacobian 的转置作用在上一层梯度上。

若 $z=Wy$，则

$$
\frac{\partial L}{\partial y}=W^\top\frac{\partial L}{\partial z},
\qquad
\frac{\partial L}{\partial W}=\frac{\partial L}{\partial z}y^\top.
$$

这解释了为什么全参数训练要保存激活：反向传播需要它们；零阶方法试图用仅前向查询绕开这一点，见 [[Lecture-11_Zeroth-Order-Optimizers#2. 用函数值估计梯度|第 11 讲]]。

Softmax 的 Jacobian 可写成

$$
J_{\operatorname{softmax}}(s)=\operatorname{Diag}(p)-pp^\top.
$$

它与分类损失的组合常被化简为 $p-q$。向量的 log-sum-exp 为

$$
\operatorname{LSE}(s)=\log\sum_i e^{s_i},
\qquad
\nabla\operatorname{LSE}(s)=\operatorname{softmax}(s).
$$

这是梯度关系，不是反函数关系；softmax 也不是可逆映射，因为给所有 logits 加同一常数不会改变输出。不要把向量 LSE 的结论无条件推广到矩阵函数。

## 3. 内积、正交与正定

**熟悉，正定性质必须掌握。**

- 标准内积：$\langle x,y\rangle=x^\top y$。
- 加权内积：$\langle x,y\rangle_M=x^\top My$，其中 $M$ 必须对称正定。
- 正定：$v^\top Mv>0,\ \forall v\ne0$。
- 正交投影：把 $x$ 投到方向 $u$ 上，使残差与 $u$ 正交。

正定矩阵定义了可用的“新距离”，也能构成优化中的局部曲率模型。协方差矩阵、$A^\top A$ 与 Hessian（局部正定时）都常具有正半定或正定结构。

## 4. 矩阵函数与矩阵微积分

**熟悉。** 标量函数可以作用在矩阵元素上，例如逐元素平方、逐元素 sigmoid；也可以借助特征分解定义矩阵函数。优化里更重要的是矩阵微分恒等式：

$$
d(XY)=(dX)Y+X(dY),
$$

$$
d\operatorname{tr}(X)=\operatorname{tr}(dX),
\qquad
d\operatorname{tr}(AX)=\operatorname{tr}(A\,dX),
$$

其中 $X$ 在第一条中为方阵，$AX$ 在第二条中必须维度相容。若在 Frobenius 内积下已化为 $df=\operatorname{tr}(G^\top dX)$，则 $G$ 是对应梯度；因此 $\nabla_X\operatorname{tr}(X)=I$，而 $\nabla_X\operatorname{tr}(AX)=A^\top$。

矩阵微积分速查：

| 标量函数 | 变量形状 | 微分 | Frobenius 意义下的梯度形状 |
|---|---|---|---|
| $\operatorname{tr}(X)$ | $n\times n$ | $\operatorname{tr}(dX)$ | $n\times n$：$I$ |
| $\operatorname{tr}(AX)$ | $X\colon m\times n$，$A\colon n\times m$ | $\operatorname{tr}(A\,dX)$ | $m\times n$：$A^\top$ |
| $\operatorname{tr}(AXB)$ | 各因子维度相容 | $\operatorname{tr}(BA\,dX)$ | $A^\top B^\top$ |

先核对变量与结果形状，再使用转置公式；不要把微分本身误写成一个梯度。

**了解。** Fréchet 可微、Gâteaux 可微与可微性之间的完整关系；知道矩阵空间可以赋予内积和范数即可。

## 5. 范数

**核心掌握常用范数及用途。**

- 1-范数：$\|x\|_1=\sum_i|x_i|$，促进稀疏。
- 2-范数：$\|x\|_2=\sqrt{x^\top x}$，欧氏长度。
- $\infty$-范数：$\|x\|_\infty=\max_i|x_i|$，最大坐标幅度。
- Frobenius 范数：$\|X\|_F^2=\sum_{ij}X_{ij}^2$，等价于把矩阵展平成向量后的 2-范数。

向量范数需满足非负性、正齐次性与三角不等式。平方欧氏长度不满足正齐次性，所以不是范数。

矩阵算子范数定义为

$$
\|A\|_p=\max_{x\ne0}\frac{\|Ax\|_p}{\|x\|_p}.
$$

常用结论：

$$
\|A\|_1=\max_j\sum_i|A_{ij}|,
\qquad
\|A\|_\infty=\max_i\sum_j|A_{ij}|,
$$

$$
\|A\|_2=\sigma_{\max}(A).
$$

对偶范数定义为

$$
\|y\|_*=\max_{\|x\|\le1}y^\top x.
$$

它与低秩、压缩和压缩感知中的凸松弛相关。完整约束优化证明可留到专门课程。

## 6. SVD 与低秩

**核心掌握。** 任意矩阵 $X\in\mathbb R^{m\times n}$ 可做奇异值分解

$$
X=U\Sigma V^\top,
$$

非零奇异值满足

$$
\sigma_i(X)=\sqrt{\lambda_i(X^\top X)}.
$$

矩阵的 2-范数为最大奇异值；Frobenius 范数为奇异值的欧氏长度：

$$
\|X\|_F=\sqrt{\sum_i\sigma_i(X)^2}.
$$

只保留前 $r$ 个奇异值得到 rank-$r$ 近似。Eckart–Young 定理说明 Frobenius 范数下 SVD 截断是最佳低秩近似：

$$
\|X-X_r\|_F^2=\sum_{i>r}\sigma_i^2.
$$

该结论会成为 GaLore 的核心假设：若梯度接近低秩，就在低维子空间保存优化器状态。需要同时记住它的风险——噪声梯度未必低秩，见 [[Lecture-10_Subspace-Optimizers#2. GaLore 与其收敛风险|子空间优化]]。

## 仅作了解

- 核范数（奇异值之和）和更一般的 Schatten 范数。
- SVD 的完整存在性证明、奇异值扰动界。
- 所有对偶范数公式；先熟记 $\|\cdot\|_1\leftrightarrow\|\cdot\|_\infty$ 与 $\|\cdot\|_2$ 自对偶。

## 自测

1. 写出 $f(x)=W_2\sigma(W_1x)$ 的两步链式 Jacobian。
2. 为什么反向传播需要缓存激活？
3. 矩阵 2-算子范数与最大奇异值是什么关系？
4. 若奇异值快速衰减，截断 SVD 会带来什么近似误差？

## 资料定位

| 主题 | 原始资料位置 |
|---|---|
| Jacobian、链式法则与 Hessian | `01_jacobian-hessian-chain-rule_notes.pdf`，PDF p. 1-10 |
| 内积、正定、矩阵函数与矩阵微分 | `02_inner-product-matrix-functions-backprop_notes.pdf`，PDF p. 1-22 |
| 范数、对偶范数与 SVD | `03_norms-svd_notes.pdf`，PDF p. 1-18 |

基础代数结束后，学习随机变量与期望：[[Lecture-04_Probability-and-Statistics]]。

