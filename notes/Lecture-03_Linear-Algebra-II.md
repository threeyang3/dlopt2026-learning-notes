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

若 $z=g(x)$、 $y=f(z)$，则局部线性化是

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

若 $W\in\mathbb R^{m\times n}$、 $y\in\mathbb R^n$、 $z=Wy\in\mathbb R^m$，标量损失的导数采用列梯度，则

$$
\frac{\partial L}{\partial y}=W^\top\frac{\partial L}{\partial z},
\qquad
\frac{\partial L}{\partial W}=\frac{\partial L}{\partial z}y^\top.
$$

这解释了为什么全参数训练要保存激活：反向传播需要它们；零阶方法试图用仅前向查询绕开这一点，见 [[Lecture-11_Zeroth-Order-Optimizers#2. 用函数值估计梯度|第 11 讲]]。

Softmax 的 Jacobian 可写成

$$
J_{\mathrm{softmax}}(s)=\mathrm{Diag}(p)-pp^\top.
$$

取交叉熵 $L=-\sum_iq_i\log p_i$、 $\sum_iq_i=1$ 时，对 logits 的梯度为 $\nabla_sL=p-q$；不是 softmax 函数自身的 Jacobian 等于 $p-q$。向量的 log-sum-exp 为

$$
\mathrm{LSE}(s)=\log\sum_i e^{s_i},
\qquad
\nabla\mathrm{LSE}(s)=\mathrm{softmax}(s).
$$

这是梯度关系，不是反函数关系；softmax 也不是可逆映射，因为给所有 logits 加同一常数不会改变输出。不要把向量 LSE 的结论无条件推广到矩阵函数。

## 3. 内积、正交与正定

**熟悉，正定性质必须掌握。**

- 标准内积： $\langle x,y\rangle=x^\top y$。
- 加权内积： $\langle x,y\rangle_M=x^\top My$，其中 $M$ 必须对称正定。
- 正定： $v^\top Mv>0,\ \forall v\ne0$。
- 正交投影：把 $x$ 投到方向 $u$ 上，使残差与 $u$ 正交。

正定矩阵定义了可用的“新距离”，也能构成优化中的局部曲率模型。协方差矩阵、 $A^\top A$ 与 Hessian（局部正定时）都常具有正半定或正定结构。

## 4. 矩阵函数与矩阵微积分

**熟悉。** 标量函数可以作用在矩阵元素上，例如逐元素平方、逐元素 sigmoid；也可以借助特征分解定义矩阵函数。优化里更重要的是矩阵微分恒等式：

$$
d(XY)=(dX)Y+X(dY),
$$

$$
d\mathrm{tr}(X)=\mathrm{tr}(dX),
\qquad
d\mathrm{tr}(AX)=\mathrm{tr}(A\,dX),
$$

在两条迹公式中， $X$ 或乘积 $AX$ 分别应为方阵， $A$ 为固定矩阵；乘积微分 $d(XY)$ 本身不要求 $X$ 是方阵。若在 Frobenius 内积下已化为 $df=\mathrm{tr}(G^\top dX)$，则 $G$ 是对应梯度；因此 $\nabla_X\mathrm{tr}(X)=I$，而 $\nabla_X\mathrm{tr}(AX)=A^\top$。

矩阵微积分速查：

| 标量函数 | 变量形状 | 微分 | Frobenius 意义下的梯度形状 |
|---|---|---|---|
| $\mathrm{tr}(X)$ | $n\times n$ | $\mathrm{tr}(dX)$ | $n\times n$： $I$ |
| $\mathrm{tr}(AX)$ | $X\colon m\times n$， $A\colon n\times m$ | $\mathrm{tr}(A\,dX)$ | $m\times n$： $A^\top$ |
| $\mathrm{tr}(AXB)$ | 各因子维度相容 | $\mathrm{tr}(BA\,dX)$ | $A^\top B^\top$ |

先核对变量与结果形状，再使用转置公式；不要把微分本身误写成一个梯度。

**了解。** Fréchet 可微、Gâteaux 可微与可微性之间的完整关系；知道矩阵空间可以赋予内积和范数即可。

## 5. 范数

**核心掌握常用范数及用途。**

- 1-范数： $\Vert x\Vert _1=\sum_i|x_i|$，促进稀疏。
- 2-范数： $\Vert x\Vert _2=\sqrt{x^\top x}$，欧氏长度。
- $\infty$-范数： $\Vert x\Vert _\infty=\max_i|x_i|$，最大坐标幅度。
- Frobenius 范数： $\Vert X\Vert _F^2=\sum_{ij}X_{ij}^2$，等价于把矩阵展平成向量后的 2-范数。

向量范数需满足非负且仅在零向量处为零、绝对齐次性 $\Vert cx\Vert =|c|\Vert x\Vert $ 与三角不等式。平方欧氏长度不满足绝对齐次性，所以不是范数。

矩阵算子范数定义为

$$
\Vert A\Vert _p=\max_{x\ne0}\frac{\Vert Ax\Vert _p}{\Vert x\Vert _p}.
$$

常用结论：

$$
\Vert A\Vert _1=\max_j\sum_i|A_{ij}|,
\qquad
\Vert A\Vert _\infty=\max_i\sum_j|A_{ij}|,
$$

$$
\Vert A\Vert _2=\sigma_{\max}(A).
$$

对偶范数定义为

$$
\Vert y\Vert _*=\max_{\Vert x\Vert \le1}y^\top x.
$$

它与低秩、压缩和压缩感知中的凸松弛相关。完整约束优化证明可留到专门课程。

## 6. SVD 与低秩

**核心掌握。** 任意矩阵 $X\in\mathbb R^{m\times n}$ 可做奇异值分解

$$
X=U\Sigma V^\top,
$$

完整形式中 $U\in\mathbb R^{m\times m}$、 $V\in\mathbb R^{n\times n}$ 为正交矩阵， $\Sigma\in\mathbb R^{m\times n}$，奇异值按非增顺序排列。

非零奇异值满足

$$
\sigma_i(X)=\sqrt{\lambda_i(X^\top X)}.
$$

矩阵的 2-范数为最大奇异值；Frobenius 范数为奇异值的欧氏长度：

$$
\Vert X\Vert _F=\sqrt{\sum_i\sigma_i(X)^2}.
$$

只保留前 $r$ 个奇异值得到秩至多为 $r$ 的近似，其中 $0\le r\le\min(m,n)$。Eckart–Young 定理说明 Frobenius 范数下 SVD 截断是最佳低秩近似：

$$
\Vert X-X_r\Vert _F^2=\sum_{i>r}\sigma_i^2.
$$

该结论会成为 GaLore 的核心假设：若梯度接近低秩，就在低维子空间保存优化器状态。需要同时记住它的风险——噪声梯度未必低秩，见 [[Lecture-10_Subspace-Optimizers#2. GaLore 与其收敛风险|子空间优化]]。

## 仅作了解

- 核范数（奇异值之和）和更一般的 Schatten 范数。
- SVD 的完整存在性证明、奇异值扰动界。
- 所有对偶范数公式；先熟记 $\Vert \cdot\Vert _1\leftrightarrow\Vert \cdot\Vert _\infty$ 与 $\Vert \cdot\Vert _2$ 自对偶。

## 自测

1. 写出 $f(x)=W_2\sigma(W_1x)$ 的两步链式 Jacobian。
2. 为什么反向传播需要缓存激活？
3. 矩阵 2-算子范数与最大奇异值是什么关系？
4. 若奇异值快速衰减，截断 SVD 会带来什么近似误差？

## 资料定位

| 主题 | 原始资料位置 |
|---|---|
| Jacobian、softmax、LSE、链式法则 | `01` notes PDF 页序 1、3-6；链式法则后续例题见 7-10 |
| 内积、谱定理、Frobenius 微分和乘积反传 | `02` notes PDF 页序 1-3、15-22；迹梯度表为微分规则的整理 |
| 范数、算子范数、SVD 与对偶范数 | `03` notes PDF 页序 1-2、6-7、9-14、14-18 |

截断 SVD 的最佳近似式是补充的 Eckart–Young 结论；课件重点推导 SVD、范数和对偶范数，未将该最佳近似证明作为已核实的原稿定理。

基础代数结束后，学习随机变量与期望：[[Lecture-04_Probability-and-Statistics]]。

