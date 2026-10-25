# 第 9 讲：面向 LLM 的高级优化器

> **课程定位**：2026 年课程核心高级优化器。官网概要明确列出 Shampoo、SOAP、Muon、Lite、MuonM；本课件还包含 Adafactor、Adam-mini、Sophia 等扩展内容。
>
> 配套文件：`materials/Lecture-09_Advanced-Optimizers/01_advanced-optimizers_slides.pdf`

## 学习目标

围绕“省优化器状态”和“利用曲率”两条线理解 Adafactor、Adam-mini、Shampoo、SOAP、Muon、Sophia。必须能说明每种方法压缩了什么、增加什么计算、对模型结构作了什么假设，而不是只记名称。

## 1. 先算 Adam 的账

**核心掌握。** 统一使用

$$
\text{Memory}=\text{Model}+\text{Gradient}
+\text{Optimizer states}+\text{Activations}.
$$

Adam 每个参数需要一阶矩 $m$ 与二阶矩 $v$。若参数数为 $\Phi$，模型、梯度和两个优化器状态各占 $\Phi$ 个元素，合计 $4\Phi$。全部用 FP32 时约 $16\Phi$ 字节；若模型、梯度与状态都降到 16-bit，则约 $8\Phi$ 字节。7B 参数分别约为 112GB 和 56GB（十进制），尚未计入激活和额外缓冲区。因此 Adam 的 $m,v$ 是高级优化器的首要压缩目标。

## 2. Adafactor

**熟悉公式和假设。** 对矩阵参数 $W\in\mathbb R^{m\times n}$，Adam 的 $V_t$ 也约 $m\times n$。Adafactor 用行、列边缘统计近似它。令 $G\in\mathbb R^{m\times n}$，忽略时间指数和更细的数值保护，行、列平方梯度均值为

$$
r_i=\frac1n\sum_{j=1}^{n}G_{ij}^2,
\qquad
c_j=\frac1m\sum_{i=1}^{m}G_{ij}^2.
$$

一种常见的因子化二阶矩估计为

$$
\widehat V_{ij}\approx
\frac{r_i c_j}{\frac1m\sum_{k=1}^{m}r_k}.
$$

若要维护历史信息，可对 $r_i,c_j$ 分别使用指数滑动平均；不同实现还会在分母、偏置修正和数值下限上有所差异。关键是保存并更新 $O(m+n)$ 个统计量，把 Adam 的 $V$ 状态从

$$
O(mn)\quad\text{降到}\quad O(m+n).
$$

它还可以选择去掉一阶动量，只保留压缩后的二阶信息。

一个 $2\times3$ 的数值检查：若

$$
G=
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix},
$$

则 $r=(14/3,77/3)$、 $c=(8.5,14.5,22.5)$、行均值 $=91/6$。于是

$$
\widehat V_{11}
=\frac{(14/3)(8.5)}{91/6}
=\frac{34}{13}\approx2.615,
$$

而真实 $V_{11}=1$。这说明它是低成本近似，并非精确分解。

**收益与代价：**

- 显著减少优化器状态；
- 低秩假设若不成立会带来近似误差；
- 分解、RMS 计算与较小张量操作增加额外计算/内核启动；
- 不是在任何矩阵上都“免费”获得 Adam 的精度。

## 3. Adam-mini：按 Hessian 近似块结构共享学习率

**核心掌握设计思想。** Adam 给每个坐标一个二阶状态，而 Transformer 权重的 Hessian 被认为近似块对角：块内强耦合，块间联系较弱。Adam-mini 用结构知识把参数分成块，每块只维护一个学习率尺度。

典型的划分：

- Q/K 投影：按 attention head 成块；
- attention 输出与 MLP：按神经元/行成块；
- embedding 与输出层：按 token 或列成块。

Adam-mini **保留完整的一阶动量 $m$**，每个参数仍有一个一阶状态；它把逐参数的二阶自适应尺度 $v$ 改写成每个参数块共享的尺度。因此优化器状态元素数可从约 $2mn$ 降到约 $mn$，论文报告约 50% 的优化器内存节省。与此同时，逐元素 sqrt/div 可变为块级标量运算。该设计不要求改变模型前向逻辑。

**掌握边界：** “Hessian 近块对角”是结构性假设，不是精确全局结论；实际效果依赖模型结构与参数分组。

## 4. 二阶预条件的基本思想

**熟悉。** 牛顿方向为

$$
d=-H_f(x)^{-1}\nabla f(x).
$$

这个公式首先要求 Hessian 可逆；要把它解释为曲率自适应的下降方向，还需 Hessian 正定。非凸目标的 Hessian 可能不定或奇异，通常需阻尼或其他修正。对 $P$ 个参数，完整 Hessian 有 $P^2$ 个元素，大模型中存储、求逆和更新都不可行。高级方法因此寻找低维或结构近似：

- 对角 Hessian；
- Kronecker 因子；
- 低秩特征基；
- 矩阵参数的左右两侧曲率。

## 5. Shampoo 与 SOAP

**熟悉名称与核心机制。** 对参数矩阵 $W_t\in\mathbb R^{m\times n}$ 与梯度 $G_t\in\mathbb R^{m\times n}$，课件 PDF 页序 36 的 Shampoo 矩阵版采用累积统计：初始化 $L_0=\epsilon I_m$、 $R_0=\epsilon I_n$（ $\epsilon>0$），然后

$$
L_t=L_{t-1}+G_tG_t^\top,\qquad
R_t=R_{t-1}+G_t^\top G_t.
$$

更新近似使用

$$
W_{t+1}=W_t-\eta L_t^{-1/4}G_tR_t^{-1/4}.
$$

$L_t$ 为 $m\times m$、 $R_t$ 为 $n\times n$；正定初始化保证这里的负矩阵幂有定义。 $-1/4$ 来自矩阵参数两侧的预条件，而不是普通 $2\times2$ Hessian 的直接反演。采用 EMA 统计的实现是另一种变体，应同时声明衰减系数和数值保护；不能把不同版本的累积式随意拼接。实际系统通过特征分解、矩阵根与周期更新控制开销。

SOAP 把这套想法接回一阶优化：在 Shampoo 提供的特征基中运行类似 Adam/Adafactor 的更新。可把它理解为“用二阶方法寻找坐标，在好坐标里做一阶优化”。

**了解：** Shampoo 的 Kronecker 近似定理、完整特征根更新和 SOAP 实现细节。初次学习先理解两侧统计和基变换；此处等级是学习建议。

## 6. Muon：正交化矩阵更新

**熟悉。** Muon 把隐藏层权重保持为矩阵而不是展成长向量。它先累积矩阵动量

$$
M_t=\beta M_{t-1}+G_t,
$$

再近似求 $M_t$ 的极分解/正交因子（常记 $\mathrm{Orth}(M_t)$），沿更均衡的奇异方向更新：

$$
W_{t+1}=W_t-\eta\mathrm{Orth}(M_t).
$$

正交化避免少数大奇异值主导更新，使各方向获得更均衡的步长。完整 SVD 太贵，因此使用 Newton–Schulz 迭代近似正交化；每次迭代为矩阵乘法，能较好利用 GPU。

Muon 通常用于矩阵型隐藏层，embedding、输出层等仍可能沿用 Adam；它不是“没有优化器状态”，而是把更新结构与模型层的线性代数结构匹配起来。

本节是机制简式。课件 PDF 页序 44 使用归一化动量 $M_t=\beta M_{t-1}+(1-\beta)G_t$，先生成负方向再正交化；对精确极因子，正比例缩放不改变该因子，上式把符号放到最终减号中。有限次 Newton–Schulz、归一化和层尺度修正属于具体实现，不能用这个简式逐行替代。

## 7. Sophia：曲率对角与逐坐标裁剪

**熟悉。** Sophia 用对角 Hessian 估计 $h_t$ 与一阶动量，更新可抽象为

$$
W_{t+1}=W_t-\eta\,
\mathrm{clip}\!\left(
\frac{m_t}{\max(\rho h_t,\epsilon)},1\right).
$$

其中 $\rho>0$ 控制曲率尺度， $\epsilon>0$；max 和除法逐坐标执行， $\mathrm{clip}(z,1)=\min\{1,\max\{-1,z\}\}$ 也逐坐标执行。这里省略权重衰减等工程项。Hessian 对角估计可以使用：

- Hutchinson 估计：固定 $H$ 时，若随机向量满足 $\mathbb E[uu^\top]=I$，则

  $$
  \mathbb E[u\odot(Hu)]=\mathrm{diag}(H).
  $$

  独立 Rademacher 向量（各坐标为等概率的 $\pm1$）或标准高斯向量 $u\sim\mathcal N(0,I)$ 都满足该条件；任意随机向量不自动满足；
- Gauss–Newton–Bartlett（GNB）：在模型分布下，用平方梯度近似 Fisher/对角 Hessian。

Bartlett identity 在模型分布采样和允许微分与积分交换等正则条件下，连接期望负对数似然 Hessian 与得分外积/Fisher 信息；不是任意真实标签上的单次平方梯度都等于 Hessian。完整推导按讲义了解。Sophia 的重点是便宜曲率与裁剪稳定性，而非构造完整 Hessian。

## 8. 方法对照

| 方法 | 节省或利用什么 | 额外成本 | 关键假设 |
|---|---|---|---|
| Adafactor | 把矩阵二阶状态压成行列因子 | 因子统计 | $V$ 可低秩近似 |
| Adam-mini | 每块共享二阶尺度 | 参数分组 | Transformer Hessian 近似块对角 |
| Shampoo | 左右 Kronecker 曲率预条件 | 矩阵根、周期分解 | 参数拥有矩阵结构 |
| SOAP | 在二阶特征基中做一阶更新 | 基变换与状态 | 稳定特征基有帮助 |
| Muon | 正交化矩阵动量 | Newton–Schulz 迭代 | 隐藏层更适合矩阵几何 |
| Sophia | 对角曲率 + 裁剪 | Hessian 估计 | 对角曲率与裁剪足以稳定 |

## 仅作了解

- Shampoo 和 SOAP 完整论文中的硬件优化、分片实现。
- Muon 的极分解理论、Newton–Schulz 收敛分析。
- Sophia 的 Bartlett identity 全部推导；初次学习优先理解“平方梯度为何可作曲率代理”的条件和直觉。
- 各论文的具体训练基准数值；它们随模型、token 数和硬件变化。

## 课程对齐说明

官网 Lecture 9 概要列出 **Shampoo、SOAP、Muon、Lite、MuonM**。当前 72 页课件标题和正文能定位到前三种，但未发现 Lite、MuonM 的独立展开章节。这两项应标记为“需结合课堂更新确认”，不能仅凭课件目录判断其考试范围。Adafactor、Adam-mini 和 Sophia 虽出现在课件中，复习时仍应以教师当年明确范围为准。

## 常见误解

- Adafactor 的 $O(m+n)$ 是优化器状态规模，不是把模型参数或前向计算降到该规模。
- Adam-mini 保留完整一阶动量，只把逐参数二阶尺度改为块级尺度。
- Shampoo 的矩阵根和 Muon 的正交化都依赖参数具有矩阵结构；把所有权重展平后直接套公式会丢失这层含义。
- “省内存”不等于“更快”：因子分解、SVD、Newton–Schulz 或 Hessian 估计都有额外计算和实现开销。

## 自测

1. Adam 相比 SGD 多占多少量级的优化器状态？
2. Adafactor 如何把 $O(mn)$ 降成 $O(m+n)$？代价是什么？
3. Adam-mini 的分组利用了 Transformer 的哪类结构？
4. FlashAttention 减少 HBM 访问，Muon 减少矩阵动量的方向不平衡，两者分别属于什么层面的优化？

## 资料定位

| 主题 | 原始资料位置 |
|---|---|
| Adam 显存与 Adafactor | PDF 页序 2-11；行列因子见 5-9 |
| Adam-mini 与分组 | PDF 页序 12-25；分组见 18-20 |
| 二阶方法、Shampoo、SOAP | PDF 页序 26-41；Shampoo 矩阵算法见 36，SOAP 见 40-41 |
| Muon | PDF 页序 42-51；更新见 44 |
| Sophia、Hutchinson、GNB/Bartlett、裁剪 | PDF 页序 52-72；估计见 56-60，更新与裁剪见 68-70 |

**页码约定：** 本 PDF 的页序与部分页脚编号不一致，例如 PDF 第 36 页页脚为 38，第 69 页页脚为 74；以上全部使用 PDF 页序，避免把页脚数字当成文件页码。

下一讲把“压缩优化器状态”和“降低梯度/参数子空间维度”连起来：[[Lecture-10_Subspace-Optimizers]]。

