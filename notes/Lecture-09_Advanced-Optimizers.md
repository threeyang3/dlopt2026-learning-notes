# 第 9 讲：面向 LLM 的高级优化器

> 配套文件：`materials/Lecture-09_Advanced-Optimizers/01_advanced-optimizers_slides.pdf`

## 学习目标

围绕“省优化器状态”和“利用曲率”两条线理解 Adafactor、Adam-mini、Shampoo、SOAP、Muon、Sophia。必须能说明每种方法压缩了什么、增加什么计算、对模型结构作了什么假设，而不是只记名称。

## 1. 先算 Adam 的账

**核心掌握。** 统一使用

\[
\text{Memory}=\text{Model}+\text{Gradient}
+\text{Optimizer states}+\text{Activations}.
\]

Adam 每个参数需要一阶矩 \(m\) 与二阶矩 \(v\)。若参数为 \(\Phi\)，二者各占 \(\Phi\)，模型、梯度、优化器状态合计约 \(4\Phi\) 个元素；在 FP32 下粗略为 \(16\Phi\) 字节。7B 参数仅这三项就常在数十 GB 量级，尚未计入激活。因此 Adam 的 \(m,v\) 是高级优化器的首要压缩目标。

## 2. Adafactor

**熟悉公式和假设。** 对矩阵参数 \(W\in\mathbb R^{m\times n}\)，Adam 的 \(V_t\) 也约 \(m\times n\)。Adafactor 假设它可由行因子 \(R_t\in\mathbb R^m\) 与列因子 \(C_t\in\mathbb R^n\) 近似：

\[
V_t\approx R_t\cdot \frac{\mathbf 1_n^\top}{n}
\frac{\mathbf 1_m}{m}\cdot C_t.
\]

实际实现的行、列因子按平方梯度的边缘统计构造，再把状态从

\[
O(mn)\quad\text{降到}\quad O(m+n).
\]

它还可以去掉一阶动量，只保留压缩后的二阶信息。

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

与 Adam 的 \(2\Phi\) 优化器状态相比，Adam-mini 把 \(m,v\) 降到约 \(m\) 级别（只保留块统计），同时减少逐元素 sqrt/div。它复用 Adam 的块内统计设计，不要求改变模型前向逻辑。

**掌握边界：** “Hessian 近块对角”是结构性假设，不是精确全局结论；实际效果依赖模型结构与参数分组。

## 4. 二阶预条件的基本思想

**熟悉。** 牛顿方向为

\[
d=-H_f(x)^{-1}\nabla f(x).
\]

它会在曲率大的方向小步、曲率小的方向大步，但完整 Hessian 有 \(P^2\) 个元素，存储、求逆和更新都不可行。高级方法因此寻找低维或结构近似：

- 对角 Hessian；
- Kronecker 因子；
- 低秩特征基；
- 矩阵参数的左右两侧曲率。

## 5. Shampoo 与 SOAP

**熟悉名称与核心机制。** 对参数矩阵 \(W\in\mathbb R^{m\times n}\) 与梯度 \(G\in\mathbb R^{m\times n}\)，Shampoo 累积左右两边的二阶信息，例如

\[
L_t=\beta L_{t-1}+GG^\top,\qquad
R_t=\beta R_{t-1}+G^\top G.
\]

更新近似使用

\[
W_{t+1}=W_t-\eta L_t^{-1/4}G_tR_t^{-1/4}.
\]

\(-1/4\) 来自在两侧进行 Kronecker 型预条件后的综合效果，而不是普通 \(2\times2\) Hessian 的直接反演。实际系统通过特征分解、矩阵根与周期更新控制开销。

SOAP 把这套想法接回一阶优化：在 Shampoo 提供的特征基中运行类似 Adam/Adafactor 的更新。可把它理解为“用二阶方法寻找坐标，在好坐标里做一阶优化”。

**了解：** Shampoo 的 Kronecker 近似定理、完整特征根更新和 SOAP 实现细节。期中复习先掌握两侧统计和基变换。

## 6. Muon：正交化矩阵更新

**熟悉。** Muon 把隐藏层权重保持为矩阵而不是展成长向量。它先累积矩阵动量

\[
M_t=\beta M_{t-1}+G_t,
\]

再近似求 \(M_t\) 的极分解/正交因子（常记 \(\operatorname{Orth}(M_t)\)），沿更均衡的奇异方向更新：

\[
W_{t+1}=W_t-\eta\operatorname{Orth}(M_t).
\]

正交化避免少数大奇异值主导更新，使各方向获得更均衡的步长。完整 SVD 太贵，因此使用 Newton–Schulz 迭代近似正交化；每次迭代为矩阵乘法，能较好利用 GPU。

Muon 通常用于矩阵型隐藏层，embedding、输出层等仍可能沿用 Adam；它不是“没有优化器状态”，而是把更新结构与模型层的线性代数结构匹配起来。

## 7. Sophia：曲率对角与逐坐标裁剪

**熟悉。** Sophia 用对角 Hessian 估计 \(h_t\) 与一阶动量，更新可抽象为

\[
W_{t+1}=W_t-\eta\,
\operatorname{clip}\!\left(
\frac{m_t}{\max(\rho h_t,\epsilon)},1\right).
\]

其中 \(\rho\) 控制曲率阈值，clip 限定单次坐标更新幅度。Hessian 对角估计可以使用：

- Hutchinson 估计：随机向量 \(u\) 下，用 \(u\odot(Hu)\) 的无偏估计；
- Gauss–Newton–Bartlett（GNB）：在模型分布下，用平方梯度近似 Fisher/对角 Hessian。

Bartlett identity 给出一定协方差意义下“平方得分 ≈ Fisher 信息”的联系；完整推导按讲义了解。Sophia 的重点是便宜曲率与裁剪稳定性，而非构造完整 Hessian。

## 8. 方法对照

| 方法 | 节省或利用什么 | 额外成本 | 关键假设 |
|---|---|---|---|
| Adafactor | 把矩阵二阶状态压成行列因子 | 因子统计 | \(V\) 可低秩近似 |
| Adam-mini | 每块共享二阶尺度 | 参数分组 | Transformer Hessian 近似块对角 |
| Shampoo | 左右 Kronecker 曲率预条件 | 矩阵根、周期分解 | 参数拥有矩阵结构 |
| SOAP | 在二阶特征基中做一阶更新 | 基变换与状态 | 稳定特征基有帮助 |
| Muon | 正交化矩阵动量 | Newton–Schulz 迭代 | 隐藏层更适合矩阵几何 |
| Sophia | 对角曲率 + 裁剪 | Hessian 估计 | 对角曲率与裁剪足以稳定 |

## 仅作了解

- Shampoo 和 SOAP 完整论文中的硬件优化、分片实现。
- Muon 的极分解理论、Newton–Schulz 收敛分析。
- Sophia 的 Bartlett identity 全部推导；期中优先掌握“平方梯度为何可作曲率代理”的直觉。
- 各论文的具体训练基准数值；它们随模型、token 数和硬件变化。

## 自测

1. Adam 相比 SGD 多占多少量级的优化器状态？
2. Adafactor 如何把 \(O(mn)\) 降成 \(O(m+n)\)？代价是什么？
3. Adam-mini 的分组利用了 Transformer 的哪类结构？
4. FlashAttention 减少 HBM 访问，Muon 减少矩阵动量的方向不平衡，两者分别属于什么层面的优化？

下一讲把“压缩优化器状态”和“降低梯度/参数子空间维度”连起来：[[Lecture-10_Subspace-Optimizers]]。

