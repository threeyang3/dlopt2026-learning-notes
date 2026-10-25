# 第 6 讲：SGD、动量与自适应优化

> 配套文件：`materials/Lecture-06_Stochastic-Gradient-Descent/01_stochastic-gradient-descent_slides.pdf`、`02_stochastic-gradient-descent_notes.pdf`、`03_momentum-sgd_slides.pdf`、`04_momentum-sgd_notes.pdf`、`05_adaptive-sgd_slides.pdf`

## 学习目标

掌握随机目标、无偏梯度、方差与 mini-batch 的关系；能区分 SGD 的迭代速率和样本复杂度；理解 Momentum SGD 为什么不能普遍突破随机优化下界；掌握 AdaGrad、RMSProp、Adam 的更新式和适用场景。

## 1. 从经验风险到随机目标

**核心掌握。** 深度网络训练可写为

\[
\min_x F(x)=\mathbb E_{\xi\sim\mathcal D}[f(x;\xi)].
\]

数据上的经验平均只是对 \(F\) 的离散近似。第 \(k\) 次采样 \(\xi_k\)，随机梯度为 \(g_k=\nabla f(x_k;\xi_k)\)。本文采用常见假设：

\[
\mathbb E[g_k\mid x_k]=\nabla F(x_k),
\qquad
\mathbb E[\|g_k-\nabla F(x_k)\|^2\mid x_k]
\le\sigma^2.
\]

第一条是无偏，第二条是有界方差。真实深度学习中方差可能有重尾或依赖结构，因此这些假设是入门的简化。

SGD 更新仍是

\[
x_{k+1}=x_k-\gamma_k g_k.
\]

它每步只处理一个样本或一个小批量，计算便宜但方向带噪。

## 2. 随机问题与无偏梯度

**核心掌握。** 若完整损失为 \(\frac1m\sum_i f_i(x)\)，随机抽取 \(i\) 的梯度 \(\nabla f_i(x)\) 无偏，因为

\[
\mathbb E_i[\nabla f_i(x)]=\frac1m\sum_i\nabla f_i(x).
\]

mini-batch 取 \(B\) 个样本平均，方差从 \(\sigma^2\) 降到约 \(\sigma^2/B\)（独立采样时）。但这并不保证 \(B\) 倍加速：每次迭代多做 \(B\) 倍工作，且噪声可能高度相关。

## 3. SGD 的收敛图景

**核心掌握“收敛到邻域”而非精确最优。**

- 光滑非凸：\(\min_{k\le K}\mathbb E\|\nabla F(x_k)\|^2=O(1/\sqrt K)\)；更严格地说，常数步长会停在与噪声有关的邻域，衰减步长可让梯度项趋零。
- 光滑凸：平均迭代的 \(F(\bar x_K)-F^\star\) 常达到 \(O(1/\sqrt K)\) 的主阶。
- 光滑强凸：在强噪声、固定步长下只收敛到 \(\sigma^2/(\mu^2)\) 量级的邻域；适当递减步长可进一步缩小。

因此看到“SGD 收敛到 \(x^\star\)”时必须追问：是目标值、距离还是梯度范数？是否用了衰减步长？噪声假设是什么？

相较确定性 GD 的 \(1/K\)，SGD 的 \(1/\sqrt K\) 常被称作“噪声代价”；但在大数据上，每次迭代成本 \(O(1)\) 而不是 \(O(m)\)，总样本复杂度可能仍然更好。mini-batch 主要改善向量化吞吐和梯度质量。

## 4. Fourier/滤波视角的 Momentum SGD

**熟悉推导，掌握结论。** 动量 SGD 可写为

\[
m_k=\beta m_{k-1}+g_k,
\qquad
x_{k+1}=x_k-\gamma m_k.
\]

若取等价的 SGD 学习率 \(\gamma_s=\gamma/(1-\beta)\)，且 \(\beta\) 不接近 1、\(\gamma\) 足够小，则小步极限下动量方法的平均行为与标准 SGD 相近。它主要抑制高频噪声、平滑更新，不应被理解为所有问题都能把 \(1/\sqrt K\) 加速成 \(1/K\)。

Nesterov momentum SGD 在外推点 \(y_{k-1}\) 处采样梯度。理论上，在噪声 \(\sigma^2\) 很小或迭代数 \(K\) 不太大等条件下可能获得更好的瞬态；当 \(K\to\infty\) 且噪声主导时，随机一阶下界限制仍然存在。

**工程上很有用：** 大模型训练中动量能显著改善吞吐和优化质量；这与“最坏情形阶数不变”并不矛盾。

## 5. 自适应 SGD

**核心掌握更新式。** 自适应方法本质上是带对角预条件的 SGD：

\[
x_{k+1}=x_k-\gamma P_k g_k.
\]

\(P_k\) 根据历史梯度自动改变每个坐标的有效步长。它们缓解病态问题中的锯齿，但不直接使用 Hessian。

### AdaGrad

\[
v_k=v_{k-1}+g_k\odot g_k,
\qquad
x_{k+1}=x_k-\frac{\eta}{\sqrt{v_k}+\epsilon}g_k.
\]

累积历史平方梯度的坐标范数，使常见坐标的有效步长变小。\(v_k\) 单调增加，因此长期训练的有效学习率往往衰减得过快，参数一旦在后期需要明显更新，可能反映迟钝。

### RMSProp

\[
v_k=\beta v_{k-1}+(1-\beta)g_k\odot g_k,
\qquad
x_{k+1}=x_k-\frac{\eta}{\sqrt{v_k}+\epsilon}g_k.
\]

使用指数滑动平均避免 \(v_k\to\infty\)，更适合非平稳目标。\(\beta\) 越大，记忆越长。

### Adam

\[
m_k=\beta_1m_{k-1}+(1-\beta_1)g_k,
\]

\[
v_k=\beta_2v_{k-1}+(1-\beta_2)g_k\odot g_k,
\]

\[
\hat m_k=\frac{m_k}{1-\beta_1^k},
\qquad
\hat v_k=\frac{v_k}{1-\beta_2^k},
\]

\[
x_{k+1}=x_k-\frac{\eta}{\sqrt{\hat v_k}+\epsilon}\hat m_k.
\]

偏差修正消除初值为 0 造成的低估。Adam 同时有一阶平滑（类似动量）和二阶尺度估计（类似 RMSProp）。

## 6. 自适应方法不是免费的

**熟悉。** Adam 的 \(m,v\) 各占一份与参数同形状的状态；显存压力会在 [[Lecture-09_Advanced-Optimizers]] 成为重点。AdaGrad/RMSProp/Adam 的自适应步长也不自动保证适合所有任务或达到更好泛化；学习率、权重衰减和训练时长仍需验证。

## 方法比较

| 方法 | 保存的历史 | 主要优势 | 主要限制 |
|---|---|---|---|
| SGD | 无 | 简单、内存低、随机噪声可帮助探索 | 病态问题慢 |
| Momentum SGD | 一阶动量 | 平滑噪声、改善方向一致性 | 不普遍突破随机下界 |
| AdaGrad | 累积平方梯度 | 自动缩放、坐标自适应 | 有效学习率持续衰减 |
| RMSProp | 滑动平方梯度 | 适合非平稳噪声 | 仍需调超参数 |
| Adam | 一阶动量 + 二阶累积 | 稳定、易用 | 优化器状态显存高 |

## 仅作了解

- mini-batch 收敛定理的完整常数推导与不同采样顺序假设。
- Momentum SGD 的 Fourier 分析和下界构造。
- Adam 的所有收敛变体与泛化争议；先掌握更新式和成本。

## 自测

1. 无偏梯度是否意味着梯度没有噪声？二者能否同时成立？
2. 为什么固定步长强凸 SGD 往往只能到噪声邻域？
3. 写出 Momentum SGD、AdaGrad、RMSProp、Adam 的更新式。
4. 小批量增大 \(B\) 倍，方差与每次迭代成本各如何变化？

了解参数如何进入 Transformer 后，才能把这些优化器成本落实为实际显存：[[Lecture-07_Transformer-and-LLM-I]]。

