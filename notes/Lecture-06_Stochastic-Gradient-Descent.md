# 第 6 讲：SGD、动量与自适应优化

> **课程定位**：2026 年课程核心优化主线；主要复用 DLOpt 2024 资料。
>
> 配套文件：`materials/Lecture-06_Stochastic-Gradient-Descent/01_stochastic-gradient-descent_slides.pdf`、`02_stochastic-gradient-descent_notes.pdf`、`03_momentum-sgd_slides.pdf`、`04_momentum-sgd_notes.pdf`、`05_adaptive-sgd_slides.pdf`

## 学习目标

掌握随机目标、无偏梯度、方差与 mini-batch 的关系；能区分 SGD 的迭代速率和样本复杂度；理解 Momentum SGD 为什么不能普遍突破随机优化下界；掌握 AdaGrad、RMSProp、Adam 的更新式和适用场景。

## 1. 从经验风险到随机目标

**核心掌握。** 深度网络训练可写为

$$
\min_x F(x)=\mathbb E_{\xi\sim\mathcal D}[f(x;\xi)].
$$

数据上的经验平均只是对 $F$ 的离散近似。第 $k$ 次采样 $\xi_k$，随机梯度为 $g_k=\nabla f(x_k;\xi_k)$。令 $\mathcal F_k$ 表示采样 $\xi_k$ 之前的迭代历史（包含 $x_k$），本文采用课件的条件假设：

$$
\mathbb E[g_k\mid\mathcal F_k]=\nabla F(x_k),
\qquad
\mathbb E[\Vert g_k-\nabla F(x_k)\Vert ^2\mid\mathcal F_k]
\le\sigma^2.
$$

第一条是无偏，第二条是有界方差。真实深度学习中方差可能有重尾或依赖结构，因此这些假设是入门的简化。

SGD 更新仍是

$$
x_{k+1}=x_k-\gamma_k g_k.
$$

它每步只处理一个样本或一个小批量，计算便宜但方向带噪。

## 2. 随机问题与无偏梯度

**核心掌握。** 若完整损失为 $\frac1m\sum_i f_i(x)$，均匀随机抽取 $i$ 的梯度 $\nabla f_i(x)$ 无偏，因为

$$
\mathbb E_i[\nabla f_i(x)]=\frac1m\sum_i\nabla f_i(x).
$$

mini-batch 取 $B$ 个样本平均；给定历史后，各噪声独立、均值为零且方差至多 $\sigma^2$ 时，平均噪声方差至多 $\sigma^2/B$。只有各样本方差恰为同一个值时才有相应等式。但这并不保证 $B$ 倍加速：每次迭代多做 $B$ 倍工作，且噪声可能高度相关；无放回抽样还需另计有限总体修正。

## 3. SGD 的收敛图景

**核心掌握固定步长的噪声底与有限步长计划的区别。** 下列速率都要求前述条件无偏和有界方差，以及相应的目标假设。非凸情形还需 $F^\star=\inf_xF(x)>-\infty$；凸情形假设存在最优点且初始距离有限。

**光滑非凸。** 固定步长 $0\lt \gamma\le1/L$、 $K$ 次更新满足

$$
\frac1K\sum_{k=0}^{K-1}\mathbb E\Vert \nabla F(x_k)\Vert ^2
\le\frac{2\Delta_0}{\gamma K}+L\gamma\sigma^2,
\qquad\Delta_0=F(x_0)-F^\star.
$$

给定有限预算 $K$，取 $\gamma=\min\lbrace 1/L,c/\sqrt K\rbrace $（ $c>0$ 固定）可得到 $O(1/\sqrt K)$ 主阶。该步长在这一次运行内固定、随预算变化；它与逐步递减的 $\gamma_k$ 不同。一般的逐步递减计划需要另做加权求和分析，不能断言任意递减都给出同一速率。
- 光滑凸：合适的预算相关步长下， $\mathbb E[F(\bar x_K)]-F^\star$ 有 $O(1/\sqrt K)$ 主阶， $\bar x_K=K^{-1}\sum_{k=0}^{K-1}x_k$。
- 光滑强凸：固定步长下通常收敛到由 $\gamma,\sigma^2,\mu$ 共同决定的噪声邻域；适当递减步长可进一步缩小。

因此看到“SGD 收敛到 $x^\star$”时必须追问：是目标值、距离还是梯度范数？是否用了衰减步长？噪声假设是什么？

相较确定性 GD 的 $1/K$，SGD 的 $1/\sqrt K$ 常被称作“噪声代价”；但在大数据上，每次迭代成本 $O(1)$ 而不是 $O(m)$，总样本复杂度可能仍然更好。mini-batch 主要改善向量化吞吐和梯度质量。

| 情形 | 常见输出量 | 典型速率或现象 | 必须同时说明的条件 |
|---|---|---|---|
| 非凸 | $\min_{0\le k\lt K}\mathbb E\Vert \nabla F(x_k)\Vert ^2$ | $O(1/\sqrt K)$；固定步长另有噪声底 | 条件无偏、有界方差、 $L$-光滑、函数有下界、预算相关步长 |
| 凸 | 平均迭代的目标间隙 | $O(1/\sqrt K)$ 型主阶 | 无偏、有界方差、凸且光滑、合适步长 |
| 强凸 | 最优间隙或到最优点的距离 | 固定步长到噪声邻域；递减步长可继续缩小 | $\mu$-强凸、噪声模型、步长计划 |

不同教材会用不同步长得到不同常数，不能只抄一个 $O(\cdot)$ 而省略条件。

以一维二次目标 $f(x)=\frac\mu2x^2$、 $\mu>0$、随机梯度 $g_k=\mu x_k+\xi_k$ 为例。取确定初值、独立同分布且独立于此前历史的噪声，满足 $\mathbb E\xi_k=0$、 $\mathrm{Var}(\xi_k)=\sigma^2$，固定步长得到

$$
x_{k+1}=(1-\gamma\mu)x_k-\gamma\xi_k.
$$

当 $0\lt \gamma\mu\lt 2$ 时，稳态参数二阶矩为

$$
\lim_{k\to\infty}\mathbb E[x_k^2]
=\frac{\gamma\sigma^2}{2\mu-\gamma\mu^2}.
$$

因此小步长时约为 $\gamma\sigma^2/(2\mu)$。这里衡量的是参数距离平方；目标值误差、梯度范数或非二次目标会得到不同表达式。

这是笔记补充的可手算例子：条件零均值使交叉项为零，故 $\mathbb E[x_{k+1}^2]=(1-\gamma\mu)^2\mathbb E[x_k^2]+\gamma^2\sigma^2$，再求几何级数得到上式。仅有每次噪声的无条件均值为零而无跨时刻条件，不能据此推导。

## 4. Fourier/滤波视角的 Momentum SGD

**熟悉推导，掌握结论。** 动量 SGD 可写为

$$
m_k=\beta m_{k-1}+g_k,
\qquad
x_{k+1}=x_k-\gamma m_k.
$$

若目标 $F$ 光滑强凸，满足原定理的随机噪声等条件，并取 SGD 学习率 $\gamma_s=\gamma/(1-\beta)$、 $\beta$ 有界地远离 1、 $\gamma$ 足够小，则课件第 12-13 页描述的小步极限下，动量方法与标准 SGD 的行为近似等价。这不是逐步轨迹严格相等，也不能直接推广到一般非凸网络。动量可以平滑更新，但不应被理解为所有问题都能把 $1/\sqrt K$ 加速成 $1/K$。

Nesterov momentum SGD 在外推点 $y_{k-1}$ 处采样梯度。理论上，在噪声 $\sigma^2$ 很小或迭代数 $K$ 不太大等条件下可能获得更好的瞬态；当 $K\to\infty$ 且噪声主导时，随机一阶下界限制仍然存在。

**工程上很有用：** 大模型训练中动量能显著改善吞吐和优化质量；这与“最坏情形阶数不变”并不矛盾。

## 5. 自适应 SGD

**核心掌握更新式。** 自适应方法本质上是带对角预条件的 SGD：

$$
x_{k+1}=x_k-\gamma P_k g_k.
$$

$P_k$ 根据历史及当前梯度改变每个坐标的有效步长。它们尝试缓解病态问题中的锯齿，但不直接使用 Hessian。下列平方、平方根、乘除均按坐标执行；状态从零初始化， $\eta>0$、 $\epsilon>0$，Momentum/RMSProp 的 $0\le\beta\lt1$。若动量递推从 $k=0$ 开始，应取 $m_{-1}=0$。

### AdaGrad

$$
v_k=v_{k-1}+g_k\odot g_k,
\qquad
x_{k+1}=x_k-\frac{\eta}{\sqrt{v_k}+\epsilon}g_k.
$$

累积历史平方梯度的坐标范数，使常见坐标的有效步长变小。 $v_k$ 单调增加，因此长期训练的有效学习率往往衰减得过快，参数一旦在后期需要明显更新，可能反映迟钝。

### RMSProp

$$
v_k=\beta v_{k-1}+(1-\beta)g_k\odot g_k,
\qquad
x_{k+1}=x_k-\frac{\eta}{\sqrt{v_k}+\epsilon}g_k.
$$

使用指数滑动平均，避免无衰减地累积全部历史；有界梯度下 $v_k$ 保持有界。它不保证任意无界梯度序列的 $v_k$ 都不发散。 $0\le\beta\lt 1$ 越大，记忆越长。

### Adam

为避免首步偏差修正分母为零，本小节从 $k=1$ 开始计数，初始化 $m_0=v_0=0$，在 $x_k$ 处取梯度。平方根、除法和乘法均逐坐标执行， $0\le\beta_1,\beta_2\lt 1$、 $\epsilon>0$。

$$
m_k=\beta_1m_{k-1}+(1-\beta_1)g_k,
$$

$$
v_k=\beta_2v_{k-1}+(1-\beta_2)g_k\odot g_k,
$$

$$
\hat m_k=\frac{m_k}{1-\beta_1^k},
\qquad
\hat v_k=\frac{v_k}{1-\beta_2^k},
$$

$$
x_{k+1}=x_k-\frac{\eta}{\sqrt{\hat v_k}+\epsilon}\hat m_k.
$$

除以 $1-\beta^k$ 使历史权重和为 1；梯度矩平稳时消除零初始化带来的矩估计低估，非平稳时仍是对历史的加权平均，不保证等于当前梯度矩。Adam 同时有一阶平滑（类似动量）和二阶尺度估计（类似 RMSProp）。

adaptive slides 第 18 页省略偏差修正；这里按原始 Adam 论文补全。它修正的是零初始化导致的矩估计偏置，不意味着经过非线性除法后的最终更新方向就是无偏梯度。

## 6. 自适应方法不是免费的

**熟悉。** Adam 的 $m,v$ 各占一份与参数同形状的状态；显存压力会在 [[Lecture-09_Advanced-Optimizers]] 成为重点。AdaGrad/RMSProp/Adam 的自适应步长也不自动保证适合所有任务或达到更好泛化；学习率、权重衰减和训练时长仍需验证。

## 方法比较

| 方法 | 保存的历史 | 主要优势 | 主要限制 |
|---|---|---|---|
| SGD | 无 | 简单、内存低、随机噪声可帮助探索 | 病态问题慢 |
| Momentum SGD | 一阶动量 | 平滑噪声、改善方向一致性 | 不普遍突破随机下界 |
| AdaGrad | 累积平方梯度 | 自动缩放、坐标自适应 | 有效学习率持续衰减 |
| RMSProp | 滑动平方梯度 | 适合非平稳噪声 | 仍需调超参数 |
| Adam | 一阶动量 + 二阶累积 | 稳定、易用 | 优化器状态显存高 |

## 常见误解

- 固定步长强凸 SGD 的误差邻域不只由 $\sigma^2/\mu^2$ 决定，学习率也进入表达式。
- “收敛到驻点”不等于“收敛到全局最优”；不同凸性下的误差量不同。
- Momentum 在工程上常显著加速，但不能在所有噪声主导的随机优化问题上突破一阶下界。
- 增大 batch 会降低独立采样的梯度方差，但不会自动让总计算成本按同样倍数下降。

## 仅作了解

- mini-batch 收敛定理的完整常数推导与不同采样顺序假设。
- Momentum SGD 的 Fourier 分析和下界构造。
- Adam 的所有收敛变体与泛化争议；先掌握更新式和成本。

## 自测

1. 无偏梯度是否意味着梯度没有噪声？二者能否同时成立？
2. 为什么固定步长强凸 SGD 往往只能到噪声邻域？
3. 写出 Momentum SGD、AdaGrad、RMSProp、Adam 的更新式。
4. 小批量增大 $B$ 倍，方差与每次迭代成本各如何变化？

## 资料定位

| 主题 | 原始资料位置 |
|---|---|
| SGD 建模、条件假设、非凸/凸/强凸界 | `01` slides PDF 页序 11、13-15、18、22-23 |
| mini-batch 与条件独立性 | `01` PDF 页序 27-29 |
| Momentum 更新与近似等价条件 | `03` momentum slides PDF 页序 8-9、12-13 |
| Nesterov SGD 与随机下界 | `03` PDF 页序 14-18 |
| AdaGrad、RMSProp、Adam | `05` adaptive slides PDF 页序 9、14、17-18；偏差修正另见 [[REFERENCES]] 中原始 Adam 论文 |

编号对应顶部配套文件，页码采用 PDF 页序。一维稳态二阶矩是本笔记的独立补充推导；不将它冒充课件中的一般强凸定理。

了解参数如何进入 Transformer 后，才能把这些优化器成本落实为实际显存：[[Lecture-07_Transformer-and-LLM-I]]。

