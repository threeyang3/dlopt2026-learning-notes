# 原始资料对照核查

核查日期：2026-10-25。基准为提交 `a231c2b`，本页记录在此基础上的修订。对象是第 5、6、9、10、11、12 讲笔记的关键公式、算法机制及其相邻条件，另核对全部 14 讲的官网主题与学习等级。不是对 35 份 PDF 的全部公式逐项证明，也未核验课堂讲授或考试口径。

方法：先读对应课件页的文本，再对提取缺失/顺序敏感的公式直接查看 PDF 原页；对原稿简写和补充结论用明确推导或原论文区分。数值测试作为辅助证据，文案字符串不作为数学正确性的独立证据。

**页码均指从 1 开始的 PDF 页序。** 课件页脚可能与之不同，尤其第 9 讲。例如该 PDF 第 36 页页脚为 38，第 69 页页脚为 74。

## 1. 第 5 讲：确定性优化

来源：[GD slides](materials/Lecture-05_Gradient-Descent/01_convexity-gradient-descent_slides.pdf)、[加速 slides](materials/Lecture-05_Gradient-Descent/03_accelerated-gradient-descent_slides.pdf)。

| 笔记结论 | 原始页序 | 核查结果与处理 |
|---|---|---|
| 凸性一阶下界、强凸、梯度 Lipschitz | GD 7-9 | 保留定义，明确开凸定义域/无约束空间；一阶下界用于分析，不能说 GD 最小化线性下界 |
| GD 更新与下降引理 | GD 11、14 | 公式相符；非驻点才有严格下降，保留非严格不等式 |
| 非凸驻点度量与求和界 | GD 17、25 | 原稿将梯度平方误差记为精度；笔记区分梯度范数与其平方，补函数下界并统一 $K$ 次更新 |
| 凸 GD 函数间隙 | GD 19 | 课件步长 $1/(2L)$；笔记采用标准 $1/L$ 版本，明确不照搬原稿常数；独立说明见下文 |
| 强凸 GD 收缩 | GD 21 | 原稿使用 $2/(L+\mu)$；笔记同时列 $1/L$ 的标准版本，分别配距离收缩因子 |
| heavy-ball 两种递推 | 加速 5、8、10-18 | 位移式与速度式一致；不把二次目标的加速结论推广到一般目标 |
| NAG 外推、动量、函数值速率 | 加速 19、22-24 | 补上凸/强凸的不同动量计划，转换更新索引，明确速率约束函数间隙 |
| Anderson、最坏情形下界 | 加速 26-27、30 | 课件历史点组合是简写；笔记按下列原论文 Algorithm 1 保留完整固定点映射、历史长度和条件，不当作课件原式逐字抄写 |

凸 GD 补充版本的独立依据：令 $x^\star$ 为最优点，凸且 $L$-光滑、 $\gamma=1/L$ 时，二次上界和凸性给出

$$
f(x_{k+1})-f^\star
\le\frac L2\bigl(\|x_k-x^\star\|^2-\|x_{k+1}-x^\star\|^2\bigr).
$$

求和并利用函数值单调性可得 $f(x_K)-f^\star\le L\|x_0-x^\star\|^2/(2K)$，其中 $K\ge1$。这解释笔记的步长为何成立，不依赖测试里的字符串。

## 2. 第 6 讲：随机与自适应优化

来源：[SGD slides](materials/Lecture-06_Stochastic-Gradient-Descent/01_stochastic-gradient-descent_slides.pdf)、[Momentum slides](materials/Lecture-06_Stochastic-Gradient-Descent/03_momentum-sgd_slides.pdf)、[Adaptive slides](materials/Lecture-06_Stochastic-Gradient-Descent/05_adaptive-sgd_slides.pdf)。

| 笔记结论 | 原始页序 | 核查结果与处理 |
|---|---|---|
| 目标、SGD 更新、无偏与方差 | SGD 11、13-14 | 条件化对象改为采样前历史，有限和采样明确均匀 |
| mini-batch 方差 $1/B$ | SGD 27-29 | 依赖给定历史后批内独立的样本，不推广到任意相关批次 |
| 非凸噪声底及 $1/\sqrt K$ | SGD 15、18 | 补完整有限步界；区分预算相关的固定步长与逐步递减，不笼统给后者同一速率 |
| 凸平均输出、强凸邻域 | SGD 22-23 | 明确期望、最优点/下界和步长计划；不把距离噪声底当成函数间隙 |
| 一维强凸稳态二阶矩 | 笔记补充 | 补独立同分布噪声条件和二阶矩递推，几何级数直接推导；不是原稿一般定理 |
| 动量递推与近似等价 | Momentum 8-9、12-13 | 补光滑强凸与小步长等条件；学习率关系不代表轨迹逐步相同 |
| 随机 NAG 与噪声下界 | Momentum 14-18 | 保留机制/瞬态解释，不把确定性加速阶数推广到噪声主导情形 |
| AdaGrad/RMSProp/Adam | Adaptive 9、14、17-18 | 坐标运算与状态含义对应；RMSProp 只有在有界梯度等前提下才保证统计量有界；Adam 偏差修正按原论文补全，并从第 1 步计数 |

非凸随机下降的独立依据：将 $-\gamma g_k$ 代入光滑上界，利用条件无偏以及 $\mathbb E[\|g_k\|^2\mid\mathcal F_k]\le\|\nabla F(x_k)\|^2+\sigma^2$，得到

$$
\mathbb E[F(x_{k+1})\mid\mathcal F_k]
\le F(x_k)-\frac\gamma2\|\nabla F(x_k)\|^2
+\frac{L\gamma^2\sigma^2}{2}
$$

（ $0<\gamma\le1/L$）。求和与函数下界给出正文的 $2\Delta_0/(\gamma K)+L\gamma\sigma^2$。

## 3. 第 9 讲：高级优化器

来源：[Advanced optimizers slides](materials/Lecture-09_Advanced-Optimizers/01_advanced-optimizers_slides.pdf)。

| 笔记结论 | 原始页序 | 核查结果与处理 |
|---|---|---|
| Adam 状态元素与字节数 | 2-3 | 元素数与精度假设对应；不含激活/额外缓冲，16-bit 状态不是所有实现的默认配方 |
| Adafactor 因子与状态规模 | 5-11 | 行/列均值版本与原稿边缘和归一化一致；数值例子为笔记补充，近似不声称精确 |
| Adam-mini 分组及完整一阶矩 | 18-22 | head/row/token 分组对应课件；具体实现仍依权重布局，不把整模型状态降为常数 |
| Newton 方向 | 26-28 | 增加可逆/正定前提，避免对非凸不定 Hessian 无条件声称下降 |
| Shampoo 左右统计与负矩阵幂 | 30-36 | 改为原稿累积版，补 $\epsilon I$ 初始化及形状；与 EMA 变体分开 |
| SOAP 特征基内的一阶更新 | 40-41 | 机制对应；未逐项核验实现中的状态换基、周期分解与所有数值保护 |
| Muon 动量与正交化 | 43-49 | 明确正文为机制简式，区别原稿归一化动量、负方向与有限次 NS 实现 |
| Sophia/Hutchinson/GNB/clip | 56-70 | 固定 Hessian 下矩条件成立；补逐坐标裁剪定义、模型分布与 Bartlett 正则条件 |

Hutchinson 的独立代数核验：固定 $H$ 时， $\mathbb E[u_i(Hu)_i]=\sum_jH_{ij}\mathbb E[u_iu_j]=H_{ii}$，只使用二阶矩条件，不依赖单一数值矩阵。

## 4. 第 10 讲：子空间与分块

来源：[GaLore/GoLore/RSO](materials/Lecture-10_Subspace-Optimizers/01_golore-rso_slides.pdf)、[BCD](materials/Lecture-10_Subspace-Optimizers/02_block-coordinate-descent_slides.pdf)、[分块训练](materials/Lecture-10_Subspace-Optimizers/03_block-wise-lisa_slides.pdf)。

| 笔记结论 | 原始页序 | 核查结果与处理 |
|---|---|---|
| 投影、回映与状态规模 | GaLore 15-16 | 左投影矩阵形状一致；一般子空间链式法则是笔记统一表达 |
| 噪声导致失效、GoLore | GaLore 23-28、31-34 | 保留分布/缩放条件，原稿随机投影期望含秩占比，不把未缩放投影称为原梯度无偏估计 |
| RSO 子问题与显存 | GaLore 42-47 | 原稿含二次正则项；正文只概述，激活收益依子空间计算图，不归于任意投影方法 |
| 精确最小化与梯度步 | BCD 5-8 | 两种算法分开，不互套收敛定理 |
| 块光滑性、条件下降与求和界 | BCD 12-15 | 第 14 页指定 $1/L_{\max}$；正文一般步长推导与特例一致，已直接查看原页 |
| 坐标友好结构 | BCD 16-20 | 将含糊的 $O(s)$ 改为相对完整更新约 $1/s$ 的成本 |
| LISA/分块激活与性能边界 | 分块 13-26 | 冻结层仍可能需要传播梯度；实验较慢不升级为普遍定理 |

## 5. 第 11 讲：零阶优化

来源：[ZO slides](materials/Lecture-11_Zeroth-Order-Optimizers/01_zeroth-order-gradient-descent_slides.pdf)、[MeZO/LOZO](materials/Lecture-11_Zeroth-Order-Optimizers/03_low-rank-zeroth-order_slides.pdf)。

| 笔记结论 | 原始页序 | 核查结果与处理 |
|---|---|---|
| 前向差分、查询数与误差 | ZO 8-10 | 共 $d+1$ 次查询，单坐标误差与全梯度误差不同；梯度稀疏不自动减少逐坐标查询 |
| 线性插值的测量方向 | ZO 14-17 | 明确按行堆叠方向与 $b=Q\nabla f+\text{误差}$；稀疏恢复需额外条件 |
| 球面估计器、球体平滑 | ZO 18-20；下列 Flaxman 论文 Lemma 1 | 修复期望括号范围，完整包含方向向量；原稿只给估计器与偏差界，平滑恒等式标为补充 |
| 与维度相关的步长/随机保证 | ZO 21、25-26 | 本笔记只描述条件依赖；原稿部分简式略去期望，不能当作每条随机路径的保证 |
| MeZO 中心差分与种子恢复 | MeZO/LOZO 6-10 | 两次前向及恢复顺序对应；模型/状态/瞬时激活仍有成本 |
| LOZO 低秩扰动及子空间机制 | MeZO/LOZO 13-20 | 概述对应；未逐项复核全部高阶矩与收敛常数 |

## 6. 第 12 讲：分布式优化

来源：[Decentralized](materials/Lecture-12_Distributed-Optimizers/02_decentralized-learning_slides.pdf)、[Local/compressed](materials/Lecture-12_Distributed-Optimizers/03_communication-compression-local-learning_slides.pdf)。

| 笔记结论 | 原始页序 | 核查结果与处理 |
|---|---|---|
| PSGD 聚合、线性加速 | Decentralized 4-7 | 共同参数与独立噪声、噪声主导区间是关键，不无条件承诺节点数倍加速 |
| DSGD 参数/梯度顺序 | Decentralized 12、20 | 原稿先本地更新后混合，明确对应第二式；Lian 文献另一顺序保留为扩展对照 |
| 双随机混合与谱条件 | Decentralized 20-26 | 保留共识条件，纯混合共识不能替代含梯度迭代的分析 |
| 异质性与梯度跟踪 | Decentralized 34-40 | 共同点与不同点的梯度平均分开；本轮核对机制，未重证全部 GT 定理 |
| Local SGD/FedAvg 双层递推 | Local/compressed 9-10 | 新增通信轮 $r$ 与本地步 $t$，含初始化、随机梯度、全参与平均；修正原稿右侧下标排版笔误 |
| SCAFFOLD 与客户端参与 | Local/compressed 18-22、27-28 | 保留机制；本节公式限等权全参与，不冒充所有 FedAvg 设置 |
| 无偏压缩、Top-$k$ | Local/compressed 36-37、42-43 | 无偏与收缩性质分开；Random-$k$ 是补充例子，声明概率/缩放与 $1\le k\le d$ |
| Error Feedback | Local/compressed 45 | 明确原稿为 EF21 梯度差跟踪版，不与经典残差累计版混为一式 |
| 总通信代价 | Local/compressed 51-57 | 保留每轮代价乘轮数的解释；不跨噪声/目标假设套用收益结论 |

第 12 讲其余流水线/张量/3D 并行属于结构示意，本轮未重新逐图审核所有调度与算例。拓扑部分核对主题和机制，不把所有附录谱隙证明列为已完成。

## 7. 独立来源、教学组织与维护边界

- Anderson 完整固定点算法：[Mai–Johansson，Algorithm 1](https://arxiv.org/abs/1910.08590)。
- 球体平滑与球面估计器：[Flaxman–Kalai–McMahan，Lemma 1](https://arxiv.org/abs/cs/0408007)。
- Adam 偏差修正：[Kingma–Ba，Algorithm 1](https://arxiv.org/abs/1412.6980)。
- 14 讲的官网主题/跨年归属/建议学习等级核对见 [课程范围](notes/COURSE_SCOPE.md)；没有教师明确说明的内容不写成考试重点。
- 本轮没有增加零散测试组。原有块路径枚举设定上限，保留小型算例，防止编辑规模后使 CI 指数膨胀。
- 本页给出的独立推导、来源和反例比文案字符串提供更直接的数学依据；它们仍不能替代其余材料的系统审查或形式化证明。

## 8. 阅读器检查

Obsidian 阅读视图已实测：第 9 讲 Adafactor 多行矩阵和分式、第 10 讲条件期望与长收敛式、第 11 讲球体平滑期望括号、第 12 讲 Local SGD 初始化/本地步/平均公式。所选位置显示正常，未把此结果扩张为全库逐页验收。

2026-10-26 远端状态复核：内容提交 `4c2b4f4` 已上传，远端分支 SHA 一致，[该提交 CI 通过](https://github.com/threeyang3/dlopt2026-learning-notes/actions/runs/40204150714)。本轮 GitHub 视觉复查尚未完成：桌面浏览器操作被用户操作中断，独立浏览器未登录此私有仓库。文件/链接/公式定界符检查和 CI 均不能替代实际网页渲染。
