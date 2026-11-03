# 补充单元：语言模型基础——词嵌入、n-gram 与 RNN

**课程定位**：用户补充材料，不改变 DLOpt 2026 原 14 讲的编号，也不宣称它原本就在 2026 课程清单中。建议学完 [[Lecture-03_Linear-Algebra-II]]、[[Lecture-04_Probability-and-Statistics]] 后，在 [[Lecture-07_Transformer-and-LLM-I]] 之前阅读。

**主要来源**：Kun Yuan，《Language Models》，用户上传的 `03_langmodel.pdf`，56 页。[仓库原件](../materials/Supplement-Language-Models/03_langmodel.pdf)与[作者公开原件](https://raw.githubusercontent.com/kunyuan827/kunyuan827.github.io/master/teaching/DLOpt2024/03_langmodel.pdf)字节一致，SHA-256 为 `ca4316886f225f3324b81d99a3e0ee97efd8ed42ca6bbe73fd158478d8441419`。原件不作修改。第 3 页注明词嵌入材料来自 Deep Learning Specialization，语言模型材料来自 Stanford CS224n。

**阅读约定**：页码均为 PDF 页序。“课件主线”概括原件；“推导补充”“实现约定”“勘误与边界”是本笔记添加的解释，不冒充课件原文。网站是浏览器本地的小型教学实验，不是预训练模型，不衡量真实语言理解能力。

## 学习目标

- **核心掌握**：one-hot 与 embedding 查表的关系；n-gram 的条件概率与计数估计；RNN 的共享参数递推；next-token 交叉熵的输入/标签错位关系。
- **熟悉**：固定窗口模型与递归状态的区别；采样生成、训练与推理；长路径的 Jacobian 连乘；双向与多层 RNN 的适用场景。
- **了解**：词向量类比、课件中的生成案例以及更复杂的循环结构。不要将示意向量当成真实模型的语义规律。

## 1. 从 one-hot 到词嵌入（第 4–10 页）

### 1.1 one-hot 表示能标识词，却不表达语义相似性

词表大小为 $V$，第 $i$ 个词的 one-hot 向量为 $o_i\in\mathbb R^V$，仅第 $i$ 个分量等于 1。不同词之间满足

$$
o_i^\top o_j=0,\qquad \|o_i-o_j\|_2=\sqrt{2}\quad(i\ne j).
$$

这是表示几何的性质，不是说两个词在自然语言中无关。课件以 apple、orange 和 juice 说明：只看 one-hot 距离，不能表达这些词的语义联系。[原件第 5 页](../materials/Supplement-Language-Models/03_langmodel.pdf#page=5)

### 1.2 嵌入矩阵乘 one-hot 就是取一列

沿用课件的**列向量约定**：嵌入维度为 $d$，矩阵 $E\in\mathbb R^{d\times V}$，则

$$
e_i=Eo_i=E_{:,i}\in\mathbb R^d.
$$

形状是 $(d\times V)(V\times1)=d\times1$。实现中可直接索引对应列，无需构造完整 one-hot。某些软件把表存成 $V\times d$ 并按行索引，这是存储约定差异。[原件第 10 页](../materials/Supplement-Language-Models/03_langmodel.pdf#page=10)

课件第 6 页给出六个词、四个人工命名坐标的示意矩阵。网站沿用该表的数字，允许选择两个词，对照 one-hot 与嵌入余弦相似度。余弦定义为

$$
\cos(a,b)=\frac{a^\top b}{\|a\|_2\|b\|_2},\qquad \|a\|_2\|b\|_2>0.
$$

**边界**：这些是人工示意数，不是训练后的语义测量；实际 embedding 的单个维度通常不能直接命名为 gender、royal、age、food。第 6 页关于“接近”的直觉还取决于使用哪些维度和哪一种距离，不能由文字描述取代数值计算；相似度高也不等于同义词。

### 1.3 类比是近似关系，不是语义定律

按第 9 页的中间等式移项：

$$
e_{\mathrm{man}}-e_{\mathrm{woman}}\approx e_{\mathrm{king}}-e_w
\quad\Longrightarrow\quad
e_w\approx e_{\mathrm{king}}-e_{\mathrm{man}}+e_{\mathrm{woman}}.
$$

**勘误**：原件第 9 页末行搜索式在 woman 项前显示减号，与上面的移项结果不一致。本笔记明确采用中间等式导出的加号，不改写原 PDF。若用余弦搜索候选词，应排除题目已经给出的词，并注意零范数情形。第 8 页“orange and orange”按上下文应为“orange and apple”。这些勘误不意味着任意 embedding 都支持稳定的词向量类比。

## 2. 语言模型与 n-gram（第 11–23 页）

### 2.1 预测的是一个分布

**课件主线**：给定已出现的词，语言模型输出下一个词的条件分布，而非只有一个确定答案：

$$
p_\theta(x_{t+1}\mid x_1,\ldots,x_t),\qquad x_{t+1}\in\mathcal V.
$$

**推导补充**：若还定义了起始分布或 BOS 条件，则链式法则给出序列概率：

$$
p(x_1,\ldots,x_T)=p(x_1)\prod_{t=2}^{T}p(x_t\mid x_1,\ldots,x_{t-1}).
$$

应区分词级课件示例和实际 tokenizer 的 token；网站的“词”就是显式小词表中的符号，不模拟 BPE。

### 2.2 n 表示整个 n-gram 的长度，历史只有 n−1 个词

n-gram 假设用最近 $n-1$ 个词近似完整历史。写 $c_t=(x_{t-n+2},\ldots,x_t)$，则在上下文计数为正时，最大似然计数估计为

$$
\widehat p(w\mid c_t)=\frac{C(c_t,w)}{\sum_{v\in\mathcal V}C(c_t,v)}.
$$

若把 $C(c_t)$ 定义为具有后继预测事件的上下文总次数，分母就是 $C(c_t)$。统计时需明确句子边界；不能随意把上一句最后一个词与下一句第一个词拼接成句内样本。[原件第 16 页](../materials/Supplement-Language-Models/03_langmodel.pdf#page=16)

课件第 17 页的 4-gram 只保留 `students opened their` 三个历史词。若上下文出现 1000 次，接 books 出现 400 次、接 exams 出现 100 次，则这两项概率分别为 0.4、0.1。**不能把只列出的两项再归一化成 0.8 和 0.2**，其余 0.5 属于其他后继词。

### 2.3 稀疏性、平滑与生成

若上下文出现过，但某个后继词未出现，未平滑模型给它概率 0；若整个上下文未出现，分母为 0，经验条件分布未定义。扩大 $n$ 能延长显式历史，但通常加重计数稀疏与存储成本；课件第 22–23 页强调了这一权衡。

**实现补充（非课件展开的算法）**：网站提供 add-$\alpha$ 平滑以演示边界：

$$
\widehat p_\alpha(w\mid c)=\frac{C(c,w)+\alpha}{C(c)+\alpha V},\qquad \alpha\ge0.
$$

当 $C(c)=0,\alpha>0$ 时，得到小词表上的均匀分布；这并不表示模型“理解了”陌生上下文。当两者均为 0 时，网站明确显示未定义并停止生成，而不是静默填入均匀分布。已知目标概率为 0 时，负对数似然为 $+\infty$，不能为了界面好看偷偷截断为有限值。

网站使用**自建的 12 句语料**：books/exams/laptops/minds 分别出现于同一上下文 4/1/3/2 次，另外两句是水果果汁示例。这不是原件实验数据，仅使前两项比例与第 17 页一致。每句独立添加 BOS 历史和 EOS 目标，BOS 不属于输出词表。可以调节 $n$、平滑和随机种子，逐步从当前条件分布抽样，遇 EOS 或无定义上下文停止。[课件生成过程：第 19–22 页](../materials/Supplement-Language-Models/03_langmodel.pdf#page=19)

## 3. 固定窗口神经语言模型（第 24–25 页）

窗口含 $k$ 个词，每个嵌入维度 $d$，拼接得到 $e\in\mathbb R^{kd}$。隐层宽度为 $H$，则

$$
h=f(We+b_1),\qquad z=Uh+b_2,\qquad \widehat y=\mathrm{softmax}(z),
$$

其中 $W\in\mathbb R^{H\times kd}$、 $b_1\in\mathbb R^H$、 $U\in\mathbb R^{V\times H}$、 $b_2\in\mathbb R^V$。

**区别**：n-gram 用离散计数表；固定窗口神经模型通过嵌入、隐层和输出参数共享统计信息，但输入仍受窗口限制。增加窗口宽度会增加这里的输入权重规模；不能把这项结论直接套用到参数跨时间共享的 RNN。

网站对本节保留公式和笔记讲解，不声称额外实现了一个固定窗口模型训练器。

## 4. RNN 语言模型：共享参数，逐步更新状态（第 26–29 页）

在课件列向量约定下，输入 $x_t\in\mathbb R^V$ 是 one-hot，隐状态 $h_t\in\mathbb R^H$：

$$
e_t=Ex_t,\qquad a_t=W_hh_{t-1}+W_ee_t+b_1,\qquad h_t=\sigma(a_t),
$$

$$
z_t=Uh_t+b_2,\qquad \widehat y_t=\mathrm{softmax}(z_t).
$$

维度依次为： $E:d\times V$； $W_h:H\times H$； $W_e:H\times d$； $U:V\times H$；偏置为 $H$ 与 $V$ 维。各时间步使用同一组参数； $h_0$ 需明确初始化。[原件第 27 页](../materials/Supplement-Language-Models/03_langmodel.pdf#page=27)

**网站约定**： $V=7,d=3,H=2$，使用 $\tanh$、固定 $h_0=0$，总共 54 个可训练标量：

$$
dV+H^2+Hd+H+VH+V=21+4+6+2+14+7=54.
$$

权重随机初始化且不预训练，输出不应被解释为合理自然语言概率。页面允许播放输入序列，检查 embedding、前一隐藏状态、递归项、输入项、当前隐藏状态和下一个词分布。改变观察时间步**不会训练或重新初始化参数**。

**边界**：课件“序列变长不增加模型大小”指参数数目，不是总计算量或训练激活内存不增长。能够接收任意长度序列也不等于能可靠保留任意长距离信息。

## 5. 训练目标、BPTT 与生成（第 30–40 页）

### 5.1 标签是下一个词，不是当前词

为避免尾端越界，令监督序列为 $x_1,\ldots,x_{T+1}$；有 $T$ 对输入/目标。若最后一个真实词也需被用作输入，就显式接 EOS 作为末尾目标。令 $y_t$ 为 $x_{t+1}$ 的 one-hot：

$$
\ell_t=-\sum_{v=1}^{V}y_{t,v}\log\widehat y_{t,v}=-\log\widehat y_{t,x_{t+1}},\qquad
J(\theta)=\frac1T\sum_{t=1}^{T}\ell_t.
$$

用自然对数时单位为 nat。稳定实现使用 $\ell_t=\log\sum_v\exp(z_{t,v})-z_{t,x_{t+1}}$，计算 log-sum-exp 时先减最大值，避免溢出。网站同时显示当前步损失与整个训练句子的平均损失，两者不混用。[原件第 30–35 页](../materials/Supplement-Language-Models/03_langmodel.pdf#page=30)

### 5.2 完整 BPTT（推导补充，不是原件的逐式复现）

课件给出了训练目标与展开图，但没有在这份文件中完整推导 BPTT。以下用于解释网站的单句训练器。对平均损失，有

$$
\delta^z_t=\frac{\widehat y_t-y_t}{T},\qquad
\delta^a_t=\left(U^\top\delta^z_t+W_h^\top\delta^a_{t+1}\right)\odot(1-h_t\odot h_t),
$$

从 $t=T$ 向前递推，边界 $\delta^a_{T+1}=0$。参数共享意味着各时间步贡献相加：
$$
\nabla_UJ=\sum_t\delta^z_th_t^\top,\qquad
\nabla_{W_h}J=\sum_t\delta^a_th_{t-1}^\top,\qquad
\nabla_{W_e}J=\sum_t\delta^a_te_t^\top.
$$

偏置梯度分别为 $\sum_t\delta^z_t$ 与 $\sum_t\delta^a_t$；输入词对应的嵌入列累加 $W_e^\top\delta^a_t$。重复出现的词必须累加到同一列，不能覆盖前一次贡献。54 个参数均可通过中心有限差分独立检验。

点击“训练一步”才执行 $\theta\leftarrow\theta-\eta\nabla J$；播放只展示已计算轨迹。网站最多演示 50 步单句梯度下降，不使用真实语料库、不提供泛化指标，也不保证任意步长每一步都下降。

### 5.3 批量训练与逐词生成的输入不同

第 36 页说明实际训练按句子/文档或批次进行，而非一次保存整个语料库的计算图。第 37 页展示的是标注为 Transformer 的 $(B,T)$ 输入组织示意，不是 RNN 独有格式。**实现边界补充**：变长批次需要区分 padding 与有效目标；数据打包与跨文档状态传递需要明确策略，网站单句演示不实现这些机制。

训练中输入真实前缀、预测真实后继；生成时则从输出分布采样，再将生成词作为之后的输入。第 38–40 页展示的是已训练 RNN 的案例，不能把它们当作本网站随机小模型的能力证明。网站提供 n-gram 逐词采样，不声称复现课件中的 RNN 长文本生成质量。

## 6. 为什么长距离梯度会消失或爆炸？（第 41–43 页）

**课件主线**：远处的训练信号沿很多时间步传播后可能很弱，模型因而难以学习长距离依赖；也可能出现梯度爆炸。第 41 页只概述 residual connections、LSTM、attention 等思路，没有给出这些机制的完整推导。

**推导补充**：对 $\tanh$ RNN，记

$$
J_t=\frac{\partial h_t}{\partial h_{t-1}}=\mathrm{diag}(1-h_t^2)W_h,
\qquad
\frac{\partial h_t}{\partial h_s}=J_tJ_{t-1}\cdots J_{s+1}.
$$

所以若所有因子谱范数均不超过 $\rho<1$，乘积范数不超过 $\rho^{t-s}$。反之，仅有某个权重范数大于 1，并不足以证明实际梯度必然爆炸；饱和导数、方向和连乘结构也重要。

网站用**标量、零输入的教学递推** $h_t=\tanh(wh_{t-1})$ 展示

$$
\frac{\partial h_T}{\partial h_0}=\prod_{t=1}^{T}w(1-h_t^2).
$$

当 $h_0=0$ 时，沿零轨迹的导数恰好是 $w^T$；非零初态可能进入饱和区，使导数显著减小。页面显示的是对初始状态的敏感度，**不是对参数 $w$ 的梯度，也不是完整训练损失的梯度**。

## 7. RNN 家族、双向与多层结构（第 44–56 页）

第 45–49 页按输入输出序列长度组织模型：一对一；一对多（如音乐生成）；多对一（如情感分类）；等长多对多（如序列标注）；不等长多对多（如机器翻译）。这是一组任务结构示例，不表示只有 RNN 才能完成这些任务。

**双向 RNN（第 50–53 页）**：前向状态编码左侧历史，后向状态编码右侧信息，通常拼接为 $[\overrightarrow h_t;\overleftarrow h_t]$。它们是两个方向的递推，不是让单个循环方程在同一时刻互相依赖。课件第 52 页双箭头只是简图。

**适用边界**：第 53 页“不能用于 language modeling”应放在本课件的左到右 next-word 场景理解：不能先读取整句的未来真值，再把利用该信息的预测当作合法的因果预测。已知完整输入的编码、序列标注等任务可利用双向上下文；不能把这句话推广成“所有语言建模任务都必须单向”。

**多层 RNN（第 54–56 页）**：上一层在同一时刻的输出作为下一层输入。时间维展开与深度方向堆叠是两回事。以 $h_t^{(0)}=e_t$ 为例，补充写法为

$$
h_t^{(\ell)}=\tanh\left(W_h^{(\ell)}h_{t-1}^{(\ell)}+W_x^{(\ell)}h_t^{(\ell-1)}+b^{(\ell)}\right).
$$

第 56 页中的具体层数来自特定历史实验，不是所有任务的最佳层数准则。本模块讲解双向和多层结构，但动态计算只实现单向单层 RNN，不能把示意文字当成已经实现这些扩展。

## 仅作了解

这份课件不完整覆盖 word2vec 的训练目标、LSTM 门控、BPE、Transformer 全结构及大规模训练系统；本单元不会用外部知识悄悄补成“原课件内容”。后续注意力与 Transformer 主线见 [[Lecture-07_Transformer-and-LLM-I]]；计算及显存问题见 [[Lecture-08_Transformer-and-LLM-II]]。RNN 梯度消失不意味着所有 RNN 都无法学习任何长程关系。

## 自测

1. $E\in\mathbb R^{4\times6}$ 乘第 5 个 one-hot 后是什么？**答**：第 5 列，即课件示例 apple 向量；不是第 5 行。
2. 4-gram 有几个历史词？**答**：3 个；n 包含待预测的下一个词。
3. 400/1000 和 100/1000 为什么不能再归一化？**答**：尚有其他后继词占据 0.5 概率质量。
4. 未见过的上下文且 $\alpha=0$ 时怎么办？**答**：分布未定义；需要明确另设回退/平滑，而非偷偷赋均匀分布。
5. 输入 the 时，训练目标是不是 the？**答**：在课件例子中是 students，即下一个词。
6. RNN 序列变长时参数数目增长吗？**答**：共享参数数目不随时间步增长，但计算和保存的激活可能增长。
7. 网站 $h_0=0,w=0.5,T=3$ 的标量状态导数是多少？**答**： $0.5^3=0.125$；隐藏状态本身仍为 0。
8. 为什么 BPTT 中同一 embedding 列可能接收多次梯度？**答**：同一词可在多个时间位置出现，而嵌入参数共享。
9. 双向整句编码能直接当作不看未来的 next-token 预测吗？**答**：不能把已读到目标未来的信息泄漏进因果预测。

## 资料定位

| PDF 页序 | 内容 | 网站/笔记范围 |
|---|---|---|
| 1–3 | 标题、目录、课程来源 | 保留来源与归属 |
| 4–10 | one-hot、语义示意、词类比、E 查表 | 嵌入实验；明确两处勘误 |
| 11–23 | next-word、n-gram、计数、生成与稀疏性 | 计数和采样实验；平滑标为补充 |
| 24–25 | 固定窗口神经模型 | 公式讲解，不另设训练器 |
| 26–29 | RNN 展开、参数共享、优势 | 单层 tanh RNN 前向轨迹 |
| 30–37 | 交叉熵、序列训练、批次组织 | 单句训练；BPTT 推导标为补充 |
| 38–40 | 已训练 RNN 的生成例子 | 仅介绍，不承诺复现效果 |
| 41–43 | 梯度传播与长程依赖困难 | 标量敏感度演示，非 LSTM 实现 |
| 44–56 | 任务结构、双向、多层 RNN | 笔记讲解与适用边界 |

[打开原始 PDF](../materials/Supplement-Language-Models/03_langmodel.pdf) · [返回笔记索引](README.md)
