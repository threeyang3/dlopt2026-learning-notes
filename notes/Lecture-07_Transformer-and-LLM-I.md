# 第 7 讲：Seq2Seq、Attention 与 Transformer

> **课程定位**：2026 年课程核心 Transformer 基础；复用 DLOpt 2024 课件。
>
> 配套文件：`materials/Lecture-07_Transformer-and-LLM-I/01_transformer-attention_slides.pdf`

## 学习目标

理解从 RNN encoder-decoder 到 attention、self-attention、multi-head attention 和 Transformer 的演化；能写出 attention 矩阵形式和计算复杂度，并能解释位置编码、残差、LayerNorm 与因果掩码的作用。

## 1. Seq2Seq 与 RNN 的瓶颈

**核心掌握问题，熟悉历史结构。** Seq2Seq 用编码器把输入序列压成隐状态，再由解码器逐 token 输出。典型 RNN 的问题包括：

- 长距离信息必须经过固定大小隐状态，形成信息瓶颈。
- 两个远距离 token 的交互路径随距离线性增长。
- 时间步串行，GPU 并行度低，训练长序列慢。

Attention 不取代全部结构，而是让解码器在每一步直接读取编码器所有隐藏状态，缩短远距离依赖路径。现代 LLM 常为 decoder-only，但 encoder-decoder 和 cross-attention 仍用于翻译、检索增强与多模态模型。

## 2. Attention 是加权平均

**核心掌握。** 给定 decoder 当前查询 $s$ 和 encoder 状态 $h_i$，先算相关性分数，再归一化：

$$
e_i=\mathrm{score}(s,h_i),\qquad
\alpha_i=\frac{\exp(e_i)}{\sum_j\exp(e_j)},
$$

$$
c=\sum_i\alpha_i h_i.
$$

常见加性分数 $e_i=v^\top\tanh(W_1s+W_2h_i)$，乘性分数 $e_i=s^\top Wh_i$。输出是所有 value 的加权平均，权重可解释为“当前步把注意放到哪里”。

## 3. Self-Attention

**核心掌握矩阵公式。** 同一序列通过三个线性映射得到

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

取 $X\in\mathbb R^{N\times d_{\mathrm{model}}}$， $W_Q,W_K\in\mathbb R^{d_{\mathrm{model}}\times d_k}$、 $W_V\in\mathbb R^{d_{\mathrm{model}}\times d_v}$，则 $Q,K\in\mathbb R^{N\times d_k}$、 $V\in\mathbb R^{N\times d_v}$。

缩放点积 attention 为

$$
\mathrm{Attn}(Q,K,V)
=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
$$

- $QK^\top$ 每行表示一个 query 对所有 key 的匹配分数。
- $1/\sqrt{d_k}$ 控制点积随维数增长造成的数值方差。
- $M$ 可为因果 mask：未来位置设为 $-\infty$，softmax 后权重为 0。
- softmax 按行归一化，输出形状为 $N\times d_v$；每行必须至少有一个未屏蔽的 key，整行均为 $-\infty$ 会使普通 softmax 未定义。

单个 self-attention 层的主要计算是 $QK^\top$ 和 softmax 权重乘 $V$，复杂度约为

$$
O(N^2d),
$$

显式存储 $N\times N$ 注意力矩阵需 $O(N^2)$ 内存。长上下文的主要挑战来自这里，FlashAttention 见 [[Lecture-08_Transformer-and-LLM-II#3. FlashAttention：减少 HBM 访问|第 8 讲]]。

上述复杂度专指 attention 核心并假定 $d_k,d_v$ 同阶为 $d$，不包含输入/输出投影的 $O(Nd_{\mathrm{model}}^2)$ 成本。多头时要计入各头，不能把单头显式矩阵大小当作整个模型显存。

## 4. 为什么需要多头

**核心掌握结构。** 每个头有自己的 $W_Q^{(i)},W_K^{(i)},W_V^{(i)}$，在不同子空间学习关注模式：

$$
Q_i=XW_Q^{(i)},\qquad
K_i=XW_K^{(i)},\qquad
V_i=XW_V^{(i)},
$$

$$
\mathrm{head}_i
=\mathrm{Attn}(Q_i,K_i,V_i),
$$

再将头拼接并线性映射：

$$
\mathrm{MHA}(X)
=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_a)W_O.
$$

多头不是简单复制；它让模型同时关注词法、句法、指代等不同关系。头数与每头宽度影响参数、FLOPs 与 KV Cache，见 [[Lecture-14_Efficient-Fine-Tuning-and-Inference#3. 自回归推理与 KV Cache]]。

## 5. 位置编码、残差与非线性的必要性

**核心掌握四个作用。**

1. **位置编码**：普通双向 self-attention 对 token 排列具有等变性，需要显式位置信号来表达“谁在谁之前”。因果掩码也会限制信息流向，但它只定义允许关注的区域，不能替代位置编码；两者提供的信息不同。可用可学习位置向量、正弦编码或 RoPE 等方法。
2. **非线性**：在注意力权重固定的条件下，attention 输出对 $V$ 是线性的；完整的 self-attention 因 softmax 和输入投影而是输入 $X$ 的非线性函数。前馈网络与激活函数进一步增加表达能力。
3. **残差连接**：每层输出写为 $x+\mathrm{Sublayer}(x)$，让信息和梯度有直接通道，深层网络更容易训练。
4. **归一化**：LayerNorm 对单个 token 的特征做标准化，稳定激活尺度；Pre-LN/Norm 差异影响深层训练稳定性。

## 6. Transformer 结构

**核心掌握数据流。**

1. token 经 embedding 加上位置信息；
2. 每层先做 self-attention；
3. 接残差与归一化；
4. 通过两层 MLP（常把宽度扩为 $4d$）；
5. 再接残差与归一化；
6. 最后经语言模型头输出词表 logits。

Encoder 不遮蔽未来，Decoder 用因果 mask；Encoder-Decoder 模型在 decoder 中还有 cross-attention，其 query 来自 decoder，key/value 来自 encoder。

## 7. 计算与表达力的取舍

Attention 能直接连接任意两 token，并行度高，但稠密核心计算随长度二次增长，朴素实现还显式存储二次大小矩阵；RNN 则沿时间串行。FlashAttention 保留精确稠密 attention，主要改变 IO/存储；稀疏和线性 attention 才改变连接或计算形式，不能把三者都当成精度与成本的同一种折中。

## 仅作了解

- 所有 positional encoding 的三角恒等式与 RoPE 的完整推导。
- Transformer 论文中的每个 BLEU 指标和对机器翻译实验的历史比较。
- LayerNorm 的变体细节；先掌握“稳定尺度”和放置位置。

## 自测

1. 写出缩放点积 attention，并解释三个矩阵的维度。
2. 为什么 self-attention 需要位置编码？
3. 因果 mask 如何保证自回归训练不看到未来？
4. $N$ 从 1,000 增至 10,000，显式注意力矩阵约增长多少倍？

## 资料定位

| 主题 | 原始资料位置 |
|---|---|
| Seq2Seq、RNN 瓶颈、分数与加权和 | PDF 页序 10-13、16、25、29-30 |
| Self-attention、行 softmax、缩放 | PDF 页序 36-40；本笔记另显式列出矩阵维度与 mask 条件 |
| 位置、非线性、多头与 Transformer | PDF 页序 45-59、60-65 |
| 交叉注意力 | PDF 页序 66-67；复杂度为按矩阵乘法维度的补充分析 |

下一讲把这些维度变成参数、FLOPs 和显存公式：[[Lecture-08_Transformer-and-LLM-II]]。

