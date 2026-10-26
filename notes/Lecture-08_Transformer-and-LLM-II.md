# 第 8 讲：Transformer 的参数、计算、显存与 FlashAttention

> **课程定位**：2026 年课程核心 Transformer 系统分析；复用 DLOpt 2024 课件。
>
> 配套文件：`materials/Lecture-08_Transformer-and-LLM-II/01_transformer-parameter-computation_slides.pdf`、`02_transformer-memory-cost_slides.pdf`、`03_flash-attention_slides.pdf`

## 学习目标

能用 $l,h,v,s,b$ 等符号估算 decoder-only Transformer 的参数、FLOPs、训练显存和 attention 的 HBM 通信；理解 FlashAttention 的 tiling、kernel fusion 与 online softmax，知道它节省的是中间内存和显存访问，不是把 $O(N^2d)$ 计算变成线性。

## 1. 记号与参数估算

**核心掌握量级。** 假设：

- 层数 $l$，隐藏宽度 $h$，词表大小 $v$，头数 $a$，序列长度 $s$，micro-batch 为 $b$；
- 每个注意力头有 $W_Q,W_K,W_V$，并有一个输出矩阵 $W_O$；
- 前馈网络把 $h$ 投到 $4h$ 再投回 $h$。

每层 attention 参数约 $4h^2$，FFN 参数约 $8h^2$，因此每层约 $12h^2$；再乘 $l$：

$$
P_{\text{body}}\approx12lh^2.
$$

输入词嵌入约 $vh$，输出头也常为 $vh$。若输入嵌入与输出投影共享权重，词表相关参数约为 $vh$：

$$
P_{\text{tied}}\approx vh+12lh^2.
$$

若二者不共享，则

$$
P_{\text{untied}}\approx2vh+12lh^2.
$$

后续 $12lh^2$ 的 Transformer 主体估算不变；引用“约 $12lh^2$”时通常忽略词表项。

这是标准多头 attention 加两层宽度 $4h$ FFN 的近似账本；GQA/MQA、门控 FFN、不同中间宽度及 embedding 共享会改变常数。课件虽用 LLaMA 作数值比较，也不能据此将 $12lh^2$ 当成所有 LLaMA 架构的精确结构公式。

一个带可训练 gain/bias 的 LayerNorm 有 $2h$ 个参数；若每 block 有两个，则是 $4h$，还需另计末层归一化。RMSNorm 等变体的计数不同。这些线性于 $h$ 的项在主项估算中省略，精确账本仍须计算。

## 2. FLOPs

**核心掌握主项，熟悉常数。** 一次矩阵乘法 $X_{s\times h}W_{h\times h'}$ 的主计算量约为 $2shh'$ FLOPs（每个乘加计 2）。

- 输入 token embedding 通常是按 ID 查表，主项是访存而非稠密矩阵乘；输出头是 $s\times h$ 乘 $h\times v$，约 $2svh$ FLOPs（每个 token 为 $2vh$）。
- 每层 attention 的 Q/K/V/O 投影与 attention 点积、加权：主项约 $8sh^2+4s^2h$。
- 每层 FFN：约 $16sh^2$。
- 因而每层前向约 $24sh^2+4s^2h$；长序列时 $4s^2h$ 才会显著成为主项。

在主体矩阵乘占主项时，全模型前向可近似为每序列 $2Ps$、每批 $2Pbs$，或每 token $2P$；当词表项占比明显时，输入查表与共享权重的计数会影响该近似。反向传播约为前向的 2 倍，所以前向加反向共约 3 倍前向，训练常用

$$
\text{training FLOPs}\approx6PD,
$$

其中 $D$ 是处理的 token 数。这个“$6PD$”是主项估算，不含注意力二次项、激活重算、通信和硬件利用率损失。

## 3. FlashAttention：减少 HBM 访问

**核心掌握动机与机制。** 标准实现常把 $N\times N$ 的 logits、概率矩阵写回 GPU 的 HBM，再读回做 softmax 和乘 $V$。Attention 的 FLOPs 是 $O(N^2d)$，但在长序列下，读写大中间矩阵可能比算术更耗时。

FlashAttention 的三个关键词：

1. **Tiling**：把 $Q,K,V$ 切成能放进片上 SRAM 的块，在块内完成 attention。
2. **Kernel fusion**：把 scale、mask、softmax、乘 $V$ 尽量融合，避免把完整 attention 矩阵落到 HBM。
3. **Online softmax**：分块处理时维护运行最大值与归一化因子，流式组合每块输出。

Online softmax 的直觉：读完新块 $j$ 后更新

$$
m_j=\max(m_{j-1},\max_i x_i^{(j)}),
$$

$$
\ell_j=e^{m_{j-1}-m_j}\ell_{j-1}
+\sum_i e^{x_i^{(j)}-m_j},
$$

对一条 query 行，若本块 value 为 $v_i^{(j)}\in\mathbb R^{d_v}$，可维护未归一化加权和

$$
z_j=e^{m_{j-1}-m_j}z_{j-1}
+\sum_i e^{x_i^{(j)}-m_j}v_i^{(j)},
\qquad o_j=z_j/\ell_j.
$$

首个含有效有限 logit 的块直接初始化 $m_1,\ell_1,z_1$，空块/全 mask 块跳过；只有 $\ell_j>0$ 才做除法。遍历完整行后 $o_j$ 才是最终 attention 输出，提前的 $o_j$ 只对已读块归一化；无需保存整行，不代表无需读取其余块。此递推是对课件第 21-24 页分块合并式的等价展开。

**重要边界：** FlashAttention 保留数学上的精确稠密 attention，有限精度和运算顺序仍可产生数值差异。它避免在 HBM 中存储完整 attention 中间矩阵，反向时重算部分中间量。在原论文的两级存储模型中，单头宽度为 $d$、 SRAM 容量 $M$ 按可存储的元素数计，且 $d\le M\le Nd$ 时，HBM 读写量从标准实现的 $\Theta(Nd+N^2)$ 改为 $\Theta(N^2d^2/M)$；不能将该界无条件用于任意 $M$、字节单位或所有硬件执行时间。见 [FlashAttention 原论文第 3.2 节](https://arxiv.org/abs/2205.14135)。

## 4. 训练显存五分量

**核心掌握。** 统一使用

标题“五分量”将 Adam 一阶、二阶状态分别计数；下式把二者合并为 Optimizer states，因此写成四个类别。不能漏掉其中任一状态。

$$
\text{Memory}
=\text{Model}+\text{Gradient}
+\text{Optimizer states}
+\text{Activations}.
$$

若参数量为 $P$，一种常被比较的 **BF16 模型/梯度 + 两个 16-bit Adam 状态** 账本为：

| 分量 | 每参数字节 | 7B 参数量级 |
|---|---:|---:|
| BF16 模型参数 | 2 | 14 GB |
| BF16 梯度 | 2 | 14 GB |
| Adam 一阶、二阶状态（各 2 bytes） | 4 | 28 GB |
| 激活 | 依赖 batch/序列/模型 | 沿用内存高效优化器课件的特定设置约 2 GB |
| 合计（不含额外主权重和临时缓冲区） | 8 + 激活 | 约 58 GB |

这里 28GB 明确对应“两个 16-bit 状态”。若两个状态都存成 FP32，则需约 $8P$ 字节，即 7B 参数约 56GB（十进制），不是 28GB。若另存 FP32 主权重、FP32 梯度或通信缓冲，还要继续加项。GPT-3 175B 的模型、梯度和优化器状态合计已达 TB 级。激活为重算、LoRA、子空间和零阶方法提供了节省空间，见 [[Lecture-09_Advanced-Optimizers]]、[[Lecture-10_Subspace-Optimizers]]、[[Lecture-11_Zeroth-Order-Optimizers]]。

上表的 2 GB 激活不是对任意 7B 模型的估计，出处是第 10 讲 `01_golore-rso` PDF 页序 7-8 的特定账本。本讲 `02` memory slides PDF 页序 14 则按朴素保存策略估算激活元素数为 $b\lbrace l(2s^2a+14sh)+2sv+sh\rbrace $；融合算子、重计算和序列/批大小都会改变实际峰值。通信暂存另计，因此这里的四个常驻类别不等于实际总峰值只含四项。

## 5. 计算、带宽与容量三本账

**核心掌握区分。**

- **算力（FLOPs）**：矩阵乘通常算力密集，可用 Tensor Core 加速。
- **带宽（Bytes moved）**：大量小算子或 attention 中间矩阵会受 HBM 带宽限制。
- **容量（Bytes stored）**：模型、状态、激活和 cache 是否放得下。

优化只能减少真实瓶颈，才能改善端到端时间。例如减少 FLOPs 但增加同步和内存访问可能更慢。

## 仅作了解

- FlashAttention 对应 CUDA 实现、warp 映射和每个 kernel 的具体 tile 大小。
- 所有注意力反向传播公式；掌握“重算换存储”和数值稳定性即可。
- 不同 GPU 的 SRAM 容量与 Tensor Core 微架构细节。

## 自测

1. 用 $l,h,v$ 写 decoder-only 参数的主项估算。
2. 为什么稠密 attention 的计算和朴素显式矩阵存储随序列长度二次增长？FlashAttention 改变了其中哪项？
3. FlashAttention 为什么需要 online softmax？
4. FlashAttention 是否把精确 attention 变成线性时间？它主要减少了什么？

## 资料定位

| 主题 | 原始资料位置 |
|---|---|
| 参数量与架构假设 | `01` parameter slides PDF 页序 6-14、16-24 |
| FLOPs 与输入 embedding 口径区别 | `01` PDF 页序 26-35；原稿第 27 页按 one-hot 稠密乘法计输入 embedding，笔记采用实际查表口径 |
| 训练显存、激活与例子 | `02` memory slides PDF 页序 4-5、12-17；7B 的 2 GB 激活另见第 10 讲课件 |
| Online softmax、融合与 HBM | `03` FlashAttention slides PDF 页序 19-24、27-39；IO 适用条件补充自原论文 |

下一步在同一显存框架中审视 Adam、二阶近似和优化器状态：[[Lecture-09_Advanced-Optimizers]]。
