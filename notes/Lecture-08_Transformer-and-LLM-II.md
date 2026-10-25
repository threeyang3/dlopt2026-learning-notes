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

层归一化的 $2h$ 参数相对很小，但在逐层显存方程中不能假装完全不存在。

## 2. FLOPs

**核心掌握主项，熟悉常数。** 一次矩阵乘法 $X_{s\times h}W_{h\times h'}$ 的主计算量约为 $2shh'$ FLOPs（每个乘加计 2）。

- 输入 token embedding 通常是按 ID 查表，主项是访存而非稠密矩阵乘；输出头是 $s\times h$ 乘 $h\times v$，约 $2svh$ FLOPs（每个 token 为 $2vh$）。
- 每层 attention 的 Q/K/V/O 投影与 attention 点积、加权：主项约 $8sh^2+4s^2h$。
- 每层 FFN：约 $16sh^2$。
- 因而每层前向约 $24sh^2+4s^2h$；长序列时 $4s^2h$ 才会显著成为主项。

对全模型，前向主项可近似 $2Ps$（prefill 整段）或每 token $2P$ 的口径；反向传播约为前向的 2 倍，所以前向加反向共约 3 倍前向，训练常用

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

输出也按指数权重重新缩放并累加。这样无需先看到整行就得到数值稳定的 softmax。

**重要边界：** FlashAttention 仍是精确 attention，而不是近似稀疏 attention；它减少中间矩阵的存储和 HBM 读写。因为不能读取完整矩阵，反向传播通常需要重算部分中间量。粗略 HBM 访问量可从 $O(N^2)$ 降到 $O(N^2d^2/M)$ 量级，$M$ 是片上存储规模。

## 4. 训练显存五分量

**核心掌握。** 统一使用

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
| 激活 | 依赖 batch/序列/模型 | 课件示例约 2 GB |
| 合计（不含额外主权重和临时缓冲区） | 8 + 激活 | 约 58 GB |

这里 28GB 明确对应“两个 16-bit 状态”。若两个状态都存成 FP32，则需约 $8P$ 字节，即 7B 参数约 56GB（十进制），不是 28GB。若另存 FP32 主权重、FP32 梯度或通信缓冲，还要继续加项。GPT-3 175B 的模型、梯度和优化器状态合计已达 TB 级。激活为重算、LoRA、子空间和零阶方法提供了节省空间，见 [[Lecture-09_Advanced-Optimizers]]、[[Lecture-10_Subspace-Optimizers]]、[[Lecture-11_Zeroth-Order-Optimizers]]。

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
2. 为什么 attention 长序列时显存和计算都按 $O(N^2)$ 增长？
3. FlashAttention 为什么需要 online softmax？
4. FlashAttention 是否把精确 attention 变成线性时间？它主要减少了什么？

## 资料定位

| 主题 | 原始资料位置 |
|---|---|
| 参数量与 LLaMA 示例 | parameter slides PDF p. 3-24 |
| 前向、反向 FLOPs | parameter slides PDF p. 26-36 |
| 训练显存分解与案例 | memory slides PDF p. 2-17 |
| FlashAttention、online softmax 与 HBM | FlashAttention slides PDF p. 1-39 |

下一步在同一显存框架中审视 Adam、二阶近似和优化器状态：[[Lecture-09_Advanced-Optimizers]]。
