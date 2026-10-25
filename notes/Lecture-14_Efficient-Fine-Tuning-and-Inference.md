# 第 14 讲：参数高效微调与高效推理

> **课程定位**：2026 年课程核心微调与推理；复用 DLOpt 2024 课件，部分性能数字属于文献实验条件而非保证值。
>
> 配套文件：`materials/Lecture-14_Efficient-Fine-Tuning-and-Inference/01_efficient-fine-tuning_slides.pdf`、`02_efficient-inference_slides.pdf`

## 学习目标

把“训练阶段省什么”和“推理阶段省什么”分开：前者主要缩小可训练参数、梯度与优化器状态，后者主要管理 KV Cache、注意力计算和请求调度。应能说明 LoRA、DoRA、LISA、GQA/MLA、StreamingLLM、vLLM 和投机解码各解决哪一环。

## 1. 先选微调策略

**核心掌握决策。**

- **Prompting**：不改权重，适合快速原型和已有能力。
- **RAG**：运行时检索外部资料，适合动态、可追溯的知识。
- **Fine-tuning**：用数据稳定改变行为/格式，适合固定任务和专业能力。

微调可以看作从预训练的良好初始化出发，继续最小化任务损失。全参数微调保存全部参数的梯度、优化器状态和激活，对数十亿参数模型很贵；为每个下游任务保存整份权重也不方便部署。

## 2. 参数高效微调方法

### BitFit

**了解。** 只训练偏置 $b$，冻结权重和多数参数。实现简单，但表达能力有限。

### Adapter

**熟悉。** 在每个 Transformer 模块插入小的 bottleneck 网络，冻结主体参数，只训练 adapter。它会改变层结构，部署时需兼容 adapter 模块。

### LoRA

**核心掌握。** 冻结原权重 $W_0$，学习低秩增量：

$$
W'=W_0+\Delta W
=W_0+\frac{\alpha}{r}BA,
$$

$$
B\in\mathbb R^{d\times r},\qquad
A\in\mathbb R^{r\times k},\qquad
r\ll\min(d,k).
$$

可训练参数从 $dk$ 降为 $r(d+k)$。通常初始化 $A$ 很小/随机、 $B=0$，使增量初始为 0；这样训练起点与原模型一致。推理可把 $BA$ 合并进 $W_0$，但需要先确认量化、融合与精度策略。

**内存收益：** 主要节省被冻结参数的梯度与 Adam 状态；模型权重和部分激活仍然存在。LoRA 不自动减少前向矩阵乘的 FLOPs，部署合并后才可能获得低延迟。

### LoRA+

**熟悉。** 对 $A,B$ 使用不同学习率，避免二者同尺度更新导致收敛慢。初始化为零的因子和随机因子的更新需求不相同，因此统一学习率并非最优。

### DoRA

**熟悉。** 把权重分成幅度与方向：

$$
W=m\frac{V}{\|V\|_c},
$$

训练幅度 $m$ 与低秩方向更新 $V$。它把“方向改变”和“尺度改变”解耦，更贴近全参数微调的行为。实现会增加规范化/分解计算。

### LISA

**熟悉。** 逐层激活一部分 Transformer block，冻结其余层。它节省非激活层的梯度、优化器状态和激活，但有层间更新不同步，通常适合微调而非从零预训练。它和 [[Lecture-10_Subspace-Optimizers#6. 分块/逐层训练 LISA|第 10 讲]] 是同一系统思想。

## 3. 自回归推理与 KV Cache

**核心掌握。** 生成 token $t$ 时，模型条件于之前所有 token。每个新 token 的注意力需要历史 token 的 K、V；若每步重算，就浪费大量前向计算。KV Cache 保存历史层、头、token 的 $K,V$。

粗略显存：

$$
\text{KV bytes}\approx
2\times b\times s\times l\times h\times\text{bytes per element},
$$

其中 2 对应 K 和 V；若多头维度与 $h$ 不同，要按实际每层 KV 宽度计算。上下文长、批量大、层多时，KV Cache 会成为长上下文推理的主内存项。

推理有两个阶段：

- **Prefill**：并行处理 prompt，矩阵较大，往往 compute-bound；
- **Decode**：每步只生成少量 token，需要加载历史和权重，往往 memory-bandwidth-bound。

## 4. 减少 KV Cache

### 架构共享

- **MQA**：多个 query 头共享一个 K/V 头，显著缩小缓存；
- **GQA**：将 query 头分组，每组共享一组 K/V，折中质量与带宽；
- **MLA**：联合压缩 K/V 的低秩潜在表示，DeepSeek-V2 使用该路线。

共享 K/V 通常需要预训练配方或继续微调来适配，不是简单运行时开关。

### 稀疏与淘汰

- 观察 attention 具有稀疏性，只保留重要 token；
- H2O 用 heavy hitters 动态淘汰，保留对生成影响大的历史；
- StreamingLLM 观察到 attention sink：即使初始 token 语义不突出，也常获得大权重；保留部分 sink token 与最近窗口可稳定流式生成；
- Quest 通过块级相关性/预测，按需读取相关 KV 块，减少 eviction 后不可恢复的问题。

### 跨层共享与量化

KV Cache 在相邻层可能相似，可跨层共享。量化时可针对 key/value 的不同分布分别处理，例如 key 用 per-channel、value 用 per-token 量化；长上下文的异常通道会影响策略。

## 5. vLLM 与 PagedAttention

**核心掌握系统思想。** 朴素批处理为每个请求预分配最大可能长度的连续 KV 空间，导致：

- 内部碎片：预留过长；
- 外部碎片：不同长度请求很难填满；
- 无法方便共享相同前缀。

PagedAttention 借鉴操作系统虚拟内存和分页：把 KV 存成固定大小 block，逻辑连续、物理可不连续，用 block table 把逻辑块映射到物理块，并按需分配。

收益是提升显存利用率、允许请求细粒度拼接与退出，并支持前缀共享；讲义报告在相近延迟下吞吐提高约 2-4 倍。这个数字来自特定模型、负载和基线对比，不是任意部署的保证。它主要解决 **KV 管理效率**，不等同于减少单次 attention 的数学 FLOPs。

## 6. 投机解码

**熟悉。** 小 draft model 一次提出 $K$ 个候选 token，大 target model 并行验证；接受前缀，遇到拒绝后回退。验证机制设计正确时输出分布可与 target model 保持一致，而不是以质量换速度。

它适合“预测容易、验证可并行”的负载；K 过小收益有限，K 过大浪费 draft 计算。讲义报告在合适设置下约 2-3 倍加速，具体数值依赖模型对、数据和硬件。

## 7. 训练与推理成本对照

| 技术 | 训练收益 | 推理收益 | 关键代价 |
|---|---|---|---|
| LoRA/DoRA | **可训练参数**及其梯度、优化器状态大幅缩小 | 合并后可能降低增量成本 | 子空间表达有限 |
| Adapter | 缩小可训练参数 | 保留额外层 | 结构改动 |
| LISA | 缩小梯度/状态/激活 | 不直接改善推理 | 收敛较慢 |
| GQA/MLA | 训练适配 | 减少 KV Cache 与带宽 | 架构与训练成本 |
| H2O/StreamingLLM | 不适用 | 减少历史 KV | 可能丢失远距离信息 |
| PagedAttention | 不适用 | 提升缓存利用率与吞吐 | 内存管理复杂 |
| 投机解码 | 不适用 | 并行验证多个 token | draft 误差/浪费 |

## 仅作了解

- LoRA 各层的秩分配、DoRA 的完整实验结果。
- 所有 attention sink 的理论解释、Quest 的预测器和复杂量化粒度。
- PagedAttention 的 kernel 与调度器实现。
- 投机解码的严格接受概率推导；核心是“用小模型提议、大模型验证”。

## 自测

1. LoRA 可训练参数为什么从 $dk$ 降成 $r(d+k)$？
2. 为什么 LoRA 节省训练显存，却不一定减少普通推理 FLOPs？
3. Prefill 与 decode 的瓶颈分别是什么？
4. PagedAttention 解决的是 KV Cache 的计算、容量还是碎片问题？
5. 投机解码在什么条件下能提高速度而不改变 target model 的生成分布？

## 资料定位

| 主题 | 原始资料位置 |
|---|---|
| 微调决策、BitFit、Adapter、LoRA、DoRA、LISA | fine-tuning slides PDF p. 2-38 |
| 自回归推理、KV Cache 与压缩 | inference slides PDF p. 3-28 |
| 投机解码、vLLM/PagedAttention | inference slides PDF p. 29-45 |

回到学习路线：[学习笔记索引](README.md) 或 [课程总览](../README.md)。
