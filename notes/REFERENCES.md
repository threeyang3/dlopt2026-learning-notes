# 主要延伸文献

本页只列与课程主题直接相关的原始论文、标准教材或项目页，便于从笔记回到方法来源。课程 PDF 的页码定位仍放在各 Lecture 笔记末尾。

## 数学与基础优化

| 讲次 | 文献 |
|---|---|
| 02-03 | Yuan, Song. *微积分与线性代数基础*：仓库内 DLOpt 2026 notes |
| 05 | Y. Nesterov. *Introductory Lectures on Convex Optimization*. Springer, 2003. |
| 05 | D. G. Anderson. “Iterative Procedures for Nonlinear Integral Equations.” 1965. |
| 06 | D. P. Kingma, J. Ba. [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). 2014. |
| 06 | J. Duchi et al. “Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.” JMLR, 2011. |

## Transformer 与系统

| 讲次 | 文献 |
|---|---|
| 07 | A. Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 2017. |
| 08 | T. Dao et al. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135). 2022. |
| 12 | M. Shoeybi et al. [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053). 2019. |
| 12 | B. McMahan et al. [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629). 2016. |
| 12 | Y. Lin et al. [Deep Gradient Compression](https://arxiv.org/abs/1712.01887). 2017. |

## 高级优化器

| 讲次 | 文献 |
|---|---|
| 09 | N. Shazeer, M. Stern. [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235). 2018. |
| 09 | V. Gupta et al. [Shampoo: Preconditioned Stochastic Tensor Optimization](https://arxiv.org/abs/1802.09568). 2018. |
| 09 | V. Vyas et al. [SOAP: Improving and Stabilizing Shampoo using Adam](https://arxiv.org/abs/2409.11321). 2024. |
| 09 | K. Jordan et al. [Muon optimizer project](https://github.com/KellerJordan/modded-nanogpt). |
| 09 | H. Liu et al. [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342). 2023. |

## 子空间、零阶与内存高效方法

| 讲次 | 文献 |
|---|---|
| 10 | J. Zhao et al. [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507). 2024. |
| 10 | Y. Chen et al. “A Memory Efficient Randomized Subspace Optimization Method for Training Large Language Models.” ICML, 2025. |
| 10 | H. Pan et al. [LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning](https://arxiv.org/abs/2403.17919). 2024. |
| 11 | S. Malladi et al. [Fine-Tuning Language Models with Just Forward Passes](https://arxiv.org/abs/2305.17333). 2023. |
| 11 | Y. Chen et al. “Enhancing Zeroth-Order Fine-Tuning for Language Models with Low-Rank Structures.” ICLR, 2025. |

## 低精度、微调与推理

| 讲次 | 文献 |
|---|---|
| 13 | P. Micikevicius et al. [Mixed Precision Training](https://arxiv.org/abs/1710.03740). 2017. |
| 13 | N. Mellempudi et al. [Mixed Precision Training with 8-bit Floating Point](https://arxiv.org/abs/1905.12334). 2019. |
| 13 | NVIDIA. [NVFP4: Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/). |
| 14 | E. J. Hu et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). 2021. |
| 14 | S. Liu et al. [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353). 2024. |
| 14 | W. Kwon et al. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). 2023. |
| 14 | G. Xiao et al. [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453). 2023. |
| 14 | Y. Leviathan et al. [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192). 2022. |

## 使用原则

- 笔记中的课程定义优先；论文用于理解方法来源、假设和实验边界。
- 论文报告的速度或显存数字与具体模型、硬件、batch、序列长度和基线有关，不能直接作为部署保证。
- 若论文与当年课件在符号或实现细节上不同，应在笔记中明确说明版本差异。

