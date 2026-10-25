# 原始资料清单

课程主页当前有 14 个 Lecture，共 35 个唯一 PDF、1,179 页、约 128.3 MiB。下表的页码范围由 PDF 文件本身读取；网页上重复指向同一文件的条目只计一次。

| Lecture | 文件与页码范围 | 页数 | 主要内容 |
|---|---|---:|---|
| 01 Introduction | `01_introduction-to-deep-learning_slides.pdf` 1-22；`02_introduction-to-llm_slides.pdf` 1-65 | 87 | DNN 建模、优化困难、LLM、ChatGPT 训练流水线与 prompt/RAG |
| 02 Linear Algebra I | `01_gradient-and-hessian_notes.pdf` 1-10；`02_linear-transform-eigenvalues_notes.pdf` 1-12 | 22 | 梯度、Hessian、线性映射、相似矩阵、特征值 |
| 03 Linear Algebra II | `01_jacobian-hessian-chain-rule_notes.pdf` 1-10；`02_inner-product-matrix-functions-backprop_notes.pdf` 1-22；`03_norms-svd_notes.pdf` 1-18 | 50 | Jacobian、链式法则、正定、矩阵函数、范数与 SVD |
| 04 Probability and Statistics | `01_probability-statistics_notes.pdf` 1-34 | 34 | 事件、条件概率、分布、期望、方差、边缘/条件分布与协方差 |
| 05 Gradient Descent | `01_convexity-gradient-descent_slides.pdf` 1-28；`02_convexity-gradient-descent_notes.pdf` 1-9；`03_accelerated-gradient-descent_slides.pdf` 1-34；`04_accelerated-gradient-descent_notes.pdf` 1-11 | 82 | 凸性、强凸、光滑、GD、Polyak、NAG、Anderson 与下界 |
| 06 Stochastic Gradient Descent | `01_stochastic-gradient-descent_slides.pdf` 1-32；`02_stochastic-gradient-descent_notes.pdf` 1-11；`03_momentum-sgd_slides.pdf` 1-21；`04_momentum-sgd_notes.pdf` 1-6；`05_adaptive-sgd_slides.pdf` 1-21 | 91 | SGD、mini-batch、Momentum SGD、AdaGrad、RMSProp、Adam |
| 07 Transformer and LLM I | `01_transformer-attention_slides.pdf` 1-71 | 71 | Seq2Seq、RNN 瓶颈、attention、self-attention、multi-head 与 Transformer |
| 08 Transformer and LLM II | `01_transformer-parameter-computation_slides.pdf` 1-36；`02_transformer-memory-cost_slides.pdf` 1-17；`03_flash-attention_slides.pdf` 1-39 | 92 | Transformer 参数、FLOPs、显存分解、HBM 与 FlashAttention |
| 09 Advanced Optimizers | `01_advanced-optimizers_slides.pdf` 1-72 | 72 | Adafactor、Adam-mini、Shampoo、SOAP、Muon、Sophia |
| 10 Subspace Optimizers | `01_golore-rso_slides.pdf` 1-61；`02_block-coordinate-descent_slides.pdf` 1-20；`03_block-wise-lisa_slides.pdf` 1-26 | 107 | GaLore、GoLore、RSO、块坐标下降、LISA |
| 11 Zeroth-Order Optimizers | `01_zeroth-order-gradient-descent_slides.pdf` 1-28；`02_zeroth-order-optimization_notes.pdf` 1-16；`03_low-rank-zeroth-order_slides.pdf` 1-28 | 72 | 有限差分、线性插值、球面平滑、MeZO、LOZO |
| 12 Distributed Optimizers | `01_parallelism_slides.pdf` 1-53；`02_decentralized-learning_slides.pdf` 1-91；`03_communication-compression-local-learning_slides.pdf` 1-72 | 216 | PSGD、3D 并行、去中心化 SGD、局部 SGD、压缩与拓扑 |
| 13 Mixed-Precision Training | `01_mixed-precision-foundations_slides.pdf` 1-46；`02_fp8-nvfp4_slides.pdf` 1-54 | 100 | FP16/FP32、主权重、loss scaling、FP8、NVFP4 与硬件 |
| 14 Efficient Fine-Tuning and Inference | `01_efficient-fine-tuning_slides.pdf` 1-38；`02_efficient-inference_slides.pdf` 1-45 | 83 | BitFit、Adapter、LoRA、DoRA、LISA、KV Cache、vLLM |

## 来源说明

- 当前课程页直接引用的 35 个 PDF 均已收录。
- 课程主页快照保存在 `course_page.html`，用于追溯当前 Materials 列表。
- DLOpt 2026 的线性代数与概率讲义原本是课程参考书章节，本次按 Lecture 切分归档，但 PDF 内保留原章节编号。
- Lecture 4 在网站上有两个条目指向同一份 `probability_chapter.pdf`，没有重复下载。
- 当前网页注释掉的历史材料未计作当前课程内容；原链接仍可从课程主页追溯。
- 精确的“原文件路径 -> 规范文件名”映射见 [source_manifest.tsv](source_manifest.tsv)。

## 重新下载与校验

在包含本目录的工作目录中运行：

```bash
bash dlopt2026_learning_pack/download_materials.sh
```

脚本会跳过已经是 PDF 的目标文件；若想强制刷新，删除对应 PDF 后重跑。PDF 的页数和文件类型已检查，但课程主页更新后仍应重新核对讲次与文件数。
