# 课程范围与资料归属

本页用于避免把“课件里出现的内容”直接等同于“当年考试重点”。判断顺序应为：

1. 教师当年课堂说明；
2. 2026 课程主页 `Materials` 概要；
3. 对应 PDF 正文；
4. 跨年份完整课件和延伸论文。

| Lecture | 2026 官网强调 | 当前资料与笔记状态 |
|---|---|---|
| 01 Introduction | 深度学习导论、LLM 导论 | 2026 导论 + 2025 LLM slides；属课程核心 |
| 02 Linear Algebra I | 梯度/Hessian、线性变换、特征值 | 2026 notes；属数学核心 |
| 03 Linear Algebra II | Jacobian/Hessian、链式法则、矩阵函数、范数、SVD | 2026 notes；属数学核心 |
| 04 Probability and Statistics | 概率、随机变量、期望、方差、条件分布 | 2026 notes；属数学核心 |
| 05 Gradient Descent | 凸性、GD、Momentum、NAG、Anderson | 综合 LLM 2025 与 DLOpt 2024；属优化核心 |
| 06 Stochastic Gradient Descent | SGD、Momentum SGD、AdaGrad、RMSProp、Adam | DLOpt 2024 slides/notes；属优化核心 |
| 07 Transformer and LLM I | Word Embedding、Seq2Seq、Attention、Transformer | DLOpt 2024 slides；属模型基础核心 |
| 08 Transformer and LLM II | Transformer 参数/计算/显存、Memory Access、FlashAttention | DLOpt 2024 slides；属系统基础核心 |
| 09 Advanced Optimizer | Shampoo、SOAP、Muon、Lite、MuonM | 课件还含 Adafactor、Adam-mini、Sophia；Lite/MuonM 在当前 72 页课件中未找到独立章节，需按课堂更新确认 |
| 10 Subspace Optimizer | Adam-mini、内存高效优化器、BCD、分块训练 | Adam-mini 详解放第 9 讲；本讲深入 GaLore/GoLore/RSO、BCD、LISA |
| 11 Zeroth-Order Optimizer | 有限差分、线性插值、球面平滑、MeZO、LOZO | 2025 slides + notes；属高效训练核心 |
| 12 Distributed Optimizer | 三种并行、去中心化、压缩与 Local Learning | DLOpt 2024 slides；属训练系统核心 |
| 13 Mixed-Precision Training | FP16/FP32、混合精度、FP8、NVFP4 | DLOpt 2024 + 2025 slides；属训练系统核心 |
| 14 Efficient Fine-Tuning and Inference | Adapter/LoRA/DoRA/LISA、KV Cache/vLLM | DLOpt 2024 slides；属适配与部署核心 |

## 标记规则

- **课程核心**：官网明确列出且当前资料中有实质展开。
- **跨年补充**：来自同一课程历史版本、用于补足背景或后续方法的内容。
- **需确认**：官网列名与当前 PDF 正文未能一一对应，不能据此推断考试范围。

Lite、MuonM 目前归入“需确认”。如果课堂或教师后续材料给出定义、更新式和实验设置，应把它们补成独立小节，而不是只在方法对照表中增加名称。

