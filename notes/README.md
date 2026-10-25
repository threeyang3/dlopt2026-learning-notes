# 学习笔记索引

这套笔记按照课程当前 14 个 Lecture 组织，用 **核心掌握 / 熟悉 / 了解** 提供学习建议，每篇末尾有自测题。`[[...]]` 是 Obsidian 风格双链；原生 GitHub 不解析这种链接，下面的讲次索引使用标准 Markdown 链接，专题路线保留双链便于 Obsidian 使用。

- 每篇开头都有“课程定位”，说明当年核心与跨年补充内容。
- 每篇末尾都有“资料定位”，可快速回到对应 PDF 页码。
- 课程概要、课件覆盖和待确认内容的对照见 [课程范围](COURSE_SCOPE.md)。
- 原始论文与项目入口集中在 [延伸文献](REFERENCES.md)。
- 后续增补内容遵循 [编修规范](NOTE_STANDARD.md) 的定义、条件、例子、反例和来源结构。
- 第 5、6、9、10、11、12 讲的原始页码与版本差异见 [对照核查](../SOURCE_AUDIT.md)。
- 公式统一使用 Obsidian/MathJax 的 `$...$` 与 `$$...$$`。

## 按课程顺序

1. [第 1 讲：课程导论与深度学习建模](Lecture-01_Introduction.md)
2. [第 2 讲：线性代数 I](Lecture-02_Linear-Algebra-I.md)
3. [第 3 讲：线性代数 II](Lecture-03_Linear-Algebra-II.md)
4. [第 4 讲：概率与统计](Lecture-04_Probability-and-Statistics.md)
5. [第 5 讲：梯度下降与加速](Lecture-05_Gradient-Descent.md)
6. [第 6 讲：随机梯度下降](Lecture-06_Stochastic-Gradient-Descent.md)
7. [第 7 讲：Transformer 与 LLM I](Lecture-07_Transformer-and-LLM-I.md)
8. [第 8 讲：Transformer 与 LLM II](Lecture-08_Transformer-and-LLM-II.md)
9. [第 9 讲：面向 LLM 的高级优化器](Lecture-09_Advanced-Optimizers.md)
10. [第 10 讲：子空间与分块优化](Lecture-10_Subspace-Optimizers.md)
11. [第 11 讲：零阶优化](Lecture-11_Zeroth-Order-Optimizers.md)
12. [第 12 讲：分布式优化](Lecture-12_Distributed-Optimizers.md)
13. [第 13 讲：混合精度训练](Lecture-13_Mixed-Precision-Training.md)
14. [第 14 讲：高效微调与推理](Lecture-14_Efficient-Fine-Tuning-and-Inference.md)

## 三条复习主线

### 1. 数学 → 收敛率

[[Lecture-02_Linear-Algebra-I]] → [[Lecture-03_Linear-Algebra-II]] → [[Lecture-05_Gradient-Descent]] → [[Lecture-06_Stochastic-Gradient-Descent]]

核心问题：目标函数的曲率决定“能不能走快”；随机噪声决定“最终能停多准”。

### 2. Attention → 成本 → 高效算法

[[Lecture-07_Transformer-and-LLM-I]] → [[Lecture-08_Transformer-and-LLM-II]] → [[Lecture-09_Advanced-Optimizers]] → [[Lecture-10_Subspace-Optimizers]] → [[Lecture-11_Zeroth-Order-Optimizers]]

核心问题：先分清模型参数、梯度、优化器状态、激活和 KV Cache，再判断方法节省了什么。

### 3. 单机训练 → 集群训练 → 部署

[[Lecture-08_Transformer-and-LLM-II]] → [[Lecture-12_Distributed-Optimizers]] → [[Lecture-13_Mixed-Precision-Training]] → [[Lecture-14_Efficient-Fine-Tuning-and-Inference]]

核心问题：复杂度 $O(\cdot)$ 之外，通信、显存带宽、数值范围和硬件支持经常决定真实速度。

