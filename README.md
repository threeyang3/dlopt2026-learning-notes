# 深度学习最优化：DLOpt 2026 学习资料与笔记

[![Verify course notes](https://github.com/threeyang3/dlopt2026-learning-notes/actions/workflows/verify.yml/badge.svg)](https://github.com/threeyang3/dlopt2026-learning-notes/actions/workflows/verify.yml)

本目录整理自 [PKU DLOpt 2026 课程主页](https://kunyuan827.github.io/dlopt2026/) 当前 `Materials` 区域。网站共有 14 讲、35 个唯一 PDF，合计 1,179 页。资料来自 DLOpt 2024、DLOpt 2025、DLOpt 2026 以及 LLM 2025 的配套内容，已按本课程当前的 Lecture 归属重新组织。

本仓库持续开展体系化审校，重点检查数学条件、公式维度、数值格式、课程归属、Obsidian/GitHub 渲染和关键公式的数值回归。修订明细见 [审校修订记录](REVISION_NOTES.md)。

第 5、6、9、10、11、12 讲的关键公式与原始课件页码、版本差异和实际核查边界见 [原始资料对照核查](SOURCE_AUDIT.md)。

## 目录

```text
dlopt2026_learning_pack/
├── README.md                    # 课程总览与学习路线
├── REVISION_NOTES.md            # 本轮审校与修正记录
├── SOURCE_AUDIT.md              # 关键公式的原始课件对照与核查边界
├── course_page.html             # 课程主页快照
├── source_manifest.tsv          # 原链接、原文件名、角色与规范文件名
├── checksums.sha256             # 35 份 PDF 的 SHA-256 校验和
├── download_materials.sh        # 可重复运行的下载脚本
├── resource_inventory.md        # 每个 PDF 的内容与页数索引
├── .github/workflows/verify.yml # GitHub Actions 自动验证
├── tools/check_notes.py         # 双链、页内锚点、公式和清单检查
├── tools/test_math_notes.py     # 关键公式的数值回归测试
├── materials/                   # 按 Lecture 重命名并归档的原始 PDF
│   ├── Lecture-01_Introduction/
│   └── ...
└── notes/                       # 按 Lecture 组织的 Markdown 学习笔记
    ├── README.md
    ├── COURSE_SCOPE.md          # 当年课程范围与跨年资料归属
    ├── REFERENCES.md            # 主要原始论文与项目入口
    ├── NOTE_STANDARD.md         # 后续笔记的统一编修规范
    ├── Lecture-01_Introduction.md
    └── ...
```

文件名采用 `讲次内顺序_主题_文档角色.pdf`，例如：

```text
01_gradient-and-hessian_notes.pdf
03_flash-attention_slides.pdf
```

原始文件名与 URL 没有丢弃，全部记录在 [来源清单](source_manifest.tsv) 中。这样既方便按讲次学习，也能追溯网站更新。

## 学习路径

课程可以分成六个阶段。前一阶段没有掌握就进入后一阶段，容易把“算法名称”误当成理解。

| 阶段 | Lecture | 主线 | 进入下一阶段前应能回答 |
|---|---|---|---|
| 数学基础 | 01-04 | 深度学习问题建模；梯度、Jacobian、Hessian；矩阵结构；概率期望 | 为什么 DNN 训练是高维、非凸、大数据上的随机优化？ |
| 确定性优化 | 05 | 凸性、平滑性、GD、动量与加速 | 收敛率中的 $L,\mu$ 如何影响步长与迭代数？ |
| 随机优化 | 06 | SGD、小批量、动量 SGD、AdaGrad、RMSProp、Adam | 噪声为何使 SGD 只能稳到邻域？批量如何影响样本复杂度？ |
| 模型与系统基础 | 07-08 | Seq2Seq、Attention、Transformer；参数、FLOPs、显存、FlashAttention | $N$、批大小和模型宽度分别把哪一项成本放大？ |
| LLM 高效训练 | 09-13 | 高级优化器、子空间、零阶、分布式、低精度 | 每项技术究竟节省参数、梯度、优化器状态、激活还是通信？ |
| 高效适配与推理 | 14 | LoRA、DoRA、KV Cache、vLLM、投机解码 | 训练节省与推理节省分别发生在哪个阶段？ |

建议按 Lecture 顺序初次学习；复习时可按以下专题跳跃：

- 数学与收敛：[Lecture 02](notes/Lecture-02_Linear-Algebra-I.md) → [Lecture 03](notes/Lecture-03_Linear-Algebra-II.md) → [Lecture 05](notes/Lecture-05_Gradient-Descent.md) → [Lecture 06](notes/Lecture-06_Stochastic-Gradient-Descent.md)
- 自适应与二阶：[Lecture 06](notes/Lecture-06_Stochastic-Gradient-Descent.md) → [Lecture 09](notes/Lecture-09_Advanced-Optimizers.md)
- 显存主导项：[Lecture 08](notes/Lecture-08_Transformer-and-LLM-II.md) → [Lecture 09](notes/Lecture-09_Advanced-Optimizers.md) → [Lecture 10](notes/Lecture-10_Subspace-Optimizers.md) → [Lecture 11](notes/Lecture-11_Zeroth-Order-Optimizers.md)
- 训练系统：[Lecture 12](notes/Lecture-12_Distributed-Optimizers.md) → [Lecture 13](notes/Lecture-13_Mixed-Precision-Training.md)
- 部署：[Lecture 07](notes/Lecture-07_Transformer-and-LLM-I.md) → [Lecture 14](notes/Lecture-14_Efficient-Fine-Tuning-and-Inference.md)

## 掌握等级

每篇笔记都使用三级标记：

- **核心掌握**：必须能准确解释、写出关键式子，并说明适用条件。
- **熟悉**：应能判断何时使用，能把算法接回主线，不要求完整复现证明。
- **了解**：知道问题背景、代表方法和主要结论；复习或实际使用时能查回资料即可。

“核心”不等于“会背”。例如 Adam 的核心是理解一阶动量、二阶累积与偏差修正各自解决什么；完整收敛证明则可按讲义选择性掌握。

以上等级是笔记的学习建议，课程主页并未给出所有知识点的考核等级，仓库也没有完整课堂范围记录。

每篇讲义的“课程定位”说明当年核心与跨年补充内容；详细对照见 [课程范围与资料归属](notes/COURSE_SCOPE.md)。课件概要、课件正文和课堂说明不一致时，优先采用教师当年的明确说明。

## 使用建议

1. 先读本讲笔记的“学习目标”和“先修关系”，再浏览原 PDF 的标题页与目录。
2. 第一遍只看核心内容，合上资料写出算法更新式、代价公式和一个反例。
3. 第二遍阅读“仅作了解”部分，把新方法和已经掌握的方法做一句话比较。
4. 最后完成每篇末尾的自测题，并沿双链回到相关讲次。
5. 遇到显存或计算成本，统一回到“模型参数 + 梯度 + 优化器状态 + 激活 + 通信”五分量框架。

公式已统一为 Obsidian/MathJax 的 `$...$` 与 `$$...$$` 语法，并避开 GitHub 预览不支持的常见宏。每篇末尾的“资料定位”表给出重要主题在 PDF 中的页码。

## 完整性说明

- 网站中注释掉的历史内容不属于当前 `Materials`，本次未混入当前讲次。
- 同一 Lecture 4 的两个条目指向同一份概率讲义，因此只保存一份。
- 所有 35 个文件均已通过 PDF 文件类型与页数检查。内容索引见 [资源清单](resource_inventory.md)。
- 笔记中的推导用于串连讲义，符号可能与不同年份原稿略有差异；引用具体结论时以对应 PDF 和笔记顶部的配套文件为准。
- 下载脚本会在获取后检查 PDF 文件类型，并用 `checksums.sha256` 验证内容没有变化。
- 在仓库根目录运行 `python3 tools/check_notes.py` 可检查双链、相对链接、公式定界符、课程标签和资料清单。
- 运行 `python3 tools/test_math_notes.py` 可执行关键公式的纯标准库数值测试，无需安装 NumPy。
- 数值冒烟测试和文档回归用于发现具体错误，不构成对全部数学结论的自动证明。
- GitHub Actions 会在 push 和 pull request 时自动运行以上两项检查。

## 版权与使用范围

`materials/` 中的 PDF 版权归原作者及原课程机构所有，仅作个人学习与课程复习留存；本仓库不应被视为这些原始课件的再发布授权。

