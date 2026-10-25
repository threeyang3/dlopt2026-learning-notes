# 第 1 讲：课程导论与深度学习建模

> 配套文件：`materials/Lecture-01_Introduction/01_introduction-to-deep-learning_slides.pdf`、`02_introduction-to-llm_slides.pdf`

## 学习目标

完成本讲后，应能把“训练一个深度网络”改写成优化问题，并说清它为什么难；同时知道大语言模型从预训练基座到助手模型的基本流程，为 [[Lecture-07_Transformer-and-LLM-I]] 和 [[Lecture-14_Efficient-Fine-Tuning-and-Inference]] 建立全局地图。

## 1. 从分类器到训练目标

**核心掌握。** 多分类器先为类别 \(c\) 计算分数

\[
s_c=w_c^\top h+b_c,
\]

再经 softmax 变成概率

\[
p_c=\frac{e^{s_c}}{\sum_{j=1}^{C}e^{s_j}}.
\]

真实标签使用 one-hot 向量 \(q\)，交叉熵为

\[
\ell(p,q)=-\sum_{c=1}^{C}q_c\log p_c.
\]

若样本 \((a_i,b_i)\) 上的模型是 \(h(x;a_i)\)、损失是 \(L(h(x;a_i),b_i)\)，参数 \(x\) 通过

\[
\min_x\quad F(x)=\frac1m\sum_{i=1}^{m}L(h(x;a_i),b_i)
\]

学习得到。注意这里同时出现了“数据平均”和“非线性模型”两层结构，后面分别演化为 mini-batch 和深度网络。

常见的 DNN 结构是

\[
a\longrightarrow \text{feature extractor}\longrightarrow \text{classifier}\longrightarrow \hat b.
\]

卷积网络、Transformer 的差别主要在特征提取器；最后的分类或语言模型头仍然可以写成“仿射变换 + softmax + 损失”。

## 2. 为什么深度学习优化困难

**核心掌握。** 讲义把困难概括为三个来源：

1. **非凸**：损失面可能有局部极小、鞍点和不平坦区域，无法保证找到全局最优。
2. **高维**：参数可能有百万到万亿个，精确 Hessian、二阶法或网格搜索都不可行。
3. **大数据**：计算完整目标梯度需要遍历大量样本，因而必须使用随机或小批量估计。

因此实际训练通常追求“足够好的参数”和“稳定下降到驻点”，而不是严格求解全局最小。这个现实目标会在 [[Lecture-05_Gradient-Descent#4. 收敛分析的三条路线|GD 的非凸收敛]] 和 [[Lecture-06_Stochastic-Gradient-Descent#3. SGD 的收敛图景|SGD 的收敛分析]] 中变成可证明的结论。

## 3. LLM 是什么

**核心掌握。** 语言模型可看成给定上下文后预测下一 token 的概率函数：

\[
p_\theta(x_t\mid x_{<t}).
\]

参数 \(\theta\) 是把高维输入映射到词表概率的神经网络权重。自回归生成就是反复采样或选取下一 token，再接回上下文。训练数据、模型大小和训练 token 数共同决定能力；不能只看参数量。

## 4. ChatGPT 式训练流水线

**核心掌握四个阶段的角色，熟悉训练细节。**

| 阶段 | 输入 | 学到什么 | 主要代价 |
|---|---|---|---|
| Pretraining | 大规模无标注文本 | 一般语言规律与世界知识 | 最大，占绝大部分训练资源 |
| Supervised Fine-tuning, SFT | 高质量 `(prompt, response)` 对 | 按助手方式回答 | 数据标注与较小规模训练 |
| Reward modeling, RM | 同一 prompt 的多个回答排序 | 给回答质量打分 | 人类偏好比较数据 |
| Reinforcement learning | 模型生成回答 + RM 打分 | 偏向高奖励回答 | 采样、稳定性与奖励误差 |

预训练先通过分词把文本编码成整数序列，再使用 Transformer 最小化下一 token 交叉熵。预训练模型具有强迁移能力，但直接部署时未必会稳定遵循指令；SFT、RM 和 RL 把“会续写文本”变成“像助手一样回答”。

**了解。** RLHF 的具体强化学习算法、奖励模型训练目标以及不同厂商的数据配比；这些不是本课程的主推导对象。

## 5. 如何高效使用 LLM

讲义把 prompting、RAG、工具调用和微调放在同一决策框架中：

- **Chain of Thought（思维链）**：把复杂任务拆成中间步骤。
- **Tree of Thoughts**：展开多个思路，评价后继续搜索。
- **Prompt ensemble / reflection**：多提示集成、自我检查，降低单次推理偏差。
- **RAG**：检索外部资料，再把相关内容放进上下文，适合更新快或私有的知识。
- **工具调用**：把计算、查询、代码执行交给外部系统。
- **Fine-tuning**：用数据改变模型行为，适合专门任务，但成本和维护要求更高。

选择原则：上下文能解决就用 prompting/RAG；需要实时知识优先 RAG；需要稳定输出格式、专业行为或特定能力时才优先微调。LoRA 等低成本微调将在 [[Lecture-14_Efficient-Fine-Tuning-and-Inference#2. 参数高效微调方法|第 14 讲]] 展开。

## 6. 关键术语

- **Token**：模型处理文本的基本离散单位，不一定等于一个英文单词。
- **Vocabulary**：所有可能 token 的有限集合。
- **Base model**：只经过预训练的下一 token 预测模型。
- **Assistant model**：经过指令微调与偏好优化、更适合对话的模型。
- **Inference**：固定参数后执行前向计算并生成结果；与 training 分开计成本。

## 仅作了解

- 课程评分、助教信息和微信群的行政信息。
- 具体模型排行榜、某代 LLaMA/GPT 的参数量和能力宣传。
- Prompt engineering 的具体案例；重在实际任务中迭代验证。

## 自测

1. 写出多分类交叉熵和 DNN 平均训练目标。
2. 为什么“非凸”和“高维”不能通过更精确的梯度公式自动消失？
3. Pretraining、SFT、RM、RL 各自解决什么问题？
4. 一个需要引用当天新闻的问答系统，prompting、RAG、微调中应优先考虑哪两种？

下一讲从最基本的导数与曲率开始：[[Lecture-02_Linear-Algebra-I]]。

