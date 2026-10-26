export type Verification = "verified" | "partially_verified" | "unverified";
export interface LearningModule {
  lectures: number[];
  goal: string;
  prerequisites: string;
  conditions: string;
  misconception: string;
  experiment: string;
  status: Verification;
  evidence: string;
}
export const modules: Record<string, LearningModule> = {
  gd: {
    lectures: [5],
    goal: "从曲率与梯度解释更新，区分 Heavy-ball、NAG 与 GD 的评估点。",
    prerequisites: "梯度、谱分解、欧氏范数（第 2–3 讲）",
    conditions:
      "二维对称正定二次函数；μ=1；确定性精确梯度。一般光滑下降证明独立说明。",
    misconception:
      "下降界不等于一般非凸线性收敛；超过稳定边界不意味着每个特殊初值都发散。",
    experiment:
      "保持初值与目标不变，改变 γ；观察谱因子、函数间隙与陡方向振荡是否一致。",
    status: "verified",
    evidence:
      "仅限本页 SPD 递推和 GD 下降推导：解析梯度/手算更新/下降不等式数值测试；课件 GD 11、14，加速 5、19、23 页已读取，14、23 页另看原图。不涵盖整讲所有定理。",
  },
  sgd: {
    lectures: [4, 6],
    goal: "区分单样本、批平均与真实梯度，检查 Adam 矩状态和修正。",
    prerequisites: "GD、条件期望、独立性与方差",
    conditions: "独立同分布的加性高斯噪声，每坐标方差 σ²；矩状态零初始化。",
    misconception: "无偏不等于无噪声；偏差修正不使 Adam 最终方向无偏。",
    experiment: "固定种子对比 B=1 与 B=32，再比较 16 条路径的波动与样本代价。",
    status: "partially_verified",
    evidence:
      "随机重现、批平均、经验方差和 Adam 首步有测试；课件 Adam 第 18 页已读，偏差修正按现有笔记补充，尚未独立逐式核对原论文。",
  },
  linear: {
    lectures: [2, 3],
    goal: "把矩阵、线性变换、奇异值与低秩残差联系起来。",
    prerequisites: "矩阵乘法、正交基、Frobenius 范数",
    conditions: "有限实矩阵 G∈R³ˣ²；r=1 或 2；SVD 由 ml-matrix 实际计算。",
    misconception:
      "二维图只显示前两行，不代表完整 3D 变换；低秩表示不必在小矩阵上省存储。",
    experiment: "缩小一列尺度，观察尾部奇异值与秩一误差同步变化。",
    status: "partially_verified",
    evidence:
      "已知奇异值、正交性、尺寸与完全重构有独立数值测试；SVD 最优性证明与原稿全部内容未在本轮复核。",
  },
  subspace: {
    lectures: [3, 10],
    goal: "检查 PᵀG 与 PPᵀG 的形状和投影误差，区分表示与状态。",
    prerequisites: "SVD、正交投影、Adam 状态",
    conditions: "P 是前 r 个左奇异向量；PᵀP=I；仅模拟投影，不执行完整 GaLore。",
    misconception: "低秩梯度表示、优化器状态压缩和反向激活节省不是同一件事。",
    experiment: "比较 r=1/2 的残差和基+两个压缩状态总元素数。",
    status: "partially_verified",
    evidence:
      "投影维度、正交性和解析残差有测试；实际 GaLore 周期换基和历史状态不在实现范围。",
  },
  zeroth: {
    lectures: [11],
    goal: "用查询值恢复近似方向，分离截断误差、方向方差和查询噪声。",
    prerequisites: "梯度、Taylor 展开、球面与高斯分布",
    conditions:
      "给定可解析光滑四次函数；τ>0；球面为均匀角度，高斯为独立标准正态。",
    misconception: "球面与球体分布不同；一个随机方向不等于完整梯度。",
    experiment: "先关闭噪声缩小 τ，再打开查询噪声观察误差反向增大。",
    status: "partially_verified",
    evidence:
      "解析梯度、中心差分误差阶数、查询数、方向归一化和 MeZO 缩放有测试；平滑恒等式沿用原笔记，未本轮重读全部原论文。",
  },
  distributed: {
    lectures: [12],
    goal: "区分梯度聚合、参数混合与本地多步更新，核算通信账本。",
    prerequisites: "SGD、均匀平均、矩阵混合",
    conditions:
      "二维异质强凸二次目标；等权全参与；独立噪声；DSGD 使用先更新后环混合。",
    misconception: "连通图不意味着一次全局平均；每轮压缩不等于总通信更少。",
    experiment: "对比 Local K=1 与 PSGD，再增大 K 和异质性观察共识误差。",
    status: "partially_verified",
    evidence:
      "K=1 等价、DSGD 均值、非零共识、Random-k 经验无偏与字节累加有测试；未测真实网络。",
  },
  attention: {
    lectures: [7],
    goal: "逐步计算 Q/K/V、mask、softmax、双头拼接和输出投影。",
    prerequisites: "矩阵乘法、指数与 softmax",
    conditions: "N≤6，模型宽度4，两个头各宽2；因果 mask 保留自身。",
    misconception:
      "权重不是所有语境下可靠的语义重要性；attention 核心不含所有 Transformer 组件。",
    experiment: "切换 mask 与种子，检查未来位置权重和行权重和。",
    status: "partially_verified",
    evidence:
      "手算均匀 attention、行归一化、mask 和数值稳定性有测试；未核验所有模型架构。",
  },
  flash: {
    lectures: [8],
    goal: "通过运行最大值、归一化和与累积输出理解分块机制。",
    prerequisites: "缩放 attention、稳定 softmax",
    conditions:
      "同一 query 的有限有效 keys；因果未来 keys 排除；分块最后输出与稠密对照。",
    misconception: "减少中间矩阵 IO 不意味着精确 attention 变成线性计算。",
    experiment: "改变块大小，检查每块 m/ℓ/z 的变化与最终输出不变性。",
    status: "partially_verified",
    evidence:
      "多种块大小与独立稠密结果数值一致；课件21页原图已核对。没有实现或测量 GPU kernel。",
  },
  precision: {
    lectures: [13],
    goal: "观察指数、尾数、次正规数与低精度更新丢失。",
    prerequisites: "二进制浮点、相对精度",
    conditions:
      "最近偶数舍入；保留次正规数；FP32 采用 Math.fround；不模拟硬件 flush-to-zero。",
    misconception: "同样指数范围不等于同样精度；缩放不能挽救已经溢出的值。",
    experiment: "输入65520、2⁻²⁴附近数值与1.001，对比范围和分辨率。",
    status: "partially_verified",
    evidence:
      "FP16 精确边界、偶数中点、次正规、溢出与 BF16 对比测试通过；未硬件逐编码穷举。",
  },
  inference: {
    lectures: [14],
    goal: "核算 LoRA 因子和逐 token KV 形状与内存。",
    prerequisites: "矩阵秩、Attention 头与张量形状",
    conditions:
      "LoRA W为4×3；KV各层头数、头宽一致，只计有效token和给定元素字节。",
    misconception:
      "LoRA 不消除模型权重；KV 线性增长与 attention 权重矩阵二次增长是不同账本。",
    experiment: "翻倍 batch、token 或层数，检查缓存估算是否线性变化。",
    status: "partially_verified",
    evidence:
      "LoRA 独立参数计数和 K/V 两份内存手算有测试；没有真实训练、推理或硬件峰值测量。",
  },
  resources: {
    lectures: [9],
    goal: "按统一状态口径比较优化器内存与额外计算。",
    prerequisites: "Adam、矩阵乘法与渐近复杂度",
    conditions:
      "单个矩阵、声明状态精度；Adafactor 无一阶矩，Shampoo 矩阵累积版，Muon 仅动量核心。",
    misconception: "持久状态不等于总训练显存；特定实现性能不构成普遍保证。",
    experiment: "增大一边矩阵维度，观察行列因子和平方矩阵统计的成本变化。",
    status: "partially_verified",
    evidence:
      "计数来自明确状态形状；算法条件参照现有第9讲，尚未独立复核所有官方实现。",
  },
};
