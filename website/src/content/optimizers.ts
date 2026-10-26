export const optimizerComparisons = [
  {
    name: "Adam",
    mechanism: "一阶 EMA 与逐坐标二阶矩预条件",
    states: "m, v",
    memory: "2mn",
    count: (m: number, n: number) => 2 * m * n,
    compute: "O(mn) 每步",
    assumptions: "β₁, β₂ ∈ [0,1)，ε > 0；收敛需额外目标与步长条件",
    limits: "自适应不保证普遍收敛或泛化优于 SGD；额外权重、梯度另计",
  },
  {
    name: "Adafactor（无一阶矩）",
    mechanism: "用行列边缘统计近似二阶矩",
    states: "行 r、列 c",
    memory: "m+n",
    count: (m: number, n: number) => m + n,
    compute: "O(mn) 每步",
    assumptions: "矩阵权重；分母正或有数值保护",
    limits:
      "因子化是近似；启用一阶矩还需 mn 状态，真实实现含 clipping 与相对步长",
  },
  {
    name: "Shampoo（矩阵版）",
    mechanism: "L⁻¹/⁴ G R⁻¹/⁴ 两侧预条件",
    states: "L ∈ Rᵐˣᵐ, R ∈ Rⁿˣⁿ",
    memory: "m²+n²",
    count: (m: number, n: number) => m * m + n * n,
    compute: "统计与应用 O(m²n+mn²)；分解 O(m³+n³)",
    assumptions: "εI 正定初始化；矩阵负幂有定义",
    limits: "大矩阵需分块与周期分解；这里未计额外动量或缓存的预条件矩阵",
  },
  {
    name: "Muon（动量核心）",
    mechanism: "动量矩阵经正交化产生方向",
    states: "动量 M",
    memory: "mn（持久核心状态）",
    count: (m: number, n: number) => m * n,
    compute: "NS 每次 O(mn min(m,n))；精确 SVD 代价依形状",
    assumptions: "主要用于二维矩阵参数；NS 系数、归一化与步数依实现",
    limits:
      "临时正交化缓冲不在状态计数内；非矩阵参数常需其他优化器；无普遍硬件收益保证",
  },
];
