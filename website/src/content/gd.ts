export const gdDerivation = [
  {
    formula: String.raw`f(y)\le f(x)+\nabla f(x)^\top(y-x)+\frac L2\|y-x\|_2^2`,
    explanation:
      "L-光滑的下降引理：梯度为 L-Lipschitz 时，沿线段积分梯度变化得到二次上界。这里不需要凸性。",
    rule: "梯度 Lipschitz + 沿线段积分",
    conditions: "f 在 Rᵈ 上可微，梯度为 L-Lipschitz；L > 0",
    variables: "x：当前点；y：候选点；L：梯度变化上界",
  },
  {
    formula: String.raw`y=x_{k+1}=x_k-\gamma g_k,\quad g_k=\nabla f(x_k),\quad y-x_k=-\gamma g_k`,
    explanation:
      "将 GD 的候选点代入上界。梯度取在当前点 xₖ；Nesterov 则在外推点评估，不能直接混用。",
    rule: "GD 更新定义",
    conditions: "γ > 0，精确梯度",
    variables: "γ：学习率；gₖ：当前梯度",
  },
  {
    formula: String.raw`g_k^\top(-\gamma g_k)=-\gamma\|g_k\|_2^2,\quad\|-\gamma g_k\|_2^2=\gamma^2\|g_k\|_2^2`,
    explanation:
      "用向量与自身的内积等于欧氏范数平方；范数平方把标量系数变为 γ²。",
    rule: "内积与欧氏范数定义",
    conditions: "欧氏几何；γ 为实数",
    variables: "‖gₖ‖₂：梯度长度",
  },
  {
    formula: String.raw`f(x_{k+1})\le f(x_k)-\left(\gamma-\frac{L\gamma^2}{2}\right)\|g_k\|_2^2`,
    explanation:
      "合并线性下降项和二次曲率代价。0 < γ < 2/L 时系数为正，非驻点得到严格下降。",
    rule: "代入、合并同类项",
    conditions: "L-光滑；GD 精确梯度",
    variables: "γ − Lγ²/2：保证下降的系数",
  },
  {
    formula: String.raw`0<\gamma\le\frac1L\Rightarrow\gamma-\frac{L\gamma^2}{2}\ge\frac\gamma2\Rightarrow f(x_{k+1})\le f(x_k)-\frac\gamma2\|g_k\|_2^2`,
    explanation:
      "γL ≤ 1 使曲率代价最多消耗一半下降量。这是单步下降界；一般非凸目标还需函数有下界等条件才能推出驻点速率，不能据此宣称线性收敛。",
    rule: "步长条件 γL ≤ 1",
    conditions: "0 < γ ≤ 1/L；线性收敛另需额外结构",
    variables: "下降界只约束这一步的函数值",
  },
];
export const gdQuestions = [
  {
    question: "在本实验 SPD 二次目标上，学习率越过 2/L 后会怎样？",
    options: [
      "所有方向都一定更快",
      "最大曲率方向有非零初始分量时会发散",
      "总能得到线性收敛",
    ],
    answer: 1,
    explanation:
      "该特征方向误差乘以 1−γL。γ>2/L 时其绝对值大于 1；若初值恰好没有该方向分量，则不能断言这条特殊轨迹发散。",
  },
  {
    question: "狭长等高线中，GD 为什么可能左右摆动？",
    options: [
      "各方向曲率不同，共同步长可能使陡方向反复过冲",
      "梯度计算一定有误",
      "因为所有椭圆都非凸",
    ],
    answer: 0,
    explanation:
      "当 1/L < γ < 2/L 时，最大曲率方向的因子为负且绝对值小于 1，表现为衰减振荡；较平缓方向仍可能移动很慢。",
  },
  {
    question: "从 L-光滑的单步下降界能直接推出什么？",
    options: [
      "任意非凸目标全局线性收敛",
      "0<γ≤1/L 时精确 GD 函数值不增",
      "任意步长都稳定",
    ],
    answer: 1,
    explanation:
      "下降不等于全局最优或线性收敛。驻点求和界还需函数有下界；强凸收缩需要强凸等条件。",
  },
];
