# 第 5 讲：梯度下降、动量与加速

> 配套文件：`materials/Lecture-05_Gradient-Descent/01_convexity-gradient-descent_slides.pdf`、`02_convexity-gradient-descent_notes.pdf`、`03_accelerated-gradient-descent_slides.pdf`、`04_accelerated-gradient-descent_notes.pdf`

## 学习目标

掌握凸集、凸函数、强凸与 \(L\)-光滑的定义和几何含义，会推导并解释 GD 在非凸、凸、强凸三类问题上的收敛率；理解 Polyak、Nesterov 和 Anderson 加速的差别与边界。

## 1. 凸性的四条等价线索

**核心掌握。** 集合 \(\mathcal C\) 凸，指任意 \(x,y\in\mathcal C\) 的连线仍在集合内：

\[
\theta x+(1-\theta)y\in\mathcal C,\qquad \forall\theta\in[0,1].
\]

函数 \(f\) 凸，是指其上方图（epigraph）为凸集，也可写成 Jensen 不等式

\[
f(\theta x+(1-\theta)y)
\le\theta f(x)+(1-\theta)f(y).
\]

若 \(f\) 可微，凸性等价于“一阶切线是全局下界”：

\[
f(y)\ge f(x)+\nabla f(x)^\top(y-x).
\]

若二阶可微，凸性对应 \(H_f(x)\succeq0\)。这条线性下界很重要：GD 每一步都沿着一个可靠的下界移动。

## 2. 凸性、平滑性与曲率

**核心掌握。** \(\mu\)-强凸通常写成

\[
f(y)\ge f(x)+\nabla f(x)^\top(y-x)+\frac\mu2\|y-x\|^2.
\]

\(L\)-光滑对应梯度的 Lipschitz 连续性：

\[
\|\nabla f(x)-\nabla f(y)\|\le L\|x-y\|.
\]

当 \(f\) 二阶可微时，强凸与光滑分别给出

\[
\mu I\preceq H_f(x)\preceq LI.
\]

因此 \(L/\mu\) 是条件数。强凸保证最低曲率为正，因而有一个唯一最小值；光滑保证函数不会被过大的步长直接推飞。

## 3. 梯度下降与驻点

**核心掌握。** 无约束问题为 \(\min_x f(x)\)，GD 更新

\[
x_{k+1}=x_k-\gamma\nabla f(x_k).
\]

驻点满足 \(\nabla f(x^\star)=0\)。它可能是全局最小、局部最小、鞍点或非严格平地；仅凭梯度为零不能判断性质，还要看二阶曲率或邻域函数值。

SGD、Adam 等后续算法仍然以“寻找好的驻点或足够低的函数值”为基本目标，见 [[Lecture-06_Stochastic-Gradient-Descent]]。

## 4. 收敛分析的三条路线

**核心掌握结论，熟悉证明的第一步。** 由 \(L\)-光滑的下降引理，

\[
f(x_{k+1})
\le f(x_k)-\left(\gamma-\frac{L\gamma^2}{2}\right)
\|\nabla f(x_k)\|^2.
\]

当 \(0<\gamma\le1/L\) 时，右侧每一步下降并在 \(1/L\) 附近最优。

| 情形 | 常用步长 | 可保证的收敛 | 达到误差 \(\epsilon\) 的迭代量级 |
|---|---|---|---|
| 光滑非凸 | \(\gamma=1/L\) | \(\min_{k\le K}\|\nabla f(x_k)\|^2=O(1/K)\) | \(O(1/\epsilon^2)\) 达到梯度范数 \(\epsilon\) |
| 光滑凸 | \(\gamma=1/L\) | \(f(x_K)-f^\star=O(LD^2/K)\) | \(O(LD^2/\epsilon)\) |
| \(L\)-光滑、\(\mu\)-强凸 | \(\gamma=2/(L+\mu)\) | 函数或距离按 \((1-\mu/L)^K\) 收缩 | \(O((L/\mu)\log(1/\epsilon))\) |

这里 \(D=\|x_0-x^\star\|\)。不要只记“\(O(1/K)\)”：要标明它约束的是梯度范数平方还是最优间隙；若要求梯度范数本身为 \(\epsilon\)，非凸复杂度是 \(O(1/\epsilon^2)\)。

收敛证明通常先把 GD 代入下降不等式，再对 \(k\) 求和得到 telescoping；凸问题再使用凸性的线性下界。完整推导按讲义习题掌握即可，但必须知道步长和 \(L\) 的关系。

## 5. 病态问题与动量

**核心掌握更新式与直觉。** 病态二次问题的条件数很大，普通 GD 在两个方向用同一学习率，导致一个方向振荡、另一个方向几乎不动。

Polyak/heavy-ball 动量：

\[
x_{k+1}=x_k-\gamma\nabla f(x_k)+\beta(x_k-x_{k-1}).
\]

历史位移 \(x_k-x_{k-1}\) 起到“惯性”作用。等价的速度写法是

\[
v_{k+1}=\beta v_k-\gamma\nabla f(x_k),\qquad
x_{k+1}=x_k+v_{k+1}.
\]

在二次目标上可以显式分析特征方向上的二阶递推。适当选择 \(\beta\) 能加快收敛，但 heavy-ball 的加速结论主要适用于二次问题；一般光滑凸问题不能直接把“动量”当作无条件加速。

## 6. Nesterov 加速

**核心掌握。** Nesterov 的起点并非只累积当前位置的梯度，而是在外推点评估梯度：

\[
y_k=x_k+\beta_k(x_k-x_{k-1}),
\qquad
x_{k+1}=y_k-\gamma\nabla f(y_k).
\]

取 \(\gamma=1/L\)、合适的 \(\beta_k\) 后，光滑凸问题达到

\[
f(x_K)-f^\star=O\left(\frac{LD^2}{K^2}\right),
\]

强凸情形可达到 \((1-\sqrt{\mu/L})^K\) 型收缩。与 GD 相比，凸复杂度从 \(O(L/\epsilon)\) 降为 \(O(\sqrt{L/\epsilon})\)。一阶光滑凸优化的下界说明 NAG 在阶数意义下已最优。

**直观但不是“魔法”。** Nesterov 通过外推点补偿动量方向，减少了过冲；低阶平滑一阶算法中，想从 \(1/K\) 提升到 \(1/K^2\)，几乎需要这种“前瞻梯度”结构。

## 7. Anderson acceleration

**熟悉。** Anderson 方法用最近 \(m+1\) 个梯度或残差做仿射组合：

\[
x_{k+1}=\sum_{i=0}^{m}\alpha_{k,i}x_{k-i},
\qquad
\sum_i\alpha_{k,i}=1,
\]

并选系数使组合后的残差尽量小。它适合求解 \(F(x)=0\) 的固定点问题，实践上可显著加速病态问题。

**边界：** Anderson 不自动获得比 NAG 更好的最坏情形阶数；讲义用下界和实验强调其“实际有效、理论有限”。不要把实验加速误写成所有凸问题的严格优势。

## 8. 选择指南

- 光滑凸问题、计算梯度昂贵：NAG 是理论首选。
- 非凸神经网络：Nesterov/SGD 变体仍是工程主力，但经典最坏情形保证变弱。
- 希望保留二阶历史但目标非二次：Anderson 是实用工具，需做稳定性和阻尼。
- 仅凭“动量”标签不能判断收敛，必须同时看步长、曲率、噪声和是否确定性。

## 仅作了解

- 所有矩阵特征值计算和 heavy-ball 的完整参数最优化推导。
- Nesterov 下界证明中对抗性函数构造。
- Anderson 的完整局部收敛条件与不同记忆长度实现。

## 自测

1. 写出可微凸函数的一阶下界和 \(L\)-光滑的定义。
2. 为什么 GD 的非凸复杂度不能简单写成 \(O(1/\epsilon)\)？
3. Polyak 和 Nesterov 分别在哪一点使用历史信息？
4. 强凸 GD 的 \((1-\mu/L)^K\) 与条件数有什么关系？

一旦训练目标变成数据集上的平均，梯度只能抽样估计：[[Lecture-06_Stochastic-Gradient-Descent]]。

