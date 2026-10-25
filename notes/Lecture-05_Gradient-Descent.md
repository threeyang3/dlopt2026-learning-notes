# 第 5 讲：梯度下降、动量与加速

> **课程定位**：2026 年课程核心优化主线；内容综合 LLM 2025 与 DLOpt 2024 资料。
>
> 配套文件：`materials/Lecture-05_Gradient-Descent/01_convexity-gradient-descent_slides.pdf`、`02_convexity-gradient-descent_notes.pdf`、`03_accelerated-gradient-descent_slides.pdf`、`04_accelerated-gradient-descent_notes.pdf`

## 学习目标

掌握凸集、凸函数、强凸与 $L$-光滑的定义和几何含义，会推导并解释 GD 在非凸、凸、强凸三类问题上的收敛率；理解 Polyak、Nesterov 和 Anderson 加速的差别与边界。

## 1. 凸性的四条等价线索

**核心掌握。** 集合 $\mathcal C$ 凸，指任意 $x,y\in\mathcal C$ 的连线仍在集合内：

$$
\theta x+(1-\theta)y\in\mathcal C,\qquad \forall\theta\in[0,1].
$$

函数 $f$ 凸，是指其上方图（epigraph）为凸集，也可写成 Jensen 不等式

$$
f(\theta x+(1-\theta)y)
\le\theta f(x)+(1-\theta)f(y).
$$

若 $f$ 可微，凸性等价于“一阶切线是全局下界”：

$$
f(y)\ge f(x)+\nabla f(x)^\top(y-x).
$$

若二阶可微，凸性对应 $H_f(x)\succeq0$。这条线性下界很重要：GD 每一步都沿着一个可靠的下界移动。

## 2. 凸性、平滑性与曲率

**核心掌握。** $\mu$-强凸通常写成

$$
f(y)\ge f(x)+\nabla f(x)^\top(y-x)+\frac\mu2\|y-x\|^2.
$$

$L$-光滑对应梯度的 Lipschitz 连续性：

$$
\|\nabla f(x)-\nabla f(y)\|\le L\|x-y\|.
$$

当 $f$ 二阶可微时，强凸与光滑分别给出

$$
\mu I\preceq H_f(x)\preceq LI.
$$

因此 $L/\mu$ 是条件数。强凸保证最低曲率为正，因而有一个唯一最小值；光滑保证函数不会被过大的步长直接推飞。

## 3. 梯度下降与驻点

**核心掌握。** 无约束问题为 $\min_x f(x)$，GD 更新

$$
x_{k+1}=x_k-\gamma\nabla f(x_k).
$$

驻点满足 $\nabla f(x^\star)=0$。它可能是全局最小、局部最小、鞍点或非严格平地；仅凭梯度为零不能判断性质，还要看二阶曲率或邻域函数值。

SGD、Adam 等后续算法仍然以“寻找好的驻点或足够低的函数值”为基本目标，见 [[Lecture-06_Stochastic-Gradient-Descent]]。

## 4. 收敛分析的三条路线

**核心掌握结论，熟悉证明的第一步。** 由 $L$-光滑的下降引理，

$$
f(x_{k+1})
\le f(x_k)-\left(\gamma-\frac{L\gamma^2}{2}\right)
\|\nabla f(x_k)\|^2.
$$

当 $0<\gamma\le1/L$ 时，右侧每一步下降并在 $1/L$ 附近最优。

| 情形 | 常用步长 | 可保证的收敛 | 达到误差 $\epsilon$ 的迭代量级 |
|---|---|---|---|
| 光滑非凸 | $\gamma=1/L$ | $\min_{k\le K}\|\nabla f(x_k)\|^2=O(1/K)$ | $O(1/\epsilon^2)$ 达到梯度范数 $\epsilon$ |
| 光滑凸 | $\gamma=1/L$ | $f(x_K)-f^\star=O(LD^2/K)$ | $O(LD^2/\epsilon)$ |
| $L$-光滑、 $\mu$-强凸 | $\gamma=1/L$ | 距离按 $(1-\mu/L)^K$ 收缩 | $O((L/\mu)\log(1/\epsilon))$ |
| $L$-光滑、 $\mu$-强凸 | $\gamma=2/(L+\mu)$ | 距离按 $((L-\mu)/(L+\mu))^K$ 收缩 | $O((L/\mu)\log(1/\epsilon))$ |

这里 $D=\|x_0-x^\star\|$。不要只记“$O(1/K)$”：要标明它约束的是梯度范数平方还是最优间隙；若要求梯度范数本身为 $\epsilon$，非凸复杂度是 $O(1/\epsilon^2)$。

非凸速率的完整表述还需要 $f$ 有下界，即 $f^\star=\inf_x f(x)>-\infty$，并使用定理要求的步长。仅有 $L$-光滑和逐步下降并不足以单独推出统一的 $O(1/K)$ 速率。

对二次函数，距离收缩可直接由

$$
\|x_{k+1}-x^\star\|_2
\le
\max_{\lambda\in[\mu,L]}|1-\gamma\lambda|
\|x_k-x^\star\|_2
$$

得到。两种步长对应的收缩因子分别为 $1-\mu/L$ 与 $(L-\mu)/(L+\mu)$，不能交叉使用。函数值误差还需要结合强凸或光滑界，不能把所有量都写成同一个 $q^K$。

收敛证明通常先把 GD 代入下降不等式，再对 $k$ 求和得到 telescoping；凸问题再使用凸性的线性下界。完整推导按讲义习题掌握即可，但必须知道步长和 $L$ 的关系。

## 5. 病态问题与动量

**核心掌握更新式与直觉。** 病态二次问题的条件数很大，普通 GD 在两个方向用同一学习率，导致一个方向振荡、另一个方向几乎不动。

Polyak/heavy-ball 动量：

$$
x_{k+1}=x_k-\gamma\nabla f(x_k)+\beta(x_k-x_{k-1}).
$$

历史位移 $x_k-x_{k-1}$ 起到“惯性”作用。等价的速度写法是

$$
v_{k+1}=\beta v_k-\gamma\nabla f(x_k),\qquad
x_{k+1}=x_k+v_{k+1}.
$$

在二次目标上可以显式分析特征方向上的二阶递推。适当选择 $\beta$ 能加快收敛，但 heavy-ball 的加速结论主要适用于二次问题；一般光滑凸问题不能直接把“动量”当作无条件加速。

## 6. Nesterov 加速

**核心掌握。** Nesterov 的起点并非只累积当前位置的梯度，而是在外推点评估梯度：

$$
y_k=x_k+\beta_k(x_k-x_{k-1}),
\qquad
x_{k+1}=y_k-\gamma\nabla f(y_k).
$$

取 $\gamma=1/L$、合适的 $\beta_k$ 后，光滑凸问题达到

$$
f(x_K)-f^\star=O\left(\frac{LD^2}{K^2}\right),
$$

强凸情形可达到 $(1-\sqrt{\mu/L})^K$ 型收缩。与 GD 相比，凸复杂度从 $O(L/\epsilon)$ 降为 $O(\sqrt{L/\epsilon})$。一阶光滑凸优化的下界说明 NAG 在阶数意义下已最优。

**直观但不是“魔法”。** Nesterov 通过外推点补偿动量方向，减少了过冲；低阶平滑一阶算法中，想从 $1/K$ 提升到 $1/K^2$，几乎需要这种“前瞻梯度”结构。

## 7. Anderson acceleration

**熟悉。** Anderson 方法加速固定点迭代，而不是直接对历史参数作任意平均。对固定点方程

$$
x=G(x),
\qquad
r(x)=G(x)-x,
$$

先选仿射系数，使历史残差的组合尽量小：

$$
\alpha_k^\star
=\arg\min_{\sum_i\alpha_{k,i}=1}
\left\|\sum_i\alpha_{k,i}r(x_{k-i})\right\|_2^2.
$$

无阻尼的常见 Type-II 更新为

$$
x_{k+1}
=\sum_i\alpha_{k,i}^\star G(x_{k-i}).
$$

在梯度下降中常取 $G(x)=x-\gamma\nabla f(x)$，此时 $G$ 的固定点就是梯度零点。记忆长度 $m$ 决定可使用的历史，阻尼系数用于抑制系数振荡和数值不稳定。它适合求解 $x=G(x)$ 或等价地 $r(x)=0$ 的固定点问题，实践上可显著加速病态问题。

**边界：** Anderson 不自动获得比 NAG 更好的最坏情形阶数；讲义用下界和实验强调其“实际有效、理论有限”。不要把实验加速误写成所有凸问题的严格优势。

## 8. 选择指南

- 光滑凸问题、计算梯度昂贵：NAG 是理论首选。
- 非凸神经网络：Nesterov/SGD 变体仍是工程主力，但经典最坏情形保证变弱。
- 希望保留二阶历史但目标非二次：Anderson 是实用工具，需做稳定性和阻尼。
- 仅凭“动量”标签不能判断收敛，必须同时看步长、曲率、噪声和是否确定性。

## 常见误解

- 强凸收缩因子与步长必须配套： $\gamma=1/L$ 对应 $1-\mu/L$，不能与 $\gamma=2/(L+\mu)$ 的结果互换。
- Nesterov 的最优性针对一阶光滑凸问题的阶数，不表示它在所有非凸深度网络上必然优于 SGD 或 Adam。
- Anderson 不是对历史参数的简单加权平均；真正参与更新的是固定点映射 $G$ 的输出。
- 非凸 $O(1/K)$ 结论不能脱离函数有下界和规定步长单独使用。

## 仅作了解

- 所有矩阵特征值计算和 heavy-ball 的完整参数最优化推导。
- Nesterov 下界证明中对抗性函数构造。
- Anderson 的完整局部收敛条件与不同记忆长度实现。

## 自测

1. 写出可微凸函数的一阶下界和 $L$-光滑的定义。
2. 为什么 GD 的非凸复杂度不能简单写成 $O(1/\epsilon)$？
3. Polyak 和 Nesterov 分别在哪一点使用历史信息？
4. 强凸 GD 的 $(1-\mu/L)^K$ 与条件数有什么关系？

## 资料定位

| 主题 | 原始资料位置 |
|---|---|
| 凸集、凸函数、强凸与光滑 | slides PDF p. 3-10；notes PDF p. 1-3 |
| GD 更新与三类收敛率 | slides PDF p. 11-28；notes PDF p. 2-6 |
| Polyak、NAG、Anderson 与下界 | accelerated slides p. 3-34；notes PDF p. 1-11 |

一旦训练目标变成数据集上的平均，梯度只能抽样估计：[[Lecture-06_Stochastic-Gradient-Descent]]。

