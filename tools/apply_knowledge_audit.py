from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def replace(path, old, new):
    p = ROOT / path
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'{path}: expected one exact replacement, found {count}: {old[:90]!r}')
    p.write_text(text.replace(old, new))

def section(path, start, end, new):
    p = ROOT / path
    text = p.read_text()
    if text.count(start) != 1 or text.count(end) != 1:
        raise RuntimeError(f'{path}: ambiguous section boundaries')
    a, b = text.index(start), text.index(end)
    if b < a:
        raise RuntimeError(f'{path}: reversed section boundaries')
    p.write_text(text[:a] + new + text[b:])

def write(path, content):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)

replace('notes/Lecture-05_Gradient-Descent.md',
        '一旦训练目标变成数据集上的平均，梯度只能抽样估计：',
        '当训练目标是大规模数据集上的平均损失时，完整梯度仍可通过遍历全部样本计算；由于成本较高，实践中常用随机样本或小批量梯度进行估计。有限和的梯度等于各项梯度的平均，而不是在数学上只能抽样：')

replace('notes/Lecture-07_Transformer-and-LLM-I.md',
        '1. **位置编码**：普通双向 self-attention 对 token 排列具有等变性，需要显式位置信号来表达“谁在谁之前”。因果掩码也会限制信息流向，但它只定义允许关注的区域，不能替代位置编码；两者提供的信息不同。可用可学习位置向量、正弦编码或 RoPE 等方法。',
        '1. **位置编码与因果掩码**：不含位置信号且使用全连接双向注意力时，self-attention 对 token 排列具有等变性，不能凭该结构区分绝对顺序。固定的因果掩码则引入非对称可见关系，不能直接套用前一种情形的结论。位置编码显式提供位置信息，因果掩码限制信息流向，二者机制不同；但不能断言所有因果 Transformer 都必须使用显式位置编码。已有研究发现不带显式位置编码的因果语言模型也能学习隐式位置信息，见 [Haviv 等（2022）](https://aclanthology.org/2022.findings-emnlp.99/)。这不保证任意模型删除位置编码后性能不变，也不保证长度外推能力相同。可学习位置向量、正弦编码或 RoPE 的选择需结合架构、任务和长度泛化要求。')
replace('notes/Lecture-07_Transformer-and-LLM-I.md',
        '完整的 self-attention 因 softmax 和输入投影而是输入 $X$ 的非线性函数。',
        '输入投影本身是线性的，但 $QK^\\top$ 中输入相关因子的乘积以及 softmax，使完整 self-attention 通常成为输入 $X$ 的非线性函数。')
replace('notes/Lecture-07_Transformer-and-LLM-I.md',
        '2. 为什么 self-attention 需要位置编码？',
        '2. 无位置信号的全连接双向 self-attention 有什么置换对称性？为什么不能据此断言所有因果 Transformer 都必须使用显式位置编码？')

replace('notes/Lecture-10_Subspace-Optimizers.md',
        '若分布偏置、缩放错误或秩过低，投影仍可能有很大偏差。',
        '分布不满足各向同性或缩放不正确时，估计可能有偏；在正确缩放且满足相应随机模型时，降低秩通常增加方差和单次估计误差，而不必引入统计偏差。')
replace('notes/Lecture-10_Subspace-Optimizers.md',
        '在无偏缩放、各向同性投影的比较中，维数降低通常增加方差；不能把这一趋势当作任意分布或未缩放估计器的统一方差公式。必须区分“低秩状态节省显存”和“无偏随机梯度”，两者是不同层次。',
        r'''例如，对固定 $g\in\mathbb R^d$，设随机正交基 $P\in\mathbb R^{d\times r}$ 满足 $P^\top P=I_r$、 $\mathbb E[PP^\top]=(r/d)I_d$，其中 $1\le r\le d$。定义

$$
\widehat g=\frac dr PP^\top g.
$$

则

$$
\mathbb E[\widehat g]=g,\qquad
\mathbb E\Vert\widehat g-g\Vert_2^2
=\left(\frac dr-1\right)\Vert g\Vert_2^2.
$$

令 $Q=PP^\top$，利用 $Q^\top=Q$、 $Q^2=Q$ 可得 $\mathbb E\Vert\widehat g\Vert_2^2=(d/r)\Vert g\Vert_2^2$，再展开误差平方即得上式。若 $g$ 也是随机量，需要对给定 $g$ 的条件分布满足相应各向同性条件，不能把与 $g$ 相关的任意选基直接代入。

这说明即使 $r=1$，正确缩放的估计也可保持无偏，但方差可能很大。上述恒等式只针对该线性投影估计器；经过 Adam 等非线性变换后的更新不因此自动无偏。必须区分“低秩状态节省显存”和“无偏随机梯度”，两者是不同层次。''')

replace('notes/Lecture-09_Advanced-Optimizers.md',
        '- attention 输出与 MLP：按神经元/行成块；\n- embedding 与输出层：按 token 或列成块。',
        '- V 投影、attention 输出与 MLP：按输出神经元成块；\n- embedding 与输出层：按 token 成块。\n\n本段分组参照 [Adam-mini 原论文 arXiv v7，第 2.3 节](https://arxiv.org/html/2406.16793v7)，而非将不同版本的策略混为一谈；具体对应矩阵的行还是列取决于权重布局，实际实现应核对使用的版本。')

replace('website/src/labs/LinearLab.tsx',
        '左投影先把 3 维行空间压缩成 r 维。P 的列正交保证 PPᵀ 是正交投影。',
        'G 的每个列向量属于三维环境空间 R³；左乘 Pᵀ 得到子空间中的 r 维坐标，再左乘 P 回映。G 的行向量属于 R²，列空间维数至多为 2；P 的列正交保证 PPᵀ 是正交投影。')

write('website/src/engines/attention-demo.ts', r'''import { attention, multiply, type Mat } from "./matrix";
import { seeded } from "./optimization";

/** Fixed model weights and a prefix-stable input stream are independent controls. */
export function createAttentionDemo(
  tokens: number,
  weightSeed: number,
  causal: boolean,
  inputSeed = 42,
) {
  if (!Number.isInteger(tokens) || tokens < 1)
    throw Error("Token 数量必须是正整数");
  const weightRng = seeded(weightSeed);
  const inputRng = seeded(inputSeed);
  const matrix = (rows: number, cols: number, rng: () => number): Mat =>
    Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => 2 * rng() - 1),
    );
  const X = matrix(tokens, 4, inputRng);
  const heads = Array.from({ length: 2 }, () => {
    const WQ = matrix(4, 2, weightRng);
    const WK = matrix(4, 2, weightRng);
    const WV = matrix(4, 2, weightRng);
    const Q = multiply(X, WQ);
    const K = multiply(X, WK);
    const V = multiply(X, WV);
    return { WQ, WK, WV, Q, K, V, ...attention(Q, K, V, causal) };
  });
  const concat = heads[0].output.map((row, i) => [...row, ...heads[1].output[i]]);
  const WO = matrix(4, 4, weightRng);
  return { X, heads, concat, WO, output: multiply(concat, WO) };
}
''')
section('website/src/labs/AttentionLab.tsx',
        'import {\n  attention,', 'import { MatrixView }',
        'import { onlineAttention } from "../engines/matrix";\nimport { createAttentionDemo } from "../engines/attention-demo";\n')
replace('website/src/labs/AttentionLab.tsx',
        'const [seed, setSeed] = useState(42),',
        'const [seed, setSeed] = useState(42),\n    [inputSeed, setInputSeed] = useState(42),')
section('website/src/labs/AttentionLab.tsx',
        '  const d = useMemo(() => {', '  const row = Math.min(query, tokens - 1),',
        '  const d = useMemo(\n    () => createAttentionDemo(tokens, seed, causal, inputSeed),\n    [seed, inputSeed, tokens, causal],\n  );\n')
replace('website/src/labs/AttentionLab.tsx',
        '            <label className="check">',
        '''            <Slider
              label="输入序列种子"
              min={1}
              max={100}
              value={inputSeed}
              onChange={(n) => change(() => setInputSeed(n))}
            />
            <label className="check">''')
replace('website/src/labs/AttentionLab.tsx',
        '确定种子直接映射到实际矩阵；播放仅改变显示阶段，不改变计算结果。双头宽度各',
        '权重与输入使用独立随机流；只改变 token 数时，权重与已有输入前缀保持不变。因果模式下，追加未来 token 不改变已有位置输出；双向注意力则无此保证。播放仅改变显示阶段。双头宽度各')
replace('website/src/labs/AttentionLab.tsx',
        '<MatrixView name="最终输出" data={d.output} />',
        '<div data-testid="attention-output" data-output={JSON.stringify(d.output)}><MatrixView name="最终输出" data={d.output} /></div>')

write('website/src/engines/zeroth.ts', r'''import { normal, seeded, type Vec } from "./optimization";
export const objective = (x: Vec) =>
  x[0] ** 4 / 4 + x[0] ** 2 / 2 + 2 * x[1] ** 2;
export const trueGradient = (x: Vec): Vec => [x[0] ** 3 + x[0], 4 * x[1]];
export type Estimator =
  "坐标前向" | "坐标中心" | "球面单侧" | "球面双侧" | "MeZO 高斯双侧";
export interface QueryPair {
  coordinate: number | null;
  direction: Vec;
  plus: Vec;
  minus: Vec;
  fPlus: number;
  fMinus: number;
  denominator: number;
  multiplier: number;
  coefficient: number;
  contribution: Vec;
}
export function estimate(
  x: Vec,
  tau: number,
  method: Estimator,
  seed: number,
  noise = 0,
) {
  if (!(Number.isFinite(tau) && tau > 0)) throw Error("τ 必须有限且大于零");
  if (!(Number.isFinite(noise) && noise >= 0) || !x.every(Number.isFinite))
    throw Error("查询点必须有限，噪声标准差必须有限且非负");
  const rng = seeded(seed);
  let queries = 0;
  const query = (y: Vec) => {
    queries++;
    return objective(y) + noise * normal(rng);
  };
  const details: QueryPair[] = [];
  const record = (
    direction: Vec,
    coordinate: number | null,
    centered: boolean,
    multiplier: number,
    base?: number,
  ) => {
    const plus: Vec = [x[0] + tau * direction[0], x[1] + tau * direction[1]];
    const minus: Vec = centered
      ? [x[0] - tau * direction[0], x[1] - tau * direction[1]]
      : [...x];
    const fPlus = query(plus);
    const fMinus = base === undefined ? query(minus) : base;
    const denominator = centered ? 2 * tau : tau;
    const coefficient = (multiplier * (fPlus - fMinus)) / denominator;
    const contribution: Vec = [coefficient * direction[0], coefficient * direction[1]];
    details.push({ coordinate, direction, plus, minus, fPlus, fMinus,
      denominator, multiplier, coefficient, contribution });
  };
  if (method.startsWith("坐标")) {
    const centered = method === "坐标中心";
    // The forward estimator shares exactly one noisy baseline across coordinates.
    const base = centered ? undefined : query(x);
    record([1, 0], 0, centered, 1, base);
    record([0, 1], 1, centered, 1, base);
  } else {
    let u: Vec;
    if (method === "MeZO 高斯双侧") u = [normal(rng), normal(rng)];
    else {
      const angle = rng() * 2 * Math.PI;
      u = [Math.cos(angle), Math.sin(angle)];
    }
    record(u, null, method !== "球面单侧", method === "MeZO 高斯双侧" ? 1 : 2);
  }
  const g = details.reduce<Vec>(
    (sum, pair) => [sum[0] + pair.contribution[0], sum[1] + pair.contribution[1]],
    [0, 0],
  );
  const truth = trueGradient(x);
  const last = details[details.length - 1];
  return {
    g, details,
    // Legacy summary fields always describe the same final query pair.
    u: last.direction, plus: last.plus, minus: last.minus,
    fp: last.fPlus, fm: last.fMinus, queries, truth,
    error: Math.hypot(g[0] - truth[0], g[1] - truth[1]),
  };
}
''')
replace('website/src/labs/ZerothLab.tsx',
        '[x, setX] = useState(1);',
        '[x, setX] = useState(1),\n    [coordinate, setCoordinate] = useState(0);')
replace('website/src/labs/ZerothLab.tsx',
        'd = estimate([x, 0.7], tau, method, seed, noise);',
        'd = estimate([x, 0.7], tau, method, seed, noise),\n    pair = d.details[Math.min(coordinate, d.details.length - 1)];')
replace('website/src/labs/ZerothLab.tsx',
        'onChange={(e) => setMethod(e.target.value as Estimator)}',
        'onChange={(e) => { setMethod(e.target.value as Estimator); setCoordinate(0); }}')
replace('website/src/labs/ZerothLab.tsx',
        '<h2>本次查询与方向</h2>',
        '''<h2>本次查询与方向</h2>
          {d.details.length > 1 && (
            <label>查看坐标查询
              <select aria-label="查看坐标查询" value={coordinate}
                onChange={(e) => setCoordinate(Number(e.target.value))}>
                <option value={0}>坐标 x₁</option>
                <option value={1}>坐标 x₂</option>
              </select>
            </label>
          )}
          <p data-testid="query-pair" data-direction={JSON.stringify(pair.direction)}
            data-plus={JSON.stringify(pair.plus)} data-minus={JSON.stringify(pair.minus)}>
            {pair.coordinate === null ? "随机方向 u" : `当前坐标方向 e${pair.coordinate + 1}`}：{vector(pair.direction)}
          </p>''')
for old, new in [('vector(d.plus)', 'vector(pair.plus)'), ('vector(d.minus)', 'vector(pair.minus)'), ('vector([d.fp, d.fm])', 'vector([pair.fPlus, pair.fMinus])'), ('fmt(d.fp - d.fm)', 'fmt(pair.fPlus - pair.fMinus)')]:
    replace('website/src/labs/ZerothLab.tsx', old, new)
replace('website/src/labs/ZerothLab.tsx',
        '["随机方向 u", vector(d.u)],',
        '["差分分母", fmt(pair.denominator)],\n              ["缩放因子", fmt(pair.multiplier)],\n              ["差分系数", fmt(pair.coefficient)],\n              ["本组方向贡献", vector(pair.contribution)],')
replace('website/src/labs/ZerothLab.tsx',
        '坐标法显示最后一个坐标的扰动查询。噪声来自每次查询独立生成的高斯值，另含',
        '坐标法可逐项查看查询，完整梯度由各坐标贡献相加。前向差分复用同一次基准查询；各次实际查询使用独立高斯噪声，因此共享基准会使坐标估计误差相关。另含')
replace('website/src/labs/ZerothLab.tsx', 'plus={d.plus}', 'plus={pair.plus}')
replace('website/src/labs/ZerothLab.tsx', 'minus={d.minus}', 'minus={pair.minus}')

replace('website/src/engines/distributed.ts',
        'export function distributed(',
        '''export interface DistributedOptions {
  curvatureSpread?: number;
  noiseStd?: number;
}
export function localGradient(x: Vec, target: Vec, curvature: number): Vec {
  return [curvature * (x[0] - target[0]), curvature * (x[1] - target[1])];
}
export function distributed(''')
replace('website/src/engines/distributed.ts',
        '  heterogeneity: number,', '  targetSpread: number,')
replace('website/src/engines/distributed.ts',
        '  steps = 100,\n) {\n  const rng = seeded(seed);',
        '''  steps = 100,
  options: DistributedOptions = {},
) {
  const curvatureSpread = options.curvatureSpread ?? 1;
  const noiseStd = options.noiseStd ?? 0.15;
  if (![targetSpread, curvatureSpread, noiseStd].every((x) => Number.isFinite(x) && x >= 0))
    throw Error("目标分散、曲率差异与噪声标准差必须有限且非负");
  const rng = seeded(seed);''')
replace('website/src/engines/distributed.ts', 'heterogeneity * Math.cos', 'targetSpread * Math.cos')
replace('website/src/engines/distributed.ts', 'heterogeneity * Math.sin', 'targetSpread * Math.sin')
replace('website/src/engines/distributed.ts', 'curvature: 1 + i / nodes,', 'curvature: 1 + curvatureSpread * i / nodes,')
section('website/src/engines/distributed.ts',
        '    const gradients = workers.map(', '    const comm = method !== "Local SGD"',
        '''    const gradients = workers.map((w): Vec => {
      const g = localGradient(w.x, w.target, w.curvature);
      return [g[0] + noiseStd * normal(rng), g[1] + noiseStd * normal(rng)];
    });
''')
replace('website/src/labs/DistributedLab.tsx',
        '[heterogeneity, setHeterogeneity] = useState(2),',
        '[targetSpread, setTargetSpread] = useState(2),\n    [curvatureSpread, setCurvatureSpread] = useState(1),\n    [noiseStd, setNoiseStd] = useState(0.15),')
replace('website/src/labs/DistributedLab.tsx', '          heterogeneity,', '          targetSpread,')
replace('website/src/labs/DistributedLab.tsx',
        '          seed,\n        ),', '          seed,\n          100,\n          { curvatureSpread, noiseStd },\n        ),')
replace('website/src/labs/DistributedLab.tsx',
        '[nodes, method, interval, heterogeneity, rate, compression, seed],',
        '[nodes, method, interval, targetSpread, curvatureSpread, noiseStd, rate, compression, seed],')
replace('website/src/labs/DistributedLab.tsx',
        '每个节点拥有不同曲率与最优点。真实执行本地更新、聚合与带索引的压缩通信。',
        '分别控制本地最优点、曲率差异与采样噪声。真实执行本地更新、聚合与带索引的压缩通信。')
replace('website/src/labs/DistributedLab.tsx', 'label="数据异质性"', 'label="本地最优点分散程度"')
replace('website/src/labs/DistributedLab.tsx', 'value={heterogeneity}', 'value={targetSpread}')
replace('website/src/labs/DistributedLab.tsx', 'setHeterogeneity(n)', 'setTargetSpread(n)')
replace('website/src/labs/DistributedLab.tsx',
        '          <Slider\n            label="节点学习率"',
        '''          <Slider label="曲率差异" min={0} max={2} step={0.1}
            value={curvatureSpread} onChange={(n) => change(() => setCurvatureSpread(n))} />
          <Slider label="每坐标梯度噪声标准差" min={0} max={0.5} step={0.01}
            value={noiseStd} onChange={(n) => change(() => setNoiseStd(n))} />
          <button onClick={() => change(() => {
            setTargetSpread(0); setCurvatureSpread(0); setNoiseStd(0);
          })}>同质无噪声对照</button>
          <p className="notice">
            同质目标要求最优点分散与曲率差异都为零；共同最优点不等于相同目标。
            同质目标下独立采样噪声仍可能产生不同随机梯度，对照按钮同时关闭噪声。
          </p>
          <Slider
            label="节点学习率"''')
replace('website/src/labs/DistributedLab.tsx',
        '<tr key={i}>',
        '<tr key={i} data-testid={`worker-${i}`} data-gradient={JSON.stringify(w.gradient)} data-parameters={JSON.stringify(w.x)}>')
replace('website/src/labs/DistributedLab.tsx',
        '在共同参数上平均（可压缩的）随机梯度。独立噪声每坐标标准差 0.15。',
        '在共同参数上平均（可压缩的）随机梯度。曲率 aᵢ = 1 + δ·i/n，i 从 0 起，δ = {curvatureSpread}；独立噪声每坐标标准差为 {noiseStd}。')
replace('website/src/content/modules.ts',
        '二维异质强凸二次目标；等权全参与；独立噪声；DSGD 使用先更新后环混合。',
        '二维强凸二次目标；最优点分散、曲率差异和独立噪声分别可控；等权全参与；DSGD 先更新后环混合。')
replace('website/src/content/modules.ts',
        '对比 Local K=1 与 PSGD，再增大 K 和异质性观察共识误差。',
        '先用同质无噪声对照检查节点一致性，再分别增大最优点分散、曲率差异和噪声；比较 Local K=1 与 PSGD。')

write('website/src/tests/knowledge-regressions.test.ts', r'''import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { createAttentionDemo } from "../engines/attention-demo";
import { distributed, localGradient } from "../engines/distributed";
import { estimate, type Estimator } from "../engines/zeroth";

const weights = (d: ReturnType<typeof createAttentionDemo>) => ({
  heads: d.heads.map(({ WQ, WK, WV }) => ({ WQ, WK, WV })), WO: d.WO,
});
describe("知识与教学变量回归", () => {
  it("改变 token 数只追加输入，不更换模型；因果前缀输出保持不变", () => {
    for (const seed of [1, 42, 99]) {
      const short = createAttentionDemo(2, seed, true, 7);
      for (const count of [3, 4, 6]) {
        const long = createAttentionDemo(count, seed, true, 7);
        expect(weights(long)).toEqual(weights(short));
        expect(long.X.slice(0, 2)).toEqual(short.X);
        long.output.slice(0, 2).forEach((row, i) => row.forEach((v, j) =>
          expect(v).toBeCloseTo(short.output[i][j], 12)));
      }
    }
  });
  it("输入种子与权重种子互不改变对方控制的对象", () => {
    const baseline = createAttentionDemo(4, 42, true, 7);
    const input = createAttentionDemo(4, 42, true, 8);
    const model = createAttentionDemo(4, 43, true, 7);
    expect(weights(input)).toEqual(weights(baseline));
    expect(input.X).not.toEqual(baseline.X);
    expect(model.X).toEqual(baseline.X);
    expect(weights(model)).not.toEqual(weights(baseline));
    const bidirectional = createAttentionDemo(4, 42, false, 7);
    expect(weights(bidirectional)).toEqual(weights(baseline));
    expect(bidirectional.X).toEqual(baseline.X);
  });
  it("双向 attention 不冒用因果前缀不变性", () => {
    const short = createAttentionDemo(2, 42, false);
    const long = createAttentionDemo(6, 42, false);
    expect(weights(short)).toEqual(weights(long));
    expect(long.output[0]).not.toEqual(short.output[0]);
  });
  it("逐坐标记录的方向、查询点、差分系数与完整梯度相符", () => {
    const x: [number, number] = [1, 0.7];
    for (const method of ["坐标前向", "坐标中心"] as Estimator[]) {
      const d = estimate(x, 0.1, method, 42);
      expect(d.queries).toBe(method === "坐标前向" ? 3 : 4);
      expect(d.details).toHaveLength(2);
      d.details.forEach((p, i) => {
        expect(p.coordinate).toBe(i);
        expect(p.direction).toEqual(i === 0 ? [1, 0] : [0, 1]);
        for (let j = 0; j < 2; j++) {
          expect(p.plus[j] - x[j]).toBeCloseTo(0.1 * p.direction[j], 12);
          expect(x[j] - p.minus[j]).toBeCloseTo(
            method === "坐标中心" ? 0.1 * p.direction[j] : 0, 12);
        }
        expect(p.coefficient).toBeCloseTo((p.fPlus - p.fMinus) / p.denominator, 12);
        expect(p.contribution[i]).toBeCloseTo(d.g[i], 12);
      });
      expect(d.u).toEqual([0, 1]);
      expect(d.plus).toEqual(d.details[1].plus);
      expect(d.fp).toBe(d.details[1].fPlus);
    }
    const center = estimate(x, 0.1, "坐标中心", 42);
    expect(center.g[0]).toBeCloseTo(2.01, 12);
    expect(center.g[1]).toBeCloseTo(2.8, 12);
  });
  it("坐标前向共用一次有噪声基准，随机方向记录保持正确缩放", () => {
    const forward = estimate([1, 0.7], 0.1, "坐标前向", 42, 0.01);
    expect(forward.queries).toBe(3);
    expect(forward.details[0].fMinus).toBe(forward.details[1].fMinus);
    for (const method of ["球面单侧", "球面双侧", "MeZO 高斯双侧"] as Estimator[]) {
      const d = estimate([1, 0.7], 0.1, method, 42);
      const p = d.details[0];
      expect(d.queries).toBe(2);
      expect(p.coordinate).toBeNull();
      expect(p.multiplier).toBe(method === "MeZO 高斯双侧" ? 1 : 2);
      expect(d.g).toEqual(p.contribution);
      p.direction.forEach((u, j) => expect(p.plus[j] - [1, 0.7][j]).toBeCloseTo(0.1 * u, 12));
    }
  });
  it("同质无噪声目标在共同点的梯度一致，Local 多步不产生分歧", () => {
    const run = distributed(4, "Local SGD", 5, 0, 0.1, "无压缩", 42, 12,
      { curvatureSpread: 0, noiseStd: 0 });
    run.forEach((s) => {
      s.workers.forEach((w) => {
        expect(w.curvature).toBe(1);
        expect(w.target.every((v) => v === 0)).toBe(true);
        w.x.forEach((v, j) => expect(v).toBeCloseTo(s.workers[0].x[j], 12));
        w.gradient.forEach((v, j) => expect(v).toBeCloseTo(s.workers[0].gradient[j], 12));
      });
      expect(s.consensus).toBeCloseTo(0, 12);
    });
  });
  it("共同最优点仍可有曲率异质性，目标同质与采样噪声分开", () => {
    const curved = distributed(2, "Local SGD", 5, 0, 0.1, "无压缩", 42, 1,
      { curvatureSpread: 1, noiseStd: 0 });
    expect(curved[1].workers[0].gradient).toEqual([-3, 2]);
    expect(curved[1].workers[1].gradient).toEqual([-4.5, 3]);
    expect(localGradient([-3, 2], [0, 0], 1)).toEqual([-3, 2]);
    const noisy = distributed(4, "Local SGD", 5, 0, 0.1, "无压缩", 42, 1,
      { curvatureSpread: 0, noiseStd: 0.15 });
    expect(noisy[1].consensus).toBeGreaterThan(0);
  });
  it("显式默认配置保留原来的分布式数值模型", () => {
    expect(distributed(4, "PSGD", 1, 2, 0.1, "无压缩", 42, 5)).toEqual(
      distributed(4, "PSGD", 1, 2, 0.1, "无压缩", 42, 5,
        { curvatureSpread: 1, noiseStd: 0.15 }));
  });
  it("精确枚举随机坐标子空间：无偏不等于低方差", () => {
    const g = [1, -2, 3, -4];
    const norm2 = g.reduce((s, x) => s + x * x, 0);
    for (const rank of [1, 2, 3, 4]) {
      const subsets = Array.from({ length: 16 }, (_, mask) =>
        [0, 1, 2, 3].filter((i) => mask & (1 << i))).filter((s) => s.length === rank);
      const estimates = subsets.map((s) => g.map((v, i) => s.includes(i) ? 4 * v / rank : 0));
      g.forEach((v, i) => expect(estimates.reduce((s, e) => s + e[i], 0) / estimates.length).toBeCloseTo(v, 12));
      const mse = estimates.reduce((s, e) => s + e.reduce((a, v, i) => a + (v - g[i]) ** 2, 0), 0) / estimates.length;
      expect(mse).toBeCloseTo((4 / rank - 1) * norm2, 12);
    }
  });
  it("笔记与解释不再使用本轮确认的错误断言", () => {
    const note = (file: string) => readFileSync(resolve("../notes", file), "utf8");
    expect(note("Lecture-05_Gradient-Descent.md")).not.toContain("梯度只能抽样估计");
    expect(note("Lecture-07_Transformer-and-LLM-I.md")).toContain("2022.findings-emnlp.99");
    expect(note("Lecture-10_Subspace-Optimizers.md")).toContain("不必引入统计偏差");
    expect(readFileSync(resolve("src/labs/LinearLab.tsx"), "utf8")).not.toContain("3 维行空间");
  });
});
''')

write('website/e2e/knowledge.spec.ts', r'''import { test, expect } from "@playwright/test";

test("追加未来 token 保持模型和因果前缀输出", async ({ page }) => {
  await page.goto("./#attention");
  await page.getByRole("slider", { name: "Token 数量 N" }).fill("2");
  await page.getByRole("slider", { name: "迭代位置" }).fill("5");
  const result = page.getByTestId("attention-output");
  const before: number[][] = JSON.parse((await result.getAttribute("data-output"))!);
  await page.getByRole("slider", { name: "Token 数量 N" }).fill("6");
  await page.getByRole("slider", { name: "迭代位置" }).fill("5");
  const after: number[][] = JSON.parse((await result.getAttribute("data-output"))!);
  expect(after).toHaveLength(6);
  before.forEach((row, i) => row.forEach((v, j) => expect(after[i][j]).toBeCloseTo(v, 12)));
  await expect(page.getByRole("slider", { name: "权重随机种子" })).toHaveValue("42");
  await expect(page.getByRole("slider", { name: "输入序列种子" })).toHaveValue("42");
});

test("零阶坐标查询方向与实际扰动匹配，随机模式不保留旧坐标", async ({ page }) => {
  await page.goto("./#zeroth");
  const pair = page.getByTestId("query-pair");
  await expect(pair).toHaveAttribute("data-direction", "[1,0]");
  await page.getByLabel("查看坐标查询").selectOption("1");
  await expect(pair).toHaveAttribute("data-direction", "[0,1]");
  const plus: number[] = JSON.parse((await pair.getAttribute("data-plus"))!);
  const minus: number[] = JSON.parse((await pair.getAttribute("data-minus"))!);
  expect(plus[0]).toBe(1);
  expect(minus[0]).toBe(1);
  expect(plus[1]).toBeCloseTo(0.71, 12);
  expect(minus[1]).toBeCloseTo(0.69, 12);
  await expect(pair).toContainText("当前坐标方向 e2");
  await page.getByLabel("零阶估计器").selectOption("球面双侧");
  await expect(page.getByLabel("查看坐标查询")).toHaveCount(0);
  await expect(pair).toContainText("随机方向 u");
  await page.getByLabel("零阶估计器").selectOption("坐标中心");
  await expect(pair).toHaveAttribute("data-direction", "[1,0]");
});

test("分布式同质无噪声预设消除目标差异，而非仅共享最优点", async ({ page }) => {
  await page.goto("./#distributed");
  await page.getByRole("button", { name: "同质无噪声对照" }).click();
  for (const name of ["本地最优点分散程度", "曲率差异", "每坐标梯度噪声标准差"])
    await expect(page.getByRole("slider", { name, exact: true })).toHaveValue("0");
  await page.getByRole("slider", { name: "迭代位置" }).fill("3");
  const first = page.getByTestId("worker-0");
  const parameters = await first.getAttribute("data-parameters");
  const gradient = await first.getAttribute("data-gradient");
  for (let i = 1; i < 4; i++) {
    await expect(page.getByTestId(`worker-${i}`)).toHaveAttribute("data-parameters", parameters!);
    await expect(page.getByTestId(`worker-${i}`)).toHaveAttribute("data-gradient", gradient!);
  }
  await page.getByRole("slider", { name: "曲率差异", exact: true }).fill("1");
  await expect(page.getByTestId("iteration")).toHaveText("0");
  await page.getByRole("button", { name: "前进一步" }).click();
  expect(await page.getByTestId("worker-0").getAttribute("data-gradient"))
    .not.toBe(await page.getByTestId("worker-1").getAttribute("data-gradient"));
});
''')

p = ROOT / 'REVISION_NOTES.md'
p.write_text(p.read_text() + '''

## 知识性与实验一致性修订（2026-11-01）

- 第 5 讲：完整有限和梯度可以计算，抽样是计算代价选择，不是数学上的唯一途径。
- 第 7 讲：区分无位置双向注意力的置换等变性与因果掩码的顺序信息，补充 Haviv 等（2022）的原始研究；明确输入投影本身线性。
- 第 9 讲：注明 Adam-mini 分组参照 arXiv v7，区分输出神经元与矩阵行列布局。
- 第 10 讲：分开统计偏差与方差，补充各向同性随机投影的无偏与均方误差推导。
- 网站：修正列向量环境空间的说明；零阶方法记录每一组查询与方向，并允许切换坐标；Attention 使用独立的权重与输入随机流；分布式模型分别控制最优点分散、曲率和噪声，增加同质无噪声预设。
- 回归：新增因果前缀不变性、查询元数据一致性、同质目标及投影均方误差测试；验收以相应提交的实际 CI 记录为准，不将新增测试等同于已通过。
''')
p = ROOT / 'website/VERIFICATION.md'
p.write_text(p.read_text() + '''

## 知识与实验变量一致性（2026-11-01）

本轮仅修复知识审查确认的问题，不升级整讲核验等级，不声称逐页重校全部 PDF。

- Attention：权重种子与输入种子分离；追加 token 保留权重和输入前缀。因果模式测试已有前缀输出不变；双向模式不承诺这一性质。
- 零阶：逐坐标查询记录包含真实方向、正负/基准点、函数值、分母、维度因子和方向贡献。前向法共享一次基准查询，故噪声查询独立不意味着各坐标估计误差独立。
- 分布式：本地最优点分散为零不等于目标同质；曲率差异独立控制。对照预设同时消除目标差异和采样噪声，用于检查共同初始化下的节点一致性。
- 低秩：左投影作用于列向量所在环境空间，不能称作三维行空间。随机坐标子空间的精确枚举独立核算无偏性与均方误差，未推广为完整 GaLore/Adam 更新无偏。
- 新增测试位于 `src/tests/knowledge-regressions.test.ts` 与 `e2e/knowledge.spec.ts`。与既有全部测试一起执行；结果以当前候选提交及发布提交对应的 CI 日志为准。
''')
print('Applied knowledge-audit fixes with exact-match guards.')
