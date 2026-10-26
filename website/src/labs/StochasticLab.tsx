import { useMemo, useState } from "react";
import {
  batchGradient,
  seeded,
  stochastic,
  type Quadratic,
  type StochasticState,
} from "../engines/optimization";
import {
  Slider,
  Playback,
  usePlayback,
  fmt,
  vector,
} from "../components/Controls";
import { Landscape, Curve } from "../components/Plots";
import { MathFormula } from "../components/Math";
import { Quiz } from "../components/Quiz";
import { Sources } from "../components/Sources";
export function StochasticLab() {
  const [batch, setBatch] = useState(8),
    [sigma, setSigma] = useState(2),
    [seed, setSeed] = useState(42),
    [rate, setRate] = useState(0.05),
    [algorithm, setAlgorithm] = useState<"SGD" | "Momentum SGD" | "Adam">(
      "SGD",
    ),
    [b1, setB1] = useState(0.9),
    [b2, setB2] = useState(0.999);
  const q = useMemo<Quadratic>(
    () => ({ mu: 1, L: 8, angle: 0.4, optimum: [0, 0] }),
    [],
  );
  const runs = useMemo(
    () =>
      Array.from({ length: 16 }, (_, i) =>
        stochastic(
          q,
          [-4, 3],
          algorithm,
          rate,
          batch,
          sigma,
          seed + i,
          100,
          b1,
          b2,
        ),
      ),
    [q, algorithm, rate, batch, sigma, seed, b1, b2],
  );
  const player = usePlayback(100),
    s = runs[0][player.index] as StochasticState;
  const mean = runs[0].map((s, k) => ({
    ...s,
    value: runs.reduce((a, r) => a + r[k].value, 0) / runs.length,
  }));
  const band = mean.map((_, k) => {
    const sorted = runs.map((r) => r[k].value).sort((a, b) => a - b);
    return { low: sorted[1], high: sorted[14] };
  });
  const empirical = useMemo(() => {
    const r = seeded(seed),
      samples = Array.from(
        { length: 2000 },
        () => batchGradient(q, [0, 0], batch, sigma, r).mean[0],
      ),
      m = samples.reduce((a, b) => a + b) / samples.length;
    return samples.reduce((a, b) => a + (b - m) ** 2, 0) / (samples.length - 1);
  }, [q, batch, sigma, seed]);
  function change(action: () => void) {
    player.reset();
    action();
  }
  return (
    <>
      <div className="lab-intro">
        <p>Lecture 04 / 06</p>
        <h1>随机梯度与 Adam</h1>
        <p>
          同一个二次目标，加入独立高斯梯度噪声。固定种子观察批平均、路径波动与历史状态。
        </p>
      </div>
      <div className="experiment">
        <div className="visual-panel">
          <Landscape
            q={q}
            series={[{ name: algorithm, states: runs[0] }]}
            index={player.index}
          />
          <Playback player={player} max={100} />
        </div>
        <aside className="control-panel">
          <h2>采样设置</h2>
          <label>
            算法
            <select
              aria-label="随机算法"
              value={algorithm}
              onChange={(e) =>
                change(() => setAlgorithm(e.target.value as typeof algorithm))
              }
            >
              {["SGD", "Momentum SGD", "Adam"].map((a) => (
                <option key={a}>{a}</option>
              ))}
            </select>
          </label>
          <Slider
            label="批量 B"
            min={1}
            max={64}
            value={batch}
            onChange={(n) => change(() => setBatch(n))}
          />
          <Slider
            label="每坐标噪声标准差 σ"
            min={0}
            max={5}
            step={0.1}
            value={sigma}
            onChange={(n) => change(() => setSigma(n))}
          />
          <Slider
            label="随机学习率"
            min={0.001}
            max={0.15}
            step={0.001}
            value={rate}
            onChange={(n) => change(() => setRate(n))}
          />
          <Slider
            label="随机种子"
            min={1}
            max={200}
            value={seed}
            onChange={(n) => change(() => setSeed(n))}
          />
          <Slider
            label="β₁ / 动量"
            min={0}
            max={0.98}
            step={0.01}
            value={b1}
            onChange={(n) => change(() => setB1(n))}
          />
          <Slider
            label="Adam β₂"
            min={0.9}
            max={0.999}
            step={0.001}
            value={b2}
            onChange={(n) => change(() => setB2(n))}
          />
          <p className="notice">
            每坐标 Var(g_B | x) = σ²/B = {fmt(sigma ** 2 / batch)}；2000
            次独立批采样经验方差 = {fmt(empirical)}。总噪声二阶矩为 2σ²/B。
          </p>
        </aside>
      </div>
      <div className="two-column">
        <section>
          <h2>16 次运行的均值与波动</h2>
          <Curve
            series={[
              { name: "均值", states: mean },
              { name: "当前种子", states: runs[0] },
            ]}
            index={player.index}
            band={band}
          />
          <p className="muted">
            实线：16 个种子的函数间隙均值；虚线：当前种子路径；阴影：排序后第 2
            至第 15 条数值的区间，是经验波动区间，不是置信区间。每步处理 B
            个独立样本，总样本数 = {player.index * batch}。
          </p>
        </section>
        <section>
          <h2>本步随机梯度账本</h2>
          <dl>
            {[
              ["单样本梯度", vector(s.sample)],
              ["批平均梯度", vector(s.mean)],
              ["更新前真实梯度", vector(s.truth)],
              ["批噪声长度", fmt(s.noise)],
              ["一阶矩 m", vector(s.m)],
              ["二阶矩 v", vector(s.v)],
              ["修正一阶矩 m̂", vector(s.mhat)],
              ["修正二阶矩 v̂", vector(s.vhat)],
              ["更新后 x", vector(s.x)],
              ["函数间隙", fmt(s.value)],
            ].map(([k, v]) => (
              <div style={{ display: "contents" }} key={k}>
                <dt>{k}</dt>
                <dd>{v}</dd>
              </div>
            ))}
          </dl>
          {player.index === 0 && (
            <p className="muted">尚未采样，梯度账本从第 1 步开始。</p>
          )}
        </section>
      </div>
      <section>
        <h2>噪声、批量与历史权重</h2>
        <MathFormula
          tex={String.raw`g_i(x)=\nabla f(x)+\xi_i,\quad\xi_i\overset{\mathrm{iid}}\sim\mathcal N(0,\sigma^2I_2),\quad g_B=\frac1B\sum_i g_i`}
        />
        <MathFormula
          tex={String.raw`\mathrm{Cov}(g_B\mid x)=\frac{\sigma^2}{B}I_2`}
        />
        <p>
          条件独立、同方差使交叉协方差为零。相关样本、无放回有限总体和异方差需重新计算，不能无条件套用这一等式。
        </p>
        <MathFormula
          tex={String.raw`m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2`}
        />
        <MathFormula
          tex={String.raw`\hat m_t=\frac{m_t}{1-\beta_1^t},\quad\hat v_t=\frac{v_t}{1-\beta_2^t},\quad x_t=x_{t-1}-\gamma\frac{\hat m_t}{\sqrt{\hat v_t}+10^{-8}}`}
        />
        <p>
          Adam 从 t = 1 开始计数，先在 xₜ₋₁
          采样，所有矩运算逐坐标进行。偏差修正消除零初始化的权重和偏差，不保证最终非线性更新是无偏梯度。本页是独立高斯噪声教学模型，并非训练真实数据集。Momentum
          SGD 采用 mₜ = βmₜ₋₁ + gₜ。
        </p>
      </section>
      <Quiz
        questions={[
          {
            question: "增大独立同方差批量 B 的直接结果是什么？",
            options: [
              "方差缩小为 σ²/B，但每步样本计算增多",
              "必然得到 B 倍墙钟加速",
              "噪声完全消失",
            ],
            answer: 0,
            explanation:
              "方差规律依赖条件独立性与同方差。梯度成本、并行吞吐和通信决定实际耗时。",
          },
        ]}
      />
      <Sources ids={[4, 6]} />
    </>
  );
}
