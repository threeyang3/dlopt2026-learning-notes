import { useState } from "react";
import { estimate, type Estimator } from "../engines/zeroth";
import { Slider, fmt, vector } from "../components/Controls";
import { MathFormula } from "../components/Math";
import { Sources } from "../components/Sources";
import { QueryPlot } from "../components/QueryPlot";
import { Quiz } from "../components/Quiz";
export function ZerothLab() {
  const [method, setMethod] = useState<Estimator>("坐标中心"),
    [exponent, setExponent] = useState(-2),
    [noise, setNoise] = useState(0),
    [seed, setSeed] = useState(42),
    [x, setX] = useState(1),
    [coordinate, setCoordinate] = useState(0);
  const tau = 10 ** exponent,
    d = estimate([x, 0.7], tau, method, seed, noise),
    pair = d.details[Math.min(coordinate, d.details.length - 1)];
  return (
    <>
      <div className="lab-intro">
        <p>Lecture 11</p>
        <h1>只看函数值，能找到方向吗？</h1>
        <p>非二次目标 f(x) = x₁⁴/4 + x₁²/2 + 2x₂²，使差分截断偏差真实可见。</p>
      </div>
      <div className="lab-body two-column">
        <div>
          <label>
            梯度估计器
            <select
              aria-label="零阶估计器"
              value={method}
              onChange={(e) => {
                setMethod(e.target.value as Estimator);
                setCoordinate(0);
              }}
            >
              {[
                "坐标前向",
                "坐标中心",
                "球面单侧",
                "球面双侧",
                "MeZO 高斯双侧",
              ].map((a) => (
                <option key={a}>{a}</option>
              ))}
            </select>
          </label>
          <Slider
            label="log₁₀ τ"
            min={-12}
            max={0}
            step={0.1}
            value={exponent}
            onChange={setExponent}
          />
          <Slider
            label="查询噪声标准差"
            min={0}
            max={0.01}
            step={0.0001}
            value={noise}
            onChange={setNoise}
          />
          <Slider
            label="查询点 x₁"
            min={-0.5}
            max={2}
            step={0.1}
            value={x}
            onChange={setX}
          />
          <Slider
            label="方向与噪声种子"
            min={1}
            max={200}
            value={seed}
            onChange={setSeed}
          />
          <p>
            τ = {fmt(tau)}；函数查询 {d.queries} 次；估计误差 {fmt(d.error)}。
          </p>
        </div>
        <div>
          <h2>本次查询与方向</h2>
          {d.details.length > 1 && (
            <label>
              查看坐标查询
              <select
                aria-label="查看坐标查询"
                value={coordinate}
                onChange={(e) => setCoordinate(Number(e.target.value))}
              >
                <option value={0}>坐标 x₁</option>
                <option value={1}>坐标 x₂</option>
              </select>
            </label>
          )}
          <p
            data-testid="query-pair"
            data-direction={JSON.stringify(pair.direction)}
            data-plus={JSON.stringify(pair.plus)}
            data-minus={JSON.stringify(pair.minus)}
          >
            {pair.coordinate === null
              ? "随机方向 u"
              : `当前坐标方向 e${pair.coordinate + 1}`}
            ：{vector(pair.direction)}
          </p>
          <dl>
            {[
              ["正扰动点", vector(pair.plus)],
              ["负扰动 / 基准点", vector(pair.minus)],
              ["f₊ / f₋", vector([pair.fPlus, pair.fMinus])],
              ["函数差", fmt(pair.fPlus - pair.fMinus)],
              ["估计梯度", vector(d.g)],
              ["解析梯度", vector(d.truth)],
              ["差分分母", fmt(pair.denominator)],
              ["缩放因子", fmt(pair.multiplier)],
              ["差分系数", fmt(pair.coefficient)],
              ["本组方向贡献", vector(pair.contribution)],
            ].map(([k, v]) => (
              <div style={{ display: "contents" }} key={k}>
                <dt>{k}</dt>
                <dd>{v}</dd>
              </div>
            ))}
          </dl>
          <p className="muted">
            坐标法可逐项查看查询，完整梯度由各坐标贡献相加。前向差分复用同一次基准查询；各次实际查询使用独立高斯噪声，因此共享基准会使坐标估计误差相关。另含
            JavaScript 浮点舍入。固定种子便于隔离 τ 的影响。
          </p>
        </div>
      </div>
      <QueryPlot
        x={[x, 0.7]}
        plus={pair.plus}
        minus={pair.minus}
        estimated={d.g}
        truth={d.truth}
      />
      <section>
        <h2>改变尺度，分清两种误差</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>τ</th>
                <th>坐标前向误差</th>
                <th>坐标中心误差</th>
              </tr>
            </thead>
            <tbody>
              {[-1, -2, -3, -5, -8, -12].map((e) => (
                <tr key={e}>
                  <td>10^{e}</td>
                  <td>
                    {fmt(
                      estimate([x, 0.7], 10 ** e, "坐标前向", seed, noise)
                        .error,
                    )}
                  </td>
                  <td>
                    {fmt(
                      estimate([x, 0.7], 10 ** e, "坐标中心", seed, noise)
                        .error,
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <MathFormula
          tex={String.raw`\widehat g_{\mathrm{sphere}}=d\,\frac{f(x+\tau u)-f(x-\tau u)}{2\tau}u,\quad u\sim\mathrm{Unif}(S^{d-1}),\quad d=2`}
        />
        <MathFormula
          tex={String.raw`\widehat g_{\mathrm{MeZO}}=\frac{f(x+\tau u)-f(x-\tau u)}{2\tau}u,\quad u\sim\mathcal N(0,I_d)`}
        />
        <p>
          二维球面通过均匀角度采样，不是从球体中采样。球面估计器带维度因子
          d；高斯 MeZO 不带。单次估计一般不等于解析梯度。
        </p>
        <MathFormula
          tex={String.raw`f_\tau(x)=\mathbb E_{v\sim\mathrm{Unif}(B_d)}f(x+\tau v),\quad \mathbb E[\widehat g_{\mathrm{sphere}}]=\nabla f_\tau(x)`}
        />
        <p>
          在邻域连续可微及相应可积条件下，球面单侧与对称双侧的期望对应球体平滑梯度；高斯双侧对应高斯平滑梯度。截断偏差、方向方差和函数查询噪声是三个不同来源。τ
          太小会放大差分舍入及查询噪声。
        </p>
      </section>
      <Quiz
        questions={[
          {
            question: "单位球面采样和单位球体平滑有什么关系？",
            options: [
              "两者是同一分布",
              "球面方向估计器的期望可对应球体平滑梯度",
              "单方向总能恢复原梯度",
            ],
            answer: 1,
            explanation:
              "估计方向在边界球面，平滑积分在球体内；恒等式需要规则性与正确维度因子。",
          },
        ]}
      />
      <Sources ids={[11]} />
    </>
  );
}
