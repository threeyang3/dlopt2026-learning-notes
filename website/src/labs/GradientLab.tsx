import { useMemo, useState } from "react";
import {
  trajectory,
  trajectoryStatus,
  comparisonLimit,
  gradient,
  value,
  hessian,
  type Algorithm,
  type Quadratic,
  type Vec,
} from "../engines/optimization";
import { Landscape, Curve, seriesColor } from "../components/Plots";
import {
  Slider,
  Playback,
  usePlayback,
  fmt,
  vector,
} from "../components/Controls";
import { MathFormula } from "../components/Math";
import { Derivation } from "../components/Derivation";
import { Quiz } from "../components/Quiz";
import { gdDerivation, gdQuestions } from "../content/gd";
export function GradientLab({ preview = false }: { preview?: boolean }) {
  const [condition, setCondition] = useState(12),
    [angle, setAngle] = useState(30),
    [rate, setRate] = useState(0.08),
    [beta, setBeta] = useState(0.65),
    [start, setStart] = useState<Vec>([-4, 3]),
    [optimum, setOptimum] = useState<Vec>([0, 0]),
    [algorithm, setAlgorithm] = useState<Algorithm>("GD"),
    [compare, setCompare] = useState(true),
    [metric, setMetric] = useState<"value" | "norm" | "error">("value"),
    [log, setLog] = useState(true);
  const q: Quadratic = useMemo(
    () => ({ mu: 1, L: condition, angle: (angle * Math.PI) / 180, optimum }),
    [condition, angle, optimum],
  );
  const series = useMemo(
    () =>
      (compare
        ? (["GD", "Heavy-ball", "Nesterov"] as Algorithm[])
        : [algorithm]
      ).map((name) => ({
        name,
        ...trajectory(q, start, name, rate, beta),
      })),
    [q, start, rate, beta, algorithm, compare],
  );
  const max = comparisonLimit(series),
    player = usePlayback(max),
    current = series.find((s) => s.name === algorithm) ?? series[0],
    stateIndex = Math.min(player.index, current.states.length - 1),
    s = current.states[stateIndex],
    previous = current.states[Math.max(0, stateIndex - 1)],
    A = hessian(q),
    g = gradient(q, s.evaluation),
    actual = player.index ? value(q, s.x) - value(q, previous.x) : 0;
  function change(action: () => void) {
    player.reset();
    action();
  }
  return (
    <>
      <div className="lab-intro">
        <p>Lecture 05 · 确定性优化</p>
        <h1>{preview ? "让优化的每一步，都看得见。" : "梯度下降实验室"}</h1>
        <p>通过动态实验、公式推导和交互式比较理解深度学习最优化。</p>
      </div>
      <div className="experiment">
        <div className="visual-panel">
          <div className="section-heading">
            <h2>曲率、方向与轨迹</h2>
            <span>二维 SPD 二次目标</span>
          </div>
          <div className="legend">
            {series.map((s, i) => (
              <span key={s.name} style={{ color: seriesColor(s.name, i) }}>
                {" "}
                {i === 0 ? "━" : i === 1 ? "┄" : "┈"} {s.name}
              </span>
            ))}
          </div>
          <Landscape
            q={q}
            series={series}
            index={player.index}
            onStart={(x) => change(() => setStart(x))}
          />
          <Playback player={player} max={max} />
          <ul
            className="trajectory-status"
            aria-label="各算法计算状态"
            aria-live="polite"
          >
            {series.map((run) => (
              <li
                key={run.name}
                data-testid={"run-" + run.name}
                data-state-iteration={Math.min(
                  player.index,
                  run.states.length - 1,
                )}
              >
                {run.name}：{trajectoryStatus(run, player.index)}；显示状态 k=
                {Math.min(player.index, run.states.length - 1)}
              </li>
            ))}
          </ul>
        </div>
        <aside className="control-panel">
          <h2>改变条件，观察结果</h2>
          <label>
            观察算法
            <select
              value={algorithm}
              onChange={(e) =>
                change(() => setAlgorithm(e.target.value as Algorithm))
              }
            >
              {["GD", "Heavy-ball", "Nesterov"].map((a) => (
                <option key={a}>{a}</option>
              ))}
            </select>
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={compare}
              onChange={(e) => change(() => setCompare(e.target.checked))}
            />
            同时比较三种算法
          </label>
          <div className="presets">
            {[
              ["圆形", 1, 0],
              ["狭长", 24, 0],
              ["旋转椭圆", 12, 30],
            ].map(([name, k, a]) => (
              <button
                key={name}
                onClick={() =>
                  change(() => {
                    setCondition(+k);
                    setAngle(+a);
                  })
                }
              >
                {name}
              </button>
            ))}
          </div>
          <Slider
            label="条件数 κ = L/μ"
            min={1}
            max={40}
            value={condition}
            onChange={(n) => change(() => setCondition(n))}
          />
          <Slider
            label="旋转角度 θ (°)"
            min={0}
            max={90}
            value={angle}
            onChange={(n) => change(() => setAngle(n))}
          />
          <Slider
            label="学习率 γ"
            min={0.001}
            max={0.5}
            step={0.001}
            value={rate}
            onChange={(n) => change(() => setRate(n))}
          />
          <Slider
            label="动量 β"
            min={0}
            max={0.98}
            step={0.01}
            value={beta}
            onChange={(n) => change(() => setBeta(n))}
          />
          <div className="toolbar">
            <button onClick={() => change(() => setRate(1 / q.L))}>
              γ = 1/L
            </button>
            <button
              onClick={() =>
                change(() => {
                  setRate(1 / q.L);
                  setBeta((Math.sqrt(q.L) - 1) / (Math.sqrt(q.L) + 1));
                })
              }
            >
              强凸 NAG 参数
            </button>
          </div>
          {[0, 1].map((i) => (
            <Slider
              key={i}
              label={`初始坐标 x${i + 1}`}
              min={-5}
              max={5}
              step={0.1}
              value={start[i]}
              onChange={(n) =>
                change(() =>
                  setStart(start.map((v, j) => (i === j ? n : v)) as Vec),
                )
              }
            />
          ))}
          <details>
            <summary>修改最优点 x★</summary>
            {[0, 1].map((i) => (
              <Slider
                key={i}
                label={`最优点坐标 ${i + 1}`}
                min={-2}
                max={2}
                step={0.1}
                value={optimum[i]}
                onChange={(n) =>
                  change(() =>
                    setOptimum(optimum.map((v, j) => (i === j ? n : v)) as Vec),
                  )
                }
              />
            ))}
          </details>
          <p className="notice">
            μ = 1，L = {q.L}。GD 对所有初值收敛的区间为 0 &lt; γ &lt;{" "}
            {fmt(2 / q.L)}。当前谱收缩因子 ρ ={" "}
            {fmt(Math.max(Math.abs(1 - rate), Math.abs(1 - rate * q.L)))}。
          </p>
        </aside>
      </div>
      <div className="metrics" aria-live="polite">
        {[
          ["当前算法", current.name + `（状态 k=${stateIndex}）`],
          ["f(xₖ) − f★", fmt(s.value)],
          ["‖∇f(xₖ)‖₂", fmt(s.norm)],
          ["‖xₖ − x★‖₂", fmt(s.error)],
        ].map(([l, v]) => (
          <div key={l}>
            <span>{l}</span>
            <strong>{v}</strong>
          </div>
        ))}
      </div>
      {!preview && (
        <>
          <div className="two-column">
            <section>
              <div className="section-heading">
                <h2>收敛，不只看一条曲线</h2>
                <label className="check">
                  <input
                    type="checkbox"
                    checked={log}
                    onChange={(e) => setLog(e.target.checked)}
                  />
                  对数轴
                </label>
              </div>
              <select
                aria-label="收敛指标"
                value={metric}
                onChange={(e) => setMetric(e.target.value as typeof metric)}
              >
                <option value="value">函数间隙 f(xₖ) − f★</option>
                <option value="norm">梯度范数</option>
                <option value="error">参数误差</option>
              </select>
              <Curve
                series={series}
                index={player.index}
                metric={metric}
                log={log}
              />
              <p className="muted">
                相同目标、初值、γ、β，均为每步一次梯度。对数轴将零及低于 10⁻¹⁶
                的值截在 10⁻¹⁶；横轴是迭代数，并非硬件耗时。NAG 推荐参数仅适用于
                L-光滑、μ-强凸设定。
              </p>
            </section>
            <section className="state-panel">
              <h2>第 {stateIndex} 步的计算账本</h2>
              <p>{trajectoryStatus(current, player.index)}</p>
              <dl>
                <dt>参数 xₖ</dt>
                <dd>{vector(s.x)}</dd>
                <dt>上一步参数</dt>
                <dd>{vector(previous.x)}</dd>
                <dt>本次梯度评估点</dt>
                <dd>{vector(s.evaluation)}</dd>
                <dt>评估点梯度</dt>
                <dd>{vector(g)}</dd>
                <dt>当前位置梯度</dt>
                <dd>{vector(s.gradient)}</dd>
                <dt>速度 / 使用的方向</dt>
                <dd>{vector(s.velocity)}</dd>
                <dt>本步函数值变化</dt>
                <dd>{fmt(actual)}</dd>
              </dl>
              <p>
                相对初值函数间隙比值：
                {current.states[0].value === 0
                  ? "初值已是最优点"
                  : fmt(s.value / current.states[0].value)}
                。
                {s.value > 1e12
                  ? "当前数值已快速增大，请回退观察过冲或降低步长。"
                  : "比较当前坐标与曲线，检查振荡是否伴随误差下降。"}
              </p>
              <p className="muted">
                超过 10⁵⁰
                的参数误差会停止计算，保留越界时的有限状态，非有限的下一次尝试仅记录发生位置。各算法独立停止，指标停止后不延长为虚构的新迭代。每个按钮恢复保存状态，播放速度不影响递推结果。
              </p>
            </section>
          </div>
          <section>
            <h2>同一个目标，三种更新</h2>
            <MathFormula
              tex={String.raw`f(x)=\tfrac12(x-x^\star)^\top A(x-x^\star),\quad A=R_\theta\mathrm{diag}(1,\kappa)R_\theta^\top,\quad \nabla f(x)=A(x-x^\star)`}
            />
            <p>
              A = {vector(A[0])}，{vector(A[1])}；形状 2×2，秩 2，特征值 1 与{" "}
              {q.L}。旋转保持正定性。
            </p>
            <MathFormula
              tex={String.raw`\text{GD: }x_{k+1}=x_k-\gamma\nabla f(x_k)`}
            />
            <MathFormula
              tex={String.raw`\text{Heavy-ball: }v_{k+1}=\beta v_k+\nabla f(x_k),\quad x_{k+1}=x_k-\gamma v_{k+1}`}
            />
            <p>
              速度约定 v₀ = 0；与笔记的位移速度形式相差 −γ
              的缩放。每步多保留一个向量。
            </p>
            <MathFormula
              tex={String.raw`\text{NAG: }y_k=x_k+\beta(x_k-x_{k-1}),\quad x_{k+1}=y_k-\gamma\nabla f(y_k)`}
            />
            <p>
              x₋₁ = x₀。这里采用固定动量的常见强凸参数化；Heavy-ball 与 NAG
              的梯度评估点不同。任意手动 β 不自动享有推荐参数的收敛保证。
            </p>
          </section>
          <section>
            <h2>一次更新，需要多少资源？</h2>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>算法</th>
                    <th>梯度评估</th>
                    <th>额外持久状态</th>
                    <th>向量运算</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>GD</td>
                    <td>1 次，在 xₖ</td>
                    <td>无历史向量</td>
                    <td>O(d)</td>
                  </tr>
                  <tr>
                    <td>Heavy-ball</td>
                    <td>1 次，在 xₖ</td>
                    <td>速度 v，d 个元素</td>
                    <td>O(d)</td>
                  </tr>
                  <tr>
                    <td>NAG</td>
                    <td>1 次，在 yₖ</td>
                    <td>上一步参数，d 个元素</td>
                    <td>O(d)，另需外推临时量</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="muted">
              三者共同的当前参数与梯度未计入额外状态；本页
              d=2，全部轨迹历史是学习界面的存储，不是算法必需状态。一次梯度成本在同一二次目标上可比，迭代数不能直接代表真实模型的墙钟时间。
            </p>
          </section>
          <Derivation steps={gdDerivation} />
          <section className="numeric-proof">
            <h2>把下降界代入真实数字</h2>
            <p>
              以当前 GD 状态为 x，计算下一步，即使正在观察其他算法也独立核验
              GD。
            </p>
            {(() => {
              const gdRun = series.find((run) => run.name === "GD"),
                gd =
                  gdRun?.states[
                    Math.min(player.index, gdRun.states.length - 1)
                  ] ?? s,
                gg = gradient(q, gd.x),
                next = value(q, [
                  gd.x[0] - rate * gg[0],
                  gd.x[1] - rate * gg[1],
                ]),
                bound =
                  gd.value - (rate - (q.L * rate ** 2) / 2) * gd.norm ** 2;
              return (
                <p>
                  f(x − γ∇f(x)) = {fmt(next)} ≤ 二次上界 {fmt(bound)}；差值{" "}
                  {fmt(bound - next)}。简化的 γ/2 界当前
                  {rate <= 1 / q.L ? "适用" : "不适用：γ > 1/L"}。
                </p>
              );
            })()}
          </section>
          <Quiz questions={gdQuestions} />
        </>
      )}
    </>
  );
}
