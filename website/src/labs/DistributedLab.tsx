import { useMemo, useState } from "react";
import {
  distributed,
  type DistributedMethod,
  type Compression,
} from "../engines/distributed";
import {
  Slider,
  Playback,
  usePlayback,
  fmt,
  vector,
} from "../components/Controls";
import { Curve, colors } from "../components/Plots";
import { state } from "../engines/optimization";
import { MathFormula } from "../components/Math";
import { Sources } from "../components/Sources";
import { Quiz } from "../components/Quiz";
export function DistributedLab() {
  const [nodes, setNodes] = useState(4),
    [method, setMethod] = useState<DistributedMethod>("Local SGD"),
    [interval, setInterval] = useState(5),
    [heterogeneity, setHeterogeneity] = useState(2),
    [compression, setCompression] = useState<Compression>("无压缩"),
    [rate, setRate] = useState(0.1),
    [seed, setSeed] = useState(42);
  const states = useMemo(
      () =>
        distributed(
          nodes,
          method,
          interval,
          heterogeneity,
          rate,
          compression,
          seed,
        ),
      [nodes, method, interval, heterogeneity, rate, compression, seed],
    ),
    player = usePlayback(100),
    s = states[player.index];
  function change(action: () => void) {
    player.reset();
    action();
  }
  const series = [
    {
      name: "全局目标 F(x̄)",
      states: states.map((s) => ({
        ...state({ mu: 1, L: 1, angle: 0, optimum: [0, 0] }, s.average),
        value: s.global,
      })),
    },
    {
      name: "共识误差",
      states: states.map((s) => ({
        ...state({ mu: 1, L: 1, angle: 0, optimum: [0, 0] }, s.average),
        value: s.consensus,
      })),
    },
  ];
  return (
    <>
      <div className="lab-intro">
        <p>Lecture 12</p>
        <h1>分布式优化：走多远，再通信？</h1>
        <p>
          每个节点拥有不同曲率与最优点。真实执行本地更新、聚合与带索引的压缩通信。
        </p>
      </div>
      <div className="experiment">
        <div className="visual-panel">
          <svg viewBox="0 0 560 420" role="img" aria-label="各节点参数轨迹">
            <path d="M20,210H540M280,20V400" stroke="var(--line)" />
            {s.workers.map((w, i) => (
              <g key={i}>
                <polyline
                  points={states
                    .slice(0, player.index + 1)
                    .map(
                      (s) =>
                        `${280 + s.workers[i].x[0] * 45},${210 - s.workers[i].x[1] * 45}`,
                    )
                    .join(" ")}
                  fill="none"
                  stroke={colors[i % colors.length]}
                  strokeWidth="2"
                />
                <circle
                  cx={280 + w.x[0] * 45}
                  cy={210 - w.x[1] * 45}
                  r="6"
                  fill={colors[i % colors.length]}
                />
                <text
                  x={286 + w.x[0] * 45}
                  y={200 - w.x[1] * 45}
                  fill={colors[i % colors.length]}
                  fontSize="12"
                >
                  节点 {i + 1}
                </text>
                <text
                  x={280 + w.target[0] * 45}
                  y={210 - w.target[1] * 45}
                  fill={colors[i % colors.length]}
                >
                  ×
                </text>
              </g>
            ))}
            <text x="15" y="410" fontSize="12" fill="var(--muted)">
              × 本地最优点；轨迹为参数；坐标单位每格 45 px
            </text>
          </svg>
          <Playback player={player} max={100} />
        </div>
        <aside className="control-panel">
          <h2>通信与本地设置</h2>
          <label>
            算法
            <select
              value={method}
              onChange={(e) =>
                change(() => setMethod(e.target.value as DistributedMethod))
              }
            >
              {["PSGD", "DSGD", "Local SGD"].map((a) => (
                <option key={a}>{a}</option>
              ))}
            </select>
          </label>
          <Slider
            label="节点数量"
            min={2}
            max={6}
            value={nodes}
            onChange={(n) => change(() => setNodes(n))}
          />
          <Slider
            label="本地步数 K"
            min={1}
            max={15}
            value={interval}
            onChange={(n) => change(() => setInterval(n))}
          />
          <Slider
            label="数据异质性"
            min={0}
            max={4}
            step={0.1}
            value={heterogeneity}
            onChange={(n) => change(() => setHeterogeneity(n))}
          />
          <Slider
            label="节点学习率"
            min={0.01}
            max={0.4}
            step={0.01}
            value={rate}
            onChange={(n) => change(() => setRate(n))}
          />
          <Slider
            label="通信随机种子"
            min={1}
            max={200}
            value={seed}
            onChange={(n) => change(() => setSeed(n))}
          />
          <label>
            PSGD 梯度压缩
            <select
              disabled={method !== "PSGD"}
              value={compression}
              onChange={(e) =>
                change(() => setCompression(e.target.value as Compression))
              }
            >
              {["无压缩", "Random-k", "Top-k"].map((a) => (
                <option key={a}>{a}</option>
              ))}
            </select>
          </label>
          <p className="muted">
            压缩只用于 PSGD 梯度聚合；DSGD 与 Local SGD 传递完整参数。
          </p>
        </aside>
      </div>
      <div className="metrics">
        {[
          ["全局 F(x̄)", fmt(s.global)],
          ["均方共识误差", fmt(s.consensus)],
          ["通信轮", String(s.rounds)],
          ["累计逻辑通信", s.bytes + " bytes"],
        ].map(([k, v]) => (
          <div key={k}>
            <span>{k}</span>
            <strong>{v}</strong>
          </div>
        ))}
      </div>
      <Curve series={series} index={player.index} />
      <p className="muted">
        实线：全局目标 F(x̄)；虚线：均方共识误差。压缩坐标值与索引各 4
        bytes（本模拟 d=2、k=1），下发平均梯度按稠密 8 bytes
        计。小维度稀疏传输并未省字节。PSGD / Local
        采用每节点上行和下行账本；DSGD 计每节点两条有向邻居消息，n=2
        时重复边按两条消息计；未模拟真实 collective 或延迟。
      </p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>节点</th>
              <th>本地参数</th>
              <th>本步随机梯度</th>
              <th>本地目标</th>
              <th>更新数</th>
              <th>通信状态</th>
            </tr>
          </thead>
          <tbody>
            {s.workers.map((w, i) => (
              <tr key={i}>
                <td>{i + 1}</td>
                <td>{vector(w.x)}</td>
                <td>{vector(w.gradient)}</td>
                <td>
                  a={fmt(w.curvature)}，x★={vector(w.target)}
                </td>
                <td>{w.updates}</td>
                <td>{w.communicating ? "已通信" : "本地计算"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <section>
        <MathFormula
          tex={String.raw`F_i(x)=\tfrac{a_i}2\|x-c_i\|^2,\quad F(x)=\tfrac1n\sum_iF_i(x)`}
        />
        <MathFormula tex={String.raw`\text{DSGD: }X_{k+1}=(X_k-\gamma G_k)W`} />
        <p>
          本页 DSGD 先更新再混合，环上自己与左右邻居各占 1/3，W
          对称双随机。Local SGD 每 K 步均匀平均参数；PSGD
          在共同参数上平均（可压缩的）随机梯度。独立噪声每坐标标准差 0.15。
        </p>
        <MathFormula
          tex={String.raw`[C(x)]_i=\frac dk x_i\mathbb1\{i\in S\},\quad\mathbb E[C(x)]=x,\quad d=2,\ k=1`}
        />
        <p>
          Random-k 均匀保留一个坐标并乘 2；Top-k
          按幅值保留一个且不缩放，通常有偏。本教学模型没有误差反馈。共同初始化、K=1
          和完整均匀聚合时 Local SGD 与 PSGD 对应；环混合与梯度平均不能混同。
        </p>
      </section>
      <Quiz
        questions={[
          {
            question: "未缩放 Random-k 的期望是什么？",
            options: ["总是 x", "(k/d)x", "零"],
            answer: 1,
            explanation:
              "坐标被选中概率为 k/d，需乘 d/k 才可恢复期望。Top-k 的选择依赖坐标幅值，不能套用该概率。",
          },
        ]}
      />
      <Sources ids={[12]} />
    </>
  );
}
