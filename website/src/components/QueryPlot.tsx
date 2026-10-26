import { objective } from "../engines/zeroth";
import type { Vec } from "../engines/optimization";
export function QueryPlot({
  x,
  plus,
  minus,
  estimated,
  truth,
}: {
  x: Vec;
  plus: Vec;
  minus: Vec;
  estimated: Vec;
  truth: Vec;
}) {
  const maxX = Math.max(2.5, Math.abs(plus[0]), Math.abs(minus[0])) + 0.2,
    minX = -maxX,
    project = (v: number) => 50 + ((v - minX) / (maxX - minX)) * 500;
  const samples = Array.from({ length: 100 }, (_, i) => {
    const a = minX + ((maxX - minX) * i) / 99;
    return [a, objective([a, x[1]])];
  });
  const maxY = Math.max(...samples.map((p) => p[1]));
  const py = (v: number) => 190 - (v / maxY) * 160;
  const origin = [280, 160],
    length = Math.max(Math.hypot(...estimated), Math.hypot(...truth), 1),
    end = (v: Vec) => [
      origin[0] + (v[0] / length) * 100,
      origin[1] - (v[1] / length) * 100,
    ];
  return (
    <div className="two-column">
      <svg
        viewBox="0 0 600 240"
        role="img"
        aria-label="目标函数切片与正负查询点"
      >
        <path d="M50,20V190H570" fill="none" stroke="var(--muted)" />
        <polyline
          points={samples.map(([a, v]) => `${project(a)},${py(v)}`).join(" ")}
          stroke="var(--series-1)"
          strokeWidth="2"
          fill="none"
        />
        {[
          [x, "x", "var(--ink)"],
          [plus, "x+", "var(--series-2)"],
          [minus, "x−", "var(--series-3)"],
        ].map(([v, name, color], i) => {
          const point = v as Vec;
          return (
            <g key={name as string}>
              <circle
                cx={project(point[0])}
                cy={py(objective([point[0], x[1]]))}
                r="4"
                fill={color as string}
              />
              <text
                x={project(point[0]) + 6}
                y={py(objective([point[0], x[1]])) - 10 - i * 15}
                fill={color as string}
                fontSize="12"
              >
                {name as string}
              </text>
            </g>
          );
        })}
        <text x="50" y="220" fontSize="12" fill="var(--muted)">
          固定 x₂ 的真实函数切片；标记只取查询点的 x₁ 投影
        </text>
      </svg>
      <svg
        viewBox="0 0 560 240"
        role="img"
        aria-label="估计梯度与解析梯度方向比较"
      >
        <path d="M80,160H480M280,30V220" stroke="var(--line)" />
        {[
          [truth, "解析梯度", "var(--series-1)"],
          [estimated, "估计梯度", "var(--series-2)"],
        ].map(([v, label, color], i) => {
          const e = end(v as Vec);
          return (
            <g key={i}>
              <line
                x1={origin[0]}
                y1={origin[1]}
                x2={e[0]}
                y2={e[1]}
                stroke={color as string}
                strokeWidth="3"
                strokeDasharray={i ? "6 3" : undefined}
              />
              <circle cx={e[0]} cy={e[1]} r="4" fill={color as string} />
              <text
                x={e[0] + 8}
                y={e[1] + i * 18}
                fill={color as string}
                fontSize="13"
              >
                {label as string}
              </text>
            </g>
          );
        })}
        <text x="70" y="230" fontSize="12" fill="var(--muted)">
          两个方向采用相同缩放；显示梯度而非负梯度
        </text>
      </svg>
    </div>
  );
}
