import { useId } from "react";
import {
  gradient,
  norm,
  type Quadratic,
  type State,
  type Vec,
  type Trajectory,
  trajectoryStatus,
} from "../engines/optimization";
export const colors = Array.from(
  { length: 6 },
  (_, i) => `var(--series-${i + 1})`,
);
export const seriesColor = (name: string, index: number) =>
  colors[
    (
      {
        GD: 0,
        "Heavy-ball": 1,
        Nesterov: 2,
        SGD: 0,
        "Momentum SGD": 1,
        Adam: 2,
      } as Record<string, number>
    )[name] ?? index % 6
  ];
export interface Series {
  name: string;
  states: State[];
  termination?: Trajectory["termination"];
}
export function Landscape({
  q,
  series,
  index,
  onStart,
}: {
  q: Quadratic;
  series: Series[];
  index: number;
  onStart?: (x: Vec) => void;
}) {
  const id = useId().replace(/:/g, "");
  const project = (x: Vec) => [300 + 42 * x[0], 240 - 42 * x[1]];
  const p = (x: Vec) => project(x).join(",");
  const levelPoints = (r: number) =>
    Array.from({ length: 101 }, (_, i) => {
      const t = (i * Math.PI * 2) / 100,
        a = (r * Math.cos(t)) / Math.sqrt(q.mu),
        b = (r * Math.sin(t)) / Math.sqrt(q.L);
      return p([
        q.optimum[0] + a * Math.cos(q.angle) - b * Math.sin(q.angle),
        q.optimum[1] + a * Math.sin(q.angle) + b * Math.cos(q.angle),
      ]);
    }).join(" ");
  return (
    <svg
      className="landscape"
      viewBox="0 0 600 480"
      role="img"
      aria-label="目标函数等高线、算法轨迹和当前梯度"
    >
      <defs>
        <clipPath id={id + "clip"}>
          <rect x="20" y="15" width="560" height="440" />
        </clipPath>
        <marker
          id={id + "arrow"}
          markerWidth="7"
          markerHeight="7"
          refX="5"
          refY="3"
          orient="auto"
        >
          <path d="M0,0 L6,3 L0,6" fill="none" stroke="currentColor" />
        </marker>
      </defs>
      <rect width="600" height="480" fill="var(--plot)" />
      {Array.from({ length: 13 }, (_, i) => (
        <g key={i} className="grid">
          <path d={`M${48 + i * 42},15V455`} />
          <path d={`M20,${30 + i * 35}H580`} />
        </g>
      ))}
      <g clipPath={`url(#${id}clip)`}>
        {[1, 2, 3, 4, 5, 6, 8, 10, 13, 17, 22, 29].map((r) => (
          <polyline
            key={r}
            points={levelPoints(r)}
            fill="none"
            stroke="var(--contour)"
          />
        ))}
        <path d="M20,240H580M300,15V455" stroke="var(--muted)" opacity=".5" />
        {series.map((s, i) => {
          const current = s.states[Math.min(index, s.states.length - 1)],
            g = gradient(q, current.x),
            length = norm(g),
            end = length
              ? ([
                  current.x[0] - (g[0] / length) * 1.1,
                  current.x[1] - (g[1] / length) * 1.1,
                ] as Vec)
              : current.x;
          return (
            <g key={s.name} style={{ color: seriesColor(s.name, i) }}>
              <polyline
                points={s.states
                  .slice(0, index + 1)
                  .map((s) => p(s.x))
                  .join(" ")}
                stroke={seriesColor(s.name, i)}
                strokeWidth="2.6"
                strokeDasharray={i === 1 ? "7 3" : i === 2 ? "2 3" : undefined}
                fill="none"
              />
              <line
                x1={project(current.x)[0]}
                y1={project(current.x)[1]}
                x2={project(end)[0]}
                y2={project(end)[1]}
                stroke={seriesColor(s.name, i)}
                strokeWidth="2"
                markerEnd={`url(#${id}arrow)`}
              />
              <circle
                cx={project(current.x)[0]}
                cy={project(current.x)[1]}
                r={5}
                fill={seriesColor(s.name, i)}
                stroke="var(--surface)"
                strokeWidth="2"
              />
              <text
                x={project(current.x)[0] + 8}
                y={project(current.x)[1] - 8 - i * 16}
                fill={seriesColor(s.name, i)}
                fontSize="12"
              >
                {s.name}
              </text>
            </g>
          );
        })}
        <circle
          cx={project(q.optimum)[0]}
          cy={project(q.optimum)[1]}
          r="5"
          fill="var(--ink)"
        />
        <text
          x={project(q.optimum)[0] + 8}
          y={project(q.optimum)[1] + 17}
          fill="var(--ink)"
          fontSize="12"
        >
          x★
        </text>
      </g>
      <text x="568" y="261" fill="var(--ink)">
        x₁
      </text>
      <text x="311" y="30" fill="var(--ink)">
        x₂
      </text>
      <text x="25" y="476" fill="var(--muted)" fontSize="12">
        坐标范围固定，超出窗口的轨迹裁剪；箭头为归一化 −∇f
      </text>
      {series.map(
        (run, i) =>
          run.termination && (
            <text
              key={run.name}
              x="25"
              y={32 + i * 17}
              fill={seriesColor(run.name, i)}
              fontSize="11"
            >
              <title>
                {trajectoryStatus(
                  { states: run.states, termination: run.termination },
                  index,
                )}
              </title>
              {run.name}：
              {index < run.termination.iteration
                ? "运行中"
                : `${run.termination.reason === "diverged" ? "数值终止" : "预算结束"} k=${run.termination.iteration}`}
            </text>
          ),
      )}
      {onStart && (
        <rect
          x="20"
          y="15"
          width="560"
          height="440"
          fill="transparent"
          style={{ cursor: "crosshair" }}
          onClick={(e) => {
            const r = e.currentTarget.ownerSVGElement!.getBoundingClientRect();
            onStart([
              ((600 * (e.clientX - r.left)) / r.width - 300) / 42,
              (240 - (480 * (e.clientY - r.top)) / r.height) / 42,
            ]);
          }}
        />
      )}
    </svg>
  );
}
export function Curve({
  series,
  index,
  metric = "value",
  log = true,
  band,
}: {
  series: Series[];
  index: number;
  metric?: "value" | "norm" | "error";
  log?: boolean;
  band?: { low: number; high: number }[];
}) {
  const transform = (v: number) => (log ? Math.log10(Math.max(v, 1e-16)) : v);
  const all = series
    .flatMap((s) =>
      s.states.slice(0, index + 1).map((x) => transform(x[metric])),
    )
    .filter(Number.isFinite);
  if (band)
    all.push(
      ...band
        .slice(0, index + 1)
        .flatMap((b) => [transform(b.low), transform(b.high)]),
    );
  const lo = Math.min(...all, 0),
    hi = Math.max(...all, lo + 1),
    y = (v: number) => 180 - ((transform(v) - lo) / (hi - lo)) * 155;
  const length = Math.max(
      1,
      ...series.map((s) =>
        Math.max(s.states.length - 1, s.termination?.iteration ?? 0),
      ),
    ),
    x = (k: number) => 62 + (k / length) * 510;
  return (
    <svg
      className="curve"
      viewBox="0 0 600 220"
      role="img"
      aria-label="收敛指标曲线"
    >
      <path d="M62,20V180H580" fill="none" stroke="var(--muted)" />
      {[0, 1, 2, 3].map((i) => (
        <g key={i}>
          <path d={`M62,${25 + (i * 155) / 3}H580`} stroke="var(--line)" />
          <text x="3" y={29 + (i * 155) / 3} fontSize="11" fill="var(--muted)">
            {(log
              ? 10 ** (hi - (i * (hi - lo)) / 3)
              : hi - (i * (hi - lo)) / 3
            ).toExponential(1)}
          </text>
        </g>
      ))}
      {band && (
        <polygon
          fill={colors[0]}
          opacity=".13"
          points={[
            ...band.slice(0, index + 1).map((b, k) => `${x(k)},${y(b.high)}`),
            ...band
              .slice(0, index + 1)
              .map((b, k) => `${x(k)},${y(b.low)}`)
              .reverse(),
          ].join(" ")}
        />
      )}
      {series.map((s, i) => (
        <polyline
          data-testid={"curve-" + s.name}
          key={s.name}
          points={s.states
            .slice(0, index + 1)
            .map((v, k) => `${x(k)},${y(v[metric])}`)
            .join(" ")}
          fill="none"
          stroke={seriesColor(s.name, i)}
          strokeWidth="2.3"
          strokeDasharray={i === 1 ? "7 3" : i === 2 ? "2 3" : undefined}
        />
      ))}
      <text x="62" y="203" fill="var(--muted)" fontSize="12">
        0
      </text>
      <text x="505" y="203" fill="var(--muted)" fontSize="12">
        迭代 k / {length}
      </text>
    </svg>
  );
}
