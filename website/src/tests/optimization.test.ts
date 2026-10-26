import { describe, it, expect } from "vitest";
import {
  gradient,
  value,
  state,
  step,
  trajectory,
  seeded,
  batchGradient,
  stochastic,
  hessian,
  type Quadratic,
} from "../engines/optimization";
const q: Quadratic = { mu: 2, L: 8, angle: 0, optimum: [1, -1] };
describe("解析二次函数与优化递推", () => {
  it("正确计算梯度、最优点与范数", () => {
    expect(gradient(q, [3, 2])).toEqual([4, 24]);
    expect(value(q, q.optimum)).toBe(0);
    expect(value(q, [3, 2])).toBe(40);
    expect(state(q, [3, 2]).norm).toBeCloseTo(Math.sqrt(592));
    expect(() => hessian({ ...q, mu: 0 })).toThrow();
  });
  it("旋转二次函数梯度与独立中心差分一致", () => {
    const rotated = { ...q, angle: 0.7 },
      x: [number, number] = [0.3, -0.8],
      h = 1e-5;
    for (let i = 0; i < 2; i++) {
      const plus = [...x] as typeof x,
        minus = [...x] as typeof x;
      plus[i] += h;
      minus[i] -= h;
      expect(gradient(rotated, x)[i]).toBeCloseTo(
        (value(rotated, plus) - value(rotated, minus)) / (2 * h),
        7,
      );
    }
  });
  it("GD 与手算一致", () => {
    expect(step(q, state(q, [3, 2]), "GD", 0.1, 0).x).toEqual([
      2.6, -0.40000000000000036,
    ]);
  });
  it("Heavy-ball 速度累积并保持约定", () => {
    const s = state(q, [3, 2], [3, 2], [1, 2]),
      next = step(q, s, "Heavy-ball", 0.1, 0.5);
    expect(next.velocity).toEqual([4.5, 25]);
    expect(next.x[0]).toBeCloseTo(2.55);
    expect(next.x[1]).toBeCloseTo(-0.5);
  });
  it("Nesterov 在外推点评估", () => {
    const next = step(q, state(q, [3, 2], [2, 1]), "Nesterov", 0.1, 0.5);
    expect(next.evaluation).toEqual([3.5, 2.5]);
    expect(next.x[0]).toBeCloseTo(3);
    expect(next.x[1]).toBeCloseTo(-0.3);
  });
  it("稳定区间下降，超界真实发散", () => {
    const stable = trajectory(q, [3, 2], "GD", 0.1, 0),
      unstable = trajectory(q, [3, 2], "GD", 0.3, 0);
    expect(stable.at(-1)!.value).toBeLessThan(1e-15);
    expect(unstable.at(-1)!.value).toBeGreaterThan(1e10);
  });
  it("光滑下降界与数值一致", () => {
    for (const rate of [0.01, 0.08, 0.125, 0.2]) {
      const s = state(q, [3, 2]),
        next = step(q, s, "GD", rate, 0);
      expect(next.value).toBeLessThanOrEqual(
        s.value - (rate - (q.L * rate ** 2) / 2) * s.norm ** 2 + 1e-12,
      );
    }
  });
});
describe("采样、mini-batch 与 Adam", () => {
  it("随机种子可重复且不同种子不同", () => {
    const a = seeded(42),
      b = seeded(42),
      c = seeded(43);
    expect(Array.from({ length: 10 }, a)).toEqual(
      Array.from({ length: 10 }, b),
    );
    expect(seeded(42)()).not.toBe(c());
  });
  it("mini-batch 真正求平均", () => {
    const b = batchGradient(q, [2, 1], 4, 1, seeded(1));
    for (let i = 0; i < 2; i++)
      expect(b.mean[i]).toBeCloseTo(
        b.samples.reduce((a, s) => a + s[i], 0) / 4,
      );
  });
  it("独立高斯批平均的经验方差接近 σ²/B", () => {
    for (const B of [1, 8]) {
      const rng = seeded(7),
        n = 60000,
        noise = Array.from(
          { length: n },
          () => batchGradient(q, q.optimum, B, 2, rng).mean[0],
        ),
        mean = noise.reduce((a, b) => a + b) / n,
        variance = noise.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
      expect(variance).toBeCloseTo(4 / B, 1);
    }
  });
  it("Adam 首步修正为真实梯度与平方，更新接近符号步", () => {
    const s = stochastic(q, [3, 2], "Adam", 0.1, 1, 0, 1, 1)[1];
    expect(s.mhat[0]).toBeCloseTo(4);
    expect(s.vhat[0]).toBeCloseTo(16);
    expect(s.x[0]).toBeCloseTo(2.9);
    expect(s.x[1]).toBeCloseTo(1.9);
  });
});
