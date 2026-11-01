import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { createAttentionDemo } from "../engines/attention-demo";
import { distributed, localGradient } from "../engines/distributed";
import { estimate, type Estimator } from "../engines/zeroth";

const weights = (d: ReturnType<typeof createAttentionDemo>) => ({
  heads: d.heads.map(({ WQ, WK, WV }) => ({ WQ, WK, WV })),
  WO: d.WO,
});
describe("知识与教学变量回归", () => {
  it("改变 token 数只追加输入，不更换模型；因果前缀输出保持不变", () => {
    for (const seed of [1, 42, 99]) {
      const short = createAttentionDemo(2, seed, true, 7);
      for (const count of [3, 4, 6]) {
        const long = createAttentionDemo(count, seed, true, 7);
        expect(weights(long)).toEqual(weights(short));
        expect(long.X.slice(0, 2)).toEqual(short.X);
        long.output
          .slice(0, 2)
          .forEach((row, i) =>
            row.forEach((v, j) =>
              expect(v).toBeCloseTo(short.output[i][j], 12),
            ),
          );
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
            method === "坐标中心" ? 0.1 * p.direction[j] : 0,
            12,
          );
        }
        expect(p.coefficient).toBeCloseTo(
          (p.fPlus - p.fMinus) / p.denominator,
          12,
        );
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
    for (const method of [
      "球面单侧",
      "球面双侧",
      "MeZO 高斯双侧",
    ] as Estimator[]) {
      const d = estimate([1, 0.7], 0.1, method, 42);
      const p = d.details[0];
      expect(d.queries).toBe(2);
      expect(p.coordinate).toBeNull();
      expect(p.multiplier).toBe(method === "MeZO 高斯双侧" ? 1 : 2);
      expect(d.g).toEqual(p.contribution);
      p.direction.forEach((u, j) =>
        expect(p.plus[j] - [1, 0.7][j]).toBeCloseTo(0.1 * u, 12),
      );
    }
  });
  it("同质无噪声目标在共同点的梯度一致，Local 多步不产生分歧", () => {
    const run = distributed(4, "Local SGD", 5, 0, 0.1, "无压缩", 42, 12, {
      curvatureSpread: 0,
      noiseStd: 0,
    });
    run.forEach((s) => {
      s.workers.forEach((w) => {
        expect(w.curvature).toBe(1);
        expect(w.target.every((v) => v === 0)).toBe(true);
        w.x.forEach((v, j) => expect(v).toBeCloseTo(s.workers[0].x[j], 12));
        w.gradient.forEach((v, j) =>
          expect(v).toBeCloseTo(s.workers[0].gradient[j], 12),
        );
      });
      expect(s.consensus).toBeCloseTo(0, 12);
    });
  });
  it("共同最优点仍可有曲率异质性，目标同质与采样噪声分开", () => {
    const curved = distributed(2, "Local SGD", 5, 0, 0.1, "无压缩", 42, 1, {
      curvatureSpread: 1,
      noiseStd: 0,
    });
    expect(curved[1].workers[0].gradient).toEqual([-3, 2]);
    expect(curved[1].workers[1].gradient).toEqual([-4.5, 3]);
    expect(localGradient([-3, 2], [0, 0], 1)).toEqual([-3, 2]);
    const noisy = distributed(4, "Local SGD", 5, 0, 0.1, "无压缩", 42, 1, {
      curvatureSpread: 0,
      noiseStd: 0.15,
    });
    expect(noisy[1].consensus).toBeGreaterThan(0);
  });
  it("显式默认配置保留原来的分布式数值模型", () => {
    expect(distributed(4, "PSGD", 1, 2, 0.1, "无压缩", 42, 5)).toEqual(
      distributed(4, "PSGD", 1, 2, 0.1, "无压缩", 42, 5, {
        curvatureSpread: 1,
        noiseStd: 0.15,
      }),
    );
  });
  it("精确枚举随机坐标子空间：无偏不等于低方差", () => {
    const g = [1, -2, 3, -4];
    const norm2 = g.reduce((s, x) => s + x * x, 0);
    for (const rank of [1, 2, 3, 4]) {
      const subsets = Array.from({ length: 16 }, (_, mask) =>
        [0, 1, 2, 3].filter((i) => mask & (1 << i)),
      ).filter((s) => s.length === rank);
      const estimates = subsets.map((s) =>
        g.map((v, i) => (s.includes(i) ? (4 * v) / rank : 0)),
      );
      g.forEach((v, i) =>
        expect(
          estimates.reduce((s, e) => s + e[i], 0) / estimates.length,
        ).toBeCloseTo(v, 12),
      );
      const mse =
        estimates.reduce(
          (s, e) => s + e.reduce((a, v, i) => a + (v - g[i]) ** 2, 0),
          0,
        ) / estimates.length;
      expect(mse).toBeCloseTo((4 / rank - 1) * norm2, 12);
    }
  });
  it("笔记与解释不再使用本轮确认的错误断言", () => {
    const note = (file: string) =>
      readFileSync(resolve("../notes", file), "utf8");
    expect(note("Lecture-05_Gradient-Descent.md")).not.toContain(
      "梯度只能抽样估计",
    );
    expect(note("Lecture-07_Transformer-and-LLM-I.md")).toContain(
      "2022.findings-emnlp.99",
    );
    expect(note("Lecture-10_Subspace-Optimizers.md")).toContain(
      "不必引入统计偏差",
    );
    expect(
      readFileSync(resolve("src/labs/LinearLab.tsx"), "utf8"),
    ).not.toContain("3 维行空间");
  });
});
