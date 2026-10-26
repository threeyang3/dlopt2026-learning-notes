import { describe, it, expect } from "vitest";
import {
  lowRank,
  multiply,
  transpose,
  attention,
  onlineAttention,
  softmax,
  kvBytes,
  loraCount,
} from "../engines/matrix";
import { estimate, objective, trueGradient } from "../engines/zeroth";
import { compress, distributed } from "../engines/distributed";
import { quantize, formatBounds } from "../engines/precision";
import { seeded } from "../engines/optimization";
describe("矩阵、投影与 Attention", () => {
  it("已知奇异值矩阵截断误差与正交条件", () => {
    const G = [
        [3, 0],
        [0, 2],
        [0, 0],
      ],
      r = lowRank(G, 1);
    expect(r.singular).toEqual([3, 2]);
    expect(r.error).toBeCloseTo(2);
    expect(r.projected).toHaveLength(1);
    expect(r.projected[0]).toHaveLength(2);
    expect(r.lifted).toHaveLength(3);
    expect(multiply(transpose(r.P), r.P)[0][0]).toBeCloseTo(1);
    expect(lowRank(G, 2).error).toBeCloseTo(0);
    expect(() => multiply(r.P, G)).toThrow();
  });
  it("稠密 attention 与手算，mask 不窥未来", () => {
    const Q = [
        [0, 0],
        [0, 0],
      ],
      K = [
        [1, 0],
        [0, 1],
      ],
      V = [
        [2, 4],
        [6, 8],
      ],
      a = attention(Q, K, V);
    expect(a.weights).toEqual([
      [0.5, 0.5],
      [0.5, 0.5],
    ]);
    expect(a.output).toEqual([
      [4, 6],
      [4, 6],
    ]);
    expect(attention(Q, K, V, true).output[0]).toEqual([2, 4]);
    expect(() => softmax([-Infinity, -Infinity])).toThrow();
    expect(softmax([1000, 1001]).reduce((a, b) => a + b)).toBeCloseTo(1);
  });
  it("在线 attention 多种块大小与独立稠密计算一致", () => {
    const Q = [[2, -1]],
      K = [
        [1, 2],
        [3, 4],
        [-2, 1],
        [5, -2],
      ],
      V = [
        [1, 0],
        [0, 2],
        [-3, 1],
        [4, 2],
      ],
      expected = attention(Q, K, V).output[0];
    for (const size of [1, 2, 3, 4]) {
      const states = onlineAttention(Q[0], K, V, size);
      states
        .at(-1)!
        .output.forEach((x, i) => expect(x).toBeCloseTo(expected[i], 12));
    }
  });
  it("LoRA 和 KV cache 计数包括两个因子和 K/V 两份", () => {
    expect(loraCount(4, 3, 2)).toBe(14);
    expect(kvBytes(2, 10, 3, 4, 8, 2)).toBe(7680);
  });
});
describe("差分与随机方向", () => {
  it("中心差分与解析梯度一致且二阶误差", () => {
    const x: [number, number] = [1, 0.7],
      large = estimate(x, 0.1, "坐标中心", 1),
      small = estimate(x, 0.01, "坐标中心", 1);
    expect(large.error / small.error).toBeCloseTo(100, 6);
    expect(small.g[0]).toBeCloseTo(trueGradient(x)[0], 3);
    expect(large.queries).toBe(4);
    expect(estimate(x, 0.1, "坐标前向", 1).queries).toBe(3);
    expect(objective([0, 0])).toBe(0);
  });
  it("球面与高斯方向遵循不同范数与维度缩放", () => {
    const s = estimate([1, 0.7], 0.01, "球面双侧", 42),
      m = estimate([1, 0.7], 0.01, "MeZO 高斯双侧", 42);
    expect(Math.hypot(...s.u)).toBeCloseTo(1);
    expect(s.g[0]).toBeCloseTo(((2 * (s.fp - s.fm)) / 0.02) * s.u[0]);
    expect(m.g[0]).toBeCloseTo(((m.fp - m.fm) / 0.02) * m.u[0]);
    expect(m.queries).toBe(2);
    expect(estimate([1, 0.7], 0.01, "MeZO 高斯双侧", 42)).toEqual(m);
  });
});
describe("分布式、通信与浮点", () => {
  it("无偏 Random-k 的经验均值与 Top-k 幅值规则", () => {
    const rng = seeded(11),
      N = 50000,
      sum = [0, 0];
    for (let i = 0; i < N; i++) {
      const c = compress([1, -3], "Random-k", 1, rng);
      sum[0] += c[0] / N;
      sum[1] += c[1] / N;
    }
    expect(sum[0]).toBeCloseTo(1, 1);
    expect(sum[1]).toBeCloseTo(-3, 1);
    expect(compress([1, -3], "Top-k", 1, rng)).toEqual([0, -3]);
  });
  it("共同初值 Local K=1 与 PSGD 逐步相同", () => {
    const a = distributed(4, "PSGD", 1, 2, 0.1, "无压缩", 42, 10),
      b = distributed(4, "Local SGD", 1, 2, 0.1, "无压缩", 42, 10);
    a.forEach((s, k) =>
      s.workers.forEach((w, i) =>
        w.x.forEach((v, j) => expect(v).toBeCloseTo(b[k].workers[i].x[j], 12)),
      ),
    );
  });
  it("DSGD 保留均值且不同于全局一次平均", () => {
    const p = distributed(4, "PSGD", 1, 2, 0.1, "无压缩", 42, 1)[1],
      d = distributed(4, "DSGD", 1, 2, 0.1, "无压缩", 42, 1)[1];
    expect(d.average[0]).toBeCloseTo(p.average[0]);
    expect(d.consensus).toBeGreaterThan(0);
    expect(p.consensus).toBe(0);
  });
  it("Local 通信轮与字节和独立种子重复", () => {
    const a = distributed(3, "Local SGD", 4, 2, 0.1, "无压缩", 42, 10);
    expect(a.at(-1)!.rounds).toBe(2);
    expect(a.at(-1)!.bytes).toBe(3 * 16 * 2);
    expect(a).toEqual(distributed(3, "Local SGD", 4, 2, 0.1, "无压缩", 42, 10));
  });
  it("FP16 边界与偶数舍入、次正规数和溢出", () => {
    expect(formatBounds("FP16")).toEqual({
      max: 65504,
      normal: 2 ** -14,
      subnormal: 2 ** -24,
    });
    expect(quantize(1 + 2 ** -11, "FP16")).toBe(1);
    expect(quantize(1 + 3 * 2 ** -11, "FP16")).toBe(1 + 2 ** -9);
    expect(quantize(2 ** -24, "FP16")).toBe(2 ** -24);
    expect(quantize(2 ** -25, "FP16")).toBe(0);
    expect(quantize(65504, "FP16")).toBe(65504);
    expect(quantize(65520, "FP16")).toBe(Infinity);
    expect(quantize(-70000, "FP16")).toBe(-Infinity);
  });
  it("BF16 比 FP16 范围宽但 1 附近精度低", () => {
    expect(quantize(1e10, "BF16")).toBeGreaterThan(1e9);
    expect(quantize(1e10, "FP16")).toBe(Infinity);
    expect(quantize(1.001, "BF16")).toBe(1);
    expect(quantize(1.001, "FP16")).not.toBe(1);
  });
});
