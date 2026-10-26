import { normal, seeded, type Vec } from "./optimization";
export const objective = (x: Vec) =>
  x[0] ** 4 / 4 + x[0] ** 2 / 2 + 2 * x[1] ** 2;
export const trueGradient = (x: Vec): Vec => [x[0] ** 3 + x[0], 4 * x[1]];
export type Estimator =
  "坐标前向" | "坐标中心" | "球面单侧" | "球面双侧" | "MeZO 高斯双侧";
export function estimate(
  x: Vec,
  tau: number,
  method: Estimator,
  seed: number,
  noise = 0,
) {
  if (!(tau > 0)) throw Error("τ > 0");
  const rng = seeded(seed),
    f = (y: Vec) => objective(y) + noise * normal(rng);
  let queries = 0;
  const query = (y: Vec) => {
    queries++;
    return f(y);
  };
  let u: Vec = [1, 0],
    plus: Vec = x,
    minus: Vec = x,
    fp = 0,
    fm = 0,
    g: Vec = [0, 0];
  if (method.startsWith("坐标")) {
    const base = method === "坐标前向" ? query(x) : 0;
    g = [0, 1].map((i) => {
      const p = [...x] as Vec,
        m = [...x] as Vec;
      p[i] += tau;
      m[i] -= tau;
      plus = p;
      minus = method === "坐标前向" ? x : m;
      fp = query(p);
      fm = method === "坐标前向" ? base : query(m);
      return (fp - fm) / (method === "坐标前向" ? tau : 2 * tau);
    }) as Vec;
  } else {
    if (method === "MeZO 高斯双侧") u = [normal(rng), normal(rng)];
    else {
      const a = rng() * 2 * Math.PI;
      u = [Math.cos(a), Math.sin(a)];
    }
    plus = [x[0] + tau * u[0], x[1] + tau * u[1]];
    minus = method === "球面单侧" ? x : [x[0] - tau * u[0], x[1] - tau * u[1]];
    fp = query(plus);
    fm = query(minus);
    const factor = method === "MeZO 高斯双侧" ? 1 : 2,
      denominator = method === "球面单侧" ? tau : 2 * tau;
    g = [
      ((factor * (fp - fm)) / denominator) * u[0],
      ((factor * (fp - fm)) / denominator) * u[1],
    ];
  }
  const truth = trueGradient(x);
  return {
    g,
    u,
    plus,
    minus,
    fp,
    fm,
    queries,
    truth,
    error: Math.hypot(g[0] - truth[0], g[1] - truth[1]),
  };
}
