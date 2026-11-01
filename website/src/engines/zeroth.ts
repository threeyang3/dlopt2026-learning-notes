import { normal, seeded, type Vec } from "./optimization";
export const objective = (x: Vec) =>
  x[0] ** 4 / 4 + x[0] ** 2 / 2 + 2 * x[1] ** 2;
export const trueGradient = (x: Vec): Vec => [x[0] ** 3 + x[0], 4 * x[1]];
export type Estimator =
  "坐标前向" | "坐标中心" | "球面单侧" | "球面双侧" | "MeZO 高斯双侧";
export interface QueryPair {
  coordinate: number | null;
  direction: Vec;
  plus: Vec;
  minus: Vec;
  fPlus: number;
  fMinus: number;
  denominator: number;
  multiplier: number;
  coefficient: number;
  contribution: Vec;
}
export function estimate(
  x: Vec,
  tau: number,
  method: Estimator,
  seed: number,
  noise = 0,
) {
  if (!(Number.isFinite(tau) && tau > 0)) throw Error("τ 必须有限且大于零");
  if (!(Number.isFinite(noise) && noise >= 0) || !x.every(Number.isFinite))
    throw Error("查询点必须有限，噪声标准差必须有限且非负");
  const rng = seeded(seed);
  let queries = 0;
  const query = (y: Vec) => {
    queries++;
    return objective(y) + noise * normal(rng);
  };
  const details: QueryPair[] = [];
  const record = (
    direction: Vec,
    coordinate: number | null,
    centered: boolean,
    multiplier: number,
    base?: number,
  ) => {
    const plus: Vec = [x[0] + tau * direction[0], x[1] + tau * direction[1]];
    const minus: Vec = centered
      ? [x[0] - tau * direction[0], x[1] - tau * direction[1]]
      : [...x];
    const fPlus = query(plus);
    const fMinus = base === undefined ? query(minus) : base;
    const denominator = centered ? 2 * tau : tau;
    const coefficient = (multiplier * (fPlus - fMinus)) / denominator;
    const contribution: Vec = [
      coefficient * direction[0],
      coefficient * direction[1],
    ];
    details.push({
      coordinate,
      direction,
      plus,
      minus,
      fPlus,
      fMinus,
      denominator,
      multiplier,
      coefficient,
      contribution,
    });
  };
  if (method.startsWith("坐标")) {
    const centered = method === "坐标中心";
    // The forward estimator shares exactly one noisy baseline across coordinates.
    const base = centered ? undefined : query(x);
    record([1, 0], 0, centered, 1, base);
    record([0, 1], 1, centered, 1, base);
  } else {
    let u: Vec;
    if (method === "MeZO 高斯双侧") u = [normal(rng), normal(rng)];
    else {
      const angle = rng() * 2 * Math.PI;
      u = [Math.cos(angle), Math.sin(angle)];
    }
    record(u, null, method !== "球面单侧", method === "MeZO 高斯双侧" ? 1 : 2);
  }
  const g = details.reduce<Vec>(
    (sum, pair) => [
      sum[0] + pair.contribution[0],
      sum[1] + pair.contribution[1],
    ],
    [0, 0],
  );
  const truth = trueGradient(x);
  const last = details[details.length - 1];
  return {
    g,
    details,
    // Legacy summary fields always describe the same final query pair.
    u: last.direction,
    plus: last.plus,
    minus: last.minus,
    fp: last.fPlus,
    fm: last.fMinus,
    queries,
    truth,
    error: Math.hypot(g[0] - truth[0], g[1] - truth[1]),
  };
}
