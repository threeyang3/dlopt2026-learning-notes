export type Vec = [number, number];
export type Algorithm = "GD" | "Heavy-ball" | "Nesterov";
export interface Quadratic {
  mu: number;
  L: number;
  angle: number;
  optimum: Vec;
}
export interface State {
  x: Vec;
  previous: Vec;
  velocity: Vec;
  gradient: Vec;
  evaluation: Vec;
  value: number;
  norm: number;
  error: number;
}
export const add = (a: Vec, b: Vec): Vec => [a[0] + b[0], a[1] + b[1]];
export const scale = (a: Vec, s: number): Vec => [a[0] * s, a[1] * s];
export const norm = (a: Vec) => Math.hypot(...a);
export function hessian(q: Quadratic): [Vec, Vec] {
  if (!(
    q.mu > 0 &&
    q.L >= q.mu &&
    [q.mu, q.L, q.angle, ...q.optimum].every(Number.isFinite)
  ))
    throw Error("需要有限的 0 < μ ≤ L");
  const c = Math.cos(q.angle),
    s = Math.sin(q.angle);
  return [
    [q.mu * c * c + q.L * s * s, (q.mu - q.L) * c * s],
    [(q.mu - q.L) * c * s, q.mu * s * s + q.L * c * c],
  ];
}
export function gradient(q: Quadratic, x: Vec): Vec {
  const A = hessian(q),
    d = add(x, scale(q.optimum, -1));
  return [A[0][0] * d[0] + A[0][1] * d[1], A[1][0] * d[0] + A[1][1] * d[1]];
}
export function value(q: Quadratic, x: Vec) {
  const d = add(x, scale(q.optimum, -1)),
    g = gradient(q, x);
  return (d[0] * g[0] + d[1] * g[1]) / 2;
}
export function state(
  q: Quadratic,
  x: Vec,
  previous: Vec = x,
  velocity: Vec = [0, 0],
  evaluation: Vec = x,
): State {
  const g = gradient(q, x);
  return {
    x,
    previous,
    velocity,
    evaluation,
    gradient: g,
    value: value(q, x),
    norm: norm(g),
    error: norm(add(x, scale(q.optimum, -1))),
  };
}
export function step(
  q: Quadratic,
  s: State,
  algorithm: Algorithm,
  rate: number,
  beta: number,
): State {
  if (!(rate > 0 && Number.isFinite(rate) && beta >= 0 && beta < 1))
    throw Error("需要 γ > 0，0 ≤ β < 1");
  const evaluation =
    algorithm === "Nesterov"
      ? add(s.x, scale(add(s.x, scale(s.previous, -1)), beta))
      : s.x;
  const g = gradient(q, evaluation),
    velocity = algorithm === "Heavy-ball" ? add(scale(s.velocity, beta), g) : g;
  const x = add(evaluation, scale(velocity, -rate));
  return state(q, x, s.x, velocity, evaluation);
}
export function trajectory(
  q: Quadratic,
  x: Vec,
  algorithm: Algorithm,
  rate: number,
  beta: number,
  count = 120,
) {
  const states = [state(q, x)];
  for (let k = 0; k < count; k++) {
    const next = step(q, states[k], algorithm, rate, beta);
    if (!Number.isFinite(next.value) || next.error > 1e50) break;
    states.push(next);
  }
  return states;
}
export function seeded(seed: number) {
  let a = seed >>> 0;
  return () => {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function normal(rng: () => number) {
  return (
    Math.sqrt(-2 * Math.log(Math.max(rng(), Number.EPSILON))) *
    Math.cos(2 * Math.PI * rng())
  );
}
export function batchGradient(
  q: Quadratic,
  x: Vec,
  batch: number,
  sigma: number,
  rng: () => number,
) {
  if (!Number.isInteger(batch) || batch < 1) throw Error("batch 必须为正整数");
  const truth = gradient(q, x);
  const samples = Array.from({ length: batch }, (): Vec =>
    add(truth, [sigma * normal(rng), sigma * normal(rng)]),
  );
  const mean = scale(samples.reduce(add, [0, 0]), 1 / batch);
  return { truth, samples, mean };
}
export interface StochasticState extends State {
  m: Vec;
  v: Vec;
  mhat: Vec;
  vhat: Vec;
  sample: Vec;
  mean: Vec;
  truth: Vec;
  noise: number;
}
export function stochastic(
  q: Quadratic,
  x: Vec,
  algorithm: "SGD" | "Momentum SGD" | "Adam",
  rate: number,
  batch: number,
  sigma: number,
  seed: number,
  count = 120,
  b1 = 0.9,
  b2 = 0.999,
): StochasticState[] {
  const rng = seeded(seed);
  let m: Vec = [0, 0],
    v: Vec = [0, 0];
  const result: StochasticState[] = [
    {
      ...state(q, x),
      m,
      v,
      mhat: m,
      vhat: v,
      sample: [0, 0],
      mean: [0, 0],
      truth: gradient(q, x),
      noise: 0,
    },
  ];
  for (let t = 1; t <= count; t++) {
    const b = batchGradient(q, x, batch, sigma, rng);
    let mhat: Vec = [0, 0],
      vhat: Vec = [0, 0],
      update = b.mean;
    if (algorithm === "Momentum SGD") {
      m = add(scale(m, b1), b.mean);
      update = m;
    }
    if (algorithm === "Adam") {
      m = add(scale(m, b1), scale(b.mean, 1 - b1));
      v = [
        b2 * v[0] + (1 - b2) * b.mean[0] ** 2,
        b2 * v[1] + (1 - b2) * b.mean[1] ** 2,
      ];
      mhat = scale(m, 1 / (1 - b1 ** t));
      vhat = scale(v, 1 / (1 - b2 ** t));
      update = [
        mhat[0] / (Math.sqrt(vhat[0]) + 1e-8),
        mhat[1] / (Math.sqrt(vhat[1]) + 1e-8),
      ];
    }
    const previous = x;
    x = add(x, scale(update, -rate));
    const next = {
      ...state(q, x, previous, update, previous),
      m: [...m] as Vec,
      v: [...v] as Vec,
      mhat,
      vhat,
      sample: b.samples[0],
      mean: b.mean,
      truth: b.truth,
      noise: norm(add(b.mean, scale(b.truth, -1))),
    };
    if (!Number.isFinite(next.value) || next.error > 1e50) break;
    result.push(next);
  }
  return result;
}
