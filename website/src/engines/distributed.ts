import { seeded, normal, type Vec } from "./optimization";
export type DistributedMethod = "PSGD" | "DSGD" | "Local SGD";
export type Compression = "无压缩" | "Random-k" | "Top-k";
export function compress(
  x: Vec,
  mode: Compression,
  k: number,
  rng: () => number,
): Vec {
  if (k === 2 || mode === "无压缩") return [...x];
  const index =
    mode === "Top-k"
      ? Math.abs(x[0]) >= Math.abs(x[1])
        ? 0
        : 1
      : rng() < 0.5
        ? 0
        : 1;
  return index === 0
    ? [x[0] * (mode === "Random-k" ? 2 : 1), 0]
    : [0, x[1] * (mode === "Random-k" ? 2 : 1)];
}
export interface Worker {
  x: Vec;
  gradient: Vec;
  target: Vec;
  curvature: number;
  updates: number;
  communicating: boolean;
}
export function distributed(
  nodes: number,
  method: DistributedMethod,
  interval: number,
  heterogeneity: number,
  rate: number,
  mode: Compression,
  seed: number,
  steps = 100,
) {
  const rng = seeded(seed);
  let workers: Worker[] = Array.from({ length: nodes }, (_, i) => ({
    x: [-3, 2],
    gradient: [0, 0],
    target: [
      heterogeneity * Math.cos((i * 2 * Math.PI) / nodes),
      heterogeneity * Math.sin((i * 2 * Math.PI) / nodes),
    ],
    curvature: 1 + i / nodes,
    updates: 0,
    communicating: false,
  }));
  let rounds = 0,
    bytes = 0;
  const mean = (xs: Vec[]): Vec => [
    xs.reduce((s, x) => s + x[0], 0) / nodes,
    xs.reduce((s, x) => s + x[1], 0) / nodes,
  ];
  const snapshot = () => {
    const average = mean(workers.map((w) => w.x)),
      consensus =
        workers.reduce(
          (s, w) => s + (w.x[0] - average[0]) ** 2 + (w.x[1] - average[1]) ** 2,
          0,
        ) / nodes,
      global =
        workers.reduce(
          (s, w) =>
            s +
            (w.curvature *
              ((average[0] - w.target[0]) ** 2 +
                (average[1] - w.target[1]) ** 2)) /
              2,
          0,
        ) / nodes;
    return {
      workers: structuredClone(workers),
      average,
      consensus,
      global,
      rounds,
      bytes,
    };
  };
  const states = [snapshot()];
  for (let t = 1; t <= steps; t++) {
    const gradients = workers.map(
      (w) =>
        [
          w.curvature * (w.x[0] - w.target[0]) + 0.15 * normal(rng),
          w.curvature * (w.x[1] - w.target[1]) + 0.15 * normal(rng),
        ] as Vec,
    );
    const comm = method !== "Local SGD" || t % interval === 0;
    if (method === "PSGD") {
      const g = mean(gradients.map((g) => compress(g, mode, 1, rng))),
        x = mean(workers.map((w) => w.x));
      workers = workers.map((w, i) => ({
        ...w,
        x: [x[0] - rate * g[0], x[1] - rate * g[1]],
        gradient: gradients[i],
        updates: t,
        communicating: true,
      }));
      rounds++;
      const payload = mode === "无压缩" ? 8 : 8;
      bytes += nodes * (payload + 8);
    } else {
      const local = workers.map(
        (w, i) =>
          [
            w.x[0] - rate * gradients[i][0],
            w.x[1] - rate * gradients[i][1],
          ] as Vec,
      );
      let mixed = local;
      if (method === "DSGD") {
        mixed = local.map(
          (x, i) =>
            [0, 1].map(
              (j) =>
                (x[j] +
                  local[(i + 1) % nodes][j] +
                  local[(i + nodes - 1) % nodes][j]) /
                3,
            ) as Vec,
        );
        rounds++;
        bytes += nodes * 2 * 8;
      }
      if (method === "Local SGD" && comm) {
        const x = mean(local);
        mixed = local.map(() => [...x] as Vec);
        rounds++;
        bytes += nodes * 16;
      }
      workers = workers.map((w, i) => ({
        ...w,
        x: mixed[i],
        gradient: gradients[i],
        updates: t,
        communicating: comm,
      }));
    }
    states.push(snapshot());
  }
  return states;
}
