import { seeded } from "./optimization";

// PDF p.6: deliberately illustrative axes, not a trained embedding model.
export const embeddingWords = [
  "man",
  "woman",
  "king",
  "queen",
  "apple",
  "orange",
] as const;
export const embeddingAxes = ["gender（示意）", "royal", "age", "food"];
export const slideEmbedding = [
  [-1, 1, -0.95, 0.97, 0, 0.01],
  [0.01, 0.02, 0.93, 0.95, -0.01, 0],
  [0.51, 0.47, 0.7, 0.69, 0.03, -0.02],
  [0.04, 0.01, 0.02, 0.01, 0.95, 0.97],
];
export function embeddingColumn(index: number) {
  if (!Number.isInteger(index) || index < 0 || index >= embeddingWords.length)
    throw Error("Invalid vocabulary index");
  return slideEmbedding.map((row) => row[index]);
}
export function cosine(a: number[], b: number[]) {
  if (
    !a.length ||
    a.length !== b.length ||
    ![...a, ...b].every(Number.isFinite)
  )
    throw Error("Invalid vectors");
  const norm = Math.hypot(...a) * Math.hypot(...b);
  return norm === 0 ? null : a.reduce((s, x, i) => s + x * b[i], 0) / norm;
}

export const toyCorpus: readonly (readonly string[])[] = [
  ...Array.from({ length: 4 }, () => [
    "the",
    "students",
    "opened",
    "their",
    "books",
  ]),
  ["the", "students", "opened", "their", "exams"],
  ...Array.from({ length: 3 }, () => [
    "the",
    "students",
    "opened",
    "their",
    "laptops",
  ]),
  ...Array.from({ length: 2 }, () => [
    "the",
    "students",
    "opened",
    "their",
    "minds",
  ]),
  ["i", "like", "orange", "juice"],
  ["i", "like", "apple", "juice"],
];
export interface NgramModel {
  order: number;
  vocabulary: string[];
  counts: Map<string, number[]>;
}
const contextKey = (context: readonly string[], order: number) =>
  JSON.stringify(order === 1 ? [] : context.slice(-(order - 1)));
export function buildNgram(
  corpus: readonly (readonly string[])[],
  order: number,
): NgramModel {
  if (
    !Number.isInteger(order) ||
    order < 1 ||
    order > 4 ||
    !corpus.length ||
    corpus.some(
      (s) => !s.length || s.some((w) => !w || w === "<bos>" || w === "<eos>"),
    )
  )
    throw Error(
      "Use nonempty sentences and n in 1..4; boundaries are added internally",
    );
  const vocabulary = [...new Set(corpus.flat()), "<eos>"];
  const counts = new Map<string, number[]>();
  for (const sentence of corpus) {
    const padded = [
      ...Array<string>(order - 1).fill("<bos>"),
      ...sentence,
      "<eos>",
    ];
    for (let i = order - 1; i < padded.length; i++) {
      const key = contextKey(padded.slice(0, i), order);
      const row = counts.get(key) ?? Array<number>(vocabulary.length).fill(0);
      row[vocabulary.indexOf(padded[i])] += 1;
      counts.set(key, row);
    }
  }
  return { order, vocabulary, counts };
}
export function ngramDistribution(
  model: NgramModel,
  context: readonly string[],
  alpha = 0,
) {
  if (!Number.isFinite(alpha) || alpha < 0)
    throw Error("alpha must be nonnegative");
  const counts = [
    ...(model.counts.get(contextKey(context, model.order)) ??
      Array<number>(model.vocabulary.length).fill(0)),
  ];
  const total = counts.reduce((s, c) => s + c, 0);
  const denominator = total + alpha * counts.length;
  return {
    counts,
    total,
    denominator,
    probabilities:
      denominator === 0 ? null : counts.map((c) => (c + alpha) / denominator),
  };
}
export function sampleToken(probabilities: readonly number[], u: number) {
  const sum = probabilities.reduce((a, b) => a + b, 0);
  if (
    !probabilities.length ||
    probabilities.some((p) => !Number.isFinite(p) || p < 0) ||
    Math.abs(sum - 1) > 1e-10 ||
    !Number.isFinite(u) ||
    u < 0 ||
    u >= 1
  )
    throw Error("Invalid categorical distribution or uniform variate");
  let cumulative = 0;
  for (let i = 0; i < probabilities.length; i++) {
    cumulative += probabilities[i];
    if (u < cumulative) return i;
  }
  for (let i = probabilities.length - 1; i >= 0; i--) {
    if (probabilities[i] > 0) return i;
  }
  throw Error("Empty categorical support");
}
export function generateNgram(
  model: NgramModel,
  context: string[],
  alpha: number,
  seed: number,
  steps: number,
) {
  if (!Number.isInteger(steps) || steps < 0 || steps > 20)
    throw Error("Invalid generation length");
  const rng = seeded(seed),
    tokens = [...context];
  let reason: "limit" | "eos" | "unseen" = "limit";
  for (let i = 0; i < steps; i++) {
    const distribution = ngramDistribution(model, tokens, alpha);
    if (!distribution.probabilities) {
      reason = "unseen";
      break;
    }
    const word =
      model.vocabulary[sampleToken(distribution.probabilities, rng())];
    tokens.push(word);
    if (word === "<eos>") {
      reason = "eos";
      break;
    }
  }
  return { tokens, reason };
}
export function categoricalNll(probability: number) {
  if (!Number.isFinite(probability) || probability < 0 || probability > 1)
    throw Error("Invalid probability");
  return -Math.log(probability); // p=0 deliberately remains +Infinity.
}

export const rnnVocabulary = [
  "the",
  "students",
  "opened",
  "their",
  "books",
  "exams",
  "<eos>",
];
export type Matrix = number[][];
export interface RnnWeights {
  E: Matrix;
  Wh: Matrix;
  We: Matrix;
  U: Matrix;
  b1: number[];
  b2: number[];
}
export interface RnnState {
  input: number;
  target: number;
  embedding: number[];
  previous: number[];
  recurrent: number[];
  incoming: number[];
  preactivation: number[];
  hidden: number[];
  logits: number[];
  probabilities: number[];
  loss: number;
}
const zeros = (rows: number, cols: number): Matrix =>
  Array.from({ length: rows }, () => Array<number>(cols).fill(0));
const mv = (a: Matrix, x: number[]) =>
  a.map((row) => row.reduce((s, v, j) => s + v * x[j], 0));
const transposeMv = (a: Matrix, x: number[]) =>
  a[0].map((_, j) => a.reduce((s, row, i) => s + row[j] * x[i], 0));
export function makeRnnWeights(seed: number): RnnWeights {
  const rng = seeded(seed),
    matrix = (r: number, c: number) =>
      Array.from({ length: r }, () =>
        Array.from({ length: c }, () => rng() - 0.5),
      );
  return {
    E: matrix(3, 7),
    Wh: matrix(2, 2),
    We: matrix(2, 3),
    U: matrix(7, 2),
    b1: [0, 0],
    b2: Array<number>(7).fill(0),
  };
}
function validateRnn(w: RnnWeights, inputs: number[], targets: number[]) {
  const shape = (a: Matrix, r: number, c: number) =>
    a.length === r &&
    a.every((row) => row.length === c && row.every(Number.isFinite));
  if (
    !shape(w.E, 3, 7) ||
    !shape(w.Wh, 2, 2) ||
    !shape(w.We, 2, 3) ||
    !shape(w.U, 7, 2) ||
    w.b1.length !== 2 ||
    w.b2.length !== 7 ||
    ![...w.b1, ...w.b2].every(Number.isFinite) ||
    !inputs.length ||
    inputs.length > 30 ||
    inputs.length !== targets.length ||
    [...inputs, ...targets].some((i) => !Number.isInteger(i) || i < 0 || i >= 7)
  )
    throw Error("Invalid toy RNN dimensions or token indices");
}
export function rnnForward(w: RnnWeights, inputs: number[], targets: number[]) {
  validateRnn(w, inputs, targets);
  let previous = [0, 0];
  const states: RnnState[] = inputs.map((input, t) => {
    const embedding = w.E.map((row) => row[input]);
    const recurrent = mv(w.Wh, previous),
      incoming = mv(w.We, embedding);
    const preactivation = recurrent.map((v, j) => v + incoming[j] + w.b1[j]);
    const hidden = preactivation.map(Math.tanh);
    const logits = mv(w.U, hidden).map((v, j) => v + w.b2[j]);
    const max = Math.max(...logits),
      exps = logits.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0),
      probabilities = exps.map((v) => v / sum);
    const loss = max + Math.log(sum) - logits[targets[t]];
    const state = {
      input,
      target: targets[t],
      embedding,
      previous: [...previous],
      recurrent,
      incoming,
      preactivation,
      hidden,
      logits,
      probabilities,
      loss,
    };
    previous = hidden;
    return state;
  });
  if (states.some((s) => !Number.isFinite(s.loss)))
    throw Error("Non-finite RNN loss");
  return {
    states,
    loss: states.reduce((s, x) => s + x.loss, 0) / states.length,
  };
}
// Supplementary derivation: full BPTT of mean next-token CE, including E.
export function rnnGradients(
  w: RnnWeights,
  inputs: number[],
  targets: number[],
) {
  const forward = rnnForward(w, inputs, targets),
    T = inputs.length;
  const g: RnnWeights = {
    E: zeros(3, 7),
    Wh: zeros(2, 2),
    We: zeros(2, 3),
    U: zeros(7, 2),
    b1: [0, 0],
    b2: Array<number>(7).fill(0),
  };
  const outerAdd = (a: Matrix, x: number[], y: number[]) =>
    x.forEach((v, i) =>
      y.forEach((u, j) => {
        a[i][j] += v * u;
      }),
    );
  let dhNext = [0, 0];
  for (let t = T - 1; t >= 0; t--) {
    const s = forward.states[t];
    const dz = s.probabilities.map(
      (p, j) => (p - (j === s.target ? 1 : 0)) / T,
    );
    outerAdd(g.U, dz, s.hidden);
    dz.forEach((v, j) => {
      g.b2[j] += v;
    });
    const dh = transposeMv(w.U, dz).map((v, j) => v + dhNext[j]);
    const da = dh.map((v, j) => v * (1 - s.hidden[j] ** 2));
    outerAdd(g.Wh, da, s.previous);
    outerAdd(g.We, da, s.embedding);
    da.forEach((v, j) => {
      g.b1[j] += v;
    });
    transposeMv(w.We, da).forEach((v, j) => {
      g.E[j][s.input] += v;
    });
    dhNext = transposeMv(w.Wh, da);
  }
  return { ...forward, gradients: g };
}
export function trainRnnStep(
  w: RnnWeights,
  inputs: number[],
  targets: number[],
  rate: number,
) {
  if (!Number.isFinite(rate) || rate <= 0 || rate > 0.5)
    throw Error("Invalid learning rate");
  const { gradients: g } = rnnGradients(w, inputs, targets);
  const update = (a: Matrix, b: Matrix) =>
    a.map((row, i) => row.map((v, j) => v - rate * b[i][j]));
  return {
    E: update(w.E, g.E),
    Wh: update(w.Wh, g.Wh),
    We: update(w.We, g.We),
    U: update(w.U, g.U),
    b1: w.b1.map((v, j) => v - rate * g.b1[j]),
    b2: w.b2.map((v, j) => v - rate * g.b2[j]),
  };
}
// Scalar zero-input tanh recurrence: derivative w.r.t. initial hidden state,
// not the parameter gradient and not an implementation of LSTM or attention.
export function scalarMemory(weight: number, initial: number, length: number) {
  if (
    !Number.isFinite(weight) ||
    Math.abs(weight) > 2 ||
    !Number.isFinite(initial) ||
    Math.abs(initial) > 1 ||
    !Number.isInteger(length) ||
    length < 0 ||
    length > 30
  )
    throw Error("Invalid scalar recurrence");
  const states = [{ step: 0, hidden: initial, local: 1, derivative: 1 }];
  for (let step = 1; step <= length; step++) {
    const previous = states[step - 1],
      hidden = Math.tanh(weight * previous.hidden);
    const local = weight * (1 - hidden ** 2);
    states.push({
      step,
      hidden,
      local,
      derivative: previous.derivative * local,
    });
  }
  return states;
}
