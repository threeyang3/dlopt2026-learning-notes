import { Matrix, SingularValueDecomposition } from "ml-matrix";
export type Mat = number[][];
export const transpose = (a: Mat): Mat =>
  a[0].map((_, j) => a.map((row) => row[j]));
export function multiply(a: Mat, b: Mat): Mat {
  if (
    !a.length ||
    !b.length ||
    a[0].length !== b.length ||
    a.some((r) => r.length !== a[0].length) ||
    b.some((r) => r.length !== b[0].length)
  )
    throw Error("矩阵维度不相容");
  return a.map((row) =>
    b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)),
  );
}
export const frobenius = (a: Mat) =>
  Math.sqrt(a.flat().reduce((s, x) => s + x * x, 0));
export function lowRank(a: Mat, rank: number) {
  if (
    !Number.isInteger(rank) ||
    rank < 1 ||
    rank > Math.min(a.length, a[0].length)
  )
    throw Error("无效秩");
  const svd = new SingularValueDecomposition(new Matrix(a), {
    autoTranspose: true,
  });
  const U = svd.leftSingularVectors.to2DArray(),
    V = svd.rightSingularVectors.to2DArray(),
    singular = svd.diagonal;
  const P = U.map((row) => row.slice(0, rank)),
    projected = multiply(transpose(P), a),
    lifted = multiply(P, projected),
    error = frobenius(a.map((row, i) => row.map((x, j) => x - lifted[i][j])));
  return {
    U,
    V,
    P,
    singular,
    projected,
    lifted,
    error,
    rank: singular.filter((x) => x > Math.max(...singular) * 1e-12).length,
  };
}
export function softmax(row: number[]) {
  const m = Math.max(...row);
  if (!Number.isFinite(m)) throw Error("至少一个有效有限 logit");
  const e = row.map((x) => Math.exp(x - m)),
    sum = e.reduce((a, b) => a + b);
  return e.map((v) => v / sum);
}
export function attention(Q: Mat, K: Mat, V: Mat, causal = false) {
  if (K.length !== V.length || Q[0].length !== K[0].length)
    throw Error("Attention 维度不相容");
  const logits = multiply(Q, transpose(K)).map((row, i) =>
      row.map((v, j) =>
        causal && j > i ? -Infinity : v / Math.sqrt(Q[0].length),
      ),
    ),
    weights = logits.map(softmax);
  return { logits, weights, output: multiply(weights, V) };
}
export function onlineAttention(
  q: number[],
  K: Mat,
  V: Mat,
  blockSize: number,
) {
  if (!Number.isInteger(blockSize) || blockSize < 1)
    throw Error("块大小必须为正整数");
  let m = -Infinity,
    l = 0,
    z = V[0].map(() => 0);
  const states = [];
  for (let start = 0; start < K.length; start += blockSize) {
    const end = Math.min(start + blockSize, K.length),
      logits = K.slice(start, end).map(
        (k) => q.reduce((s, v, j) => s + v * k[j], 0) / Math.sqrt(q.length),
      ),
      nextM = Math.max(m, ...logits),
      rescale = Math.exp(m - nextM),
      exps = logits.map((s) => Math.exp(s - nextM));
    l = rescale * l + exps.reduce((a, b) => a + b);
    z = z.map(
      (v, j) =>
        v * rescale + exps.reduce((a, e, k) => a + e * V[start + k][j], 0),
    );
    m = nextM;
    states.push({
      start,
      end,
      m,
      l,
      z: [...z],
      output: z.map((v) => v / l),
      logits,
    });
  }
  return states;
}
export function loraCount(rows: number, cols: number, rank: number) {
  return rank * (rows + cols);
}
export function kvBytes(
  batch: number,
  tokens: number,
  layers: number,
  heads: number,
  width: number,
  bytes: number,
) {
  return 2 * batch * tokens * layers * heads * width * bytes;
}
