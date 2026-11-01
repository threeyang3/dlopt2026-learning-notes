import { attention, multiply, type Mat } from "./matrix";
import { seeded } from "./optimization";

/** Fixed model weights and a prefix-stable input stream are independent controls. */
export function createAttentionDemo(
  tokens: number,
  weightSeed: number,
  causal: boolean,
  inputSeed = 42,
) {
  if (!Number.isInteger(tokens) || tokens < 1)
    throw Error("Token 数量必须是正整数");
  const weightRng = seeded(weightSeed);
  // Separate streams even when both UI seed controls have the same value.
  const inputRng = seeded(inputSeed ^ 0x9e3779b9);
  const matrix = (rows: number, cols: number, rng: () => number): Mat =>
    Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => 2 * rng() - 1),
    );
  const X = matrix(tokens, 4, inputRng);
  const heads = Array.from({ length: 2 }, () => {
    const WQ = matrix(4, 2, weightRng);
    const WK = matrix(4, 2, weightRng);
    const WV = matrix(4, 2, weightRng);
    const Q = multiply(X, WQ);
    const K = multiply(X, WK);
    const V = multiply(X, WV);
    return { WQ, WK, WV, Q, K, V, ...attention(Q, K, V, causal) };
  });
  const concat = heads[0].output.map((row, i) => [
    ...row,
    ...heads[1].output[i],
  ]);
  const WO = matrix(4, 4, weightRng);
  return { X, heads, concat, WO, output: multiply(concat, WO) };
}
