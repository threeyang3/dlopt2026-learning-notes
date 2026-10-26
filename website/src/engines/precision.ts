export type Format = "FP32" | "FP16" | "BF16";
export const formats = {
  FP32: { exponent: 8, mantissa: 23, bias: 127 },
  FP16: { exponent: 5, mantissa: 10, bias: 15 },
  BF16: { exponent: 8, mantissa: 7, bias: 127 },
};
function evenRound(x: number) {
  const floor = Math.floor(x),
    frac = x - floor;
  return frac === 0.5 ? (floor % 2 === 0 ? floor : floor + 1) : Math.round(x);
}
export function quantize(x: number, format: Format) {
  if (format === "FP32") return Math.fround(x);
  if (!Number.isFinite(x) || x === 0) return x;
  const f = formats[format],
    abs = Math.abs(x),
    emax = 2 ** f.exponent - 2 - f.bias,
    emin = 1 - f.bias,
    exponent = Math.max(emin, Math.floor(Math.log2(abs))),
    spacing = 2 ** (exponent - f.mantissa),
    rounded = evenRound(abs / spacing) * spacing;
  return rounded >= 2 ** (emax + 1)
    ? Math.sign(x) * Infinity
    : Math.sign(x) * rounded;
}
export function formatBounds(format: Format) {
  const f = formats[format];
  return {
    max: (2 - 2 ** -f.mantissa) * 2 ** (2 ** f.exponent - 2 - f.bias),
    normal: 2 ** (1 - f.bias),
    subnormal: 2 ** (1 - f.bias - f.mantissa),
  };
}
