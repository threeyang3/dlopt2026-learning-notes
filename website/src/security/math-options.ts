import type { KatexOptions } from "katex";

/** Shared limits for interactive formulas and Markdown mathematics. */
export const mathOptions: KatexOptions = {
  trust: false,
  strict: "ignore",
  maxExpand: 1000,
  maxSize: 20,
};
