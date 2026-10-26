import data from "./course.json";
export const course = data;
export const labNames: Record<string, string> = {
  gd: "梯度下降",
  sgd: "随机梯度与 Adam",
  linear: "线性变换与 SVD",
  subspace: "低秩投影",
  zeroth: "零阶梯度估计",
  distributed: "分布式优化",
  attention: "Attention",
  flash: "分块 Attention",
  precision: "浮点精度",
  inference: "LoRA 与 KV Cache",
  resources: "优化器资源账本",
};
export const rawNotes = import.meta.glob("../../../notes/*.md", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;
export function noteText(slug: string) {
  return rawNotes["../../../notes/" + slug + ".md"] ?? "";
}
export const resource = (path: string) =>
  import.meta.env.BASE_URL + "resources/" + path;
