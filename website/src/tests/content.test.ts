import { it, expect } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import course from "../content/course.json";
import { modules } from "../content/modules";
import { sourceReview } from "../content/source-review";
it("14 讲索引逐项映射真实笔记、目标与资料，先修顺序无环", () => {
  expect(course).toHaveLength(14);
  course.forEach((l, i) => {
    expect(l.id).toBe(i + 1);
    const text = readFileSync(resolve("..", l.note), "utf8");
    expect(text).toContain(l.title);
    expect(text).toContain(l.goals);
    l.pdfs.forEach((p) => expect(existsSync(resolve("..", p))).toBe(true));
    expect(modules[l.lab]).toBeDefined();
    expect(l.prerequisites.every((x) => x < l.id)).toBe(true);
    expect(l.status).toBe("partially_verified");
  });
});
it("抽查来源是实际存在的 PDF 且有明确证据与页序", () => {
  for (const sources of Object.values(sourceReview))
    for (const s of sources) {
      expect(existsSync(resolve("..", s.path))).toBe(true);
      expect(s.page).toBeGreaterThan(0);
      expect(s.method).toMatch(/核对/);
    }
});
