import { describe, expect, it } from "vitest";
import { headingSlug, resolveNoteHref } from "../components/Notes";

describe("课程笔记章节链接", () => {
  it("为中文 Markdown 标题生成稳定的 ID", () => {
    expect(headingSlug("2. 凸性、平滑性与曲率")).toBe("2-凸性-平滑性与曲率");
  });

  it("跨讲链接保留章节锚点而不是只打开讲次", () => {
    const href = resolveNoteHref(
      "Lecture-05_Gradient-Descent.md#2. 凸性、平滑性与曲率",
    );
    expect(href).toMatch(/^#lecture\/5\/2-/);
    expect(decodeURIComponent(href!.split("/").at(-1)!)).toBe(
      "2-凸性-平滑性与曲率",
    );
  });

  it("本讲内部锚点跳转到当前讲，外部链接保持不变", () => {
    const local = resolveNoteHref("#常见误解", 5);
    expect(local).toBe("#lecture/5/" + encodeURIComponent("常见误解"));
    expect(resolveNoteHref("https://example.com/reference#method", 5)).toBe(
      "https://example.com/reference#method",
    );
  });
});
