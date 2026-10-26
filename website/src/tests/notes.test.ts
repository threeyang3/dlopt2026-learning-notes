import { describe, expect, it } from "vitest";
import { buildHeadingIndex, headingSlug, resolveNoteHref } from "../components/Notes";
import { noteText } from "../content/course";

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

  it("每篇笔记的标题 ID 均唯一，LoRA 与 LoRA+ 不冲突", () => {
    const markdown = [
      "# 第一章",
      "### LoRA",
      "### LoRA+",
      "### LoRA",
      "### LoRA-2",
      "```md",
      "### 代码块不应成为标题",
      "```",
    ].join("\\n");
    const headings = buildHeadingIndex(markdown);
    expect(headings.map((heading) => heading.id)).toEqual([
      "第一章",
      "lora",
      "lora-2",
      "lora-3",
      "lora-2-2",
    ]);
    expect(new Set(headings.map((heading) => heading.id)).size).toBe(
      headings.length,
    );

    const original = buildHeadingIndex(
      noteText("Lecture-14_Efficient-Fine-Tuning-and-Inference"),
    );
    expect(original.find((heading) => heading.title === "LoRA")?.id).toBe(
      "lora",
    );
    expect(original.find((heading) => heading.title === "LoRA+")?.id).toBe(
      "lora-2",
    );
  });

  it("本讲及跨讲链接都使用与标题渲染相同的唯一 ID", () => {
    const path = "Lecture-14_Efficient-Fine-Tuning-and-Inference.md";
    expect(resolveNoteHref(path + "#LoRA")).toBe("#lecture/14/lora");
    expect(resolveNoteHref(path + "#LoRA+")).toBe("#lecture/14/lora-2");
    expect(resolveNoteHref(path + "#LoRA%2B")).toBe("#lecture/14/lora-2");
    expect(resolveNoteHref("#LoRA+", 14)).toBe("#lecture/14/lora-2");
    expect(resolveNoteHref("#lora-2", 14)).toBe("#lecture/14/lora-2");
  });

});
