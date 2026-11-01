import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { renderToStaticMarkup } from "react-dom/server";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import katex from "katex";
import { MathFormula } from "../components/Math";
import { Notes } from "../components/Notes";
import { mathOptions } from "../security/math-options";

describe("安全渲染与依赖一致性", () => {
  it("所有 KaTeX 副本与直接指定的修补版本一致", () => {
    const pkg = JSON.parse(readFileSync("package.json", "utf8"));
    const lock = JSON.parse(readFileSync("package-lock.json", "utf8"));
    const entries = Object.entries(lock.packages).filter(([path]) =>
      path.endsWith("/katex"),
    );
    expect(entries.length).toBeGreaterThan(0);
    for (const [, value] of entries)
      expect((value as { version: string }).version).toBe(
        pkg.dependencies.katex,
      );
    expect(pkg.overrides.katex).toBe("$katex");
  });
  it("公式组件拒绝可信外链、图片和 HTML 命令", () => {
    const expressions = [
      String.raw`\href{https://example.invalid/blocked}{x}`,
      String.raw`\includegraphics{https://example.invalid/blocked.png}`,
      String.raw`\htmlClass{injected-class}{x}`,
    ];
    for (const tex of expressions) {
      const html = renderToStaticMarkup(<MathFormula tex={tex} />);
      expect(html).not.toMatch(/<a\b|<img\b|class="injected-class"/);
    }
  });
  it("不接受从 options 原型继承的 trust 授权", () => {
    const inherited = Object.create({ trust: true });
    const html = katex.renderToString(
      String.raw`\href{https://example.invalid/blocked}{x}`,
      inherited,
    );
    expect(html).not.toMatch(/<a\b/);
  });
  it("共享配置明确关闭信任并限制展开和尺寸", () => {
    expect(mathOptions.trust).toBe(false);
    expect(mathOptions.maxExpand).toBe(1000);
    expect(mathOptions.maxSize).toBe(20);
  });
  it("Markdown 数学渲染禁用可信命令与原始 HTML", () => {
    const text =
      String.raw`$\href{https://example.invalid/blocked}{x}$` +
      '\n\n<b id="raw-html">raw</b>';
    const html = renderToStaticMarkup(
      <ReactMarkdown
        skipHtml
        remarkPlugins={[remarkMath]}
        rehypePlugins={[[rehypeKatex, mathOptions]]}
      >
        {text}
      </ReactMarkdown>,
    );
    expect(html).not.toMatch(/<a\b|id="raw-html"/);
  });
  it("新渲染器可渲染全部 14 讲公式", () => {
    for (let id = 1; id <= 14; id++) {
      const html = renderToStaticMarkup(<Notes id={id} />);
      expect(html).toContain('class="katex"');
      expect(html).not.toContain('class="katex-error"');
    }
  });
});
