import { it, expect } from "vitest";
import {
  mkdtempSync,
  mkdirSync,
  writeFileSync,
  existsSync,
  readFileSync,
  rmSync,
  symlinkSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { resolve, dirname } from "node:path";
import {
  copyResources,
  publishedResources,
  resourcePath,
} from "../../scripts/resources";
it("同一白名单控制开发访问和构建复制，不自动发布新增 Markdown 或 PDF", () => {
  const base = mkdtempSync(resolve(tmpdir(), "dlopt-publish-")),
    root = resolve(base, "repo"),
    out = resolve(base, "dist");
  try {
    for (const name of [
      ...publishedResources,
      "private-draft.md",
      "notes/private.md",
      "materials/private.pdf",
      "website/package.json",
    ]) {
      const p = resolve(root, name);
      mkdirSync(dirname(p), { recursive: true });
      writeFileSync(p, name);
    }
    copyResources(root, out);
    for (const name of publishedResources) {
      expect(resourcePath(root, "/" + name + "?download=1")).toBeDefined();
      expect(readFileSync(resolve(out, name), "utf8")).toBe(name);
    }
    for (const name of [
      "private-draft.md",
      "notes/private.md",
      "materials/private.pdf",
      "website/package.json",
    ]) {
      expect(resourcePath(root, name)).toBeUndefined();
      expect(existsSync(resolve(out, name))).toBe(false);
    }
    expect(resourcePath(root, "../README.md")).toBeUndefined();
    expect(resourcePath(root, "%E0%A4%A")).toBeUndefined();
    const escaped = resolve(base, "secret.md");
    writeFileSync(escaped, "secret");
    rmSync(resolve(root, "README.md"));
    symlinkSync(escaped, resolve(root, "README.md"));
    expect(resourcePath(root, "README.md")).toBeUndefined();
    expect(() => copyResources(root, out)).toThrow("发布资源缺失");
  } finally {
    rmSync(base, { recursive: true, force: true });
  }
});
