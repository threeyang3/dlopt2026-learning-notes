import { cpSync, mkdirSync, realpathSync, statSync } from "node:fs";
import { resolve, dirname } from "node:path";
import course from "../src/content/course.json";

// Explicit tracked publication policy. Adding a root Markdown file does not publish it.
export const publishedResources: readonly string[] = Object.freeze([
  ...new Set([
    ...course.flatMap((lecture) => [lecture.note, ...lecture.pdfs]),
    "notes/README.md",
    "notes/COURSE_SCOPE.md",
    "notes/NOTE_STANDARD.md",
    "notes/REFERENCES.md",
    "README.md",
    "SOURCE_AUDIT.md",
    "CHAPTER_AUDIT.md",
    "RENDER_AUDIT.md",
    "REVISION_NOTES.md",
    "resource_inventory.md",
    "source_manifest.tsv",
    "checksums.sha256",
    "website/README.md",
    "website/DELIVERY.md",
    "website/VERIFICATION.md",
    "website/ARCHITECTURE.md",
  ]),
]);
export function resourcePath(
  root: string,
  request: string,
): string | undefined {
  try {
    const base = realpathSync(root);
    const path = resolve(
      base,
      decodeURIComponent(request.split("?")[0].replace(/^\//, "")),
    );
    if (!publishedResources.some((name) => resolve(base, name) === path))
      return;
    // Do not follow a listed path replaced by an escaping symlink.
    if (realpathSync(path) !== path || !statSync(path).isFile()) return;
    return path;
  } catch {
    return;
  }
}
export function copyResources(root: string, output: string) {
  for (const name of publishedResources) {
    const source = resourcePath(root, name);
    if (!source) throw Error(`发布资源缺失或不是常规文件：${name}`);
    const target = resolve(output, name);
    mkdirSync(dirname(target), { recursive: true });
    cpSync(source, target);
  }
}
