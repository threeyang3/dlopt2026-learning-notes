import {
  createElement,
  isValidElement,
  type HTMLAttributes,
  type ReactNode,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";
import { mathOptions } from "../security/math-options";
import { course, noteText, resource } from "../content/course";
import supplements from "../content/supplements.json";

type HeadingTag = "h1" | "h2" | "h3" | "h4" | "h5" | "h6";

export interface NoteHeading {
  line: number;
  title: string;
  id: string;
}

function nodeText(value: ReactNode): string {
  if (typeof value === "string" || typeof value === "number")
    return String(value);
  if (Array.isArray(value)) return value.map(nodeText).join("");
  if (isValidElement<{ children?: ReactNode }>(value))
    return nodeText(value.props.children);
  return "";
}

export function headingSlug(value: string) {
  return (
    value
      .normalize("NFKC")
      .trim()
      .toLowerCase()
      .replace(/[`*_~]/g, "")
      .replace(/[^\p{L}\p{N}]+/gu, "-")
      .replace(/^-+|-+$/g, "") || "section"
  );
}

function headingTitle(value: string) {
  return value
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/[ \t]+#+[ \t]*$/, "")
    .replace(/[`*_~]/g, "")
    .trim();
}

// A shared, deterministic heading index is used both for rendered IDs and
// incoming links. Source-line keys avoid mutable counters during React renders.
export function buildHeadingIndex(markdown: string): NoteHeading[] {
  const headings: NoteHeading[] = [];
  const used = new Set<string>();
  const nextSuffix = new Map<string, number>();
  let fence: { marker: string; length: number } | undefined;
  markdown.split(/\r?\n/).forEach((line, index) => {
    const boundary = /^ {0,3}(`{3,}|~{3,})/.exec(line);
    if (boundary) {
      const marker = boundary[1][0];
      if (!fence) fence = { marker, length: boundary[1].length };
      else if (fence.marker === marker && boundary[1].length >= fence.length)
        fence = undefined;
      return;
    }
    if (fence) return;
    const match = /^ {0,3}#{1,6}[ \t]+(.+?)[ \t]*$/.exec(line);
    if (!match) return;
    const title = headingTitle(match[1]);
    const base = headingSlug(title);
    let ordinal = nextSuffix.get(base) ?? 1;
    let id = ordinal === 1 ? base : `${base}-${ordinal}`;
    while (used.has(id)) {
      ordinal += 1;
      id = `${base}-${ordinal}`;
    }
    used.add(id);
    nextSuffix.set(base, ordinal + 1);
    headings.push({ line: index + 1, title, id });
  });
  return headings;
}

function decodeFragment(value: string) {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function documentHref(slug: string, anchor?: string) {
  const lecture = course.find((entry) => entry.slug === slug);
  const supplement = supplements.find((entry) => entry.slug === slug);
  const base = lecture
    ? "#lecture/" + lecture.id
    : supplement
      ? "#supplement/" + supplement.id
      : undefined;
  if (!base) return undefined;
  if (!anchor) return base;
  const name = decodeFragment(anchor).trim();
  const headings = buildHeadingIndex(noteText(slug));
  const idForTitle = headings.find((entry) => entry.title === name)?.id;
  const idForExistingSlug = headings.find((entry) => entry.id === name)?.id;
  const idForSlug = headings.find(
    (entry) => headingSlug(entry.title) === headingSlug(name),
  )?.id;
  return (
    base +
    "/" +
    encodeURIComponent(
      idForTitle ?? idForExistingSlug ?? idForSlug ?? headingSlug(name),
    )
  );
}
function lectureHref(id: number, anchor?: string) {
  const lecture = course.find((entry) => entry.id === id);
  return lecture ? documentHref(lecture.slug, anchor) : undefined;
}

function headingRenderer(tag: HeadingTag, byLine: Map<number, string>) {
  return ({
    children,
    node,
    ...props
  }: HTMLAttributes<HTMLHeadingElement> & {
    children?: ReactNode;
    node?: { position?: { start?: { line?: number } } };
  }) =>
    createElement(
      tag,
      {
        ...props,
        id:
          byLine.get(node?.position?.start?.line ?? -1) ??
          headingSlug(nodeText(children)),
      },
      children,
    );
}

export function Notes({ id }: { id: number }) {
  const lecture = course.find((l) => l.id === id)!;
  return <NoteDocument slug={lecture.slug} lectureId={id} />;
}
export function SupplementNotes({ id }: { id: string }) {
  const entry = supplements.find((s) => s.id === id);
  return entry ? <NoteDocument slug={entry.slug} /> : null;
}
function NoteDocument({
  slug,
  lectureId,
}: {
  slug: string;
  lectureId?: number;
}) {
  const text = noteText(slug).replace(
    /\[\[([^\]|#]+)(?:#([^\]|]+))?(?:\|([^\]]+))?\]\]/g,
    (_, name, anchor, label) => {
      const target =
        documentHref(name, anchor) ??
        name + ".md" + (anchor ? "#" + encodeURIComponent(anchor) : "");
      return `[${label || anchor || name}](${target})`;
    },
  );
  const byLine = new Map(
    buildHeadingIndex(text).map((heading) => [heading.line, heading.id]),
  );
  const renderHeading = (tag: HeadingTag) => headingRenderer(tag, byLine);
  return (
    <article className="note-content">
      <ReactMarkdown
        remarkPlugins={[remarkMath, remarkGfm]}
        skipHtml
        rehypePlugins={[[rehypeKatex, mathOptions]]}
        components={{
          h1: renderHeading("h1"),
          h2: renderHeading("h2"),
          h3: renderHeading("h3"),
          h4: renderHeading("h4"),
          h5: renderHeading("h5"),
          h6: renderHeading("h6"),
          a: ({ href, children }) => (
            <a href={resolveNoteHref(href, lectureId, slug)}>{children}</a>
          ),
        }}
      >
        {text}
      </ReactMarkdown>
    </article>
  );
}

export function resolveNoteHref(
  href?: string,
  currentLectureId?: number,
  currentSlug?: string,
) {
  if (!href) return undefined;
  if (/^(https?:|mailto:|#lecture\/|#supplement\/)/.test(href)) return href;

  const hashIndex = href.indexOf("#");
  const clean = hashIndex >= 0 ? href.slice(0, hashIndex) : href;
  const fragment =
    hashIndex >= 0 ? decodeFragment(href.slice(hashIndex + 1)) : "";

  if (!clean && currentLectureId && fragment)
    return lectureHref(currentLectureId, fragment);

  if (!clean && currentSlug && fragment)
    return documentHref(currentSlug, fragment);
  const document = [...course, ...supplements].find(
    (l) => clean === l.slug + ".md" || clean.endsWith("/" + l.slug + ".md"),
  );
  if (document) return documentHref(document.slug, fragment || undefined);

  const path = new URL(href, "https://course.invalid/notes/current.md");
  return resource(path.pathname.slice(1)) + path.hash;
}
