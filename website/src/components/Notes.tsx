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
import { course, noteText, resource } from "../content/course";

type HeadingTag = "h1" | "h2" | "h3" | "h4" | "h5" | "h6";

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

function decodeFragment(value: string) {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function lectureHref(id: number, anchor?: string) {
  const base = "#lecture/" + id;
  return anchor
    ? base + "/" + encodeURIComponent(headingSlug(decodeFragment(anchor)))
    : base;
}

function headingRenderer(tag: HeadingTag) {
  return ({
    children,
    node: _node,
    ...props
  }: HTMLAttributes<HTMLHeadingElement> & {
    children?: ReactNode;
    node?: unknown;
  }) =>
    createElement(
      tag,
      { ...props, id: headingSlug(nodeText(children)) },
      children,
    );
}

export function Notes({ id }: { id: number }) {
  const lecture = course.find((l) => l.id === id)!;
  const text = noteText(lecture.slug).replace(
    /\[\[([^\]|#]+)(?:#([^\]|]+))?(?:\|([^\]]+))?\]\]/g,
    (_, name, anchor, label) => {
      const other = course.find((l) => l.slug === name);
      const target = other
        ? lectureHref(other.id, anchor)
        : name + ".md" + (anchor ? "#" + encodeURIComponent(anchor) : "");
      return `[${label || anchor || name}](${target})`;
    },
  );
  return (
    <article className="note-content">
      <ReactMarkdown
        remarkPlugins={[remarkMath, remarkGfm]}
        rehypePlugins={[[rehypeKatex, { strict: "ignore" }]]}
        components={{
          h1: headingRenderer("h1"),
          h2: headingRenderer("h2"),
          h3: headingRenderer("h3"),
          h4: headingRenderer("h4"),
          h5: headingRenderer("h5"),
          h6: headingRenderer("h6"),
          a: ({ href, children }) => (
            <a href={resolveNoteHref(href, id)}>{children}</a>
          ),
        }}
      >
        {text}
      </ReactMarkdown>
    </article>
  );
}

export function resolveNoteHref(href?: string, currentLectureId?: number) {
  if (!href) return undefined;
  if (/^(https?:|mailto:|#lecture\/)/.test(href)) return href;

  const hashIndex = href.indexOf("#");
  const clean = hashIndex >= 0 ? href.slice(0, hashIndex) : href;
  const fragment =
    hashIndex >= 0 ? decodeFragment(href.slice(hashIndex + 1)) : "";

  if (!clean && currentLectureId && fragment)
    return lectureHref(currentLectureId, fragment);

  const lecture = course.find(
    (l) => clean === l.slug + ".md" || clean.endsWith("/" + l.slug + ".md"),
  );
  if (lecture) return lectureHref(lecture.id, fragment || undefined);

  const path = new URL(href, "https://course.invalid/notes/current.md");
  return resource(path.pathname.slice(1)) + path.hash;
}
