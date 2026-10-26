import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";
import { course, noteText, resource } from "../content/course";
export function Notes({ id }: { id: number }) {
  const lecture = course.find((l) => l.id === id)!;
  const text = noteText(lecture.slug).replace(
    /\[\[([^\]|#]+)(?:#([^\]|]+))?(?:\|([^\]]+))?\]\]/g,
    (_, name, _anchor, label) => {
      const other = course.find((l) => l.slug === name);
      return `[${label || name}](${other ? "#lecture/" + other.id : name + ".md"})`;
    },
  );
  return (
    <article className="note-content">
      <ReactMarkdown
        remarkPlugins={[remarkMath, remarkGfm]}
        rehypePlugins={[[rehypeKatex, { strict: "ignore" }]]}
        components={{
          a: ({ href, children }) => (
            <a href={resolveNoteHref(href)}>{children}</a>
          ),
        }}
      >
        {text}
      </ReactMarkdown>
    </article>
  );
}

export function resolveNoteHref(href?: string) {
  if (!href) return undefined;
  if (/^(https?:|mailto:|#lecture\/)/.test(href)) return href;
  const clean = href.split("#")[0];
  const lecture = course.find((l) => clean === l.slug + ".md");
  if (lecture) return "#lecture/" + lecture.id;
  const path = new URL(href, "https://course.invalid/notes/current.md");
  return resource(path.pathname.slice(1)) + path.hash;
}
