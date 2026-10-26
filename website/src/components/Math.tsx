import katex from "katex";
export function MathFormula({
  tex,
  block = true,
}: {
  tex: string;
  block?: boolean;
}) {
  return (
    <span
      className={block ? "formula" : "inline-formula"}
      dangerouslySetInnerHTML={{
        __html: katex.renderToString(tex, {
          displayMode: block,
          throwOnError: true,
          strict: "ignore",
        }),
      }}
    />
  );
}
