import katex from "katex";
import { mathOptions } from "../security/math-options";
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
          ...mathOptions,
        }),
      }}
    />
  );
}
