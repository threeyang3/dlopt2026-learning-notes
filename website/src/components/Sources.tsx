import { course, resource } from "../content/course";
import { sourceReview } from "../content/source-review";
export function Sources({ ids }: { ids: number[] }) {
  return (
    <section className="sources">
      <h2>回到原始资料</h2>
      {ids.map((id) => {
        const l = course.find((x) => x.id === id)!;
        return (
          <details key={id}>
            <summary>
              Lecture {id} · {l.english} · partially_verified
            </summary>
            <p>{l.evidence}</p>
            <a href={resource(l.note)}>原始 Markdown</a>
            {sourceReview[id] && (
              <ul>
                {sourceReview[id].map((r) => (
                  <li key={r.path + r.page}>
                    <a href={resource(r.path) + "#page=" + r.page}>
                      {r.topic} · PDF 页序 {r.page}
                    </a>
                    （2026-10-26，{r.method}）
                  </li>
                ))}
              </ul>
            )}
            <ul>
              {l.pdfs.map((p) => (
                <li key={p}>
                  <a href={resource(p)}>{p.split("/").pop()}</a>
                  （其他页码见原笔记；仅上述抽查页在本轮核对，不代表整份 PDF
                  已核验）
                </li>
              ))}
            </ul>
            <a href={"#lecture/" + id}>阅读全文与原笔记参考文献</a>
          </details>
        );
      })}
      <p>
        <a href={resource("notes/REFERENCES.md")}>论文与官方文档索引</a> ·{" "}
        <a href={resource("SOURCE_AUDIT.md")}>既有逐式审校记录</a>
      </p>
    </section>
  );
}
