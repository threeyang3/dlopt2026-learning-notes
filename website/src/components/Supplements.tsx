import supplements from "../content/supplements.json";
import { resource } from "../content/course";
export function SupplementCatalog() {
  return (
    <section className="supplement-catalog">
      <h2>补充单元</h2>
      <p>不改变原 14 讲编号和进度；建议在 Transformer 之前学习。</p>
      {supplements.map((entry) => (
        <article className="notice" key={entry.id}>
          <h3>{entry.title}</h3>
          <p>{entry.evidence}</p>
          <a className="button" href={"#" + entry.lab}>
            打开语言模型实验室
          </a>{" "}
          <a href={"#supplement/" + entry.id}>阅读补充笔记</a>
          {" · "}
          <a href={resource(entry.pdf)}>原始 PDF（{entry.pages} 页）</a>
        </article>
      ))}
    </section>
  );
}
export function SupplementSources({ id }: { id: string }) {
  const entry = supplements.find((s) => s.id === id);
  if (!entry) return null;
  return (
    <section className="sources">
      <h2>补充材料与核验范围</h2>
      <p>{entry.evidence}</p>
      <p>
        原件 56 页：p.4–10 嵌入；p.11–23 n-gram；p.24–25 固定窗口；p.26–40 RNN
        与训练；p.41–43 梯度；p.44–56 双向与多层。
      </p>
      <p>
        <a href={resource(entry.pdf)}>仓库原始 PDF</a>
        {" · "}
        <a href={entry.sourceUrl}>作者公开版本</a>
        {" · "}
        <a href={resource(entry.note)}>Markdown 笔记</a>
      </p>
      <p>
        状态：{entry.status}。计数、递推、BPTT
        和边界有测试；示意数据不是预训练模型。平滑、BPTT
        与标量敏感度是明确标注的补充。
      </p>
    </section>
  );
}
