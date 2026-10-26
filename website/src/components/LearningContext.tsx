import { modules } from "../content/modules";
export function LearningContext({ lab }: { lab: string }) {
  const m = modules[lab];
  if (!m) return null;
  return (
    <details className="learning-context">
      <summary>学习目标、条件与核验范围 · {m.status}</summary>
      <dl>
        <dt>学习目标</dt>
        <dd>{m.goal}</dd>
        <dt>先修知识</dt>
        <dd>{m.prerequisites}</dd>
        <dt>适用条件</dt>
        <dd>{m.conditions}</dd>
        <dt>建议实验</dt>
        <dd>{m.experiment}</dd>
        <dt>常见误解</dt>
        <dd>{m.misconception}</dd>
        <dt>核验依据</dt>
        <dd>{m.evidence}</dd>
      </dl>
      <p>
        关联课程：
        {m.lectures.map((id) => (
          <a key={id} href={"#lecture/" + id}>
            第 {id} 讲{" "}
          </a>
        ))}
        。verified 仅适用于声明范围，不表示整讲或全部课件已核验。
      </p>
    </details>
  );
}
