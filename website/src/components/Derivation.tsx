import { useState } from "react";
import { MathFormula } from "./Math";
export interface DerivationStep {
  formula: string;
  explanation: string;
  rule: string;
  conditions: string;
  variables: string;
}
export function Derivation({ steps }: { steps: DerivationStep[] }) {
  const [index, setIndex] = useState(0),
    [full, setFull] = useState(false),
    [explain, setExplain] = useState(true);
  return (
    <section className="derivation">
      <div className="section-heading">
        <h2>从公式到这一步</h2>
        <span>
          {index + 1} / {steps.length}
        </span>
      </div>
      <div className="toolbar">
        <button
          disabled={index === 0}
          onClick={() => {
            setFull(false);
            setIndex(index - 1);
          }}
        >
          上一步推导
        </button>
        <button
          disabled={index === steps.length - 1}
          onClick={() => {
            setFull(false);
            setIndex(index + 1);
          }}
        >
          下一步推导
        </button>
        <button aria-pressed={full} onClick={() => setFull(!full)}>
          {full ? "逐步查看" : "完整推导"}
        </button>
        <button aria-pressed={explain} onClick={() => setExplain(!explain)}>
          {explain ? "隐藏说明" : "显示说明"}
        </button>
      </div>
      {steps.map(
        (s, i) =>
          (full || i === index) && (
            <article
              key={i}
              className={i === index ? "proof-step current" : "proof-step"}
            >
              <span className="step-label">
                步骤 {i + 1} · {s.rule}
              </span>
              <MathFormula tex={s.formula} />
              {explain && (
                <>
                  <p>{s.explanation}</p>
                  <p className="muted">
                    条件：{s.conditions}
                    <br />
                    {s.variables}
                  </p>
                </>
              )}
            </article>
          ),
      )}
    </section>
  );
}
