import { useState } from "react";
export interface Question {
  question: string;
  options: string[];
  answer: number;
  explanation: string;
}
export function Quiz({ questions }: { questions: Question[] }) {
  const [choices, setChoices] = useState<Record<number, number>>({}),
    [shown, setShown] = useState<Record<number, boolean>>({});
  return (
    <section>
      <h2>先预测，再实验</h2>
      {questions.map((q, i) => (
        <fieldset className="quiz" key={i}>
          <legend>
            {i + 1}. {q.question}
          </legend>
          {q.options.map((o, j) => (
            <label key={o}>
              <input
                type="radio"
                name={"quiz" + i}
                checked={choices[i] === j}
                onChange={() => {
                  setChoices({ ...choices, [i]: j });
                  setShown({ ...shown, [i]: false });
                }}
              />
              {o}
            </label>
          ))}
          <button
            disabled={choices[i] === undefined}
            onClick={() => setShown({ ...shown, [i]: true })}
          >
            检查答案 {i + 1}
          </button>
          {shown[i] && (
            <p role="status">
              {choices[i] === q.answer ? "回答正确。" : "再想一想。"}
              {q.explanation}
            </p>
          )}
        </fieldset>
      ))}
    </section>
  );
}
