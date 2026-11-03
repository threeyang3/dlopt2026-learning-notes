import { useMemo, useRef, useState } from "react";
import {
  Slider,
  Playback,
  usePlayback,
  fmt,
  vector,
} from "../components/Controls";
import { MatrixView } from "../components/Matrix";
import { MathFormula } from "../components/Math";
import { Quiz } from "../components/Quiz";
import { SupplementSources } from "../components/Supplements";
import { resource } from "../content/course";
import {
  embeddingWords,
  embeddingAxes,
  slideEmbedding,
  embeddingColumn,
  cosine,
  toyCorpus,
  buildNgram,
  ngramDistribution,
  generateNgram,
  categoricalNll,
  makeRnnWeights,
  rnnVocabulary,
  rnnForward,
  trainRnnStep,
  scalarMemory,
} from "../engines/language-models";

const pdf = "materials/Supplement-Language-Models/03_langmodel.pdf";
const panels = ["嵌入查表", "n-gram 计数与生成", "RNN 前向与训练", "梯度传播"];
export function LanguageModelsLab() {
  const [active, setActive] = useState(0);
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);
  return (
    <>
      <div className="lab-intro">
        <p>补充单元 · 建议在第 7 讲之前学习</p>
        <h1>语言模型：从词向量到循环状态</h1>
        <p>
          依据 56 页《Language
          Models》课件。以下是可复算的小型教学模型，不是预训练
          LLM。切换面板会重置该面板的实验。
        </p>
        <a className="button" href="#supplement/language-models">
          阅读完整补充笔记
        </a>{" "}
        <a href={resource(pdf)}>打开原始 PDF</a>
      </div>
      <div className="lm-tabs" role="tablist" aria-label="语言模型实验主题">
        {panels.map((title, i) => (
          <button
            key={title}
            role="tab"
            id={`lm-tab-${i}`}
            aria-selected={active === i}
            aria-controls="lm-panel"
            tabIndex={active === i ? 0 : -1}
            ref={(el) => {
              buttons.current[i] = el;
            }}
            onClick={() => setActive(i)}
            onKeyDown={(event) => {
              let next = i;
              if (event.key === "ArrowRight") next = (i + 1) % panels.length;
              else if (event.key === "ArrowLeft")
                next = (i + panels.length - 1) % panels.length;
              else if (event.key === "Home") next = 0;
              else if (event.key === "End") next = panels.length - 1;
              else return;
              event.preventDefault();
              setActive(next);
              buttons.current[next]?.focus();
            }}
          >
            {title}
          </button>
        ))}
      </div>
      <section
        id="lm-panel"
        role="tabpanel"
        aria-labelledby={`lm-tab-${active}`}
        className="lab-body lm-panel"
      >
        {active === 0 && <EmbeddingPanel />}
        {active === 1 && <NgramPanel />}
        {active === 2 && <RnnPanel />}
        {active === 3 && <GradientPanel />}
      </section>
      <Quiz
        questions={[
          {
            question: "4-gram 的条件历史含几个词？",
            options: ["4 个", "3 个", "全部前缀"],
            answer: 1,
            explanation:
              "n 包含待预测词，所以历史为 n−1。PDF 第 17 页的条件是 students opened their。",
          },
          {
            question: "播放 RNN 时间步会改变权重吗？",
            options: ["每步自动训练", "不会，只有训练按钮更新参数"],
            answer: 1,
            explanation:
              "前向递推使用同一组共享参数。训练另行通过平均交叉熵和 BPTT 计算梯度。",
          },
          {
            question:
              "RNN 参数数目不随序列长度增长，是否意味着训练内存不增长？",
            options: ["是", "否，保存的激活仍可能增长"],
            answer: 1,
            explanation: "参数共享不消除每个时间步的状态及反向传播中间量。",
          },
        ]}
      />
      <SupplementSources id="language-models" />
    </>
  );
}
function EmbeddingPanel() {
  const [first, setFirst] = useState(4),
    [second, setSecond] = useState(5);
  const e = embeddingColumn(first),
    other = embeddingColumn(second);
  const oneHot = embeddingWords.map((_, i) => [i === first ? 1 : 0]);
  return (
    <>
      <h2>嵌入矩阵 × one-hot = 取出对应列</h2>
      <p>
        PDF p.4–10；下面的 4×6 数字来自第 6
        页人工示意表，不是训练结果。坐标依次为 {embeddingAxes.join("、")}。
      </p>
      <div className="lm-controls">
        <label>
          观察词{" "}
          <select
            aria-label="观察词"
            value={first}
            onChange={(e) => setFirst(+e.target.value)}
          >
            {embeddingWords.map((w, i) => (
              <option key={w} value={i}>
                {w}
              </option>
            ))}
          </select>
        </label>
        <label>
          比较词{" "}
          <select
            aria-label="比较词"
            value={second}
            onChange={(e) => setSecond(+e.target.value)}
          >
            {embeddingWords.map((w, i) => (
              <option key={w} value={i}>
                {w}
              </option>
            ))}
          </select>
        </label>
      </div>
      <MathFormula
        tex={String.raw`E\in\mathbb R^{4\times6},\quad e_i=Eo_i=E_{:,i}`}
      />
      <div className="lm-scroll">
        <MatrixView name="E（课件示意）" data={slideEmbedding} />
      </div>
      <div className="two-column">
        <MatrixView name="one-hot o" data={oneHot} />
        <div data-testid="embedding-result" data-vector={JSON.stringify(e)}>
          <MatrixView name="查表结果 e" data={e.map((v) => [v])} />
          <p>one-hot 余弦：{first === second ? "1" : "0"}</p>
          <p>
            嵌入余弦：
            <output data-testid="embedding-cosine">
              {fmt(cosine(e, other)!)}
            </output>
          </p>
        </div>
      </div>
      <p className="notice">
        词向量相似度依赖度量与具体数值，不是同义关系的证明。类比和第 9
        页符号勘误见补充笔记；此处不训练 embedding。
      </p>
    </>
  );
}
const contexts = [
  ["the", "students", "opened", "their"],
  ["i", "like", "orange"],
  ["books", "i", "their"],
];
function NgramPanel() {
  const [order, setOrder] = useState(4),
    [alpha, setAlpha] = useState(0),
    [context, setContext] = useState(0),
    [seed, setSeed] = useState(42),
    [steps, setSteps] = useState(0);
  const model = useMemo(() => buildNgram(toyCorpus, order), [order]);
  const dist = ngramDistribution(model, contexts[context], alpha);
  const generated = generateNgram(model, contexts[context], alpha, seed, steps);
  function change(fn: () => void) {
    setSteps(0);
    fn();
  }
  return (
    <>
      <h2>用真实计数得到下一个词分布</h2>
      <p>
        课件 p.15–23。使用 12 句自建语料，非原始实验数据；四类后继计数为
        4/1/3/2，比例对应课件 books=0.4、exams=0.1。add-α 是明确标注的教学补充。
      </p>
      <details>
        <summary>查看全部教学语料与边界约定</summary>
        <ol>
          {toyCorpus.map((s, i) => (
            <li key={i}>{s.join(" ")}</li>
          ))}
        </ol>
        <p>
          各句独立添加 n−1 个 BOS 和一个 EOS，不跨句拼接。BOS 不属于输出词表。
        </p>
      </details>
      <div className="two-column">
        <div>
          <Slider
            label="n-gram 阶数 n"
            min={1}
            max={4}
            value={order}
            onChange={(v) => change(() => setOrder(v))}
          />
          <Slider
            label="平滑 α（补充）"
            min={0}
            max={1}
            step={0.1}
            value={alpha}
            onChange={(v) => change(() => setAlpha(v))}
          />
          <Slider
            label="生成随机种子"
            min={1}
            max={100}
            value={seed}
            onChange={(v) => change(() => setSeed(v))}
          />
        </div>
        <div>
          <label>
            上下文{" "}
            <select
              aria-label="n-gram 上下文"
              value={context}
              onChange={(e) => change(() => setContext(+e.target.value))}
            >
              {contexts.map((c, i) => (
                <option key={i} value={i}>
                  {c.join(" ")}
                  {i === 2 ? "（4-gram 未见）" : ""}
                </option>
              ))}
            </select>
          </label>
          <p>
            实际条件历史：
            <code>
              {order === 1
                ? "∅（unigram）"
                : contexts[context].slice(-(order - 1)).join(" ")}
            </code>
          </p>
          <p>
            上下文计数 C(c) = {dist.total}；词表 V = {model.vocabulary.length}
            ；分母 = {fmt(dist.denominator)}
          </p>
        </div>
      </div>
      <MathFormula
        tex={String.raw`\widehat p_\alpha(w\mid c)=\frac{C(c,w)+\alpha}{C(c)+\alpha V}`}
      />
      {!dist.probabilities && (
        <p role="status">
          未见上下文且 α=0：条件分布未定义，不能采样。可显式开启平滑。
        </p>
      )}
      <div
        className="lm-scroll"
        data-testid="ngram-distribution"
        data-probabilities={JSON.stringify(dist.probabilities)}
      >
        <table>
          <caption>固定初始上下文的计数与概率（不是生成后的上下文）</caption>
          <thead>
            <tr>
              <th>后继词</th>
              <th>计数</th>
              <th>概率</th>
              <th>该词为目标时 NLL</th>
            </tr>
          </thead>
          <tbody>
            {model.vocabulary.map((word, i) => (
              <tr key={word}>
                <th scope="row">{word}</th>
                <td>{dist.counts[i]}</td>
                <td>
                  {dist.probabilities ? fmt(dist.probabilities[i]) : "未定义"}
                </td>
                <td>
                  {dist.probabilities
                    ? fmt(categoricalNll(dist.probabilities[i]))
                    : "未定义"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="lm-controls">
        <button
          disabled={
            steps >= 8 || generated.reason !== "limit" || !dist.probabilities
          }
          onClick={() => setSteps((n) => n + 1)}
        >
          采样下一个词
        </button>
        <button onClick={() => setSteps(0)}>重置生成</button>
      </div>
      <p data-testid="ngram-generated">{generated.tokens.join(" ")}</p>
      <p aria-live="polite">
        停止原因：
        {generated.reason === "eos"
          ? "遇到 EOS"
          : generated.reason === "unseen"
            ? "生成遇到未定义上下文"
            : steps >= 8
              ? "达到 8 步上限"
              : "可继续采样"}
      </p>
    </>
  );
}
function RnnPanel() {
  const [seed, setSeed] = useState(42),
    [weights, setWeights] = useState(() => makeRnnWeights(42)),
    [ending, setEnding] = useState(4),
    [rate, setRate] = useState(0.1),
    [trained, setTrained] = useState(0);
  const inputs = [0, 1, 2, 3, ending],
    targets = [1, 2, 3, ending, 6];
  const forward = rnnForward(weights, inputs, targets),
    player = usePlayback(inputs.length - 1);
  const current = forward.states[player.index];
  const initialLoss = rnnForward(makeRnnWeights(seed), inputs, targets).loss;
  function reset(newSeed = seed) {
    setSeed(newSeed);
    setWeights(makeRnnWeights(newSeed));
    setTrained(0);
    player.reset();
  }
  return (
    <>
      <h2>共享权重的 RNN：前向播放与训练分离</h2>
      <p>
        PDF p.26–37。7 个词、3 维 embedding、2 维 tanh 状态，h₀=0；54
        个参数随机初始化。每个时间步预测下一个词，最后预测 EOS。
      </p>
      <div className="two-column">
        <div>
          <Slider
            label="RNN 初始化种子"
            min={1}
            max={100}
            value={seed}
            onChange={reset}
          />
          <Slider
            label="RNN 学习率"
            min={0.01}
            max={0.3}
            step={0.01}
            value={rate}
            onChange={setRate}
          />
        </div>
        <div>
          <label>
            训练句子{" "}
            <select
              aria-label="RNN 训练句子"
              value={ending}
              onChange={(e) => {
                setEnding(+e.target.value);
                reset();
              }}
            >
              <option value={4}>the students opened their books</option>
              <option value={5}>the students opened their exams</option>
            </select>
          </label>
          <p>改变句子或种子重置模型；学习率只影响下一次训练。</p>
        </div>
      </div>
      <Playback player={player} max={inputs.length - 1} />
      <p>
        迭代位置在此表示观察的时间步（从 0
        开始），不表示优化次数。播放不会更改权重。
      </p>
      <ol className="lm-timeline" aria-label="RNN 时间展开">
        {inputs.map((token, i) => (
          <li key={i}>
            <button
              aria-current={player.index === i ? "step" : undefined}
              onClick={() => {
                player.setPlaying(false);
                player.setIndex(i);
              }}
            >
              t={i + 1} · {rnnVocabulary[token]}
            </button>
          </li>
        ))}
      </ol>
      <MathFormula
        tex={String.raw`e_t=Ex_t,\quad h_t=\tanh(W_hh_{t-1}+W_ee_t+b_1),\quad\widehat y_t=\mathrm{softmax}(Uh_t+b_2)`}
      />
      <div
        className="notice"
        data-testid="rnn-state"
        data-hidden={JSON.stringify(current.hidden)}
        data-input={current.input}
        data-target={current.target}
      >
        <p>
          <strong>
            输入 {rnnVocabulary[current.input]} → 目标{" "}
            {rnnVocabulary[current.target]}
          </strong>
        </p>
        <p>
          eₜ = {vector(current.embedding)}；hₜ₋₁ = {vector(current.previous)}
        </p>
        <p>
          递归项 Wₕhₜ₋₁ = {vector(current.recurrent)}；输入项 Wₑeₜ ={" "}
          {vector(current.incoming)}
        </p>
        <p>
          aₜ = {vector(current.preactivation)}；hₜ = {vector(current.hidden)}
        </p>
        <p>当前步交叉熵：{fmt(current.loss)} nat</p>
      </div>
      <div className="lm-scroll">
        <table
          data-testid="rnn-probabilities"
          data-probabilities={JSON.stringify(current.probabilities)}
        >
          <caption>当前步下一个词分布</caption>
          <thead>
            <tr>
              <th>词</th>
              <th>概率</th>
              <th>目标</th>
            </tr>
          </thead>
          <tbody>
            {rnnVocabulary.map((w, i) => (
              <tr key={w}>
                <th scope="row">{w}</th>
                <td>{fmt(current.probabilities[i])}</td>
                <td>{current.target === i ? "是" : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="lm-controls">
        <button
          disabled={trained >= 50}
          onClick={() => {
            setWeights((w) => trainRnnStep(w, inputs, targets, rate));
            setTrained((n) => n + 1);
            player.reset();
          }}
        >
          训练一步（完整 BPTT）
        </button>
        <button onClick={() => reset()}>重置 RNN 参数</button>
      </div>
      <p>
        训练步数：<output data-testid="rnn-training-steps">{trained}</output>
        /50；初始平均损失 {fmt(initialLoss)}；当前平均损失{" "}
        <output data-testid="rnn-loss" data-loss={forward.loss}>
          {fmt(forward.loss)}
        </output>{" "}
        nat。
      </p>
      <p>
        完整 BPTT 与单句梯度下降为教学补充；训练包括 E、Wₕ、Wₑ、U
        和偏置。没有批次、截断 BPTT 或泛化评估，不保证任意步长单调下降。
      </p>
      <details>
        <summary>检查共享参数与形状</summary>
        <div className="lm-scroll">
          <MatrixView name="E" data={weights.E} />
          <MatrixView name="W_h" data={weights.Wh} />
          <MatrixView name="W_e" data={weights.We} />
          <MatrixView name="U" data={weights.U} />
          <p>
            b₁={vector(weights.b1)}；b₂={vector(weights.b2)}
          </p>
        </div>
      </details>
    </>
  );
}
function GradientPanel() {
  const [weight, setWeight] = useState(0.5),
    [initial, setInitial] = useState(0),
    [length, setLength] = useState(10);
  const states = scalarMemory(weight, initial, length),
    final = states[states.length - 1];
  return (
    <>
      <h2>长路径梯度：每一步导数的连乘</h2>
      <p>
        PDF p.41–43 的现象，加上笔记中的标量推导补充。以下是零输入 tanh
        递推，不是完整 RNN 损失的参数梯度。
      </p>
      <Slider
        label="递归权重 w"
        min={-1.8}
        max={1.8}
        step={0.1}
        value={weight}
        onChange={setWeight}
      />
      <Slider
        label="初始隐藏状态 h0"
        min={-0.9}
        max={0.9}
        step={0.1}
        value={initial}
        onChange={setInitial}
      />
      <Slider
        label="传播步数 T"
        min={1}
        max={30}
        value={length}
        onChange={setLength}
      />
      <MathFormula
        tex={String.raw`h_t=\tanh(wh_{t-1}),\qquad\frac{\partial h_T}{\partial h_0}=\prod_{t=1}^{T}w(1-h_t^2)`}
      />
      <p>
        最终状态 {fmt(final.hidden)}；对初态的导数{" "}
        <output data-testid="rnn-sensitivity" data-value={final.derivative}>
          {fmt(final.derivative)}
        </output>
        。
      </p>
      <div className="lm-scroll">
        <table>
          <caption>实际轨迹与连乘，负号也保留</caption>
          <thead>
            <tr>
              <th>t</th>
              <th>hₜ</th>
              <th>局部导数</th>
              <th>对 h₀ 的导数</th>
            </tr>
          </thead>
          <tbody>
            {states.map((s) => (
              <tr key={s.step}>
                <th scope="row">{s.step}</th>
                <td>{fmt(s.hidden)}</td>
                <td>{s.step === 0 ? "初态" : fmt(s.local)}</td>
                <td>{fmt(s.derivative)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="notice">
        h₀=0 时状态保持零，但局部敏感度可按 wᵀ 增长；非零初态可能饱和。不能仅凭
        |w|&gt;1 就断言任意轨迹必然梯度爆炸。双向、多层和 LSTM
        的边界见补充笔记。
      </p>
    </>
  );
}
