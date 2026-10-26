import { useMemo, useState } from "react";
import { multiply, loraCount, kvBytes } from "../engines/matrix";
import { seeded } from "../engines/optimization";
import { Slider, Playback, usePlayback, fmt } from "../components/Controls";
import { MatrixView } from "../components/Matrix";
import { MathFormula } from "../components/Math";
import { Quiz } from "../components/Quiz";
import { Sources } from "../components/Sources";
export function InferenceLab() {
  const [rank, setRank] = useState(1),
    [alpha, setAlpha] = useState(2),
    [batch, setBatch] = useState(1),
    [tokens, setTokens] = useState(1024),
    [layers, setLayers] = useState(32),
    [heads, setHeads] = useState(8),
    [width, setWidth] = useState(128),
    [bytes, setBytes] = useState(2);
  const d = useMemo(() => {
    const rng = seeded(42),
      matrix = (m: number, n: number) =>
        Array.from({ length: m }, () =>
          Array.from({ length: n }, () => rng() - 0.5),
        ),
      W = matrix(4, 3),
      A = matrix(rank, 3),
      B = matrix(4, rank),
      BA = multiply(B, A).map((row) => row.map((x) => (alpha / rank) * x)),
      adapted = W.map((row, i) => row.map((x, j) => x + BA[i][j]));
    return { W, A, B, BA, adapted };
  }, [rank, alpha]);
  const player = usePlayback(tokens),
    length = player.index,
    memory = kvBytes(batch, length, layers, heads, width, bytes);
  return (
    <>
      <div className="lab-intro">
        <p>Lecture 14</p>
        <h1>LoRA 与 KV Cache</h1>
        <p>
          分开计算训练中的低秩增量与推理中的历史缓存。参数数量是解析计数，缓存内存是明确假设下的估算。
        </p>
      </div>
      <div className="lab-body">
        <h2>LoRA：4×3 权重的低秩增量</h2>
        <div className="two-column">
          <Slider
            label="LoRA 秩 r"
            min={1}
            max={3}
            value={rank}
            onChange={setRank}
          />
          <Slider
            label="LoRA α"
            min={1}
            max={12}
            value={alpha}
            onChange={setAlpha}
          />
        </div>
        <div className="matrix-group">
          <MatrixView name="冻结 W" data={d.W} />
          <MatrixView name="A" data={d.A} />
          <MatrixView name="B" data={d.B} />
          <MatrixView name="α/r BA" data={d.BA} />
          <MatrixView name="W′" data={d.adapted} />
        </div>
        <MathFormula
          tex={String.raw`W'=W+\frac\alpha r BA,\quad B\in\mathbb R^{4\times r},\quad A\in\mathbb R^{r\times3}`}
        />
        <p>
          完整权重 12 个元素；可训练因子 {loraCount(4, 3, rank)} 个，比例{" "}
          {fmt(loraCount(4, 3, rank) / 12)}。在小矩阵与较高秩时，LoRA
          不一定更省参数。示例展示非零因子合成，并未训练模型；常见初始化 B=0
          会使初始增量为零。
        </p>
        <p>
          冻结权重仍需保存。主要节省梯度与优化器状态，激活和前向计算不自动消失；可合并性与量化、精度配方有关。
        </p>
      </div>
      <section className="lab-body">
        <h2>逐 token 增长的 K/V 缓存</h2>
        <div className="two-column">
          <div>
            <Slider
              label="Batch size"
              min={1}
              max={16}
              value={batch}
              onChange={setBatch}
            />
            <Slider
              label="最大生成长度"
              min={32}
              max={8192}
              step={32}
              value={tokens}
              onChange={(n) => {
                player.reset();
                setTokens(n);
              }}
            />
            <Slider
              label="层数"
              min={1}
              max={80}
              value={layers}
              onChange={setLayers}
            />
          </div>
          <div>
            <Slider
              label="KV 头数"
              min={1}
              max={32}
              value={heads}
              onChange={setHeads}
            />
            <Slider
              label="每头宽度"
              min={16}
              max={256}
              step={16}
              value={width}
              onChange={setWidth}
            />
            <label>
              每元素字节
              <select value={bytes} onChange={(e) => setBytes(+e.target.value)}>
                <option value={4}>FP32 · 4 bytes</option>
                <option value={2}>FP16 / BF16 · 2 bytes</option>
                <option value={1}>8-bit · 1 byte（未计 scale）</option>
              </select>
            </label>
          </div>
        </div>
        <Playback player={player} max={tokens} />
        <MathFormula
          tex={String.raw`\mathrm{KV\ bytes}=2\,b\,s\,l\,n_{\mathrm{kv}}\,d_{\mathrm{head}}\,\mathrm{bytes/element}`}
        />
        <p className="stat-line">
          每层 K 与 V 各为 [{batch}, {heads}, {length}, {width}]；层数 {layers}
          。当前 {memory.toLocaleString()} bytes = {fmt(memory / 2 ** 30)}{" "}
          GiB；长度上限时{" "}
          {fmt(kvBytes(batch, tokens, layers, heads, width, bytes) / 2 ** 30)}{" "}
          GiB。
        </p>
        <svg
          viewBox="0 0 600 100"
          role="img"
          aria-label="缓存内存随 token 线性增长"
        >
          <rect x="10" y="20" width="570" height="26" fill="var(--soft)" />
          <rect
            x="10"
            y="20"
            width={(570 * length) / tokens}
            height="26"
            fill="#2167ae"
          />
          <text x="10" y="75" fill="var(--ink)" fontSize="13">
            {length} / {tokens} tokens · 每 token{" "}
            {kvBytes(batch, 1, layers, heads, width, bytes).toLocaleString()}{" "}
            bytes
          </text>
        </svg>
        <p>
          假设所有层使用相同 KV 头数和头宽、K/V 同精度，只计有效
          token。不含分页空闲、元数据、临时激活、量化 scale 与模型权重。MHA 可令
          KV 总宽度等于隐藏宽度；GQA/MQA 按实际 KV 头数计，MLA
          则需不同缓存表示。这里生成的是缓存计数，未执行语言模型推理。
        </p>
      </section>
      <Quiz
        questions={[
          {
            question: "序列长度翻倍时，该模型的 KV 有效元素内存如何变化？",
            options: ["翻倍（其他配置固定）", "平方增长", "不变"],
            answer: 0,
            explanation:
              "缓存各 token 的 K、V，元素数线性于序列长度；显式 attention 权重矩阵的平方增长是另一个账本。",
          },
        ]}
      />
      <Sources ids={[14]} />
    </>
  );
}
