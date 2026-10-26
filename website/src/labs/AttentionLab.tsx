import { useMemo, useState } from "react";
import {
  attention,
  multiply,
  onlineAttention,
  type Mat,
} from "../engines/matrix";
import { seeded } from "../engines/optimization";
import { MatrixView } from "../components/Matrix";
import {
  Slider,
  Playback,
  usePlayback,
  fmt,
  vector,
} from "../components/Controls";
import { MathFormula } from "../components/Math";
import { Quiz } from "../components/Quiz";
import { Sources } from "../components/Sources";
export function AttentionLab({ flash = false }: { flash?: boolean }) {
  const [seed, setSeed] = useState(42),
    [tokens, setTokens] = useState(4),
    [causal, setCausal] = useState(true),
    [block, setBlock] = useState(2),
    [query, setQuery] = useState(0);
  const d = useMemo(() => {
    const rng = seeded(seed),
      matrix = (m: number, n: number): Mat =>
        Array.from({ length: m }, () =>
          Array.from({ length: n }, () => 2 * rng() - 1),
        ),
      X = matrix(tokens, 4),
      heads = Array.from({ length: 2 }, () => {
        const WQ = matrix(4, 2),
          WK = matrix(4, 2),
          WV = matrix(4, 2),
          Q = multiply(X, WQ),
          K = multiply(X, WK),
          V = multiply(X, WV);
        return { WQ, WK, WV, Q, K, V, ...attention(Q, K, V, causal) };
      });
    const concat = heads[0].output.map((row, i) => [
        ...row,
        ...heads[1].output[i],
      ]),
      WO = matrix(4, 4);
    return { X, heads, concat, WO, output: multiply(concat, WO) };
  }, [seed, tokens, causal]);
  const row = Math.min(query, tokens - 1),
    K = causal ? d.heads[0].K.slice(0, row + 1) : d.heads[0].K,
    V = causal ? d.heads[0].V.slice(0, row + 1) : d.heads[0].V,
    online = onlineAttention(d.heads[0].Q[row], K, V, block),
    max = flash ? online.length - 1 : 5,
    player = usePlayback(max),
    current = online[Math.min(player.index, online.length - 1)];
  function change(action: () => void) {
    player.reset();
    action();
  }
  return (
    <>
      <div className="lab-intro">
        <p>Lecture {flash ? "08" : "07"}</p>
        <h1>
          {flash
            ? "FlashAttention：分块合并的数学"
            : "Attention，从 token 到加权和"}
        </h1>
        <p>
          随机生成的小型教学权重，实际计算两个头的 Q、K、V、softmax
          与输出投影。维度与每一步数值可检查。
        </p>
      </div>
      <div className="lab-body">
        <div className="two-column">
          <div>
            <Slider
              label="Token 数量 N"
              min={2}
              max={6}
              value={tokens}
              onChange={(n) =>
                change(() => {
                  setTokens(n);
                  setQuery(0);
                })
              }
            />
            <Slider
              label="权重随机种子"
              min={1}
              max={100}
              value={seed}
              onChange={(n) => change(() => setSeed(n))}
            />
            <label className="check">
              <input
                type="checkbox"
                checked={causal}
                onChange={(e) => change(() => setCausal(e.target.checked))}
              />
              因果 mask（每行保留自身）
            </label>
            {flash && (
              <>
                <Slider
                  label="K/V 块大小"
                  min={1}
                  max={3}
                  value={block}
                  onChange={(n) => change(() => setBlock(n))}
                />
                <Slider
                  label="观察 query 行（从 0 起）"
                  min={0}
                  max={tokens - 1}
                  value={row}
                  onChange={(n) => change(() => setQuery(n))}
                />
              </>
            )}
          </div>
          <div>
            <h2>{flash ? "流式状态" : "计算阶段"}</h2>
            <p>
              {flash
                ? `当前处理 keys [${current.start}, ${current.end})；query = ${row}`
                : [
                    "1. Token embedding",
                    "2. Q/K/V 线性投影",
                    "3. 缩放点积与 mask",
                    "4. 行 softmax",
                    "5. Value 加权求和",
                    "6. 双头拼接与输出投影",
                  ][player.index]}
            </p>
            <Playback player={player} max={max} />
            <p className="muted">
              确定种子直接映射到实际矩阵；播放仅改变显示阶段，不改变计算结果。双头宽度各
              2，模型宽度 4；不包含位置编码、残差、归一化或训练。
            </p>
          </div>
        </div>
        {flash ? (
          <>
            <div className="matrix-group">
              <MatrixView name="Q（头 1）" data={d.heads[0].Q} />
              <MatrixView
                name="当前 K 块"
                data={K.slice(current.start, current.end)}
              />
              <MatrixView
                name="当前 V 块"
                data={V.slice(current.start, current.end)}
              />
            </div>
            <dl>
              <dt>运行最大值 m</dt>
              <dd>{fmt(current.m)}</dd>
              <dt>归一化和 ℓ</dt>
              <dd>{fmt(current.l)}</dd>
              <dt>未归一化输出 z</dt>
              <dd>{vector(current.z)}</dd>
              <dt>当前块输出 z/ℓ</dt>
              <dd>{vector(current.output)}</dd>
              <dt>完整稠密输出</dt>
              <dd>{vector(d.heads[0].output[row])}</dd>
            </dl>
            <p>
              完整分块计算与稠密结果的最大差值：
              {fmt(
                Math.max(
                  ...online
                    .at(-1)!
                    .output.map((x, i) =>
                      Math.abs(x - d.heads[0].output[row][i]),
                    ),
                ),
              )}
              。尚未遍历完整行时，当前输出只对已读的 keys 归一化。
            </p>
            <MathFormula
              tex={String.raw`m'=\max(m,\max S_j),\quad \ell'=e^{m-m'}\ell+\sum_i e^{S_{j,i}-m'}`}
            />
            <MathFormula
              tex={String.raw`z'=e^{m-m'}z+\sum_i e^{S_{j,i}-m'}V_{j,i},\quad o'=z'/\ell'`}
            />
            <p>
              首块从有效有限 logits 初始化；因果 mask 的未来 keys
              在进入分块前排除。仅需保存当前行 m、ℓ、z
              与块缓冲。教学代码为对照仍计算稠密矩阵，不能将本页浏览器内存占用视作
              FlashAttention kernel 的内存测量。
            </p>
            <p>
              传统核心显式 logits / weights 各 N² 元素，本例各 {tokens * tokens}
              ；教学流式每行统计保存 1 + 1 + dᵥ = 4 元素，另计 Q/K/V/O
              和块缓冲。真实 GPU 的 tile、SRAM 和反向重算需进一步实现；本页不测
              GPU 性能。
            </p>
          </>
        ) : (
          <>
            <div className="matrix-group">
              <MatrixView name="Embedding X" data={d.X} />
              {player.index >= 1 && (
                <>
                  <MatrixView name="WQ（头 1）" data={d.heads[0].WQ} />
                  <MatrixView name="Q = XWQ" data={d.heads[0].Q} />
                  <MatrixView name="K = XWK" data={d.heads[0].K} />
                  <MatrixView name="V = XWV" data={d.heads[0].V} />
                </>
              )}
            </div>
            {player.index >= 2 && (
              <div className="matrix-group">
                <MatrixView name="QKᵀ/√2 + mask" data={d.heads[0].logits} />
                {player.index >= 3 && (
                  <MatrixView name="行归一化权重" data={d.heads[0].weights} />
                )}
                {player.index >= 4 && (
                  <MatrixView
                    name="头 1 输出 = 权重 × V"
                    data={d.heads[0].output}
                  />
                )}
              </div>
            )}
            {player.index >= 5 && (
              <div className="matrix-group">
                <MatrixView name="头 2 输出" data={d.heads[1].output} />
                <MatrixView name="拼接" data={d.concat} />
                <MatrixView name="WO" data={d.WO} />
                <MatrixView name="最终输出" data={d.output} />
              </div>
            )}
            <MathFormula
              tex={String.raw`Q_i=XW_Q^{(i)},\quad K_i=XW_K^{(i)},\quad V_i=XW_V^{(i)},\quad H_i=\mathrm{softmax}\!\left(\frac{Q_iK_i^\top}{\sqrt2}+M\right)V_i`}
            />
            <MathFormula
              tex={String.raw`\mathrm{MHA}(X)=\mathrm{Concat}(H_1,H_2)W_O`}
            />
            <p>
              softmax 先减去行最大值提高稳定性，因果 mask 的 −∞
              变为零权重。当前头 1 各行权重和：
              {d.heads[0].weights
                .map((r) => fmt(r.reduce((a, b) => a + b)))
                .join(", ")}
              。Attention
              权重表示本次加权机制，不能直接当作所有语境下可靠的语义重要性。
            </p>
          </>
        )}
      </div>
      <Quiz
        questions={[
          {
            question: flash
              ? "分块 attention 主要避免保存什么？"
              : "Attention softmax 沿哪个维度归一化？",
            options: flash
              ? ["整个 N×N 中间注意力矩阵", "所有模型权重", "所有 Q/K/V"]
              : [
                  "每个 query 对允许 keys 的一行",
                  "整个模型所有层",
                  "只沿 value 特征",
                ],
            answer: 0,
            explanation: flash
              ? "分块与在线归一化改变 IO 和中间存储，不把精确稠密 attention 的算术复杂度降为线性。"
              : "每行归一化为 1，且至少有一个有效 key，否则整行 −∞ 时普通 softmax 无定义。",
          },
        ]}
      />
      <Sources ids={flash ? [8] : [7]} />
    </>
  );
}
