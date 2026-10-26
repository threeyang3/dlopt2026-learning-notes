import { useMemo, useState } from "react";
import { lowRank, type Mat } from "../engines/matrix";
import { Slider, fmt } from "../components/Controls";
import { MatrixView } from "../components/Matrix";
import { MathFormula } from "../components/Math";
import { Sources } from "../components/Sources";
import { Quiz } from "../components/Quiz";
export function LinearLab({ subspace = false }: { subspace?: boolean }) {
  const [rank, setRank] = useState(1),
    [a, setA] = useState(3),
    [b, setB] = useState(1),
    [angle, setAngle] = useState(30);
  const matrix = useMemo<Mat>(() => {
    const c = Math.cos((angle * Math.PI) / 180),
      s = Math.sin((angle * Math.PI) / 180);
    return [
      [a * c, -b * s],
      [a * s, b * c],
      [0.4 * a, 0.1 * b],
    ];
  }, [a, b, angle]);
  const d = lowRank(matrix, rank),
    c = Math.cos((angle * Math.PI) / 180),
    s = Math.sin((angle * Math.PI) / 180);
  const geometry = (transformed: boolean) =>
    Array.from({ length: 81 }, (_, i) => {
      const t = (i * Math.PI) / 40,
        x = Math.cos(t),
        y = Math.sin(t);
      return `${180 + 45 * (transformed ? a * c * x - b * s * y : x)},${150 - 45 * (transformed ? a * s * x + b * c * y : y)}`;
    }).join(" ");
  return (
    <>
      <div className="lab-intro">
        <p>Lecture 02 / 03 / 10</p>
        <h1>{subspace ? "低秩投影与状态压缩" : "线性变换与 SVD"}</h1>
        <p>
          修改真实的 3×2
          矩阵，观察正交投影、奇异值和低秩近似。二维图展示前两行对应的线性映射。
        </p>
      </div>
      <div className="lab-body">
        <div className="two-column">
          <div>
            <Slider
              label="第一列尺度"
              min={0.2}
              max={4}
              step={0.1}
              value={a}
              onChange={setA}
            />
            <Slider
              label="第二列尺度"
              min={0.2}
              max={4}
              step={0.1}
              value={b}
              onChange={setB}
            />
            <Slider
              label="旋转 θ"
              min={0}
              max={90}
              value={angle}
              onChange={setAngle}
            />
            <Slider
              label="保留秩 r"
              min={1}
              max={2}
              value={rank}
              onChange={setRank}
            />
            <p>
              矩阵 G 形状 3×2，数值秩 {d.rank}；奇异值{" "}
              {d.singular.map(fmt).join(", ")}；谱条件数{" "}
              {fmt(d.singular[0] / d.singular[1])}。
            </p>
          </div>
          <svg
            viewBox="0 0 360 300"
            role="img"
            aria-label="单位圆经矩阵变换成为椭圆"
          >
            <path d="M10,150H350M180,10V290" stroke="var(--line)" />
            <polyline
              points={geometry(false)}
              fill="none"
              stroke="var(--series-3)"
              strokeDasharray="5 4"
            />
            <polyline
              points={geometry(true)}
              fill="none"
              stroke="var(--series-1)"
              strokeWidth="2"
            />
            <text x="15" y="25" fill="var(--ink)" fontSize="12">
              虚线：单位圆；实线：前两行变换
            </text>
            <text x="330" y="170" fill="var(--ink)">
              x₁
            </text>
            <text x="190" y="25" fill="var(--ink)">
              x₂
            </text>
          </svg>
        </div>
        <div className="matrix-group">
          <MatrixView name="原始梯度 G" data={matrix} />
          <MatrixView name="正交基 P" data={d.P} />
          <MatrixView name="投影 PᵀG" data={d.projected} />
          <MatrixView name="回映 PPᵀG" data={d.lifted} />
        </div>
        <p
          className="stat-line"
          data-testid="projection-error"
          data-error={d.error}
        >
          Frobenius 残差 = {fmt(d.error)}；被舍弃奇异值平方和 ={" "}
          {fmt(d.singular.slice(rank).reduce((v, s) => v + s * s, 0))}；残差平方
          = {fmt(d.error ** 2)}。
        </p>
        <MathFormula
          tex={String.raw`G=U\Sigma V^\top,\quad P=U_{:,1:r}\in\mathbb R^{3\times r},\quad P^\top P=I_r,\quad\widetilde G=P^\top G\in\mathbb R^{r\times2},\quad\widehat G=P\widetilde G`}
        />
        <div className="matrix-group">
          <MatrixView name="紧致左奇异向量 U" data={d.U} />
          <MatrixView name="右奇异向量 V" data={d.V} />
        </div>
        <p>
          前 r 个左奇异向量定义正交投影。Eckart–Young 的最佳秩 r
          近似结论需对有限实矩阵在 Frobenius
          范数下理解；这是笔记中的标准补充定理，尚未在本轮独立复核原始教材证明。
        </p>
        <h2>GaLore 教学模型：表示不等于优化器</h2>
        <p>
          本页只实现 SVD 投影与回映，没有声称执行完整 GaLore 的 Adam
          状态换基和周期刷新。原矩阵表示 6 个元素；低秩投影表示基与压缩梯度共{" "}
          {3 * rank + 2 * rank} 个元素。低秩 Adam 两个状态加基共{" "}
          {2 * rank * 2 + 3 * rank} 个元素，对照完整 Adam 状态 12
          个元素。在这样的小矩阵中低秩并不必然节省。
        </p>
        <p>
          权重和临时完整梯度不在上述状态计数内。维度大、秩小才可能显著省状态；投影不自动节省反向激活，也不能被称为原梯度的无偏估计。
        </p>
      </div>
      <Quiz
        questions={[
          {
            question: "P 为 3×r，G 为 3×2，哪个乘法正确？",
            options: ["PG", "PᵀG 为 r×2，再乘 P 回映", "GP 为 2×r"],
            answer: 1,
            explanation:
              "左投影先把 3 维行空间压缩成 r 维。P 的列正交保证 PPᵀ 是正交投影。",
          },
        ]}
      />
      <Sources ids={[2, 3, 10]} />
    </>
  );
}
