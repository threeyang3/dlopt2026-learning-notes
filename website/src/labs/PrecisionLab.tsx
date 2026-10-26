import { useState } from "react";
import {
  formats,
  formatBounds,
  quantize,
  type Format,
} from "../engines/precision";
import { Slider, fmt } from "../components/Controls";
import { Sources } from "../components/Sources";
import { Quiz } from "../components/Quiz";
export function PrecisionLab() {
  const [coefficient, setCoefficient] = useState(1.001),
    [exponent, setExponent] = useState(0),
    [scale, setScale] = useState(1),
    [raw, setRaw] = useState("1.001");
  const x = coefficient * 10 ** exponent;
  return (
    <>
      <div className="lab-intro">
        <p>Lecture 13</p>
        <h1>有限精度，如何吞掉一次更新？</h1>
        <p>
          实现 round-to-nearest, ties-to-even，保留次正规数的格式教学模型；可与
          FP32 原生转换比较。
        </p>
      </div>
      <div className="lab-body">
        <div className="two-column">
          <div>
            <Slider
              label="有效系数"
              min={-2}
              max={2}
              step={0.001}
              value={coefficient}
              onChange={setCoefficient}
            />
            <Slider
              label="十进制指数"
              min={-45}
              max={39}
              value={exponent}
              onChange={setExponent}
            />
            <Slider
              label="Loss scale = 2ˢ"
              min={0}
              max={24}
              value={scale}
              onChange={setScale}
            />
            <p>输入 x = {fmt(x)}</p>
            <label>
              直接输入待转换数值
              <input
                aria-label="待转换数值"
                value={raw}
                onChange={(e) => setRaw(e.target.value)}
              />
            </label>
          </div>
          <div>
            <h2>舍入策略与范围</h2>
            <p>
              指数决定范围，尾数决定相对分辨率。这里模拟标量转换，不模拟 GEMM
              累加、次正规清零或真实训练。Loss scaling
              不能恢复已溢出的梯度；检测到非有限值时应跳过无效更新。
            </p>
            <p>
              FP16 的 loss scaling 与主权重策略不能无条件推广到
              BF16。共享指数范围也不表示尾数精度相同。
            </p>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>格式</th>
                <th>指数 / 尾数</th>
                <th>最大有限值</th>
                <th>最小正规 / 次正规</th>
                <th>转换 x</th>
                <th>误差</th>
                <th>直接输入转换</th>
                <th>缩放后再还原</th>
              </tr>
            </thead>
            <tbody>
              {(Object.keys(formats) as Format[]).map((f) => {
                const bounds = formatBounds(f),
                  rounded = quantize(x, f);
                return (
                  <tr key={f}>
                    <td>{f}</td>
                    <td>
                      {formats[f].exponent} / {formats[f].mantissa}
                    </td>
                    <td>{fmt(bounds.max)}</td>
                    <td>
                      {fmt(bounds.normal)} / {fmt(bounds.subnormal)}
                    </td>
                    <td>{fmt(rounded)}</td>
                    <td>{fmt(Math.abs(x - rounded))}</td>
                    <td>{fmt(quantize(Number(raw), f))}</td>
                    <td>{fmt(quantize(x * 2 ** scale, f) / 2 ** scale)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <h2>小更新累积：1000 次减去 10⁻⁵</h2>
        <p>
          初值为 1。每步重新舍入的低精度权重，与持久 FP32 主权重对照。保留 FP32
          历史后最终显示低精度副本。
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>格式</th>
                <th>每步低精度保存</th>
                <th>FP32 主权重最终转换</th>
              </tr>
            </thead>
            <tbody>
              {(Object.keys(formats) as Format[]).map((f) => {
                let low = 1,
                  master = 1;
                for (let i = 0; i < 1000; i++) {
                  low = quantize(low - 1e-5, f);
                  master = Math.fround(master - 1e-5);
                }
                return (
                  <tr key={f}>
                    <td>{f}</td>
                    <td>{fmt(low)}</td>
                    <td>{fmt(quantize(master, f))}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
      <Quiz
        questions={[
          {
            question: "BF16 相比 FP16 的主要取舍是什么？",
            options: [
              "指数更宽、尾数更短",
              "完全相同",
              "必然需要同样的 loss scaling",
            ],
            answer: 0,
            explanation:
              "BF16 为 8 位指数、7 位尾数，FP16 为 5 位指数、10 位尾数。范围与精度需要分别比较。",
          },
        ]}
      />
      <Sources ids={[13]} />
    </>
  );
}
