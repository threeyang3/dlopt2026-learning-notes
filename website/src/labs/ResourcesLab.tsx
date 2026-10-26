import { useState } from "react";
import { optimizerComparisons } from "../content/optimizers";
import { Slider, fmt } from "../components/Controls";
import { MathFormula } from "../components/Math";
import { Sources } from "../components/Sources";
export function ResourcesLab() {
  const [m, setM] = useState(1024),
    [n, setN] = useState(1024),
    [bytes, setBytes] = useState(4);
  return (
    <>
      <div className="lab-intro">
        <p>Lecture 09</p>
        <h1>优化器，究竟保存了什么？</h1>
        <p>
          统一比较单个 m×n
          矩阵参数的持久核心状态。该页计算资源账本，未执行完整优化器或硬件计时。
        </p>
      </div>
      <div className="lab-body">
        <div className="two-column">
          <Slider
            label="矩阵行数 m"
            min={64}
            max={4096}
            step={64}
            value={m}
            onChange={setM}
          />
          <Slider
            label="矩阵列数 n"
            min={64}
            max={4096}
            step={64}
            value={n}
            onChange={setN}
          />
        </div>
        <label>
          状态精度
          <select value={bytes} onChange={(e) => setBytes(+e.target.value)}>
            <option value={4}>FP32</option>
            <option value={2}>16-bit 状态假设</option>
          </select>
        </label>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>方法 / 状态</th>
                <th>更新机制</th>
                <th>状态复杂度与估算</th>
                <th>计算复杂度</th>
                <th>条件与限制</th>
              </tr>
            </thead>
            <tbody>
              {optimizerComparisons.map((o) => (
                <tr key={o.name}>
                  <td>
                    <b>{o.name}</b>
                    <br />
                    {o.states}
                  </td>
                  <td>{o.mechanism}</td>
                  <td>
                    {o.memory}
                    <br />
                    {o.count(m, n).toLocaleString()} 元素
                    <br />
                    {fmt((o.count(m, n) * bytes) / 2 ** 20)} MiB
                  </td>
                  <td>{o.compute}</td>
                  <td>
                    {o.assumptions}
                    <br />
                    {o.limits}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <MathFormula
          tex={String.raw`L_t=\epsilon I+\sum_{s=1}^tG_sG_s^\top,\quad R_t=\epsilon I+\sum_{s=1}^tG_s^\top G_s,\quad \Delta W_t=-\eta L_t^{-1/4}G_tR_t^{-1/4}`}
        />
        <p>
          Shampoo 使用课程的累积统计矩阵版，不能与 EMA 变体无说明拼接。Adafactor
          为不保存一阶矩的比较配置。Muon
          为动量核心计数，实际实现还可能保留其他状态。
        </p>
        <p>
          上述元素不含权重、梯度、激活、FP32 主权重、分解缓存、通信或 allocator
          暂存。真实峰值及端到端速度不能由持久状态复杂度直接推断。当前公开资料中
          Lite / MuonM 的独立定义未确认，故没有编造更新式。
        </p>
      </div>
      <Sources ids={[9]} />
    </>
  );
}
