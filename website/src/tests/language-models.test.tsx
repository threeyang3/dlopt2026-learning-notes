import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { renderToStaticMarkup } from "react-dom/server";
import { SupplementNotes, resolveNoteHref, buildHeadingIndex } from "../components/Notes";
import supplements from "../content/supplements.json";
import { publishedResources } from "../../scripts/resources";
import { embeddingColumn, cosine, toyCorpus, buildNgram, ngramDistribution, sampleToken, categoricalNll, generateNgram, makeRnnWeights, rnnForward, rnnGradients, trainRnnStep, scalarMemory, type RnnWeights } from "../engines/language-models";

describe("Language Models 来源与阅读", () => {
  it("原件哈希、字节数及发布路径与附件一致", () => {
    const s = supplements[0];
    const pdf = readFileSync(resolve("..", s.pdf));
    expect(pdf.length).toBe(7225650);
    expect(createHash("sha256").update(pdf).digest("hex")).toBe("ca4316886f225f3324b81d99a3e0ee97efd8ed42ca6bbe73fd158478d8441419");
    expect(s.sha256).toBe(createHash("sha256").update(pdf).digest("hex"));
    for (const path of [s.note, s.pdf]) expect(publishedResources).toContain(path);
  });
  it("补充笔记安全渲染、标题唯一、正课与补充双向链接可解", () => {
    const s = supplements[0], text = readFileSync(resolve("..", s.note), "utf8");
    const html = renderToStaticMarkup(<SupplementNotes id={s.id} />);
    expect(html).toContain('class="katex"');
    expect(html).not.toContain('class="katex-error"');
    expect(html).toContain('href="#lecture/7"');
    const headings = buildHeadingIndex(text);
    expect(new Set(headings.map((h) => h.id)).size).toBe(headings.length);
    expect(resolveNoteHref(s.slug + ".md", 7)).toBe("#supplement/language-models");
    const h = headings.find((entry) => entry.title.startsWith("4. RNN"))!;
    const link = "#supplement/language-models/" + encodeURIComponent(h.id);
    expect(resolveNoteHref(s.slug + ".md#" + h.title, 7)).toBe(link);
    expect(resolveNoteHref("#" + h.title, undefined, s.slug)).toBe(link);
    expect(resolveNoteHref(link)).toBe(link);
    expect(text).toContain("勘误"); expect(text).toContain("推导补充");
  });
});
describe("embedding 与 n-gram", () => {
  it("查表是取列，非取行；one-hot 不承载语义距离", () => {
    expect(embeddingColumn(4)).toEqual([0, -0.01, 0.03, 0.95]);
    expect(embeddingColumn(5)).toEqual([0.01, 0, -0.02, 0.97]);
    expect(cosine([1, 0], [0, 1])).toBe(0);
    expect(cosine([0, 0], [1, 0])).toBeNull();
    expect(() => embeddingColumn(6)).toThrow();
  });
  it("4-gram 只条件于最后三词，保留所有未展示类别的质量", () => {
    const model = buildNgram(toyCorpus, 4);
    const d = ngramDistribution(model, ["ignored", "students", "opened", "their"]);
    expect(d.total).toBe(10);
    expect(d.probabilities![model.vocabulary.indexOf("books")]).toBe(0.4);
    expect(d.probabilities![model.vocabulary.indexOf("exams")]).toBe(0.1);
    expect(d.probabilities!.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 14);
  });
  it("unigram 不错误地保留全部历史；句末 EOS 纳入词表", () => {
    const m = buildNgram([["a", "b"], ["a", "c"]], 1);
    const d = ngramDistribution(m, ["anything"]);
    expect(d.total).toBe(6);
    expect(d.probabilities).toEqual([2 / 6, 1 / 6, 1 / 6, 2 / 6]);
    expect(ngramDistribution(m, []).probabilities).toEqual(d.probabilities);
  });
  it("句子边界不产生跨句 n-gram，BOS 不在输出空间", () => {
    const m = buildNgram([["a", "b"], ["c", "d"]], 2);
    const d = ngramDistribution(m, ["b"]);
    expect(d.counts[m.vocabulary.indexOf("c")]).toBe(0);
    expect(d.probabilities![m.vocabulary.indexOf("<eos>")]).toBe(1);
    expect(m.vocabulary).not.toContain("<bos>");
    expect(ngramDistribution(m, ["<bos>"]).total).toBe(2);
  });
  it("未见上下文保持未定义，显式 add-alpha 才得到均匀分布", () => {
    const m = buildNgram([["a", "b"]], 2);
    expect(ngramDistribution(m, ["new"]).probabilities).toBeNull();
    expect(ngramDistribution(m, ["new"], 1).probabilities).toEqual([1 / 3, 1 / 3, 1 / 3]);
    expect(categoricalNll(0)).toBe(Infinity);
    expect(categoricalNll(0.4)).toBeCloseTo(-Math.log(0.4), 14);
    expect(() => ngramDistribution(m, [], -1)).toThrow();
  });
  it("已见上下文的平滑公式与手算一致", () => {
    const m = buildNgram([["a", "b"], ["a", "c"]], 2);
    expect(ngramDistribution(m, ["a"], 1).probabilities).toEqual([1 / 6, 2 / 6, 2 / 6, 1 / 6]);
  });
  it("采样不选零概率词，EOS 与未定义上下文都停止生成", () => {
    expect(sampleToken([0, 1, 0], 0)).toBe(1);
    expect(sampleToken([0.4, 0.6, 0], 0.4)).toBe(1);
    expect(() => sampleToken([0.4, 0.1], 0)).toThrow();
    const m = buildNgram([["a", "b"]], 2);
    expect(generateNgram(m, ["b"], 0, 42, 8)).toEqual({ tokens: ["b", "<eos>"], reason: "eos" });
    expect(generateNgram(m, ["new"], 0, 42, 8).reason).toBe("unseen");
  });
  it("增加生成步数不重新改变已有随机前缀", () => {
    const m = buildNgram(toyCorpus, 1);
    for (const seed of [1, 42, 100]) {
      const short = generateNgram(m, ["the"], 0.5, seed, 1);
      const long = generateNgram(m, ["the"], 0.5, seed, 5);
      expect(long.tokens.slice(0, short.tokens.length)).toEqual(short.tokens);
    }
  });
});
describe("RNN 与完整 BPTT", () => {
  it("零模型为均匀分布，CE 与输出偏置梯度有手算对照", () => {
    const w = makeRnnWeights(42);
    for (const key of ["E", "Wh", "We", "U"] as const) w[key] = w[key].map((row) => row.map(() => 0));
    const result = rnnGradients(w, [0, 1], [1, 2]);
    expect(result.loss).toBeCloseTo(Math.log(7), 14);
    expect(result.states[0].hidden).toEqual([0, 0]);
    expect(result.gradients.b2[1]).toBeCloseTo(1 / 7 - 0.5, 14);
    expect(result.gradients.b2[0]).toBeCloseTo(1 / 7, 14);
    expect(result.gradients.Wh.flat()).toEqual([0, 0, 0, 0]);
  });
  it("固定权重的前缀状态不依赖未来 token，前向不改变参数", () => {
    const w = makeRnnWeights(7), before = JSON.stringify(w);
    const a = rnnForward(w, [0, 1], [1, 2]);
    const b = rnnForward(w, [0, 1, 2, 3], [1, 2, 3, 4]);
    expect(b.states.slice(0, 2)).toEqual(a.states);
    expect(JSON.stringify(w)).toBe(before);
    a.states.forEach((s) => expect(s.probabilities.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 14));
  });
  it("全部 54 个梯度与中心有限差分一致（含重复词累加）", () => {
    const w = makeRnnWeights(42), input = [0, 1, 0, 1], target = [1, 0, 1, 6];
    const { gradients: g } = rnnGradients(w, input, target);
    let checked = 0;
    function check(get: (a: RnnWeights) => number, set: (a: RnnWeights, n: number) => void, analytic: number) {
      const plus = structuredClone(w), minus = structuredClone(w), eps = 1e-6;
      set(plus, get(w) + eps); set(minus, get(w) - eps);
      const numeric = (rnnForward(plus, input, target).loss - rnnForward(minus, input, target).loss) / (2 * eps);
      expect(Math.abs(numeric - analytic)).toBeLessThan(3e-7); checked++;
    }
    for (const key of ["E", "Wh", "We", "U"] as const)
      w[key].forEach((row, i) => row.forEach((_, j) => check((a) => a[key][i][j], (a, n) => { a[key][i][j] = n; }, g[key][i][j])));
    for (const key of ["b1", "b2"] as const)
      w[key].forEach((_, i) => check((a) => a[key][i], (a, n) => { a[key][i] = n; }, g[key][i]));
    expect(checked).toBe(54);
  });
  it("小步训练降低给定样例平均损失而不原地修改权重", () => {
    const w = makeRnnWeights(42), snapshot = JSON.stringify(w), inputs = [0, 1, 2, 3, 4], targets = [1, 2, 3, 4, 6];
    const next = trainRnnStep(w, inputs, targets, 0.01);
    expect(rnnForward(next, inputs, targets).loss).toBeLessThan(rnnForward(w, inputs, targets).loss);
    expect(JSON.stringify(w)).toBe(snapshot);
  });
  it("拒绝错位长度、非法词、错误矩阵和非有限参数", () => {
    const w = makeRnnWeights(1);
    expect(() => rnnForward(w, [0], [1, 2])).toThrow();
    expect(() => rnnForward(w, [7], [1])).toThrow();
    expect(() => rnnForward({ ...w, E: [[1]] }, [0], [1])).toThrow();
    expect(() => trainRnnStep(w, [0], [1], 0)).toThrow();
  });
});
describe("标量梯度传播", () => {
  it("零轨迹的初态导数是 w^T，负号不被绝对值吞掉", () => {
    expect(scalarMemory(0.5, 0, 3)[3]).toEqual({ step: 3, hidden: 0, local: 0.5, derivative: 0.125 });
    expect(scalarMemory(-1.2, 0, 3)[3].derivative).toBeCloseTo(-1.728, 12);
  });
  it("非零轨迹按实际饱和导数计算，匹配初态有限差分", () => {
    const w = 1.2, h = 0.3, n = 10, eps = 1e-6;
    const actual = scalarMemory(w, h, n)[n].derivative;
    const finiteDifference = (scalarMemory(w, h + eps, n)[n].hidden - scalarMemory(w, h - eps, n)[n].hidden) / (2 * eps);
    expect(actual).toBeCloseTo(finiteDifference, 7);
    expect(Math.abs(actual)).toBeLessThan(w ** n);
  });
});
