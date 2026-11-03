from pathlib import Path
import json
root = Path.cwd()
def read(p): return (root / p).read_text(encoding='utf-8')
def write(p, t): (root / p).write_text(t, encoding='utf-8')
assert 'SupplementCatalog' not in read('website/src/App.tsx'), 'Integration already applied'
s = read('website/src/components/Notes.tsx')
s = s.replace('import { course, noteText, resource } from "../content/course";', 'import { course, noteText, resource } from "../content/course";\nimport supplements from "../content/supplements.json";')
a = s.index('function lectureHref('); b = s.index('\nfunction headingRenderer', a)
s = s[:a] + '''function documentHref(slug: string, anchor?: string) {
  const lecture = course.find((entry) => entry.slug === slug);
  const supplement = supplements.find((entry) => entry.slug === slug);
  const base = lecture ? "#lecture/" + lecture.id : supplement ? "#supplement/" + supplement.id : undefined;
  if (!base) return undefined;
  if (!anchor) return base;
  const name = decodeFragment(anchor).trim();
  const headings = buildHeadingIndex(noteText(slug));
  const idForTitle = headings.find((entry) => entry.title === name)?.id;
  const idForExistingSlug = headings.find((entry) => entry.id === name)?.id;
  const idForSlug = headings.find((entry) => headingSlug(entry.title) === headingSlug(name))?.id;
  return base + "/" + encodeURIComponent(idForTitle ?? idForExistingSlug ?? idForSlug ?? headingSlug(name));
}
function lectureHref(id: number, anchor?: string) {
  const lecture = course.find((entry) => entry.id === id);
  return lecture ? documentHref(lecture.slug, anchor) : undefined;
}
''' + s[b:]
s = s.replace('  const text = noteText(lecture.slug).replace(', '''  return <NoteDocument slug={lecture.slug} lectureId={id} />;
}
export function SupplementNotes({ id }: { id: string }) {
  const entry = supplements.find((s) => s.id === id);
  return entry ? <NoteDocument slug={entry.slug} /> : null;
}
function NoteDocument({ slug, lectureId }: { slug: string; lectureId?: number }) {
  const text = noteText(slug).replace(''')
s = s.replace('''      const other = course.find((l) => l.slug === name);
      const target = other
        ? lectureHref(other.id, anchor)
        : name + ".md" + (anchor ? "#" + encodeURIComponent(anchor) : "");''', '''      const target = documentHref(name, anchor) ?? name + ".md" + (anchor ? "#" + encodeURIComponent(anchor) : "");''')
s = s.replace('resolveNoteHref(href, id)', 'resolveNoteHref(href, lectureId, slug)')
s = s.replace('export function resolveNoteHref(href?: string, currentLectureId?: number)', 'export function resolveNoteHref(href?: string, currentLectureId?: number, currentSlug?: string)')
s = s.replace(r'/^(https?:|mailto:|#lecture\/)/', r'/^(https?:|mailto:|#lecture\/|#supplement\/)/')
s = s.replace('''  const lecture = course.find(
    (l) => clean === l.slug + ".md" || clean.endsWith("/" + l.slug + ".md"),
  );
  if (lecture) return lectureHref(lecture.id, fragment || undefined);''', '''  if (!clean && currentSlug && fragment) return documentHref(currentSlug, fragment);
  const document = [...course, ...supplements].find(
    (l) => clean === l.slug + ".md" || clean.endsWith("/" + l.slug + ".md"),
  );
  if (document) return documentHref(document.slug, fragment || undefined);''')
write('website/src/components/Notes.tsx', s)
s = read('website/src/App.tsx')
s = s.replace('import { useEffect, useRef, useState }', 'import { lazy, Suspense, useEffect, useRef, useState }')
s = s.replace('import { Notes }', 'import { Notes, SupplementNotes }')
s = s.replace('function routeAnchor(route: string) {', '''import { SupplementCatalog, SupplementSources } from "./components/Supplements";
import supplements from "./content/supplements.json";
const LanguageModelsLab = lazy(() => import("./labs/LanguageModelsLab").then((m) => ({ default: m.LanguageModelsLab })));
function routeAnchor(route: string) {''')
s = s.replace(r'/^lecture\/\d+\/(.+)$/', r'/^(?:lecture\/\d+|supplement\/[a-z0-9-]+)\/(.+)$/')
s = s.replace('''  return (
    <>
      <a
        className="skip"''', '''  const supplement = route.startsWith("supplement/") ? supplements.find((entry) => entry.id === route.split("/")[1]) : undefined;
  return (
    <>
      <a
        className="skip"''', 1)
s = s.replace('          {route === "sgd" && <StochasticLab />}', '''          {route === "language-models" && <Suspense fallback={<p role="status">正在载入语言模型实验室…</p>}><LanguageModelsLab /></Suspense>}
          {supplement && <>
            <div className="lab-intro"><p>补充单元 · 不计入原 14 讲编号</p><h1>{supplement.title}</h1><p>{supplement.evidence}</p><a className="button primary" href={"#" + supplement.lab}>打开语言模型实验室</a></div>
            <SupplementNotes id={supplement.id} />
            <SupplementSources id={supplement.id} />
          </>}
          {route === "sgd" && <StochasticLab />}''')
s = s.replace('            !lecture && (', '            !lecture && !supplement && (')
s = s.replace('              <Sources ids={course.map((x) => x.id)} />', '''              <SupplementCatalog />
              <SupplementSources id="language-models" />
              <Sources ids={course.map((x) => x.id)} />''')
s = s.replace('              <Notes id={lecture.id} />', '''              {(lecture.id === 1 || lecture.id === 7) && <SupplementCatalog />}
              <Notes id={lecture.id} />''')
s = s.replace('''      <section className="notice">
        <strong>笔记与动态实验的关系</strong>''', '''      <SupplementCatalog />
      <section className="notice">
        <strong>笔记与动态实验的关系</strong>''')
pos = s.index('function CourseMap(')
s = s[:pos] + s[pos:].replace('''      </div>
    </section>''', '''      </div>
      <SupplementCatalog />
    </section>''', 1)
write('website/src/App.tsx', s)
s = read('website/src/content/course.ts').replace('  attention: "Attention",', '  "language-models": "语言模型与 RNN",\n  attention: "Attention",')
write('website/src/content/course.ts', s)
s = read('website/src/content/modules.ts'); pos = s.index('  attention: {')
s = s[:pos] + '''  "language-models": {
    lectures: [1, 3, 4, 7],
    goal: "从 one-hot 查表、n-gram 计数走到共享状态、next-token 损失与梯度传播。",
    prerequisites: "矩阵乘法、条件概率、softmax、链式法则",
    conditions: "补充课件 56 页；人工示意 embedding、自建 12 句语料、7 词/2 状态的单向单层 tanh RNN。",
    misconception: "可处理长序列不等于可靠记住长距离；参数共享不意味着激活内存不增长。",
    experiment: "检查未见上下文；区分播放和训练；比较零初态与饱和轨迹的导数连乘。",
    status: "partially_verified",
    evidence: "原件页序、计数边界、完整 BPTT 有限差分和浏览器操作有测试；平滑与梯度推导标为补充，不声称实现预训练 LLM、双向或多层 RNN。",
  },
''' + s[pos:]
write('website/src/content/modules.ts', s)
s = read('website/scripts/resources.ts').replace('import course from "../src/content/course.json";', 'import course from "../src/content/course.json";\nimport supplements from "../src/content/supplements.json";')
s = s.replace('    "notes/README.md",', '    ...supplements.flatMap((entry) => [entry.note, entry.pdf]),\n    "notes/README.md",')
write('website/scripts/resources.ts', s)
write('website/src/styles/app.css', read('website/src/styles/app.css') + '''
.lm-panel, .supplement-catalog { min-width: 0; }
.lm-tabs, .lm-controls, .lm-timeline { display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 0; }
.lm-tabs [aria-selected="true"], .lm-timeline [aria-current="step"] { background: var(--soft); border-color: var(--accent); font-weight: 700; }
.lm-controls label { display: grid; gap: 6px; }
.lm-panel select { max-width: 100%; }
.lm-timeline { list-style: none; padding: 0; }
.lm-scroll { max-width: 100%; overflow: auto; margin: 16px 0; }
.lm-scroll table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
.lm-scroll th, .lm-scroll td { text-align: left; padding: 8px; border-bottom: 1px solid var(--line); }
.lm-scroll caption { text-align: left; color: var(--muted); margin-bottom: 8px; }
.lm-panel p, .supplement-catalog p { overflow-wrap: anywhere; }
.lm-panel .notice { margin: 16px 0; }
''')
write('source_manifest.tsv', read('source_manifest.tsv').rstrip() + '\nSupplement-Language-Models\tSlides-01\t03_langmodel.pdf\thttps://raw.githubusercontent.com/kunyuan827/kunyuan827.github.io/master/teaching/DLOpt2024/03_langmodel.pdf\tmaterials/Supplement-Language-Models/03_langmodel.pdf\n')
write('checksums.sha256', read('checksums.sha256').rstrip() + '\nca4316886f225f3324b81d99a3e0ee97efd8ed42ca6bbe73fd158478d8441419  materials/Supplement-Language-Models/03_langmodel.pdf\n')
s = read('tools/check_notes.py').replace('import csv', 'import csv\nimport json')
s = s.replace('    md_files = markdown_files(ROOT)', '''    supplements = json.loads((ROOT / "website/src/content/supplements.json").read_text(encoding="utf-8"))
    supplement_pdfs = {entry["pdf"] for entry in supplements}
    supplement_notes = {entry["note"] for entry in supplements}
    if len(supplement_pdfs) != len(supplements) or len(supplement_notes) != len(supplements):
        errors.append("duplicate supplemental PDF/note registration")
    actual_supplements = {str(path.relative_to(ROOT)) for path in NOTES.glob("Supplement-*.md")}
    if actual_supplements != supplement_notes:
        errors.append("supplemental notes differ from explicit content index")
    for entry in supplements:
        if not (ROOT / entry["note"]).is_file():
            errors.append(f"missing supplement note: {entry['note']}")
        target = ROOT / entry["pdf"]
        if target.is_file() and (target.stat().st_size != entry["bytes"] or hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]):
            errors.append(f"supplement differs from source identity: {entry['pdf']}")

    md_files = markdown_files(ROOT)''')
s = s.replace('    for path in LECTURE_FILES:\n', '    for path in LECTURE_FILES + sorted(NOTES.glob("Supplement-*.md")):\n')
s = s.replace('''    if len(rows) != 35:
        errors.append(f"expected 35 manifest rows, found {len(rows)}")''', '''    expected_count = 35 + len(supplements)
    core_rows = [row for row in rows if row["lecture"].startswith("Lecture-")]
    supplemental_rows = [row for row in rows if not row["lecture"].startswith("Lecture-")]
    if len(core_rows) != 35 or len(rows) != expected_count:
        errors.append(f"expected 35 core + {len(supplements)} supplemental manifest rows, found {len(rows)}")
    if {row["target_path"] for row in supplemental_rows} != supplement_pdfs:
        errors.append("supplemental manifest paths differ from explicit content index")
    manifest_paths = {row["target_path"] for row in rows}
    if len(manifest_paths) != len(rows):
        errors.append("duplicate manifest target")
    if manifest_paths != {str(path.relative_to(ROOT)) for path in (ROOT / "materials").rglob("*.pdf")}:
        errors.append("PDF files differ from registered manifest")''')
s = s.replace('''    if len(checksum_lines) != 35:
        errors.append(f"expected 35 checksum rows, found {len(checksum_lines)}")''', '''    if len(checksum_lines) != expected_count:
        errors.append(f"expected {expected_count} checksum rows, found {len(checksum_lines)}")
    checksum_paths = [line.split(maxsplit=1)[1].strip() for line in checksum_lines if len(line.split(maxsplit=1)) == 2]
    if len(set(checksum_paths)) != len(checksum_paths) or set(checksum_paths) != manifest_paths:
        errors.append("checksum paths must match manifest exactly, without duplicates")''')
s = s.replace('        "OK: 14 notes, 35 materials, links, math delimiters, and SHA-256 checksums"', '        f"OK: 14 core notes + {len(supplements)} supplements, {expected_count} materials, links, math delimiters, and SHA-256 checksums"')
write('tools/check_notes.py', s)
for path in ['website/package.json', 'website/package-lock.json']:
    data = json.loads(read(path)); data['version'] = '1.1.0'
    if 'packages' in data: data['packages']['']['version'] = '1.1.0'
    write(path, json.dumps(data, ensure_ascii=False, indent=2) + '\n')
s = read('website/e2e/core.spec.ts').replace('    "resources",\n    "sources",', '    "resources",\n    "language-models",\n    "supplement/language-models",\n    "sources",')
write('website/e2e/core.spec.ts', s)
additions = {
'notes/README.md': '''
## 补充单元（不改变原 14 讲编号）

[语言模型基础：词嵌入、n-gram 与 RNN](Supplement-Language-Models.md)依据用户补充的 56 页课件编写，建议在第 7 讲之前学习。原件、公式页码、实现边界和勘误均在笔记中标注。
''',
'README.md': '''
## 新增补充单元：Language Models（2026-11-03）

原课程的 14 讲、35 份 PDF 保持不变；另收录 1 份用户提供的语言模型课件和 1 篇补充笔记。当前仓库共 36 份 PDF、1,235 页。原课程统计与历史审校记录不等于补充单元已经在当年课程清单内。

- [原始课件：03_langmodel.pdf，56 页](materials/Supplement-Language-Models/03_langmodel.pdf)
- [补充笔记：词嵌入、n-gram 与 RNN](notes/Supplement-Language-Models.md)
- [网站语言模型实验室](https://threeyang.top/dlopt2026-learning-notes/#language-models)

网站提供嵌入查表、n-gram 计数与采样、单层 RNN 前向/BPTT 单步训练、标量梯度传播；每个示例均标注数据来源和教学范围。不调整原课件、不改变原 14 讲进度、不增加新依赖。
''',
'resource_inventory.md': '''
## 用户补充资料（2026-11-03）

原课程清单仍为上述 35 份；新增资料单独登记，不冒充官网原 14 讲的一部分。当前总计 **36 份 PDF、1,235 页**。

| 单元 | 文件 | 页数 | 来源及内容 |
|---|---|---:|---|
| Language Models | [03_langmodel.pdf](materials/Supplement-Language-Models/03_langmodel.pdf) | 56 | 用户附件与作者 DLOpt2024 公开原件字节一致；词嵌入、n-gram、固定窗口、RNN、交叉熵、双向与多层结构 |

SHA-256：`ca4316886f225f3324b81d99a3e0ee97efd8ed42ca6bbe73fd158478d8441419`；7,225,650 字节。已纳入总 manifest、校验和与网站显式发布名单，原件不修改。
''',
'notes/COURSE_SCOPE.md': '''
## 用户补充的语言模型先修单元（2026-11-03）

`Supplement-Language-Models.md` 和 `materials/Supplement-Language-Models/03_langmodel.pdf` 是用户追加材料，虽来自同一教师公开资料，但没有据此改写 2026 年原始课程清单。建议置于第 7 讲之前；课程核心仍为原 14 讲，补充笔记与实验不计入其掌握进度。
''',
'REVISION_NOTES.md': '''
## 2026-11-03：语言模型补充单元

新增原件 `03_langmodel.pdf`，56 页，附件与作者公开版本 SHA-256 一致。新增独立学习笔记，按课件顺序覆盖 embedding、n-gram、固定窗口、RNN、训练、梯度现象、双向和多层结构，并明确第 8/9 页勘误及推导补充。

网站新增语言模型实验室、补充笔记阅读和章节深链，接入首页/地图/导论/移动导航/资料入口。原 14 讲保留原编号和进度；manifest、checksum、资源白名单及校验器同步接入第 36 份 PDF。核心回归未删除，另加计数边界、梯度数值校验和浏览器回归。
''',
'website/RELEASE_NOTES.md': '''
## 1.1.0 — Language Models supplement (2026-11-03)

Add the user-provided 56-page PDF unchanged, a page-linked Chinese study note, and four local-computation teaching panels: embedding lookup, n-gram counts/generation, single-layer RNN/BPTT, and scalar recurrent sensitivity. Preserve the original 14-lecture numbering/progress and 35 core PDF checksums; explicitly register one supplemental PDF and note. Include separate source-derived content, derivations, errata and implementation limits. No new dependencies, domain or repository-visibility changes. Retain security audit gates and add independent finite-difference and browser regressions.
''',
'website/VERIFICATION.md': '''
## 2026-11-03：Language Models 补充单元

原件 SHA-256 与用户上传一致，56 页，未修改 PDF。笔记保留课件的 embedding → n-gram → 固定窗口 → RNN → 训练 → 梯度现象 → 双向/多层组织；页 8/9 的勘误、平滑、BPTT、标量敏感度均显式区分原述和补充。

验证覆盖：课件表的查表、n-gram 真计数/归一化/未见上下文/EOS/不跨句、固定种子的生成前缀、7 词两维状态 RNN 的 54 参数完整 BPTT 中心有限差分（含重复词）、共享权重前缀不变性、单步训练、标量初态导数；网页测试覆盖入口、播放与训练分离、状态重置、章节深链、PDF、移动布局。

边界：模型没有预训练，语料是 12 句自建样例；固定窗口、双向、多层和 LSTM 仅在笔记中解释，不声称全部实现。校验与数值测试不是逐页数学审定，也不保证训练步长总能使损失下降。
''',
'website/README.md': '''
## 语言模型补充模块（1.1.0）

入口 `#language-models`；在线笔记 `#supplement/language-models`，支持其标题深链。原 14 讲不重排；另收录 56 页 `03_langmodel.pdf` 和对应笔记，总资料增至 36 份 PDF。实验含真实计数/采样、共享 RNN 前向/完整 BPTT 及标量导数连乘，不新增依赖。原 35 份资料的摘要仍保持不变；新增资源必须通过显式清单、校验和与发布白名单。
''',
'website/DELIVERY.md': '''
## 1.1.0 内容扩展（2026-11-03）

以上 14 讲/35 PDF 的数字是历史基线。本版另外增加 1 个用户提供的语言模型补充单元（56 页 PDF 与学习笔记），合计 14 篇核心笔记 + 1 篇补充笔记、36 份 PDF。新增实验入口 `#language-models` 与阅读入口 `#supplement/language-models`。保留原编号、进度、依赖安全策略以及所有既有数学回归；原件哈希及本版新增验证范围见 VERIFICATION.md 与 source_manifest.tsv。
'''
}
for path, addition in additions.items(): write(path, read(path) + addition)
for path in ['notes/Lecture-01_Introduction.md', 'notes/Lecture-07_Transformer-and-LLM-I.md']:
    write(path, read(path) + '\n补充先修阅读：[[Supplement-Language-Models]]（词嵌入、n-gram、RNN 与 next-token 交叉熵；不改变本讲课程定位）。\n')
