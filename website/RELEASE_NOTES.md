# 1.0.1 稳定化说明 · 2026-11-01

基于 a0088cf，保留上一轮知识性修复及全部既有测试。

- 定向升级 Vitest 至 5.0.5、KaTeX 至 0.18.2，统一 Markdown 数学链的 KaTeX 副本。
- 交互公式与笔记共用显式 trust=false、展开与尺寸限制；笔记忽略原始 HTML。
- 增加安全渲染和全部 14 讲公式的回归测试。
- PR 和发布均阻断全部已知级别的 audit 告警；增加每周审计与 Dependabot 更新 PR。
- 固定 Actions SHA，收紧构建/部署权限，不持久保存 checkout 凭证。
- Pages 产物 version.json 记录版本与构建提交。

最终发布条件：PR 检查通过后 squash 合并，main 的完整生产门禁与 Pages deploy 成功。具体结果以对应 PR / Actions 为准。本文件不预先宣称尚未完成的检查通过。

安全范围、基线告警、版本选择及回退边界见根目录 SECURITY.md。课程 PDF、域名和公开范围不变。

## 1.1.0 — Language Models supplement (2026-11-03)

Add the user-provided 56-page PDF unchanged, a page-linked Chinese study note, and four local-computation teaching panels: embedding lookup, n-gram counts/generation, single-layer RNN/BPTT, and scalar recurrent sensitivity. Preserve the original 14-lecture numbering/progress and 35 core PDF checksums; explicitly register one supplemental PDF and note. Include separate source-derived content, derivations, errata and implementation limits. No new dependencies, domain or repository-visibility changes. Retain security audit gates and add independent finite-difference and browser regressions.
