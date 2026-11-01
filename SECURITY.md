# 安全维护与稳定版验收

## 维护范围

1.0.1 是本轮稳定化版本。发布必须以对应提交的 PR 检查、生产子路径测试与 Pages 部署成功为准；版本号本身不证明检查成功，也不表示不存在未知漏洞。

## 2026-11-01 基线审计

基线为 `a0088cf2d7d013a405d950ab7237c85c96bb2ec0`，原始审计见 [Actions 41283414093](https://github.com/threeyang3/dlopt2026-learning-notes/actions/runs/41283414093)。完整依赖树报告 7 个受影响包计数（4 low、1 moderate、2 critical）；生产依赖报告 4 low。计数包含依赖传播，不等于七个互不相关漏洞。

| 依赖链 | 风险与作用范围 | 修复 |
|---|---|---|
| Vitest / @vitest/mocker | 开发测试服务的 mock 路径读取和注册问题，不是静态 Pages 的服务端 | 升级 Vitest 3.2.7 至 5.0.5 |
| Vitest / Tinypool | 已有原型污染等前置条件下的 worker 配置问题，影响 Node 测试工具链 | 升级测试框架以消除受影响依赖链；按最终锁文件审计验收 |
| KaTeX 与 Markdown 数学依赖 | 已有原型污染或可影响配置原型时，可能绕过 trust；并不说明本网站已被利用 | KaTeX 0.18.2，所有直接/间接副本统一；两个渲染入口显式 trust=false，并限制宏展开和尺寸 |

上游依据：[KaTeX](https://github.com/KaTeX/KaTeX/security/advisories/GHSA-238p-pmpm-9mq7)、[Tinypool worker options](https://github.com/advisories/GHSA-5gmw-xhrv-c9v3)、[Tinypool run options](https://github.com/tinylibs/tinypool/security/advisories/GHSA-85c8-ppgw-ccpr)、[Vitest mock 路径](https://github.com/advisories/GHSA-82fw-gwwq-j7x9)、[mock 注册](https://github.com/advisories/GHSA-qxg6-89mg-rqv8)。

`overrides.katex = "$katex"` 是有意的兼容策略：仅升级顶层仍可能让 remark/rehype 拉入受影响的旧版；不能降级数学插件来隐藏告警。所有公式和 Markdown 渲染都必须回归；上游支持安全范围后应重新评估 override。

Vitest 4.1.12 在本轮 npm 10 依赖解析中失败，未发布；最终选用符合现有 Node 22 / Vite 7 要求的 5.0.5。没有使用 `--force`、`--legacy-peer-deps`，也没有忽略或白名单化告警。其余直接依赖固定到基线锁文件中的版本；传递依赖按新的锁文件验收。

## 发布门禁

PR 和 main 发布均执行 `npm ci --ignore-scripts --no-audit`，随后执行 `npm run audit` 与 `npm run audit:runtime`。audit 的阈值为 low，任何已知级别的告警或服务错误都阻断。安装阶段的 `--no-audit` 仅避免重复查询，不取消随后必需的审计步骤。

继续执行格式、类型、所有单元与桌面/移动测试、生产构建、原笔记与 PDF 校验。安装不自动执行依赖生命周期脚本；未来确需某个安装脚本时须单独评估，不能整体放宽。

直接依赖精确固定，锁文件提交。Dependabot 每周提出 npm / Actions 更新 PR，不自动合并；独立每周审计检查未改动的锁文件是否出现新告警。

Actions 固定完整提交 SHA。checkout 不持久保存凭证，构建/测试只有 contents:read。pages:write 和 id-token:write 只在 main 的部署 job 中授予。没有修改管理员设置，也不声称已启用分支保护；维护者应在 Settings 中要求 PR 与成功检查，文件中的门禁不能阻止管理员绕过。

## 追溯与报告

Pages 构建完成后生成 version.json，包含版本和构建提交。发布成功与线上 HTTPS/CDN/设备验收是不同检查，不相互替代。需要回退时通过 PR 与同一门禁，不回到已知受影响依赖。

请勿在公开 issue 提交密钥或可直接滥用的漏洞细节。若私密漏洞报告已启用，优先使用；否则通过维护者已有的私密渠道联系。本文件不声称已开启该管理员功能。

域名、仓库可见性及课程 PDF 保持不变。公开资源仍可下载；安全检查不代替版权许可或访问控制，也不等于全部数学内容已经独立审定。
