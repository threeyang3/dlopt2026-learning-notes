# DLOpt 2026 动态学习实验室交付报告

实施目录为仓库内 website/。原14讲Markdown、35份PDF和原数学回归脚本保持不变。

## 文件与架构

新增 website/ 独立 React + TypeScript + Vite 应用，包含 content/课程与证据、engines/纯数学、labs/实验、components/推导/播放/图形/笔记、tests/数值和内容测试、e2e/浏览器测试。详细文件见源码目录及 README。

根 README 增加网站入口；新增 `.github/workflows/website.yml` 和 `pages.yml`。`tools/check_notes.py` 仅调整遍历范围以排除 node_modules、dist、test-results 等生成目录，保留所有原资料、链接、数学定界符和SHA断言；新增 `tools/test_check_notes.py` 验证此范围及错误仍可被检出。

## 实际功能与计算

- 阶段 A/B：14讲地图、原笔记KaTeX阅读、先修/来源/进度、GD/HB/NAG真实二维递推、旋转SPD目标、条件数/步长/动量/初值/最优点、历史播放/单步/回退、三指标曲线、数值下降界、五步完整推导、三题预测。
- 阶段 C 教学范围：真实高斯SGD/mini-batch/Momentum/Adam、16种子均值波动、实际SVD/投影、五类差分估计、PSGD/DSGD/Local与两种稀疏梯度压缩。
- 阶段 D 教学范围：双头attention、在线softmax分块对照、浮点舍入与小更新、LoRA矩阵合成与KV缓存计数、四种优化器状态/复杂度对照。
- 阶段 E：键盘、减少动态效果、主题与移动布局检查；本地构建和测试、Pages配置与文档。远端部署未执行。

上述并非所有真实训练算法的完整复现。各模块的计算逻辑、适用条件和未完成项见 README 与 VERIFICATION。

## 验证结果

- TypeScript 检查通过，生产构建通过，格式检查通过。
- Vitest：25项通过，覆盖数学递推、独立解析参考、随机统计与课程来源一致性。
- Playwright：桌面/手机共12项通过，遍历11个实验入口、首页、来源页和14讲阅读器，检查公式、运行错误、参数、播放回退、预测题、本地PDF、参考链接、键盘、减少动态效果和主题持久化。
- 额外使用生产构建在 `/dlopt2026-learning-notes/` 子路径运行同一组12项E2E，全部通过；不把开发模式测试当作部署路径验证。
- 原资料校验通过：14讲、35份PDF、链接、数学定界符、SHA-256；原13组数值/内容回归通过；新增校验器范围测试2项通过。
- 已查看桌面/手机首页和深色GD截图，修复共同初值标签重叠与深色曲线对比度。
- 上述结果为本地验收；远端CI以GitHub Actions对应提交的实际运行结果为准。

## 运行与部署

```bash
cd website
npm ci
npm run dev
npm run typecheck
npm test
npx playwright install chromium
npm run e2e
npm run build
npm run preview
```

Pages仅配置手动工作流，尚无已验证的上线地址；源码上传与站点部署分别验收。静态构建包含原资料并使用hash路由，子路径由 BASE_PATH 指定。

## 下一阶段最有价值的工作

1. 把SGD条件期望下降、球体平滑、低秩投影误差等证明编排为与GD同等完整的可复用逐步推导。
2. 将GaLore状态换基、Shampoo/Muon完整更新做成小矩阵逐步执行器，以经过核对的具体实现版本为准。
3. 扩充分布式拓扑、更多维度/可调k、误差反馈及真实通信模型；再独立核对更多PDF与论文原文。
4. 在仓库Pages设置就绪后发布并实测线上hash入口、PDF及手机交互。

## 实际文件清单

修改：`README.md`、`tools/check_notes.py`。新增：

```text
.github/workflows/pages.yml
.github/workflows/website.yml
tools/test_check_notes.py
website/.gitignore
website/ARCHITECTURE.md
website/DELIVERY.md
website/README.md
website/VERIFICATION.md
website/e2e/core.spec.ts
website/index.html
website/package-lock.json
website/package.json
website/playwright.config.ts
website/scripts/index-course.py
website/src/App.tsx
website/src/components/Controls.tsx
website/src/components/Derivation.tsx
website/src/components/LearningContext.tsx
website/src/components/Math.tsx
website/src/components/Matrix.tsx
website/src/components/Notes.tsx
website/src/components/Plots.tsx
website/src/components/QueryPlot.tsx
website/src/components/Quiz.tsx
website/src/components/Sources.tsx
website/src/content/course.json
website/src/content/course.ts
website/src/content/gd.ts
website/src/content/modules.ts
website/src/content/optimizers.ts
website/src/content/source-review.ts
website/src/engines/distributed.ts
website/src/engines/matrix.ts
website/src/engines/optimization.ts
website/src/engines/precision.ts
website/src/engines/zeroth.ts
website/src/labs/AttentionLab.tsx
website/src/labs/DistributedLab.tsx
website/src/labs/GradientLab.tsx
website/src/labs/InferenceLab.tsx
website/src/labs/LinearLab.tsx
website/src/labs/PrecisionLab.tsx
website/src/labs/ResourcesLab.tsx
website/src/labs/StochasticLab.tsx
website/src/labs/ZerothLab.tsx
website/src/main.tsx
website/src/styles/app.css
website/src/tests/advanced.test.ts
website/src/tests/content.test.ts
website/src/tests/optimization.test.ts
website/tsconfig.json
website/vite.config.ts
```
