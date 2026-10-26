# DLOpt 2026 Interactive Learning Lab

通过动态实验、公式推导和交互式比较理解深度学习最优化。独立静态应用，计算在浏览器本地完成，不需要账户或服务器数据库。

## 开发与验证

要求 Node.js 22.12+（建议 Node 22 LTS）、npm；课程索引生成和原仓库检查需要 Python 3。

```bash
cd website
npm ci
npm run dev
# 浏览器打开终端输出地址，默认 http://127.0.0.1:5173/

npm run typecheck
npm run format:check
npm test
npx playwright install chromium
npm run e2e
npm run build
npm run preview
```

生产子路径验证（与 Pages 设置一致）：

```bash
BASE_PATH=/dlopt2026-learning-notes/ npm run build
E2E_PREVIEW=1 BASE_PATH=/dlopt2026-learning-notes/ TEST_BASE_PATH=/dlopt2026-learning-notes/ npm run e2e
```

Linux CI 用 `npx playwright install --with-deps chromium`。首次安装依赖与浏览器需要网络。开发服务器和预览服务器只监听本机。原资料在开发和构建后均通过 `resources/` 本地提供，原始笔记不被改写；原课程 Markdown 同时经 Vite raw import 由阅读器渲染。

```bash
# 从仓库根目录运行原有校验
python3 tools/check_notes.py
python3 tools/test_math_notes.py
# 笔记更新后重新提取结构化课程目录
python3 website/scripts/index-course.py
```

## 架构

- `src/content/course.json`：从 14 讲笔记提取的课程索引、学习目标、先修、PDF 与范围，生成脚本只读原资料。
- `src/content/modules.ts`：各实验目标、数学条件、常见误解、建议实验与核验范围。
- `src/content/gd.ts`：GD 逐步推导与预测题。
- `src/content/source-review.ts`：本轮实际读取的 PDF 抽查页，区分文本和原图证据。
- `src/engines/`：与 React 无关的纯计算引擎。二次函数及随机优化、矩阵/SVD/attention、零阶估计、分布式模拟、精度转换。
- `src/labs/`：实验界面，调用纯引擎；SVG 仅绘图，不生成算法数值。
- `src/components/`：离散播放与历史回退、参数控制、轨迹/曲线、矩阵、KaTeX、推导、自测、原笔记阅读与来源。
- `src/tests/`：手算参考、数值验证与源资料一致性；`e2e/`：桌面/移动端浏览器检查。

页面使用 hash 路由，适合静态目录部署与直接刷新；主题与自报学习进度存于 localStorage。若存储被禁用仍可使用。算法轨迹按离散步预计算，播放定时器只改变查看位置。减少动态效果设置下放慢播放，并停用装饰动画；不自动播放。

## 已实现实验

| 实验 | 真实计算范围 |
|---|---|
| GD | SPD 旋转二次目标、最优点与初值、GD / Heavy-ball / 固定动量 NAG、同目标同初值比较、三类指标、线性/对数轴、单步/回退、下降界数字核验、五步推导、三题 |
| SGD / Adam | 独立高斯样本、mini-batch、Momentum、完整偏差修正 Adam、16 次种子运行均值与经验波动区间、2000 次批噪声经验方差 |
| 线性代数 / 子空间 | 实际 SVD、线性几何、秩/条件数、正交投影和回映、残差、基与优化器状态计数 |
| 零阶 | 坐标前向/中心、球面单侧/双侧、高斯 MeZO，四次函数、真实查询值与次数、方向/误差/切片，尺度与查询噪声 |
| 分布式 | 独立本地目标、PSGD、先更新后环混合 DSGD、Local SGD、均匀平均、无偏 Random-k / Top-k（仅 PSGD 梯度）、逐节点状态与通信账本 |
| Attention / FlashAttention | 双头 Q/K/V、因果 mask、稳定行 softmax、拼接/投影，分块在线 m/ℓ/z、稠密数值对照 |
| 混合精度 | FP32 / FP16 / BF16 最近偶数舍入、次正规、上下溢、小更新丢失、缩放后还原 |
| LoRA / KV Cache | 小矩阵因子合成、参数计数、逐 token KV 形状与字节估算 |
| 资源比较 | Adam、无一阶矩 Adafactor、矩阵 Shampoo、Muon 动量核心的持久状态与复杂度账本 |

所有非 GD 模块是有明确范围的教学模型。SVD 投影不是完整 GaLore；资源比较不是完整 Shampoo / Muon 执行器；FlashAttention 不是 GPU kernel；KV Cache 是估算，不执行语言模型。不能据此推断真实训练峰值、吞吐或泛化。

## 数学与来源状态

课程整讲统一 `partially_verified`；仅 GD 明确范围内的递推、条件和下降推导为 `verified`，依据见模块数据与 `VERIFICATION.md`。`unverified` 保留用于未来未经核验的模块，不为凑状态分类人为降级或升级。

PDF 页码仅收录本轮实际读取的页。原笔记的其他页码仍在阅读器中显示，代表原记录，不能当作本轮独立核验。原笔记与资料的内容未被覆盖。原稿差异见 `VERIFICATION.md`；第 9 讲 Lite / MuonM 继续标记需确认。

## 添加新实验

1. 先阅读原笔记、课程范围与相关 PDF，确定一个边界清楚的可计算问题。
2. 在 `src/engines/` 写纯计算，先以独立手算/解析参考建立数值测试。
3. 在 `src/content/modules.ts` 声明目标、条件、误解、证据状态；已实际核对的页添加到 `source-review.ts`。
4. 在 `src/labs/` 组合参数、图形和状态。迭代实验复用 Playback；静态矩阵/资源实验通过参数直接重算，不添加没有语义的播放按钮。
5. 将入口注册到 `content/course.ts` 的 labNames 和 App 路由。需要推导时复用 Derivation，题目复用 Quiz。
6. 扩展 `e2e/` 的页面与参数测试，运行新旧所有校验。

## 部署

网站源码随仓库版本管理；尚未部署。远端 CI 状态以 GitHub Actions 对应提交的结果为准，不能将工作流文件存在视作检查或部署成功。

- 验证：`.github/workflows/website.yml` 在 push/PR/manual 运行类型检查、数值测试、构建、桌面/移动 E2E 和原有资料检查；原 `verify.yml` 保留。
- Pages：仓库 Settings → Pages → Source 选择 GitHub Actions，推送代码后手动触发 `Deploy learning lab to Pages`。工作流设置仓库子路径并上传 `website/dist`。
- 私有仓库的 Pages 可用性与站点可见性取决于账户/组织设置。构建会携带全部 35 份 PDF（约128 MiB），不依赖私有 GitHub raw 链接。
- 其他静态托管：上传 `dist/`；部署根路径可直接 `npm run build`，固定子路径用 `BASE_PATH=/your-path/ npm run build`。支持相对 base，使用 hash 路由无需服务器 fallback。

将站点放入仓库名以外的 Pages 自定义路径时应调整 BASE_PATH；仓库首页型 Pages 需要 `/`。只有实际部署后验证首页、带 hash 实验与 PDF 链接可访问，才能报告已上线。
