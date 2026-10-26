import { test, expect } from "@playwright/test";
test("GD 单步、回退、改参和自测", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("./#gd");
  await expect(
    page.getByRole("heading", { name: "梯度下降实验室", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "前进一步" }).click();
  await expect(page.getByTestId("iteration")).toHaveText("1");
  await page.getByRole("button", { name: "后退一步" }).click();
  await expect(page.getByTestId("iteration")).toHaveText("0");
  await page.getByRole("button", { name: "播放", exact: true }).click();
  await expect(page.getByTestId("iteration")).not.toHaveText("0");
  await page.getByRole("button", { name: "暂停", exact: true }).click();
  await page.getByRole("button", { name: "圆形", exact: true }).click();
  await expect(page.getByTestId("iteration")).toHaveText("0");
  await page.getByRole("button", { name: "下一步推导" }).click();
  await expect(page.getByText("步骤 2 · GD 更新定义")).toBeVisible();
  await page.getByLabel("最大曲率方向有非零初始分量时会发散").check();
  await page.getByRole("button", { name: "检查答案 1", exact: true }).click();
  await expect(page.locator("p[role=status]")).toContainText("回答正确");
  expect(errors).toEqual([]);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});
test("课程地图和原始笔记可读、进度保留", async ({ page }) => {
  await page.goto("./#map");
  await expect(page.locator(".course-map article")).toHaveCount(14);
  await page.locator(".course-map input").first().check();
  await page.reload();
  await expect(page.locator(".course-map input").first()).toBeChecked();
  await page.goto("./#lecture/5");
  await expect(page.locator(".note-content .katex").first()).toBeVisible();
  await expect(page.locator(".katex-error")).toHaveCount(0);
});
test("所有实验与 14 讲公式可渲染，无运行错误或横向溢出", async ({
  page,
}, info) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  for (const route of [
    "home",
    "intro",
    "gd",
    "sgd",
    "linear",
    "subspace",
    "zeroth",
    "distributed",
    "attention",
    "flash",
    "precision",
    "inference",
    "resources",
    "sources",
    ...Array.from({ length: 14 }, (_, i) => "lecture/" + (i + 1)),
  ]) {
    await page.goto("./#" + route);
    await expect(page.locator("main h1").first()).toBeVisible();
    await expect(page.locator(".katex-error")).toHaveCount(0);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      route + " 溢出",
    ).toBe(true);
    if (route === "home" || route === "gd")
      await page.screenshot({
        path: info.outputPath(route + ".png"),
        fullPage: false,
      });
  }
  expect(errors).toEqual([]);
});
test("随机、矩阵、Attention 与缓存控件驱动计算", async ({ page }) => {
  await page.goto("./#sgd");
  await page.getByLabel("随机算法").selectOption("Adam");
  await page.getByRole("button", { name: "前进一步" }).click();
  await expect(page.locator("main")).toContainText("修正一阶矩");
  await page.goto("./#linear");
  const rank = page.getByRole("slider", { name: "保留秩 r" });
  await rank.focus();
  await page.keyboard.press("End");
  expect(
    Number(
      await page.getByTestId("projection-error").getAttribute("data-error"),
    ),
  ).toBeLessThan(1e-12);
  await page.goto("./#attention");
  await page.getByRole("slider", { name: "迭代位置" }).fill("5");
  await expect(page.getByRole("heading", { name: "最终输出" })).toBeVisible();
  await page.goto("./#inference");
  await page.getByRole("button", { name: "前进一步" }).click();
  await expect(page.locator("main")).toContainText(
    "每层 K 与 V 各为 [1, 8, 1, 128]",
  );
});
test("本地原始资源与参考链接可访问", async ({ page, request }) => {
  await page.goto("./#lecture/5");
  const ref = page
    .locator(".note-content a")
    .filter({ hasText: "第 11 讲" })
    .first();
  if (await ref.count())
    await expect(ref).toHaveAttribute("href", /#lecture\//);
  const note = await request.get(
    "resources/notes/Lecture-05_Gradient-Descent.md",
  );
  expect(note.status()).toBe(200);
  expect(await note.text()).toContain("梯度下降");
  const pdf = await request.get(
    "resources/materials/Lecture-05_Gradient-Descent/" +
      "01_convexity-gradient-descent_slides.pdf",
  );
  expect(pdf.status()).toBe(200);
  expect(pdf.headers()["content-type"]).toContain("application/pdf");
  expect((await pdf.body()).subarray(0, 4).toString()).toBe("%PDF");
  await page.goto("./#lecture/6");
  const bib = page
    .locator(".note-content a")
    .filter({ hasText: "REFERENCES" })
    .first();
  await expect(bib).toHaveAttribute(
    "href",
    /resources\/notes\/REFERENCES\.md$/,
  );
});
test("键盘操作、减少动态效果与主题持久化", async ({ page }, info) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("./#gd");
  const step = page.getByRole("button", { name: "前进一步" });
  await step.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("iteration")).toHaveText("1");
  await page.getByRole("button", { name: "切换主题" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(page.getByTestId("iteration")).toHaveText("0");
  await page.screenshot({
    path: info.outputPath("gd-dark.png"),
    fullPage: false,
  });
});

test("键盘跳过导航保持当前路由并聚焦主要内容", async ({ page }) => {
  for (const route of ["gd", "lecture/5"]) {
    await page.goto("./#" + route);
    await page.reload();
    await page.keyboard.press("Tab");
    const skip = page.getByRole("link", { name: "跳至主要内容" });
    await expect(skip).toBeFocused();
    const before = page.url();
    await page.keyboard.press("Enter");
    await expect(page.locator("main")).toBeFocused();
    expect(page.url()).toBe(before);
    await expect(page.getByRole("heading", { name: "未找到页面" })).toHaveCount(
      0,
    );
  }
});
test("比较中 NAG 越界不截断 Heavy-ball，终止后指标保持最后真实状态", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("./#gd");
  await page.getByRole("slider", { name: "条件数 κ = L/μ" }).fill("40");
  const scrub = page.getByRole("slider", { name: "迭代位置" });
  await expect(scrub).toHaveAttribute("max", "120");
  await scrub.fill("110");
  const nag = page.getByTestId("run-Nesterov"),
    hb = page.getByTestId("run-Heavy-ball");
  await expect(nag).toContainText("数值发散保护");
  const last = Number(await nag.getAttribute("data-state-iteration"));
  expect(last).toBeLessThan(110);
  await expect(hb).toHaveAttribute("data-state-iteration", "110");
  await page.getByLabel("观察算法").selectOption("Nesterov");
  // Algorithm selection resets playback; seek past its independent end again.
  await scrub.fill("120");
  await expect(page.locator(".state-panel h2")).toHaveText(
    `第 ${last} 步的计算账本`,
  );
  await expect(hb).toHaveAttribute("data-state-iteration", "120");
  await expect(hb).toContainText("达到迭代预算");
  expect(
    (await page.getByTestId("curve-Nesterov").getAttribute("points"))!
      .trim()
      .split(/\s+/),
  ).toHaveLength(last + 1);
  expect(
    (await page.getByTestId("curve-Heavy-ball").getAttribute("points"))!
      .trim()
      .split(/\s+/),
  ).toHaveLength(121);
  await page.getByRole("button", { name: "后退一步" }).click();
  await expect(nag).toHaveAttribute("data-state-iteration", String(last));
  expect(errors).toEqual([]);
});
test("资源端点拒绝未发布的项目文件", async ({ request }) => {
  const response = await request.get("resources/website/package.json");
  expect(response.status()).toBe(404);
  const encoded = await request.get(
    "resources/notes/%2e%2e/website/package.json",
  );
  expect(encoded.status()).toBe(404);
});

test("跨讲笔记链接保留章节锚点并滚动到目标标题", async ({ page }) => {
  await page.goto("./#lecture/2");
  const link = page
    .locator('.note-content a[href^="#lecture/5/"]')
    .filter({ hasText: "第 5 讲" })
    .first();
  await expect(link).toBeVisible();
  await expect(link).toHaveAttribute("href", /^#lecture\/5\/2-/);
  await link.click();
  await expect(page).toHaveURL(/#lecture\/5\/2-/);

  const target = page.locator('.note-content h2[id="2-凸性-平滑性与曲率"]');
  await expect(target).toBeVisible();
  await expect
    .poll(() =>
      target.evaluate((element) => element.getBoundingClientRect().top),
    )
    .toBeGreaterThanOrEqual(0);
  await expect
    .poll(() =>
      target.evaluate((element) => element.getBoundingClientRect().top),
    )
    .toBeLessThan(300);
});

test("收敛曲线播放时纵轴范围和既有点位置保持稳定", async ({ page }) => {
  await page.goto("./#gd");
  const scrub = page.getByRole("slider", { name: "迭代位置" });
  const chart = page.locator("svg.curve").first();
  const gd = page.getByTestId("curve-GD");

  await scrub.fill("40");
  const domain = await chart.getAttribute("data-y-domain");
  const ticks = await chart.locator("text").allTextContents();
  const prefix = (await gd.getAttribute("points"))!.trim().split(/\s+/);

  await scrub.fill("80");
  expect(await chart.getAttribute("data-y-domain")).toBe(domain);
  expect(await chart.locator("text").allTextContents()).toEqual(ticks);
  const later = (await gd.getAttribute("points"))!.trim().split(/\s+/);
  expect(later.slice(0, prefix.length)).toEqual(prefix);
});

test("课程导论展示四阶段，并明确第 1 讲的关联实验", async ({ page }) => {
  await page.goto("./#intro");
  await expect(
    page.getByRole("heading", { name: "从优化问题到高效大模型训练" }),
  ).toBeVisible();
  await expect(page.locator(".intro-stages article")).toHaveCount(4);
  await page.getByRole("link", { name: "从第 1 讲开始" }).click();
  await expect(page).toHaveURL(/#lecture\/1$/);
  await expect(
    page.getByRole("link", { name: "关联实验：Attention" }),
  ).toHaveAttribute("href", "#attention");
  await expect(
    page.getByRole("link", { name: "课程导论与学习路线" }),
  ).toHaveAttribute("href", "#intro");
});


test("第 14 讲 LoRA 与 LoRA+ 标题 ID 唯一，锚点精确导航", async ({
  page,
}) => {
  await page.goto("./#lecture/14");
  const original = page.locator('.note-content h3[id="lora"]');
  const plus = page.locator('.note-content h3[id="lora-2"]');
  await expect(original).toHaveText("LoRA");
  await expect(plus).toHaveText("LoRA+");
  expect(
    await page.locator(".note-content [id]").evaluateAll((nodes) => {
      const ids = nodes.map((node) => node.id);
      return new Set(ids).size === ids.length;
    }),
  ).toBe(true);

  await page.goto("./#lecture/14/lora-2");
  await expect(plus).toBeVisible();
  await expect
    .poll(() =>
      plus.evaluate((element) => element.getBoundingClientRect().top),
    )
    .toBeGreaterThanOrEqual(0);
  await expect
    .poll(() =>
      plus.evaluate((element) => element.getBoundingClientRect().top),
    )
    .toBeLessThan(300);

  await page.reload();
  await expect(plus).toBeVisible();
  await expect
    .poll(() =>
      plus.evaluate((element) => element.getBoundingClientRect().top),
    )
    .toBeLessThan(300);
});
