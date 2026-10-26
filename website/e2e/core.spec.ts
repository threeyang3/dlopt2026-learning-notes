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
    "resources/materials/Lecture-05_Gradient-Descent/01_convexity-gradient-descent_slides.pdf",
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
