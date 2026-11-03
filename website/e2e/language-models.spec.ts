import { test, expect } from "@playwright/test";

test("补充单元入口保留 14 讲进度，笔记深链与 PDF 可访问", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("./#map");
  await expect(page.locator(".course-map article")).toHaveCount(14);
  await page.getByRole("link", { name: "阅读补充笔记", exact: true }).click();
  await expect(page.locator(".note-content .katex").first()).toBeVisible();
  await expect(page.locator(".katex-error")).toHaveCount(0);
  const id = await page.locator(".note-content h2").filter({ hasText: "4. RNN" }).getAttribute("id");
  await page.goto("./#supplement/language-models/" + encodeURIComponent(id!));
  await page.reload();
  await expect.poll(async () => page.locator(".note-content h2").filter({ hasText: "4. RNN" }).evaluate((el) => Math.abs(el.getBoundingClientRect().top))).toBeLessThan(100);
  const pdf = await page.request.get("./resources/materials/Supplement-Language-Models/03_langmodel.pdf");
  expect(pdf.ok()).toBe(true);
  expect((await pdf.body()).subarray(0, 5).toString()).toBe("%PDF-");
  await page.goto("./#lecture/7");
  await page.locator(".note-content a[href='#supplement/language-models']").click();
  await expect(page).toHaveURL(/#supplement\/language-models$/);
  expect(errors).toEqual([]);
});

test("嵌入查表、键盘页签和小屏导航可用", async ({ page, isMobile }, info) => {
  await page.goto("./#language-models");
  if (isMobile) {
    await page.getByRole("button", { name: "打开导航菜单" }).click();
    await expect(page.getByRole("navigation", { name: "移动主导航" }).getByRole("link", { name: "语言模型与 RNN", exact: true })).toBeVisible();
    await page.getByRole("navigation", { name: "移动主导航" }).getByRole("link", { name: "语言模型与 RNN", exact: true }).click();
  }
  await expect(page.getByTestId("embedding-result")).toHaveAttribute("data-vector", "[0,-0.01,0.03,0.95]");
  await page.getByLabel("观察词", { exact: true }).selectOption("5");
  await expect(page.getByTestId("embedding-result")).toHaveAttribute("data-vector", "[0.01,0,-0.02,0.97]");
  await page.getByRole("tab", { name: "嵌入查表", exact: true }).focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("tab", { name: "n-gram 计数与生成" })).toBeFocused();
  for (const title of ["嵌入查表", "n-gram 计数与生成", "RNN 前向与训练", "梯度传播"]) {
    await page.getByRole("tab", { name: title, exact: true }).click();
    await expect(page.locator(".katex-error")).toHaveCount(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), title).toBe(true);
  }
  await page.screenshot({ path: info.outputPath("language-model-gradient.png"), fullPage: false });
});

test("n-gram 未定义分布不偷偷采样，显式平滑恢复生成", async ({ page }) => {
  await page.goto("./#language-models");
  await page.getByRole("tab", { name: "n-gram 计数与生成" }).click();
  const distribution = page.getByTestId("ngram-distribution");
  const p: number[] = JSON.parse((await distribution.getAttribute("data-probabilities"))!);
  expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  await page.getByLabel("n-gram 上下文").selectOption("2");
  await expect(distribution).toHaveAttribute("data-probabilities", "null");
  await expect(page.getByRole("button", { name: "采样下一个词" })).toBeDisabled();
  await page.getByRole("slider", { name: "平滑 α（补充）" }).fill("1");
  await expect(distribution).not.toHaveAttribute("data-probabilities", "null");
  await page.getByRole("button", { name: "采样下一个词" }).click();
  expect((await page.getByTestId("ngram-generated").innerText()).split(" ")).toHaveLength(4);
  await page.getByRole("button", { name: "重置生成" }).click();
  await expect(page.getByTestId("ngram-generated")).toHaveText("books i their");
});

test("RNN 播放不训练，训练更新损失，改参清空旧轨迹", async ({ page }, info) => {
  await page.goto("./#language-models");
  await page.getByRole("tab", { name: "RNN 前向与训练" }).click();
  const loss = page.getByTestId("rnn-loss");
  const before = Number(await loss.getAttribute("data-loss"));
  await expect(page.getByTestId("rnn-state")).toHaveAttribute("data-input", "0");
  await expect(page.getByTestId("rnn-state")).toHaveAttribute("data-target", "1");
  await page.getByRole("button", { name: "前进一步" }).click();
  await expect(page.getByTestId("rnn-state")).toHaveAttribute("data-input", "1");
  await expect(page.getByTestId("rnn-training-steps")).toHaveText("0");
  expect(Number(await loss.getAttribute("data-loss"))).toBe(before);
  await page.getByRole("button", { name: "播放", exact: true }).click();
  await expect(page.getByTestId("iteration")).not.toHaveText("1");
  await page.getByRole("button", { name: "训练一步（完整 BPTT）" }).click();
  await expect(page.getByTestId("iteration")).toHaveText("0");
  await expect(page.getByTestId("rnn-training-steps")).toHaveText("1");
  expect(Number(await loss.getAttribute("data-loss"))).toBeLessThan(before);
  await page.getByRole("slider", { name: "RNN 初始化种子" }).fill("10");
  await expect(page.getByTestId("rnn-training-steps")).toHaveText("0");
  await expect(page.getByTestId("iteration")).toHaveText("0");
  const p: number[] = JSON.parse((await page.getByTestId("rnn-probabilities").getAttribute("data-probabilities"))!);
  expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  await page.screenshot({ path: info.outputPath("language-model-rnn.png"), fullPage: false });
});

test("标量梯度显示有符号连乘及饱和效应", async ({ page }) => {
  await page.goto("./#language-models");
  await page.getByRole("tab", { name: "梯度传播", exact: true }).click();
  await page.getByRole("slider", { name: "传播步数 T" }).fill("3");
  const result = page.getByTestId("rnn-sensitivity");
  expect(Number(await result.getAttribute("data-value"))).toBeCloseTo(0.125, 12);
  await page.getByRole("slider", { name: "递归权重 w" }).fill("-1.2");
  expect(Number(await result.getAttribute("data-value"))).toBeCloseTo(-1.728, 12);
  await page.getByRole("slider", { name: "初始隐藏状态 h0" }).fill("0.7");
  expect(Math.abs(Number(await result.getAttribute("data-value")))).toBeLessThan(1.728);
});
