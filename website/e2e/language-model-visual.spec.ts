import { test, expect } from "@playwright/test";

test("语言模型各面板保留可读视觉证据", async ({ page }, info) => {
  await page.goto("./#language-models");
  const titles = [
    "嵌入查表",
    "n-gram 计数与生成",
    "RNN 前向与训练",
    "梯度传播",
  ];
  for (let i = 0; i < titles.length; i++) {
    await page.getByRole("tab", { name: titles[i], exact: true }).click();
    const panel = page.getByRole("tabpanel");
    await expect(panel).toBeVisible();
    await page.evaluate(() => document.fonts.ready);
    await expect(panel.locator(".katex-error")).toHaveCount(0);
    await panel.screenshot({ path: info.outputPath(`panel-${i}.png`) });
  }
});
