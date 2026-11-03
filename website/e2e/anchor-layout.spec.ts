import { test, expect } from "@playwright/test";

test("慢字体下深链刷新和历史返回仍对齐标题", async ({ page }) => {
  await page.route("**/*.woff2", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 200));
    await route.continue();
  });
  const id = "4-rnn-语言模型-共享参数-逐步更新状态-第-26-29-页";
  const heading = page
    .locator(".note-content h2")
    .filter({ hasText: "4. RNN" });
  const aligned = async () => {
    await expect(heading).toHaveAttribute("id", id);
    await expect
      .poll(async () =>
        heading.evaluate((el) => Math.abs(el.getBoundingClientRect().top)),
      )
      .toBeLessThan(100);
    // innerWidth grows with mobile overflow; compare against the emulated device.
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth),
    ).toBeLessThanOrEqual(page.viewportSize()!.width);
  };
  await page.goto("./#supplement/language-models/" + encodeURIComponent(id));
  await aligned();
  await page.reload();
  await aligned();
  await page.goto("./#map");
  await page.goBack();
  await aligned();
});
