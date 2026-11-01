import { test, expect } from "@playwright/test";

test("追加未来 token 保持模型和因果前缀输出", async ({ page }) => {
  await page.goto("./#attention");
  await page.getByRole("slider", { name: "Token 数量 N" }).fill("2");
  await page.getByRole("slider", { name: "迭代位置" }).fill("5");
  const result = page.getByTestId("attention-output");
  const before: number[][] = JSON.parse(
    (await result.getAttribute("data-output"))!,
  );
  await page.getByRole("slider", { name: "Token 数量 N" }).fill("6");
  await page.getByRole("slider", { name: "迭代位置" }).fill("5");
  const after: number[][] = JSON.parse(
    (await result.getAttribute("data-output"))!,
  );
  expect(after).toHaveLength(6);
  before.forEach((row, i) =>
    row.forEach((v, j) => expect(after[i][j]).toBeCloseTo(v, 12)),
  );
  await expect(page.getByRole("slider", { name: "权重随机种子" })).toHaveValue(
    "42",
  );
  await expect(page.getByRole("slider", { name: "输入序列种子" })).toHaveValue(
    "42",
  );
});

test("零阶坐标查询方向与实际扰动匹配，随机模式不保留旧坐标", async ({
  page,
}) => {
  await page.goto("./#zeroth");
  const pair = page.getByTestId("query-pair");
  await expect(pair).toHaveAttribute("data-direction", "[1,0]");
  await page.getByLabel("查看坐标查询").selectOption("1");
  await expect(pair).toHaveAttribute("data-direction", "[0,1]");
  const plus: number[] = JSON.parse((await pair.getAttribute("data-plus"))!);
  const minus: number[] = JSON.parse((await pair.getAttribute("data-minus"))!);
  expect(plus[0]).toBe(1);
  expect(minus[0]).toBe(1);
  expect(plus[1]).toBeCloseTo(0.71, 12);
  expect(minus[1]).toBeCloseTo(0.69, 12);
  await expect(pair).toContainText("当前坐标方向 e2");
  await page.getByLabel("零阶估计器").selectOption("球面双侧");
  await expect(page.getByLabel("查看坐标查询")).toHaveCount(0);
  await expect(pair).toContainText("随机方向 u");
  await page.getByLabel("零阶估计器").selectOption("坐标中心");
  await expect(pair).toHaveAttribute("data-direction", "[1,0]");
});

test("分布式同质无噪声预设消除目标差异，而非仅共享最优点", async ({ page }) => {
  await page.goto("./#distributed");
  await page.getByRole("button", { name: "同质无噪声对照" }).click();
  for (const name of ["本地最优点分散程度", "曲率差异", "每坐标梯度噪声标准差"])
    await expect(page.getByRole("slider", { name, exact: true })).toHaveValue(
      "0",
    );
  await page.getByRole("slider", { name: "迭代位置" }).fill("3");
  const first = page.getByTestId("worker-0");
  const parameters = await first.getAttribute("data-parameters");
  const gradient = await first.getAttribute("data-gradient");
  for (let i = 1; i < 4; i++) {
    await expect(page.getByTestId(`worker-${i}`)).toHaveAttribute(
      "data-parameters",
      parameters!,
    );
    await expect(page.getByTestId(`worker-${i}`)).toHaveAttribute(
      "data-gradient",
      gradient!,
    );
  }
  await page.getByRole("slider", { name: "曲率差异", exact: true }).fill("1");
  await expect(page.getByTestId("iteration")).toHaveText("0");
  await page.getByRole("button", { name: "前进一步" }).click();
  expect(
    await page.getByTestId("worker-0").getAttribute("data-gradient"),
  ).not.toBe(await page.getByTestId("worker-1").getAttribute("data-gradient"));
});
