import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  use: {
    baseURL: "http://127.0.0.1:4173" + (process.env.TEST_BASE_PATH || "/"),
    trace: "retain-on-failure",
  },
  webServer: {
    command: process.env.E2E_PREVIEW
      ? "npm run preview -- --port 4173"
      : "npm run dev -- --port 4173",
    url: "http://127.0.0.1:4173" + (process.env.TEST_BASE_PATH || "/"),
    reuseExistingServer: !process.env.CI,
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    {
      name: "mobile",
      use: { ...devices["iPhone 13"], defaultBrowserType: "chromium" },
    },
  ],
});
