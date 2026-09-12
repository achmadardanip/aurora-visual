import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "../tests/e2e",
  timeout: 45000,
  use: {
    baseURL: process.env.AURORA_BROWSER_URL || "http://127.0.0.1:5171",
    viewport: { width: 1280, height: 900 },
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  reporter: [
    ["list"],
    ["json", { outputFile: "../artifacts/reports/playwright.json" }],
  ],
});
