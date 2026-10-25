import { defineConfig, devices } from "@playwright/test";

// End-to-end tests against the packaged app (TDD §9.2): `qlassifier ui --no-browser --port 8765 --seeds 2`.
// CI starts the server from the installed wheel; locally Playwright starts it unless one is running.
const PORT = Number(process.env.E2E_PORT ?? 8765);
const BASE = process.env.E2E_BASE_URL ?? `http://127.0.0.1:${PORT}`;
const SERVER = process.env.E2E_SERVER_CMD ?? `qlassifier ui --no-browser --port ${PORT} --seeds 2`;

export default defineConfig({
  testDir: "e2e",
  // Runs are server-wide state (one at a time): one worker, files in order.
  workers: 1,
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  timeout: 60_000,
  expect: { timeout: 10_000 },
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : [["list"]],
  use: {
    baseURL: BASE,
    viewport: { width: 1440, height: 900 },
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } },
    },
  ],
  webServer: {
    command: SERVER,
    url: `${BASE}/api/v1/meta`,
    reuseExistingServer: true,
    timeout: 60_000,
    stdout: "ignore",
    stderr: "pipe",
  },
});
