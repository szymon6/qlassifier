// Shared e2e fixtures (TDD §9.2).
// - NFR-2: every request of every test must stay on 127.0.0.1 (auto fixture, asserted per test).
// - Helpers wait on events and attributes, never on sleeps (TDD §12.1: e2e timing flakiness).
import { expect, test as base, type APIRequestContext, type Page } from "@playwright/test";

export const test = base.extend<{ loopbackOnly: void }>({
  loopbackOnly: [
    async ({ context }, use) => {
      const outside: string[] = [];
      context.on("request", (request) => {
        const url = new URL(request.url());
        if (url.protocol === "data:" || url.protocol === "blob:" || url.protocol === "about:")
          return;
        if (url.hostname !== "127.0.0.1") outside.push(request.url());
      });
      await use();
      expect(outside, "requests that left 127.0.0.1 (NFR-2)").toEqual([]);
    },
    { auto: true },
  ],
});

export { expect };

/** Waits until the server has no active run (runs are server-wide). */
export async function waitIdle(request: APIRequestContext, timeoutMs = 240_000): Promise<void> {
  await expect
    .poll(
      async () => {
        const meta = await (await request.get("/api/v1/meta")).json();
        const run = meta.active_run;
        if (run && (run.status === "running" || run.status === "cancelling")) {
          await request.post(`/api/v1/runs/${run.id}/cancel`, { data: {} });
          return run.status;
        }
        return "idle";
      },
      { timeout: timeoutMs, intervals: [250, 500, 1000] },
    )
    .toBe("idle");
}

/** Opens the dashboard and waits for the first leaderboard paint. */
export async function openDashboard(page: Page, query = ""): Promise<void> {
  await page.goto(`/${query}`);
  await expect(page.locator('[data-testid="leaderboard"][data-ready="true"]')).toBeVisible();
}

export function progress(page: Page) {
  return page.getByRole("progressbar", { name: "Run progress" });
}

/** done / total of the run ring. */
export async function runProgress(page: Page): Promise<{ done: number; total: number }> {
  const el = progress(page);
  return {
    done: Number(await el.getAttribute("aria-valuenow")),
    total: Number(await el.getAttribute("aria-valuemax")),
  };
}
