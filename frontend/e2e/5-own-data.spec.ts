// Own data, run from code, shown in the dashboard (README "Dashboard and benchmark"; TDD §6.2 "saved from code").
// Real flow: `qlassifier benchmark my_data.csv` in a fresh folder, then `qlassifier ui` started in that folder reads
// ./qlassifier-benchmark/ by default. Own server on its own port, so the other specs keep their five tiles.
import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { copyFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { expect, test } from "./fixtures";

// CI installs the wheel in a fresh venv; locally `qlassifier` from the active venv.
const CLI = process.env.E2E_QLASSIFIER ?? "qlassifier";
const PORT = Number(process.env.E2E_OWN_DATA_PORT ?? 8766);
const BASE = `http://127.0.0.1:${PORT}`;

let folder = "";
let server: ChildProcess | null = null;

test.describe("own data", () => {
  test.beforeAll(async () => {
    test.setTimeout(300_000);
    folder = mkdtempSync(path.join(tmpdir(), "qlassifier-e2e-"));
    copyFileSync(path.join(__dirname, "data", "my_data.csv"), path.join(folder, "my_data.csv"));
    const bench = spawnSync(
      CLI,
      ["benchmark", "my_data.csv", "--target", "target", "--seeds", "2", "--jobs", "2"],
      { cwd: folder, encoding: "utf8", timeout: 280_000 },
    );
    expect(bench.status, bench.stderr).toBe(0);
    server = spawn(CLI, ["ui", "--no-browser", "--port", String(PORT), "--seeds", "2"], {
      cwd: folder,
      stdio: "ignore",
    });
    await expect
      .poll(
        async () => {
          try {
            return (await fetch(`${BASE}/api/v1/meta`)).status;
          } catch {
            return 0;
          }
        },
        { timeout: 60_000 },
      )
      .toBe(200);
  });

  test.afterAll(() => {
    server?.kill("SIGINT");
    if (folder) rmSync(folder, { recursive: true, force: true });
  });

  test("a CSV benchmarked from the CLI shows as a tile with its leaderboard", async ({ page }) => {
    await page.goto(`${BASE}/`);
    await expect(page.locator('[data-testid="leaderboard"][data-ready="true"]')).toBeVisible();
    const tile = page.getByRole("radio", { name: "my_data.csv", exact: false });
    await expect(tile).toBeVisible();
    await expect(tile).toHaveAttribute("aria-checked", "false");
    await tile.click();
    await expect(page.getByTestId("leaderboard")).toHaveAttribute("data-dataset", "my_data");
    await expect(page.getByTestId("row-qlassifier")).toHaveAttribute("data-emphasis", "ours");
    await expect(page.getByTestId("row-zz_qsvc")).toBeVisible();
    await expect(page).toHaveURL(/[?&]d=my_data/);
  });

  test("a link to the saved result opens it directly", async ({ page }) => {
    await page.goto(`${BASE}/?d=my_data`);
    await expect(page.locator('[data-testid="leaderboard"][data-ready="true"]')).toBeVisible();
    await expect(page.getByTestId("leaderboard")).toHaveAttribute("data-dataset", "my_data");
    await expect(page.getByRole("radio", { name: "my_data.csv", exact: false })).toHaveAttribute(
      "aria-checked",
      "true",
    );
  });
});
