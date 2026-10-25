// Captures the dashboard at 1440 × 900 for visual review and the README hero (TDD §8, US-4).
//   node scripts/screenshots.mjs [baseUrl] [outDir]
// Needs a running `qlassifier ui --no-browser --port 8765` (default base URL).
// Writes: light.png, dark.png, protein-light.png, protein-dark.png, running-light.png, running-dark.png.
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { chromium } from "@playwright/test";

const base = (process.argv[2] ?? "http://127.0.0.1:8765").replace(/\/$/, "");
const out = resolve(process.argv[3] ?? "e2e/screenshots");
await mkdir(out, { recursive: true });

const browser = await chromium.launch();

async function open(theme, query = "") {
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 2,
    colorScheme: theme,
    reducedMotion: "reduce",
  });
  const page = await context.newPage();
  await page.goto(`${base}/${query}`);
  await page.locator('[data-testid="leaderboard"][data-ready="true"]').waitFor();
  await page.evaluate(() => document.fonts.ready);
  return { context, page };
}

async function shot(page, name) {
  await page.mouse.move(0, 0);
  await page.screenshot({ path: `${out}/${name}.png` });
  console.log(`${out}/${name}.png`);
}

for (const theme of ["light", "dark"]) {
  const { context, page } = await open(theme);
  await shot(page, theme);
  await context.close();
  const protein = await open(theme, "?d=protein_angles");
  await shot(protein.page, `protein-${theme}`);
  await protein.context.close();
}

// Running: start a run on Sonar and capture once a third of the tasks are done.
for (const theme of ["light", "dark"]) {
  const { context, page } = await open(theme, "?d=sonar");
  await page.getByRole("button", { name: "Run benchmark" }).click();
  await page.waitForFunction(() => {
    const el = document.querySelector('[data-testid="run-progress"]');
    const done = Number(el?.getAttribute("data-done") ?? 0);
    const total = Number(el?.getAttribute("data-total") ?? 0);
    return total > 0 && done / total >= 0.35;
  });
  await page.waitForTimeout(300);
  await shot(page, `running-${theme}`);
  await page.getByRole("button", { name: "Stop run" }).click();
  await page.locator('[data-testid="run"][data-state="stopped"]').waitFor();
  await context.close();
}

await browser.close();
