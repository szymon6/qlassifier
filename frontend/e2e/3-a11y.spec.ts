// US-4 a11y: axe-core 0 AA violations in both themes, keyboard-only path, reduced motion (TDD §8.6, §9.2).
import AxeBuilder from "@axe-core/playwright";
import { expect, openDashboard, test } from "./fixtures";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"];

for (const theme of ["light", "dark"] as const) {
  test(`axe-core finds no WCAG AA violations (${theme})`, async ({ page }) => {
    await page.emulateMedia({ colorScheme: theme, reducedMotion: "reduce" });
    await openDashboard(page);
    await expect(page.locator("html")).toHaveClass(theme === "dark" ? /dark/ : /^(?!.*dark).*/);
    const idle = await new AxeBuilder({ page }).withTags(TAGS).analyze();
    expect(
      idle.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target).join(" ")}`),
    ).toEqual([]);
    // Focused state: hover card + dimmed rows.
    await page.getByTestId("row-qlassifier").hover();
    const focused = await new AxeBuilder({ page }).withTags(TAGS).analyze();
    expect(focused.violations.map((v) => v.id)).toEqual([]);
    // Geometry dataset (twin row, Ramachandran tile).
    await page.getByTestId("tile-protein_angles").click();
    await expect(page.getByTestId("leaderboard")).toHaveAttribute("data-dataset", "protein_angles");
    const protein = await new AxeBuilder({ page }).withTags(TAGS).analyze();
    expect(protein.violations.map((v) => v.id)).toEqual([]);
  });
}

test("the theme button flips light and dark and is remembered", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" });
  await openDashboard(page);
  const theme = page.getByRole("button", { name: "Dark theme" });
  await expect(theme).toHaveAttribute("aria-pressed", "false");
  await theme.click();
  await expect(page.locator("html")).toHaveClass(/dark/);
  await expect(theme).toHaveAttribute("aria-pressed", "true");
  await page.reload();
  await expect(page.locator("html")).toHaveClass(/dark/);
  await page.getByRole("button", { name: "Dark theme" }).click();
  await expect(page.locator("html")).not.toHaveClass(/dark/);
});

test("keyboard only: every control in order, arrows inside groups, Enter / Space, Esc", async ({
  page,
}) => {
  await openDashboard(page);
  await page.locator("body").focus();
  const order: string[] = [];
  for (let i = 0; i < 6; i++) {
    await page.keyboard.press("Tab");
    order.push(
      await page.evaluate(() => {
        const el = document.activeElement as HTMLElement | null;
        if (!el) return "none";
        const group = el.closest('[role="radiogroup"]')?.getAttribute("aria-label");
        if (group) return `radio:${group}`;
        if (el.getAttribute("role") === "option") return "row";
        const chart = el.closest("[data-testid]")?.getAttribute("data-testid");
        if (el.getAttribute("role") === "img") return `chart:${chart}`;
        return `button:${el.getAttribute("aria-label")}`;
      }),
    );
  }
  expect(order).toEqual([
    "radio:Dataset",
    "radio:Metric",
    "button:Dark theme",
    "row",
    "chart:noise",
    "chart:cost",
  ]);
  // Visible focus ring on a row.
  await page.keyboard.press("Shift+Tab");
  await page.keyboard.press("Shift+Tab");
  const outline = await page.evaluate(() => {
    const el = document.activeElement as HTMLElement;
    const s = getComputedStyle(el);
    return { role: el.getAttribute("role"), shadow: s.boxShadow, outline: s.outlineStyle };
  });
  expect(outline.role).toBe("option");
  expect(outline.shadow !== "none" || outline.outline !== "none").toBe(true);
  // Rows: arrows move the focus (linked highlight), Enter pins, Esc clears.
  const first = await page.evaluate(() => document.activeElement?.getAttribute("data-model"));
  await page.keyboard.press("ArrowDown");
  const second = await page.evaluate(() => document.activeElement?.getAttribute("data-model"));
  expect(second).not.toBe(first);
  await expect(page.getByTestId(`row-${second}`)).toHaveAttribute("data-focus", "focus");
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(new RegExp(`[?&]h=${second}`));
  await page.keyboard.press("Escape");
  await expect(page).not.toHaveURL(/[?&]h=/);
  // Groups: arrows select inside the dataset tiles; Space selects in the metric switch.
  await page
    .getByRole("radiogroup", { name: "Dataset" })
    .getByRole("radio", { checked: true })
    .focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByTestId("leaderboard")).toHaveAttribute("data-dataset", "ionosphere");
  await page.keyboard.press("Tab");
  await page.keyboard.press("ArrowRight");
  await expect(page.getByTestId("leaderboard")).toHaveAttribute("data-metric", "acc");
  await page.keyboard.press("ArrowLeft");
  await page.keyboard.press("Space");
  await expect(page.getByTestId("leaderboard")).toHaveAttribute("data-metric", "ba");
});

test("controls are at least 32 px", async ({ page }) => {
  await openDashboard(page);
  const small = await page.evaluate(
    () =>
      [...document.querySelectorAll('header button, header [role="radio"]')]
        .map((el) => el.getBoundingClientRect())
        .filter((r) => r.width < 32 || r.height < 32).length,
  );
  expect(small).toBe(0);
});

test("reduced motion: no tweening on data changes", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await openDashboard(page);
  await expect(page.getByTestId("dashboard")).toHaveAttribute("data-reduced-motion", "true");
  await page.getByRole("radio", { name: "accuracy", exact: true }).click();
  const tweening = await page.evaluate(() => document.documentElement.dataset.tweening ?? "no");
  expect(tweening).toBe("no");
});

test("with motion allowed, data changes tween briefly", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await openDashboard(page);
  await expect(page.getByTestId("dashboard")).toHaveAttribute("data-reduced-motion", "false");
  await page.evaluate(() => {
    (window as unknown as { __tweened: boolean }).__tweened = false;
    new MutationObserver(() => {
      if (document.documentElement.dataset.tweening)
        (window as unknown as { __tweened: boolean }).__tweened = true;
    }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-tweening"] });
  });
  await page.getByRole("radio", { name: "accuracy", exact: true }).click();
  await expect
    .poll(() => page.evaluate(() => (window as unknown as { __tweened: boolean }).__tweened))
    .toBe(true);
  await expect(page.locator("html")).not.toHaveAttribute("data-tweening", "true");
});
