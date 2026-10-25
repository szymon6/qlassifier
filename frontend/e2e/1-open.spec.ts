// US-4 open / pick / read / share / simplicity, NFR-5 render time (TDD §9.2).
import { expect, openDashboard, test } from "./fixtures";

test.describe("open", () => {
  test("opens on the cached Breast Cancer leaderboard, Qlassifier emphasised, nothing below the fold", async ({
    page,
  }) => {
    await openDashboard(page);
    await expect(page.getByTestId("leaderboard")).toHaveAttribute("data-dataset", "breast_cancer");
    await expect(page.getByTestId("row-qlassifier")).toHaveAttribute("data-emphasis", "ours");
    await expect(page.getByRole("radio", { name: "Breast Cancer", exact: false })).toHaveAttribute(
      "aria-checked",
      "true",
    );
    for (const id of ["top-bar", "chips", "leaderboard", "noise", "cost", "footer"]) {
      const box = await page.getByTestId(id).boundingBox();
      expect(box, id).not.toBeNull();
      expect(box!.y + box!.height, id).toBeLessThanOrEqual(900);
      expect(box!.x + box!.width, id).toBeLessThanOrEqual(1440);
    }
    const size = await page.evaluate(() => ({
      h: document.documentElement.scrollHeight,
      w: document.documentElement.scrollWidth,
    }));
    expect(size.h).toBeLessThanOrEqual(900);
    expect(size.w).toBeLessThanOrEqual(1440);
  });

  test("renders the cached result in under a second (NFR-5)", async ({ page }) => {
    await openDashboard(page);
    const ms = await page.evaluate(
      () => performance.getEntriesByName("leaderboard-ready")[0]?.startTime ?? Infinity,
    );
    expect(ms).toBeLessThan(1000);
  });

  test("shows the three proof chips with numbers", async ({ page }) => {
    await openDashboard(page);
    for (const id of ["zz", "noise", "cost"]) {
      const chip = page.getByTestId(`chip-${id}`);
      await expect(chip).toBeVisible();
      await expect(chip).not.toHaveAttribute("data-value", "");
    }
    await expect(page.getByTestId("chip-zz")).toContainText("pp");
    await expect(page.getByTestId("chip-cost")).toContainText("×");
  });
});

test.describe("pick", () => {
  test("glyph tiles have accessible names; click and arrow keys switch the dataset and the URL", async ({
    page,
  }) => {
    await openDashboard(page);
    const tiles = page.getByRole("radiogroup", { name: "Dataset" }).getByRole("radio");
    await expect(tiles).toHaveCount(5);
    const names = await tiles.evaluateAll((els) => els.map((e) => e.getAttribute("aria-label")));
    expect(names.every((n) => n && n.length > 2)).toBe(true);
    await page.getByTestId("tile-sonar").click();
    await expect(page.getByTestId("leaderboard")).toHaveAttribute("data-dataset", "sonar");
    await expect(page).toHaveURL(/[?&]d=sonar/);
    await page.keyboard.press("ArrowRight");
    await expect(page.getByTestId("leaderboard")).toHaveAttribute("data-dataset", "heart");
    await expect(page).toHaveURL(/[?&]d=heart/);
    await page.keyboard.press("ArrowRight");
    await expect(page.getByTestId("leaderboard")).toHaveAttribute("data-dataset", "protein_angles");
    // Geometry dataset: the classical twin sits right under Qlassifier (rows tween for 240 ms:
    // poll until the order on screen has settled).
    await expect
      .poll(async () => {
        const order = await page
          .getByRole("listbox", { name: /Leaderboard/ })
          .getByRole("option")
          .evaluateAll((els) =>
            els.map((e) => [e.getAttribute("data-model"), Number(e.getBoundingClientRect().top)]),
          );
        order.sort((a, b) => (a[1] as number) - (b[1] as number));
        const ids = order.map((o) => o[0]);
        return ids[ids.indexOf("qlassifier") + 1];
      })
      .toBe("classical_twin");
    await page.keyboard.press("ArrowRight");
    await expect(page.getByTestId("leaderboard")).toHaveAttribute("data-dataset", "breast_cancer");
    await expect(page).not.toHaveURL(/[?&]d=/);
  });
});

test.describe("read", () => {
  test("leaderboard has error bars; noise and cost charts are present", async ({ page }) => {
    await openDashboard(page);
    expect(await page.locator("[data-ci]").count()).toBeGreaterThanOrEqual(10);
    expect(await page.locator("[data-dot]").count()).toBeGreaterThanOrEqual(10 * 10);
    await expect(page.getByTestId("tie-band").first()).toBeVisible();
    // A vertical SVG line has an empty box: check it is drawn, not "visible".
    await expect(page.getByTestId("majority-rule")).toBeAttached();
    await expect(page.getByTestId("noise").locator("[data-series]").first()).toBeVisible();
    await expect(page.getByTestId("cost").locator("g[data-model]").first()).toBeVisible();
    await expect(page.getByTestId("noise").getByRole("img")).toHaveAccessibleName(
      /Qlassifier loses at most/,
    );
    await expect(page.getByTestId("cost").getByRole("img")).toHaveAccessibleName(
      /Qlassifier needs/,
    );
  });

  test("hover highlights the model in every view; click pins (URL h); Esc clears", async ({
    page,
  }) => {
    await openDashboard(page);
    await page.getByTestId("row-qlassifier").hover();
    await expect(page.getByTestId("row-qlassifier")).toHaveAttribute("data-focus", "focus");
    await expect(page.getByTestId("row-zz_qsvc")).toHaveAttribute("data-focus", "dim");
    await expect(
      page.getByTestId("noise").locator('[data-model="qlassifier"]').first(),
    ).toHaveAttribute("data-focus", "focus");
    await expect(page.getByTestId("cost").locator('g[data-model="qlassifier"]')).toHaveAttribute(
      "data-focus",
      "focus",
    );
    await expect(page.getByTestId("cost").locator('g[data-model="zz_qsvc"]')).toHaveAttribute(
      "data-focus",
      "dim",
    );
    await expect(page.getByTestId("hover-card")).toContainText("95% CI");
    await page.getByTestId("row-qlassifier").click();
    await expect(page).toHaveURL(/[?&]h=qlassifier/);
    await page.mouse.move(5, 5);
    await expect(page.getByTestId("row-qlassifier")).toHaveAttribute("data-pinned", "true");
    await expect(page.getByTestId("row-zz_qsvc")).toHaveAttribute("data-focus", "dim");
    await page.keyboard.press("Escape");
    await expect(page).not.toHaveURL(/[?&]h=/);
    await expect(page.getByTestId("row-zz_qsvc")).toHaveAttribute("data-focus", "none");
  });

  test("hovering a chart line focuses the same model in the leaderboard", async ({ page }) => {
    await openDashboard(page);
    await page.getByTestId("cost").locator('g[data-model="zz_qsvc"]').hover();
    await expect(page.getByTestId("row-zz_qsvc")).toHaveAttribute("data-focus", "focus");
    await expect(page.getByTestId("row-qlassifier")).toHaveAttribute("data-focus", "dim");
  });

  test("the metric switch re-sorts the leaderboard", async ({ page }) => {
    await openDashboard(page);
    await page.getByRole("radio", { name: "ROC-AUC" }).click();
    await expect(page.getByTestId("leaderboard")).toHaveAttribute("data-metric", "auc");
    await expect(page).toHaveURL(/[?&]m=auc/);
    await expect(page.getByTestId("leaderboard").getByRole("heading")).toContainText("ROC-AUC");
  });
});

test.describe("share", () => {
  test("the URL keeps the view; opening it restores d, m, h", async ({ page }) => {
    await openDashboard(page, "?d=heart&m=f1");
    await page.getByTestId("row-mlp").click();
    await expect(page).toHaveURL(/h=mlp/);
    const url = page.url();
    expect(new URL(url).search).toBe("?d=heart&m=f1&h=mlp");
    await page.goto(url);
    await expect(page.locator('[data-testid="leaderboard"][data-ready="true"]')).toBeVisible();
    await expect(page.getByTestId("leaderboard")).toHaveAttribute("data-dataset", "heart");
    await expect(page.getByTestId("leaderboard")).toHaveAttribute("data-metric", "f1");
    await expect(page.getByTestId("row-mlp")).toHaveAttribute("aria-selected", "true");
    await expect(page.getByRole("radio", { name: "F1" })).toHaveAttribute("aria-checked", "true");
  });

  test("unknown m and h are ignored", async ({ page }) => {
    await openDashboard(page, "?m=rmse&h=nope");
    await expect(page.getByTestId("leaderboard")).toHaveAttribute("data-metric", "ba");
    await expect(page.locator('[aria-selected="true"]')).toHaveCount(0);
    await expect(page).not.toHaveURL(/[?&](m|h)=/);
  });
});

test.describe("simplicity", () => {
  test("exactly three controls outside the charts", async ({ page }) => {
    await openDashboard(page);
    const count = await page.evaluate(() => {
      const inChart = (el: Element) =>
        el.closest("[data-testid=leaderboard],[data-testid=noise],[data-testid=cost]");
      const groups = [...document.querySelectorAll('[role="radiogroup"]')].filter(
        (el) => !inChart(el),
      );
      const buttons = [
        ...document.querySelectorAll('button, a[href], input, select, textarea, [role="button"]'),
      ].filter(
        (el) =>
          !inChart(el) &&
          !el.closest('[role="radiogroup"]') &&
          !el.closest("[data-sonner-toaster]"),
      );
      return { groups: groups.length, buttons: buttons.length };
    });
    expect(count.groups + count.buttons).toBe(3);
    expect(count.groups).toBe(2);
  });
});
