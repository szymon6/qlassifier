// US-5: the dashboard shows the API's error text as a toast (TDD §6.5, §8.7, §9.2).
import { expect, test } from "./fixtures";

test("an unknown dataset in the URL shows the API error text and falls back to the default", async ({
  page,
  request,
}) => {
  const api = await (await request.get("/api/v1/results/iris")).json();
  expect(api.error.code).toBe("unknown_dataset");
  await page.goto("/?d=iris");
  const toast = page.locator('[data-testid="toast"][data-code="unknown_dataset"]');
  await expect(toast).toBeVisible();
  await expect(toast).toHaveAttribute("role", "status");
  await expect(toast.getByTestId("toast-message")).toHaveText(api.error.message);
  await expect(toast.getByTestId("toast-hint")).toHaveText(api.error.hint);
  await expect(page.locator('[data-testid="leaderboard"][data-ready="true"]')).toBeVisible();
  await expect(page.getByTestId("leaderboard")).toHaveAttribute("data-dataset", "breast_cancer");
  await expect(page).not.toHaveURL(/[?&]d=/);
});

test("a stopped server shows 'Server stopped.'", async ({ page }) => {
  // The page loads, but every API call fails: the server is gone.
  await page.route("**/api/v1/**", (route) => route.abort("connectionrefused"));
  await page.goto("/");
  const toast = page.locator('[data-testid="toast"][data-code="server_stopped"]');
  await expect(toast.getByTestId("toast-message")).toHaveText("Server stopped.");
  await expect(toast.getByTestId("toast-hint")).toHaveText("Run `qlassifier ui` again.");
});
