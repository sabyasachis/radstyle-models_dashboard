import {test, expect} from "@playwright/test";

test("explores both datasets and catalog ID/OOD without cross-dataset relabeling", async ({page}) => {
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/");
  await expect(page.locator("#match-count")).toHaveText("858 / 858");
  await expect(page.locator(".model-card")).toHaveCount(4);
  await page.locator("#partition").selectOption("id");
  await expect(page.locator("#match-count")).toHaveText("683 / 858");
  await expect(page.locator("#case-tags")).toContainText("ID persona");
  await expect(page.locator('[data-model="disjoint_nonthinking"]')).toContainText("Cross-dataset");
  await expect(page.locator("#case-tags")).not.toContainText("OOD");
  await page.locator("#partition").selectOption("ood");
  await expect(page.locator("#match-count")).toHaveText("175 / 858");
  await page.getByLabel("Disjoint", {exact: true}).check();
  await expect(page.locator("#match-count")).toHaveText("154 / 829");
  await expect(page.locator("#case-tags")).toContainText("OOD persona");
  await page.locator("#reset").click();
  await expect(page.locator("#match-count")).toHaveText("829 / 829");
  await page.locator("#persona").selectOption("1");
  await expect(page.locator("#case-tags")).toContainText("Catalog rank 1 / 50");
  await page.locator("#search").fill("impossible-term-no-result-123456");
  await expect(page.locator("#empty")).toBeVisible();
  await page.locator("#reset").click();
  await expect(page.locator("#case-content")).toBeVisible();
  await page.getByLabel("Base thinking", {exact: true}).uncheck();
  await expect(page.locator(".model-card")).toHaveCount(3);
  await page.locator("#case-next").click();
  const title = await page.locator("#case-title").textContent();
  await page.reload();
  await expect(page.locator("#case-title")).toHaveText(title);
  expect(errors).toEqual([]);
});

test("deep links work under the project path and mobile layout does not overflow", async ({page}) => {
  await page.setViewportSize({width: 390, height: 844});
  await page.goto("/?dataset=disjoint&case=disjoint-0042");
  await expect(page.locator("#case-title")).toHaveText("disjoint-0042");
  await expect(page.locator("#findings")).not.toBeEmpty();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test("clinical text is not interpreted as markup", async ({page}) => {
  await page.route("**/data/joint.json", async route => {
    const response = await route.fetch();
    const data = await response.json();
    data.cases[0].findings = '<img src="https://invalid.example/" onerror="window.injected=true"><script>window.injected=true</script>';
    await route.fulfill({json: data});
  });
  await page.goto("/?dataset=joint&case=joined-0001");
  await expect(page.locator("#findings")).toContainText("<script>");
  await expect(page.locator("#findings img, #findings script")).toHaveCount(0);
  expect(await page.evaluate(() => window.injected)).toBeUndefined();
});

test("data errors are visible, not empty success", async ({page}) => {
  await page.route("**/data/joint.json", route => route.fulfill({status: 503, body: "Unavailable"}));
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("HTTP 503");
  await expect(page.locator("#workspace")).toBeHidden();
});
