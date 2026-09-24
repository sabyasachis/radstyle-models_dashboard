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

test("persona tab shows real tokens, matched epoch-3 controls, and both kinds of OOD", async ({page, context}, testInfo) => {
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto("/");
  await page.getByRole("tab", {name: "Persona-token conditioning"}).click();
  await expect(page.locator("#match-count")).toHaveText("858 / 858");
  await expect(page.locator(".model-card")).toHaveCount(5);
  await expect(page.locator("#model-count")).toHaveText("5");
  await expect(page.locator("#tab-persona")).toHaveAttribute("aria-selected", "true");
  await expect(page.getByRole("tabpanel")).toHaveAttribute("aria-labelledby", "tab-persona");
  await expect(page.getByRole("tabpanel").locator("#comparison")).toBeVisible();
  await page.locator("#persona").selectOption("1");
  const joint = page.locator('[data-model="joint-conditioned"]');
  const cross = page.locator('[data-model="disjoint-conditioned"]');
  await expect(joint).toContainText("<|persona_001|>");
  await expect(joint).toContainText("Dataset ID");
  await expect(joint).toContainText("Persona seen (ID)");
  await expect(joint).toContainText("Final epoch 3 · checkpoint 923");
  await expect(cross).toContainText("<|persona_unknown|>");
  await expect(cross).toContainText("Dataset OOD");
  await expect(cross).toContainText("Persona unseen (OOD)");
  await expect(page.locator("#case-tags")).toContainText("Catalog ID persona");
  await expect(page.locator('[data-model="joint-unconditioned"]')).toContainText("No persona token");
  await expect(page.locator('[data-model="joint-unconditioned"]')).toContainText("Final epoch 3 · checkpoint 923");
  await page.screenshot({path: testInfo.outputPath("persona-desktop.png"), fullPage: true});
  await page.locator("#share").click();
  const shared = await page.evaluate(() => navigator.clipboard.readText());
  expect(new URL(shared).searchParams.get("experiment")).toBe("persona");
  expect(new URL(shared).searchParams.has("persona")).toBe(false);
  await page.locator("#persona").selectOption("41");
  await expect(joint).toContainText("<|persona_unknown|>");
  await expect(joint).toContainText("Persona unseen (OOD)");
  await page.getByLabel("Disjoint", {exact: true}).check();
  await page.locator("#reset").click();
  await expect(page.locator("#match-count")).toHaveText("829 / 829");
  await page.locator("#persona").selectOption("1");
  await expect(cross).toContainText("<|persona_001|>");
  await expect(cross).toContainText("Dataset ID");
  await expect(joint).toContainText("Dataset OOD");
  await page.getByLabel("Base", {exact: true}).uncheck();
  await expect(page.locator(".model-card")).toHaveCount(4);
  const selected = await page.locator("#case-title").textContent();
  await page.getByRole("tab", {name: "Original experiment"}).click();
  await expect(page.locator(".model-card")).toHaveCount(4);
  await expect(page.locator('[data-model="disjoint_nonthinking"]')).toContainText("Selected checkpoint 600");
  await expect(page.locator("#case-title")).toHaveText(selected);
  await page.getByRole("tab", {name: "Persona-token conditioning"}).click();
  await expect(page.locator('[data-model="base"]')).toHaveCount(0);
  await expect(page.locator("#case-title")).toHaveText(selected);
  await page.reload();
  await expect(page.locator("#tab-persona")).toHaveAttribute("aria-selected", "true");
  await expect(page.locator("#case-title")).toHaveText(selected);
  expect(errors).toEqual([]);
});

test("tabs support keyboard navigation, back/forward, and mobile deep links", async ({page}) => {
  await page.setViewportSize({width: 390, height: 844});
  await page.goto("/?experiment=persona&dataset=disjoint&case=disjoint-0042");
  await expect(page.locator("#case-title")).toHaveText("disjoint-0042");
  await expect(page.locator(".model-card")).toHaveCount(5);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.locator("#tab-persona").focus();
  await page.keyboard.press("ArrowLeft");
  await expect(page.locator("#tab-original")).toBeFocused();
  await expect(page.locator('[data-model="base_thinking"]')).toBeVisible();
  await page.goBack();
  await expect(page.locator("#tab-persona")).toHaveAttribute("aria-selected", "true");
  await expect(page.locator('[data-model="joint-conditioned"]')).toBeVisible();
  await page.goForward();
  await expect(page.locator("#tab-original")).toHaveAttribute("aria-selected", "true");
  await page.keyboard.press("End");
  await expect(page.locator("#tab-persona")).toBeFocused();
  await expect(page.locator(".model-card")).toHaveCount(5);
});

test("stale tab loads cannot overwrite the current experiment", async ({page}) => {
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  let requested;
  const pending = new Promise(resolve => { requested = resolve; });
  await page.route("**/data/persona/joint.json", async route => {
    requested();
    await gate;
    await route.continue();
  });
  await page.goto("/");
  await expect(page.locator(".model-card")).toHaveCount(4);
  await page.locator("#tab-persona").click();
  await pending;
  await page.locator("#tab-original").click();
  await expect(page.locator('[data-model="base_thinking"]')).toBeVisible();
  const loaded = page.waitForResponse("**/data/persona/joint.json");
  release();
  await loaded;
  await expect(page.locator("#tab-original")).toHaveAttribute("aria-selected", "true");
  await expect(page.locator(".model-card")).toHaveCount(4);
  await expect(page.locator('[data-model="joint-conditioned"]')).toHaveCount(0);
});

test("persona data failures can recover by returning to the original tab", async ({page}) => {
  await page.route("**/data/persona/joint.json", route => route.fulfill({status: 503, body: "Unavailable"}));
  await page.goto("/?experiment=persona");
  await expect(page.getByRole("alert")).toContainText("HTTP 503");
  await expect(page.locator("#workspace")).toBeHidden();
  await page.locator("#tab-original").click();
  await expect(page.locator(".model-card")).toHaveCount(4);
  await expect(page.getByRole("alert")).toBeHidden();
});
