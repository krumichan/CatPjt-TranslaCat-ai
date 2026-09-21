/** Configure the fresh QA Google user through the ordinary settings UI. */
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { ownedSpeakingBrowser, requireNormalOwner } from "./qa_owned_speaking_browser.mjs";

const campaign = process.env.QA_CAMPAIGN_DIR;
const feRoot = process.env.QA_FE_ROOT;
if (!campaign || !feRoot)
    throw new Error("Only an explicit owned QA Google profile is allowed");
const owned = ownedSpeakingBrowser(campaign);
const output = path.join(owned.output, "normal-setting-onboarding.json");
if (fs.existsSync(output)) throw new Error("Preserve prior onboarding evidence");
const require = createRequire(path.join(feRoot, "package.json"));
const { chromium } = require("@playwright/test");
const report = { status: "STARTED", providerCalls: 0, normalBrowserSetting: true, events: [] };
const save = (event, details = {}) => {
    report.events.push({ at: new Date().toISOString(), event, ...details });
    fs.writeFileSync(output, JSON.stringify(report, null, 2));
};
save("SETTINGS_UI_STARTED");
const context = await chromium.launchPersistentContext(owned.profile, {
    headless: false, executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    args: ["--no-first-run", "--no-default-browser-check"], viewport: { width: 1280, height: 900 },
});
try {
    const origin = "http://localhost:3000";
    const page = context.pages()[0] ?? await context.newPage();
    const response = await page.request.get(`${origin}/api/auth/session`);
    const session = response.ok() ? await response.json() : {};
    const publicId = requireNormalOwner(session, owned.expectedPublicId);
    await page.goto(`${origin}/ja/language-learning/settings`, { waitUntil: "domcontentloaded" });
    const selects = page.locator("label select");
    await selects.first().waitFor({ state: "visible", timeout: 30_000 });
    const before = await selects.evaluateAll((nodes) => nodes.slice(0, 2).map((node) => node.value));
    save("NORMAL_SETTINGS_VISIBLE", { publicId, before });
    await selects.nth(0).selectOption("ko");
    await selects.nth(1).selectOption("ja");
    const after = await selects.evaluateAll((nodes) => nodes.slice(0, 2).map((node) => node.value));
    if (after[0].toLowerCase() !== "ko" || after[1].toLowerCase() !== "ja")
        throw new Error("QA language settings not selected");
    const saveButton = page.locator("section button").filter({ hasText: /保存|Save/i }).first();
    await saveButton.waitFor({ state: "visible", timeout: 30_000 });
    if (!await saveButton.isEnabled()) throw new Error("Ordinary setting form not valid");
    await saveButton.click();
    await page.getByRole("status").waitFor({ state: "visible", timeout: 30_000 });
    save("NORMAL_SETTINGS_SAVED", { publicId, after, statusText: await page.getByRole("status").innerText() });
    await page.goto(`${origin}/ja/language-learning/speaking`, { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(5000);
    save("POST_ONBOARDING_SPEAKING", { url: page.url(), visibleText: (await page.locator("body").innerText()).slice(0, 1400) });
    report.status = "NORMAL_SETTINGS_SAVED";
} catch (error) {
    report.status = "FAILED_PARTIAL";
    save("SETTINGS_FAILED", { errorType: error?.name ?? "Error", errorMessage: String(error?.message ?? error).slice(0, 240) });
    throw error;
} finally {
    save("SETTINGS_UI_FINISHED", { status: report.status });
    await context.close();
}
