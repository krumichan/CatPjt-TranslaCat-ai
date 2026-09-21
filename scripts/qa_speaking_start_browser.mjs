/** Start one owned QA Speaking mode through normal Google-authenticated Chrome. */
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { ownedSpeakingBrowser, requireNormalOwner } from "./qa_owned_speaking_browser.mjs";

const feRoot = process.env.QA_FE_ROOT;
const campaign = process.env.QA_CAMPAIGN_DIR;
const mode = process.env.QA_MODE;
const topic = process.env.QA_TOPIC;
const label = process.env.QA_RUN_LABEL ?? "initial";
if (!feRoot || !campaign || !["READ_ALOUD", "GUIDED", "FREE"].includes(mode) || !topic
    || !/^[a-z0-9][a-z0-9-]{0,40}$/.test(label)) throw new Error("Exact QA input required");
const owned = ownedSpeakingBrowser(campaign);
const output = path.join(owned.output, `speaking-${mode.toLowerCase()}-start-browser-${label}.json`);
if (fs.existsSync(output)) throw new Error("Existing start evidence; never create again for same label");
const require = createRequire(path.join(feRoot, "package.json"));
const { chromium } = require("@playwright/test");
const report = { mode, topic, status: "RUNNING", events: [] };
const save = (event, details = {}) => {
    report.events.push({ at: new Date().toISOString(), event, ...details });
    fs.writeFileSync(output, JSON.stringify(report, null, 2));
};
save("RUN_STARTED");
const context = await chromium.launchPersistentContext(owned.profile, {
    headless: false, executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    args: ["--no-first-run", "--no-default-browser-check"], viewport: { width: 1280, height: 900 },
});
try {
    const origin = "http://localhost:3000";
    const page = context.pages()[0] ?? await context.newPage();
    const auth = await page.request.get(`${origin}/api/auth/session`);
    const session = auth.ok() ? await auth.json() : {};
    requireNormalOwner(session, owned.expectedPublicId);
    await page.goto(`${origin}/ja/language-learning/speaking`, { waitUntil: "domcontentloaded" });
    await page.getByTestId("speaking-start-page").waitFor({ state: "visible", timeout: 30_000 });
    const card = page.getByTestId(`speaking-mode-${mode}`);
    const modeButton = card.getByRole("button");
    await modeButton.waitFor({ state: "visible", timeout: 30_000 });
    await modeButton.click();
    await page.getByTestId("speaking-topic-mode-custom").click();
    await page.getByTestId("speaking-custom-topic-input").fill(topic);
    const create = page.getByRole("button", { name: /Speakingを開始|Start Speaking/i });
    if (!await create.isEnabled()) throw new Error("QA mode topic failed the current UI validation");
    await create.click();
    save("SESSION_CREATE_UI_SUBMITTED");
    await page.waitForURL(/\/speaking\/\d+$/, { timeout: 120_000 });
    const match = page.url().match(/\/speaking\/(\d+)$/);
    if (!match) throw new Error("SESSION_ID_NOT_VISIBLE");
    report.status = "COMPLETED_BROWSER_FLOW";
    save("SESSION_CREATED", { sessionId: Number(match[1]), url: page.url() });
} catch (error) {
    report.status = "FAILED_PARTIAL";
    save("RUN_FAILED", { errorType: error?.name ?? "Error", errorMessage: String(error?.message ?? error).slice(0, 240) });
    throw error;
} finally {
    await context.close();
}
