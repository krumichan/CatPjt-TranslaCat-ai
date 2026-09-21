/** Read-only normal Google login and fresh QA BE identity preflight. */
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { ownedSpeakingBrowser, requireNormalOwner } from "./qa_owned_speaking_browser.mjs";

const campaign = process.env.QA_CAMPAIGN_DIR;
const feRoot = process.env.QA_FE_ROOT;
const label = process.env.QA_RUN_LABEL ?? "initial";
if (!campaign || !feRoot)
    throw new Error("Explicit owned QA campaign and FE root required");
if (!/^[a-z0-9][a-z0-9-]{0,40}$/.test(label)) throw new Error("Invalid QA label");
const owned = ownedSpeakingBrowser(campaign);
const output = path.join(owned.output, `normal-google-auth-preflight-${label}.json`);
if (fs.existsSync(output)) throw new Error("Preserve existing auth preflight");
const require = createRequire(path.join(feRoot, "package.json"));
const { chromium } = require("@playwright/test");
const report = { status: "STARTED", readOnly: true, providerCalls: 0, events: [] };
const save = (event, details = {}) => {
    report.events.push({ event, at: new Date().toISOString(), ...details });
    fs.writeFileSync(output, JSON.stringify(report, null, 2));
};
save("AUTH_PREFLIGHT_STARTED");
const context = await chromium.launchPersistentContext(owned.profile, {
    headless: false, executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    args: ["--no-first-run", "--no-default-browser-check"],
    viewport: { width: 1280, height: 900 },
});
try {
    const origin = "http://localhost:3000";
    const page = context.pages()[0] ?? await context.newPage();
    const network = [];
    page.on("response", (response) => {
        const url = new URL(response.url());
        if (url.origin === origin && (url.pathname.startsWith("/api/") || url.pathname.startsWith("/_next/")))
            network.push({ path: url.pathname, status: response.status() });
    });
    page.on("requestfailed", (request) => {
        const url = new URL(request.url());
        if (url.origin === origin)
            network.push({ path: url.pathname, failure: request.failure()?.errorText ?? "unknown" });
    });
    const auth = await page.request.get(`${origin}/api/auth/session`);
    const session = auth.ok() ? await auth.json() : {};
    const publicId = requireNormalOwner(session, null);
    save("NORMAL_GOOGLE_SESSION_PRESENT", { publicId });
    await page.goto(`${origin}/ja/language-learning/speaking`, { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(5000);
    save("SPEAKING_ROUTE_LOADED", { url: page.url(), visibleText: (await page.locator("body").innerText()).slice(0, 1600), network });
    await page.getByTestId("speaking-start-page").waitFor({ state: "visible", timeout: 30_000 });
    const visible = await page.locator("body").innerText();
    if (!/自由会話|Free Conversation/i.test(visible)) throw new Error("FREE_MODE_NOT_VISIBLE");
    save("FRESH_QA_SPEAKING_UI_VISIBLE", { publicId, url: page.url() });
    report.status = "NORMAL_AUTH_UI_READY";
} catch (error) {
    report.status = "FAILED_READ_ONLY";
    save("AUTH_PREFLIGHT_FAILED", { errorType: error?.name ?? "Error",
        errorMessage: String(error?.message ?? error).slice(0, 240) });
    throw error;
} finally {
    save("AUTH_PREFLIGHT_FINISHED", { status: report.status });
    await context.close();
}
