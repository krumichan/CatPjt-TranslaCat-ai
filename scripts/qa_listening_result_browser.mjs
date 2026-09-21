/** Read-only authenticated QA Listening result re-entry after WAV policy changes. */
import { createRequire } from "node:module";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const feRoot = process.env.QA_FE_ROOT;
const campaign = process.env.QA_CAMPAIGN_DIR;
const sessionId = Number(process.env.QA_LISTENING_SESSION_ID);
const label = process.env.QA_RUN_LABEL;
if (!feRoot || !campaign || !Number.isSafeInteger(sessionId) || sessionId <= 0
    || !label || !/^[a-z0-9][a-z0-9-]{0,40}$/.test(label)) {
    throw new Error("Exact owned QA FE, campaign, session, and label required");
}
const manifest = JSON.parse(fs.readFileSync(path.join(campaign, "integration-environment", "manifest.json"), "utf8"));
if (manifest.owner !== "translacat-isolated-qa"
    || manifest.campaign !== "openai-speech-campaign-20260920"
    || manifest.ports.be !== 18083) throw new Error("Owned QA environment changed");
const output = path.join(campaign, `listening-session${sessionId}-browser-result-${label}.json`);
if (fs.existsSync(output)) throw new Error("Browser result evidence already exists");
const require = createRequire(path.join(feRoot, "package.json"));
const { chromium } = require("@playwright/test");
const report = { sessionId, status: "RUNNING", browserReadOnly: true, providerCallsStarted: 0 };
const save = () => fs.writeFileSync(output, JSON.stringify(report, null, 2));
save();
const context = await chromium.launchPersistentContext(path.join(campaign, "chrome-virtual-mic-profile"), {
    headless: false, executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    args: ["--no-first-run", "--no-default-browser-check"], viewport: { width: 1280, height: 900 },
});
try {
    const origin = "http://localhost:3000";
    const page = context.pages()[0] ?? await context.newPage();
    const response = await page.request.get(`${origin}/api/auth/session`);
    const session = response.ok() ? await response.json() : {};
    if ((session.publicId ?? session.user?.publicId) !== "TC-GA5T-4LRB"
        || !(session.accessToken ?? session.user?.accessToken)) throw new Error("Normal QA Google session unavailable");
    await page.goto(`${origin}/ja/language-learning/listening/session/${sessionId}/result`, {
        waitUntil: "domcontentloaded",
    });
    await page.getByTestId("listening-result-page").waitFor({ state: "visible", timeout: 30_000 });
    const attempts = await page.locator('[data-testid^="listening-result-attempt-"]').count();
    const evaluatedCount = page.getByTestId("listening-result-evaluated-count");
    // The count panel is intentionally hidden when no evaluation is pending or failed.
    const evaluated = await evaluatedCount.count() ? await evaluatedCount.innerText() : null;
    const hasPartialNotice = await page.getByTestId("listening-result-partial-notice").count() > 0;
    const hasEvaluationStatus = await page.getByTestId("listening-result-evaluation-status").count() > 0;
    const body = await page.locator("body").innerText();
    report.status = "RESULT_VISIBLE";
    report.url = page.url();
    report.attemptsVisible = attempts;
    report.evaluatedCountText = evaluated;
    report.hasPartialNotice = hasPartialNotice;
    report.hasEvaluationStatus = hasEvaluationStatus;
    report.bodySha256 = crypto.createHash("sha256").update(body).digest("hex");
    if (attempts !== 5 || hasPartialNotice || hasEvaluationStatus
        || (evaluated !== null && Number(evaluated.trim()) !== 5)) {
        report.status = "RESULT_COUNT_MISMATCH";
        save();
        throw new Error("Saved result UI does not show five evaluated attempts");
    }
    save();
    console.log(JSON.stringify({ sessionId, status: report.status, attempts, evaluated }));
} catch (error) {
    report.status = "FAILED_PARTIAL";
    report.errorType = error?.name ?? "Error";
    report.errorMessage = String(error?.message ?? error).slice(0, 240);
    save();
    throw error;
} finally {
    await context.close();
}
