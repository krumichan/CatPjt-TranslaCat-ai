/** One-shot UI evaluation of a completed owned QA FREE session. */
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { ownedSpeakingBrowser, requireNormalOwner } from "./qa_owned_speaking_browser.mjs";

const campaign = process.env.QA_CAMPAIGN_DIR;
const feRoot = process.env.QA_FE_ROOT;
const sessionId = Number(process.env.QA_SESSION_ID);
if (!campaign || !feRoot || !Number.isSafeInteger(sessionId) || sessionId <= 0) {
    throw new Error("Owned campaign, FE root, and session ID required");
}
const owned = ownedSpeakingBrowser(campaign);
const output = path.join(owned.output, `speaking-free-session${sessionId}-finish.json`);
if (fs.existsSync(output)) throw new Error("One-shot evaluation evidence already exists");
const require = createRequire(path.join(feRoot, "package.json"));
const { chromium } = require("@playwright/test");
const report = { sessionId, status: "STARTED", browserEvaluation: true, events: [] };
const save = (event, details = {}) => {
    report.events.push({ at: new Date().toISOString(), event, ...details });
    fs.writeFileSync(output, JSON.stringify(report, null, 2));
};
save("FINISH_STARTED");
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
    await page.goto(`${origin}/ja/language-learning/speaking/${sessionId}`, { waitUntil: "domcontentloaded" });
    await page.getByTestId("speaking-session-page").waitFor({ state: "visible", timeout: 30_000 });
    await page.waitForFunction(() => document.querySelectorAll('[data-testid^="speaking-turn-"]').length === 5,
        null, { timeout: 30_000 });
    const evaluate = page.getByRole("button", { name: /終了して評価を受ける|終了してコーチングを受ける|Finish and evaluate|Finish and receive coaching/i });
    await evaluate.waitFor({ state: "visible", timeout: 30_000 });
    if (!await evaluate.isEnabled()) throw new Error("FIVE_TURNS_NOT_EVALUATION_ELIGIBLE");
    const coaching = /コーチング|coaching/i.test(await evaluate.innerText());
    const eligibility = (await page.locator("body").innerText()).match(/有効Turn[^\n]+/);
    save(coaching ? "FIVE_TURNS_COACHING_COMPLETION_READY" : "FIVE_TURNS_EVALUATION_ELIGIBLE",
        { summary: eligibility?.[0] ?? null, resultKind: coaching ? "SESSION_COACHING" : "SCORED_EVALUATION" });
    page.once("dialog", (dialog) => dialog.accept());
    await evaluate.click();
    save(coaching ? "COACHING_UI_SUBMITTED" : "EVALUATION_UI_SUBMITTED");
    await page.waitForURL(new RegExp(`/speaking/${sessionId}/evaluation`), { timeout: 120_000 });
    await page.waitForFunction(() => /総合スコア|評価データが不足しています|評価失敗|今回の会話コーチング|コーチングを準備できませんでした/.test(document.body.innerText),
        null, { timeout: 120_000 });
    const resultText = await page.locator("body").innerText();
    const resultKind = /今回の会話コーチング/.test(resultText) ? "COACHING_VISIBLE"
        : /総合スコア/.test(resultText) ? "SCORE_VISIBLE"
        : /評価データが不足しています/.test(resultText) ? "INSUFFICIENT_EVIDENCE_VISIBLE" : "OTHER_NO_SCORE";
    report.status = "COMPLETED_BROWSER_FLOW";
    save("EVALUATION_PAGE_VISIBLE", { url: page.url(), resultKind, visibleText: resultText });
} catch (error) {
    report.status = "FAILED_PARTIAL";
    save("FINISH_FAILED", { errorType: error?.name ?? "Error", errorMessage: String(error?.message ?? error).slice(0, 240) });
    throw error;
} finally {
    save("FINISH_ENDED", { status: report.status });
    await context.close();
}
