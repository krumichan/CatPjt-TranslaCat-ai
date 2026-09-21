/** Read-only browser re-entry for exact owned QA Speaking sessions. */
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { ownedSpeakingBrowser, requireNormalOwner } from "./qa_owned_speaking_browser.mjs";

const feRoot = process.env.QA_FE_ROOT;
const campaign = process.env.QA_CAMPAIGN_DIR;
const sessions = (process.env.QA_SESSION_IDS ?? "").split(",").map(Number);
const label = process.env.QA_RUN_LABEL ?? "initial";
if (!feRoot || !campaign || sessions.length === 0 || sessions.some((id) => !Number.isSafeInteger(id) || id <= 0)
    || !/^[a-z0-9][a-z0-9-]{0,40}$/.test(label)) throw new Error("Exact QA input required");
const owned = ownedSpeakingBrowser(campaign);
const output = path.join(owned.output, `speaking-reentry-browser-${label}.json`);
if (fs.existsSync(output)) throw new Error("Existing QA evidence; never overwrite");
const require = createRequire(path.join(feRoot, "package.json"));
const { chromium } = require("@playwright/test");
const report = { readOnly: true, startedAt: new Date().toISOString(), status: "RUNNING", sessions: [] };
const save = () => fs.writeFileSync(output, JSON.stringify(report, null, 2));
save();
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
    for (const id of sessions) {
        const item = { sessionId: id };
        report.sessions.push(item);
        save();
        await page.goto(`${origin}/ja/language-learning/speaking/${id}`, { waitUntil: "domcontentloaded" });
        await page.getByTestId("speaking-session-page").waitFor({ state: "visible", timeout: 30_000 });
        item.savedTurnCards = await page.locator('[data-testid^="speaking-turn-"]').count();
        await page.goto(`${origin}/ja/language-learning/speaking/${id}/evaluation`, { waitUntil: "domcontentloaded" });
        await page.waitForFunction(() => /総合スコア|評価データが不足しています|今回の会話コーチング|データを読み込めませんでした/.test(document.body.innerText),
            null, { timeout: 30_000 });
        const firstBody = await page.locator("body").innerText();
        item.firstResultKind = /今回の会話コーチング/.test(firstBody) ? "COACHING"
            : /総合スコア/.test(firstBody) ? "SCORE"
            : /評価データが不足しています/.test(firstBody) ? "INSUFFICIENT" : "LOAD_ERROR";
        if (item.firstResultKind === "COACHING") {
            item.coachingPageCopy = {
                coachingHeaderVisible: /Speaking会話コーチング/.test(firstBody),
                legacySkillDescriptionVisible: /共通5 Skill|Speaking 3 Skill/.test(firstBody),
                unverifiedAsrCautionVisible: /音声認識文に対するフィードバック/.test(firstBody),
            };
            if (!item.coachingPageCopy.coachingHeaderVisible
                || item.coachingPageCopy.legacySkillDescriptionVisible
                || !item.coachingPageCopy.unverifiedAsrCautionVisible) {
                throw new Error("COACHING_PAGE_COPY_CONTRACT_MISMATCH");
            }
        }
        await page.reload({ waitUntil: "domcontentloaded" });
        await page.waitForFunction(() => /総合スコア|評価データが不足しています|今回の会話コーチング|データを読み込めませんでした/.test(document.body.innerText),
            null, { timeout: 30_000 });
        const body = await page.locator("body").innerText();
        item.reloadResultKind = /今回の会話コーチング/.test(body) ? "COACHING"
            : /総合スコア/.test(body) ? "SCORE"
            : /評価データが不足しています/.test(body) ? "INSUFFICIENT" : "LOAD_ERROR";
        item.failedProblemCount = (body.match(/評価失敗/g) ?? []).length;
        item.evaluatedProblemCount = (body.match(/評価完了/g) ?? []).length;
        save();
    }
    report.status = "COMPLETED_READ_ONLY";
    save();
} catch (error) {
    report.status = "FAILED_PARTIAL";
    report.errorType = error?.name ?? "Error";
    report.errorMessage = String(error?.message ?? error).slice(0, 240);
    save();
    throw error;
} finally {
    await context.close();
}
