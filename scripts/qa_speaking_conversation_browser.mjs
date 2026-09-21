/** Owned QA GUIDED/FREE browser flow using Chrome MediaRecorder and synthetic mic. */
import { createRequire } from "node:module";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const feRoot = process.env.QA_FE_ROOT;
const campaign = process.env.QA_CAMPAIGN_DIR;
const wav = process.env.QA_SYNTHETIC_MIC_WAV;
const sessionId = Number(process.env.QA_SESSION_ID);
const mode = process.env.QA_MODE;
const label = process.env.QA_RUN_LABEL ?? "initial";
if (!feRoot || !campaign || !wav || !Number.isSafeInteger(sessionId) || sessionId <= 0
    || !["GUIDED", "FREE"].includes(mode) || !/^[a-z0-9][a-z0-9-]{0,40}$/.test(label)) {
    throw new Error("Exact owned QA paths, session, mode, and safe run label required");
}
const manifest = JSON.parse(fs.readFileSync(path.join(campaign, "integration-environment", "manifest.json"), "utf8"));
if (manifest.owner !== "translacat-isolated-qa" || manifest.campaign !== "openai-speech-campaign-20260920"
    || manifest.ports.be !== 18083 || !fs.existsSync(wav)) throw new Error("Owned QA environment changed");
const output = path.join(campaign, `speaking-${mode.toLowerCase()}-session${sessionId}-browser-${label}.json`);
if (fs.existsSync(output)) throw new Error("Existing QA evidence: never repeat same run");
const require = createRequire(path.join(feRoot, "package.json"));
const { chromium } = require("@playwright/test");
const report = { sessionId, mode, syntheticAudio: true, browserRecorder: true,
    sourceWavSha256: crypto.createHash("sha256").update(fs.readFileSync(wav)).digest("hex"),
    startedAt: new Date().toISOString(), status: "RUNNING", events: [] };
const save = (event, details = {}) => {
    report.events.push({ at: new Date().toISOString(), event, ...details });
    fs.writeFileSync(output, JSON.stringify(report, null, 2));
};
save("RUN_STARTED");
const context = await chromium.launchPersistentContext(path.join(campaign, "chrome-virtual-mic-profile"), {
    headless: false, executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
        `--use-file-for-fake-audio-capture=${wav}`, "--no-first-run", "--no-default-browser-check"],
    viewport: { width: 1280, height: 900 },
});
try {
    const origin = "http://localhost:3000";
    await context.grantPermissions(["microphone"], { origin });
    const page = context.pages()[0] ?? await context.newPage();
    const auth = await page.request.get(`${origin}/api/auth/session`);
    const session = auth.ok() ? await auth.json() : {};
    if ((session.publicId ?? session.user?.publicId) !== "TC-GA5T-4LRB"
        || !(session.accessToken ?? session.user?.accessToken)) throw new Error("QA Google session unavailable");
    await page.goto(`${origin}/ja/language-learning/speaking/${sessionId}`, { waitUntil: "domcontentloaded" });
    await page.getByTestId("speaking-session-page").waitFor({ state: "visible", timeout: 30_000 });
    const permission = page.getByRole("button", { name: /マイク権限を許可|Allow microphone/i });
    if (await permission.isVisible().catch(() => false)) await permission.click();
    await page.waitForFunction((expectedMode) => {
        const body = document.body.innerText;
        return /0\s*\/\s*20/.test(body)
            && (expectedMode !== "GUIDED" || body.includes("Speakingガイド"));
    }, mode, { timeout: 30_000 });
    const turns = page.locator('[data-testid^="speaking-turn-"]');
    if (await turns.count() !== 0) throw new Error("Session already has turns; never replay prefix");
    for (let turn = 1; turn <= 5; turn++) {
        const start = page.getByRole("button", { name: /録音開始|Start recording/i });
        await start.waitFor({ state: "visible", timeout: 90_000 });
        if (!await start.isEnabled()) throw new Error(`Recorder unavailable before turn ${turn}`);
        await start.click();
        save("MEDIARECORDER_STARTED", { turn });
        await page.waitForTimeout(13_500);
        await page.getByRole("button", { name: /録音停止|Stop recording/i }).click();
        const preview = page.locator("#speaking-recorder audio");
        await preview.waitFor({ state: "visible", timeout: 15_000 });
        const media = await preview.evaluate(async (element) => {
            const response = await fetch(element.src);
            const blob = await response.blob();
            return { bytes: blob.size, mimeType: blob.type };
        });
        if (media.bytes < 1000) throw new Error("EMPTY_MEDIARECORDER_BLOB");
        save("MEDIARECORDER_STOPPED", { turn, ...media });
        await page.getByRole("button", { name: /録音を送信|送信|Submit recording/i }).click();
        save("UI_SUBMIT_CLICKED", { turn });
        await page.locator(`[data-testid="speaking-turn-${turn}"]`).waitFor({ state: "visible", timeout: 120_000 });
        await page.waitForFunction((index) => {
            const card = document.querySelector(`[data-testid="speaking-turn-${index}"]`);
            return card && /完了/.test(card.textContent ?? "") && !/処理中|失敗/.test(card.textContent ?? "");
        }, turn, { timeout: 120_000 });
        save("TURN_READY_VISIBLE", { turn, persistedCardCount: await turns.count() });
        if (turn < 5) {
            await page.waitForFunction(() => [...document.querySelectorAll("button")].some(
                (button) => /録音開始/.test(button.textContent ?? "") && !button.disabled
            ), null, { timeout: 120_000 });
        }
    }
    const evaluate = page.getByRole("button", { name: /終了して評価を受ける|Finish and evaluate/i });
    await evaluate.waitFor({ state: "visible", timeout: 60_000 });
    if (!await evaluate.isEnabled()) throw new Error("FIVE_TURNS_NOT_EVALUATION_ELIGIBLE");
    const eligibility = (await page.locator("body").innerText()).match(/有効Turn[^\n]+/);
    save("EVALUATION_ELIGIBLE", { summary: eligibility?.[0] ?? null });
    page.once("dialog", (dialog) => dialog.accept());
    await evaluate.click();
    save("EVALUATION_UI_SUBMITTED");
    await page.waitForURL(new RegExp(`/speaking/${sessionId}/evaluation`), { timeout: 120_000 });
    save("EVALUATION_PAGE_VISIBLE", { url: page.url() });
    await page.waitForFunction(() => /総合スコア|評価データが不足しています|評価失敗/.test(document.body.innerText),
        null, { timeout: 120_000 });
    const resultText = await page.locator("body").innerText();
    report.status = "COMPLETED_BROWSER_FLOW";
    save("RUN_COMPLETED", { resultKind: /総合スコア/.test(resultText) ? "SCORE_VISIBLE"
        : /評価データが不足しています/.test(resultText) ? "INSUFFICIENT_EVIDENCE_VISIBLE" : "OTHER_NO_SCORE",
        url: page.url() });
} catch (error) {
    report.status = "FAILED_PARTIAL";
    save("RUN_FAILED", { errorType: error?.name ?? "Error", errorMessage: String(error?.message ?? error).slice(0, 240) });
    throw error;
} finally {
    await context.close();
}
