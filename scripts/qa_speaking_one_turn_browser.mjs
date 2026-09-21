/** One authentic Chrome MediaRecorder turn, with distinct synthesized QA input.
 * Inspect the visible current prompt first; choose a context-aware reply, make
 * a separate budgeted WAV, then relaunch the same authenticated profile once.
 * No direct BE POST substitutes for the browser microphone/upload path.
 */
import { createRequire } from "node:module";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { ownedSpeakingBrowser, requireNormalOwner } from "./qa_owned_speaking_browser.mjs";

const feRoot = process.env.QA_FE_ROOT;
const campaign = process.env.QA_CAMPAIGN_DIR;
const closure = path.join(campaign ?? "", "closure-20260920-v1");
const sessionId = Number(process.env.QA_SESSION_ID);
const turn = Number(process.env.QA_TURN);
const mode = process.env.QA_MODE ?? "FREE";
const wav = process.env.QA_SYNTHETIC_MIC_WAV;
const action = process.env.QA_ACTION;
if (!feRoot || !campaign || !Number.isInteger(sessionId) || sessionId <= 0
    || !Number.isInteger(turn) || turn < 1 || turn > 5
    || !["FREE", "READ_ALOUD"].includes(mode)
    || !["inspect", "record"].includes(action) || (action === "record" && !wav)) {
    throw new Error("Owned QA FE/campaign/session/turn/action and WAV for record required");
}
const owned = ownedSpeakingBrowser(campaign);
if (!fs.existsSync(closure) || (action === "record" && !fs.existsSync(wav)))
    throw new Error("Owned QA inputs changed");
const output = path.join(owned.output, `speaking-${mode.toLowerCase()}-session${sessionId}-turn${turn}-${action}.json`);
if (fs.existsSync(output)) throw new Error("One-shot turn action already has preserved evidence");
const require = createRequire(path.join(feRoot, "package.json"));
const { chromium } = require("@playwright/test");
const report = { sessionId, turn, mode, action, status: "STARTED", syntheticAudio: action === "record",
    browserRecorder: action === "record", events: [] };
const save = (event, details = {}) => {
    report.events.push({ at: new Date().toISOString(), event, ...details });
    fs.writeFileSync(output, JSON.stringify(report, null, 2));
};
save("ACTION_STARTED");
const args = ["--no-first-run", "--no-default-browser-check"];
if (action === "record") args.push("--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
    `--use-file-for-fake-audio-capture=${wav}`);
const context = await chromium.launchPersistentContext(owned.profile, {
    headless: false, executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    args, viewport: { width: 1280, height: 900 },
});
try {
    const origin = "http://localhost:3000";
    if (action === "record") await context.grantPermissions(["microphone"], { origin });
    const page = context.pages()[0] ?? await context.newPage();
    const auth = await page.request.get(`${origin}/api/auth/session`);
    const session = auth.ok() ? await auth.json() : {};
    requireNormalOwner(session, owned.expectedPublicId);
    await page.goto(`${origin}/ja/language-learning/speaking/${sessionId}`, { waitUntil: "domcontentloaded" });
    await page.getByTestId("speaking-session-page").waitFor({ state: "visible", timeout: 30_000 });
    const cards = page.locator('[data-testid^="speaking-turn-"]');
    const expectedPrefix = mode === "READ_ALOUD" ? (turn - 1) * 2 : turn - 1;
    await page.waitForFunction((count) => document.querySelectorAll('[data-testid^="speaking-turn-"]').length === count,
        expectedPrefix, { timeout: 30_000 });
    if (mode === "READ_ALOUD") await page.getByTestId("read-aloud-problem-panel")
        .getByText(new RegExp(`問題 ${turn} / 5|問題 ${turn}/5`)).first()
        .waitFor({ state: "visible", timeout: 30_000 });
    const visible = await page.locator("body").innerText();
    save("PREFIX_AND_CURRENT_PROMPT_VISIBLE", { persistedTurnCount: await cards.count(),
        visibleText: visible, url: page.url() });
    if (action === "inspect") {
        report.status = "INSPECTED_NO_PROVIDER_OR_UPLOAD";
    } else {
        const bytes = fs.readFileSync(wav);
        const hash = crypto.createHash("sha256").update(bytes).digest("hex");
        if (bytes.toString("ascii", 0, 4) !== "RIFF" || bytes.toString("ascii", 8, 12) !== "WAVE") {
            throw new Error("Explicit QA mic input must be a real WAV");
        }
        const sampleRate = bytes.readUInt32LE(24);
        const blockAlign = bytes.readUInt16LE(32);
        if (!sampleRate || !blockAlign || bytes.length <= 44) throw new Error("QA WAV header invalid");
        const captureMs = Math.max(1_300, Math.min(30_000,
            Math.round((bytes.length - 44) / (sampleRate * blockAlign) * 1000) + 250));
        save("SYNTHETIC_INPUT_SELECTED", { wavSha256: hash, bytes: bytes.length,
            captureMs, sourceDurationSeconds: (bytes.length - 44) / (sampleRate * blockAlign) });
        const permission = page.getByRole("button", { name: /マイク権限を許可|Allow microphone/i });
        if (await permission.isVisible().catch(() => false)) await permission.click();
        const attempts = mode === "READ_ALOUD" ? 2 : 1;
        for (let attempt = 1; attempt <= attempts; attempt++) {
            const index = expectedPrefix + attempt;
            const start = page.getByRole("button", { name: /録音開始|Start recording/i });
            await start.waitFor({ state: "visible", timeout: 45_000 });
            await start.click();
            save("MEDIARECORDER_STARTED", { attempt, index });
            await page.waitForTimeout(captureMs);
            await page.getByRole("button", { name: /録音停止|Stop recording/i }).click();
            const preview = page.locator("#speaking-recorder audio");
            await preview.waitFor({ state: "visible", timeout: 15_000 });
            const media = await preview.evaluate(async (element) => {
                const response = await fetch(element.src);
                const blob = await response.blob();
                return { bytes: blob.size, mimeType: blob.type };
            });
            if (media.bytes < 1000) throw new Error("Empty browser MediaRecorder output");
            save("MEDIARECORDER_STOPPED", { attempt, index, ...media });
            await page.getByRole("button", { name: /録音を送信|送信|Submit recording/i }).click();
            save("UI_SUBMIT_CLICKED", { attempt, index });
            await page.locator(`[data-testid="speaking-turn-${index}"]`)
                .waitFor({ state: "visible", timeout: 120_000 });
            await page.waitForFunction((cardIndex) => {
                const card = document.querySelector(`[data-testid="speaking-turn-${cardIndex}"]`);
                return card && /完了/.test(card.textContent ?? "") && !/処理中|失敗/.test(card.textContent ?? "");
            }, index, { timeout: 120_000 });
            save("TURN_READY_AFTER_UPLOAD_STT_AND_ASSISTANT", { attempt, index,
                persistedTurnCount: await cards.count(), visibleText: await page.locator("body").innerText() });
        }
        if (mode === "READ_ALOUD") {
            const evaluate = page.getByTestId("read-aloud-problem-panel")
                .getByRole("button", { name: /評価|次へ|Evaluate/i });
            if (!await evaluate.isEnabled()) throw new Error(`PROBLEM_${turn}_EVALUATION_DISABLED`);
            await evaluate.click();
            if (turn < 5) await page.getByTestId("read-aloud-problem-panel")
                .getByText(new RegExp(`問題 ${turn + 1} / 5|問題 ${turn + 1}/5`)).first()
                .waitFor({ state: "visible", timeout: 90_000 });
            else await page.waitForURL(new RegExp(`/speaking/${sessionId}/evaluation`), { timeout: 90_000 });
            save("PROBLEM_EVALUATION_UI_ADVANCED", { problem: turn, url: page.url(),
                visibleText: await page.locator("body").innerText() });
        }
        report.status = "TURN_READY_BROWSER_MEDIARECORDER";
    }
} catch (error) {
    report.status = "FAILED_PARTIAL";
    save("ACTION_FAILED", { errorType: error?.name ?? "Error", errorMessage: String(error?.message ?? error).slice(0, 240) });
    throw error;
} finally {
    save("ACTION_FINISHED", { status: report.status });
    await context.close();
}
