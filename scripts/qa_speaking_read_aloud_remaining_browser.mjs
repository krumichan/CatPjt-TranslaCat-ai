/** Finish the remaining owned QA READ_ALOUD problems using real MediaRecorder. */
import { createRequire } from "node:module";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const feRoot = process.env.QA_FE_ROOT;
const campaign = process.env.QA_CAMPAIGN_DIR;
const wav = process.env.QA_SYNTHETIC_MIC_WAV;
const sessionId = Number(process.env.QA_SESSION_ID ?? "5");
const firstProblem = Number(process.env.QA_START_PROBLEM ?? "2");
if (!feRoot || !campaign || !wav || !Number.isSafeInteger(sessionId) || sessionId <= 0
    || !Number.isSafeInteger(firstProblem) || firstProblem < 1 || firstProblem > 5) {
    throw new Error("Exact QA paths, session ID, and first problem required");
}
const manifest = JSON.parse(fs.readFileSync(path.join(campaign, "integration-environment", "manifest.json"), "utf8"));
if (manifest.owner !== "translacat-isolated-qa" || manifest.campaign !== "openai-speech-campaign-20260920"
    || manifest.ports.be !== 18083 || !fs.existsSync(wav)) throw new Error("Owned QA environment changed");
const runLabel = process.env.QA_RUN_LABEL ?? "initial";
if (!/^[a-z0-9][a-z0-9-]{0,40}$/.test(runLabel)) throw new Error("Safe run label required");
const resumeFromTurn25 = sessionId === 5 && ["resume-after-ready-turn25", "resume-after-rendered-linebreak"].includes(runLabel);
const output = path.join(campaign, sessionId === 5 && runLabel === "initial"
    ? "speaking-read-aloud-session5-remaining-browser.json"
    : `speaking-read-aloud-session${sessionId}-browser-${runLabel}.json`);
if (fs.existsSync(output)) throw new Error("Existing QA evidence: never repeat the same run");
const require = createRequire(path.join(feRoot, "package.json"));
const { chromium } = require("@playwright/test");
const report = { sessionId, firstProblem, syntheticAudio: true, browserRecorder: true,
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
    const permission = page.getByRole("button", { name: /マイク権限を許可|Allow microphone/i });
    if (await permission.isVisible().catch(() => false)) await permission.click();
    if (firstProblem === 1) {
        await page.getByTestId("read-aloud-problem-panel").waitFor({ state: "visible", timeout: 30_000 });
        if (await page.locator('[data-testid^="speaking-turn-"]').count() !== 0) {
            throw new Error("New-session recorder QA must not replay saved turns");
        }
    }
    if (resumeFromTurn25) {
        await page.waitForFunction(() => /問題\s*2\s*·\s*発話\s*1\s*完了/.test(document.body.innerText),
            null, { timeout: 20_000 });
        save("PERSISTED_PREFIX_REUSED", { problem: 2, attempt: 1, originalTurnId: 25 });
    }
    for (let problem = firstProblem; problem <= 5; problem++) {
        const panel = page.locator('[data-testid="read-aloud-problem-panel"]');
        await panel.getByText(new RegExp(`問題 ${problem} / 5|問題 ${problem}/5`)).first()
            .waitFor({ state: "visible", timeout: 90_000 });
        save("PROBLEM_STARTED", { problem });
        for (let attempt = problem === 2 && resumeFromTurn25 ? 2 : 1;
            attempt <= 2; attempt++) {
            const start = page.getByRole("button", { name: /録音開始|Start recording/i });
            await start.waitFor({ state: "visible", timeout: 90_000 });
            await start.click();
            save("MEDIARECORDER_STARTED", { problem, attempt });
            await page.waitForTimeout(4200);
            await page.getByRole("button", { name: /録音停止|Stop recording/i }).click();
            const preview = page.locator("#speaking-recorder audio");
            await preview.waitFor({ state: "visible", timeout: 15_000 });
            const media = await preview.evaluate(async (element) => {
                const response = await fetch(element.src);
                const blob = await response.blob();
                return { bytes: blob.size, mimeType: blob.type };
            });
            if (media.bytes < 1000) throw new Error("EMPTY_MEDIARECORDER_BLOB");
            save("MEDIARECORDER_STOPPED", { problem, attempt, ...media });
            await page.getByRole("button", { name: /録音を送信|送信|Submit recording/i }).click();
            save("UI_SUBMIT_CLICKED", { problem, attempt });
            const readyPattern = `問題\\s*${problem}\\s*·\\s*発話\\s*${attempt}\\s*完了`;
            try {
                await page.waitForFunction((pattern) => new RegExp(pattern).test(document.body.innerText),
                    readyPattern, { timeout: 40_000 });
            } catch {
                const body = await page.locator("body").innerText();
                save("UI_TURN_READY_NOT_OBSERVED", { problem, attempt,
                    bodySha256: crypto.createHash("sha256").update(body).digest("hex"),
                    includesReadyText: new RegExp(readyPattern).test(body),
                    includesProcessing: /処理中|採点中|送信中/.test(body),
                    alertCount: await page.getByRole("alert").count() });
                await page.reload({ waitUntil: "domcontentloaded" });
                await page.waitForFunction((pattern) => new RegExp(pattern).test(document.body.innerText),
                    readyPattern, { timeout: 20_000 });
                save("UI_READY_AFTER_REENTRY", { problem, attempt });
            }
            save("TURN_VISIBLE_READY", { problem, attempt });
        }
        const evaluate = panel.getByRole("button", { name: /評価|次へ|Evaluate/i });
        if (!await evaluate.isEnabled()) throw new Error(`EVALUATION_NOT_READY_${problem}`);
        await evaluate.click();
        save("EVALUATION_UI_SUBMITTED", { problem });
        if (problem < 5) {
            await panel.getByText(new RegExp(`問題 ${problem + 1} / 5|問題 ${problem + 1}/5`)).first()
                .waitFor({ state: "visible", timeout: 90_000 });
        } else {
            await page.waitForURL(new RegExp(`/speaking/${sessionId}/evaluation`), { timeout: 90_000 });
        }
        save("EVALUATION_UI_ADVANCED", { problem, url: page.url() });
    }
    report.status = "COMPLETED_BROWSER_FLOW";
    save("RUN_COMPLETED", { url: page.url() });
} catch (error) {
    report.status = "FAILED_PARTIAL";
    save("RUN_FAILED", { errorType: error?.name ?? "Error", errorMessage: String(error?.message ?? error).slice(0, 240) });
    throw error;
} finally {
    await context.close();
}
