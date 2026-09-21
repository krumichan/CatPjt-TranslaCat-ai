/**
 * Owned QA browser with a synthetic WAV as Chromium's microphone input.
 * The page uses its real getUserMedia/MediaRecorder/upload/STT path.
 * No authentication is forged: a normal Google session is required.
 */
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";

const feRoot = process.env.QA_FE_ROOT;
const campaign = process.env.QA_CAMPAIGN_DIR;
const wav = process.env.QA_SYNTHETIC_MIC_WAV;
const chrome = process.env.QA_CHROME_PATH;
const baseURL = process.env.QA_BROWSER_ORIGIN ?? "http://localhost:3000";
const runLabel = process.env.QA_BROWSER_RUN_LABEL ?? "read-aloud-attempt-1";
if (!/^[a-z0-9][a-z0-9-]{0,50}$/.test(runLabel)) throw new Error("Safe QA run label required");
if (!feRoot || !campaign || !wav || !chrome) {
    throw new Error("QA_FE_ROOT, QA_CAMPAIGN_DIR, QA_SYNTHETIC_MIC_WAV and QA_CHROME_PATH are required");
}
if (!fs.existsSync(wav) || !fs.existsSync(chrome)) {
    throw new Error("Synthetic WAV or Chrome executable is missing");
}
const manifest = JSON.parse(fs.readFileSync(path.join(campaign, "integration-environment", "manifest.json"), "utf8"));
if (manifest.owner !== "translacat-isolated-qa" || manifest.campaign !== "openai-speech-campaign-20260920") {
    throw new Error("QA campaign ownership check failed");
}
const require = createRequire(path.join(feRoot, "package.json"));
const { chromium } = require("@playwright/test");
const output = path.join(campaign, `speaking-virtual-mic-browser-${runLabel}.json`);
if (fs.existsSync(output)) throw new Error("QA browser artifact already exists; do not overwrite evidence");
const events = [];
const startedAt = Date.now();
const snapshot = (phase, detail = {}) => {
    events.push({ at: new Date().toISOString(), phase, ...detail });
    fs.writeFileSync(output, JSON.stringify({ syntheticAudio: true, origin: baseURL,
        sourceWavSha256: require("node:crypto").createHash("sha256").update(fs.readFileSync(wav)).digest("hex"),
        elapsedMs: Date.now() - startedAt, events }, null, 2));
};

const context = await chromium.launchPersistentContext(path.join(campaign, "chrome-virtual-mic-profile"), {
    headless: false,
    executablePath: chrome,
    args: [
        "--use-fake-device-for-media-stream",
        "--use-fake-ui-for-media-stream",
        `--use-file-for-fake-audio-capture=${wav}`,
        "--no-first-run",
        "--no-default-browser-check",
    ],
    viewport: { width: 1280, height: 900 },
});
try {
    await context.grantPermissions(["microphone"], { origin: baseURL });
    const page = context.pages()[0] ?? await context.newPage();
    await page.goto(`${baseURL}/ja/language-learning/speaking/5`, { waitUntil: "domcontentloaded" });
    snapshot("BROWSER_OPENED", { url: page.url() });
    // User completes normal Google OAuth only if this isolated profile has no valid session.
    const authDeadline = Date.now() + 10 * 60_000;
    let publicId = null;
    while (Date.now() < authDeadline) {
        const sessionResponse = await page.request.get(`${baseURL}/api/auth/session`).catch(() => null);
        if (sessionResponse?.ok()) {
            const session = await sessionResponse.json().catch(() => ({}));
            publicId = session.publicId ?? session.user?.publicId ?? null;
            if (publicId && (session.accessToken ?? session.user?.accessToken)) break;
        }
        await page.waitForTimeout(2000);
    }
    if (!publicId) throw new Error("NORMAL_GOOGLE_AUTH_NOT_COMPLETED");
    snapshot("AUTHENTICATED", { publicId });
    await page.goto(`${baseURL}/ja/language-learning/speaking/5`, { waitUntil: "domcontentloaded" });
    const permissionButton = page.getByRole("button", { name: /マイク権限を許可|Allow microphone/i });
    if (await permissionButton.isVisible().catch(() => false)) {
        await permissionButton.click();
    }
    const start = page.getByRole("button", { name: /録音開始|Start recording/i });
    await start.waitFor({ state: "visible", timeout: 30_000 });
    await start.click();
    snapshot("MEDIARECORDER_STARTED", { pageUrl: page.url() });
    await page.waitForTimeout(4200);
    await page.getByRole("button", { name: /録音停止|Stop recording/i }).click();
    const preview = page.locator("#speaking-recorder audio");
    await preview.waitFor({ state: "visible", timeout: 15_000 });
    const media = await preview.evaluate(async (element) => {
        const response = await fetch(element.src);
        const blob = await response.blob();
        return { bytes: blob.size, mimeType: blob.type };
    });
    snapshot("MEDIARECORDER_STOPPED", media);
    if (media.bytes < 1000) throw new Error("EMPTY_MEDIARECORDER_BLOB");
    const submit = page.getByRole("button", { name: /録音を送信|送信|Submit recording/i });
    await submit.click();
    snapshot("UI_SUBMIT_CLICKED");
    await page.waitForTimeout(30_000);
    const body = await page.locator("body").innerText();
    snapshot("POST_SUBMIT_UI", { url: page.url(), hasTranscript: /文字起こし|認識|transcript|発話/.test(body),
        alertText: (await page.getByRole("alert").allInnerTexts()).map((value) => value.slice(0, 200)) });
    if (process.env.QA_EVALUATE_READ_ALOUD === "1") {
        const evaluate = page.locator('[data-testid="read-aloud-problem-panel"] button').filter({
            hasText: /評価|次へ|Evaluate/i,
        });
        await evaluate.waitFor({ state: "visible", timeout: 15_000 });
        if (await evaluate.isDisabled()) throw new Error("READ_ALOUD_EVALUATION_NOT_READY");
        await evaluate.click();
        snapshot("UI_EVALUATE_CLICKED");
        await page.waitForTimeout(30_000);
        snapshot("POST_EVALUATION_UI", { url: page.url(),
            alertText: (await page.getByRole("alert").allInnerTexts()).map((value) => value.slice(0, 200)) });
    }
} catch (error) {
    snapshot("FAILED", { errorType: error?.name ?? "Error", errorMessage: String(error?.message ?? error).slice(0, 250) });
    throw error;
} finally {
    await context.close();
}
