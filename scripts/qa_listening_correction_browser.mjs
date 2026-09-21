/** One-shot normal-login UI playback of the owned QA duration-corrected item. */
import { createRequire } from "node:module";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const campaign = process.env.QA_CAMPAIGN_DIR;
const feRoot = process.env.QA_FE_ROOT;
const reentry = process.env.QA_REENTRY === "1";
const trace = process.env.QA_TRACE === "1";
if (!campaign || !feRoot) throw new Error("Exact owned campaign and FE required");
const environment = path.join(campaign, "integration-environment");
const closure = path.join(campaign, "closure-20260920-v1");
const manifest = JSON.parse(fs.readFileSync(path.join(environment, "manifest.json"), "utf8"));
const clone = JSON.parse(fs.readFileSync(path.join(closure, "listening-be-clone-verify.json"), "utf8"));
const learning = JSON.parse(fs.readFileSync(path.join(closure, "listening-be-clone-learning-count-fixture.json"), "utf8"));
if (manifest.owner !== "translacat-isolated-qa" || manifest.campaign !== "openai-speech-campaign-20260920"
    || manifest.ports.be !== 18083 || clone.newSetId !== 11 || clone.result !== "BE_READY_FIVE_ITEMS"
    || clone.correctedItem?.id !== 62 || clone.correctedItem.measuredAudioMs < 5000
    || clone.correctedItem.measuredAudioMs > 12000
    || learning.status !== "LEARNING_COUNT_NORMALIZED_FOR_NEW_BROWSER_USER") {
    throw new Error("Owned corrected Listening fixture authority changed");
}
const output = path.join(closure, trace ? "listening-be-clone-browser-playback-trace.json" : reentry
    ? "listening-be-clone-browser-playback-reentry.json" : "listening-be-clone-browser-playback.json");
if (fs.existsSync(output)) throw new Error("One-shot browser playback evidence already exists");
const require = createRequire(path.join(feRoot, "package.json"));
const { chromium } = require("@playwright/test");
const report = { status: "STARTED", setId: 11, correctedItemId: 62,
    normalGoogleAuthentication: false, browserPlayback: false, events: [] };
const save = (event, details = {}) => {
    report.events.push({ at: new Date().toISOString(), event, ...details });
    fs.writeFileSync(output, JSON.stringify(report, null, 2));
};
save("BROWSER_STARTED");
const context = await chromium.launchPersistentContext(path.join(campaign, "chrome-virtual-mic-profile"), {
    headless: false, executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    args: ["--no-first-run", "--no-default-browser-check"], viewport: { width: 1280, height: 900 },
});
try {
    const origin = "http://localhost:3000";
    const page = context.pages()[0] ?? await context.newPage();
    const auth = await page.request.get(`${origin}/api/auth/session`);
    const session = auth.ok() ? await auth.json() : {};
    if ((session.publicId ?? session.user?.publicId) !== "TC-GA5T-4LRB"
        || !(session.accessToken ?? session.user?.accessToken)) throw new Error("Normal Google QA auth unavailable");
    report.normalGoogleAuthentication = true;
    const audioResponses = [];
    const audioRequests = [];
    context.on("request", (request) => {
        if (request.url().includes("/language-learning/listening/items/")) {
            audioRequests.push({ urlPath: new URL(request.url()).pathname, method: request.method() });
        }
    });
    context.on("response", (response) => {
        if (response.url().includes("/language-learning/listening/items/")) {
            audioResponses.push({ urlPath: new URL(response.url()).pathname,
                status: response.status(), mime: response.headers()["content-type"] });
        }
    });
    if (reentry || trace) {
        await page.goto(`${origin}/ja/language-learning/listening/session/10`, { waitUntil: "domcontentloaded" });
        await page.getByTestId("listening-session-page").waitFor({ state: "visible", timeout: 30_000 });
        report.sessionUrl = page.url();
        save("EXISTING_SESSION_REENTERED_READ_ONLY", { sessionUrl: page.url() });
    } else {
        await page.goto(`${origin}/ja/language-learning/listening`, { waitUntil: "domcontentloaded" });
    const card = page.getByTestId("listening-mode-DICTATION");
    await card.waitFor({ state: "visible", timeout: 30_000 });
    const action = card.getByRole("button");
    if (!await action.isEnabled()) throw new Error("DICTATION_NORMAL_UI_ACTION_DISABLED");
    await action.click();
    await page.waitForURL(/\/language-learning\/listening\/session\/\d+$/, { timeout: 60_000 });
    await page.getByTestId("listening-session-page").waitFor({ state: "visible", timeout: 30_000 });
    report.sessionUrl = page.url();
    save("NORMAL_UI_SESSION_STARTED", { sessionUrl: page.url() });
    for (let index = 1; index <= 4; index++) {
        const skip = page.getByRole("button", { name: /スキップ|건너뛰기|Skip/i });
        await skip.waitFor({ state: "visible", timeout: 30_000 });
        page.once("dialog", (dialog) => dialog.accept());
        await skip.click();
        await page.waitForFunction((expected) => document.body.innerText.includes(`${expected} / 5`)
            || document.body.innerText.includes(`${expected}/5`), index + 1, { timeout: 30_000 });
        save("QA_PREFIX_SKIPPED_THROUGH_UI", { index });
    }
    }
    const player = page.getByTestId("listening-reference-audio");
    await player.waitFor({ state: "visible", timeout: 30_000 });
    await player.getByRole("button").first().click();
    await page.waitForFunction(() => {
        const element = document.querySelector('[data-testid="listening-reference-audio"] audio');
        return element instanceof HTMLAudioElement && Number.isFinite(element.duration)
            && element.duration >= 5 && element.duration <= 12 && element.currentTime > 0.1;
    }, null, { timeout: 30_000 });
    const playback = await player.locator("audio").evaluate(async (element) => {
        const audio = element;
        const response = await fetch(audio.src);
        const blob = await response.blob();
        const bytes = await blob.arrayBuffer();
        const decoder = new AudioContext();
        try {
            const decoded = await decoder.decodeAudioData(bytes.slice(0));
            return { currentTime: audio.currentTime, durationSeconds: audio.duration,
                mime: blob.type, bytes: bytes.byteLength, decodedSeconds: decoded.duration,
                decodedChannels: decoded.numberOfChannels, decodedSampleRate: decoded.sampleRate };
        } finally {
            await decoder.close();
        }
    });
    await page.waitForFunction(() => {
        const element = document.querySelector('[data-testid="listening-reference-audio"] audio');
        return element instanceof HTMLAudioElement && element.ended;
    }, null, { timeout: 20_000 });
    save("PLAYBACK_OBSERVED_BEFORE_NETWORK_ASSERTION", { playback, audioRequests, audioResponses });
    if (!audioResponses.some((response) => response.urlPath.includes("/items/62/audio")
        && response.status === 200)) {
        if (!trace) throw new Error("Item62 authenticated BE audio response missing");
        report.status = "BROWSER_WAV_DECODED_BE_FETCH_UNOBSERVED";
        report.browserPlayback = true;
        save("AUTHENTICATED_SESSION_AUDIO_DECODED_FETCH_NOT_CORROBORATED",
            { playback, audioRequests, audioResponses });
    } else {
        report.status = "PLAYBACK_COMPLETED_BROWSER";
        report.browserPlayback = true;
        save("CORRECTED_ITEM_PLAYBACK_AND_DECODE_COMPLETED", { playback, audioRequests, audioResponses });
    }
} catch (error) {
    report.status = "FAILED_PARTIAL";
    save("BROWSER_FAILED", { errorType: error?.name ?? "Error", errorMessage: String(error?.message ?? error).slice(0, 240) });
    throw error;
} finally {
    save("BROWSER_FINISHED", { status: report.status });
    await context.close();
}
