/** Owned normal-Google Reading selected-model UI start/observe/solve/re-entry. */
import { createRequire } from "node:module";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const campaign = process.env.QA_CAMPAIGN_DIR;
const feRoot = process.env.QA_FE_ROOT;
const action = process.env.QA_READING_ACTION;
const mode = process.env.QA_READING_MODE ?? "COMPREHENSION";
const cardIndex = { COMPREHENSION: 0, STRUCTURE: 1, CONTEXT_INFERENCE: 2 }[mode];
if (!campaign || !feRoot || !["inspect", "start", "observe", "solve", "resume", "finish", "reenter", "verify"].includes(action)) {
    throw new Error("Owned campaign, FE root, and explicit Reading action required");
}
if (cardIndex === undefined) throw new Error("Unsupported Reading mode");
const manifest = JSON.parse(fs.readFileSync(path.join(campaign, "integration-environment", "manifest.json"), "utf8"));
const preflight = JSON.parse(fs.readFileSync(path.join(campaign, "closure-20260920-v1",
    "reading-selected-be-preflight.json"), "utf8"));
if (manifest.owner !== "translacat-isolated-qa" || manifest.campaign !== "openai-speech-campaign-20260920"
    || manifest.ports.be !== 18083 || preflight.userId !== 3 || preflight.baseLevelScore !== 30
    || preflight.existingReadingSets.some((set) => set.mode === mode && set.learningDate === "2026-09-21")) {
    throw new Error("Exact normal Google QA Reading owner/day/profile changed");
}
const closure = path.join(campaign, "closure-20260920-v1");
const suffix = mode === "COMPREHENSION" ? "" : `-${mode.toLowerCase().replaceAll("_", "-")}`;
const output = path.join(closure, `reading-selected-be-browser-${action}${suffix}.json`);
if (fs.existsSync(output)) throw new Error("One-shot Reading browser action already exists");
const require = createRequire(path.join(feRoot, "package.json"));
const { chromium } = require("@playwright/test");
const report = { action, status: "STARTED", normalGoogleAuthentication: false, events: [] };
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
    const token = session.accessToken ?? session.user?.accessToken;
    if ((session.publicId ?? session.user?.publicId) !== "TC-GA5T-4LRB" || !token) {
        throw new Error("Normal Google QA authentication unavailable");
    }
    report.normalGoogleAuthentication = true;
    if (action === "inspect" || action === "start" || action === "observe") {
        if (action !== "observe") {
            await page.goto(`${origin}/ja/language-learning/reading`, { waitUntil: "domcontentloaded" });
            await page.locator("article").first().waitFor({ state: "visible", timeout: 30_000 });
            const cards = await page.locator("article").evaluateAll((nodes) => nodes.map((node) => ({
                title: node.querySelector("h3")?.textContent?.trim() ?? "",
                action: node.querySelector("button")?.textContent?.trim() ?? "",
            })));
            save("LANDING_VISIBLE", { cards });
        }
        if (action === "inspect") report.status = "INSPECTED_NO_GENERATION";
        else {
            const start = performance.now();
            let setId;
            if (action === "start") {
                await page.locator("article").nth(cardIndex).getByRole("button").click();
                await page.waitForURL(/\/language-learning\/reading\/session\/\d+$/, { timeout: 60_000 });
                setId = Number(page.url().split("/").at(-1));
            } else {
                const prior = JSON.parse(fs.readFileSync(path.join(closure,
                    `reading-selected-be-browser-start${suffix}.json`), "utf8"));
                if (prior.status !== "FAILED_PARTIAL" || !Number.isSafeInteger(prior.setId)) {
                    throw new Error("Read-only observation requires owned failed instrumentation with real set");
                }
                setId = prior.setId;
                await page.goto(`${origin}/ja/language-learning/reading/session/${setId}`,
                    { waitUntil: "domcontentloaded" });
            }
            if (!Number.isSafeInteger(setId) || setId <= 0) throw new Error("New Reading set ID invalid");
            report.setId = setId;
            save(action === "start" ? "NORMAL_UI_GENERATION_STARTED" : "NORMAL_UI_EXISTING_SET_OBSERVED",
                { setId, url: page.url() });
            const endpoint = `http://127.0.0.1:18083/api/v1/language-learning/practice/sets/${setId}`;
            const seen = new Set();
            let sawFirstPassage = false;
            for (let poll = 0; poll < 150; poll++) {
                const response = await page.request.get(endpoint, { headers: { Authorization: `Bearer ${token}` } });
                if (!response.ok()) throw new Error(`Normal bearer status GET failed: ${response.status()}`);
                const envelope = await response.json();
                const view = envelope.body;
                if (view?.practiceSetId !== setId || view.domain !== "READING"
                    || view.mode !== mode || view.questionCount !== 5
                    || !Number.isInteger(view.complexityBand) || view.complexityBand < 1
                    || view.complexityBand > 5) {
                    throw new Error("Public Reading set identity/selected band mismatch");
                }
                const count = view.questions?.length ?? 0;
                if (!seen.has(`${count}:${view.generationStatus}`)) {
                    seen.add(`${count}:${view.generationStatus}`);
                    save("BE_PUBLIC_GENERATION_PROGRESS", { poll, count, generationStatus: view.generationStatus,
                        millisecondsFromUiStart: Math.round(performance.now() - start),
                        questionIds: view.questions?.map((q) => q.questionId),
                        passageIds: view.questions?.map((q) => q.passageId),
                        failureMessage: view.generationFailureMessage ?? null });
                }
                if (count > 0 && !sawFirstPassage) {
                    if (count !== 3) throw new Error("P1_NOT_ATOMIC_THREE_QUESTIONS");
                    sawFirstPassage = true;
                    if (action === "start") report.firstQuestionDelayMs = Math.round(performance.now() - start);
                    save("FIRST_PASSAGE_ATOMICALLY_PUBLISHED", { count,
                        firstQuestionDelayMs: report.firstQuestionDelayMs ?? null,
                        delayedObservation: action === "observe" });
                }
                if (count === 5 && view.generationStatus === "READY") {
                    const p1 = view.questions.slice(0, 3);
                    const p2 = view.questions.slice(3);
                    if (p1.some((q) => q.passageId !== "p1" || q.passageText !== p1[0].passageText)
                        || p2.some((q) => q.passageId !== "p2" || q.passageText !== p2[0].passageText)) {
                        throw new Error("Published 3+2 passage binding mismatch");
                    }
                    report.status = "READY_FIVE_REAL_BE_DB";
                    report.totalGenerationMs = Math.round(performance.now() - start);
                    save("FIVE_QUESTIONS_READY", { totalGenerationMs: report.totalGenerationMs,
                        p1Sha256: crypto.createHash("sha256").update(p1[0].passageText).digest("hex"),
                        p2Sha256: crypto.createHash("sha256").update(p2[0].passageText).digest("hex") });
                    break;
                }
                if (["FAILED", "PARTIAL"].includes(view.generationStatus) && view.generationFailureMessage) {
                    report.status = "GENERATION_TERMINAL_WITH_PRESERVED_PREFIX";
                    break;
                }
                await page.waitForTimeout(4_000);
            }
            if (report.status === "STARTED") report.status = "GENERATION_INCOMPLETE_WITHIN_BOUNDED_POLL";
        }
    } else {
        const readyPath = path.join(closure, `reading-selected-be-browser-observe${suffix}.json`);
        const started = JSON.parse(fs.readFileSync(fs.existsSync(readyPath) ? readyPath
            : path.join(closure, `reading-selected-be-browser-start${suffix}.json`), "utf8"));
        if (started.status !== "READY_FIVE_REAL_BE_DB" || !Number.isSafeInteger(started.setId)) {
            throw new Error("Only a ready, previously started owned Reading set can be solved/re-entered");
        }
        report.setId = started.setId;
        await page.goto(`${origin}/ja/language-learning/reading/session/${started.setId}`, { waitUntil: "domcontentloaded" });
        if (["solve", "resume", "finish"].includes(action)) {
            const firstOrder = action === "finish" ? 3 : action === "resume" ? 2 : 1;
            if (action === "resume" || action === "finish") {
                const prior = JSON.parse(fs.readFileSync(path.join(closure,
                    `reading-selected-be-browser-${action === "finish" ? "resume" : "solve"}${suffix}.json`), "utf8"));
                if (prior.status !== "FAILED_PARTIAL" || prior.setId !== started.setId
                    || prior.events.filter((event) => event.event === "NORMAL_UI_ANSWER_SUBMITTED").length !== 1) {
                    throw new Error("Resume requires the exact one-answer QA browser interruption");
                }
            }
            for (let order = firstOrder; order <= 5; order++) {
                const heading = page.getByRole("heading", { name: new RegExp(`${order}.*5`) });
                try {
                    await heading.waitFor({ state: "visible", timeout: 30_000 });
                } catch (error) {
                    const numericHeadings = (await page.locator("h2").allTextContents())
                        .map((text) => text.replace(/[^0-9/ ]/g, "").trim());
                    save("QUESTION_PROGRESS_NOT_VISIBLE", { order, numericHeadings });
                    throw error;
                }
                const options = page.locator("button").filter({ has: page.locator("span") });
                const optionA = options.filter({ hasText: /^A/ }).first();
                await optionA.click();
                await page.getByRole("button", { name: /回答|送信|提出|Submit|確認/i }).last().click();
                await page.getByText(/正解|不正解|Correct|Incorrect/).first().waitFor({ state: "visible", timeout: 30_000 });
                save("NORMAL_UI_ANSWER_SUBMITTED", { order, choice: "A", qualityGoldClaim: false });
                if (order < 5) await page.getByRole("button", { name: "次の問題", exact: true }).click();
            }
            report.status = "FIVE_UI_ANSWERS_SUBMITTED";
        } else {
            await page.reload({ waitUntil: "domcontentloaded" });
            await page.getByRole("heading", { name: "学習結果", exact: true })
                .waitFor({ state: "visible", timeout: 30_000 });
            const body = await page.locator("body").innerText();
            report.status = "REENTRY_VISIBLE";
            save("RESULT_AFTER_RELOAD", { completedVisible: /結果|学習完了|Result/.test(body),
                pageUrl: page.url(), bodySha256: crypto.createHash("sha256").update(body).digest("hex") });
        }
    }
} catch (error) {
    report.status = "FAILED_PARTIAL";
    save("BROWSER_FAILED", { errorType: error?.name ?? "Error", errorMessage: String(error?.message ?? error).slice(0, 250) });
    throw error;
} finally {
    save("BROWSER_FINISHED", { status: report.status });
    await context.close();
}
