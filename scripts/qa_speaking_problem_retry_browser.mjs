/** One normal-auth QA retry of an already FAILED READ_ALOUD problem. */
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";

const feRoot = process.env.QA_FE_ROOT;
const campaign = process.env.QA_CAMPAIGN_DIR;
if (!feRoot || !campaign) throw new Error("QA_FE_ROOT and QA_CAMPAIGN_DIR required");
const manifest = JSON.parse(fs.readFileSync(path.join(campaign, "integration-environment", "manifest.json"), "utf8"));
if (manifest.owner !== "translacat-isolated-qa" || manifest.campaign !== "openai-speech-campaign-20260920"
    || manifest.ports.be !== 18083) throw new Error("Owned QA BE binding changed");
const require = createRequire(path.join(feRoot, "package.json"));
const { chromium } = require("@playwright/test");
const output = path.join(campaign, "speaking-read-aloud-session5-problem1-retry-diagnostic.json");
if (fs.existsSync(output)) throw new Error("Retry artifact already exists; refusing duplicate call");
const report = { syntheticAudio: true, sessionId: 5, problemIndex: 1,
    normalGoogleAuth: true, providerCallsStartedByScript: 0, startedAt: new Date().toISOString() };
const save = () => fs.writeFileSync(output, JSON.stringify(report, null, 2));
save();
const context = await chromium.launchPersistentContext(path.join(campaign, "chrome-virtual-mic-profile"), {
    headless: false,
    executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    args: ["--no-first-run", "--no-default-browser-check"],
});
try {
    const page = context.pages()[0] ?? await context.newPage();
    const response = await page.request.get("http://localhost:3000/api/auth/session");
    if (!response.ok()) throw new Error("Normal Google session unavailable");
    const session = await response.json();
    const publicId = session.publicId ?? session.user?.publicId;
    const token = session.accessToken ?? session.user?.accessToken;
    if (publicId !== "TC-GA5T-4LRB" || typeof token !== "string") {
        throw new Error("QA Google identity mismatch");
    }
    report.authenticatedPublicId = publicId;
    report.providerCallsStartedByScript = "BE_DISPATCH_BOUND_ONLY";
    report.requestStartedAt = new Date().toISOString();
    save();
    const retry = await page.request.post(
        "http://127.0.0.1:18083/api/v1/language-learning/speaking/sessions/5/read-aloud/problems/1/evaluation/retry",
        { headers: { Authorization: `Bearer ${token}` }, timeout: 30_000 },
    );
    const body = await retry.json().catch(() => ({}));
    report.httpStatus = retry.status();
    report.responseStatus = body.body?.status ?? body.status ?? null;
    report.responseCode = body.code ?? body.errorCode ?? null;
    report.requestFinishedAt = new Date().toISOString();
    save();
} catch (error) {
    report.errorType = error?.name ?? "Error";
    report.errorMessage = String(error?.message ?? error).slice(0, 180);
    report.requestFinishedAt = new Date().toISOString();
    save();
    throw error;
} finally {
    await context.close();
}
