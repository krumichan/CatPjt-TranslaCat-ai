/** Read-only view of the normal-auth QA Chrome profile; starts no provider work. */
import { createRequire } from "node:module";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const campaign = process.env.QA_CAMPAIGN_DIR;
const feRoot = process.env.QA_FE_ROOT;
if (!campaign || !feRoot) throw new Error("QA_CAMPAIGN_DIR and QA_FE_ROOT required");
const manifest = JSON.parse(fs.readFileSync(path.join(campaign, "integration-environment", "manifest.json"), "utf8"));
if (manifest.owner !== "translacat-isolated-qa" || manifest.campaign !== "openai-speech-campaign-20260920") {
    throw new Error("Wrong QA environment");
}
const output = path.join(campaign, "speaking-session5-chrome-readonly-snapshot.json");
if (fs.existsSync(output)) throw new Error("Snapshot already exists");
const require = createRequire(path.join(feRoot, "package.json"));
const { chromium } = require("@playwright/test");
const context = await chromium.launchPersistentContext(path.join(campaign, "chrome-virtual-mic-profile"), {
    headless: false, executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    args: ["--no-first-run", "--no-default-browser-check"],
});
try {
    const page = context.pages()[0] ?? await context.newPage();
    const response = await page.request.get("http://localhost:3000/api/auth/session");
    const session = response.ok() ? await response.json() : {};
    await page.goto("http://localhost:3000/ja/language-learning/speaking/5", { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(3000);
    const body = await page.locator("body").innerText();
    const result = { url: page.url(), authenticatedPublicId: session.publicId ?? session.user?.publicId ?? null,
        hasAccessToken: Boolean(session.accessToken ?? session.user?.accessToken),
        bodySha256: crypto.createHash("sha256").update(body).digest("hex"),
        hasReadyTurn25Label: body.includes("問題 2 · 発話 1 完了"),
        hasSessionHeading: body.includes("AI Speaking Session"),
        hasError: /エラー|失敗|error/i.test(body),
        firstCharacters: body.slice(0, 1500),
        providerCallsStarted: 0 };
    fs.writeFileSync(output, JSON.stringify(result, null, 2));
    console.log(JSON.stringify({ url: result.url, authenticatedPublicId: result.authenticatedPublicId,
        hasReadyTurn25Label: result.hasReadyTurn25Label, hasSessionHeading: result.hasSessionHeading,
        hasError: result.hasError }));
} finally {
    await context.close();
}
