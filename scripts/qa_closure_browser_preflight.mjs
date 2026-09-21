/** Read-only normal-auth and mode-card preflight for the owned QA profile. */
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";

const campaign = process.env.QA_CAMPAIGN_DIR;
const feRoot = process.env.QA_FE_ROOT;
const output = process.env.QA_OUTPUT;
if (!campaign || !feRoot || !output || fs.existsSync(output)) throw new Error("Fresh owned QA paths required");
const manifest = JSON.parse(fs.readFileSync(path.join(campaign, "integration-environment", "manifest.json"), "utf8"));
if (manifest.owner !== "translacat-isolated-qa" || manifest.campaign !== "openai-speech-campaign-20260920"
    || manifest.ports.be !== 18083) throw new Error("Owned QA environment changed");
const require = createRequire(path.join(feRoot, "package.json"));
const { chromium } = require("@playwright/test");
const result = { startedAt: new Date().toISOString(), readOnly: true, providerStarts: 0,
    status: "STARTED", cards: {} };
const save = () => fs.writeFileSync(output, JSON.stringify(result, null, 2));
save();
const context = await chromium.launchPersistentContext(path.join(campaign, "chrome-virtual-mic-profile"), {
    headless: false, executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    args: ["--no-first-run", "--no-default-browser-check"], viewport: { width: 1280, height: 900 },
});
try {
    const page = context.pages()[0] ?? await context.newPage();
    const response = await page.request.get("http://localhost:3000/api/auth/session");
    const session = response.ok() ? await response.json() : {};
    result.authenticatedPublicId = session.publicId ?? session.user?.publicId ?? null;
    result.hasAccessToken = Boolean(session.accessToken ?? session.user?.accessToken);
    if (result.authenticatedPublicId !== "TC-GA5T-4LRB" || !result.hasAccessToken) {
        result.status = "NORMAL_LOGIN_REQUIRED";
    } else {
        await page.goto("http://localhost:3000/ja/language-learning/speaking", { waitUntil: "domcontentloaded" });
        await page.getByTestId("speaking-start-page").waitFor({ state: "visible", timeout: 30000 });
        for (const mode of ["READ_ALOUD", "FREE", "GUIDED"]) {
            const card = page.getByTestId(`speaking-mode-${mode}`);
            result.cards[mode] = {
                visible: await card.isVisible(),
                buttonCount: await card.getByRole("button").count(),
                linkCount: await card.getByRole("link").count(),
            };
        }
        result.status = "READ_ONLY_COMPLETED";
    }
} catch (error) {
    result.status = "FAILED";
    result.errorType = error?.name ?? "Error";
    result.errorMessage = String(error?.message ?? error).slice(0, 180);
} finally {
    save();
    await context.close();
}
console.log(JSON.stringify(result));
