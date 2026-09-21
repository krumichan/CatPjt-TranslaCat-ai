/** Save an exact normal Google QA session for the existing BE QA HTTP client. */
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";

const feRoot = process.env.QA_FE_ROOT;
const campaign = process.env.QA_CAMPAIGN_DIR;
if (!feRoot || !campaign) throw new Error("Exact QA paths required");
const manifest = JSON.parse(fs.readFileSync(path.join(campaign, "integration-environment", "manifest.json"), "utf8"));
if (manifest.owner !== "translacat-isolated-qa" || manifest.campaign !== "openai-speech-campaign-20260920"
    || manifest.ports.be !== 18083) throw new Error("Owned QA environment changed");
const output = path.join(campaign, "integration-environment", "google-browser-session.private.json");
if (fs.existsSync(output)) throw new Error("Existing session descriptor must not be overwritten");
const require = createRequire(path.join(feRoot, "package.json"));
const { chromium } = require("@playwright/test");
const context = await chromium.launchPersistentContext(path.join(campaign, "chrome-virtual-mic-profile"), {
    headless: false, executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    args: ["--no-first-run", "--no-default-browser-check"], viewport: { width: 1280, height: 900 },
});
try {
    const page = context.pages()[0] ?? await context.newPage();
    const response = await page.request.get("http://localhost:3000/api/auth/session");
    if (!response.ok()) throw new Error("Normal QA Google session unavailable");
    const session = await response.json();
    const token = session.accessToken ?? session.user?.accessToken;
    const publicId = session.publicId ?? session.user?.publicId;
    if (publicId !== "TC-GA5T-4LRB" || typeof token !== "string") {
        throw new Error("Expected exact Google QA user and normal BE access token");
    }
    const jwt = JSON.parse(Buffer.from(token.split(".")[1], "base64url").toString("utf8"));
    const expires = Number(jwt.exp) * 1000;
    if (!Number.isFinite(expires) || expires <= Date.now() + 10 * 60_000) {
        throw new Error("Normal BE token expires too soon for serial QA; refresh login first");
    }
    const descriptor = { source: "normal-nextauth-google-session", publicId,
        accessToken: token, accessTokenExpires: expires, recordedAt: new Date().toISOString() };
    fs.writeFileSync(output, JSON.stringify(descriptor), { flag: "wx", mode: 0o600 });
    console.log(JSON.stringify({ status: "CAPTURED_PRIVATE_NORMAL_SESSION", publicId,
        remainingMinutes: Math.floor((expires - Date.now()) / 60_000), providerCalls: 0 }));
} finally {
    await context.close();
}
