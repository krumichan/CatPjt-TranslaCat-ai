/** Resolve only the original or explicitly owned FREE-v3 QA browser environment. */
import fs from "node:fs";
import path from "node:path";

export function ownedSpeakingBrowser(campaign) {
    const original = JSON.parse(fs.readFileSync(path.join(campaign, "integration-environment", "manifest.json"), "utf8"));
    if (original.owner !== "translacat-isolated-qa"
        || original.campaign !== "openai-speech-campaign-20260920"
        || original.ports.be !== 18083) throw new Error("Original owned QA environment changed");
    const selectors = ["QA_FREE_V3_OWNED", "QA_FREE_V3_REVISION_RETEST", "QA_FREE_COACHING_V1", "QA_FREE_COACHING_V1_RETEST"];
    const selected = selectors.filter(key => process.env[key] === "1");
    if (selected.length === 0) return {
        profile: path.join(campaign, "chrome-virtual-mic-profile"),
        output: path.join(campaign, "closure-20260920-v1"),
        expectedPublicId: "TC-GA5T-4LRB",
    };
    if (selected.length !== 1)
        throw new Error("Choose exactly one QA schema");
    const retest = process.env.QA_FREE_V3_REVISION_RETEST === "1";
    const coaching = process.env.QA_FREE_COACHING_V1 === "1";
    const coachingRetest = process.env.QA_FREE_COACHING_V1_RETEST === "1";
    const output = path.join(campaign, "closure-20260920-v1",
        coachingRetest ? "free-coaching-v1-retest-owned-qa"
            : coaching ? "free-coaching-v1-owned-qa"
            : retest ? "free-v3-retest-owned-qa" : "free-v3-owned-qa");
    const schema = coachingRetest
        ? "translacat_qa_openai_speech_campaign_20260920_coaching_v1_retest"
        : coaching
        ? "translacat_qa_openai_speech_campaign_20260920_coaching_v1"
        : retest
        ? "translacat_qa_openai_speech_campaign_20260920_free_v3_retest"
        : "translacat_qa_openai_speech_campaign_20260920_free_v3";
    const fresh = JSON.parse(fs.readFileSync(path.join(output, "manifest.json"), "utf8"));
    if (fresh.owner !== original.owner || fresh.campaign !== original.campaign
        || fresh.originalDatabase !== original.names.database
        || fresh.database !== schema
        || !fresh.mysqlContainerId
        || fresh.beProcess?.database !== fresh.database
        || fresh.status !== "BE_HEALTHY_NORMAL_LOGIN_REQUIRED") {
        throw new Error("Fresh QA schema ownership or active BE binding changed");
    }
    return { profile: path.join(output, "chrome-normal-login"), output,
        expectedPublicId: null };
}

export function requireNormalOwner(session, expectedPublicId) {
    const publicId = session.publicId ?? session.user?.publicId;
    if (typeof publicId !== "string" || !publicId.trim()
        || !(session.accessToken ?? session.user?.accessToken)
        || (expectedPublicId && publicId !== expectedPublicId)) {
        throw new Error("Normal Google QA session unavailable");
    }
    return publicId;
}
