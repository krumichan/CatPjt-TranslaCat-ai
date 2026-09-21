from __future__ import annotations

import io
import json
import wave

import pytest

from scripts import qa_campaign_be_api as qa


def client(tmp_path, monkeypatch):
    manifest = {"campaign": "test-qa", "names": {"database": "only_qa"}}
    monkeypatch.setattr(qa.bootstrap, "_owned_manifest", lambda directory: manifest)
    qa.bootstrap._write_json(tmp_path / "secrets.private.json", {
        "QA_APP_EMAIL": "qa-test-qa@example.invalid", "QA_APP_ACCESS_TOKEN": "private-token",
    })
    qa.bootstrap._write_json(tmp_path / "auth-reproduction.json", {"registeredUserId": 1})
    return qa.Client(tmp_path)


def test_route_identity_and_provider_boundary():
    assert qa.route("reading-start", {"mode": "STRUCTURE"}) == (
        "GET", "/api/v1/language-learning/practice/today?domain=READING&mode=STRUCTURE", True,
    )
    with pytest.raises(ValueError):
        qa.route("reading-start", {"mode": "../settings"})
    assert qa.route("vocabulary-start", {}) == (
        "GET", "/api/v1/language-learning/practice/today?domain=VOCABULARY&mode=CONTEXTUAL_CHOICE", True,
    )
    assert qa.route("listening-submit", {"attempt_id": 4})[-1] is True
    for value in (True, 0, -1, "1/../../production"):
        with pytest.raises(ValueError):
            qa.route("vocabulary-status", {"set_id": value})


def test_browser_identity_is_existing_normal_google_user_and_cannot_seed_legacy_fixture(tmp_path, monkeypatch):
    import time
    client(tmp_path, monkeypatch)
    session = tmp_path / "normal-session.private.json"
    qa.bootstrap._write_json(session, {"source":"normal-nextauth-google-session", "publicId":"test-public-id",
                                      "accessToken":"offline-secret", "accessTokenExpires":time.time()*1000+60000})
    queries = []
    monkeypatch.setattr(qa.bootstrap, "_qa_mysql_query", lambda _m, _p, q: queries.append(q) or "2")
    value = qa.Client(tmp_path, browser_session=session, output_directory=tmp_path/"browser-run")
    assert value.user_id == 2 and value._access_token() == "offline-secret"
    assert "social_type='GOOGLE'" in queries[0] and "SELECT id" in queries[0]
    with pytest.raises(ValueError, match="Legacy synthetic"):
        value.seed_profile()
    saved = json.loads(session.read_text(encoding="utf-8"))
    saved["publicId"] = "changed-identity"
    qa.bootstrap._write_json(session, saved)
    with pytest.raises(ValueError, match="identity changed"):
        value._access_token()


def test_provider_call_refused_before_network_or_ledger(tmp_path, monkeypatch):
    value = client(tmp_path, monkeypatch)
    monkeypatch.setattr(qa.urllib.request, "build_opener", lambda *args: pytest.fail("network forbidden"))
    with pytest.raises(ValueError, match="allow-provider"):
        value.call("vocabulary-start")
    assert not value.ledger_path.exists()


def test_expired_browser_auth_does_not_consume_http_mutation_or_persist_credentials(tmp_path, monkeypatch):
    value = client(tmp_path, monkeypatch)
    session = tmp_path / "normal-session.private.json"
    value.browser_session = session
    value.browser_public_id = "test-public-id"
    saved_session = {
        "source": "normal-nextauth-google-session", "publicId": value.browser_public_id,
        "accessToken": "expired-offline-secret", "accessTokenExpires": 0,
    }
    qa.bootstrap._write_json(session, saved_session)
    starts = []

    class Opener:
        def open(self, request, timeout):
            starts.append(request)
            # The uncertain-I/O boundary must still be durable before HTTP.
            ledger = json.loads(value.ledger_path.read_text(encoding="utf-8"))
            assert ledger[-1]["status"] == "STARTED"
            assert request.get_header("Authorization") == "Bearer refreshed-offline-secret"
            raise TimeoutError("network outcome is uncertain")

    monkeypatch.setattr(qa.urllib.request, "build_opener", lambda *args: Opener())
    payload = {"learningMode": "SUMMARY", "itemCount": 5}
    with pytest.raises(ValueError, match="must refresh"):
        value.call("listening-create", payload=payload, allow_provider=True)
    assert starts == []
    assert not value.ledger_path.exists()
    assert list(value.output.glob("*-listening-create.json")) == []

    # A real normal-session refresh, not a ledger reset, permits the first HTTP start.
    saved_session.update(accessToken="refreshed-offline-secret",
                         accessTokenExpires=qa.time.time() * 1000 + 60000)
    qa.bootstrap._write_json(session, saved_session)
    with pytest.raises(TimeoutError):
        value.call("listening-create", payload=payload, allow_provider=True)
    with pytest.raises(ValueError, match="One-shot"):
        value.call("listening-create", payload=payload, allow_provider=True)
    assert len(starts) == 1
    ledger_text = value.ledger_path.read_text(encoding="utf-8")
    assert "expired-offline-secret" not in ledger_text
    assert "refreshed-offline-secret" not in ledger_text
    assert json.loads(ledger_text)[0]["status"] == "FAILED"


def test_browser_identity_preflight_failure_does_not_create_http_ledger(tmp_path, monkeypatch):
    value = client(tmp_path, monkeypatch)
    session = tmp_path / "normal-session.private.json"
    value.browser_session = session
    value.browser_public_id = "original-public-id"
    qa.bootstrap._write_json(session, {"publicId": "other-public-id", "accessToken": "private",
                                      "accessTokenExpires": qa.time.time() * 1000 + 60000})
    monkeypatch.setattr(qa.urllib.request, "build_opener", lambda *args: pytest.fail("network forbidden"))
    with pytest.raises(ValueError, match="identity changed"):
        value.call("reading-start", identifiers={"mode": "STRUCTURE"}, allow_provider=True)
    assert not value.ledger_path.exists()
    assert list(value.output.glob("*-reading-start.json")) == []


def test_failed_write_is_flushed_and_never_automatically_repeated(tmp_path, monkeypatch):
    value = client(tmp_path, monkeypatch)
    starts = []

    class Opener:
        def open(self, request, timeout):
            starts.append(request)
            raise TimeoutError("sensitive provider value must not appear")

    monkeypatch.setattr(qa.urllib.request, "build_opener", lambda *args: Opener())
    with pytest.raises(TimeoutError):
        value.call("vocabulary-answer", identifiers={"question_id": 3}, payload={"answer": ["A"]})
    with pytest.raises(ValueError, match="One-shot"):
        value.call("vocabulary-answer", identifiers={"question_id": 3}, payload={"answer": ["B"]})
    assert len(starts) == 1
    saved = value.ledger_path.read_text(encoding="utf-8")
    assert "private-token" not in saved and "sensitive provider" not in saved
    assert json.loads(saved)[0]["error"] == {"type": "TimeoutError"}
    assert len(list(value.output.glob("0001-vocabulary-answer.json"))) == 1


def test_keyboard_interrupt_retains_partial_record(tmp_path, monkeypatch):
    value = client(tmp_path, monkeypatch)

    class Opener:
        def open(self, *args, **kwargs):
            raise KeyboardInterrupt

    monkeypatch.setattr(qa.urllib.request, "build_opener", lambda *args: Opener())
    with pytest.raises(KeyboardInterrupt):
        value.call("profile")
    record = json.loads(value.ledger_path.read_text(encoding="utf-8"))[-1]
    assert record["status"] == "INTERRUPTED"
    assert record["finishedAt"] and record["latencyMs"] >= 0


def test_real_wav_structure_and_mime_not_just_nonempty_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 1600)
    metadata = qa.audio_metadata(output.getvalue(), "audio/wav")
    assert metadata["mimeMatches"] is True
    assert metadata["durationSeconds"] == 0.1
    assert qa.audio_metadata(output.getvalue(), "application/json")["mimeMatches"] is False
    assert qa.audio_metadata(b"not-audio", "audio/wav")["format"] == "UNKNOWN"


def test_audio_upload_never_claims_microphone_capture(tmp_path, monkeypatch):
    value = client(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="explicit WAV"):
        value.call("speaking-turn", identifiers={"session_id": 1}, allow_provider=True, payload={})
    with pytest.raises(ValueError, match="exactly5"):
        value.call("listening-create", allow_provider=True, payload={"itemCount": 6})


def test_profile_fixture_does_not_overwrite_existing_progress(tmp_path, monkeypatch):
    value = client(tmp_path, monkeypatch)
    queries = []

    def database(manifest, private, query):
        queries.append(query)
        return "1" if len(queries) == 1 else "ACTIVE\t84"

    monkeypatch.setattr(qa.bootstrap, "_qa_mysql_query", database)
    with pytest.raises(ValueError, match="noninitial"):
        value.seed_profile()
    assert len(queries) == 2 and all(query.startswith("SELECT") for query in queries)


def test_summary_omits_raw_question_and_private_headers():
    result = qa.summary({"response": {"body": {
        "practiceSetId": 4, "questions": [{"prompt": "raw-content"}], "accessToken": "secret",
    }}})
    assert result["body"] == {"practiceSetId": 4}
    assert "secret" not in str(result) and "raw-content" not in str(result)


def test_poll_is_read_only_bounded_and_stops_at_terminal(tmp_path, monkeypatch):
    value = client(tmp_path, monkeypatch)
    starts = []

    def respond(action, **kwargs):
        starts.append(action)
        return {"httpStatus": 200, "response": {"body": {
            "generationStatus": "PENDING" if len(starts) == 1 else "PARTIAL",
        }}}

    monkeypatch.setattr(value, "call", respond)
    monkeypatch.setattr(qa.time, "sleep", lambda seconds: None)
    result = value.poll("vocabulary-status", {"set_id": 1}, max_polls=10, max_seconds=60)
    assert len(starts) == 2 and result["response"]["body"]["generationStatus"] == "PARTIAL"
    with pytest.raises(ValueError, match="read-only"):
        value.poll("vocabulary-retry-once", {"set_id": 1}, max_polls=10, max_seconds=60)
    assert len(starts) == 2


def test_normal_multiple_speaking_turns_have_source_identity_not_payload_retry(tmp_path, monkeypatch):
    value = client(tmp_path, monkeypatch)

    class Response:
        code = 200
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            return b'{"body":{"turnId":1}}'

    class Opener:
        def open(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(qa.urllib.request, "build_opener", lambda *args: Opener())
    for order in (1, 2):
        value.call("speaking-upload-grant", identifiers={"session_id": 1}, payload={"turnIndex": order})
    with pytest.raises(ValueError, match="One-shot"):
        value.call("speaking-upload-grant", identifiers={"session_id": 1}, payload={"turnIndex": 2, "idempotencyKey": "different"})
    assert len(json.loads(value.ledger_path.read_text(encoding="utf-8"))) == 2


def test_private_persisted_snapshot_uses_owned_user_scope_and_raw_content_not_stdout(tmp_path, monkeypatch):
    value = client(tmp_path, monkeypatch)
    queries = []
    request = json.dumps({"vocabularyPlan": {"slots": []}}).encode().hex()

    def query(manifest, private, sql):
        queries.append(sql)
        if len(queries) == 1:
            return f"1\tPARTIAL\t{request}\tfresh-token\t2026-09-19 12:00:00"
        return "7\t1\t746172676574\t746172676574\t5b2241225d\t0\tsha256"

    monkeypatch.setattr(qa.bootstrap, "_qa_mysql_query", query)
    report = value.practice_snapshot(1)
    assert report["persistedQuestionCount"] == 1 and report["persistedPlanPresent"] is True
    assert "target" not in json.dumps(report)
    assert all("user_id=1" in sql and "SELECT" in sql for sql in queries)
    artifact = json.loads(qa.Path(report["privateArtifact"]).read_text(encoding="utf-8"))
    assert artifact["persistedQuestions"][0]["correctAnswer"] == ["A"]
    assert artifact["generationToken"] == "fresh-token"


def test_listening_partial_progress_is_not_terminal_poll_state(tmp_path, monkeypatch):
    value = client(tmp_path, monkeypatch)
    calls = []

    def respond(*args, **kwargs):
        calls.append(1)
        return {"httpStatus": 200, "response": {"body": {
            "status": "PARTIAL" if len(calls) == 1 else "READY", "generationInProgress": len(calls) == 1,
        }}}

    monkeypatch.setattr(value, "call", respond)
    monkeypatch.setattr(qa.time, "sleep", lambda seconds: None)
    result = value.poll("listening-status", {"set_id": 1}, max_polls=3, max_seconds=60)
    assert len(calls) == 2 and result["response"]["body"]["status"] == "READY"


def test_private_capture_still_redacts_ephemeral_upload_credentials():
    original = {"body": {"uploadToken": "ephemeral", "context": {"password": "secret"},
                          "prompt": "private development QA content"}}
    captured = qa.redact_credentials(original)
    assert captured["body"]["uploadToken"] == "[REDACTED]"
    assert captured["body"]["context"]["password"] == "[REDACTED]"
    assert captured["body"]["prompt"] == original["body"]["prompt"]
    assert original["body"]["uploadToken"] == "ephemeral"


def test_source_daily_set_generation_complete_can_still_have_pending_tts(tmp_path, monkeypatch):
    value = client(tmp_path, monkeypatch)
    calls = []

    def respond(*args, **kwargs):
        calls.append(1)
        return {"httpStatus": 200, "response": {"body": {
            "status": "PARTIAL" if len(calls) == 1 else "READY", "generationInProgress": False,
            "physicalItemCount": 5, "readyItemCount": 4 if len(calls) == 1 else 5,
            "items": [{"status": "TTS_PENDING" if len(calls) == 1 else "READY"}],
        }}}

    monkeypatch.setattr(value, "call", respond)
    monkeypatch.setattr(qa.time, "sleep", lambda seconds: None)
    result = value.poll("listening-status", {"set_id": 2}, max_polls=3, max_seconds=60)
    assert len(calls) == 2 and result["response"]["body"]["readyItemCount"] == 5


def test_persistence_inventory_is_read_only_and_scoped_to_owned_synthetic_user(tmp_path, monkeypatch):
    value = client(tmp_path, monkeypatch)
    queries = []

    def query(manifest, private, sql):
        queries.append(sql)
        return "1\tEVALUATED"

    monkeypatch.setattr(qa.bootstrap, "_qa_mysql_query", query)
    report = value.persistence_inventory()
    assert len(queries) == 11
    assert all(sql.startswith("SELECT ") and "user_id=1" in sql for sql in queries)
    assert all("password" not in sql and "upload_token" not in sql for sql in queries)
    assert report["counts"]["listeningEvaluations"] == 1


def test_speaking_failed_job_snapshot_preserves_exact_private_request_without_replay(tmp_path, monkeypatch):
    value = client(tmp_path, monkeypatch)
    queries = []
    original = {"sessionId": "1", "evaluationScope": "READ_ALOUD_PROBLEM", "userTurns": [{"turnId": "1", "durationSeconds": 3.171}]}

    def query(manifest, private, sql):
        queries.append(sql)
        return "1\tFAILED\tSPEAKING_EVALUATION_FAILED\t" + json.dumps(original).encode().hex()

    monkeypatch.setattr(qa.bootstrap, "_qa_mysql_query", query)
    report = value.speaking_evaluation_snapshot(1, 1)
    assert report["status"] == "FAILED"
    assert "userTurns" not in str(report)
    saved = json.loads(qa.Path(report["privateArtifact"]).read_text())
    assert saved["request"] == original
    assert len(queries) == 1 and "s.user_id=1" in queries[0] and "j.problem_index=1" in queries[0]
    with pytest.raises(ValueError):
        value.speaking_evaluation_snapshot(True, 1)
