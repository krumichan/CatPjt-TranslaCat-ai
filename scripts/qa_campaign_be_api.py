"""Serial, explicit public-API QA against this campaign's isolated BE only.

No provider-bearing action starts without --allow-provider-work. This is not a
provider budget: the AI QA server must separately enforce the campaign ledger.
Every HTTP start is durably recorded before network I/O; writes never retry.
Raw responses/audio remain in the owned private directory, never stdout.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
import urllib.error
import urllib.request
import wave

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import qa_campaign_integration as bootstrap


BASE = "/api/v1/language-learning"
ROUTES = {
    "settings": ("GET", "/settings", False),
    "configure": ("PATCH", "/settings", False),
    "profile": ("GET", "/profile", False),
    "reading-start": ("GET", "/practice/today?domain=READING&mode={mode}", True),
    "reading-status": ("GET", "/practice/sets/{set_id}", False),
    "reading-answer": ("POST", "/practice/questions/{question_id}/answers", False),
    "vocabulary-start": ("GET", "/practice/today?domain=VOCABULARY&mode=CONTEXTUAL_CHOICE", True),
    "vocabulary-status": ("GET", "/practice/sets/{set_id}", False),
    "vocabulary-answer": ("POST", "/practice/questions/{question_id}/answers", False),
    "vocabulary-retry-once": ("POST", "/practice/sets/{set_id}/retry-generation", True),
    "listening-create": ("POST", "/listening/daily-sets", True),
    "listening-status": ("GET", "/listening/daily-sets/{set_id}", False),
    "listening-session-create": ("POST", "/listening/sessions", False),
    "listening-session": ("GET", "/listening/sessions/{session_id}", False),
    "listening-item": ("GET", "/listening/sessions/{session_id}/items/{item_id}", False),
    "listening-audio": ("GET", "/listening/items/{item_id}/audio", False),
    "listening-answer": ("POST", "/listening/attempts/{attempt_id}/responses/{task_type}", False),
    "listening-submit": ("POST", "/listening/attempts/{attempt_id}/submit", True),
    "listening-complete": ("POST", "/listening/sessions/{session_id}/complete", False),
    "listening-result": ("GET", "/listening/sessions/{session_id}/result", False),
    "speaking-create": ("POST", "/speaking/sessions", True),
    "speaking-session": ("GET", "/speaking/sessions/{session_id}", False),
    "speaking-active": ("GET", "/speaking/sessions/active", False),
    "speaking-upload-grant": ("POST", "/speaking/sessions/{session_id}/turns/upload-url", False),
    "speaking-rerecord-grant": ("POST", "/speaking/sessions/{session_id}/turns/{turn_id}/rerecord/upload-url", False),
    "speaking-turn": ("POST", "/speaking/sessions/{session_id}/turns", True),
    "speaking-turn-status": ("GET", "/speaking/sessions/{session_id}/turns/{turn_id}", False),
    "speaking-turn-retry-once": ("POST", "/speaking/sessions/{session_id}/turns/{turn_id}/retry", True),
    "speaking-complete": ("POST", "/speaking/sessions/{session_id}/complete", True),
    "speaking-evaluation": ("GET", "/speaking/sessions/{session_id}/evaluation", False),
    "speaking-evaluation-retry-once": ("POST", "/speaking/sessions/{session_id}/evaluation/retry", True),
    "speaking-problem-evaluate": ("POST", "/speaking/sessions/{session_id}/read-aloud/problems/{problem_index}/evaluate", True),
    "speaking-problem-retry-once": ("POST", "/speaking/sessions/{session_id}/read-aloud/problems/{problem_index}/evaluation/retry", True),
    "speaking-problem-evaluations": ("GET", "/speaking/sessions/{session_id}/read-aloud/problems", False),
    "speaking-opening-audio": ("GET", "/speaking/sessions/{session_id}/audio/opening", False),
    "speaking-user-audio": ("GET", "/speaking/sessions/{session_id}/turns/{turn_id}/audio/user", False),
    "speaking-assistant-audio": ("GET", "/speaking/sessions/{session_id}/turns/{turn_id}/audio", False),
}
TASK_TYPES = {"DICTATION", "COMPREHENSION", "SUMMARY", "INTERPRETATION", "REPEAT_AFTER_AUDIO"}


def redact_credentials(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "[REDACTED]" if key.lower() in {
            "authorization", "uploadtoken", "uploadurl", "presignedurl", "signedurl",
            "accesstoken", "refreshtoken", "password", "apikey",
        } else redact_credentials(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_credentials(item) for item in value]
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def route(action: str, identifiers: dict[str, Any]) -> tuple[str, str, bool]:
    method, template, provider = ROUTES[action]
    for name in re.findall(r"\{([^}]+)\}", template):
        value = identifiers.get(name)
        if name == "task_type":
            if value not in TASK_TYPES:
                raise ValueError("Unsupported source-defined Listening task")
        elif name == "mode":
            if value not in {"COMPREHENSION", "STRUCTURE", "CONTEXT_INFERENCE"}:
                raise ValueError("Unsupported source-defined Reading mode")
        elif type(value) is not int or value <= 0:
            raise ValueError(f"Positive integer {name} required")
    return method, BASE + template.format(**identifiers), provider


def audio_metadata(body: bytes, content_type: str) -> dict[str, Any]:
    result: dict[str, Any] = {"bytes": len(body), "contentType": content_type,
                              "sha256": hashlib.sha256(body).hexdigest()}
    if body.startswith(b"RIFF") and body[8:12] == b"WAVE":
        with wave.open(io.BytesIO(body), "rb") as source:
            result.update({"format": "WAV", "channels": source.getnchannels(),
                           "sampleRate": source.getframerate(), "frames": source.getnframes(),
                           "durationSeconds": source.getnframes() / source.getframerate()})
        result["mimeMatches"] = content_type.split(";")[0] in {"audio/wav", "audio/x-wav", "audio/wave"}
    elif body.startswith(b"ID3") or body[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}:
        result.update({"format": "MP3", "mimeMatches": content_type.split(";")[0] == "audio/mpeg"})
    elif body.startswith(b"OggS"):
        result.update({"format": "OGG", "mimeMatches": content_type.split(";")[0] in {"audio/ogg", "application/ogg"}})
    else:
        result.update({"format": "UNKNOWN", "mimeMatches": False})
    return result


def multipart(context: dict[str, Any], audio_path: Path) -> tuple[bytes, str]:
    if audio_path.stat().st_size > 15 * 1024 * 1024:
        raise ValueError("QA audio max15MiB")
    audio = audio_path.read_bytes()
    metadata = audio_metadata(audio, "audio/wav")
    if metadata["format"] != "WAV":
        raise ValueError("Explicit QA source audio must be WAV; no codec inference")
    boundary = "qa-" + os.urandom(16).hex()
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="context"\r\nContent-Type: application/json\r\n\r\n'.encode()
        + json.dumps(context, ensure_ascii=False).encode("utf-8")
        + f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="qa.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
        + audio + f"\r\n--{boundary}--\r\n".encode()
    )
    return body, f"multipart/form-data; boundary={boundary}"


@contextmanager
def exclusive(directory: Path):
    lock = directory / "be-api.lock"
    handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        yield
    finally:
        os.close(handle)
        lock.unlink()


class Client:
    def __init__(self, directory: Path, *, browser_session: Path | None = None,
                 output_directory: Path | None = None, campaign_directory: Path | None = None):
        self.directory = directory.resolve()
        self.campaign_directory = campaign_directory.resolve() if campaign_directory else self.directory.parent
        self.manifest = bootstrap._owned_manifest(self.directory)
        self.private = json.loads((self.directory / "secrets.private.json").read_text(encoding="utf-8"))
        self.browser_session = browser_session
        self.browser_public_id = None
        if browser_session is None:
            auth = json.loads((self.directory / "auth-reproduction.json").read_text(encoding="utf-8"))
            self.user_id = auth["registeredUserId"]
            if (type(self.user_id) is not int or self.user_id <= 0
                    or self.private.get("QA_APP_EMAIL") != f"qa-{self.manifest['campaign']}@example.invalid"
                    or not self.private.get("QA_APP_ACCESS_TOKEN")):
                raise ValueError("Exact synthetic registered QA identity and normal login token required")
        else:
            session = json.loads(browser_session.read_text(encoding="utf-8"))
            public_id = session.get("publicId")
            if (session.get("source") != "normal-nextauth-google-session"
                    or not isinstance(public_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,20}", public_id)
                    or not session.get("accessToken")):
                raise ValueError("Normal browser login session required; never mint QA tokens")
            self.browser_public_id = public_id
            identity = bootstrap._qa_mysql_query(self.manifest, self.private,
                "SELECT id FROM user WHERE social_type='GOOGLE' AND public_id=CONVERT(0x"
                + public_id.encode().hex() + " USING utf8mb4);")
            if not identity.isdigit() or int(identity) <= 0:
                raise ValueError("Browser user does not exist in the verified isolated QA DB")
            self.user_id = int(identity)
        self.output = output_directory.resolve() if output_directory else self.directory / "be-api"
        self.output.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.output / "ledger.json"
        identity_path = self.output / "runtime-identity.json"
        if not identity_path.exists():
            process = self.manifest.get("beProcess", {})
            command = process.get("command", [])
            source_files = (
                "domain/languagelearning/practice/controller/ReadingVocabularyPracticeController.java",
                "domain/languagelearning/practice/service/PracticePersistenceService.java",
                "domain/languagelearning/listening/controller/ListeningDailySetController.java",
                "domain/languagelearning/listening/daily/service/ListeningGenerationTransactionService.java",
                "domain/languagelearning/speaking/turn/controller/SpeakingTurnController.java",
            )
            root = Path(process.get("cwd", ".")) / "src/main/java/jp/co/translacat"
            hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                      for name in source_files if (root / name).is_file()}
            jar_hash = None
            if "-jar" in command:
                jar = Path(command[command.index("-jar") + 1])
                with jar.open("rb") as stream:
                    jar_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            bootstrap._write_json(identity_path, {
                "recordedAt": utc_now(), "pid": process.get("pid"), "currentJarSha256": jar_hash,
                "currentSourceSha256": hashes,
                "limitation": "Hash observed now, not proof that a subsequently overwritten jar matches running process",
                "instanceBoundary": self.manifest.get("hostNetworkBoundary"),
            })

    def _access_token(self) -> str:
        if self.browser_session is None:
            return self.private["QA_APP_ACCESS_TOKEN"]
        session = json.loads(self.browser_session.read_text(encoding="utf-8"))
        if session.get("publicId") != self.browser_public_id:
            raise ValueError("Browser identity changed; refusing cross-user QA")
        if session.get("accessTokenExpires", 0) <= time.time() * 1000 + 5000:
            raise ValueError("Normal browser session must refresh; no auth bypass")
        return session["accessToken"]

    def call(self, action: str, *, identifiers: dict[str, Any] | None = None,
             payload: dict[str, Any] | None = None, allow_provider: bool = False,
             audio_path: Path | None = None, timeout: float = 40) -> dict[str, Any]:
        method, path, provider = route(action, identifiers or {})
        if provider and not allow_provider:
            raise ValueError("Provider-bearing action requires explicit --allow-provider-work and shared AI budget server")
        if not 1 <= timeout <= 600:
            raise ValueError("Explicit HTTP timeout must be1..600seconds")
        if method == "GET" and payload is not None:
            raise ValueError("GET payload forbidden")
        if action == "listening-create" and (not payload or payload.get("itemCount") != 5):
            raise ValueError("This campaign requires exactly5 Listening items")
        if action == "speaking-turn" and audio_path is None:
            raise ValueError("Speaking turn requires explicit WAV; do not fake microphone capture")
        content_type = "application/json"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        if audio_path is not None:
            if action != "speaking-turn" or payload is None:
                raise ValueError("Audio only supported for source-defined Speaking turn context")
            data, content_type = multipart(payload, audio_path)
        # Write identity intentionally excludes payload: never re-answer or retry
        # the same entity by changing its body after an uncertain request outcome.
        identity = method + " " + path
        if action == "listening-create":
            identity += " " + str((payload or {}).get("learningMode"))
        identity_keys = {"listening-session-create": "dailySetId", "speaking-create": "practiceMode",
                         "speaking-upload-grant": "turnIndex", "speaking-turn": "turnId"}
        if action in identity_keys:
            discriminator = (payload or {}).get(identity_keys[action])
            if discriminator is None:
                raise ValueError("Source-defined mutation identity missing")
            identity += " " + str(discriminator)
        if action == "speaking-turn" and (payload or {}).get("rerecord") is True:
            identity += " RERECORD_ONCE"
        with exclusive(self.output):
            ledger = json.loads(self.ledger_path.read_text(encoding="utf-8")) if self.ledger_path.exists() else []
            mutation = method != "GET" or provider
            if mutation and any(entry["identity"] == identity for entry in ledger):
                raise ValueError("One-shot mutation already started; inspect outcome instead of automatic retry")
            # Authentication is local preflight, not an HTTP attempt. An expired
            # normal session must not consume a one-shot mutation before I/O.
            # Keep the token in memory only; uncertain HTTP starts stay durable.
            access_token = self._access_token()
            record: dict[str, Any] = {
                "ordinal": len(ledger) + 1, "action": action, "identity": identity,
                "startedAt": utc_now(), "status": "STARTED", "providerBearing": provider,
                "request": {"method": method, "path": path, "body": payload},
                "timeoutSeconds": timeout,
            }
            ledger.append(record)
            bootstrap._write_json(self.ledger_path, redact_credentials(ledger))
            begin = time.monotonic()
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{bootstrap.PORTS['be']}{path}", method=method, data=data,
                    headers={"Authorization": "Bearer " + access_token, "Content-Type": content_type},
                )
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), bootstrap._NoRedirect())
                try:
                    response = opener.open(request, timeout=timeout)
                except urllib.error.HTTPError as error:
                    response = error
                with response:
                    raw = response.read(20 * 1024 * 1024 + 1)
                    if len(raw) > 20 * 1024 * 1024:
                        raise ValueError("QA response max20MiB exceeded")
                    record["httpStatus"] = response.code
                    mime = response.headers.get("Content-Type", "")
                if action.endswith("-audio") and record["httpStatus"] == 200:
                    audio_file = self.output / f"{record['ordinal']:04d}-audio.bin"
                    audio_file.write_bytes(raw)
                    record["audio"] = audio_metadata(raw, mime)
                    record["audio"]["path"] = str(audio_file)
                else:
                    try:
                        record["response"] = json.loads(raw)
                    except (UnicodeError, ValueError):
                        record["response"] = {"nonJsonBytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                record["status"] = "COMPLETED" if record["httpStatus"] < 400 else "HTTP_FAILED"
            except BaseException as error:
                record["status"] = "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED"
                record["error"] = {"type": type(error).__name__}
                raise
            finally:
                record["latencyMs"] = round((time.monotonic() - begin) * 1000)
                record["finishedAt"] = utc_now()
                bootstrap._write_json(self.ledger_path, redact_credentials(ledger))
                bootstrap._write_json(self.output / f"{record['ordinal']:04d}-{action}.json", redact_credentials(record))
        return record

    def poll(self, action: str, identifiers: dict[str, Any], *, max_polls: int,
             max_seconds: float, interval_seconds: float = 10) -> dict[str, Any]:
        if action not in {"vocabulary-status", "reading-status", "listening-status"}:
            raise ValueError("Only read-only generation status can be polled")
        if not 1 <= max_polls <= 60 or not 1 <= max_seconds <= 600 or not 5 <= interval_seconds <= 30:
            raise ValueError("Poll bounds:1..60 polls,1..600seconds,5..30second interval")
        began = time.monotonic()
        last: dict[str, Any] = {}
        for index in range(max_polls):
            remaining = max_seconds - (time.monotonic() - began)
            if remaining < 1:
                break
            last = self.call(action, identifiers=identifiers, timeout=min(40, remaining))
            body = (last.get("response") or {}).get("body") or {}
            state = body.get("generationStatus") if action in {"vocabulary-status", "reading-status"} else body.get("status")
            terminal = state in {"READY", "PARTIAL", "FAILED", "COMPLETED"}
            if action == "listening-status" and (body.get("generationInProgress") is True
                    or any(item.get("status") == "TTS_PENDING" for item in body.get("items", []))):
                terminal = False
            if last.get("httpStatus") != 200 or terminal:
                break
            remaining = max_seconds - (time.monotonic() - began)
            if index + 1 >= max_polls or remaining < interval_seconds:
                break
            time.sleep(interval_seconds)
        return last

    def seed_profile(self) -> dict[str, Any]:
        """Explicit synthetic precondition, NOT evidence of an executed Level Test."""
        if self.browser_session is not None:
            raise ValueError("Legacy synthetic fixture must not target the browser account")
        artifact = self.output / "profile-fixture.json"
        if artifact.exists():
            raise ValueError("Synthetic profile fixture is one-shot")
        email_hex = self.private["QA_APP_EMAIL"].encode().hex()
        identity = f"id={self.user_id} AND email=CONVERT(0x{email_hex} USING utf8mb4) AND social_type='LOCAL'"
        def query(sql: str) -> str:
            return bootstrap._qa_mysql_query(self.manifest, self.private, sql)
        if query(f"SELECT COUNT(*) FROM user WHERE {identity};") != "1":
            raise ValueError("Owned synthetic user fingerprint mismatch")
        before = query(f"SELECT state,base_level_score FROM language_learning_profile WHERE user_id={self.user_id};")
        if before and before != "LEVEL_TEST_REQUIRED\tNULL":
            raise ValueError("Existing noninitial profile must not be overwritten")
        report = {"userId": self.user_id, "before": before, "synthetic": True,
                  "actualLevelTestRun": False, "baseLevelScore": 75, "status": "STARTED"}
        bootstrap._write_json(artifact, report)
        if before:
            sql = ("UPDATE language_learning_profile SET state='CALIBRATING',base_level_score=75,"
                   "calibration_started_date=NULL,calibration_completed_date=NULL "
                   f"WHERE user_id={self.user_id} AND state='LEVEL_TEST_REQUIRED' AND base_level_score IS NULL;")
        else:
            sql = ("INSERT INTO language_learning_profile "
                   "(user_id,profile_version,state,base_level_score,evaluation_count,confidence,trend,"
                   "additional_signals_json,created_at,updated_at,created_by,updated_by) "
                   f"SELECT id,'PROFILE','CALIBRATING',75,0,0,'stable','{{}}',NOW(),NOW(),'QA_FIXTURE','QA_FIXTURE' FROM user WHERE {identity};")
        if query(sql + " SELECT ROW_COUNT();") != "1":
            raise ValueError("Fixture expected exactly1 owned profile")
        report.update({"after": query(f"SELECT state,base_level_score FROM language_learning_profile WHERE user_id={self.user_id};"), "status": "COMPLETED"})
        bootstrap._write_json(artifact, report)
        return report

    def counts(self) -> dict[str, Any]:
        query = (
            "SELECT s.id,s.domain,s.mode,s.generation_status,COUNT(DISTINCT q.id),COUNT(DISTINCT a.id) "
            "FROM language_learning_practice_set s LEFT JOIN language_learning_practice_question q ON q.practice_set_id=s.id "
            "LEFT JOIN language_learning_practice_attempt a ON a.question_id=q.id "
            f"WHERE s.user_id={self.user_id} GROUP BY s.id,s.domain,s.mode,s.generation_status ORDER BY s.id;"
        )
        rows = bootstrap._qa_mysql_query(self.manifest, self.private, query)
        report = {"userId": self.user_id, "columns": ["setId", "domain", "mode", "generationStatus", "questionRows", "attemptRows"],
                  "rows": [row.split("\t") for row in rows.splitlines()], "queryType": "SELECT_ONLY"}
        bootstrap._write_json(self.output / f"counts-{time.time_ns()}.json", report)
        return report

    def practice_snapshot(self, set_id: int) -> dict[str, Any]:
        if type(set_id) is not int or set_id <= 0:
            raise ValueError("Positive owned practice set ID required")
        raw = bootstrap._qa_mysql_query(self.manifest, self.private, (
            "SELECT id,generation_status,HEX(generation_request_json),generation_token,generation_started_at "
            f"FROM language_learning_practice_set WHERE user_id={self.user_id} AND id={set_id};"
        ))
        if not raw or len(raw.splitlines()) != 1:
            raise ValueError("Expected exactly one synthetic user's practice set")
        identity, status, encoded, generation_token, generation_started_at = raw.split("\t")
        request = json.loads(bytes.fromhex(encoded).decode("utf-8"))
        rows = bootstrap._qa_mysql_query(self.manifest, self.private, (
            "SELECT q.id,q.order_no,HEX(q.canonical_key),HEX(q.target_expression),HEX(q.correct_answer_json),q.review_target,"
            "SHA2(CONCAT_WS('|',q.prompt,q.options_json,q.correct_answer_json,q.target_expression,q.canonical_key,q.explanation_origin,q.explanation_learning),256) "
            "FROM language_learning_practice_question q JOIN language_learning_practice_set s ON s.id=q.practice_set_id "
            f"WHERE s.user_id={self.user_id} AND s.id={set_id} ORDER BY q.order_no;"
        ))
        questions = []

        def decode(value: str) -> str | None:
            return None if value == "NULL" else bytes.fromhex(value).decode("utf-8")

        for row in rows.splitlines():
            question_id, order, canonical, target, answer, review, content_hash = row.split("\t")
            questions.append({"questionId": int(question_id), "order": int(order), "canonicalKey": decode(canonical),
                              "targetExpression": decode(target), "correctAnswer": json.loads(decode(answer) or "null"),
                              "reviewTarget": review == "1", "contentSha256": content_hash})
        path = self.output / f"practice-snapshot-{set_id}-{time.time_ns()}.json"
        bootstrap._write_json(path, {"practiceSetId": int(identity), "generationStatus": status,
                                    "generationToken": generation_token, "generationStartedAt": generation_started_at,
                                    "generationRequestSnapshot": request, "persistedQuestions": questions,
                                    "boundary": "QA-only synthetic account; reference answers are test oracle, not human performance"})
        return {"practiceSetId": int(identity), "generationStatus": status, "persistedQuestionCount": len(questions),
                "persistedPlanPresent": request.get("vocabularyPlan") is not None, "privateArtifact": str(path)}

    def listening_snapshot(self, set_id: int) -> dict[str, Any]:
        if type(set_id) is not int or set_id <= 0:
            raise ValueError("Positive owned Listening set ID required")
        raw = bootstrap._qa_mysql_query(self.manifest, self.private, (
            "SELECT id,status,target_item_count,physical_item_count,learning_mode,difficulty "
            f"FROM language_learning_listening_daily_set WHERE user_id={self.user_id} AND id={set_id};"
        ))
        if not raw or len(raw.splitlines()) != 1:
            raise ValueError("Expected exactly one synthetic user's Listening set")
        identity, status, target_count, physical_count, mode, difficulty = raw.split("\t")
        rows = bootstrap._qa_mysql_query(self.manifest, self.private, (
            "SELECT i.id,i.item_index,i.status,HEX(i.source_text),HEX(i.generation_metadata),i.audio_duration_ms,i.audio_content_type,"
            "i.replacement_sequence,i.failure_reason,i.audio_checksum,i.audio_object_key "
            "FROM language_learning_listening_item i JOIN language_learning_listening_daily_set s ON s.id=i.daily_set_id "
            f"WHERE s.user_id={self.user_id} AND s.id={set_id} ORDER BY i.item_index,i.replacement_sequence;"
        ))
        items = []
        for row in rows.splitlines():
            item_id, index, item_status, script, metadata, duration, mime, sequence, failure, checksum, object_key = row.split("\t")
            items.append({"itemId": int(item_id), "itemIndex": int(index), "status": item_status,
                          "sourceText": bytes.fromhex(script).decode("utf-8"),
                          "generationMetadata": json.loads(bytes.fromhex(metadata).decode("utf-8")),
                          "audioDurationMs": None if duration == "NULL" else int(duration), "audioContentType": mime,
                          "replacementSequence": int(sequence), "failureReason": None if failure == "NULL" else failure,
                          "audioChecksum": None if checksum == "NULL" else checksum,
                          "audioObjectKey": None if object_key == "NULL" else object_key})
        path = self.output / f"listening-snapshot-{set_id}-{time.time_ns()}.json"
        bootstrap._write_json(path, {"dailySetId": int(identity), "status": status, "mode": mode, "difficulty": difficulty,
                                    "targetItemCount": int(target_count), "physicalItemCount": int(physical_count),
                                    "items": items, "boundary": "QA-only synthetic reference oracle, not human comprehension evidence"})
        return {"dailySetId": int(identity), "status": status, "mode": mode, "persistedItemCount": len(items),
                "privateArtifact": str(path)}

    def prepare_fixtures(self) -> dict[str, Any]:
        directory = self.output / "fixtures"
        directory.mkdir(exist_ok=False)
        for mode in ("DICTATION", "COMPREHENSION", "SUMMARY"):
            bootstrap._write_json(directory / f"listening-{mode.lower()}.json", {
                "itemCount": 5, "difficulty": "MY_LEVEL", "learningMode": mode,
                "idempotencyKey": f"{self.manifest['campaign']}-be-{mode.lower()}-1",
            })
        for mode in ("READ_ALOUD", "GUIDED", "FREE"):
            bootstrap._write_json(directory / f"speaking-{mode.lower()}.json", {
                "topicId": None, "keywordBasedTopic": False, "customTopic": "職場で予定を相談する",
                "goal": "予定の変更を丁寧に伝えて、代わりの日程を相談する", "persona": "同僚",
                "practiceMode": mode, "conversationStartMode": "AI_FIRST",
                "correctionMode": "CONVERSATION" if mode == "FREE" else "COACHING",
                "targetMinutes": 5, "voiceId": "marin", "playbackSpeed": "NORMAL",
                "idempotencyKey": f"{self.manifest['campaign']}-be-{mode.lower()}-1",
            })
        return {"status": "PREPARED_NO_HTTP_CALL", "directory": str(directory), "fixtureCount": 6}

    def renew_auth(self) -> dict[str, Any]:
        if self.browser_session is not None:
            raise ValueError("Refresh the same normal Google browser session; never switch QA identities")
        status, response = bootstrap._qa_request("/api/v1/auth/login", {
            "email": self.private["QA_APP_EMAIL"], "password": self.private["QA_APP_PASSWORD"],
        })
        token = (response.get("body") or {}).get("accessToken")
        report = {"httpStatus": status, "tokenReceived": bool(token), "providerCalls": 0,
                  "boundary": "Normal password login for the same synthetic QA account"}
        if status == 200 and isinstance(token, str) and token:
            self.private["QA_APP_ACCESS_TOKEN"] = token
            bootstrap._write_json(self.directory / "secrets.private.json", self.private)
        bootstrap._write_json(self.output / f"auth-renewal-{time.time_ns()}.json", report)
        return report

    def persistence_inventory(self) -> dict[str, Any]:
        """Read-only identities/status/scores/hashes from this owned synthetic user."""
        uid = self.user_id
        queries = {
            "practiceSets": f"SELECT id,status,generation_status,question_count,SHA2(generation_request_json,256) FROM language_learning_practice_set WHERE user_id={uid} ORDER BY id;",
            "practiceQuestions": "SELECT q.id,q.practice_set_id,q.order_no,SHA2(CONCAT_WS('|',q.prompt,q.options_json,q.correct_answer_json,q.target_expression,q.canonical_key),256) FROM language_learning_practice_question q JOIN language_learning_practice_set s ON s.id=q.practice_set_id " + f"WHERE s.user_id={uid} ORDER BY q.id;",
            "listeningSets": f"SELECT id,learning_mode,status,target_item_count,physical_item_count FROM language_learning_listening_daily_set WHERE user_id={uid} ORDER BY id;",
            "listeningItems": "SELECT i.id,i.daily_set_id,i.item_index,i.status,i.audio_duration_ms,SHA2(i.source_text,256) FROM language_learning_listening_item i JOIN language_learning_listening_daily_set s ON s.id=i.daily_set_id " + f"WHERE s.user_id={uid} ORDER BY i.id;",
            "listeningSessions": f"SELECT id,daily_set_id,status,completed_item_count,evaluated_item_count FROM language_learning_listening_session WHERE user_id={uid} ORDER BY id;",
            "listeningAttempts": "SELECT a.id,a.session_id,a.item_id,a.status,a.overall_score FROM language_learning_listening_item_attempt a JOIN language_learning_listening_session s ON s.id=a.session_id " + f"WHERE s.user_id={uid} ORDER BY a.id;",
            "listeningEvaluations": "SELECT e.id,a.item_id,e.task_type,r.status,e.score,e.evaluable FROM language_learning_listening_task_evaluation e JOIN language_learning_listening_task_response r ON r.id=e.task_response_id JOIN language_learning_listening_item_attempt a ON a.id=r.attempt_id JOIN language_learning_listening_session s ON s.id=a.session_id " + f"WHERE s.user_id={uid} ORDER BY e.id;",
            "speakingSessions": f"SELECT id,practice_mode,status,evaluation_status,completed_turns,total_duration_seconds FROM language_learning_speaking_session WHERE user_id={uid} ORDER BY id;",
            "speakingTurns": "SELECT t.id,t.session_id,t.turn_index,t.status,t.recording_revision,t.duration_seconds,SHA2(t.transcript,256),t.failed_stage,t.error_code FROM language_learning_speaking_turn t JOIN language_learning_speaking_session s ON s.id=t.session_id " + f"WHERE s.user_id={uid} ORDER BY t.id;",
            "speakingEvaluations": "SELECT e.id,e.session_id,e.status,e.overall_score FROM language_learning_speaking_evaluation e JOIN language_learning_speaking_session s ON s.id=e.session_id " + f"WHERE s.user_id={uid} ORDER BY e.id;",
            "speakingProblemEvaluations": "SELECT e.id,e.session_id,e.problem_index,e.attempt_count,e.status,e.overall_score FROM language_learning_speaking_read_aloud_problem_evaluation e JOIN language_learning_speaking_session s ON s.id=e.session_id " + f"WHERE s.user_id={uid} ORDER BY e.id;",
        }
        results = {name: [line.split("\t") for line in bootstrap._qa_mysql_query(self.manifest, self.private, query).splitlines()]
                   for name, query in queries.items()}
        path = self.output / f"persistence-inventory-{time.time_ns()}.json"
        bootstrap._write_json(path, {"userId": uid, "recordedAt": utc_now(), "data": results,
                                    "sha256": hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest()})
        return {"counts": {name: len(rows) for name, rows in results.items()}, "privateArtifact": str(path)}

    def ai_runtime_identity(self) -> dict[str, Any]:
        runtime = json.loads((self.campaign_directory / "ai-server-runtime.json").read_text(encoding="utf-8"))
        return {key: runtime.get(key) for key in (
            "pid", "host", "port", "workers", "python", "modelPolicy", "sttRuntimeModel",
            "sourceSha256", "diagnosticArtifact", "cancellationBoundary",
        )}

    def speaking_evaluation_snapshot(self, session_id: int, problem: int) -> dict[str, Any]:
        if type(session_id) is not int or session_id <= 0 or type(problem) is not int or not 0 <= problem <= 5:
            raise ValueError("Source-defined session/problem identity required")
        raw = bootstrap._qa_mysql_query(self.manifest, self.private,
            "SELECT j.id,j.status,j.last_error,HEX(j.request_json) FROM language_learning_speaking_evaluation_job j "
            "JOIN language_learning_speaking_session s ON s.id=j.session_id "
            f"WHERE s.user_id={self.user_id} AND s.id={session_id} AND j.problem_index={problem};")
        if len(raw.splitlines()) != 1:
            raise ValueError("Exactly one owned evaluation job required")
        identity, status, code, encoded = raw.split("\t")
        path = self.output / f"speaking-evaluation-request-{session_id}-{problem}-{time.time_ns()}.json"
        request = json.loads(bytes.fromhex(encoded).decode("utf-8"))
        bootstrap._write_json(path, {"jobId": int(identity), "sessionId": session_id, "problemIndex": problem,
            "status": status, "failureCode": code, "request": request,
            "provenance": "Exact persisted QA BE evaluation job snapshot; not captured HTTP bytes"})
        return {"status": status, "failureCode": code, "privateArtifact": str(path)}

    def speaking_turn_snapshot(self, session_id: int, turn_id: int) -> dict[str, Any]:
        if any(type(value) is not int or value <= 0 for value in (session_id, turn_id)):
            raise ValueError("Positive session/turn identity required")
        session_fields = ("id", "origin_language", "learning_language", "topic_title", "practice_mode", "topic_category",
                          "goal", "persona", "conversation_start_mode", "correction_mode", "profile_snapshot", "selected_keywords_json",
                          "session_summary", "total_duration_seconds", "policy_snapshot", "max_turns", "opening_conversation_json")
        turn_fields = ("id", "turn_index", "problem_index", "attempt_index", "recording_revision", "idempotency_key",
                       "manual_retry_count", "transcript", "stt_confidence", "stt_segments_json", "stt_metadata_json",
                       "user_audio_object_key", "user_audio_content_type", "duration_seconds", "status", "conversation_json")
        results = {}
        for name, fields, table, predicate in (
            ("session", session_fields, "language_learning_speaking_session", f"id={session_id} AND user_id={self.user_id}"),
            ("turn", turn_fields, "language_learning_speaking_turn", f"id={turn_id} AND session_id IN(SELECT id FROM language_learning_speaking_session WHERE id={session_id} AND user_id={self.user_id})"),
        ):
            fields_sql = ",".join(f"'{field}',{field}" for field in fields)
            raw = bootstrap._qa_mysql_query(self.manifest, self.private, f"SELECT HEX(JSON_OBJECT({fields_sql})) FROM {table} WHERE {predicate};")
            if len(raw.splitlines()) != 1:
                raise ValueError("Exactly one owned session/turn snapshot required")
            results[name] = json.loads(bytes.fromhex(raw).decode("utf-8"))
        path = self.output / f"speaking-turn-source-{session_id}-{turn_id}-{time.time_ns()}.json"
        bootstrap._write_json(path, {"data": results, "provenance": "Persisted state after failed QA turn, not captured original wire request",
                                     "requestReconstructionRequiresSourceAndLedgerHashCheck": True})
        return {"sessionId": session_id, "turnId": turn_id, "privateArtifact": str(path)}


def summary(record: dict[str, Any]) -> dict[str, Any]:
    body = (record.get("response") or {}).get("body") or {}
    safe_keys = {"practiceSetId", "dailySetId", "sessionId", "generationStatus", "status", "questionCount",
                 "generatedQuestionCount", "answeredCount", "correctCount", "officialScore", "readyItemCount",
                 "physicalItemCount", "targetItemCount", "completedItemCount", "evaluatedItemCount", "generationInProgress"}
    return {"ordinal": record.get("ordinal"), "action": record.get("action"), "status": record.get("status"),
            "httpStatus": record.get("httpStatus"), "body": {key: value for key, value in body.items() if key in safe_keys} if isinstance(body, dict) else None,
            "audio": record.get("audio"), "artifactOnlyRawContent": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=[*ROUTES, "seed-profile-fixture", "counts", "prepare-fixtures", "practice-snapshot", "listening-snapshot", "renew-auth", "persistence-inventory", "speaking-evaluation-snapshot", "speaking-turn-snapshot"])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--run-label", help="Separate owned QA evidence run (letters, digits, hyphens only)")
    for name in ("set", "question", "session", "item", "attempt", "turn"):
        parser.add_argument(f"--{name}-id", type=int)
    parser.add_argument("--problem-index", type=int)
    parser.add_argument("--task-type", choices=sorted(TASK_TYPES))
    parser.add_argument("--request-json", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--allow-provider-work", action="store_true")
    parser.add_argument("--timeout", type=float, default=40)
    parser.add_argument("--max-polls", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=600)
    args = parser.parse_args()
    if args.run_label is not None and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.run_label):
        raise ValueError("Invalid QA run label")
    output_directory = args.directory / "be-api-runs" / args.run_label if args.run_label else None
    client = Client(args.directory, output_directory=output_directory)
    if args.action == "seed-profile-fixture":
        result = client.seed_profile()
    elif args.action == "counts":
        result = client.counts()
    elif args.action == "prepare-fixtures":
        result = client.prepare_fixtures()
    elif args.action == "practice-snapshot":
        result = client.practice_snapshot(args.set_id)
    elif args.action == "listening-snapshot":
        result = client.listening_snapshot(args.set_id)
    elif args.action == "renew-auth":
        result = client.renew_auth()
    elif args.action == "persistence-inventory":
        result = client.persistence_inventory()
    elif args.action == "speaking-evaluation-snapshot":
        result = client.speaking_evaluation_snapshot(args.session_id, args.problem_index)
    elif args.action == "speaking-turn-snapshot":
        result = client.speaking_turn_snapshot(args.session_id, args.turn_id)
    else:
        if args.max_polls > 1:
            result = summary(client.poll(args.action, vars(args), max_polls=args.max_polls, max_seconds=args.max_seconds))
            print(json.dumps(result, ensure_ascii=True, indent=2))
            return
        payload = json.loads(args.request_json.read_text(encoding="utf-8")) if args.request_json else None
        if args.action == "configure" and payload is None:
            payload = {"originLanguage": "ko", "learningLanguage": "ja", "timezone": "Asia/Seoul",
                       "dailySentenceCount": 5, "dailyListeningGoalCount": 5}
        result = summary(client.call(args.action, identifiers=vars(args), payload=payload,
                                     allow_provider=args.allow_provider_work, audio_path=args.audio, timeout=args.timeout))
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
