from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import qa_campaign_integration as qa


def test_atomic_qa_snapshot_survives_one_windows_sharing_violation(tmp_path, monkeypatch):
    destination = tmp_path / "diagnostic.json"
    destination.write_text('{"previous": true}', encoding="utf-8")
    real_replace = qa.os.replace
    calls = 0

    def sharing_once(source, target):
        nonlocal calls
        calls += 1
        if calls == 1:
            assert json.loads(destination.read_text(encoding="utf-8")) == {"previous": True}
            raise PermissionError(5, "temporary sharing violation")
        return real_replace(source, target)

    monkeypatch.setattr(qa.os, "replace", sharing_once)
    monkeypatch.setattr(qa.time, "sleep", lambda delay: None)
    qa._write_json(destination, {"complete": True})
    assert calls == 2
    assert json.loads(destination.read_text(encoding="utf-8")) == {"complete": True}


def test_atomic_qa_snapshot_does_not_mask_permanent_permission_error(tmp_path, monkeypatch):
    destination = tmp_path / "diagnostic.json"
    destination.write_text('{"previous": true}', encoding="utf-8")
    monkeypatch.setattr(qa.os, "replace", lambda *_: (_ for _ in ()).throw(PermissionError(5)))
    monkeypatch.setattr(qa.time, "sleep", lambda delay: None)
    with pytest.raises(PermissionError):
        qa._write_json(destination, {"complete": True})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"previous": True}


def _report(directory: Path) -> dict:
    return {
        "owner": qa.OWNER, "campaign": "test-isolated",
        "names": qa.resource_names("test-isolated"), "ports": qa.PORTS,
        "images": {kind: {"id": f"sha256:{kind}", "digests": []} for kind in ("mysql", "redis")},
        "readyForFreshInfrastructure": True, "status": "PREPARED",
        "infraHealth": {"hostPublishedPortsReady": {"mysql": True, "redis": True}},
        "directory": str(directory), "createdResources": [],
    }


@pytest.mark.parametrize("value", ["../prod", "production;docker", "UPPER", "", "aa", "a" * 41])
def test_campaign_rejects_unsafe_names(value):
    with pytest.raises(ValueError):
        qa.resource_names(value)


def test_properties_target_only_qa_ports_local_storage_and_redact_secrets(tmp_path):
    names = qa.resource_names("test-isolated")
    properties = qa.be_properties(tmp_path, names)
    assert "127.0.0.1:13316/translacat_qa_test_isolated" in properties
    assert "ai-server.url=http://127.0.0.1:18082" in properties
    assert "spring.data.redis.port=16389" in properties
    assert "translacat.storage.type=local" in properties
    assert "ApiLoggingFilter=OFF" in properties
    assert "logging.level.jdbc=OFF" in properties
    assert "${QA_DB_PASSWORD}" in properties
    assert "${QA_JWT_SECRET}" in properties
    assert "${QA_AI_API_KEY}" in properties
    assert "${QA_GOOGLE_CLIENT_ID:qa-disabled}" in properties
    assert "http://localhost:3000" in properties
    assert "http://localhost:13010" in properties
    assert "jdbc:log4jdbc" not in properties
    assert "spring.profiles.active=local" not in properties
    assert "s3" not in properties


def test_remote_docker_host_is_refused_before_any_command(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "tcp://remote.invalid:2375")
    monkeypatch.setattr(qa, "_docker", lambda *args, **kwargs: pytest.fail("must not call Docker"))
    with pytest.raises(ValueError, match="DOCKER_HOST"):
        qa.preflight("test-isolated")


def test_nonlocal_docker_context_refused(monkeypatch):
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setattr(qa, "_docker", lambda *args, **kwargs: "remote" if args[1] == "show" else "ssh://remote.invalid")
    with pytest.raises(ValueError, match="local Windows"):
        qa.preflight("test-isolated")


def test_start_infra_rejects_existing_resources_without_writes(tmp_path, monkeypatch):
    qa._write_json(tmp_path / "manifest.json", _report(tmp_path))
    monkeypatch.setattr(qa, "preflight", lambda campaign: {"readyForFreshInfrastructure": False})
    monkeypatch.setattr(qa, "_docker", lambda *args, **kwargs: pytest.fail("no mutations"))
    with pytest.raises(ValueError, match="collision"):
        qa.start_infrastructure(tmp_path)


def test_start_infra_uses_fresh_labels_pinned_images_loopback_and_secret_env_only(tmp_path, monkeypatch):
    report = _report(tmp_path)
    qa._write_json(tmp_path / "manifest.json", report)
    qa._write_json(tmp_path / "secrets.private.json", {"QA_DB_PASSWORD": "not-a-real-db-secret", "QA_DB_ROOT_PASSWORD": "not-a-real-root-secret"})
    monkeypatch.setattr(qa, "preflight", lambda campaign: report.copy())
    calls = []

    def capture(*args, environment=None):
        calls.append((args, environment))
        return f"created-{len(calls)}"

    monkeypatch.setattr(qa, "_docker", capture)
    result = qa.start_infrastructure(tmp_path)
    assert result["status"] == "INFRASTRUCTURE_STARTED_NOT_APPLICATION_READY"
    assert len(result["createdResources"]) == 5
    assert calls[0][0][:3] == ("network", "create", "--internal")
    for args, _ in calls:
        assert "qa.owner=translacat-isolated-qa" in args
        assert "qa.campaign=test-isolated" in args
        assert "not-a-real-db-secret" not in str(args)
        assert "not-a-real-root-secret" not in str(args)
    mysql_args, mysql_env = calls[-2]
    assert "127.0.0.1:13316:3306" in mysql_args
    assert mysql_args[-1] == "sha256:mysql"
    assert "--pull=never" in mysql_args
    assert mysql_env["MYSQL_PASSWORD"] == "not-a-real-db-secret"
    assert "127.0.0.1:16389:6379" in calls[-1][0]
    persisted = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert "not-a-real" not in persisted


def test_partial_start_failure_preserves_owned_manifest_without_deleting(tmp_path, monkeypatch):
    report = _report(tmp_path)
    qa._write_json(tmp_path / "manifest.json", report)
    qa._write_json(tmp_path / "secrets.private.json", {"QA_DB_PASSWORD": "fake", "QA_DB_ROOT_PASSWORD": "fake"})
    monkeypatch.setattr(qa, "preflight", lambda campaign: report.copy())
    calls = []

    def fail_volume(*args, **kwargs):
        calls.append(args)
        if args[0] == "volume":
            raise RuntimeError("injected")
        return "network-owned"

    monkeypatch.setattr(qa, "_docker", fail_volume)
    with pytest.raises(RuntimeError, match="injected"):
        qa.start_infrastructure(tmp_path)
    persisted = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "PARTIAL_INFRASTRUCTURE_FAILURE_NO_AUTOMATIC_DELETION"
    assert persisted["createdResources"][0]["id"] == "network-owned"
    assert not any("rm" in args or "prune" in args for args in calls)


def test_bootstrap_api_refuses_generation_endpoint_before_network(monkeypatch):
    monkeypatch.setattr(qa.urllib.request, "build_opener", lambda *args: pytest.fail("no network"))
    with pytest.raises(ValueError, match="generation"):
        qa._qa_request("/api/v1/language-learning/practice/today")


def test_be_launch_does_not_inherit_developer_db_or_jvm_overrides(tmp_path, monkeypatch):
    report = _report(tmp_path)
    monkeypatch.setattr(qa, "check_infrastructure", lambda directory: report)
    monkeypatch.setattr(qa, "_ports_free", lambda: {key: True for key in qa.PORTS})
    monkeypatch.setenv("SPRING_DATASOURCE_URL", "jdbc:mysql://forbidden/production")
    monkeypatch.setenv("SPRING_APPLICATION_JSON", '{"spring":{"profiles":{"active":"prod"}}}')
    monkeypatch.setenv("JAVA_TOOL_OPTIONS", "-Dspring.datasource.url=forbidden")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-must-not-enter-be")
    qa._write_json(tmp_path / "secrets.private.json", {"QA_DB_PASSWORD": "qa-fake"})
    (tmp_path / "application-qa.properties").write_text(qa.be_properties(tmp_path, report["names"]), encoding="utf-8")
    repository = tmp_path / "be"
    libs = repository / "build" / "libs"
    libs.mkdir(parents=True)
    (libs / "qa.jar").write_bytes(b"fixture")
    invocations = []

    class Process:
        pid = 4242

    def spawn(command, **kwargs):
        invocations.append((command, kwargs))
        return Process()

    monkeypatch.setattr(qa.subprocess, "Popen", spawn)
    result = qa.start_be(tmp_path, repository, tmp_path / "java")
    assert result["beProcess"]["pid"] == 4242
    command, kwargs = invocations[0]
    assert "--spring.profiles.active=qa" in command
    assert kwargs["env"]["QA_DB_PASSWORD"] == "qa-fake"
    assert not {"SPRING_DATASOURCE_URL", "SPRING_APPLICATION_JSON", "JAVA_TOOL_OPTIONS", "OPENAI_API_KEY"} & set(kwargs["env"])
    assert "qa-fake" not in str(command)


def _auth_fixture(tmp_path, monkeypatch):
    manifest = _report(tmp_path)
    manifest["beProcess"] = {"command": ["java", "-jar", "synthetic-qa.jar"]}
    monkeypatch.setattr(qa, "_owned_manifest", lambda directory: manifest)
    qa._write_json(tmp_path / "auth-reproduction.json", {
        "campaign": "test-isolated", "registeredUserId": 1,
        "registerHttpStatus": 200, "loginHttpStatus": 500, "accessTokenReceived": False,
    })
    qa._write_json(tmp_path / "secrets.private.json", {
        "QA_DB_PASSWORD": "synthetic", "QA_APP_EMAIL": "qa-test-isolated@example.invalid",
        "QA_APP_PASSWORD": "synthetic-password",
    })
    return manifest


def test_auth_fixture_only_mutates_exact_new_synthetic_identity_and_uses_normal_login(tmp_path, monkeypatch):
    _auth_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(qa, "_bcrypt_fixture_hash", lambda *args: "$2a$12$synthetic-only")
    queries = []
    calls = []

    def query(manifest, private, sql):
        queries.append(sql)
        return "1"

    def api(path, payload):
        calls.append((path, payload))
        return 200, {"body": {"accessToken": "synthetic-private-token"}}

    monkeypatch.setattr(qa, "_qa_mysql_query", query)
    monkeypatch.setattr(qa, "_qa_request", api)
    result = qa.seed_qa_auth_fixture(tmp_path, Path("java"))
    assert result["accessTokenReceived"] is True
    assert result["providerCalls"] == 0
    assert len(queries) == 2
    for sql in queries:
        assert "id=1 AND email=CONVERT(0x" in sql
        assert "AND password=CONVERT(0x" in sql
        assert "AND social_type='LOCAL'" in sql
    assert calls[0][0] == "/api/v1/auth/login"
    assert "synthetic-private-token" not in json.dumps(result)
    original = json.loads((tmp_path / "auth-reproduction.json").read_text(encoding="utf-8"))
    assert original["loginHttpStatus"] == 500
    with pytest.raises(ValueError, match="one-shot"):
        qa.seed_qa_auth_fixture(tmp_path, Path("java"))


def test_auth_fixture_identity_mismatch_prevents_write(tmp_path, monkeypatch):
    _auth_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(qa, "_qa_mysql_query", lambda *args: "0")
    monkeypatch.setattr(qa, "_bcrypt_fixture_hash", lambda *args: pytest.fail("no hash or mutation"))
    with pytest.raises(ValueError, match="no write"):
        qa.seed_qa_auth_fixture(tmp_path, Path("java"))
    assert not (tmp_path / "auth-fixture-result.json").exists()


def test_auth_fixture_non_synthetic_email_prevents_even_sql(tmp_path, monkeypatch):
    _auth_fixture(tmp_path, monkeypatch)
    private_path = tmp_path / "secrets.private.json"
    private = json.loads(private_path.read_text(encoding="utf-8"))
    private["QA_APP_EMAIL"] = "someone@example.com"
    qa._write_json(private_path, private)
    monkeypatch.setattr(qa, "_qa_mysql_query", lambda *args: pytest.fail("no SQL"))
    with pytest.raises(ValueError, match="synthetic"):
        qa.seed_qa_auth_fixture(tmp_path, Path("java"))


def test_be_cannot_launch_when_internal_db_ready_but_host_port_unpublished(tmp_path, monkeypatch):
    manifest = _report(tmp_path)
    manifest["infraHealth"]["hostPublishedPortsReady"]["mysql"] = False
    monkeypatch.setattr(qa, "check_infrastructure", lambda directory: manifest)
    monkeypatch.setattr(qa.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("no launch"))
    with pytest.raises(ValueError, match="host-published"):
        qa.start_be(tmp_path, tmp_path, Path("java"))


def test_google_qa_audience_uses_next_env_and_never_exposes_value(tmp_path, monkeypatch):
    monkeypatch.setattr(qa, "_owned_manifest", lambda directory: _report(tmp_path))
    fe = tmp_path / "fe"
    (fe / "node_modules" / "@next" / "env").mkdir(parents=True)
    qa._write_json(tmp_path / "secrets.private.json", {"QA_DB_PASSWORD": "preserved-synthetic"})
    calls = []
    client_id = "synthetic-qa.apps.googleusercontent.com"

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"clientId": client_id}))

    monkeypatch.setattr(qa.subprocess, "run", run)
    result = qa.configure_qa_google(tmp_path, fe, Path("node"))
    private = json.loads((tmp_path / "secrets.private.json").read_text(encoding="utf-8"))
    assert private == {"QA_DB_PASSWORD": "preserved-synthetic", "QA_GOOGLE_CLIENT_ID": client_id}
    assert "loadEnvConfig(root,true" in calls[0][2]
    assert client_id not in json.dumps(result)
    assert client_id not in (tmp_path / "google-audience-configuration.json").read_text(encoding="utf-8")
    assert result["productionConfigurationChanged"] is False


def test_invalid_google_audience_does_not_write_private_config(tmp_path, monkeypatch):
    monkeypatch.setattr(qa, "_owned_manifest", lambda directory: _report(tmp_path))
    fe = tmp_path / "fe"
    (fe / "node_modules" / "@next" / "env").mkdir(parents=True)
    private = tmp_path / "secrets.private.json"
    private.write_text('{"preserved": true}', encoding="utf-8")
    monkeypatch.setattr(qa.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=json.dumps({"clientId": "invalid-sensitive-output"}),
    ))
    with pytest.raises(ValueError, match="output suppressed") as caught:
        qa.configure_qa_google(tmp_path, fe, Path("node"))
    assert "invalid-sensitive-output" not in str(caught.value)
    assert private.read_text(encoding="utf-8") == '{"preserved": true}'
