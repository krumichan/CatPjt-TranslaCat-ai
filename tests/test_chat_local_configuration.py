import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("chat_ai_local", ROOT / "scripts/chat_ai_local.py")
assert SPEC and SPEC.loader
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)
KEY = "local-contract-only-chat-credential-0001"


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    # 준비: 실제 개발 파일을 읽지 않는 합성 공급원을 만든다.
    for name in tuple(os.environ):
        if name.startswith(
            ("CHAT_", "AI_SETTINGS_", "SERVER_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY")
        ):
            monkeypatch.delenv(name, raising=False)
    for name in (
        "CHAT_AUTH_MODE",
        "CHAT_AUTH_ENVIRONMENT",
        "CHAT_SERVER_API_KEY",
        "AI_SETTINGS_ENV_FILE",
        "SERVER_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "AI_VOICE_WARM_UP",
        "OCR_WARM_UP",
    ):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    environment = tmp_path / "selected.env"
    environment.write_text(
        "SERVER_API_KEY=distinct-global-test-key\nOPENAI_API_KEY=synthetic-provider-key\nOCR_WARM_UP=true\n",
        encoding="utf-8",
    )
    secret = tmp_path / "chat-key"
    secret.write_text(KEY, encoding="ascii")
    return environment, secret


def test_prepare_reuses_single_key_and_preserves_source_files(inputs, monkeypatch):
    # 준비
    environment, secret = inputs
    before = (environment.read_bytes(), secret.read_bytes())
    monkeypatch.setenv("CHAT_AUTH_MODE", "dedicated")

    # 실행
    result = launcher.prepare_environment(environment, secret, "Development", True)

    # 검증
    assert os.environ["CHAT_SERVER_API_KEY"] == KEY
    assert os.environ["AI_SETTINGS_ENV_FILE"] == str(environment)
    assert os.environ["AI_VOICE_WARM_UP"] == "false"
    assert os.environ["OCR_WARM_UP"] == "false"
    assert (environment.read_bytes(), secret.read_bytes()) == before
    assert result["chatApiKey"] == "APPLIED"


def test_legacy_mode_reuses_shared_global_key_without_rotating_source(inputs):
    # 준비: LL/BE/CHAT이 공유하는 로컬 키를 명시한다.
    environment, secret = inputs
    environment.write_text(f"CHAT_AUTH_MODE=legacy\nSERVER_API_KEY={KEY}\n", encoding="utf-8")
    before = (environment.read_bytes(), secret.read_bytes())

    # 실행
    result = launcher.prepare_environment(environment, secret, "Development", True, "legacy")

    # 검증: dedicated 전환 없이 동일한 글로벌 인증 계약을 유지한다.
    assert result["chatAuthMode"] == "legacy"
    assert os.environ["SERVER_API_KEY"] == KEY
    assert "CHAT_SERVER_API_KEY" not in os.environ
    assert (environment.read_bytes(), secret.read_bytes()) == before


def test_legacy_mode_rejects_conflicting_global_key_without_mutation(inputs):
    # 준비
    environment, secret = inputs
    before = dict(os.environ)

    # 실행
    with pytest.raises(launcher.LocalConfigurationError):
        launcher.prepare_environment(environment, secret, "Development", True, "legacy")

    # 검증
    assert dict(os.environ) == before


def test_live_provider_credentials_are_reused_without_importing_original_server_key(inputs):
    # 준비: provider 원본 파일과 결합 실행 설정은 다른 서버간 키를 가진다.
    environment, secret = inputs
    provider_source = environment.with_name("provider-source.env")
    provider_source.write_text(
        "SERVER_API_KEY=unrelated-original-server-key\n"
        "OPENAI_API_KEY=local-only-provider-credential\nCHAT_AUTH_MODE=dedicated\n",
        encoding="utf-8",
    )
    environment.write_text("CHAT_AUTH_MODE=legacy\nAI_TEXT_PROVIDER=openai\n", encoding="utf-8")
    before = (environment.read_bytes(), provider_source.read_bytes(), secret.read_bytes())

    # 실행
    result = launcher.prepare_environment(
        environment,
        secret,
        "Development",
        False,
        "legacy",
        provider_source,
    )

    # 검증: 메모리의 provider 키만 재사용하고 모든 입력 파일은 보존한다.
    assert os.environ["SERVER_API_KEY"] == KEY
    assert os.environ["OPENAI_API_KEY"] == "local-only-provider-credential"
    assert os.environ["CHAT_AUTH_MODE"] == "legacy"
    assert "local-only-provider-credential" not in json.dumps(result)
    assert (environment.read_bytes(), provider_source.read_bytes(), secret.read_bytes()) == before


@pytest.mark.parametrize(
    "source",
    [
        "OPENAI_API_KEY=different-local-provider-key\n",
        "OPENAI_API_KEY_FILE=unsupported-secret-file\n",
        "SERVER_API_KEY=not-a-provider-credential\n",
    ],
)
def test_live_provider_source_conflicts_or_missing_values_fail_before_mutation(inputs, source):
    # 준비: 기존 선택 파일에는 이미 다른 합성 provider 키가 있다.
    environment, secret = inputs
    provider_source = environment.with_name("provider-source.env")
    provider_source.write_text(source, encoding="utf-8")
    before = dict(os.environ)

    # 실행
    with pytest.raises(launcher.LocalConfigurationError) as failure:
        launcher.prepare_environment(
            environment,
            secret,
            "Development",
            False,
            "dedicated",
            provider_source,
        )

    # 검증
    assert dict(os.environ) == before
    assert "different-local-provider-key" not in str(failure.value)


@pytest.mark.parametrize(
    "name,value",
    [
        ("CHAT_SERVER_API_KEY", "different-private-value"),
        ("CHAT_AUTH_MODE", "legacy"),
        ("CHAT_AUTH_ENVIRONMENT", "Production"),
        ("AI_SETTINGS_ENV_FILE", "different-file"),
        ("SERVER_API_KEY", "different-global-value"),
        ("OPENAI_API_KEY", "different-provider-value"),
        ("CHAT_SERVER_API_KEY_FILE", "unsupported-file"),
    ],
)
def test_conflicting_environment_is_rejected_without_mutation(inputs, monkeypatch, name, value):
    # 준비
    environment, secret = inputs
    monkeypatch.setenv(name, value)
    before = dict(os.environ)

    # 실행
    with pytest.raises(launcher.LocalConfigurationError) as failure:
        launcher.prepare_environment(environment, secret, "Development", False)

    # 검증
    assert dict(os.environ) == before
    assert value not in str(failure.value)
    assert KEY not in str(failure.value)


@pytest.mark.parametrize(
    "content",
    [
        "CHAT_SERVER_API_KEY=different-private-value\n",
        "CHAT_AUTH_MODE=legacy\n",
        "CHAT_AUTH_ENVIRONMENT=Production\n",
        f"SERVER_API_KEY={KEY}\n",
        "CHAT_SERVER_API_KEY_FILE=unsupported-file\n",
        "CHAT_AUTH_MODE=dedicated\nchat_auth_mode=dedicated\n",
        'SERVER_API_KEY="unterminated\n',
        "\ufeffSERVER_API_KEY=unsafe-bom\n",
    ],
)
def test_conflicting_or_malformed_file_is_rejected(inputs, content):
    # 준비
    environment, secret = inputs
    environment.write_text(content, encoding="utf-8")
    before = (environment.read_bytes(), secret.read_bytes(), dict(os.environ))

    # 실행
    with pytest.raises(launcher.LocalConfigurationError):
        launcher.prepare_environment(environment, secret, "Development", False)

    # 검증
    assert (environment.read_bytes(), secret.read_bytes(), dict(os.environ)) == before


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"short",
        KEY.encode() + b"\n",
        KEY.encode() + b"\r\n",
        b"\xef\xbb\xbf" + KEY.encode(),
        ('"' + KEY + '"').encode(),
        b" " + KEY.encode(),
    ],
)
def test_malformed_secret_file_is_rejected(inputs, content):
    # 준비
    environment, secret = inputs
    secret.write_bytes(content)
    before = dict(os.environ)

    # 실행
    with pytest.raises(launcher.LocalConfigurationError):
        launcher.prepare_environment(environment, secret, "Development", False)

    # 검증
    assert dict(os.environ) == before


def test_validate_only_binds_actual_settings_and_keeps_parent_environment(inputs):
    # 준비: root의 .env가 아닌 명시한 임의 파일명을 사용한다.
    environment, secret = inputs
    before = dict(os.environ)

    # 실행: 실제 Python entry point와 Settings를 새 process에서 실행한다.
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/chat_ai_local.py"),
            "--environment",
            "Development",
            "--environment-file",
            str(environment),
            "--api-key-file",
            str(secret),
            "--disable-warmup",
            "--validate-only",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=environment.parent,
    )

    # 검증
    assert result.returncode == 0, "Synthetic local launcher failed"
    summary = json.loads(result.stdout)
    assert summary["pythonSettings"] == "VERIFIED"
    assert summary["globalCredential"] == "REUSED"
    assert summary["providerCredential"] == "REUSED"
    assert KEY not in result.stdout + result.stderr
    assert "distinct-global-test-key" not in result.stdout + result.stderr
    assert dict(os.environ) == before


def test_invalid_settings_input_does_not_leak_its_value(inputs):
    # 준비
    environment, secret = inputs
    sensitive_invalid_value = "sensitive-bad-timeout-never-echo"
    with environment.open("a", encoding="utf-8") as stream:
        stream.write(f"OPENAI_REQUEST_TIMEOUT_SECONDS={sensitive_invalid_value}\n")

    # 실행
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/chat_ai_local.py"),
            "--environment",
            "Development",
            "--environment-file",
            str(environment),
            "--api-key-file",
            str(secret),
            "--validate-only",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=environment.parent,
    )

    # 검증
    assert result.returncode == 2
    assert sensitive_invalid_value not in result.stdout + result.stderr
    assert KEY not in result.stdout + result.stderr


def test_synthetic_http_fixture_rejects_inherited_personal_settings_file(inputs):
    # 준비: 테스트 fixture에는 명시한 개인 파일도 허용하지 않는다.
    environment, _ = inputs
    child_environment = os.environ | {
        "CHAT_AUTH_CONTRACT_TEST": "true",
        "AI_SETTINGS_ENV_FILE": str(environment),
    }

    # 실행: 앱/Provider import 전에 거절되므로 서버가 시작되지 않는다.
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests/chat_model_execution_http_fixture.py")],
        cwd=environment.parent,
        env=child_environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # 검증
    assert result.returncode == 1
    assert "An isolated synthetic test directory is required" in result.stderr
    assert KEY not in result.stdout + result.stderr


def test_selected_settings_file_drives_http_authentication_without_provider(inputs, monkeypatch):
    # 준비: 실제 Settings와 HTTP middleware를 잇고 Provider port만 합성 대역으로 둔다.
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from app.ai.ports import StructuredGenerationResult
    from app.api.dependencies import get_model_execution_provider
    from app.core.config import Settings

    environment, secret = inputs
    launcher.prepare_environment(environment, secret, "Development", True)
    # 소설 수신도 항상 검증하므로 테스트 앱의 별도 소설 자격증명을 준비한다.
    # CHAT/공통 키 허용과 잘못된 범위 거부 assertion은 그대로 유지한다.
    configuration = Settings(
        _env_file=environment,
        NOVEL_AUTH_ENVIRONMENT="Development",
        NOVEL_SERVER_API_KEY="local-contract-only-novel-credential-0002",
    )
    with patch("app.core.config_logger.setup_logging"):
        from app import main as application
    monkeypatch.setattr(application, "settings", configuration)
    calls = []

    class ModelFixture:
        async def execute_explicit(self, **kwargs):
            calls.append("synthetic-model")
            return StructuredGenerationResult(
                data="synthetic translation", provider="fake", model="synthetic"
            )

    original = application.app.dependency_overrides.copy()
    application.app.dependency_overrides[get_model_execution_provider] = ModelFixture
    try:
        client = TestClient(application.app)
        payload = {
            "traceId": "synthetic",
            "instructions": "synthetic",
            "messages": [{"role": "user", "content": "synthetic"}],
            "tier": "LUNA",
            "reasoningEffort": "none",
            "verbosity": "low",
            "maxOutputTokens": 128,
            "remainingMilliseconds": 1000,
            "maxProviderCalls": 1,
        }

        # 실행
        allowed = client.post(
            "/internal/v1/model/execute", json=payload, headers={"X-API-KEY": KEY}
        )
        wrong = client.post(
            "/internal/v1/model/execute", json=payload, headers={"X-API-KEY": "wrong"}
        )
        global_key = client.post(
            "/internal/v1/model/execute",
            json=payload,
            headers={"X-API-KEY": configuration.SERVER_API_KEY},
        )
        wrong_scope = client.post(
            "/internal/v1/speech/transcribe", json={}, headers={"X-API-KEY": KEY}
        )

        # 검증: Chat와 기존 LL 키는 범용 실행을 허용하고 Chat 키의 다른 기능 접근은 거부한다.
        assert allowed.status_code == 200
        assert allowed.json()["output"] == "synthetic translation"
        assert (wrong.status_code, global_key.status_code, wrong_scope.status_code) == (
            401,
            200,
            401,
        )
        assert calls == ["synthetic-model", "synthetic-model"]
    finally:
        client.close()
        application.app.dependency_overrides.clear()
        application.app.dependency_overrides.update(original)


@pytest.mark.skipif(os.name != "nt", reason="The local PowerShell launcher targets Windows")
@pytest.mark.parametrize(
    "environment_name,port,expected_exit",
    [
        ("Development", "8000", 0),
        ("Production", "8000", 1),
        ("Development", "80", 1),
    ],
)
def test_powershell_entry_point_binds_and_rejects_invalid_mode(
    inputs, environment_name, port, expected_exit
):
    # 준비: 실제 PowerShell 진입점도 synthetic source만 읽는다.
    environment, secret = inputs
    shell = shutil.which("pwsh")
    assert shell is not None, "The documented launcher requires PowerShell 7"
    before = dict(os.environ)

    # 실행
    result = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-File",
            str(ROOT / "scripts/Start-ChatAiLocal.ps1"),
            "-Environment",
            environment_name,
            "-ApiKeyFile",
            str(secret),
            "-EnvironmentFile",
            str(environment),
            "-Port",
            port,
            "-DisableWarmup",
            "-ValidateOnly",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )

    # 검증
    assert result.returncode == expected_exit
    assert KEY not in result.stdout + result.stderr
    assert dict(os.environ) == before
    if expected_exit == 0:
        assert json.loads(result.stdout)["pythonSettings"] == "VERIFIED"
