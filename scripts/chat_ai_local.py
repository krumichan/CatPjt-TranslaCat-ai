import argparse
import io
import json
import os
import secrets
import sys
from pathlib import Path

from dotenv import dotenv_values
from dotenv.parser import parse_stream


class LocalConfigurationError(Exception):
    pass


def _read_configuration(path: Path) -> dict[str, str | None]:
    # 파싱 오류에 원문을 포함하지 않으며 중복 이름의 조용한 마지막 값 우선을 막는다.
    raw = path.read_bytes()
    if len(raw) > 1024 * 1024 or raw.startswith(b"\xef\xbb\xbf"):
        raise LocalConfigurationError(
            "AI environment file must be UTF-8 without BOM, at most 1 MiB"
        )
    text = raw.decode("utf-8", errors="strict")
    seen: set[str] = set()
    for binding in parse_stream(io.StringIO(text)):
        if binding.error:
            raise LocalConfigurationError("AI environment file contains an invalid setting")
        if binding.key is None:
            continue
        name = binding.key.upper()
        if name in seen:
            raise LocalConfigurationError("AI environment file contains a duplicate setting")
        seen.add(name)
    return {name.upper(): value for name, value in dotenv_values(stream=io.StringIO(text)).items()}


def prepare_environment(
    environment_file: Path,
    api_key_file: Path,
    environment: str,
    disable_warmup: bool,
    auth_mode: str = "dedicated",
    provider_credentials_file: Path | None = None,
) -> dict[str, object]:
    # 기존 발급자/Provider 설정은 같은 파일에서 재사용하며 Development 방향키만 연결한다.
    if environment != "Development":
        raise LocalConfigurationError("This launcher accepts Development only")
    if auth_mode not in {"legacy", "dedicated"}:
        raise LocalConfigurationError("Local auth mode must be legacy or dedicated")
    for path in (environment_file, api_key_file):
        if not path.is_absolute() or not path.is_file():
            raise LocalConfigurationError("AI local launch requires existing absolute file paths")
    supplied = _read_configuration(environment_file)
    raw_key = api_key_file.read_bytes()
    if not 32 <= len(raw_key) <= 4096 or any(
        value <= 32 or value >= 127 or value in (34, 39) for value in raw_key
    ):
        raise LocalConfigurationError(
            "CHAT API key must be 32..4096 ASCII bytes without whitespace or BOM"
        )
    key = raw_key.decode("ascii")
    current = {name.upper(): value for name, value in os.environ.items()}
    expected = {
        "CHAT_AUTH_MODE": auth_mode,
        "CHAT_AUTH_ENVIRONMENT": environment,
        "CHAT_SERVER_API_KEY" if auth_mode == "dedicated" else "SERVER_API_KEY": key,
        "AI_SETTINGS_ENV_FILE": str(environment_file),
    }

    # 기존 로컬 provider 비밀만 메모리에서 재사용한다. 원본의 서버간 키는 가져오지 않는다.
    if provider_credentials_file is not None:
        if not provider_credentials_file.is_absolute() or not provider_credentials_file.is_file():
            raise LocalConfigurationError(
                "Provider credentials require an existing absolute file path"
            )
        provider_source = _read_configuration(provider_credentials_file)
        credential_names = ("OPENAI_API_KEY", "GOOGLE_API_KEY")
        if any(provider_source.get(f"{name}_FILE") for name in credential_names):
            raise LocalConfigurationError(
                "Provider credential source must contain direct local values"
            )
        credentials = {
            name: provider_source[name] for name in credential_names if provider_source.get(name)
        }
        if not credentials:
            raise LocalConfigurationError("Selected local provider credential source is empty")
        expected.update(credentials)

    # 양측에 이미 다른 값이 있으면 자동 회전하지 않는다. 미지원 _FILE도 조용히 무시하지 않는다.
    for source in (supplied, current):
        for name, value in expected.items():
            existing = source.get(name)
            if existing and not secrets.compare_digest(existing.encode(), value.encode()):
                raise LocalConfigurationError(
                    f"AI setting conflicts with the selected source: {name}"
                )
        if any(
            name.endswith("_FILE") and name != "AI_SETTINGS_ENV_FILE"
            for name in source
            if name.startswith(("CHAT_", "SERVER_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"))
        ):
            raise LocalConfigurationError(
                "AI does not support the supplied secret _FILE environment name; use -ApiKeyFile"
            )
    for name in ("SERVER_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        if supplied.get(name) and current.get(name) and supplied[name] != current[name]:
            raise LocalConfigurationError(
                f"AI secret has conflicting environment and file sources: {name}"
            )
    for source in (supplied, current):
        global_key = source.get("SERVER_API_KEY")
        if (
            auth_mode == "dedicated"
            and global_key
            and secrets.compare_digest(global_key.encode(), raw_key)
        ):
            raise LocalConfigurationError("CHAT API key must be independent from SERVER_API_KEY")

    # 이 process에만 적용한다. -DisableWarmup은 비용/모델 다운로드 없는 기동을 위한 명시 선택이다.
    os.environ.update(expected)
    if disable_warmup:
        os.environ["AI_VOICE_WARM_UP"] = "false"
        os.environ["OCR_WARM_UP"] = "false"
    return {
        "environment": environment,
        "chatAuthMode": auth_mode,
        "environmentFile": str(environment_file),
        "chatApiKey": "APPLIED",
        "warmupDisabled": disable_warmup,
        "providerCredentialSource": str(provider_credentials_file)
        if provider_credentials_file
        else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start the actual AI service with a dedicated local Chat credential"
    )
    parser.add_argument("--environment", choices=["Development"], required=True)
    parser.add_argument("--environment-file", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--auth-mode", choices=["legacy", "dedicated"], default="dedicated")
    parser.add_argument("--provider-credentials-file", type=Path)
    parser.add_argument("--synthetic-test-provider", action="store_true")
    parser.add_argument("--artifact-directory", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--disable-warmup", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    try:
        if not 1024 <= args.port <= 65535:
            raise LocalConfigurationError("AI local port must be 1024..65535")
        if args.synthetic_test_provider and args.provider_credentials_file is not None:
            raise LocalConfigurationError(
                "Synthetic execution does not accept a live credential source"
            )
        result = prepare_environment(
            args.environment_file,
            args.api_key_file,
            args.environment,
            args.disable_warmup,
            args.auth_mode,
            args.provider_credentials_file,
        )
        # 명시한 합성 실행은 기존 Provider를 호출하지 않고 작업별 산출물 경로를 사용한다.
        if args.synthetic_test_provider:
            if args.artifact_directory is None or not args.artifact_directory.is_absolute():
                raise LocalConfigurationError(
                    "Synthetic execution requires an absolute artifact directory"
                )
            args.artifact_directory.mkdir(parents=True, exist_ok=True)
            os.environ["TRANSLACAT_TEST_MODEL_EXECUTION"] = "1"
            os.environ["TRANSLACAT_TEST_RUNTIME_DIRECTORY"] = str(args.artifact_directory)
            os.environ["AI_VOICE_WARM_UP"] = "false"
            os.environ["OCR_WARM_UP"] = "false"
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from app.core.chat_auth import validate_chat_auth_configuration
        from app.core.config import settings

        validate_chat_auth_configuration(settings)
        if args.synthetic_test_provider:
            os.environ["TRANSLACAT_TEST_SERVER_API_KEY"] = settings.SERVER_API_KEY
        result["pythonSettings"] = "VERIFIED"
        result["modelExecution"] = "SYNTHETIC" if args.synthetic_test_provider else "PROVIDER"
        result["globalCredential"] = "REUSED" if settings.SERVER_API_KEY else "NEEDS_INPUT"
        result["providerCredential"] = (
            "REUSED"
            if (
                settings.OPENAI_API_KEY
                if settings.AI_TEXT_PROVIDER == "openai"
                else settings.GOOGLE_API_KEY
            )
            else "NEEDS_INPUT"
        )
        print(json.dumps(result), flush=True)
    except LocalConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2
    except Exception:
        # Pydantic의 원래 ValidationError에는 입력 값이 포함될 수 있어 그대로 출력하지 않는다.
        print(
            "AI local configuration failed validation; secret values were suppressed",
            file=sys.stderr,
        )
        return 2

    if not args.validate_only:
        import uvicorn

        application = (
            "scripts.synthetic_model_server:app" if args.synthetic_test_provider else "app.main:app"
        )
        uvicorn.run(application, host="127.0.0.1", port=args.port, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
