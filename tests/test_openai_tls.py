from __future__ import annotations

import asyncio
import json
import ssl
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Barrier, Lock, Thread
from unittest.mock import patch

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from openai import APIConnectionError, AsyncOpenAI

from app.ai.providers.openai.client import OpenAIService
from app.ai.providers.openai.http_client import create_openai_http_client
from app.ai.providers.openai.speech import OpenAISpeechService
from app.core.config import settings
from app.features.voice_translation.hosted_stt import OpenAIWhisperVoiceSttProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["receipt", "speech", "hosted_stt"])
async def test_owned_clients_load_certificates_before_parallel_connections(
    monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    # 준비: 실제 SDK를 생성하되 네트워크 요청과 실제 키는 사용하지 않는다.
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "local-test-only")
    if kind == "hosted_stt":
        provider = OpenAIWhisperVoiceSttProvider(api_key="local-test-only")
        client = provider._client
    else:
        provider = OpenAIService() if kind == "receipt" else OpenAISpeechService()
        client = provider.client
    assert client is not None

    # 실행: 병렬 연결이 실제로 공유하는 transport의 TLS 문맥을 검사한다.
    try:
        context = client._client._transport._pool._ssl_context

        # 검증: 연결마다 CA 저장소를 변경하는 truststore wrapper를 사용하지 않는다.
        assert type(context) is ssl.SSLContext
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True
        assert client.max_retries == 0
    finally:
        if kind == "hosted_stt":
            await client.close()
        else:
            await provider.shutdown()
    assert client.is_closed()


@pytest.mark.asyncio
async def test_native_context_supports_parallel_wrap_bio() -> None:
    # 준비: 한 클라이언트의 문맥을 실제로 세 스레드가 함께 사용한다.
    client = create_openai_http_client()
    context = client._transport._pool._ssl_context
    barrier = Barrier(3)

    def connect() -> None:
        for _ in range(30):
            barrier.wait(timeout=5)
            context.wrap_bio(ssl.MemoryBIO(), ssl.MemoryBIO(), server_hostname="localhost")

    # 실행: 소켓/AI 없이 실제 TLS 객체 생성 경계를 동시에 반복한다.
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(connect) for _ in range(3)]
            for future in futures:
                future.result(timeout=15)

        # 검증: 실패 없이 반환되고 보안 설정은 유지된다.
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["system", "file", "directory"])
async def test_ca_configuration_is_loaded_once(
    monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    # 준비: 기존 HTTPX2와 동일하게 명시한 CA 파일을 디렉터리보다 우선한다.
    context = ssl.create_default_context()
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    expected = {}
    if mode in {"file", "directory"}:
        monkeypatch.setenv("SSL_CERT_DIR", "/configured/ca-directory")
        expected = {"capath": "/configured/ca-directory"}
    if mode == "file":
        monkeypatch.setenv("SSL_CERT_FILE", "/configured/ca.pem")
        expected = {"cafile": "/configured/ca.pem"}

    # 실행
    with patch(
        "app.ai.providers.openai.http_client.ssl.create_default_context", return_value=context,
    ) as create:
        client = create_openai_http_client()
        try:
            # 검증: 연결 생성 때 CA를 다시 로드하는 wrapper가 없다.
            create.assert_called_once_with(**expected)
            assert client._transport._pool._ssl_context is context
            assert client.follow_redirects
        finally:
            await client.aclose()


def test_invalid_explicit_ca_file_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    # 준비
    monkeypatch.setenv("SSL_CERT_FILE", "/missing/receipt-test-ca.pem")

    # 실행 / 검증: 잘못된 CA 설정을 무시하거나 verify=False로 우회하지 않는다.
    with pytest.raises(FileNotFoundError):
        create_openai_http_client()


@pytest.fixture
def tls_server(tmp_path: Path):
    # 준비: 외부 접속 없이 신뢰/호스트 검증을 시험할 localhost 인증서를 만든다.
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
        .not_valid_after(datetime.now(UTC) + timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = tmp_path / "ca.pem", tmp_path / "key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    counts = {"active": 0, "peak": 0, "requests": 0}
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            with lock:
                counts["active"] += 1
                counts["requests"] += 1
                counts["peak"] = max(counts["peak"], counts["active"])
            try:
                time.sleep(0.04)
                body = json.dumps({"object": "list", "data": []}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            finally:
                with lock:
                    counts["active"] -= 1

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, cert_path, counts
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.asyncio
async def test_sdk_parallel_https_keeps_certificate_verification(
    monkeypatch: pytest.MonkeyPatch, tls_server,
) -> None:
    # 준비: 시험용 CA만 신뢰하고 실제 SDK의 소켓/TLS 경로를 사용한다.
    port, cert_path, counts = tls_server
    monkeypatch.setenv("SSL_CERT_FILE", str(cert_path))
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    async with AsyncOpenAI(
        api_key="local-test-only", base_url=f"https://localhost:{port}/v1/",
        http_client=create_openai_http_client(), max_retries=0,
    ) as client:
        # 실행: 매 응답 후 연결이 닫혀 새 TLS 연결도 병렬로 검증된다.
        for _ in range(10):
            results = await asyncio.gather(*(client.models.list() for _ in range(3)))
            assert all(result.data == [] for result in results)

        # 검증: 직렬화하지 않았고 30회 모두 성공했다.
        assert counts["peak"] == 3
        assert counts["requests"] == 30

    # 실행 / 검증: 신뢰된 인증서도 호스트가 다르면 거부한다.
    async with AsyncOpenAI(
        api_key="local-test-only", base_url=f"https://127.0.0.1:{port}/v1/",
        http_client=create_openai_http_client(), max_retries=0,
    ) as client:
        with pytest.raises(APIConnectionError):
            await client.models.list()

    # 실행 / 검증: 시스템이 신뢰하지 않는 인증서도 거부한다.
    monkeypatch.delenv("SSL_CERT_FILE")
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    async with AsyncOpenAI(
        api_key="local-test-only", base_url=f"https://localhost:{port}/v1/",
        http_client=create_openai_http_client(), max_retries=0,
    ) as client:
        with pytest.raises(APIConnectionError):
            await client.models.list()
    assert counts["requests"] == 30
