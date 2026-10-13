from __future__ import annotations

import os
import ssl

import httpx2
from openai import DefaultAsyncHttpx2Client


def create_openai_http_client() -> httpx2.AsyncClient:
    # CA를 연결 전에 한 번만 읽는다. truststore.wrap_bio는 동시 연결마다
    # 공유 X509_STORE를 변경하여 Linux libcrypto에서 native crash를 일으킬 수 있다.
    if cafile := os.environ.get("SSL_CERT_FILE"):
        context = ssl.create_default_context(cafile=cafile)
    elif capath := os.environ.get("SSL_CERT_DIR"):
        context = ssl.create_default_context(capath=capath)
    else:
        context = ssl.create_default_context()

    # 인증서/호스트 검증과 SDK 기본 pool/timeout은 보존한다. 연결 중 CA를 변경하지 않는다.
    return DefaultAsyncHttpx2Client(verify=context)
