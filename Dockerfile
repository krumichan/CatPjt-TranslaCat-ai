FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# PaddleOCR / OpenCV 실행에 필요한 OS 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Paddle / OpenMP / BLAS 계열 스레드 폭주 방지
# GitHub Actions에서 주입하는 앱 환경변수와 별개로,
# 컨테이너 런타임 안정성을 위해 이미지 기본값으로 둠
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=2
ENV NUMEXPR_NUM_THREADS=2
ENV FLAGS_use_mkldnn=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

HEALTHCHECK --interval=10s --timeout=3s --start-period=300s --retries=3 \
    CMD python -c "import os, urllib.request; request = urllib.request.Request('http://127.0.0.1:8000/internal/v1/voice/readiness', headers={'X-API-KEY': os.environ['SERVER_API_KEY']}); urllib.request.urlopen(request, timeout=2).read()" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
