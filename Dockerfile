# 1. Pythonの軽量イメージを使用
FROM python:3.11-slim

# 2. 作業ディレクトリを設定
WORKDIR /app

# 3. 依存関係ファイルをコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. シースコードを全てコピー
COPY . .

# 5. ポート8000でサーバーを起動
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]