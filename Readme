# TranslaCat AI Server

> 翻訳と音声認識に特化した FastAPI ベースの内部向け AI 処理サーバ

## 1. 概要

TranslaCat AI Server は、翻訳および Speech-to-Text を担当する AI 専用サービスです。  
Spring Boot Backend から分離された構成を採用し、AI モデル依存の処理を独立して扱いやすくしています。

このリポジトリの主な責務は以下の通りです。

- 単文翻訳
- 複数文のバッチ翻訳
- 音声ファイルの文字起こし
- 内部向け API Key 認証
- Gemini を利用した翻訳処理
- faster-whisper を利用した STT 処理

---

## 2. このサーバを分離している理由

AI 処理は、通常の業務 API とは性質が異なります。  
モデル依存、計算負荷、再試行制御、音声ファイル処理など、専用サービスとして切り出した方が管理しやすい要素が多いためです。

TranslaCat では、以下の考え方で分離しています。

- **Backend**: 認証、業務ロジック、DB、画面向け API、履歴管理
- **AI Server**: 翻訳、STT、AI モデル接続、処理集約

この分離により、AI 処理の改善や置き換えを、業務 API と比較的独立して進めやすくなります。

---

## 3. システム内での位置づけ

```mermaid
flowchart TD
    FE[Frontend] --> BE[Spring Boot Backend]
    BE --> AI[FastAPI AI Server]
    AI --> GEM[Gemini]
    AI --> WHISPER[faster-whisper]
```

AI Server は、原則としてフロントエンドから直接呼び出す前提ではありません。  
内部 API として Backend から利用される想定で設計されています。

---

## 4. 主な機能

### 4-1. 単文翻訳

1 つのテキストを受け取り、指定された翻訳ルールに従って翻訳します。  
短文 UI 文言、チャット、即時応答が必要なケース向けです。

**Endpoint**

```text
POST /api/v1/translate/single
```

**Input**

| 項目 | 型 | 説明 |
|---|---|---|
| `text` | string | 翻訳対象の単一テキスト |
| `type` | string | 翻訳ルール切替用タイプ |

**Output**

| 項目 | 型 | 説明 |
|---|---|---|
| `translated` | string | 翻訳済み文字列 |

### 4-2. バッチ翻訳

複数テキストをまとめて受け取り、バッチで翻訳します。  
Web 小説本文のような長文・大量データを扱うケースを想定しています。

**Endpoint**

```text
POST /api/v1/translate/batch
```

**Input**

| 項目 | 型 | 説明 |
|---|---|---|
| `texts` | string[] | 翻訳対象文字列配列 |
| `type` | string | 翻訳ルール切替用タイプ |

**Output**

| 項目 | 型 | 説明 |
|---|---|---|
| `translated` | string[] | 入力順を維持した翻訳結果配列 |

**内部処理の特徴**

- 一定件数ごとに分割して処理
- Chunk 単位の翻訳失敗時は再分割して再試行
- 最終的には 1 文単位での再試行まで実施
- 一部失敗が起きても全体欠損を減らす方向で設計

### 4-3. Speech-to-Text (STT)

アップロードされた音声ファイルを受け取り、文字列へ変換します。

**Endpoint**

```text
POST /api/v1/stt/transcribe
```

**Input**

| 項目 | 型 | 説明 |
|---|---|---|
| `file` | UploadFile | 音声ファイル |

**Output**

| 項目 | 型 | 説明 |
|---|---|---|
| `text` | string | 認識された文字列 |

**内部処理の特徴**

- 受信したファイルを一時領域へ保存
- WhisperModel を利用して文字起こし
- 処理後に一時ファイルを削除

---

## 5. 技術スタック

| 区分 | 採用技術 |
|---|---|
| Language | Python |
| Framework | FastAPI |
| ASGI Server | Uvicorn |
| Validation / Settings | Pydantic, pydantic-settings |
| Translation Model | Google Gemini |
| STT | faster-whisper |

---

## 6. 実装上の見どころ

### 6-1. `type` による翻訳ルール切り替え

翻訳 API は、単純にテキストを送るだけではなく、`type` によって内部プロンプトを切り替える構成です。  
これにより、用途別に翻訳方針を変えられる柔軟性を持たせています。

### 6-2. バッチ翻訳が単純な一括処理ではない

大量データをそのまま一度に投げるのではなく、Chunk 分割と再試行を組み合わせることで、部分失敗に対して比較的強い設計になっています。

### 6-3. 依存サービスをシングルトン的に再利用

`GeminiService` と `STTService` は依存性注入で再利用される形になっており、リクエストごとに重い初期化を避ける構成です。

### 6-4. STT は CPU ベースで扱いやすい構成

`faster-whisper` の base モデルを CPU / int8 ベースで利用しているため、GPU 必須ではありません。  
その分、実行環境の CPU 性能に影響されやすい点は考慮が必要です。

---

## 7. セキュリティ

本サーバは API Key ミドルウェアにより保護されています。  
対象リクエストには以下のヘッダが必要です。

```http
X-API-KEY: <your-server-api-key>
```

認証対象外パス:

- `/`
- `/docs`
- `/redoc`
- `/openapi.json`

それ以外のエンドポイントに対して API Key が一致しない場合、`401` を返します。

> 本サーバは内部通信用を前提としているため、インターネットへ直接公開する場合は追加のセキュリティ対策が必要です。

---

## 8. API 一覧

| Method | Path | 概要 |
|---|---|---|
| GET | `/` | ルート / 生存確認 |
| POST | `/api/v1/translate/single` | 単文翻訳 |
| POST | `/api/v1/translate/batch` | バッチ翻訳 |
| POST | `/api/v1/stt/transcribe` | 音声文字起こし |

---

## 9. 環境変数

本プロジェクトは `.env` ファイルから設定を読み込みます。

| 項目 | 用途 |
|---|---|
| `SERVER_API_KEY` | 内部 API 認証用キー |
| `GOOGLE_API_KEY` | Gemini 利用 API Key |
| `GEMINI_MODEL_NAME` | 使用する Gemini モデル名 |

### `.env` 例

```env
SERVER_API_KEY=your-internal-api-key
GOOGLE_API_KEY=your-google-api-key
GEMINI_MODEL_NAME=gemini-2.5-flash
```

---

## 10. ローカル実行方法

### 10-1. 仮想環境作成

**Windows**

```bash
python -m venv .venv
```

**macOS / Linux**

```bash
python3 -m venv .venv
```

### 10-2. 仮想環境有効化

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 10-3. 依存ライブラリインストール

```bash
pip install -r requirements.txt
```

### 10-4. サーバ起動

```bash
uvicorn app.main:app --reload
```

必要に応じてホスト / ポート指定:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 11. Swagger / OpenAPI

ローカル起動後、以下で API ドキュメントを確認できます。

```text
Swagger UI: http://localhost:8000/docs
ReDoc:      http://localhost:8000/redoc
```

---

## 12. ディレクトリ構成

```text
app
├─ api
│  ├─ dependencies.py
│  └─ v1
│     ├─ stt.py
│     └─ translate.py
├─ core
│  ├─ config.py
│  ├─ config_logger.py
│  ├─ constants.py
│  ├─ openapi.py
│  ├─ prompts.py
│  └─ utils.py
├─ schemas
│  └─ translation.py
├─ services
│  ├─ gemini_service.py
│  └─ stt_service.py
└─ main.py
```

---

## 13. 開発時の注意点

- `SERVER_API_KEY` が未設定だと、認証必須 API を正常に利用できません。
- `GOOGLE_API_KEY` が未設定だと、翻訳機能は動作しません。
- `GEMINI_MODEL_NAME` は利用可能なモデル名を設定してください。
- `faster-whisper` の初期ロードにより、初回処理時は応答が遅くなる場合があります。
- 長尺音声や大規模バッチ翻訳では CPU / メモリ負荷に注意が必要です。

---

## 14. 今後 README に追加するとよい内容

- 翻訳 `type` 一覧と用途説明
- Request / Response サンプル JSON
- Docker 起動手順
- デプロイ構成
- エラーコードポリシー
- 性能試験時の推奨設定

---

## 15. まとめ

TranslaCat AI Server は、翻訳と音声認識を担当する **内部向け AI 処理サーバ** です。  
Backend から切り離すことで責務を明確にし、AI モデル依存処理を独立して改善しやすい構成を実現しています。
