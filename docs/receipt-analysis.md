# Receipt analysis contract and operating notes

The multipart endpoint remains `POST /api/v1/account-book/receipts/analyze`.
A request contains one image (`file`) and JSON `options`. Each physical receipt
inside that image becomes a separate candidate. Deploy the matching BE/FE batch
contracts together; the old single-object response is no longer emitted.

## Response

```json
{
  "receipts": [{
    "receipt_id": "receipt-1",
    "title": "Cafe",
    "store_name": "Cafe",
    "original_amount": "12.34",
    "detected_currency_code": "USD",
    "transaction_date": "2026-09-15",
    "category_name": null,
    "memo": "Coffee",
    "confidence": 0.94,
    "currency_confidence": 0.98,
    "detected_language": "en",
    "status": "READY",
    "warnings": []
  }],
  "receipt_count": 1,
  "warnings": [],
  "ocr_engine": "vision",
  "used_ai": true
}
```

Amounts are Decimal values serialized as JSON **strings**, never binary floats.
The AI contract has no exchange-rate, converted-amount or raw-OCR-text fields.
The BE owns conversion, date policy, provider selection and persistence.
`receipt_id` is stable within the returned batch and independent of merchant names.
The maximum is 30 receipts per image, matching BE analysis/registration limits;
`RECEIPT_LIMIT_REACHED` explicitly reports truncation so the remaining receipts
can be uploaded in another image. The model output budget is 16384 tokens.

Text limits match batch registration: title/store 100, category 50, memo 500.
Unknown currency/date/amount remains null. Malformed objects produce individual
`UNREADABLE` candidates instead of invalidating other receipts. Confidence outside
0..1 is rejected rather than clamped/promoted. Amounts must be finite, positive,
within supported precision and compatible with known source minor units.

## Source detection and categories

`category_candidates` contains actual account-book category names. Selection is
exact and language-independent; missing/unmatched categories remain null for
manual resolution. `currency_code` is accepted from older clients but ignored.
It is never sent to a provider or used to choose an OCR language.

ISO currency validation uses an offline 166-code catalog from [SIX List One](https://www.six-group.com/dam/download/financial-information/data-center/iso-currrency/lists/list-one.xml),
retrieved 2026-09-19, plus BGN for historic receipts. Nonmonetary/no-currency/test
codes are excluded. The catalog includes 0/2/3/4 minor units, XCG, ZWG and XAD.
Update the snapshot for future ISO additions; provider exchange support is a
separate concern and does not depend on enabled account-book Currency rows.

A bare `$` or yen symbol cannot establish currency. Low/invalid currency confidence
leaves currency null. Ambiguous numeric dates require source date-order evidence;
the target account-book locale never supplies it. Original printed dates are
checked before accepting normalized ISO dates. Named months in different scripts
can retain the model's valid ISO date; the model still owns visual interpretation.

## Modes and fallback

- `VISION_FIRST` is the default. Vision receives the whole image and a receipts
  array schema. A partial or low-confidence result is preserved for review.
  Provider/envelope failure or no detected receipts invokes OCR plus AI.
- `VISION_ONLY` returns explicit diagnostics when vision cannot produce a batch.
- `OCR_WITH_AI` supplies every OCR line with its position/confidence to the model.
  PaddleOCR v2 and v3 results retain bounding boxes and duplicate text at distinct
  positions. No keyword truncates text after the first receipt.
- `OCR_ONLY` and OCR-AI failure return a conservative, **unconfirmed** manual-review
  draft. Only a unique explicit total can yield an amount; multiple/unknown totals
  yield null. This fallback does not claim reliable physical receipt segmentation.
  OCR-AI output lacking spatial context also cannot confirm an amount.

The OCR fallback language is an explicit source hint or configured `OCR_LANGUAGE`
(default `en`), independent of the target currency. A single Paddle language may
miss scripts in mixed-language images; `OCR_LANGUAGE_HINT_LIMITATION` exposes this.
Vision is the multilingual primary path. OCR resize/detection defaults allow a
2400-pixel longest edge (5.76 megapixels) to retain multi-receipt detail.
Both vision/OCR enforce the same 5 MB JPEG/PNG/WebP upload policy.

Identical validated payments at the same detected physical position are removed
with `DUPLICATE_RECEIPT_REMOVED`. Identical payments at different positions are
preserved. Without geometric proof, both candidates remain with duplicate warnings.
No custom computer-vision segmentation framework is introduced.

Gemini schema sanitization now preserves literal property names such as `title`;
only schema metadata is stripped. Both existing provider adapters are tested.

## Privacy and verification

Logs contain mode, count/status counts, currency codes, fallback path and latency.
Provider/OCR errors log exception type only. Images, raw OCR, payment identifiers,
merchant/memo text and provider response bodies are not logged by this flow.
No real AI/provider call is needed for tests. Visual accuracy on actual global
receipt photos still needs evaluation with deployment credentials and a consented
image corpus; the mocked suite verifies contracts, parsing and control flow.

On the provided Windows machine the original venv interpreter references a missing
base executable. Verification used bundled Python 3.12 with the existing venv's
site-packages appended to `sys.path` (preserving bundled working Pillow). Installed
library ACLs required read access outside the sandbox. A fresh workspace
`--basetemp` avoids an unrelated machine temp-directory ACL failure.

Equivalent normal-environment verification commands:

```text
python -m pytest -q -p no:cacheprovider --basetemp=test-results/temp-final --junitxml=test-results/pytest-complete.xml
ruff check .
pyright
```

The focused receipt suites cover multi/single/mixed receipts, Decimal formats,
ambiguous symbols/dates, unreadable/malformed/duplicate items, confidence bounds,
category allowlists, no FX fields, vision success/fallback, spatial OCR, privacy,
upload/API serialization, provider schema adaptation and BE text/batch limits.
Final results on 2026-09-19:

| Check | Result |
| --- | --- |
| Full `pytest -q` | 1746 passed, 0 failed/errors, 6 subtests passed, 21.49 seconds |
| Receipt tests within full suite | 85 analysis + 7 API/provider-adapter tests passed |
| Runtime warnings | 1 existing RequestsDependencyWarning for installed package versions |
| Ruff, all modified Python files | All checks passed |
| Ruff, whole repository | 9 preexisting violations (2 F401, 7 E402) |
| Pyright, all modified files + receipt endpoint/tests | 0 errors, 0 warnings |
| Pyright, whole repository | 649 errors, 0 warnings; untouched HEAD baseline 650 errors |
| `git diff --check` | Passed |

JUnit: `test-results/pytest-release.xml`. Full type diagnostics:
`test-results/pyright-full.json`; untouched HEAD comparison:
`test-results/pyright-baseline.json`. The temporary baseline source copy was
removed after comparison, so later test runs cannot collect it accidentally.

Full-repository lint/type debt was checked against unchanged source rather than
hidden: Ruff has 9 preexisting F401/E402 violations outside receipt code. Pyright
has 649 existing errors; the untouched HEAD snapshot had 650, including the old
Pillow `Image.LANCZOS` access fixed here. All touched files pass focused Ruff and
Pyright checks. Detailed JUnit/type reports are retained in ignored `test-results/`.
