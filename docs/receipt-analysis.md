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
    "purchase_total": "12.34",
    "payment_breakdown": [{
      "payment_type": "CREDIT_CARD",
      "amount": "12.34",
      "evidence": "VISA 12.34",
      "duplicate_group": "card-1"
    }],
    "cash_tendered": null,
    "change": null,
    "book_amount": "12.34",
    "amount_policy_version": "receipt-book-amount-v1",
    "amount_reason": "SETTLED_PAYMENT_EXCLUDING_LOYALTY_POINTS",
    "review_status": "READY",
    "original_amount": "12.34",
    "detected_currency_code": "USD",
    "transaction_date": "2026-09-15",
    "category_name": "식비",
    "category_source": "DEFAULT",
    "category_reason": "Prepared food purchase",
    "memo": "Coffee",
    "confidence": 0.97,
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
Validated candidates require confidence >= 0.95 for `READY`; lower-confidence text
and merchant readings remain editable `NEEDS_REVIEW` candidates. Amount-policy
warnings and missing or ambiguous fields can force review independently.

## Source detection and categories

`category_candidates` contains actual account-book category names and
`default_category_candidates` contains the session-stable application defaults.
The model reuses an appropriate existing category first, otherwise chooses a useful
default, proposes one concise new category when both lists are inadequate, or uses a broad
fallback such as 기타 when evidence is weak. `category_source` records EXISTING, DEFAULT,
NEW or FALLBACK and `category_reason` keeps a short basis for the suggestion. Validation
canonicalizes exact existing/default matches, preserves safe new names, and replaces empty,
overlong or control-character suggestions with the stable fallback. New/fallback status is
advisory and does not hide unrelated amount/date/currency review errors. Merchant, branch and
memo text remains in the printed source language. `currency_code` is accepted from older clients but ignored.
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

- `VISION_ONLY` is the default receipt path. Vision receives the whole image and a
  receipts array schema containing normalized source regions. Partial and
  low-confidence candidates are preserved for review; provider failure, timeout or
  an empty response does not invoke OCR.
- `VISION_FIRST` remains an explicit legacy option. It preserves partial Vision
  results and uses OCR plus AI only after a whole-image provider/envelope failure or
  an empty result.
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

## Vision scheduling and selective recovery

Each photo starts with one whole-image provider call. Candidates with complete
identity and financial facts are not called again. An incomplete candidate with a
valid normalized source box can receive one lossless PNG crop reread. Identity and
financial recovery share that one crop when both are missing. Crop decode, EXIF
normalization and encoding run outside the async event loop, reuse one decoded
image, enforce the configured pixel limit and preserve extra pixels below the
receipt for points, card detail and change.

Whole-image and crop calls share one process-wide scheduler. Defaults are three
total in-flight calls, at most two concurrent recoveries and twelve pending calls.
The permit is acquired only around the actual provider call, so a parent request
does not hold capacity while waiting for its crop tasks. One request has one
30-second monotonic deadline from upload handling through all automatic recovery;
new crops are skipped when less than three seconds remain. A failed or timed-out
crop leaves other candidates intact and marks only that source candidate for
review. OpenAI SDK retries are disabled (`max_retries=0`).

The repository Docker image and isolated live launcher use one Uvicorn worker, so
the process limit is also the server limit in that topology. A future multi-worker
deployment must divide this value or supply an external coordinator; the current
semaphore is not represented as a cross-process limit. OCR stays available on
demand, while receipt Vision startup suppresses Paddle warm-up.

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

`scripts/evaluate_receipt_images.py` is the bounded live evaluator. It requires a
manifest whose real images are consent-approved, de-identified and independently
reviewed, verifies every file hash, preserves bounding boxes for one-to-one physical
matching, limits concurrency to 1 or 2 and limits a run to 60 images. It uses the
configured `RECEIPT_ANALYSIS` provider/model and production prompt/schema in
`VISION_ONLY` mode. Example:

```text
python scripts/evaluate_receipt_images.py path/to/manifest.json \
  --output test-results/receipt-live.json --split core --max-images 30 --concurrency 1
```

For the 2026-09-21 release-candidate run, the user explicitly authorized external
analysis of safe derivatives of their receipts. The configured OpenAI
`gpt-5.6-luna` runtime analyzed the Papasu receipt and three un-cropped photos with
4/4/3 physical receipts. Ten independently sourced public scans covered IDR, TRY,
and MYR under recorded licenses and hashes. The cumulative ledger used 36 of a
40-call ceiling and was never reset. Live output, frozen manifests, manual pixel
adjudication, and the browser-to-MySQL proof are under
`../quality-evidence/2026-09-21-rc-sol`. Mocked tests are reported separately.

The focused receipt suites cover multi/single/mixed receipts, Decimal formats,
ambiguous symbols/dates, unreadable/malformed/duplicate items, confidence bounds,
category allowlists, no FX fields, vision success/fallback, spatial OCR, privacy,
upload/API serialization, provider schema adaptation and model-policy selection.
The exact final test commands, counts, baseline static-analysis debt, and exit codes
are retained in the release-candidate evidence report rather than copied into this
contract document.
