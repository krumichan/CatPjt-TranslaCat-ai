# app refactor notes

This refactor separates feature-level logic from provider-level AI integration.

## Main structure

```text
app/
├─ api/                 FastAPI routers and dependencies
├─ ai/                  AI provider abstraction and provider implementations
│  ├─ ports.py          Provider protocols
│  ├─ provider_factory.py
│  └─ providers/gemini/ Gemini-specific client/config
├─ common/              Generic utilities
├─ core/                App-wide settings, logging, OpenAPI, constants
├─ features/            Feature-specific prompts/services/parsers
│  ├─ chat_translation/
│  ├─ receipt/
│  └─ translation/
├─ schemas/             Request/response schemas
└─ services/            Compatibility wrappers and non-provider heavy services
```

## Why this shape

- Gemini-specific config and client code is isolated under `app/ai/providers/gemini`.
- Feature rules and prompts are isolated under `app/features/*`.
- Existing imports such as `app.services.gemini_service` and `app.core.prompts` remain available as compatibility wrappers.
- Future providers can be added under `app/ai/providers/{provider_name}` and selected from `app/ai/provider_factory.py`.
```
