# chuk-llm Roadmap

Based on the [engineering review](review.md) conducted 2026-04-23.

---

## Immediate — Fix Before Next Release

### 1. Implement or remove OpenRouter client
`chuk_llm.yaml` references `openrouter_client:OpenRouterLLMClient` which does not exist. Any attempt to use the OpenRouter provider causes a runtime `ImportError`.
- **Option A:** Implement a minimal `openrouter_client.py` (OpenRouter is OpenAI-compatible, so it can subclass `OpenAILLMClient` like the Advantage client does)
- **Option B:** Remove the OpenRouter entry from `chuk_llm.yaml` until it is implemented

### 2. Unify tool sanitization
Two separate `_sanitize_tool_names` implementations exist:
- `_mixins.py:35` — classmethod, returns dicts
- `_tool_compatibility.py:632` — instance method, returns Pydantic models

Different providers call different ones, causing type mismatch bugs. Merge into a single canonical implementation used by all providers.

### 3. Fix Gemini warning suppression
`gemini_client.py:71-130` monkey-patches Python's global warning system (`warnings.warn`, `showwarning`, `formatwarning`). Replace with `warnings.filterwarnings()` scoped to Gemini imports only.

### 4. Fix broad exception swallowing
Replace bare `except Exception: pass` blocks with at minimum a `logger.debug()` or `logger.warning()` call. `anthropic_client.py` alone has 15+ silent catches. This masks real bugs.

---

## Short-term — Next Major Version

### 5. Make provider SDKs optional dependencies
Move all 16 provider SDKs from `dependencies` to `optional-dependencies` extras in `pyproject.toml`. Users should install only what they need:
```
pip install chuk-llm[openai]
pip install chuk-llm[openai,anthropic]
pip install chuk-llm[all]
```

### 6. Fix public API wildcard import
`api/__init__.py:54` uses `from .providers import *`. Replace with an explicit `__all__` list. The current approach exports 180+ items with no clear core vs. advanced boundary.

### 7. Extract Granite formatting from WatsonX client
`watsonx_client.py` is 1,759 lines. The Granite chat template handling (lines 68–250+) and format parsing methods should be extracted into `watsonx_granite_utils.py`. Target: reduce WatsonX client to ~800 lines.

### 8. Consolidate duplicate logic
- **Image downloading** — use `_mixins.py:_download_and_encode_image()` everywhere; remove per-provider reimplementations
- **Message format conversion** — reconcile `base.py:_ensure_pydantic_messages()` and `core.py:_convert_dict_to_pydantic_messages()` into one canonical function
- **Tool call extraction** — define a shared interface; provider-specific parsing should implement it rather than being freestanding

### 9. Fix async/sync event loop handling
`api/providers.py:81-108` creates new event loops which can deadlock in async contexts. Use `event_loop_manager.py` consistently across all sync wrapper paths.

### 10. Add integration tests for tool calling and streaming
- Test tool calling for every provider using mocked API responses
- Test streaming chunk assembly and tool call deduplication
- Test concurrent client creation
- Remove `--disable-warnings` from `pytest.ini`

---

## Medium-term — Quality & Maintainability

### 11. Unified parameter validation framework
Currently each provider handles unsupported parameters differently (OpenAI maps them, Gemini relies on API errors, Mistral filters conservatively). Define a shared validation layer in the base class.

### 12. Add concrete response types
Provider clients currently return `AsyncIterator[dict[str, Any]]`. Define typed response dataclasses/Pydantic models and use them throughout.

### 13. Add pre-flight configuration validation
`ProviderConfig.client_class` defaults to empty string and fails silently. Add startup validation that checks required fields are set and that API key env vars exist before the first API call is made.

### 14. Add streaming tool call deduplication framework
Groq and Mistral each implement their own deduplication logic; OpenAI and Anthropic handle it differently. Unify into a shared streaming utility.

### 15. Consistent logging
Standardise on a module-level logger per file; make log level configurable per provider. Remove hardcoded logger suppression in `__init__.py:75-76`.

---

## Nice-to-have — Future Consideration

- **Plugin system** for registering custom provider clients without modifying the library
- **Provider health checks** — periodic validation that API keys are working
- **Cost tracking** — integrate per-provider pricing data into responses
- **Load balancing** — round-robin or least-latency routing across multiple API keys for the same provider
- **Configurable cache TTL** — currently hardcoded to 300s in `unified_config.py:71`
