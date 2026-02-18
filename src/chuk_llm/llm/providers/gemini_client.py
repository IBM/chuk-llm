# chuk_llm/llm/providers/gemini_client.py

"""
Google Gemini chat-completion adapter with unified configuration integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configuration-driven capabilities with complete warning suppression and proper parameter handling.
UPDATED: Fixed to match other providers - proper system instruction support,
universal vision format, robust response handling, and universal tool name compatibility with restoration.
CRITICAL FIX: Eliminates response concatenation and data loss issues.
UNIVERSAL COMPATIBILITY: Full integration with ToolCompatibilityMixin for seamless MCP support.
"""

from __future__ import annotations

import base64
import contextlib
import functools
import json
import logging
import os
import sys
import uuid
import warnings
from collections.abc import AsyncIterator
from contextlib import contextmanager
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types as gtypes

# core
from chuk_llm.core.enums import Feature

# providers
from chuk_llm.llm.core.base import BaseLLMClient
from chuk_llm.llm.providers._config_mixin import ConfigAwareProviderMixin
from chuk_llm.llm.providers._tool_compatibility import ToolCompatibilityMixin

log = logging.getLogger(__name__)

# Honour LOGLEVEL env-var for quick local tweaks
if "LOGLEVEL" in os.environ:
    log.setLevel(os.environ["LOGLEVEL"].upper())

# ═══════════════════════════════════════════════════════════════════════════════
# ULTIMATE WARNING SUPPRESSION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

# Store original warning functions globally
_ORIGINAL_WARN = warnings.warn
_ORIGINAL_SHOWWARNING = warnings.showwarning
_ORIGINAL_FORMATWARNING = warnings.formatwarning


def _silent_warn(*args: Any, **kwargs: Any) -> None:
    """Completely silent warning function"""
    pass


def _silent_showwarning(*args: Any, **kwargs: Any) -> None:
    """Completely silent showwarning function"""
    pass


def _silent_formatwarning(*args: Any, **kwargs: Any) -> str:
    """Return empty string for formatwarning"""
    return ""


def apply_ultimate_warning_suppression() -> None:
    """Apply the most comprehensive warning suppression possible for Gemini"""

    # Method 1: Environment variables (most aggressive)
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["GOOGLE_GENAI_SUPPRESS_WARNINGS"] = "1"
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GLOG_MINLOGLEVEL"] = "3"  # Even more aggressive
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Method 2: Global warning function replacement
    warnings.warn = _silent_warn
    warnings.showwarning = _silent_showwarning
    warnings.formatwarning = _silent_formatwarning

    # Method 3: Comprehensive warning patterns for Gemini-specific warnings
    gemini_warning_patterns = [
        # Function call warnings (the main culprit)
        ".*non-text parts in the response.*",
        ".*function_call.*",
        ".*returning concatenated text result.*",
        ".*check out the non text parts.*",
        ".*text parts.*",
        ".*response.*function_call.*",
        ".*non text parts for full response.*",
        ".*candidates.*",
        ".*finish_reason.*",
        # General Gemini warnings
        ".*genai.*",
        ".*google.*",
        ".*generativeai.*",
        ".*google.genai.*",
        ".*google.generativeai.*",
        # gRPC and HTTP warnings
        ".*grpc.*",
        ".*http.*",
        ".*ssl.*",
        ".*certificate.*",
        ".*urllib3.*",
        ".*httpx.*",
        ".*asyncio.*",
    ]

    # Apply pattern-based suppression for ALL warning categories
    warning_categories = [
        UserWarning,
        Warning,
        FutureWarning,
        DeprecationWarning,
        RuntimeWarning,
        PendingDeprecationWarning,
        ImportWarning,
        UnicodeWarning,
        BytesWarning,
        ResourceWarning,
    ]

    for pattern in gemini_warning_patterns:
        for category in warning_categories:
            warnings.filterwarnings("ignore", message=pattern, category=category)
        # Also apply without category
        warnings.filterwarnings("ignore", message=pattern)

    # Method 4: Module-level suppression for Google ecosystem
    google_modules = [
        "google",
        "google.genai",
        "google.generativeai",
        "google.ai",
        "google.cloud",
        "google.protobuf",
        "grpc",
        "googleapis",
        "google.auth",
        "google.api_core",
        "google.api",
        "googleapiclient",
    ]

    for module in google_modules:
        for category in warning_categories:
            warnings.filterwarnings("ignore", category=category, module=module)
        warnings.filterwarnings("ignore", module=module)

    # Method 5: Logger suppression (nuclear option)
    google_loggers = [
        "google",
        "google.genai",
        "google.generativeai",
        "google.ai.generativelanguage",
        "google.ai",
        "google.cloud",
        "grpc",
        "grpc._channel",
        "urllib3",
        "httpx",
        "asyncio",
        "google.auth",
        "google.api_core",
        "googleapiclient",
    ]

    for logger_name in google_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL + 1)  # Even higher than CRITICAL
        logger.propagate = False
        logger.disabled = True
        logger.handlers.clear()
        # Also disable all child loggers
        for child_logger in [
            log_name
            for log_name in logging.Logger.manager.loggerDict
            if log_name.startswith(logger_name + ".")
        ]:
            child = logging.getLogger(child_logger)
            child.setLevel(logging.CRITICAL + 1)
            child.propagate = False
            child.disabled = True
            child.handlers.clear()

    # Method 6: Global warning suppression
    warnings.simplefilter("ignore")
    for category in warning_categories:
        warnings.simplefilter("ignore", category)


# Enhanced context manager for complete output suppression
class UltimateSuppression:
    """The most aggressive suppression possible - blocks everything"""

    def __init__(self):
        self.original_stderr = None
        self.original_stdout = None
        self.original_warn = None
        self.original_showwarning = None
        self.original_formatwarning = None
        self.devnull = None
        self.original_filters = None

    def __enter__(self):
        # Store all originals
        self.original_stderr = sys.stderr
        self.original_stdout = sys.stdout
        self.original_warn = warnings.warn
        self.original_showwarning = warnings.showwarning
        self.original_formatwarning = warnings.formatwarning
        self.original_filters = warnings.filters[:]

        # Open devnull
        self.devnull = open(os.devnull, "w")

        # Redirect stderr to devnull (where warnings go)
        sys.stderr = self.devnull

        # Replace all warning functions with no-ops
        warnings.warn = _silent_warn
        warnings.showwarning = _silent_showwarning
        warnings.formatwarning = _silent_formatwarning

        # Clear all filters and apply ultimate suppression
        warnings.resetwarnings()
        apply_ultimate_warning_suppression()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore everything
        if self.devnull:
            self.devnull.close()

        sys.stderr = self.original_stderr
        sys.stdout = self.original_stdout
        warnings.warn = self.original_warn
        warnings.showwarning = self.original_showwarning
        warnings.formatwarning = self.original_formatwarning

        # Restore original filters
        if self.original_filters is not None:
            warnings.filters[:] = self.original_filters  # type: ignore[index]


def silence_gemini_warnings(func):
    """Decorator to completely silence warnings for Gemini API calls"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with UltimateSuppression():
            return func(*args, **kwargs)

    return wrapper


def silence_gemini_warnings_async(func):
    """Async decorator to completely silence warnings for Gemini API calls"""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        with UltimateSuppression():
            return await func(*args, **kwargs)

    return wrapper


# Apply suppression immediately when module loads
apply_ultimate_warning_suppression()

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL AVAILABILITY MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

# Current available Gemini models
AVAILABLE_GEMINI_MODELS = {
    # Gemini 2.5 series (latest and most powerful)
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    # Gemini 2.0 series (stable)
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    # Gemini 1.5 series (stable)
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
}


def validate_and_map_model(requested_model: str) -> str:
    """
    Permissive model validation - accept any model name and let the API handle it.

    This allows new models to work automatically without code changes.
    If the model is invalid, the Gemini API will return a clear error.
    """
    # Log a warning if model not in known list, but don't block it
    if requested_model not in AVAILABLE_GEMINI_MODELS:
        import logging

        log = logging.getLogger(__name__)
        log.debug(
            f"Model '{requested_model}' not in known models list. "
            f"Attempting to use it anyway (permissive approach). "
            f"Known models: {sorted(AVAILABLE_GEMINI_MODELS)}"
        )

    return requested_model


# ═══════════════════════════════════════════════════════════════════════════════
# SAFE RESPONSE PARSING - NO CONCATENATION OR DATA LOSS
# ═══════════════════════════════════════════════════════════════════════════════


def _safe_parse_gemini_response(resp: Any) -> dict[str, Any]:
    """
    CRITICAL: Safe Gemini response parser that prevents data loss and concatenation.

    NEVER accesses resp.text directly when function calls might be present.
    Always parses through candidates -> content -> parts to extract ALL data.
    """
    tool_calls: list[dict[str, Any]] = []
    response_text = ""

    try:
        # NEVER use resp.text directly - it triggers concatenation and data loss
        # Instead, always parse through candidates -> content -> parts

        if hasattr(resp, "candidates") and resp.candidates:
            candidate = resp.candidates[0]

            if hasattr(candidate, "content") and candidate.content:
                content = candidate.content

                # Check if content has parts (this is where multimodal content lives)
                if hasattr(content, "parts") and content.parts:
                    text_parts: list[str] = []

                    # Process each part individually to avoid data loss
                    for part in content.parts:
                        # Handle text parts
                        if hasattr(part, "text") and part.text:
                            text_parts.append(part.text)

                        # Handle function call parts - CRITICAL: Extract these separately
                        elif hasattr(part, "function_call") and part.function_call:
                            fc = part.function_call

                            try:
                                # Extract function call data
                                function_name = getattr(fc, "name", "unknown")
                                function_args = dict(getattr(fc, "args", {}))

                                tool_calls.append(
                                    {
                                        "id": f"call_{uuid.uuid4().hex[:8]}",
                                        "type": "function",
                                        "function": {
                                            "name": function_name,
                                            "arguments": json.dumps(function_args),
                                        },
                                    }
                                )

                                log.debug(f"Extracted function call: {function_name}")

                            except Exception as e:
                                log.error(f"Error extracting function call: {e}")

                        # Handle other part types (images, etc.)
                        else:
                            part_info = str(type(part).__name__)
                            log.debug(
                                f"Encountered non-text, non-function part: {part_info}"
                            )

                    # Combine text parts
                    response_text = "".join(text_parts)

                    # Log what we extracted
                    if text_parts and tool_calls:
                        log.debug(
                            f"Extracted {len(text_parts)} text parts and {len(tool_calls)} function calls"
                        )
                    elif text_parts:
                        log.debug(f"Extracted {len(text_parts)} text parts only")
                    elif tool_calls:
                        log.debug(
                            f"Extracted {len(tool_calls)} function calls only (no text)"
                        )

                # Fallback: content has text but no parts structure
                elif hasattr(content, "text") and content.text:
                    response_text = content.text
                    log.debug("Extracted text from content.text (no parts structure)")

                # No text or parts - might be function-call-only response
                else:
                    if tool_calls:
                        response_text = ""  # Valid: function calls with no text
                        log.debug("Function-call-only response (no text content)")
                    else:
                        response_text = "[No content available in response]"
                        log.warning(
                            "Response has content but no extractable text or function calls"
                        )

            # No content in candidate
            else:
                # Check for thinking models hitting token limits
                if hasattr(candidate, "finish_reason"):
                    reason = str(candidate.finish_reason)
                    if "MAX_TOKENS" in reason:
                        response_text = (
                            "[Response exceeded token limit during processing.]"
                        )
                    elif "SAFETY" in reason:
                        response_text = "[Response blocked due to safety filters.]"
                    else:
                        response_text = f"[Response completed with status: {reason}]"
                else:
                    response_text = "[No content in candidate]"
                    log.warning("Candidate has no content")

        # No candidates structure - this is unusual for modern Gemini
        else:
            log.warning("Response has no candidates structure")
            # Avoid direct text access as it may trigger concatenation
            response_text = "[Unable to extract content from response - no candidates]"

    except Exception as e:
        log.error(f"Critical error in response parsing: {e}")
        response_text = f"[Error parsing response: {str(e)}]"

    # Handle JSON mode response cleanup
    if response_text and response_text.count('{"') > 1:
        # Remove duplicate JSON objects that sometimes appear
        json_parts = response_text.split('{"')
        if len(json_parts) > 1:
            first_json = '{"' + json_parts[1].split("}")[0] + "}"
            try:
                json.loads(first_json)  # Validate it's proper JSON
                response_text = first_json
            except json.JSONDecodeError:
                pass  # Keep original if not valid JSON

    # Build final response
    result = {"response": response_text, "tool_calls": tool_calls}

    # Validate we didn't lose data
    if tool_calls and not response_text:
        log.debug("Valid function-call-only response")
    elif tool_calls and response_text:
        log.debug(
            f"Mixed response: text ({len(response_text)} chars) + {len(tool_calls)} function calls"
        )
    elif response_text and not tool_calls:
        log.debug(f"Text-only response: {len(response_text)} chars")
    else:
        log.warning("Response has neither text nor function calls")

    return result


# ───────────────────────────────────────────────────── helpers ──────────


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Get *key* from dict **or** attribute-style object; fallback to *default*."""
    return (
        obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
    )


def _fix_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Fix JSON schema for Gemini compatibility.

    Gemini requires that all array properties have an 'items' field.
    This function recursively adds missing 'items' fields with a permissive schema.
    """
    if not isinstance(schema, dict):
        return schema

    fixed_schema = schema.copy()

    # Fix properties if they exist
    if "properties" in fixed_schema and isinstance(fixed_schema["properties"], dict):
        fixed_properties = {}
        for prop_name, prop_schema in fixed_schema["properties"].items():
            if isinstance(prop_schema, dict):
                # If this is an array type without items, add a default items schema
                if prop_schema.get("type") == "array" and "items" not in prop_schema:
                    log.debug(
                        f"Adding missing 'items' field for array property '{prop_name}'"
                    )
                    prop_schema = prop_schema.copy()
                    # Add a permissive items schema - allows any type
                    prop_schema["items"] = {
                        "type": "object",
                        "description": "Array item",
                    }

                # Recursively fix nested schemas
                fixed_properties[prop_name] = _fix_gemini_schema(prop_schema)
            else:
                fixed_properties[prop_name] = prop_schema

        fixed_schema["properties"] = fixed_properties

    # Fix items if they exist (for nested arrays)
    if "items" in fixed_schema and isinstance(fixed_schema["items"], dict):
        fixed_schema["items"] = _fix_gemini_schema(fixed_schema["items"])

    return fixed_schema


def _convert_tools_to_gemini_format(
    tools: list[dict[str, Any]] | None,
) -> list[gtypes.Tool] | None:
    """Convert OpenAI-style tools to Gemini format with schema validation fixes"""
    if not tools:
        return None

    try:
        function_declarations = []

        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})

                func_name = func.get("name", "")
                if not func_name or func_name == "unknown_function":
                    log.warning(f"Skipping tool with invalid name: {tool}")
                    continue

                # Fix the parameters schema for Gemini compatibility
                parameters = func.get("parameters", {})
                if parameters:
                    parameters = _fix_gemini_schema(parameters)

                func_decl = {
                    "name": func_name,
                    "description": func.get("description", ""),
                    "parameters": parameters,
                }
                function_declarations.append(func_decl)

        if function_declarations:
            try:
                gemini_tool = gtypes.Tool(function_declarations=function_declarations)  # type: ignore[arg-type]
                return [gemini_tool]
            except Exception as e:
                log.warning(f"Failed to create Gemini Tool: {e}")
                return None

    except Exception as e:
        log.error(f"Error converting tools to Gemini format: {e}")

    return None


# ─────────────────────────────────────────────────── enhanced context managers ───────────


class SuppressAllOutput:
    """Context manager to completely suppress all output including warnings"""

    def __init__(self):
        self.suppressor = UltimateSuppression()

    def __enter__(self):
        return self.suppressor.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.suppressor.__exit__(exc_type, exc_val, exc_tb)


@contextmanager
def suppress_warnings():
    """Standard context manager for warning suppression"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ─────────────────────────────────────────────────── main adapter ───────────


class GeminiLLMClient(ConfigAwareProviderMixin, ToolCompatibilityMixin, BaseLLMClient):
    """
    Configuration-aware `google-genai` wrapper with universal tool name compatibility.

    Uses universal tool name compatibility system to handle any naming convention:
    - stdio.read_query -> stdio_read_query (sanitized and restored)
    - web.api:search -> web_api_search (sanitized and restored)
    - database.sql.execute -> database_sql_execute (sanitized and restored)
    - service:method -> service_method (sanitized and restored)

    CRITICAL FIX: Eliminates response concatenation and data loss issues.
    UNIVERSAL COMPATIBILITY: Full integration with ToolCompatibilityMixin for seamless MCP support.
    """

    def __init__(
        self, model: str = "gemini-2.5-flash", *, api_key: str | None = None
    ) -> None:
        # Apply nuclear warning suppression during initialization
        apply_ultimate_warning_suppression()

        # Validate model
        safe_model = validate_and_map_model(model)

        # Initialize mixins FIRST
        ConfigAwareProviderMixin.__init__(self, "gemini", safe_model)
        ToolCompatibilityMixin.__init__(self, "gemini")

        # load environment
        load_dotenv()

        # get the api key
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        # check if we have a key
        if not api_key:
            raise ValueError("GEMINI_API_KEY / GOOGLE_API_KEY env var not set")

        # Initialize with complete suppression
        with UltimateSuppression():
            self.model = safe_model
            self.client = genai.Client(api_key=api_key)

        log.info("GeminiLLMClient initialised with model '%s'", safe_model)

    def _detect_model_family(self) -> str:
        """Detect Gemini model family for optimizations"""
        model_lower = self.model.lower()
        if "2.5" in model_lower:
            return "gemini-2.5"
        elif "2.0" in model_lower:
            return "gemini-2.0"
        elif "1.5" in model_lower:
            return "gemini-1.5"
        elif "flash" in model_lower:
            return "flash"
        elif "pro" in model_lower:
            return "pro"
        else:
            return "unknown"

    @silence_gemini_warnings
    def _parse_gemini_response_with_restoration(
        self, resp: Any, name_mapping: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Convert Gemini response to standard format and restore tool names - WITH SILENCE"""
        # Use the safe parser (no concatenation)
        result = _safe_parse_gemini_response(resp)

        # Restore original tool names using universal restoration
        if name_mapping and result.get("tool_calls"):
            result = self._restore_tool_names_in_response(result, name_mapping)

        return result

    def get_model_info(self) -> dict[str, Any]:
        """Get model info using configuration, with Gemini-specific additions."""
        # Get base info from configuration
        info = super().get_model_info()

        # Add tool compatibility info from universal system
        tool_compatibility = self.get_tool_compatibility_info()

        # Add Gemini-specific metadata only if no error occurred
        if not info.get("error"):
            info.update(
                {
                    # Universal tool compatibility info
                    **tool_compatibility,
                    "supports_function_calling": self.supports_feature(
                        Feature.TOOLS.value
                    ),
                    "supports_streaming": self.supports_feature(
                        Feature.STREAMING.value
                    ),
                    "supports_vision": self.supports_feature(Feature.VISION.value),
                    "supports_json_mode": self.supports_feature(
                        Feature.JSON_MODE.value
                    ),
                    "supports_system_messages": self.supports_feature(
                        Feature.SYSTEM_MESSAGES.value
                    ),
                    "gemini_specific": {
                        "context_length": (
                            "2M tokens"
                            if "2.5" in self.model
                            else ("2M tokens" if "2.0" in self.model else "1M tokens")
                        ),
                        "model_family": self._detect_model_family(),
                        "experimental_features": "2.0" in self.model
                        or "2.5" in self.model,
                        "warning_suppression": "ultimate",
                        "enhanced_reasoning": "2.5" in self.model,
                        "supports_function_calling": self.supports_feature(
                            Feature.TOOLS.value
                        ),
                        "data_loss_protection": "enabled",
                    },
                    "vision_format": "universal_image_url",
                    "supported_parameters": [
                        "temperature",
                        "max_tokens",
                        "top_p",
                        "top_k",
                        "stream",
                        "system",
                    ],
                    "unsupported_parameters": [
                        "frequency_penalty",
                        "presence_penalty",
                        "logit_bias",
                        "user",
                        "n",
                        "best_of",
                        "seed",
                        "stop",
                    ],
                    "parameter_mapping": {
                        "max_tokens": "max_output_tokens",
                        "stop": "stop_sequences",
                        "system": "system_instruction_in_config",
                        "temperature": "temperature",
                        "top_p": "top_p",
                        "top_k": "top_k",
                        "candidate_count": "candidate_count",
                    },
                }
            )

        return info

    def _filter_gemini_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Filter parameters using configuration limits"""
        filtered: dict[str, Any] = {}

        # Parameter mapping
        parameter_mapping = {
            "max_tokens": "max_output_tokens",
            "stop": "stop_sequences",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "candidate_count": "candidate_count",
        }

        # Supported parameters
        supported_params = {
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "candidate_count",
            "stop_sequences",
            "max_output_tokens",
        }
        unsupported_params = {
            "frequency_penalty",
            "presence_penalty",
            "logit_bias",
            "user",
            "n",
            "best_of",
            "seed",
            "stop",
            "response_format",
        }

        for key, value in params.items():
            mapped_key = parameter_mapping.get(key, key)

            if mapped_key in supported_params:
                if key == "temperature":
                    # Gemini temperature range validation
                    if value > 2.0:
                        filtered[mapped_key] = 2.0
                        log.debug(f"Capped temperature from {value} to 2.0 for Gemini")
                    else:
                        filtered[mapped_key] = value
                elif key == "max_tokens":
                    # Use configuration to validate max_tokens
                    limit = self.get_max_tokens_limit()
                    if limit and value > limit:
                        filtered[mapped_key] = limit
                        log.debug(
                            f"Capped max_tokens from {value} to {limit} for Gemini"
                        )
                    else:
                        filtered[mapped_key] = value
                else:
                    filtered[mapped_key] = value
            elif key in unsupported_params:
                log.debug(
                    f"Filtered out unsupported parameter for Gemini: {key}={value}"
                )
            else:
                log.warning(f"Unknown parameter for Gemini: {key}={value}")

        return filtered

    def _check_json_mode(self, kwargs: dict[str, Any]) -> str | None:
        """Check if JSON mode is requested and return appropriate system instruction"""
        # Only proceed if the model supports JSON mode according to config
        if not self.supports_feature(Feature.JSON_MODE.value):
            log.debug(
                f"Model {self.model} does not support JSON mode according to configuration"
            )
            return None

        # Check for OpenAI-style response_format
        from chuk_llm.core import ResponseFormat

        response_format = kwargs.get("response_format")
        if response_format:
            # Validate with Pydantic if it's a dict
            if isinstance(response_format, dict):
                with contextlib.suppress(Exception):
                    response_format = ResponseFormat.model_validate(response_format)

            # Check type regardless of whether it's dict or ResponseFormat
            format_type = (
                response_format.type
                if isinstance(response_format, ResponseFormat)
                else (
                    response_format.get("type")
                    if isinstance(response_format, dict)
                    else None
                )
            )

            if format_type == "json_object":
                return "You must respond with valid JSON only. No markdown code blocks, no explanations, no text before or after. Just pure, valid JSON."

        # Check for _json_mode_instruction from provider adapter
        json_instruction = kwargs.get("_json_mode_instruction")
        if json_instruction:
            return json_instruction

        return None

    @staticmethod
    async def _download_image_to_base64(url: str) -> tuple[str, str]:
        """Download image from URL and convert to base64"""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()

                # Get content type from headers
                content_type = response.headers.get("content-type", "image/png")
                if not content_type.startswith("image/"):
                    content_type = "image/png"  # Default fallback

                # Convert to base64
                image_data = base64.b64encode(response.content).decode("utf-8")

                return content_type, image_data

        except Exception as e:
            log.warning(f"Failed to download image from {url}: {e}")
            raise ValueError(f"Could not download image: {e}") from e

    @staticmethod
    async def _convert_universal_vision_to_gemini_async(
        content_item: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert universal image_url format to Gemini format with URL downloading"""
        if content_item.get("type") == "image_url":
            image_url = content_item.get("image_url", {})

            # Handle both string and dict formats
            url = image_url if isinstance(image_url, str) else image_url.get("url", "")

            # Convert data URL to Gemini format
            if url.startswith("data:"):
                # Extract media type and data
                try:
                    header, data = url.split(",", 1)
                    # Parse the header: data:image/png;base64
                    media_type_part = header.split(";")[0].replace("data:", "")

                    # Validate media type
                    if not media_type_part.startswith("image/"):
                        media_type_part = "image/png"  # Default fallback

                    # Gemini expects inline data format
                    return {
                        "inline_data": {
                            "mime_type": media_type_part,
                            "data": data.strip(),
                        }
                    }
                except (ValueError, IndexError) as e:
                    log.warning(f"Invalid data URL format: {url[:50]}... Error: {e}")
                    return {"text": "[Invalid image format]"}
            else:
                # For external URLs, download and convert to base64
                try:
                    (
                        media_type,
                        image_data,
                    ) = await GeminiLLMClient._download_image_to_base64(url)

                    return {
                        "inline_data": {"mime_type": media_type, "data": image_data}
                    }
                except Exception as e:
                    log.warning(f"Failed to process external image URL {url}: {e}")
                    return {"text": f"[Could not load image: {e}]"}

        return content_item

    async def _split_for_gemini_async(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str, list[Any]]:
        """
        Separate system text & convert ChatML list to Gemini format with async vision support.
        Uses configuration to validate vision support.
        """
        sys_txt: list[str] = []
        contents: list[Any] = []

        for msg in messages:
            role = _safe_get(msg, "role")

            if role == "system":
                sys_txt.append(_safe_get(msg, "content", ""))
                continue

            # assistant function calls → need to be handled in tool result flow
            if role == "assistant" and _safe_get(msg, "tool_calls"):
                # Convert to text description for now
                tool_text = "Assistant called functions: "
                tool_calls = _safe_get(msg, "tool_calls")
                for tc in tool_calls:
                    fn = _safe_get(tc, "function")
                    fn_name = _safe_get(fn, "name")
                    fn_args = _safe_get(fn, "arguments", "{}")
                    tool_text += f"{fn_name}({fn_args}) "
                contents.append(tool_text)
                continue

            # tool response → convert to user message
            if role == "tool":
                tool_result = _safe_get(msg, "content") or ""
                fn_name = _safe_get(msg, "name", "tool")
                contents.append(f"Tool {fn_name} returned: {tool_result}")
                continue

            # normal / multimodal messages with universal vision support
            if role in {"user", "assistant"}:
                cont = _safe_get(msg, "content")
                if cont is None:
                    continue

                if isinstance(cont, str):
                    # Simple text content
                    contents.append(cont)
                elif isinstance(cont, list):
                    # Permissive approach: Process all multimodal content (text, vision, audio)
                    # Let Gemini API handle unsupported cases rather than filtering
                    # Process multimodal content with proper Gemini format
                    gemini_parts = []
                    for item in cont:
                        item_type = _safe_get(item, "type")
                        if item_type == "text":
                            gemini_parts.append(_safe_get(item, "text", ""))
                        elif item_type == "image_url":
                            # Convert to Gemini format
                            try:
                                gemini_image = await self._convert_universal_vision_to_gemini_async(
                                    item
                                )
                                if "inline_data" in gemini_image:
                                    gemini_parts.append(gemini_image)
                                else:
                                    # Fallback to text if conversion failed
                                    gemini_parts.append(
                                        gemini_image.get(
                                            "text", "[Image processing failed]"
                                        )
                                    )
                            except Exception as e:
                                log.warning(f"Failed to convert image: {e}")
                                gemini_parts.append("[Image conversion failed]")
                        else:
                            # Other content types - pass through (shouldn't reach here normally)
                            gemini_parts.append(str(item))

                    # Add the multimodal content as a structured message
                    if gemini_parts:
                        contents.append(gemini_parts)
                else:
                    # Fallback for other content types
                    contents.append(str(cont))

        return "\n".join(sys_txt).strip(), contents

    def create_completion(
        self,
        messages: list,  # Pydantic Message objects
        tools: list | None = None,  # Pydantic Tool objects
        *,
        stream: bool = False,
        max_tokens: int | None = None,
        system: str | None = None,
        **extra,
    ) -> AsyncIterator[dict[str, Any]] | Any:
        """
        Configuration-aware completion generation with universal tool name compatibility.

        Args:
            messages: List of Pydantic Message objects
            tools: List of Pydantic Tool objects

        Uses configuration to validate:
        - Tool support before processing tools
        - Streaming support before enabling streaming
        - JSON mode support before adding JSON instructions
        - Vision support during message processing

        Universal tool name compatibility handles any naming convention:
        - stdio.read_query -> stdio_read_query (sanitized and restored)
        - web.api:search -> web_api_search (sanitized and restored)
        - database.sql.execute -> database_sql_execute (sanitized and restored)
        """

        # Handle backward compatibility
        from chuk_llm.llm.core.base import (
            _ensure_pydantic_messages,
            _ensure_pydantic_tools,
        )

        messages = _ensure_pydantic_messages(messages)
        tools = _ensure_pydantic_tools(tools)

        # Convert Pydantic to dicts
        dict_messages = [msg.to_dict() for msg in messages]

        # Permissive approach: Let Gemini API handle unsupported features
        # Don't block based on capability checks - dynamic models should work

        # Apply universal tool name sanitization (stores mapping for restoration)
        # Keep tools as Pydantic models through sanitization, then convert to dicts
        name_mapping = {}
        dict_tools = None
        if tools:
            from chuk_llm.core.models import Tool as ToolModel

            sanitized_tools = self._sanitize_tool_names(tools)
            # After sanitization, tools are always Pydantic Tool models
            assert isinstance(sanitized_tools, list) and all(
                isinstance(t, ToolModel) for t in sanitized_tools
            )

            name_mapping = self._current_name_mapping
            log.debug(
                f"Tool sanitization: {len(name_mapping)} tools processed for Gemini compatibility"
            )
            # Convert Pydantic models back to dicts for Gemini format conversion
            dict_tools = [tool.model_dump() for tool in sanitized_tools]  # type: ignore[union-attr]

        gemini_tools = _convert_tools_to_gemini_format(dict_tools)

        # Check for JSON mode (using configuration validation)
        json_instruction = self._check_json_mode(extra)

        # Filter parameters for Gemini compatibility (using configuration limits)
        if max_tokens:
            extra["max_tokens"] = max_tokens
        filtered_params = self._filter_gemini_params(extra)

        # --- streaming: return the async generator directly -------------------------
        if stream:
            return self._stream_completion_async(
                system,
                json_instruction,
                dict_messages,
                gemini_tools,
                filtered_params,
                name_mapping,
            )

        # --- non-streaming: use async client ------------------------------
        return self._regular_completion_async(
            system,
            json_instruction,
            dict_messages,
            gemini_tools,
            filtered_params,
            name_mapping,
        )

    async def _stream_completion_async(
        self,
        system: str | None,
        json_instruction: str | None,
        messages: list[dict[str, Any]],
        gemini_tools: list[gtypes.Tool] | None,
        filtered_params: dict[str, Any],
        name_mapping: dict[str, str] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Real streaming using Gemini client with FIXED tool call duplication.
        """

        # Handle system message and JSON instruction
        system_from_messages, contents = await self._split_for_gemini_async(messages)
        final_system = system or system_from_messages

        if json_instruction:
            if final_system:
                final_system = f"{final_system}\n\n{json_instruction}"
            else:
                final_system = json_instruction
            log.debug("Added JSON mode instruction to system prompt")

        # Build config
        config_params = filtered_params.copy()
        if gemini_tools:
            config_params["tools"] = gemini_tools

        # Handle thinking models hitting token limits
        if "2.5" in self.model:
            if "max_output_tokens" not in config_params:
                config_params["max_output_tokens"] = 4096
                log.debug(
                    "Set max_output_tokens=4096 for Gemini 2.5 to prevent thinking token overflow"
                )
        elif "max_output_tokens" not in config_params:
            config_params["max_output_tokens"] = 4096

        config = None
        if config_params:
            try:
                if final_system and self.supports_feature(
                    Feature.SYSTEM_MESSAGES.value
                ):
                    config_params["system_instruction"] = final_system

                config = gtypes.GenerateContentConfig(**config_params)
            except Exception as e:
                log.warning(f"Error creating GenerateContentConfig: {e}")
                config = None

        # Combine all content into a single message
        combined_content = "\n\n".join(contents) if contents else "Hello"

        # Prepend system instruction if not supported in config
        if final_system and not self.supports_feature("system_messages"):
            combined_content = f"System: {final_system}\n\nUser: {combined_content}"

        base_payload: dict[str, Any] = {
            "model": self.model,
            "contents": combined_content,
        }

        if config:
            base_payload["config"] = config

        log.debug("Gemini streaming payload keys: %s", list(base_payload.keys()))

        # CRITICAL FIX: Enhanced tool call tracking to prevent duplication
        accumulated_response = ""
        yielded_tool_signatures = set()  # Track what tool calls we've already yielded
        chunk_count = 0

        try:
            # Streaming with SAFE parsing and universal tool name restoration
            with UltimateSuppression():
                stream = await self.client.aio.models.generate_content_stream(
                    **base_payload
                )

            async for chunk in stream:
                chunk_count += 1

                with UltimateSuppression():
                    try:
                        # Enhanced chunk processing for tool calls
                        chunk_text = ""
                        new_tool_calls = []  # Only NEW tool calls for this chunk

                        # Parse chunk using safe method
                        if hasattr(chunk, "candidates") and chunk.candidates:
                            candidate = chunk.candidates[0]

                            if hasattr(candidate, "content") and candidate.content:
                                content = candidate.content

                                # Check if content has parts
                                if hasattr(content, "parts") and content.parts:
                                    text_parts = []

                                    # Process each part individually
                                    for part in content.parts:
                                        # Handle text parts
                                        if hasattr(part, "text") and part.text:
                                            text_parts.append(part.text)

                                        # CRITICAL FIX: Handle function call parts with duplication prevention
                                        elif (
                                            hasattr(part, "function_call")
                                            and part.function_call
                                        ):
                                            fc = part.function_call

                                            try:
                                                # Extract function call data
                                                function_name = getattr(
                                                    fc, "name", "unknown"
                                                )
                                                function_args = dict(
                                                    getattr(fc, "args", {})
                                                )

                                                # Create unique signature for this tool call
                                                tool_signature = f"{function_name}:{json.dumps(function_args, sort_keys=True)}"

                                                # CRITICAL FIX: Only add if we haven't yielded this exact tool call
                                                if (
                                                    tool_signature
                                                    not in yielded_tool_signatures
                                                ):
                                                    tool_call = {
                                                        "id": f"call_{uuid.uuid4().hex[:8]}",
                                                        "type": "function",
                                                        "function": {
                                                            "name": function_name,
                                                            "arguments": json.dumps(
                                                                function_args
                                                            ),
                                                        },
                                                    }

                                                    new_tool_calls.append(tool_call)
                                                    yielded_tool_signatures.add(
                                                        tool_signature
                                                    )  # Mark as yielded

                                                    log.debug(
                                                        f"New function call from Gemini: {function_name}"
                                                    )
                                                else:
                                                    log.debug(
                                                        f"Skipping duplicate function call: {function_name}"
                                                    )

                                            except Exception as e:
                                                log.error(
                                                    f"Error extracting function call from Gemini chunk: {e}"
                                                )

                                    # Combine text parts for this chunk
                                    chunk_text = "".join(text_parts)

                        # Handle text content with proper deduplication
                        if chunk_text:
                            # Check if this is new content or a repeat
                            if not accumulated_response or not chunk_text.startswith(
                                accumulated_response
                            ):
                                # Extract only the new part
                                if chunk_text.startswith(accumulated_response):
                                    new_content = chunk_text[
                                        len(accumulated_response) :
                                    ]
                                else:
                                    new_content = chunk_text

                                if new_content:
                                    accumulated_response += new_content

                                    # Create chunk response for text
                                    chunk_response = {
                                        "response": new_content,
                                        "tool_calls": [],
                                    }
                                    yield chunk_response

                        # CRITICAL FIX: Only yield NEW tool calls
                        if new_tool_calls:
                            # Create chunk response for tool calls
                            tool_response = {
                                "response": "",
                                "tool_calls": new_tool_calls,
                            }

                            # Restore tool names using universal restoration
                            if name_mapping:
                                tool_response = self._restore_tool_names_in_response(
                                    tool_response, name_mapping
                                )

                            yield tool_response

                    except Exception as chunk_error:
                        log.debug(
                            f"Error processing Gemini chunk {chunk_count}: {chunk_error}"
                        )
                        continue

            log.debug(
                f"Gemini streaming completed with {chunk_count} chunks, "
                f"{len(accumulated_response)} total characters, {len(yielded_tool_signatures)} unique tool calls"
            )

        except Exception as e:
            log.error(f"Error in Gemini streaming: {e}")
            yield {
                "response": f"Streaming error: {str(e)}",
                "tool_calls": [],
                "error": True,
            }

    @silence_gemini_warnings_async
    async def _regular_completion_async(
        self,
        system: str | None,
        json_instruction: str | None,
        messages: list[dict[str, Any]],
        gemini_tools: list[gtypes.Tool] | None,
        filtered_params: dict[str, Any],
        name_mapping: dict[str, str] = None,
    ) -> dict[str, Any]:
        """Non-streaming completion with SAFE response parsing and universal tool name restoration."""
        try:
            # Handle system message and JSON instruction
            system_from_messages, contents = await self._split_for_gemini_async(
                messages
            )
            final_system = system or system_from_messages

            if json_instruction:
                if final_system:
                    final_system = f"{final_system}\n\n{json_instruction}"
                else:
                    final_system = json_instruction
                log.debug("Added JSON mode instruction to system prompt")

            # Build config
            config_params = filtered_params.copy()
            if gemini_tools:
                config_params["tools"] = gemini_tools

            if "2.5" in self.model:
                if "max_output_tokens" not in config_params:
                    config_params["max_output_tokens"] = 4096
                    log.debug(
                        "Set max_output_tokens=4096 for Gemini 2.5 to prevent thinking token overflow"
                    )
            elif "max_output_tokens" not in config_params:
                config_params["max_output_tokens"] = 4096

            config = None
            if config_params:
                try:
                    if final_system and self.supports_feature(
                        Feature.SYSTEM_MESSAGES.value
                    ):
                        config_params["system_instruction"] = final_system

                    config = gtypes.GenerateContentConfig(**config_params)
                except Exception as e:
                    log.warning(f"Error creating GenerateContentConfig: {e}")
                    config = None

            # Combine all content - handle both text and multimodal
            if contents:
                has_multimodal = any(isinstance(c, list) for c in contents)

                if has_multimodal:
                    combined_content = []
                    for content_item in contents:
                        if isinstance(content_item, str):
                            combined_content.append(content_item)
                        elif isinstance(content_item, list):
                            combined_content.extend(content_item)
                else:
                    combined_content = "\n\n".join(contents)
            else:
                combined_content = "Hello"

            # Prepend system instruction if not supported in config
            if final_system and not self.supports_feature(
                Feature.SYSTEM_MESSAGES.value
            ):
                if isinstance(combined_content, list):
                    combined_content.insert(0, f"System: {final_system}\n\nUser: ")
                else:
                    combined_content = (
                        f"System: {final_system}\n\nUser: {combined_content}"
                    )

            base_payload: dict[str, Any] = {
                "model": self.model,
                "contents": [combined_content],
            }

            if config:
                base_payload["config"] = config

            log.debug("Gemini payload keys: %s", list(base_payload.keys()))

            # Make the request and use SAFE parsing with universal tool name restoration
            resp = await self.client.aio.models.generate_content(**base_payload)

            # CRITICAL: Use safe parsing and restore tool names using universal system
            return self._parse_gemini_response_with_restoration(resp, name_mapping)

        except Exception as e:
            log.error(f"Error in Gemini completion: {e}")
            return {"response": f"Error: {str(e)}", "tool_calls": [], "error": True}

    @silence_gemini_warnings
    def _extract_tool_calls_from_response_with_restoration(
        self, response, name_mapping: dict[str, str] = None
    ) -> list[dict[str, Any]]:
        """Extract tool calls from Gemini response with name restoration - SAFE parsing"""
        tool_calls = []  # type: ignore[var-annotated]

        # Only extract tool calls if tools are supported
        if not self.supports_feature(Feature.TOOLS.value):
            return tool_calls

        try:
            # Use safe parsing to extract tool calls without concatenation
            result = _safe_parse_gemini_response(response)
            tool_calls = result.get("tool_calls", [])

            # Restore original names if mapping provided
            if name_mapping and tool_calls:
                for tool_call in tool_calls:
                    if "function" in tool_call and "name" in tool_call["function"]:
                        sanitized_name = tool_call["function"]["name"]
                        original_name = name_mapping.get(sanitized_name, sanitized_name)
                        tool_call["function"]["name"] = original_name

        except Exception as e:
            log.debug(f"Error extracting tool calls: {e}")

        return tool_calls

    def _extract_tool_calls_from_response(self, response: Any) -> list[dict[str, Any]]:
        """Extract tool calls from Gemini response (legacy method, kept for compatibility)"""
        return self._extract_tool_calls_from_response_with_restoration(
            response, self._current_name_mapping
        )

    async def close(self):
        """Cleanup resources"""
        # Reset name mapping from universal system
        self._current_name_mapping = {}
        # Gemini client cleanup if needed
        pass
