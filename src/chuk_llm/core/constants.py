"""
Core Constants
==============

All constant values used throughout the codebase.
No magic strings allowed!
"""

from enum import StrEnum

# ================================================================
# HTTP Constants
# ================================================================


class HttpMethod(StrEnum):
    """HTTP request methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class HttpHeader(StrEnum):
    """Common HTTP headers."""

    AUTHORIZATION = "Authorization"
    CONTENT_TYPE = "Content-Type"
    USER_AGENT = "User-Agent"
    RETRY_AFTER = "Retry-After"
    ACCEPT = "Accept"


class ContentTypeValue(StrEnum):
    """Content-Type header values."""

    JSON = "application/json"
    TEXT = "text/plain"
    FORM = "application/x-www-form-urlencoded"


# ================================================================
# API Endpoints
# ================================================================


class OpenAIEndpoint(StrEnum):
    """OpenAI API endpoints."""

    CHAT_COMPLETIONS = "/chat/completions"
    COMPLETIONS = "/completions"
    EMBEDDINGS = "/embeddings"
    MODELS = "/models"


class AnthropicEndpoint(StrEnum):
    """Anthropic API endpoints."""

    MESSAGES = "/v1/messages"
    COMPLETIONS = "/v1/complete"


# ================================================================
# API Base URLs
# ================================================================


class ApiBaseUrl(StrEnum):
    """Default API base URLs for each provider."""

    OPENAI = "https://api.openai.com/v1"
    ANTHROPIC = "https://api.anthropic.com"
    GROQ = "https://api.groq.com/openai/v1"
    DEEPSEEK = "https://api.deepseek.com/v1"
    TOGETHER = "https://api.together.xyz/v1"
    PERPLEXITY = "https://api.perplexity.ai"
    MISTRAL = "https://api.mistral.ai/v1"
    OPENROUTER = "https://openrouter.ai/api/v1"
    GEMINI = "https://generativelanguage.googleapis.com"
    OLLAMA = "http://localhost:11434"
    MOONSHOT = "https://api.moonshot.ai/v1"


# ================================================================
# SSE (Server-Sent Events) Constants
# ================================================================


class SSEPrefix(StrEnum):
    """Server-Sent Events prefixes."""

    DATA = "data: "
    EVENT = "event: "
    ID = "id: "
    RETRY = "retry: "


class SSEEvent(StrEnum):
    """SSE event types."""

    MESSAGE = "message"
    DONE = "[DONE]"
    ERROR = "error"


# ================================================================
# Tool/Function Call Constants
# ================================================================


class ToolType(StrEnum):
    """Tool/function call types."""

    FUNCTION = "function"


# ================================================================
# Reasoning Model Generations
# ================================================================


class ReasoningGeneration(StrEnum):
    """Reasoning model generations."""

    O1 = "o1"
    O3 = "o3"
    O4 = "o4"
    O5 = "o5"
    GPT5 = "gpt5"
    UNKNOWN = "unknown"


# ================================================================
# API Response Keys (for parsing responses)
# ================================================================


class ResponseKey(StrEnum):
    """Common API response keys."""

    CHOICES = "choices"
    MESSAGE = "message"
    DELTA = "delta"
    CONTENT = "content"
    ROLE = "role"
    TOOL_CALLS = "tool_calls"
    FUNCTION = "function"
    NAME = "name"
    ARGUMENTS = "arguments"
    ID = "id"
    INDEX = "index"
    FINISH_REASON = "finish_reason"
    USAGE = "usage"
    MODEL = "model"
    PROMPT_TOKENS = "prompt_tokens"
    COMPLETION_TOKENS = "completion_tokens"
    TOTAL_TOKENS = "total_tokens"
    REASONING_TOKENS = "reasoning_tokens"
    COMPLETION_TOKENS_DETAILS = "completion_tokens_details"
    ERROR = "error"
    ERROR_MESSAGE = "error_message"
    TYPE = "type"
    DATA = "data"  # For /v1/models response
    RESULT = "result"  # For tool execution results


# ================================================================
# Error Types
# ================================================================


class ErrorType(StrEnum):
    """LLM error types."""

    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_ERROR = "permission_error"
    NOT_FOUND_ERROR = "not_found_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    SERVER_ERROR = "server_error"
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    STREAMING_ERROR = "streaming_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"


# ================================================================
# Parameter Names (for API requests)
# ================================================================


class RequestParam(StrEnum):
    """API request parameter names (Chat Completions API)."""

    MODEL = "model"
    MESSAGES = "messages"
    TEMPERATURE = "temperature"
    MAX_TOKENS = "max_tokens"
    MAX_COMPLETION_TOKENS = "max_completion_tokens"
    TOP_P = "top_p"
    FREQUENCY_PENALTY = "frequency_penalty"
    PRESENCE_PENALTY = "presence_penalty"
    STOP = "stop"
    TOOLS = "tools"
    STREAM = "stream"
    RESPONSE_FORMAT = "response_format"
    LOGIT_BIAS = "logit_bias"


class ResponsesRequestParam(StrEnum):
    """API request parameter names (Responses API)."""

    MODEL = "model"
    INPUT = "input"
    INSTRUCTIONS = "instructions"
    PREVIOUS_RESPONSE_ID = "previous_response_id"
    STORE = "store"
    STREAM = "stream"
    TEMPERATURE = "temperature"
    MAX_OUTPUT_TOKENS = "max_output_tokens"
    MAX_TOOL_CALLS = "max_tool_calls"
    TOP_P = "top_p"
    TOOLS = "tools"
    TOOL_CHOICE = "tool_choice"
    PARALLEL_TOOL_CALLS = "parallel_tool_calls"
    TEXT = "text"
    REASONING = "reasoning"
    METADATA = "metadata"
    BACKGROUND = "background"
    TRUNCATION = "truncation"


# ================================================================
# Configuration Keys
# ================================================================


class ConfigKey(StrEnum):
    """Configuration keys."""

    PROVIDERS = "providers"
    DEFAULT_PROVIDER = "default_provider"
    PROVIDER = "provider"
    MODEL = "model"
    API_KEY = "api_key"
    API_BASE = "api_base"
    DEFAULT_MODEL = "default_model"
    MODELS = "models"
    FEATURES = "features"
    CAPABILITIES = "capabilities"
    PATTERN = "pattern"
    TIMEOUT = "timeout"
    MAX_RETRIES = "max_retries"
    ENABLED = "enabled"
    DYNAMIC_DISCOVERY = "dynamic_discovery"
    ACTIVE_PROVIDER = "active_provider"
    GLOBAL = "__global__"
    GLOBAL_ALIASES = "__global_aliases__"
    SYSTEM_PROMPT = "system_prompt"
    TEMPERATURE = "temperature"
    MAX_TOKENS = "max_tokens"


class CapabilityKey(StrEnum):
    """Model capability flag names."""

    SUPPORTS_TEXT = "supports_text"
    SUPPORTS_STREAMING = "supports_streaming"
    SUPPORTS_TOOLS = "supports_tools"
    SUPPORTS_VISION = "supports_vision"
    SUPPORTS_JSON_MODE = "supports_json_mode"
    SUPPORTS_SYSTEM_MESSAGES = "supports_system_messages"
    SUPPORTS_PARALLEL_CALLS = "supports_parallel_calls"
    SUPPORTS_REASONING = "supports_reasoning"


class AzureOpenAIParam(StrEnum):
    """Azure OpenAI specific configuration parameters."""

    API_VERSION = "api_version"
    AZURE_DEPLOYMENT = "azure_deployment"
    AZURE_ENDPOINT = "azure_endpoint"
    API_KEY = "api_key"


class ToolParam(StrEnum):
    """Tool/function call parameter names."""

    TYPE = "type"
    ID = "id"
    NAME = "name"
    DESCRIPTION = "description"
    PARAMETERS = "parameters"
    FUNCTION = "function"
    ARGUMENTS = "arguments"
    TOOL_CALL_ID = "tool_call_id"


# ================================================================
# Environment Variable Names
# ================================================================


class EnvVar(StrEnum):
    """Environment variable names."""

    # OpenAI
    OPENAI_API_KEY = "OPENAI_API_KEY"
    OPENAI_API_BASE = "OPENAI_API_BASE"
    OPENAI_ORG_ID = "OPENAI_ORG_ID"

    # Anthropic
    ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"

    # Azure
    AZURE_OPENAI_API_KEY = "AZURE_OPENAI_API_KEY"
    AZURE_OPENAI_ENDPOINT = "AZURE_OPENAI_ENDPOINT"
    AZURE_OPENAI_API_VERSION = "AZURE_OPENAI_API_VERSION"

    # Gemini
    GOOGLE_API_KEY = "GOOGLE_API_KEY"
    GEMINI_API_KEY = "GEMINI_API_KEY"

    # Others
    GROQ_API_KEY = "GROQ_API_KEY"
    MISTRAL_API_KEY = "MISTRAL_API_KEY"
    DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"
    MOONSHOT_API_KEY = "MOONSHOT_API_KEY"

    # ChukLLM specific
    CHUK_LLM_CONFIG = "CHUK_LLM_CONFIG"
    CHUK_LLM_DISABLE_SESSIONS = "CHUK_LLM_DISABLE_SESSIONS"


# ================================================================
# Default Values
# ================================================================


class Default:
    """Default configuration values."""

    TIMEOUT = 60.0
    MAX_RETRIES = 3
    MAX_CONNECTIONS = 100
    MAX_KEEPALIVE = 20
    TEMPERATURE = 0.7
    MAX_TOKENS = 4096
    CACHE_TTL = 3600


# ================================================================
# Model Patterns (for detection)
# ================================================================


class ModelPattern:
    """Regex patterns for model detection."""

    # Reasoning models
    O1_MODELS = r"o1-"
    O3_MODELS = r"o3-"
    O4_MODELS = r"o4-"
    O5_MODELS = r"o5-"
    GPT5_MODELS = r"^gpt-5"  # Must start with gpt-5

    # Vision models
    VISION_MODELS = r"(vision|gpt-4|claude-3|gemini-pro-vision)"

    # Chat models
    GPT_CHAT = r"gpt-(3\.5|4)"
    CLAUDE_CHAT = r"claude-[0-9]"


# ================================================================
# HTTP Status Codes
# ================================================================


class HttpStatus:
    """HTTP status codes."""

    OK = 200
    CREATED = 201
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    RATE_LIMIT = 429
    SERVER_ERROR = 500
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
