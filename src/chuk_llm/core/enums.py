"""
Core Enumerations
=================

Type-safe enums for providers, features, roles, and other constants.
No more magic strings!
"""

from enum import StrEnum


class Provider(StrEnum):
    """LLM Provider enumeration."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"
    GEMINI = "gemini"
    GROQ = "groq"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    WATSONX = "watsonx"
    ADVANTAGE = "advantage"
    PERPLEXITY = "perplexity"
    TOGETHER = "together"
    ANYSCALE = "anyscale"
    OPENAI_COMPATIBLE = "openai_compatible"
    OPENROUTER = "openrouter"
    MOONSHOT = "moonshot"
    LLAMA_CPP = "llamacpp"


class Feature(StrEnum):
    """Model feature capabilities."""

    TEXT = "text"  # Basic text completion capability
    STREAMING = "streaming"
    TOOLS = "tools"
    VISION = "vision"
    AUDIO_INPUT = "audio_input"  # Audio input processing
    JSON_MODE = "json_mode"
    SYSTEM_MESSAGES = "system_messages"
    PARALLEL_CALLS = "parallel_calls"
    MULTIMODAL = "multimodal"  # Multiple input modalities
    REASONING = "reasoning"

    @classmethod
    def from_string(cls, value: str) -> "Feature":
        """Convert string to Feature enum"""
        try:
            return cls(value.lower())
        except ValueError as exc:
            raise ValueError(f"Unknown feature: {value}") from exc


class MessageRole(StrEnum):
    """Chat message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"  # Deprecated but supported for backwards compatibility


class FinishReason(StrEnum):
    """Completion finish reason."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class ContentType(StrEnum):
    """Content part type for multimodal messages."""

    TEXT = "text"
    IMAGE_URL = "image_url"
    IMAGE_DATA = "image_data"
    INPUT_AUDIO = "input_audio"
    AUDIO_URL = "audio_url"
    AUDIO_DATA = "audio_data"


class ToolType(StrEnum):
    """Tool/function call types."""

    FUNCTION = "function"


class ReasoningGeneration(StrEnum):
    """Reasoning model generations."""

    O1 = "o1"
    O3 = "o3"
    O4 = "o4"
    O5 = "o5"
    GPT5 = "gpt5"
    UNKNOWN = "unknown"


class ResponsesTextFormatType(StrEnum):
    """Responses API text format types."""

    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"
