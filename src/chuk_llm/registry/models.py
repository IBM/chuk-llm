"""
Core Pydantic models for the dynamic capability registry system.

This module defines the foundational data structures for model discovery,
capability resolution, and intelligent model selection across all providers.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QualityTier(StrEnum):
    """Quality tier classification for models."""

    BEST = "best"  # Frontier models (GPT-4o, Claude 3.5 Sonnet, Gemini Pro)
    BALANCED = "balanced"  # Mid-tier (GPT-4o-mini, Claude 3.5 Haiku)
    CHEAP = "cheap"  # Budget models (GPT-3.5, local models)
    UNKNOWN = "unknown"  # Not yet classified


class ModelSpec(BaseModel):
    """
    Raw identity of a discovered model.

    This is what the registry discovers via ModelSource implementations.
    It represents the minimal information needed to identify a model.
    """

    provider: str = Field(
        ..., description="Provider name (e.g., 'openai', 'anthropic')"
    )
    name: str = Field(
        ...,
        description="Model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022')",
    )
    family: str | None = Field(
        None, description="Model family (e.g., 'gpt-4', 'claude-3')"
    )
    aliases: list[str] = Field(
        default_factory=list, description="Alternative names for this model"
    )

    model_config = ConfigDict(frozen=True)

    def __hash__(self) -> int:
        """Allow ModelSpec to be used in sets and as dict keys."""
        return hash((self.provider, self.name))

    def __str__(self) -> str:
        """Human-readable representation."""
        return f"{self.provider}:{self.name}"


class ModelCapabilities(BaseModel):
    """
    Enriched metadata about what a model can do.

    This is what CapabilityResolvers produce and what the rest of CHUK consumes.
    Capabilities are layered - later resolvers override earlier ones.
    """

    # Core capabilities (usually queryable or documented)
    max_context: int | None = Field(
        None, description="Maximum context window in tokens"
    )
    max_output_tokens: int | None = Field(
        None, description="Maximum output tokens per request"
    )

    supports_tools: bool | None = Field(
        None, description="Function/tool calling support"
    )
    supports_vision: bool | None = Field(None, description="Image/vision input support")
    supports_audio: bool | None = Field(None, description="Audio input support")
    supports_json_mode: bool | None = Field(
        None, description="Basic JSON output mode (response_format={type: json_object})"
    )
    supports_structured_outputs: bool | None = Field(
        None,
        description="Structured outputs with JSON Schema (response_format={type: json_schema})",
    )
    supports_streaming: bool | None = Field(
        None, description="Streaming response support"
    )
    supports_system_messages: bool | None = Field(
        None, description="System message support"
    )

    # Parameter compatibility
    known_params: set[str] = Field(
        default_factory=set,
        description="Supported parameters (e.g., {'temperature', 'top_p', 'max_tokens'})",
    )

    # Performance metrics (measured via testing)
    quality_tier: QualityTier = Field(
        QualityTier.UNKNOWN, description="Quality tier classification"
    )
    tokens_per_second: float | None = Field(
        None, description="Measured tokens per second from benchmarks"
    )

    # Metadata
    source: str | None = Field(None, description="Where this capability data came from")
    last_updated: str | None = Field(None, description="ISO timestamp of last update")

    model_config = ConfigDict(frozen=False)  # Mutable for merging

    def merge(self, other: ModelCapabilities) -> ModelCapabilities:
        """
        Merge another capabilities object into this one.

        Non-None values from 'other' override values in 'self'.
        This enables the layered resolver pattern.
        """
        updates = {
            k: v
            for k, v in other.model_dump(exclude_none=True).items()
            if v is not None and (k != "known_params")  # Handle sets separately
        }

        # Merge known_params sets
        if other.known_params:
            updates["known_params"] = self.known_params | other.known_params

        return self.model_copy(update=updates)


class ModelWithCapabilities(BaseModel):
    """
    Complete model information: identity + capabilities.

    This is what the registry returns and what higher layers consume.
    """

    spec: ModelSpec = Field(..., description="Model identity")
    capabilities: ModelCapabilities = Field(..., description="Model capabilities")

    model_config = ConfigDict(frozen=True)

    def __str__(self) -> str:
        """Human-readable representation."""
        caps_summary = []
        if self.capabilities.supports_tools:
            caps_summary.append("tools")
        if self.capabilities.supports_vision:
            caps_summary.append("vision")
        if self.capabilities.max_context:
            caps_summary.append(f"{self.capabilities.max_context // 1000}k ctx")

        caps_str = ", ".join(caps_summary) if caps_summary else "basic"
        return f"{self.spec} ({caps_str})"


class ModelQuery(BaseModel):
    """
    Query parameters for finding suitable models.

    Used by registry.find_best() and other intelligent selection APIs.
    """

    requires_tools: bool = False
    requires_vision: bool = False
    requires_audio: bool = False
    requires_json_mode: bool = False

    min_context: int | None = None
    max_context: int | None = None

    quality_tier: QualityTier | Literal["any"] = "any"
    min_speed_tps: float | None = None  # Minimum tokens per second

    provider: str | None = None  # Filter to specific provider
    family: str | None = None  # Filter to model family

    model_config = ConfigDict(frozen=True)

    def matches(self, model: ModelWithCapabilities) -> bool:
        """Check if a model satisfies this query."""
        caps = model.capabilities

        # Required capabilities
        if self.requires_tools and not caps.supports_tools:
            return False
        if self.requires_vision and not caps.supports_vision:
            return False
        if self.requires_audio and not caps.supports_audio:
            return False
        if self.requires_json_mode and not caps.supports_json_mode:
            return False

        # Context constraints
        if self.min_context and (
            not caps.max_context or caps.max_context < self.min_context
        ):
            return False
        if (
            self.max_context
            and caps.max_context
            and caps.max_context > self.max_context
        ):
            return False

        # Speed constraints
        if self.min_speed_tps and caps.tokens_per_second:
            if caps.tokens_per_second < self.min_speed_tps:
                return False

        # Quality tier
        if self.quality_tier != "any" and caps.quality_tier != self.quality_tier:
            return False

        # Provider/family filtering
        if self.provider and model.spec.provider != self.provider:
            return False
        if self.family and model.spec.family != self.family:
            return False

        return True
