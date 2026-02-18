# src/chuk_llm/llm/__init__.py
"""
ChukLLM LLM Module with Unified Configuration
============================================

Core LLM functionality including clients and providers with the new unified configuration system.
"""

import logging
from typing import Any, Optional  # noqa: F401

# Set up logging
logger = logging.getLogger(__name__)

# Version info
__version__ = "0.1.0"

# Unified configuration system
try:
    from ..configuration import (
        CapabilityChecker,
        ConfigManager,
        ConfigValidator,
        Feature,
        get_config,
        reset_config,
    )

    # Backward compatibility aliases
    # get_config and reset_config already imported above
    # ConfigManager already imported above

    _config_available = True
    logger.debug("Unified configuration system loaded successfully")
except ImportError as e:
    logger.debug(f"Configuration module import failed: {e}")
    _config_available = False

    # Provide fallback implementations with proper signatures
    def get_config() -> Any:  # type: ignore[misc]
        raise ImportError("Configuration module not available")

    def reset_config():  # type: ignore[misc]
        raise ImportError("Configuration module not available")

    # Create fallback classes with different names to avoid conflicts
    class _FallbackConfigManager:
        def __init__(self):
            raise ImportError("Configuration module not available")

    class _FallbackConfigValidator:
        pass

    class _FallbackCapabilityChecker:
        pass

    class _FallbackFeature:
        pass

    # Assign to original names
    ConfigManager = _FallbackConfigManager  # type: ignore[misc,assignment]
    ConfigValidator = _FallbackConfigValidator  # type: ignore[misc,assignment]
    CapabilityChecker = _FallbackCapabilityChecker  # type: ignore[misc,assignment]
    Feature = _FallbackFeature  # type: ignore[misc,assignment]


# Core LLM functionality with unified client factory
try:
    from .client import get_client, list_available_providers, validate_provider_setup
    from .core.base import BaseLLMClient

    _llm_available = True
    logger.debug("LLM client factory loaded successfully")
except ImportError as e:
    logger.debug(f"LLM client import failed: {e}")
    _llm_available = False

    def get_client(  # type: ignore[misc]
        provider: str,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> Any:  # type: ignore[misc]
        raise ImportError("LLM client module not available")

    def validate_provider_setup(provider: str) -> dict[str, Any]:  # type: ignore[misc]
        raise ImportError("LLM client module not available")

    def list_available_providers() -> dict[str, dict[str, Any]]:  # type: ignore[misc]
        raise ImportError("LLM client module not available")

    class _FallbackBaseLLMClient:
        pass

    BaseLLMClient = _FallbackBaseLLMClient  # type: ignore[misc,assignment]


# Enhanced features with unified config
try:
    from .features import (
        ProviderAdapter,
        UnifiedLLMInterface,
        find_best_provider_for_task,
        multi_provider_chat,
        quick_chat,
    )

    _features_available = True
    logger.debug("Enhanced features loaded successfully")
except ImportError as e:
    logger.debug(f"Features module import failed: {e}")
    _features_available = False

    class _FallbackUnifiedLLMInterface:
        def __init__(self, *args, **kwargs):
            raise ImportError("Features module not available")

    class _FallbackProviderAdapter:
        pass

    async def quick_chat(
        provider: str, model: str | None = None, message: str = "", **kwargs: Any
    ) -> str:
        raise ImportError("Features module not available")

    async def multi_provider_chat(
        message: str,
        providers: list[str],
        model_map: dict[str, str] | None = None,
        **chat_kwargs: Any,
    ) -> dict[str, Any]:
        raise ImportError("Features module not available")

    async def find_best_provider_for_task(  # type: ignore[misc]
        message: str,
        required_features: list[str] | None = None,
        exclude_providers: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        raise ImportError("Features module not available")

    # Assign to original names
    UnifiedLLMInterface = _FallbackUnifiedLLMInterface  # type: ignore[assignment,misc]
    ProviderAdapter = _FallbackProviderAdapter  # type: ignore[assignment,misc]


# Updated API layer
try:
    from ..api import (
        ask,
        ask_sync,
        compare_providers,
        configure,
        get_current_config,
        quick_question,
        stream,
        stream_sync,
    )
    from ..api import (
        get_client as api_get_client,
    )
    from ..api import (  # type: ignore[attr-defined]
        reset_config as api_reset_config,
    )

    # New enhanced API functions
    from ..api.core import (
        ask_json,
        multi_provider_ask,
        quick_ask,
        validate_request,
    )

    _api_available = True
    logger.debug("API layer loaded successfully")
except ImportError as e:
    logger.debug(f"API module import failed: {e}")
    _api_available = False

    # Provide fallback implementations
    async def ask(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("API module not available")

    async def stream(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("API module not available")

    def ask_sync(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("API module not available")

    def stream_sync(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("API module not available")

    def configure(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("API module not available")

    def get_current_config():  # type: ignore[misc]
        raise ImportError("API module not available")

    def api_get_client(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("API module not available")

    def api_reset_config():  # type: ignore[misc]
        raise ImportError("API module not available")

    def compare_providers(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("API module not available")

    def quick_question(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("API module not available")

    # Enhanced API fallbacks

    async def ask_json(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("API module not available")

    async def quick_ask(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("API module not available")

    async def multi_provider_ask(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("API module not available")

    def validate_request(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("API module not available")


# Conversation management
try:
    from ..api.conversation import ConversationContext, conversation

    _conversation_available = True
    logger.debug("Conversation management loaded successfully")
except ImportError as e:
    logger.debug(f"Conversation module import failed: {e}")
    _conversation_available = False

    async def conversation(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Conversation module not available")

    class ConversationContext:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("Conversation module not available")


# Enhanced utilities
try:
    # Enhanced config utilities
    from ..api.config import (
        auto_configure,
        debug_config_state,
        get_capabilities,
        list_available_setups,
        quick_setup,
        supports_feature,
        switch_provider,
        validate_config,
    )
    from ..api.utils import health_check, print_diagnostics, test_connection

    _utils_available = True
    logger.debug("Utilities loaded successfully")
except ImportError as e:
    logger.debug(f"Utils module import failed: {e}")
    _utils_available = False

    async def health_check():  # type: ignore[misc]
        raise ImportError("Utils module not available")

    async def test_connection(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Utils module not available")

    def print_diagnostics():  # type: ignore[misc]
        raise ImportError("Utils module not available")

    # Enhanced config utility fallbacks
    def debug_config_state():  # type: ignore[misc]
        raise ImportError("Utils module not available")

    def quick_setup(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Utils module not available")

    def list_available_setups():  # type: ignore[misc]
        raise ImportError("Utils module not available")

    def switch_provider(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Utils module not available")

    def auto_configure(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Utils module not available")

    def validate_config():  # type: ignore[misc]
        raise ImportError("Utils module not available")

    def get_capabilities():  # type: ignore[misc]
        raise ImportError("Utils module not available")

    def supports_feature(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("Utils module not available")


def get_version() -> str:
    """Get the current version of ChukLLM."""
    return __version__


def get_available_modules() -> dict[str, bool]:
    """Get information about which modules are available."""
    return {
        "config": _config_available,
        "llm": _llm_available,
        "features": _features_available,
        "api": _api_available,
        "conversation": _conversation_available,
        "utils": _utils_available,
    }


def check_installation() -> dict[str, Any]:
    """Check the installation status and provide comprehensive diagnostic information."""
    modules = get_available_modules()

    issues = []
    warning_messages = []

    if not modules["config"]:
        issues.append("Unified configuration module failed to import")
    if not modules["llm"]:
        issues.append("LLM client module failed to import")
    if not modules["features"]:
        warning_messages.append("Enhanced features module not available")
    if not modules["api"]:
        issues.append("API module failed to import")

    # Test configuration if available
    config_status = {}
    if modules["config"]:
        try:
            config = get_config()
            providers = config.get_all_providers()
            config_status = {
                "providers_available": len(providers),
                "providers": providers,
                "config_loaded": True,
            }
        except Exception as e:
            config_status = {"config_loaded": False, "error": str(e)}
            warning_messages.append(f"Configuration system error: {e}")

    return {
        "version": __version__,
        "modules": modules,
        "config_status": config_status,
        "issues": issues,
        "warnings": warning_messages,
        "status": (
            "healthy"
            if not issues
            else ("degraded" if not warning_messages else "partial")
        ),
    }


def quick_diagnostic() -> None:
    """Print a quick diagnostic of the installation."""
    info = check_installation()

    print(f"🤖 ChukLLM v{info['version']} - {info['status'].upper()}")
    print("\n📦 Modules:")
    for module, available in info["modules"].items():
        status = "✅" if available else "❌"
        print(f"   {status} {module}")

    if info["config_status"].get("config_loaded"):
        providers = info["config_status"]["providers"]
        print(f"\n⚙️  Configuration: {len(providers)} providers available")
        for provider in providers[:3]:  # Show first 3
            print(f"   • {provider}")
        if len(providers) > 3:
            print(f"   • ... and {len(providers) - 3} more")

    if info["issues"]:
        print("\n⚠️  Issues:")
        for issue in info["issues"]:
            print(f"   • {issue}")

    if info["warnings"]:
        print("\n💡 Warnings:")
        for warning in info["warnings"]:
            print(f"   • {warning}")


# Export main API functions
__all__ = [
    "__version__",
    "get_version",
    "get_available_modules",
    "check_installation",
    "quick_diagnostic",
    # Unified configuration system
    "get_config",
    "reset_config",
    "ConfigManager",
    "ConfigValidator",
    "CapabilityChecker",
    "Feature",
    # Backward compatibility (duplicates removed)
    # LLM clients
    "get_client",
    "BaseLLMClient",
    "validate_provider_setup",
    "list_available_providers",
]

# Add enhanced features if available
if _features_available:
    __all__.extend(
        [
            "UnifiedLLMInterface",
            "ProviderAdapter",
            "quick_chat",
            "multi_provider_chat",
            "find_best_provider_for_task",
        ]
    )

# Add API functions if available
if _api_available:
    __all__.extend(
        [
            "ask",
            "stream",
            "ask_sync",
            "stream_sync",
            "configure",
            "get_current_config",
            "api_get_client",
            "api_reset_config",
            "compare_providers",
            "quick_question",
            # Enhanced API functions
            "ask_json",
            "quick_ask",
            "multi_provider_ask",
            "validate_request",
        ]
    )

if _conversation_available:
    __all__.extend(["conversation", "ConversationContext"])

if _utils_available:
    __all__.extend(
        [
            "health_check",
            "test_connection",
            "print_diagnostics",
            # Enhanced config utilities
            "debug_config_state",
            "quick_setup",
            "list_available_setups",
            "switch_provider",
            "auto_configure",
            "validate_config",
            "get_capabilities",
            "supports_feature",
        ]
    )


# Convenience functions for easy setup
def setup_provider(provider: str, model: str | None = None, **kwargs) -> bool:
    """Quick setup function for easy provider configuration."""
    if not _utils_available:
        logger.error("Utils module not available for setup")
        return False

    try:
        return quick_setup(provider, model, **kwargs)
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return False


def auto_setup_for_task(task_type: str = "general", **requirements) -> bool:
    """Automatically configure the best provider for a task type."""
    if not _utils_available:
        logger.error("Utils module not available for auto setup")
        return False

    try:
        return auto_configure(task_type, **requirements)
    except Exception as e:
        logger.error(f"Auto setup failed: {e}")
        return False


# Add convenience functions to exports
__all__.extend(["setup_provider", "auto_setup_for_task"])
