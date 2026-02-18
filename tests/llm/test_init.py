"""Comprehensive tests for llm/__init__.py module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Any


class TestModuleImports:
    """Tests for module import handling."""

    def test_version_info(self):
        """Test version information is available."""
        from chuk_llm.llm import __version__

        assert isinstance(__version__, str)

    def test_get_version(self):
        """Test get_version function."""
        from chuk_llm.llm import get_version, __version__

        assert get_version() == __version__

    def test_get_available_modules(self):
        """Test get_available_modules returns module status."""
        from chuk_llm.llm import get_available_modules

        modules = get_available_modules()
        assert isinstance(modules, dict)
        assert "config" in modules
        assert "llm" in modules
        assert "features" in modules
        assert "api" in modules
        assert "conversation" in modules
        assert "utils" in modules

        # All values should be boolean
        for key, value in modules.items():
            assert isinstance(value, bool), f"{key} should be boolean"


class TestCheckInstallation:
    """Tests for check_installation function."""

    def test_check_installation_basic(self):
        """Test check_installation returns proper structure."""
        from chuk_llm.llm import check_installation

        info = check_installation()

        assert isinstance(info, dict)
        assert "version" in info
        assert "modules" in info
        assert "config_status" in info
        assert "issues" in info
        assert "warnings" in info
        assert "status" in info

    def test_check_installation_status_values(self):
        """Test that status has valid values."""
        from chuk_llm.llm import check_installation

        info = check_installation()
        assert info["status"] in ["healthy", "degraded", "partial"]

    def test_check_installation_issues_list(self):
        """Test issues is a list."""
        from chuk_llm.llm import check_installation

        info = check_installation()
        assert isinstance(info["issues"], list)
        assert isinstance(info["warnings"], list)

    def test_check_installation_with_config_available(self):
        """Test check_installation when config is available."""
        from chuk_llm.llm import check_installation

        info = check_installation()

        # If config is available, check config_status
        if info["modules"]["config"]:
            assert "providers_available" in info["config_status"] or "error" in info["config_status"]

    def test_check_installation_with_config_error(self):
        """Test check_installation handles config errors."""
        from chuk_llm.llm import check_installation

        with patch('chuk_llm.llm.get_config') as mock_get_config:
            mock_get_config.side_effect = Exception("Config error")

            info = check_installation()

            if info["modules"]["config"]:
                assert info["config_status"].get("config_loaded") is False
                assert "error" in info["config_status"]


class TestQuickDiagnostic:
    """Tests for quick_diagnostic function."""

    def test_quick_diagnostic_runs(self, capsys):
        """Test quick_diagnostic runs without error."""
        from chuk_llm.llm import quick_diagnostic

        quick_diagnostic()

        captured = capsys.readouterr()
        assert "ChukLLM" in captured.out
        assert "Modules:" in captured.out

    def test_quick_diagnostic_shows_version(self, capsys):
        """Test quick_diagnostic shows version."""
        from chuk_llm.llm import quick_diagnostic, __version__

        quick_diagnostic()

        captured = capsys.readouterr()
        assert __version__ in captured.out

    def test_quick_diagnostic_shows_status(self, capsys):
        """Test quick_diagnostic shows status."""
        from chuk_llm.llm import quick_diagnostic

        quick_diagnostic()

        captured = capsys.readouterr()
        # Should show one of the status values
        assert any(status in captured.out for status in ["HEALTHY", "DEGRADED", "PARTIAL"])

    def test_quick_diagnostic_shows_modules(self, capsys):
        """Test quick_diagnostic lists modules."""
        from chuk_llm.llm import quick_diagnostic

        quick_diagnostic()

        captured = capsys.readouterr()
        # Should show module status indicators
        assert "config" in captured.out
        assert "llm" in captured.out

    def test_quick_diagnostic_shows_issues_if_present(self, capsys):
        """Test quick_diagnostic shows issues section if present."""
        from chuk_llm.llm import quick_diagnostic

        with patch('chuk_llm.llm.check_installation') as mock_check:
            mock_check.return_value = {
                "version": "0.1.0",
                "status": "degraded",
                "modules": {"config": True, "llm": False},
                "config_status": {},
                "issues": ["Test issue"],
                "warnings": []
            }

            quick_diagnostic()

            captured = capsys.readouterr()
            assert "Issues:" in captured.out
            assert "Test issue" in captured.out

    def test_quick_diagnostic_shows_warnings_if_present(self, capsys):
        """Test quick_diagnostic shows warnings section if present."""
        from chuk_llm.llm import quick_diagnostic

        with patch('chuk_llm.llm.check_installation') as mock_check:
            mock_check.return_value = {
                "version": "0.1.0",
                "status": "healthy",
                "modules": {"config": True, "llm": True},
                "config_status": {},
                "issues": [],
                "warnings": ["Test warning"]
            }

            quick_diagnostic()

            captured = capsys.readouterr()
            assert "Warnings:" in captured.out
            assert "Test warning" in captured.out

    def test_quick_diagnostic_shows_providers(self, capsys):
        """Test quick_diagnostic shows providers if config loaded."""
        from chuk_llm.llm import quick_diagnostic

        with patch('chuk_llm.llm.check_installation') as mock_check:
            mock_check.return_value = {
                "version": "0.1.0",
                "status": "healthy",
                "modules": {"config": True, "llm": True},
                "config_status": {
                    "config_loaded": True,
                    "providers_available": 5,
                    "providers": ["openai", "anthropic", "google", "groq", "mistral"]
                },
                "issues": [],
                "warnings": []
            }

            quick_diagnostic()

            captured = capsys.readouterr()
            assert "Configuration:" in captured.out
            assert "5 providers" in captured.out
            assert "openai" in captured.out


class TestSetupProvider:
    """Tests for setup_provider convenience function."""

    def test_setup_provider_success(self):
        """Test setup_provider with successful setup."""
        from chuk_llm.llm import setup_provider

        with patch('chuk_llm.llm._utils_available', True), \
             patch('chuk_llm.llm.quick_setup') as mock_quick_setup:

            mock_quick_setup.return_value = True

            result = setup_provider("openai", "gpt-4")
            assert result is True
            mock_quick_setup.assert_called_once_with("openai", "gpt-4")

    def test_setup_provider_with_kwargs(self):
        """Test setup_provider passes kwargs."""
        from chuk_llm.llm import setup_provider

        with patch('chuk_llm.llm._utils_available', True), \
             patch('chuk_llm.llm.quick_setup') as mock_quick_setup:

            mock_quick_setup.return_value = True

            result = setup_provider("openai", "gpt-4", api_key="test-key")
            assert result is True
            mock_quick_setup.assert_called_once_with("openai", "gpt-4", api_key="test-key")

    def test_setup_provider_utils_unavailable(self):
        """Test setup_provider when utils not available."""
        from chuk_llm.llm import setup_provider

        with patch('chuk_llm.llm._utils_available', False):
            result = setup_provider("openai")
            assert result is False

    def test_setup_provider_with_error(self):
        """Test setup_provider handles errors."""
        from chuk_llm.llm import setup_provider

        with patch('chuk_llm.llm._utils_available', True), \
             patch('chuk_llm.llm.quick_setup') as mock_quick_setup:

            mock_quick_setup.side_effect = Exception("Setup failed")

            result = setup_provider("openai", "gpt-4")
            assert result is False


class TestAutoSetupForTask:
    """Tests for auto_setup_for_task convenience function."""

    def test_auto_setup_for_task_success(self):
        """Test auto_setup_for_task with successful setup."""
        from chuk_llm.llm import auto_setup_for_task

        with patch('chuk_llm.llm._utils_available', True), \
             patch('chuk_llm.llm.auto_configure') as mock_auto_configure:

            mock_auto_configure.return_value = True

            result = auto_setup_for_task("general")
            assert result is True
            mock_auto_configure.assert_called_once_with("general")

    def test_auto_setup_for_task_with_requirements(self):
        """Test auto_setup_for_task with requirements."""
        from chuk_llm.llm import auto_setup_for_task

        with patch('chuk_llm.llm._utils_available', True), \
             patch('chuk_llm.llm.auto_configure') as mock_auto_configure:

            mock_auto_configure.return_value = True

            result = auto_setup_for_task("vision", requires_tools=True)
            assert result is True
            mock_auto_configure.assert_called_once_with("vision", requires_tools=True)

    def test_auto_setup_for_task_utils_unavailable(self):
        """Test auto_setup_for_task when utils not available."""
        from chuk_llm.llm import auto_setup_for_task

        with patch('chuk_llm.llm._utils_available', False):
            result = auto_setup_for_task("general")
            assert result is False

    def test_auto_setup_for_task_with_error(self):
        """Test auto_setup_for_task handles errors."""
        from chuk_llm.llm import auto_setup_for_task

        with patch('chuk_llm.llm._utils_available', True), \
             patch('chuk_llm.llm.auto_configure') as mock_auto_configure:

            mock_auto_configure.side_effect = Exception("Auto configure failed")

            result = auto_setup_for_task("general")
            assert result is False


class TestFallbackImplementations:
    """Tests for fallback implementations when modules aren't available."""

    def test_fallback_get_config_raises(self):
        """Test that fallback get_config raises ImportError."""
        # We need to test the fallback path when config is not available
        # This is hard to test in practice since config is usually available
        # Just verify the fallback exists
        from chuk_llm.llm import _config_available

        # If config is available, this test is not applicable
        if not _config_available:
            from chuk_llm.llm import get_config
            with pytest.raises(ImportError):
                get_config()

    def test_fallback_reset_config_raises(self):
        """Test that fallback reset_config raises ImportError."""
        from chuk_llm.llm import _config_available

        if not _config_available:
            from chuk_llm.llm import reset_config
            with pytest.raises(ImportError):
                reset_config()

    def test_fallback_get_client_raises(self):
        """Test that fallback get_client raises ImportError."""
        from chuk_llm.llm import _llm_available

        if not _llm_available:
            from chuk_llm.llm import get_client
            with pytest.raises(ImportError):
                get_client("openai")


class TestExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Test __all__ exports exist."""
        from chuk_llm import llm

        # Check __all__ is defined
        assert hasattr(llm, '__all__')
        assert isinstance(llm.__all__, list)

        # Check core exports
        assert "get_version" in llm.__all__
        assert "get_available_modules" in llm.__all__
        assert "check_installation" in llm.__all__
        assert "quick_diagnostic" in llm.__all__

        # Check all exported items exist
        for name in llm.__all__:
            assert hasattr(llm, name), f"Exported name '{name}' not found in module"

    def test_version_exports(self):
        """Test version exports."""
        from chuk_llm.llm import __version__, get_version

        assert __version__ is not None
        assert callable(get_version)

    def test_diagnostic_exports(self):
        """Test diagnostic function exports."""
        from chuk_llm.llm import check_installation, quick_diagnostic, get_available_modules

        assert callable(check_installation)
        assert callable(quick_diagnostic)
        assert callable(get_available_modules)

    def test_convenience_exports(self):
        """Test convenience function exports."""
        from chuk_llm.llm import setup_provider, auto_setup_for_task

        assert callable(setup_provider)
        assert callable(auto_setup_for_task)

    def test_config_exports(self):
        """Test config exports are available."""
        from chuk_llm import llm

        # These should always be exported, even if fallbacks
        assert hasattr(llm, 'get_config')
        assert hasattr(llm, 'reset_config')
        assert hasattr(llm, 'ConfigManager')
        assert hasattr(llm, 'Feature')

    def test_client_exports(self):
        """Test client exports are available."""
        from chuk_llm import llm

        # These should always be exported, even if fallbacks
        assert hasattr(llm, 'get_client')
        assert hasattr(llm, 'BaseLLMClient')

    def test_features_exports_when_available(self):
        """Test features exports when module is available."""
        from chuk_llm.llm import _features_available

        if _features_available:
            from chuk_llm import llm
            assert hasattr(llm, 'UnifiedLLMInterface')
            assert hasattr(llm, 'ProviderAdapter')
            assert hasattr(llm, 'quick_chat')

    def test_api_exports_when_available(self):
        """Test API exports when module is available."""
        from chuk_llm.llm import _api_available

        if _api_available:
            from chuk_llm import llm
            assert hasattr(llm, 'ask')
            assert hasattr(llm, 'stream')
            assert hasattr(llm, 'configure')

    def test_utils_exports_when_available(self):
        """Test utils exports when module is available."""
        from chuk_llm.llm import _utils_available

        if _utils_available:
            from chuk_llm import llm
            assert hasattr(llm, 'health_check')
            assert hasattr(llm, 'test_connection')


class TestModuleAvailabilityFlags:
    """Tests for module availability flags."""

    def test_config_available_flag(self):
        """Test _config_available flag."""
        from chuk_llm.llm import _config_available

        assert isinstance(_config_available, bool)

    def test_llm_available_flag(self):
        """Test _llm_available flag."""
        from chuk_llm.llm import _llm_available

        assert isinstance(_llm_available, bool)

    def test_features_available_flag(self):
        """Test _features_available flag."""
        from chuk_llm.llm import _features_available

        assert isinstance(_features_available, bool)

    def test_api_available_flag(self):
        """Test _api_available flag."""
        from chuk_llm.llm import _api_available

        assert isinstance(_api_available, bool)

    def test_conversation_available_flag(self):
        """Test _conversation_available flag."""
        from chuk_llm.llm import _conversation_available

        assert isinstance(_conversation_available, bool)

    def test_utils_available_flag(self):
        """Test _utils_available flag."""
        from chuk_llm.llm import _utils_available

        assert isinstance(_utils_available, bool)
