"""Tests for llama.cpp server manager."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from chuk_llm.llm.providers.llamacpp_server import (
    LlamaCppServerConfig,
    LlamaCppServerManager,
)


class TestLlamaCppServerConfig:
    """Test LlamaCppServerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LlamaCppServerConfig(model_path=Path("/tmp/model.gguf"))

        assert config.model_path == Path("/tmp/model.gguf")
        assert config.host == "127.0.0.1"
        assert config.port == 8033
        assert config.ctx_size == 8192
        assert config.n_gpu_layers == -1
        assert config.server_binary is None
        assert config.timeout == 120.0
        assert config.extra_args == []

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LlamaCppServerConfig(
            model_path=Path("/models/llama.gguf"),
            host="0.0.0.0",
            port=9000,
            ctx_size=16384,
            n_gpu_layers=32,
            server_binary="/usr/local/bin/llama-server",
            timeout=60.0,
            extra_args=["--flash-attn"],
        )

        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.ctx_size == 16384
        assert config.n_gpu_layers == 32
        assert config.server_binary == "/usr/local/bin/llama-server"
        assert config.timeout == 60.0
        assert config.extra_args == ["--flash-attn"]


class TestLlamaCppServerManager:
    """Test LlamaCppServerManager."""

    def test_initialization(self):
        """Test manager initialization."""
        config = LlamaCppServerConfig(model_path=Path("/tmp/model.gguf"))
        manager = LlamaCppServerManager(config)

        assert manager.config == config
        assert manager.process is None
        assert manager.base_url == "http://127.0.0.1:8033"

    def test_base_url(self):
        """Test base URL generation."""
        config = LlamaCppServerConfig(
            model_path=Path("/tmp/model.gguf"),
            host="localhost",
            port=9000,
        )
        manager = LlamaCppServerManager(config)

        assert manager.base_url == "http://localhost:9000"

    @pytest.mark.asyncio
    async def test_find_server_binary_in_config(self):
        """Test finding server binary from config."""
        with patch("pathlib.Path.exists", return_value=True):
            config = LlamaCppServerConfig(
                model_path=Path("/tmp/model.gguf"),
                server_binary="/custom/llama-server",
            )
            manager = LlamaCppServerManager(config)

            binary = await manager._find_server_binary()
            assert binary == "/custom/llama-server"

    @pytest.mark.asyncio
    async def test_find_server_binary_not_found(self):
        """Test error when server binary not found."""
        with (
            patch("shutil.which", return_value=None),
            patch("pathlib.Path.exists", return_value=False),
        ):
            config = LlamaCppServerConfig(model_path=Path("/tmp/model.gguf"))
            manager = LlamaCppServerManager(config)

            with pytest.raises(FileNotFoundError, match="llama-server not found"):
                await manager._find_server_binary()

    @pytest.mark.asyncio
    async def test_build_command(self):
        """Test command building."""
        with patch("shutil.which", return_value="/usr/bin/llama-server"):
            config = LlamaCppServerConfig(
                model_path=Path("/models/llama.gguf"),
                host="0.0.0.0",
                port=9000,
                ctx_size=16384,
                n_gpu_layers=32,
                extra_args=["--flash-attn", "--cont-batching"],
            )
            manager = LlamaCppServerManager(config)

            cmd = await manager._build_command()

            assert cmd[0] == "/usr/bin/llama-server"
            assert "-m" in cmd
            # Convert path to string and normalize for cross-platform compatibility
            model_path_str = str(Path("/models/llama.gguf"))
            assert model_path_str in cmd
            assert "--host" in cmd
            assert "0.0.0.0" in cmd
            assert "--port" in cmd
            assert "9000" in cmd
            assert "--ctx-size" in cmd
            assert "16384" in cmd
            assert "-ngl" in cmd
            assert "32" in cmd
            assert "--flash-attn" in cmd
            assert "--cont-batching" in cmd

    @pytest.mark.asyncio
    async def test_wait_for_health_timeout(self):
        """Test health check timeout."""
        config = LlamaCppServerConfig(model_path=Path("/tmp/model.gguf"))
        manager = LlamaCppServerManager(config)

        # Mock a process that never becomes healthy
        mock_process = MagicMock()
        mock_process.returncode = None
        manager.process = mock_process

        # Mock the HTTP client to avoid actual network calls
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            with pytest.raises(TimeoutError, match="Server startup timed out"):
                await manager._wait_for_health(timeout=0.1)

    @pytest.mark.asyncio
    async def test_wait_for_health_process_died(self):
        """Test health check when process dies."""
        config = LlamaCppServerConfig(model_path=Path("/tmp/model.gguf"))
        manager = LlamaCppServerManager(config)

        # Mock a process that died
        mock_process = MagicMock()
        mock_process.returncode = 1
        manager.process = mock_process

        # Mock the HTTP client to avoid actual network calls
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            with pytest.raises(RuntimeError, match="process died with code 1"):
                await manager._wait_for_health(timeout=1.0)

    @pytest.mark.asyncio
    async def test_is_healthy_no_process(self):
        """Test is_healthy when no process running."""
        config = LlamaCppServerConfig(model_path=Path("/tmp/model.gguf"))
        manager = LlamaCppServerManager(config)

        assert await manager.is_healthy() is False

    @pytest.mark.asyncio
    async def test_is_healthy_process_dead(self):
        """Test is_healthy when process is dead."""
        config = LlamaCppServerConfig(model_path=Path("/tmp/model.gguf"))
        manager = LlamaCppServerManager(config)

        mock_process = MagicMock()
        mock_process.returncode = 1
        manager.process = mock_process

        assert await manager.is_healthy() is False

    @pytest.mark.asyncio
    async def test_is_healthy_success(self):
        """Test is_healthy when server is healthy."""
        config = LlamaCppServerConfig(model_path=Path("/tmp/model.gguf"))
        manager = LlamaCppServerManager(config)

        mock_process = MagicMock()
        mock_process.returncode = None
        manager.process = mock_process

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            assert await manager.is_healthy() is True

    @pytest.mark.asyncio
    async def test_stop_no_process(self):
        """Test stop when no process running."""
        config = LlamaCppServerConfig(model_path=Path("/tmp/model.gguf"))
        manager = LlamaCppServerManager(config)

        # Should not raise
        await manager.stop()
        assert manager.process is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        config = LlamaCppServerConfig(
            model_path=Path("/tmp/model.gguf"),
            timeout=0.1,
        )

        # Mock the server startup to avoid actual process creation
        with (
            patch.object(
                LlamaCppServerManager, "start", new_callable=AsyncMock
            ) as mock_start,
            patch.object(
                LlamaCppServerManager, "stop", new_callable=AsyncMock
            ) as mock_stop,
        ):
            async with LlamaCppServerManager(config) as manager:
                assert manager is not None

            mock_start.assert_called_once()
            mock_stop.assert_called_once()
