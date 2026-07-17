# chuk_llm/llm/providers/llamacpp_server.py
"""
llama.cpp server process manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Async-native manager for llama-server subprocess lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class LlamaCppServerConfig(BaseModel):
    """Configuration for llama.cpp server."""

    model_path: Path = Field(..., description="Path to GGUF model file")
    host: str = Field(default="127.0.0.1", description="Host to bind to")
    port: int = Field(default=8033, description="Port to bind to")
    ctx_size: int = Field(default=8192, description="Context size")
    n_gpu_layers: int = Field(default=-1, description="GPU layers (-1 = all)")
    server_binary: str | None = Field(
        default=None, description="Path to llama-server binary"
    )
    timeout: float = Field(
        default=120.0, description="Server startup timeout (including model loading)"
    )
    extra_args: list[str] = Field(default_factory=list, description="Additional args")


class LlamaCppServerManager:
    """
    Async-native manager for llama-server process.

    Handles:
    - Server process lifecycle (start/stop)
    - Health checking
    - Automatic cleanup
    """

    def __init__(self, config: LlamaCppServerConfig):
        """Initialize server manager with configuration."""
        self.config = config
        self.process: asyncio.subprocess.Process | None = None
        self._base_url = f"http://{config.host}:{config.port}"

    @property
    def base_url(self) -> str:
        """Get the base URL for the server."""
        return self._base_url

    async def _find_server_binary(self) -> str:
        """Find llama-server binary in PATH or common locations."""
        import platform

        if self.config.server_binary:
            if Path(self.config.server_binary).exists():
                return str(self.config.server_binary)
            raise FileNotFoundError(
                f"Specified llama-server binary not found: {self.config.server_binary}"
            )

        # Determine binary name based on platform
        system = platform.system()
        binary_name = "llama-server.exe" if system == "Windows" else "llama-server"

        # Try to find in PATH
        if server_path := shutil.which(binary_name):
            log.debug(f"Found llama-server in PATH: {server_path}")
            return server_path

        # Try common installation paths (platform-specific)
        common_paths = []
        if system == "Windows":
            # Windows paths
            common_paths = [
                Path.home() / "llama.cpp" / "build" / "bin" / "Release" / binary_name,
                Path.home() / "llama.cpp" / "build" / "Release" / binary_name,
                Path("C:/Program Files/llama.cpp") / binary_name,
                Path("C:/llama.cpp") / binary_name,
            ]
        elif system == "Linux":
            # Linux paths
            common_paths = [
                Path.home() / "llama.cpp" / "build" / "bin" / binary_name,
                Path("/usr/local/bin") / binary_name,
                Path("/usr/bin") / binary_name,
                Path("/opt/llama.cpp/bin") / binary_name,
            ]
        elif system == "Darwin":
            # macOS paths
            common_paths = [
                Path.home() / "llama.cpp" / "build" / "bin" / binary_name,
                Path("/usr/local/bin") / binary_name,
                Path("/opt/homebrew/bin") / binary_name,
                Path("/opt/llama.cpp/bin") / binary_name,
            ]
        else:
            # Fallback for other Unix-like systems
            common_paths = [
                Path.home() / "llama.cpp" / "build" / "bin" / binary_name,
                Path("/usr/local/bin") / binary_name,
            ]

        for path in common_paths:
            if path.exists():
                log.debug(f"Found llama-server at: {path}")
                return str(path)

        raise FileNotFoundError(
            f"llama-server not found. Install llama.cpp and ensure {binary_name} is in PATH.\n"
            f"For installation instructions, see: https://github.com/ggerganov/llama.cpp"
        )

    async def _build_command(self) -> list[str]:
        """Build command line arguments for llama-server."""
        server_binary = await self._find_server_binary()

        cmd = [
            server_binary,
            "-m",
            str(self.config.model_path),
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--ctx-size",
            str(self.config.ctx_size),
            "-ngl",
            str(self.config.n_gpu_layers),
            "--jinja",  # Enable jinja template support
            # Note: -fa (function calling aware) is intentionally not included by default
            # as it requires model-specific template support. Add via extra_args if needed.
        ]

        # Add extra args
        cmd.extend(self.config.extra_args)

        return cmd

    async def start(self) -> None:
        """Start the llama-server process."""
        if self.process is not None:
            log.warning("Server already running")
            return

        # Check if model exists
        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.config.model_path}")

        # Build command
        cmd = await self._build_command()
        log.info(f"Starting llama-server: {' '.join(cmd)}")

        # Start process
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Wait for server to be ready
        try:
            await self._wait_for_health(timeout=self.config.timeout)
            log.info(f"llama-server ready at {self._base_url}")
        except TimeoutError as e:
            # Server didn't start in time, clean up
            await self.stop()
            raise RuntimeError(
                f"llama-server failed to start within {self.config.timeout}s"
            ) from e

    async def _wait_for_health(self, timeout: float = 30.0) -> None:
        """Wait for server to become healthy and model to be loaded."""
        start_time = asyncio.get_event_loop().time()

        async with httpx.AsyncClient() as client:
            # First wait for server to respond
            while (asyncio.get_event_loop().time() - start_time) < timeout:
                try:
                    # Use /v1/models instead of /health (not all versions have /health)
                    response = await client.get(
                        f"{self._base_url}/v1/models", timeout=2.0
                    )
                    if response.status_code == 200:
                        break
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass

                # Check if process died
                if self.process and self.process.returncode is not None:
                    raise RuntimeError(
                        f"llama-server process died with code {self.process.returncode}"
                    )

                await asyncio.sleep(0.5)
            else:
                raise TimeoutError(f"Server startup timed out after {timeout}s")

            # Now wait for model to actually load by testing a completion
            log.debug("Server responding, waiting for model to load...")

            # Get the model name from /v1/models to use in health check
            try:
                models_response = await client.get(
                    f"{self._base_url}/v1/models", timeout=2.0
                )
                if models_response.status_code == 200:
                    models_data = models_response.json()
                    model_name = (
                        models_data["data"][0]["id"]
                        if models_data.get("data")
                        else str(self.config.model_path)
                    )
                else:
                    model_name = str(self.config.model_path)
            except Exception:
                model_name = str(self.config.model_path)

            while (asyncio.get_event_loop().time() - start_time) < timeout:
                try:
                    # Try a simple completion to ensure model is loaded
                    test_response = await client.post(
                        f"{self._base_url}/v1/chat/completions",
                        json={
                            "model": model_name,
                            "messages": [{"role": "user", "content": "test"}],
                            "max_tokens": 1,
                        },
                        timeout=5.0,
                    )
                    if test_response.status_code == 200:
                        log.debug("Model loaded and ready")
                        return
                    elif test_response.status_code == 503:
                        # 503 means model is still loading, keep waiting
                        log.debug("Model still loading (503), waiting...")
                    else:
                        # Some other error (400, etc.) - log it but keep waiting
                        # The model might still be initializing
                        log.debug(
                            f"Got response code {test_response.status_code} during startup, "
                            "continuing to wait for 200..."
                        )
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass

                # Check if process died
                if self.process and self.process.returncode is not None:
                    raise RuntimeError(
                        f"llama-server process died with code {self.process.returncode}"
                    )

                await asyncio.sleep(1.0)

        raise TimeoutError(f"Model loading timed out after {timeout}s")

    async def stop(self) -> None:
        """Stop the llama-server process."""
        if self.process is None:
            return

        # Check if process already exited
        if self.process.returncode is not None:
            log.debug(f"Process already exited with code {self.process.returncode}")
            self.process = None
            return

        log.info("Stopping llama-server...")

        try:
            # Try graceful shutdown first
            self.process.terminate()

            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except TimeoutError:
                # Force kill if graceful shutdown fails
                log.warning("Graceful shutdown failed, force killing...")
                self.process.kill()
                await self.process.wait()
        except ProcessLookupError:
            # Process already died - this is fine
            log.debug("Process already terminated")
        except Exception as e:
            log.debug(f"Error stopping server: {e}")

        self.process = None
        log.info("llama-server stopped")

    async def is_healthy(self) -> bool:
        """Check if server is healthy."""
        if self.process is None or self.process.returncode is not None:
            return False

        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                # Use /v1/models instead of /health (not all versions have /health)
                response = await client.get(f"{self._base_url}/v1/models")
                return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
