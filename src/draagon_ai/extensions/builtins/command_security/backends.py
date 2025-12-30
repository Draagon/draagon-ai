"""Pluggable backends for command execution.

This module provides different backends for executing shell commands:
- HTTPCommandBackend: Execute via HTTP API (default, secure)
- LocalCommandBackend: Execute locally via subprocess
- SSHCommandBackend: Execute on remote hosts via SSH

Backends are swappable - the extension works with any backend
that implements the CommandBackend protocol.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result from executing a command.

    Attributes:
        success: Whether command completed successfully
        output: Combined stdout/stderr output
        exit_code: Process exit code (0 = success)
        host: Host where command was executed
        elapsed_ms: Execution time in milliseconds
        error: Error message if execution failed
    """

    success: bool
    output: str
    exit_code: int = 0
    host: str = "local"
    elapsed_ms: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "exit_code": self.exit_code,
            "host": self.host,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }


class CommandBackend(ABC):
    """Abstract base class for command execution backends.

    Implementations must provide:
    - execute(): Execute a command and return result
    - is_available(): Check if backend is operational

    Example:
        class MyBackend(CommandBackend):
            async def execute(
                self, command: str, host: str, timeout: int
            ) -> ExecutionResult:
                # Implementation here
                pass

            async def is_available(self) -> bool:
                return True
    """

    @abstractmethod
    async def execute(
        self,
        command: str,
        host: str = "local",
        timeout: int = 30,
    ) -> ExecutionResult:
        """Execute a command.

        Args:
            command: Shell command to execute
            host: Target host (interpretation depends on backend)
            timeout: Execution timeout in seconds

        Returns:
            ExecutionResult with output and status
        """
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the backend is available.

        Returns:
            True if backend can execute commands
        """
        ...


class HTTPCommandBackend(CommandBackend):
    """Execute commands via HTTP API.

    This is the default backend. Commands are sent to a command execution
    API that handles the actual subprocess management. This provides:
    - Security isolation (API can run on different machine)
    - Audit logging at the API level
    - Rate limiting and access control

    The API should accept POST /execute with:
    - command: string
    - host: string
    - timeout: int

    And return:
    - stdout: string
    - stderr: string
    - returncode: int
    - host: string

    Example:
        backend = HTTPCommandBackend(
            base_url="http://localhost:5555",
            token="secret-token",
        )

        result = await backend.execute("docker ps", host="local")
    """

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        timeout: int = 30,
    ) -> None:
        """Initialize HTTP backend.

        Args:
            base_url: Base URL of command API (e.g., http://localhost:5555)
            token: Optional Bearer token for authentication
            timeout: Default request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.default_timeout = timeout

    async def execute(
        self,
        command: str,
        host: str = "local",
        timeout: int | None = None,
    ) -> ExecutionResult:
        """Execute command via HTTP API.

        Args:
            command: Shell command to execute
            host: Target host for the command
            timeout: Request timeout in seconds

        Returns:
            ExecutionResult with output and status
        """
        try:
            import aiohttp
        except ImportError:
            return ExecutionResult(
                success=False,
                output="",
                error="aiohttp not installed - required for HTTP backend",
            )

        start_time = time.time()
        timeout_val = timeout or self.default_timeout

        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/execute",
                    json={
                        "command": command,
                        "host": host,
                        "timeout": timeout_val,
                    },
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout_val + 5),
                ) as resp:
                    elapsed_ms = int((time.time() - start_time) * 1000)

                    if resp.status != 200:
                        error_text = await resp.text()
                        return ExecutionResult(
                            success=False,
                            output="",
                            error=f"API error {resp.status}: {error_text}",
                            elapsed_ms=elapsed_ms,
                            host=host,
                        )

                    data = await resp.json()

                    output_parts = []
                    if data.get("stdout"):
                        output_parts.append(data["stdout"].strip())
                    if data.get("stderr"):
                        output_parts.append(f"Errors: {data['stderr'].strip()}")
                    if data.get("returncode", 0) != 0:
                        output_parts.append(f"Exit code: {data['returncode']}")

                    return ExecutionResult(
                        success=data.get("returncode", 1) == 0,
                        output="\n".join(output_parts)
                        or "Command completed with no output.",
                        exit_code=data.get("returncode", 0),
                        host=data.get("host", host),
                        elapsed_ms=elapsed_ms,
                    )

        except asyncio.TimeoutError:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                output="",
                error=f"Request timed out after {timeout_val}s",
                elapsed_ms=elapsed_ms,
                host=host,
            )
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)

            if "ECONNREFUSED" in error_msg or "Connection" in error_msg:
                error_msg = "Could not connect to command API. Is the service running?"

            return ExecutionResult(
                success=False,
                output="",
                error=error_msg,
                elapsed_ms=elapsed_ms,
                host=host,
            )

    async def is_available(self) -> bool:
        """Check if the HTTP API is reachable."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    return resp.status in (200, 204)
        except Exception:
            return False


class LocalCommandBackend(CommandBackend):
    """Execute commands locally via subprocess.

    This backend runs commands directly on the local machine.
    Use with caution - the security classification should prevent
    dangerous commands from reaching here.

    Example:
        backend = LocalCommandBackend()
        result = await backend.execute("ls -la")
    """

    def __init__(
        self,
        shell: str = "/bin/bash",
        allowed_hosts: list[str] | None = None,
    ) -> None:
        """Initialize local backend.

        Args:
            shell: Shell to use for execution
            allowed_hosts: List of allowed host values (default: ["local"])
        """
        self.shell = shell
        self.allowed_hosts = allowed_hosts or ["local"]

    async def execute(
        self,
        command: str,
        host: str = "local",
        timeout: int = 30,
    ) -> ExecutionResult:
        """Execute command locally via subprocess.

        Args:
            command: Shell command to execute
            host: Must be in allowed_hosts
            timeout: Execution timeout in seconds

        Returns:
            ExecutionResult with output and status
        """
        if host not in self.allowed_hosts:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Host '{host}' not in allowed hosts: {self.allowed_hosts}",
                host=host,
            )

        start_time = time.time()

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                executable=self.shell,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                elapsed_ms = int((time.time() - start_time) * 1000)
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Command timed out after {timeout}s",
                    elapsed_ms=elapsed_ms,
                    host=host,
                )

            elapsed_ms = int((time.time() - start_time) * 1000)

            output_parts = []
            if stdout:
                output_parts.append(stdout.decode().strip())
            if stderr:
                output_parts.append(f"Errors: {stderr.decode().strip()}")
            if proc.returncode != 0:
                output_parts.append(f"Exit code: {proc.returncode}")

            return ExecutionResult(
                success=proc.returncode == 0,
                output="\n".join(output_parts) or "Command completed with no output.",
                exit_code=proc.returncode or 0,
                host=host,
                elapsed_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                elapsed_ms=elapsed_ms,
                host=host,
            )

    async def is_available(self) -> bool:
        """Local backend is always available."""
        return True


class SSHCommandBackend(CommandBackend):
    """Execute commands on remote hosts via SSH.

    This backend uses SSH to run commands on remote machines.
    Requires SSH key authentication to be set up.

    Example:
        backend = SSHCommandBackend(
            hosts={
                "server1": "user@192.168.1.100",
                "server2": "root@192.168.1.101",
            }
        )
        result = await backend.execute("docker ps", host="server1")
    """

    def __init__(
        self,
        hosts: dict[str, str],
        ssh_options: str = "-o StrictHostKeyChecking=accept-new",
    ) -> None:
        """Initialize SSH backend.

        Args:
            hosts: Mapping of host names to SSH destinations
                   e.g., {"beelink": "root@192.168.1.100"}
            ssh_options: Additional SSH options
        """
        self.hosts = hosts
        self.ssh_options = ssh_options

    async def execute(
        self,
        command: str,
        host: str = "local",
        timeout: int = 30,
    ) -> ExecutionResult:
        """Execute command via SSH.

        Args:
            command: Shell command to execute
            host: Host name from the hosts mapping
            timeout: SSH command timeout in seconds

        Returns:
            ExecutionResult with output and status
        """
        if host not in self.hosts:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Unknown host '{host}'. Available: {list(self.hosts.keys())}",
                host=host,
            )

        destination = self.hosts[host]
        # Escape command for SSH
        escaped_command = command.replace("'", "'\\''")
        ssh_command = f"ssh {self.ssh_options} {destination} '{escaped_command}'"

        start_time = time.time()

        try:
            proc = await asyncio.create_subprocess_shell(
                ssh_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                elapsed_ms = int((time.time() - start_time) * 1000)
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"SSH command timed out after {timeout}s",
                    elapsed_ms=elapsed_ms,
                    host=host,
                )

            elapsed_ms = int((time.time() - start_time) * 1000)

            output_parts = []
            if stdout:
                output_parts.append(stdout.decode().strip())
            if stderr:
                stderr_text = stderr.decode().strip()
                # Filter out SSH connection messages
                if not stderr_text.startswith("Warning:"):
                    output_parts.append(f"Errors: {stderr_text}")

            return ExecutionResult(
                success=proc.returncode == 0,
                output="\n".join(output_parts) or "Command completed with no output.",
                exit_code=proc.returncode or 0,
                host=host,
                elapsed_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                elapsed_ms=elapsed_ms,
                host=host,
            )

    async def is_available(self) -> bool:
        """Check if SSH is available."""
        try:
            proc = await asyncio.create_subprocess_shell(
                "which ssh",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()
            return proc.returncode == 0
        except Exception:
            return False
