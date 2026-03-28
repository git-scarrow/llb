from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class WorkerConfig:
    lb_url: str = _env_str("LLB_URL", os.getenv("AI_LB_URL", f"http://localhost:{os.getenv('LLB_PORT', os.getenv('AI_LB_PORT', '8002'))}"))  # COMPAT: AI_LB_* fallback remove after 2026-06-01
    model: str = _env_str("WORKER_MODEL", "auto")
    max_turns: int = _env_int("WORKER_MAX_TURNS", 5)
    timeout_secs: float = _env_float("WORKER_TIMEOUT_SECS", 60.0)
    mcp_filesystem_cmd: Optional[str] = _env_str("MCP_FILESYSTEM_CMD")
    mcp_shell_cmd: Optional[str] = _env_str("MCP_SHELL_CMD")
    mcp_startup_timeout_secs: float = _env_float("MCP_STARTUP_TIMEOUT_SECS", 10.0)


@dataclass(frozen=True)
class OrchestratorConfig:
    provider: str = _env_str("ORCH_PROVIDER", "openai")
    model: str = _env_str("ORCH_MODEL", "gpt-4o-mini")
    max_tasks: int = _env_int("ORCH_MAX_TASKS", 16)
    verify_with_llm: bool = _env_bool("ORCH_VERIFY_WITH_LLM", False)
    cli_cmd: Optional[str] = _env_str("ORCH_CLI_CMD")
