from __future__ import annotations

import importlib
import os
import shlex
import shutil
import socket
from urllib.parse import urlparse
import sys
from dataclasses import dataclass
from typing import List


DEFAULT_LB_URL = f"http://127.0.0.1:{os.getenv('LLB_PORT', os.getenv('AI_LB_PORT', '8002'))}"  # COMPAT: AI_LB_PORT fallback remove after 2026-06-01
LB_TIMEOUT_SECS = 2.0


@dataclass
class CheckResult:
    label: str
    ok: bool
    detail: str


def _check_import(module: str) -> CheckResult:
    try:
        importlib.import_module(module)
        return CheckResult(label=f"import {module}", ok=True, detail="available")
    except Exception as exc:
        return CheckResult(label=f"import {module}", ok=False, detail=f"{exc.__class__.__name__}: {exc}")


def _resolve_lb_target(url: str) -> tuple[str, int]:
    if "://" not in url:
        url = f"http://{url}"
    parsed = urlparse(url)
    if not parsed.hostname:
        raise ValueError(f"Invalid LLB_URL: {url}")
    if parsed.port is not None:
        return parsed.hostname, parsed.port
    if parsed.scheme == "https":
        return parsed.hostname, 443
    return parsed.hostname, 80


def _check_lb(url: str) -> CheckResult:
    try:
        host, port = _resolve_lb_target(url)
    except ValueError as exc:
        return CheckResult(label="llb", ok=False, detail=str(exc))
    try:
        with socket.create_connection((host, port), timeout=LB_TIMEOUT_SECS):
            return CheckResult(label=f"llb {host}:{port}", ok=True, detail=f"reachable ({url})")
    except OSError as exc:
        return CheckResult(label=f"llb {host}:{port}", ok=False, detail=f"{exc} ({url})")


def _check_command(env_var: str) -> CheckResult:
    value = os.getenv(env_var, "").strip()
    if not value:
        return CheckResult(label=env_var, ok=True, detail="not set (tools disabled)")

    parts = shlex.split(value)
    if not parts:
        return CheckResult(label=env_var, ok=False, detail="empty command")

    command = parts[0]
    if os.path.sep in command or command.startswith("."):
        if os.path.isfile(command) and os.access(command, os.X_OK):
            return CheckResult(label=env_var, ok=True, detail="executable path")
        if os.path.isfile(command):
            return CheckResult(label=env_var, ok=False, detail="path not executable")
        return CheckResult(label=env_var, ok=False, detail="path not found")

    resolved = shutil.which(command)
    if resolved:
        return CheckResult(label=env_var, ok=True, detail=f"found in PATH ({resolved})")
    return CheckResult(label=env_var, ok=False, detail="command not found in PATH")


def _print_report(results: List[CheckResult]) -> bool:
    all_ok = all(result.ok for result in results)
    for result in results:
        status = "OK" if result.ok else "MISSING"
        print(f"{status}: {result.label} - {result.detail}")
    print("Ready" if all_ok else "Missing Dependencies")
    return all_ok


def main() -> int:
    lb_url = os.getenv("LLB_URL", os.getenv("AI_LB_URL", DEFAULT_LB_URL))  # COMPAT: AI_LB_URL fallback remove after 2026-06-01
    results = [
        _check_import("httpx"),
        _check_import("openai"),
        _check_lb(lb_url),
        _check_command("MCP_FILESYSTEM_CMD"),
        _check_command("MCP_SHELL_CMD"),
    ]
    ok = _print_report(results)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
