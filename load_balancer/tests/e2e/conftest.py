"""E2E test configuration and fixtures.

All E2E tests are gated on the ``LLB_BASE_URL`` environment variable.
If it is not set the entire suite is skipped so that CI does not fail
without a live stack.

Required environment variables:
    LLB_BASE_URL   e.g. http://localhost:8000   (no trailing slash)

Optional environment variables:
    LLB_REDIS_HOST   Redis host (default: localhost)
    LLB_REDIS_PORT   Redis port (default: 6379)
    LLB_MODEL        Model name to use in tests (default: auto-discovered)
    LLB_HEALTH_TIMEOUT_SECS  Seconds to wait for /health to become green (default: 10)
"""

from __future__ import annotations

import os
import time
from typing import Optional

import httpx
import pytest

# ---------------------------------------------------------------------------
# Guard: skip entire module/session when live stack is absent
# ---------------------------------------------------------------------------

_BASE_URL: Optional[str] = os.environ.get("LLB_BASE_URL", "").rstrip("/") or None


def pytest_collection_modifyitems(config, items):
    """Skip all e2e tests when LLB_BASE_URL is not set."""
    if _BASE_URL:
        return
    skip_marker = pytest.mark.skip(
        reason="LLB_BASE_URL not set – live stack required for e2e tests"
    )
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_marker)


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def base_url() -> str:
    """Return the load-balancer base URL, skipping if not configured."""
    if not _BASE_URL:
        pytest.skip("LLB_BASE_URL not set")
    return _BASE_URL


@pytest.fixture(scope="session")
def lb_client(base_url: str) -> httpx.Client:
    """Synchronous httpx client pointed at the load balancer.
    Keeps a single connection pool for the whole session.
    """
    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        yield client


@pytest.fixture(scope="session")
def redis_client():
    """Return a redis.Redis client for state inspection, or None if redis is unavailable."""
    redis_host = os.environ.get("LLB_REDIS_HOST", "localhost")
    redis_port = int(os.environ.get("LLB_REDIS_PORT", 6379))
    try:
        import redis as _redis

        r = _redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        r.ping()
        yield r
        r.close()
    except Exception:
        yield None


@pytest.fixture(scope="session")
def model_name(lb_client: httpx.Client) -> str:
    """Discover a model to use in tests.

    Prefer ``LLB_MODEL`` env var; fall back to first model returned by
    ``GET /v1/models``.
    """
    explicit = os.environ.get("LLB_MODEL", "").strip()
    if explicit:
        return explicit
    resp = lb_client.get("/v1/models")
    assert resp.status_code == 200, f"GET /v1/models failed: {resp.status_code}"
    data = resp.json().get("data", [])
    assert data, "No models returned by /v1/models – is a backend registered?"
    return data[0]["id"]


@pytest.fixture(scope="session", autouse=True)
def wait_for_healthy(base_url: str):
    """Block until /health reports healthy (or timeout)."""
    if not _BASE_URL:
        return
    timeout = float(os.environ.get("LLB_HEALTH_TIMEOUT_SECS", 10))
    deadline = time.monotonic() + timeout
    last_exc = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base_url}/health", timeout=5.0)
            if r.status_code == 200 and r.json().get("status") == "healthy":
                return
        except Exception as exc:
            last_exc = exc
        time.sleep(0.5)
    pytest.skip(
        f"Load balancer at {base_url} did not become healthy within {timeout}s"
        f"{f': {last_exc}' if last_exc else ''}"
    )
