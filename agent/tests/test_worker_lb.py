import os

import pytest

httpx = pytest.importorskip("httpx")

from agent.config import WorkerConfig
from agent.tools.mcp_client import MCPToolRouter
from agent.types import Task
from agent.worker import Worker


def test_worker_calls_lb_if_available() -> None:
    lb_url = os.getenv("LLB_URL", os.getenv("AI_LB_URL", f"http://localhost:{os.getenv('LLB_PORT', os.getenv('AI_LB_PORT', '8002'))}"))  # COMPAT: AI_LB_* fallback remove after 2026-06-01
    try:
        resp = httpx.get(f"{lb_url}/v1/models", timeout=5.0)
    except httpx.RequestError:
        pytest.skip("LLB not reachable")
    if resp.status_code != 200:
        pytest.skip("LLB /v1/models not available")

    data = resp.json()
    models = [item.get("id") for item in data.get("data", []) if item.get("id")]
    if not models:
        pytest.skip("No models available on LLB")

    config = WorkerConfig(lb_url=lb_url, model=models[0], max_turns=2, timeout_secs=30.0)
    worker = Worker(config=config, tools=MCPToolRouter([]))
    result = worker.run(Task(id="t1", description="Say 'ok'"))

    assert result.success
    assert result.output
    assert result.lb_calls
