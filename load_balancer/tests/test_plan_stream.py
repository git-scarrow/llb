"""Tests for PLAN execution mode: streaming SSE passthrough."""

import asyncio
import json
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure package path for local src
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "load_balancer" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from load_balancer.main import app as lb_app
from load_balancer import main as lb_main
from load_balancer.execution.modes import (
    PlanEvent,
    PlanResult,
    collect_plan_result,
    execute_plan,
    execute_plan_stream,
    plan_result_to_openai_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    return asyncio.run(coro)


_SIMPLE_PLAN_JSON = json.dumps(
    {
        "goal": "test goal",
        "tasks": [
            {
                "id": "t1",
                "capability": "general",
                "prompt": "do task 1",
                "depends_on": [],
            }
        ],
    }
)

_TWO_TASK_PLAN_JSON = json.dumps(
    {
        "goal": "two-task goal",
        "tasks": [
            {
                "id": "t1",
                "capability": "general",
                "prompt": "do task 1",
                "depends_on": [],
            },
            {
                "id": "t2",
                "capability": "general",
                "prompt": "do task 2",
                "depends_on": ["t1"],
            },
        ],
    }
)

_FIVE_TASK_PLAN_JSON = json.dumps(
    {
        "goal": "five-task goal",
        "tasks": [
            {"id": f"t{i}", "capability": "general", "prompt": f"do task {i}", "depends_on": []}
            for i in range(1, 6)
        ],
    }
)


class FakeRedis:
    def __init__(self):
        self.kv = {}
        self.sets = {"nodes:healthy": set()}

    async def smembers(self, key):
        return set(self.sets.get(key, set()))

    async def scard(self, key):
        return len(self.sets.get(key, set()))

    async def get(self, key):
        return self.kv.get(key)

    async def set(self, key, val):
        self.kv[key] = val

    async def incrby(self, key, val):
        v = int(self.kv.get(key, 0)) + int(val)
        self.kv[key] = v
        return v

    async def expire(self, key, ttl):
        return True

    async def sadd(self, key, val):
        s = self.sets.setdefault(key, set())
        before = len(s)
        s.add(val)
        return 1 if len(s) > before else 0

    async def incrbyfloat(self, key, val):
        v = float(self.kv.get(key, 0.0)) + float(val)
        self.kv[key] = v
        return v

    async def close(self):
        return True


class FakeStreamResponse:
    def __init__(self, chunks, status_code=200, raise_on_enter=None):
        self._chunks = chunks
        self.status_code = status_code
        self._raise_on_enter = raise_on_enter

    async def __aenter__(self):
        if self._raise_on_enter:
            raise self._raise_on_enter
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c


class FakeHTTPClient:
    """HTTP client fake for PLAN tests.

    Supports both `.post(...)` (non-streaming) and `.stream(...)` (streaming).
    `behavior` maps node host -> {"chunks": [...], "json": dict, "status": int, "error": exc}
    """

    def __init__(self, behavior):
        self.behavior = behavior
        self._post_call_count = 0

    async def post(self, url, json=None, headers=None, timeout=None):
        node = url.split("//", 1)[1].split("/", 1)[0]
        cfg = self.behavior.get(node, {"status": 200, "response": {"choices": [{"message": {"content": ""}}]}})
        if cfg.get("error"):
            raise cfg["error"]

        class FakeResp:
            def __init__(self, body, status):
                self.status_code = status
                self._body = body

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise Exception(f"HTTP {self.status_code}")

            def json(self):
                return self._body

        return FakeResp(cfg.get("response", {"choices": [{"message": {"content": ""}}]}), cfg.get("status", 200))

    def stream(self, method, url, json=None, headers=None):
        node = url.split("//", 1)[1].split("/", 1)[0]
        cfg = self.behavior.get(node, {"chunks": [b""], "status": 200})
        if cfg.get("error"):
            return FakeStreamResponse([], raise_on_enter=cfg["error"])
        return FakeStreamResponse(cfg.get("chunks", [b""]), status_code=cfg.get("status", 200))

    async def aclose(self):
        return True


@pytest.fixture(autouse=True)
def setup_redis(monkeypatch):
    r = FakeRedis()
    lb_main.redis_client = r
    return r


def make_client():
    return TestClient(lb_app)


# ---------------------------------------------------------------------------
# Unit tests: execute_plan_stream / collect_plan_result
# ---------------------------------------------------------------------------

def test_execute_plan_stream_yields_expected_events():
    """All expected event types are produced for a simple 1-task plan."""

    call_count = 0

    async def mock_call_backend(node, messages):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _SIMPLE_PLAN_JSON
        return "task 1 result"

    async def mock_stream_backend(node, messages):
        yield b"assembled response"

    async def run_test():
        events = []
        async for event in execute_plan_stream(
            messages=[{"role": "user", "content": "do something"}],
            call_backend=mock_call_backend,
            stream_backend=mock_stream_backend,
            planner_backend="node1:9999",
            capability_nodes={},
            default_nodes=["node1:9999"],
        ):
            events.append(event)
        return events

    events = run(run_test())
    event_types = [e.event_type for e in events]

    assert "plan_decomposed" in event_types
    assert "task_started" in event_types
    assert "task_finished" in event_types
    assert "assembly_started" in event_types
    assert "token" in event_types

    # task_finished must precede first token (ordering invariant)
    finished_idx = max(i for i, e in enumerate(events) if e.event_type == "task_finished")
    first_token_idx = next(i for i, e in enumerate(events) if e.event_type == "token")
    assert finished_idx < first_token_idx, "task_finished must precede first token"


def test_collect_plan_result_reconstructs_goal_and_content():
    """collect_plan_result assembles tokens and captures goal from plan_decomposed."""

    call_count = 0

    async def mock_call_backend(node, messages):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _SIMPLE_PLAN_JSON  # decompose step
        return "task result"  # subtask dispatch step

    async def mock_stream_backend(node, messages):
        yield b"hello "
        yield b"world"

    async def run_test():
        gen = execute_plan_stream(
            messages=[{"role": "user", "content": "hi"}],
            call_backend=mock_call_backend,
            stream_backend=mock_stream_backend,
            planner_backend="node1:9999",
            capability_nodes={},
            default_nodes=["node1:9999"],
        )
        return await collect_plan_result(gen)

    result = run(run_test())
    assert result.goal == "test goal"
    assert result.final_response.success is True
    content = result.final_response.response_body["choices"][0]["message"]["content"]
    assert "hello " in content
    assert "world" in content


def test_execute_plan_emits_error_on_decompose_failure():
    """An unparseable planner response produces an error event and stops."""

    async def bad_call_backend(node, messages):
        return "this is not json at all"

    async def noop_stream_backend(node, messages):
        yield b""  # pragma: no cover

    async def run_test():
        events = []
        async for event in execute_plan_stream(
            messages=[{"role": "user", "content": "hi"}],
            call_backend=bad_call_backend,
            stream_backend=noop_stream_backend,
            planner_backend="node1:9999",
            capability_nodes={},
            default_nodes=["node1:9999"],
        ):
            events.append(event)
        return events

    events = run(run_test())
    assert len(events) == 1
    assert events[0].event_type == "error"
    assert "decompose" in events[0].data["message"].lower()


def test_execute_plan_emits_error_event_for_failed_subtask():
    """A subtask backend failure yields error + task_finished(success=False)."""

    call_count = 0

    async def mock_call_backend(node, messages):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _SIMPLE_PLAN_JSON
        raise RuntimeError("backend_down")

    async def mock_stream_backend(node, messages):
        yield b"assembled"

    async def run_test():
        events = []
        async for event in execute_plan_stream(
            messages=[{"role": "user", "content": "hi"}],
            call_backend=mock_call_backend,
            stream_backend=mock_stream_backend,
            planner_backend="node1:9999",
            capability_nodes={},
            default_nodes=["node1:9999"],
        ):
            events.append(event)
        return events

    events = run(run_test())
    finished_events = [e for e in events if e.event_type == "task_finished"]
    error_events = [e for e in events if e.event_type == "error"]

    assert len(finished_events) == 1
    assert finished_events[0].data["success"] is False

    assert any(e.data.get("task_id") == "t1" for e in error_events)


def test_five_task_plan_emits_five_started_and_finished():
    """Five independent tasks each get task_started + task_finished events."""

    call_count = 0

    async def mock_call_backend(node, messages):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _FIVE_TASK_PLAN_JSON
        return f"result {call_count}"

    async def mock_stream_backend(node, messages):
        yield b"assembled"

    async def run_test():
        events = []
        async for event in execute_plan_stream(
            messages=[{"role": "user", "content": "hi"}],
            call_backend=mock_call_backend,
            stream_backend=mock_stream_backend,
            planner_backend="node1:9999",
            capability_nodes={},
            default_nodes=["node1:9999"],
        ):
            events.append(event)
        return events

    events = run(run_test())
    started = [e for e in events if e.event_type == "task_started"]
    finished = [e for e in events if e.event_type == "task_finished"]

    assert len(started) == 5, f"expected 5 task_started, got {len(started)}"
    assert len(finished) == 5, f"expected 5 task_finished, got {len(finished)}"


def test_ordering_invariant_task_finished_before_token():
    """All task_finished events must precede the first token event."""

    call_count = 0

    async def mock_call_backend(node, messages):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _TWO_TASK_PLAN_JSON
        return "result"

    async def mock_stream_backend(node, messages):
        yield b"token1"
        yield b"token2"

    async def run_test():
        events = []
        async for event in execute_plan_stream(
            messages=[{"role": "user", "content": "hi"}],
            call_backend=mock_call_backend,
            stream_backend=mock_stream_backend,
            planner_backend="node1:9999",
            capability_nodes={},
            default_nodes=["node1:9999"],
        ):
            events.append(event)
        return events

    events = run(run_test())
    finished_indices = [i for i, e in enumerate(events) if e.event_type == "task_finished"]
    token_indices = [i for i, e in enumerate(events) if e.event_type == "token"]

    assert finished_indices, "expected at least one task_finished"
    assert token_indices, "expected at least one token"
    assert max(finished_indices) < min(token_indices), (
        "All task_finished must precede first token"
    )


def test_plan_result_to_openai_response_shape():
    """plan_result_to_openai_response returns an OpenAI-compatible structure."""
    from load_balancer.execution.modes import BackendResult

    result = PlanResult(
        goal="the goal",
        task_results={},
        final_response=BackendResult(
            backend="node1",
            success=True,
            response_body={"choices": [{"message": {"content": "hello"}}]},
        ),
    )
    resp = plan_result_to_openai_response(result)
    assert resp["object"] == "chat.completion"
    assert resp["choices"][0]["message"]["content"] == "hello"
    assert resp["choices"][0]["finish_reason"] == "stop"


def test_execute_plan_wrapper_returns_plan_result():
    """execute_plan() thin wrapper returns a PlanResult."""

    call_count = 0

    async def mock_call_backend(node, messages):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _SIMPLE_PLAN_JSON
        return "result"

    async def mock_stream_backend(node, messages):
        yield b"done"

    result = run(
        execute_plan(
            messages=[{"role": "user", "content": "hi"}],
            call_backend=mock_call_backend,
            stream_backend=mock_stream_backend,
            planner_backend="node1:9999",
            capability_nodes={},
            default_nodes=["node1:9999"],
        )
    )
    assert isinstance(result, PlanResult)
    assert result.goal == "test goal"


# ---------------------------------------------------------------------------
# HTTP endpoint tests
# ---------------------------------------------------------------------------

def _setup_node(r, node="planner:9999", model="m"):
    r.sets["nodes:healthy"] = {node}
    run(r.set(f"node:{node}:models", json.dumps({"data": [{"id": model}]})))
    return node


def test_plan_mode_sse_streaming_returns_event_stream(monkeypatch):
    """POST with X-Execution-Mode: plan + Accept: text/event-stream returns SSE."""
    r = lb_main.redis_client
    node = _setup_node(r)

    lb_main.http_client = FakeHTTPClient(
        {
            node: {
                "response": {"choices": [{"message": {"content": _SIMPLE_PLAN_JSON}}]},
                "status": 200,
                "chunks": [b"data: assembled token\n\n"],
            }
        }
    )

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            headers={
                "X-Execution-Mode": "plan",
                "Accept": "text/event-stream",
            },
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "do something"}],
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = b"".join(resp.iter_bytes())
        assert b"plan_decomposed" in body
        assert b"[DONE]" in body


def test_plan_mode_sse_event_ordering(monkeypatch):
    """All task_finished events must precede the first bare data: token line."""
    r = lb_main.redis_client
    node = _setup_node(r)

    lb_main.http_client = FakeHTTPClient(
        {
            node: {
                "response": {"choices": [{"message": {"content": _SIMPLE_PLAN_JSON}}]},
                "status": 200,
                "chunks": [b"token chunk"],
            }
        }
    )

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            headers={"X-Execution-Mode": "plan", "Accept": "text/event-stream"},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        body = b"".join(resp.iter_bytes()).decode("utf-8")

    lines = body.splitlines()

    # Named events look like:  "event: task_finished"
    # Token lines look like:   "data: <chunk>"  (a bare data line NOT preceded by an "event:" line)
    # Find the line index for the last "event: task_finished" occurrence
    task_finished_event_line = max(
        (i for i, ln in enumerate(lines) if ln.strip().startswith("event:") and "task_finished" in ln),
        default=None,
    )
    # Token data lines appear after "event: assembly_started" as bare "data:" lines
    assembly_idx = next(
        (i for i, ln in enumerate(lines) if "assembly_started" in ln), None
    )
    first_token_data_line = None
    if assembly_idx is not None:
        first_token_data_line = next(
            (
                i
                for i, ln in enumerate(lines)
                if i > assembly_idx and ln.startswith("data:") and "[DONE]" not in ln
            ),
            None,
        )

    if task_finished_event_line is not None and first_token_data_line is not None:
        assert task_finished_event_line < first_token_data_line, (
            "task_finished events must precede first token data line"
        )


def test_plan_mode_non_streaming_returns_json(monkeypatch):
    """POST with X-Execution-Mode: plan without SSE Accept returns JSON."""
    r = lb_main.redis_client
    node = _setup_node(r)

    lb_main.http_client = FakeHTTPClient(
        {
            node: {
                "response": {"choices": [{"message": {"content": _SIMPLE_PLAN_JSON}}]},
                "status": 200,
                "chunks": [b"assembled text"],
            }
        }
    )

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            headers={"X-Execution-Mode": "plan"},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert "choices" in data
        assert data["choices"][0]["message"]["role"] == "assistant"


def test_plan_mode_backward_compat_no_header(monkeypatch):
    """Requests without X-Execution-Mode are unaffected (normal streaming)."""
    r = lb_main.redis_client
    node = _setup_node(r)

    lb_main.http_client = FakeHTTPClient(
        {node: {"chunks": [b"data: normal\n\n"], "status": 200}}
    )

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        body = b"".join(resp.iter_bytes())
        assert b"normal" in body


def test_plan_mode_missing_node_returns_404(monkeypatch):
    """PLAN mode returns 404 when no eligible nodes found."""
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = set()  # no nodes

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            headers={"X-Execution-Mode": "plan", "Accept": "text/event-stream"},
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 404
