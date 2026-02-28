"""Tests for Phase 3: PLAN execution mode (Perplexity Computer-style orchestration).

Covers all 12 required scenarios:
  1.  Basic 3-subtask plan: decompose → parallel dispatch → assemble
  2.  All task IDs present in results
  3.  Invalid JSON from planner → graceful error
  4.  Planner failure → graceful error
  5.  Capability routing: subtasks dispatched to correct backends
  6.  Single atomic task
  7.  Dependency ordering (t3 depends on t1, t2)
  8.  Subtask failure does not block assembly
  9.  Planner resolver: oracle priority
  10. Planner resolver: config normalization
  11. Planner resolver: cloud fallback
  12. Planner resolver: empty config → None
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "load_balancer" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from load_balancer.execution import (
    BackendResult,
    ExecutionEngine,
    PlanTask,
    PlanResult,
)
from load_balancer.execution.modes import (
    _PLANNER_SYSTEM_PROMPT,
    _extract_text_content,
)
from load_balancer import config as lb_config
from load_balancer.main import _resolve_planner_backend


def run(coro):
    return asyncio.run(coro)


def _ok(backend: str, text: str) -> BackendResult:
    """Build a successful BackendResult with the given text content."""
    return BackendResult(
        backend=backend,
        success=True,
        status_code=200,
        response_body={
            "choices": [{"message": {"content": text}}]
        },
    )


def _fail(backend: str, error: str = "boom") -> BackendResult:
    return BackendResult(backend=backend, success=False, error=error)


# ---------------------------------------------------------------------------
# Helpers to build planner responses
# ---------------------------------------------------------------------------

def _plan_json(goal: str, tasks: list[dict]) -> str:
    return json.dumps({"goal": goal, "tasks": tasks})


def _three_task_plan() -> str:
    return _plan_json("test goal", [
        {"id": "t1", "description": "step1", "capability": "research",
         "depends_on": [], "prompt": "do step 1"},
        {"id": "t2", "description": "step2", "capability": "code",
         "depends_on": [], "prompt": "do step 2"},
        {"id": "t3", "description": "step3", "capability": "general",
         "depends_on": [], "prompt": "do step 3"},
    ])


def _dependent_plan() -> str:
    """t3 depends on t1 and t2."""
    return _plan_json("dependent goal", [
        {"id": "t1", "description": "first", "capability": "research",
         "depends_on": [], "prompt": "first"},
        {"id": "t2", "description": "second", "capability": "code",
         "depends_on": [], "prompt": "second"},
        {"id": "t3", "description": "third", "capability": "general",
         "depends_on": ["t1", "t2"], "prompt": "combine"},
    ])


def _single_task_plan() -> str:
    return _plan_json("atomic goal", [
        {"id": "t1", "description": "only step", "capability": "general",
         "depends_on": [], "prompt": "do the thing"},
    ])


# ---------------------------------------------------------------------------
# Tests 1–8: execute_plan behavior
# ---------------------------------------------------------------------------

class TestExecutePlan:

    def _build_call_backend(self, responses: dict):
        """Build a call_backend mock that returns pre-defined responses.

        responses: maps a key (matched against message content) to BackendResult.
        Falls back to a generic success if no key matches.
        """
        call_log = []  # (backend, messages) tuples

        async def call_backend(backend: str, messages: list[dict]) -> BackendResult:
            call_log.append((backend, messages))
            # Check if any key substring is in any message content
            for key, result in responses.items():
                for m in messages:
                    content = m.get("content", "")
                    if key in content:
                        return result
            return _ok(backend, "generic ok")

        return call_backend, call_log

    # 1. Basic 3-subtask plan
    def test_basic_three_subtask_plan(self):
        plan_text = _three_task_plan()
        responses = {
            "task orchestrator": _ok("planner", plan_text),      # decompose
            "do step 1": _ok("node-r", "result 1"),
            "do step 2": _ok("node-c", "result 2"),
            "do step 3": _ok("node-g", "result 3"),
            "Subtask results": _ok("planner", "final answer"),   # assemble
        }
        call_fn, log = self._build_call_backend(responses)
        engine = ExecutionEngine()
        result = run(engine.execute_plan(
            messages=[{"role": "user", "content": "test request"}],
            call_backend=call_fn,
            planner_backend="planner",
            capability_nodes={"research": ["node-r"], "code": ["node-c"]},
            default_nodes=["node-g"],
        ))
        assert result.goal == "test goal"
        assert result.final_response is not None
        assert result.final_response.success
        assert len(result.tasks) == 3

    # 2. All task IDs present in results
    def test_all_task_ids_in_results(self):
        plan_text = _three_task_plan()
        responses = {
            "task orchestrator": _ok("planner", plan_text),
            "Subtask results": _ok("planner", "assembled"),
        }
        call_fn, _ = self._build_call_backend(responses)
        engine = ExecutionEngine()
        result = run(engine.execute_plan(
            messages=[{"role": "user", "content": "go"}],
            call_backend=call_fn,
            planner_backend="planner",
            capability_nodes={},
            default_nodes=["node"],
        ))
        assert set(result.task_results.keys()) == {"t1", "t2", "t3"}

    # 3. Invalid JSON from planner
    def test_invalid_json_graceful_error(self):
        responses = {
            "task orchestrator": _ok("planner", "this is not json at all {{{"),
        }
        call_fn, _ = self._build_call_backend(responses)
        engine = ExecutionEngine()
        result = run(engine.execute_plan(
            messages=[{"role": "user", "content": "go"}],
            call_backend=call_fn,
            planner_backend="planner",
            capability_nodes={},
            default_nodes=["node"],
        ))
        assert result.error is not None
        assert "invalid JSON" in result.error.lower() or "JSON" in result.error

    # 4. Planner failure
    def test_planner_failure_graceful_error(self):
        responses = {
            "task orchestrator": _fail("planner", "backend down"),
        }
        call_fn, _ = self._build_call_backend(responses)
        engine = ExecutionEngine()
        result = run(engine.execute_plan(
            messages=[{"role": "user", "content": "go"}],
            call_backend=call_fn,
            planner_backend="planner",
            capability_nodes={},
            default_nodes=["node"],
        ))
        assert result.error is not None
        assert "Planner failed" in result.error or "failed" in result.error.lower()

    # 5. Capability routing
    def test_capability_routing_dispatches_correctly(self):
        plan_text = _plan_json("cap test", [
            {"id": "t1", "description": "research", "capability": "research",
             "depends_on": [], "prompt": "research task"},
            {"id": "t2", "description": "code", "capability": "code",
             "depends_on": [], "prompt": "code task"},
        ])
        responses = {
            "task orchestrator": _ok("planner", plan_text),
            "Subtask results": _ok("planner", "done"),
        }
        call_fn, log = self._build_call_backend(responses)
        engine = ExecutionEngine()
        result = run(engine.execute_plan(
            messages=[{"role": "user", "content": "go"}],
            call_backend=call_fn,
            planner_backend="planner",
            capability_nodes={
                "research": ["research-node"],
                "code": ["code-node"],
            },
            default_nodes=["fallback-node"],
        ))
        # Find the subtask dispatch calls (not planner calls)
        subtask_calls = [(b, m) for b, m in log
                         if not any("task orchestrator" in msg.get("content", "").lower()
                                    or "Subtask results" in msg.get("content", "")
                                    for msg in m)]
        # Filter to only user-message calls (subtask dispatches)
        dispatch_backends = set()
        for backend, msgs in subtask_calls:
            for msg in msgs:
                if "research task" in msg.get("content", ""):
                    dispatch_backends.add(("research", backend))
                if "code task" in msg.get("content", ""):
                    dispatch_backends.add(("code", backend))
        assert ("research", "research-node") in dispatch_backends
        assert ("code", "code-node") in dispatch_backends

    # 6. Single atomic task
    def test_single_atomic_task(self):
        plan_text = _single_task_plan()
        responses = {
            "task orchestrator": _ok("planner", plan_text),
            "do the thing": _ok("node", "single result"),
            "Subtask results": _ok("planner", "final single"),
        }
        call_fn, _ = self._build_call_backend(responses)
        engine = ExecutionEngine()
        result = run(engine.execute_plan(
            messages=[{"role": "user", "content": "go"}],
            call_backend=call_fn,
            planner_backend="planner",
            capability_nodes={},
            default_nodes=["node"],
        ))
        assert len(result.tasks) == 1
        assert "t1" in result.task_results
        assert result.final_response is not None

    # 7. Dependency ordering
    def test_dependency_ordering(self):
        plan_text = _dependent_plan()
        call_order = []

        async def call_backend(backend, messages):
            # Track which subtask prompts are called and when
            for m in messages:
                content = m.get("content", "")
                if content in ("first", "second", "combine"):
                    call_order.append(content)
            # Return planner JSON for decompose call
            for m in messages:
                if "task orchestrator" in m.get("content", "").lower():
                    return _ok(backend, plan_text)
                if "Subtask results" in m.get("content", ""):
                    return _ok(backend, "assembled")
            return _ok(backend, "subtask done")

        engine = ExecutionEngine()
        result = run(engine.execute_plan(
            messages=[{"role": "user", "content": "go"}],
            call_backend=call_backend,
            planner_backend="planner",
            capability_nodes={},
            default_nodes=["node"],
        ))
        # t3 ("combine") must come after both t1 ("first") and t2 ("second")
        assert "combine" in call_order
        combine_idx = call_order.index("combine")
        assert "first" in call_order[:combine_idx]
        assert "second" in call_order[:combine_idx]

    # 8. Subtask failure does not block assembly
    def test_subtask_failure_does_not_block_assembly(self):
        plan_text = _three_task_plan()

        async def call_backend(backend, messages):
            for m in messages:
                content = m.get("content", "")
                if "task orchestrator" in content.lower():
                    return _ok(backend, plan_text)
                if "do step 2" in content:
                    return _fail(backend, "step 2 exploded")
                if "Subtask results" in content:
                    return _ok(backend, "assembled with partial")
            return _ok(backend, "ok")

        engine = ExecutionEngine()
        result = run(engine.execute_plan(
            messages=[{"role": "user", "content": "go"}],
            call_backend=call_backend,
            planner_backend="planner",
            capability_nodes={},
            default_nodes=["node"],
        ))
        # Assembly should still happen
        assert result.final_response is not None
        assert result.final_response.success
        # t2 should have failed
        assert not result.task_results["t2"].success
        # t1 and t3 should have succeeded
        assert result.task_results["t1"].success
        assert result.task_results["t3"].success


# ---------------------------------------------------------------------------
# Tests 9–12: _resolve_planner_backend
# ---------------------------------------------------------------------------

class TestResolvePlannerBackend:

    def setup_method(self):
        self._planner = lb_config.PLANNER_BACKEND
        self._cloud = getattr(lb_config, "CLOUD_BACKENDS", {})

    def teardown_method(self):
        lb_config.PLANNER_BACKEND = self._planner
        lb_config.CLOUD_BACKENDS = self._cloud

    # 9. Oracle priority
    def test_oracle_takes_priority(self):
        lb_config.PLANNER_BACKEND = "some-other-backend"
        result = _resolve_planner_backend(oracle_backend="cloud:openai")
        assert result == "cloud:openai"

    # 10. Config normalization (bare name → cloud:name)
    def test_config_normalization(self):
        lb_config.PLANNER_BACKEND = "claude"
        lb_config.CLOUD_BACKENDS = {"claude": {"url": "https://api.anthropic.com", "api_key": "sk"}}
        result = _resolve_planner_backend(oracle_backend=None)
        assert result == "cloud:claude"

    def test_config_already_prefixed(self):
        lb_config.PLANNER_BACKEND = "cloud:claude"
        lb_config.CLOUD_BACKENDS = {"claude": {"url": "x", "api_key": "y"}}
        result = _resolve_planner_backend(oracle_backend=None)
        assert result == "cloud:claude"

    def test_config_local_backend(self):
        """A PLANNER_BACKEND that's not a cloud name is returned as-is."""
        lb_config.PLANNER_BACKEND = "localhost:11434"
        lb_config.CLOUD_BACKENDS = {}
        result = _resolve_planner_backend(oracle_backend=None)
        assert result == "localhost:11434"

    # 11. Cloud fallback
    def test_cloud_fallback_when_no_config(self):
        lb_config.PLANNER_BACKEND = ""
        lb_config.CLOUD_BACKENDS = {"openai": {"url": "x", "api_key": "y"}}
        result = _resolve_planner_backend(oracle_backend=None)
        assert result == "cloud:openai"

    # 12. Empty config → None
    def test_empty_config_returns_none(self):
        lb_config.PLANNER_BACKEND = ""
        lb_config.CLOUD_BACKENDS = {}
        result = _resolve_planner_backend(oracle_backend=None)
        assert result is None


# ---------------------------------------------------------------------------
# Bonus: planner system prompt brace escaping
# ---------------------------------------------------------------------------

class TestPlannerPromptEscaping:

    def test_planner_prompt_format_does_not_raise(self):
        """Verify that .format(max_tasks=N) works — literal braces are escaped."""
        result = _PLANNER_SYSTEM_PROMPT.format(max_tasks=5)
        assert "Maximum 5 tasks" in result
        # Escaped braces should render as literal braces in output
        assert '"id": "t1"' in result
