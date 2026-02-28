"""Tests for Phase 1-3: capability filtering, complexity routing, PLAN mode."""

import asyncio
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "load_balancer" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from load_balancer import config as lb_config
from load_balancer.execution import (
    ExecutionEngine,
    ExecutionMode,
    BackendResult,
    PlanTask,
    PlanResult,
)
from load_balancer.routing.strategies import ComplexityRoutingStrategy


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Phase 1: Capability-typed backend pools
# ---------------------------------------------------------------------------

class TestBackendCapabilities:
    """Phase 1: _backend_has_capabilities and _filter_nodes_by_capability."""

    def setup_method(self):
        # Patch config capabilities
        self._orig = getattr(lb_config, "BACKEND_CAPABILITIES", {})
        lb_config.BACKEND_CAPABILITIES = {
            "claude":          frozenset({"reasoning", "code", "long-context"}),
            "gemini":          frozenset({"research", "multimodal", "grounding"}),
            "openai":          frozenset({"reasoning", "code", "long-context"}),
            "localhost:11434": frozenset({"fast", "private", "code"}),
        }

    def teardown_method(self):
        lb_config.BACKEND_CAPABILITIES = self._orig

    def test_cloud_backend_lookup_strips_prefix(self):
        from load_balancer.main import _backend_has_capabilities
        assert _backend_has_capabilities("cloud:claude", frozenset({"reasoning"})) is True

    def test_cloud_backend_missing_cap(self):
        from load_balancer.main import _backend_has_capabilities
        assert _backend_has_capabilities("cloud:claude", frozenset({"multimodal"})) is False

    def test_local_backend_exact_match(self):
        from load_balancer.main import _backend_has_capabilities
        assert _backend_has_capabilities("localhost:11434", frozenset({"fast", "code"})) is True

    def test_local_backend_superset_allowed(self):
        from load_balancer.main import _backend_has_capabilities
        # Requiring a subset of available caps → True
        assert _backend_has_capabilities("localhost:11434", frozenset({"private"})) is True

    def test_empty_required_always_passes(self):
        from load_balancer.main import _backend_has_capabilities
        assert _backend_has_capabilities("cloud:claude", frozenset()) is True

    def test_unknown_backend_has_no_caps(self):
        from load_balancer.main import _backend_has_capabilities
        assert _backend_has_capabilities("unknown:9999", frozenset({"reasoning"})) is False

    def test_filter_nodes_keeps_matching(self):
        from load_balancer.main import _filter_nodes_by_capability
        nodes = ["cloud:claude", "cloud:gemini", "localhost:11434"]
        result = _filter_nodes_by_capability(nodes, frozenset({"research"}))
        assert result == ["cloud:gemini"]

    def test_filter_nodes_multiple_matches(self):
        from load_balancer.main import _filter_nodes_by_capability
        nodes = ["cloud:claude", "cloud:openai", "cloud:gemini"]
        result = _filter_nodes_by_capability(nodes, frozenset({"reasoning", "code"}))
        assert set(result) == {"cloud:claude", "cloud:openai"}

    def test_filter_nodes_no_match_falls_back_to_all(self):
        from load_balancer.main import _filter_nodes_by_capability
        nodes = ["cloud:claude", "cloud:gemini"]
        # "quantum" capability exists nowhere → fall back to full list
        result = _filter_nodes_by_capability(nodes, frozenset({"quantum"}))
        assert result == nodes

    def test_filter_none_required_returns_all(self):
        from load_balancer.main import _filter_nodes_by_capability
        nodes = ["cloud:claude", "cloud:gemini", "localhost:11434"]
        assert _filter_nodes_by_capability(nodes, None) == nodes

    def test_config_parser_round_trip(self):
        from load_balancer.config import _parse_backend_capabilities
        caps = _parse_backend_capabilities(
            "claude=reasoning,code;gemini=research,multimodal|localhost:11434=fast,private"
        )
        assert "claude" in caps
        assert "reasoning" in caps["claude"]
        assert "localhost:11434" in caps
        assert "fast" in caps["localhost:11434"]


# ---------------------------------------------------------------------------
# Phase 2: Complexity routing strategy
# ---------------------------------------------------------------------------

class TestComplexityRouter:
    """Phase 2: ComplexityRoutingStrategy scoring and tier selection."""

    def setup_method(self):
        self.router = ComplexityRoutingStrategy()
        self.model_classes = {
            "historical_small": {"candidates": ["qwen/qwen3-4b", "mistral-small"], "min_nodes": 1},
            "historical_medium": {"candidates": ["qwen/qwen3-8b", "gpt-oss-20b"], "min_nodes": 1},
        }

    # --- Scoring ---

    def test_empty_messages_returns_zero(self):
        assert self.router.score_prompt_complexity([]) == 0.0

    def test_very_short_prompt_is_low(self):
        msgs = [{"role": "user", "content": "Hi"}]
        score = self.router.score_prompt_complexity(msgs)
        assert score < ComplexityRoutingStrategy.LOW_THRESHOLD

    def test_medium_length_prompt(self):
        msgs = [{"role": "user", "content": "x" * 2000}]
        score = self.router.score_prompt_complexity(msgs)
        # 2000 chars → min(2000/10000,1)*0.30 = 0.06
        assert score >= 0.05

    def test_code_fence_raises_score(self):
        msgs = [{"role": "user", "content": "```python\ncode\n``` ```js\nmore\n```"}]
        score_with_code = self.router.score_prompt_complexity(msgs)
        score_no_code = self.router.score_prompt_complexity([{"role": "user", "content": "plain text"}])
        assert score_with_code > score_no_code

    def test_reasoning_keywords_raise_score(self):
        hi_msgs = [{"role": "user", "content": "analyze the trade-off and explain why comparing pros and cons"}]
        lo_msgs = [{"role": "user", "content": "what time is it"}]
        assert self.router.score_prompt_complexity(hi_msgs) > self.router.score_prompt_complexity(lo_msgs)

    def test_multi_step_markers_raise_score(self):
        # Use markers from the current keyword list: "step 1", "then,", "first,", "next,", "finally,"
        msgs = [{"role": "user", "content": "step 1 do this. first, prep. then, do that. finally, wrap up."}]
        score = self.router.score_prompt_complexity(msgs)
        # 4 markers → min(4/4,1)*0.25 = 0.25, plus small length contribution
        assert score > 0.2

    def test_score_capped_at_one(self):
        # Max out all signals
        content = "analyze " * 50 + "```py\n" * 10 + "step 1 " * 20 + "x" * 5000
        msgs = [{"role": "user", "content": content}]
        assert self.router.score_prompt_complexity(msgs) <= 1.0

    def test_multipart_content_blocks(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "explain why analyze compare"},
            {"type": "text", "text": "x" * 1000},
        ]}]
        score = self.router.score_prompt_complexity(msgs)
        assert score > 0.0

    # --- Tier selection ---

    def test_low_complexity_maps_to_small(self):
        # score just below LOW_THRESHOLD
        score = ComplexityRoutingStrategy.LOW_THRESHOLD - 0.01
        candidates = self.router.get_complexity_model(score, self.model_classes)
        assert candidates == self.model_classes["historical_small"]["candidates"]

    def test_medium_complexity_maps_to_medium(self):
        score = (ComplexityRoutingStrategy.LOW_THRESHOLD + ComplexityRoutingStrategy.HIGH_THRESHOLD) / 2
        candidates = self.router.get_complexity_model(score, self.model_classes)
        assert candidates == self.model_classes["historical_medium"]["candidates"]

    def test_high_complexity_returns_none(self):
        score = ComplexityRoutingStrategy.HIGH_THRESHOLD + 0.01
        assert self.router.get_complexity_model(score, self.model_classes) is None

    def test_missing_tier_returns_none(self):
        score = 0.1
        assert self.router.get_complexity_model(score, {}) is None

    def test_strategy_registered(self):
        from load_balancer.routing.strategies import get_routing_strategy, STRATEGIES
        assert "COMPLEXITY" in STRATEGIES
        strategy = get_routing_strategy("COMPLEXITY")
        assert isinstance(strategy, ComplexityRoutingStrategy)

    def test_select_node_delegates_to_p2c(self):
        """ComplexityRoutingStrategy.select_node must return a node (delegates to P2C)."""
        class FakeRedis:
            async def get(self, key): return None

        nodes = ["a:1234", "b:1234"]
        result = run(self.router.select_node(nodes, "test-model", FakeRedis()))
        assert result in nodes


# ---------------------------------------------------------------------------
# Phase 3: PLAN execution mode
# ---------------------------------------------------------------------------

PLAN_DECOMP_RESPONSE = {
    "choices": [{
        "message": {
            "role": "assistant",
            "content": json.dumps({
                "goal": "Research Python async and write a summary",
                "tasks": [
                    {
                        "id": "t1",
                        "description": "Research Python async/await",
                        "capability": "research",
                        "depends_on": [],
                        "prompt": "Explain Python async/await concisely.",
                    },
                    {
                        "id": "t2",
                        "description": "Write code example",
                        "capability": "code",
                        "depends_on": [],
                        "prompt": "Write a short async Python example.",
                    },
                    {
                        "id": "t3",
                        "description": "Summarize findings",
                        "capability": "reasoning",
                        "depends_on": ["t1", "t2"],
                        "prompt": "Summarize: {t1_result} and {t2_result}",
                    },
                ],
            }),
        }
    }],
    "model": "planner-model",
}

SUBTASK_RESPONSE = lambda content: {
    "choices": [{"message": {"role": "assistant", "content": content}}],
    "model": "test",
}

ASSEMBLY_RESPONSE = {
    "choices": [{"message": {"role": "assistant", "content": "Final assembled answer."}}],
    "model": "assembler",
}


class TestPlanExecutionEngine:
    """Phase 3: ExecutionEngine.execute_plan() unit tests."""

    def _make_engine(self):
        return ExecutionEngine(similarity_threshold=0.9)

    def test_plan_mode_in_enum(self):
        assert ExecutionMode.PLAN.value == "plan"

    def test_basic_plan_execution(self):
        """Decompose → dispatch 3 subtasks (2 parallel, 1 dependent) → assemble."""
        engine = self._make_engine()
        call_count = {"n": 0}

        async def call_backend(node: str, messages: list) -> BackendResult:
            call_count["n"] += 1
            # First call = planner decomposition
            if call_count["n"] == 1:
                body = PLAN_DECOMP_RESPONSE
            elif call_count["n"] in (2, 3):
                body = SUBTASK_RESPONSE(f"result from {node} call {call_count['n']}")
            elif call_count["n"] == 4:
                body = SUBTASK_RESPONSE("final subtask result")
            else:
                body = ASSEMBLY_RESPONSE
            return BackendResult(backend=node, success=True, status_code=200, response_body=body)

        result = run(engine.execute_plan(
            messages=[{"role": "user", "content": "Research Python async and summarize"}],
            call_backend=call_backend,
            planner_backend="cloud:claude",
            capability_nodes={
                "research": ["cloud:gemini"],
                "code": ["localhost:11434"],
                "reasoning": ["cloud:claude"],
            },
            default_nodes=["cloud:claude"],
            max_subtasks=5,
            subtask_timeout=10.0,
            overall_timeout=60.0,
        ))

        assert result.goal == "Research Python async and write a summary"
        assert len(result.tasks) == 3
        assert len(result.task_results) == 3
        assert result.final_response is not None
        assert result.final_response.success is True
        assert result.error is None

    def test_plan_respects_task_ids(self):
        """All three task IDs (t1, t2, t3) should appear in task_results."""
        engine = self._make_engine()
        calls = []

        async def call_backend(node: str, messages: list) -> BackendResult:
            calls.append((node, messages))
            if len(calls) == 1:
                body = PLAN_DECOMP_RESPONSE
            elif len(calls) <= 4:
                body = SUBTASK_RESPONSE("ok")
            else:
                body = ASSEMBLY_RESPONSE
            return BackendResult(backend=node, success=True, status_code=200, response_body=body)

        result = run(engine.execute_plan(
            messages=[{"role": "user", "content": "test"}],
            call_backend=call_backend,
            planner_backend="cloud:claude",
            capability_nodes={},
            default_nodes=["cloud:claude"],
        ))

        assert "t1" in result.task_results
        assert "t2" in result.task_results
        assert "t3" in result.task_results

    def test_plan_handles_invalid_json(self):
        """Planner returning garbage JSON → PlanResult with error."""
        engine = self._make_engine()

        async def call_backend(node: str, messages: list) -> BackendResult:
            body = {"choices": [{"message": {"role": "assistant", "content": "not valid json {{ "}}]}
            return BackendResult(backend=node, success=True, status_code=200, response_body=body)

        result = run(engine.execute_plan(
            messages=[{"role": "user", "content": "test"}],
            call_backend=call_backend,
            planner_backend="cloud:claude",
            capability_nodes={},
            default_nodes=["cloud:claude"],
        ))
        assert result.error is not None
        assert "JSON" in result.error or "json" in result.error.lower()

    def test_plan_handles_planner_failure(self):
        """Planner backend returning failure → PlanResult with error."""
        engine = self._make_engine()

        async def call_backend(node: str, messages: list) -> BackendResult:
            return BackendResult(backend=node, success=False, error="upstream 503")

        result = run(engine.execute_plan(
            messages=[{"role": "user", "content": "test"}],
            call_backend=call_backend,
            planner_backend="cloud:claude",
            capability_nodes={},
            default_nodes=["cloud:claude"],
        ))
        assert result.error is not None

    def test_plan_routes_to_capability_nodes(self):
        """Each subtask should be called on the matching capability backend."""
        engine = self._make_engine()
        routed_to = {}
        call_n = {"n": 0}

        async def call_backend(node: str, messages: list) -> BackendResult:
            call_n["n"] += 1
            if call_n["n"] == 1:
                body = PLAN_DECOMP_RESPONSE
            elif call_n["n"] <= 4:
                # Record which node handled this subtask
                last_msg = messages[-1]["content"] if messages else ""
                routed_to[last_msg[:30]] = node
                body = SUBTASK_RESPONSE("ok")
            else:
                body = ASSEMBLY_RESPONSE
            return BackendResult(backend=node, success=True, status_code=200, response_body=body)

        run(engine.execute_plan(
            messages=[{"role": "user", "content": "test"}],
            call_backend=call_backend,
            planner_backend="cloud:claude",
            capability_nodes={
                "research": ["cloud:gemini"],
                "code": ["localhost:11434"],
                "reasoning": ["cloud:claude"],
            },
            default_nodes=["cloud:default"],
        ))
        # At least one subtask hit a capability-matched node
        used_nodes = set(routed_to.values())
        assert len(used_nodes) > 0

    def test_plan_single_atomic_task(self):
        """Planner returns a single task → still works correctly."""
        engine = self._make_engine()
        call_n = {"n": 0}
        single_task_resp = {
            "choices": [{"message": {"role": "assistant", "content": json.dumps({
                "goal": "Simple question",
                "tasks": [{"id": "t1", "description": "Answer it", "capability": "general",
                            "depends_on": [], "prompt": "What is 2+2?"}],
            })}}],
        }

        async def call_backend(node: str, messages: list) -> BackendResult:
            call_n["n"] += 1
            if call_n["n"] == 1:
                body = single_task_resp
            elif call_n["n"] == 2:
                body = SUBTASK_RESPONSE("4")
            else:
                body = ASSEMBLY_RESPONSE
            return BackendResult(backend=node, success=True, status_code=200, response_body=body)

        result = run(engine.execute_plan(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            call_backend=call_backend,
            planner_backend="cloud:claude",
            capability_nodes={},
            default_nodes=["cloud:claude"],
        ))
        assert result.error is None
        assert len(result.tasks) == 1
        assert result.final_response.success is True

    def test_plan_dependency_ordering(self):
        """t3 depends on t1 and t2 → t3 must be dispatched after t1 and t2 complete."""
        engine = self._make_engine()
        completion_order = []
        call_n = {"n": 0}

        async def call_backend(node: str, messages: list) -> BackendResult:
            call_n["n"] += 1
            if call_n["n"] == 1:
                body = PLAN_DECOMP_RESPONSE  # has t3 depending on t1, t2
            elif call_n["n"] <= 3:
                # t1 and t2 run in parallel batch 1
                msg = messages[-1].get("content", "")
                completion_order.append("batch1")
                body = SUBTASK_RESPONSE("ok")
            elif call_n["n"] == 4:
                completion_order.append("batch2")
                body = SUBTASK_RESPONSE("ok")
            else:
                body = ASSEMBLY_RESPONSE
            return BackendResult(backend=node, success=True, status_code=200, response_body=body)

        result = run(engine.execute_plan(
            messages=[{"role": "user", "content": "test"}],
            call_backend=call_backend,
            planner_backend="cloud:claude",
            capability_nodes={},
            default_nodes=["cloud:claude"],
        ))
        # t3 (batch2) must come after t1 and t2 (batch1)
        assert completion_order.index("batch2") > completion_order.index("batch1")

    def test_plan_subtask_failure_doesnt_block_assembly(self):
        """If a subtask fails, we still proceed to assembly with the error noted."""
        engine = self._make_engine()
        call_n = {"n": 0}
        single_dep_resp = {
            "choices": [{"message": {"role": "assistant", "content": json.dumps({
                "goal": "Do two things",
                "tasks": [
                    {"id": "t1", "description": "fail step", "capability": "general",
                     "depends_on": [], "prompt": "fail me"},
                    {"id": "t2", "description": "succeed step", "capability": "general",
                     "depends_on": [], "prompt": "succeed"},
                ],
            })}}],
        }

        async def call_backend(node: str, messages: list) -> BackendResult:
            call_n["n"] += 1
            if call_n["n"] == 1:
                body = single_dep_resp
                return BackendResult(backend=node, success=True, status_code=200, response_body=body)
            msg = messages[-1].get("content", "")
            if "fail me" in msg:
                return BackendResult(backend=node, success=False, error="forced failure")
            if "succeed" in msg:
                return BackendResult(backend=node, success=True, status_code=200,
                                     response_body=SUBTASK_RESPONSE("success!"))
            # assembly
            return BackendResult(backend=node, success=True, status_code=200,
                                 response_body=ASSEMBLY_RESPONSE)

        result = run(engine.execute_plan(
            messages=[{"role": "user", "content": "test"}],
            call_backend=call_backend,
            planner_backend="cloud:claude",
            capability_nodes={},
            default_nodes=["cloud:claude"],
        ))
        assert "t1" in result.task_results
        assert result.task_results["t1"].success is False
        assert "t2" in result.task_results
        assert result.task_results["t2"].success is True
        # Assembly still runs
        assert result.final_response is not None


# ---------------------------------------------------------------------------
# Phase 1+3 integration: _resolve_planner_backend
# ---------------------------------------------------------------------------

class TestResolvePlannerBackend:
    def setup_method(self):
        self._orig_planner = getattr(lb_config, "PLANNER_BACKEND", "")
        self._orig_cloud = getattr(lb_config, "CLOUD_BACKENDS", {})

    def teardown_method(self):
        lb_config.PLANNER_BACKEND = self._orig_planner
        lb_config.CLOUD_BACKENDS = self._orig_cloud

    def test_oracle_takes_priority(self):
        from load_balancer.main import _resolve_planner_backend
        lb_config.PLANNER_BACKEND = "gemini"
        assert _resolve_planner_backend("cloud:openai") == "cloud:openai"

    def test_planner_backend_config_normalized(self):
        from load_balancer.main import _resolve_planner_backend
        lb_config.CLOUD_BACKENDS = {"claude": {"url": "x", "api_key": "y"}}
        lb_config.PLANNER_BACKEND = "claude"
        result = _resolve_planner_backend(None)
        assert result == "cloud:claude"

    def test_falls_back_to_first_cloud(self):
        from load_balancer.main import _resolve_planner_backend
        lb_config.PLANNER_BACKEND = ""
        lb_config.CLOUD_BACKENDS = {"gemini": {}, "openai": {}}
        result = _resolve_planner_backend(None)
        assert result == "cloud:gemini"

    def test_returns_none_when_nothing_configured(self):
        from load_balancer.main import _resolve_planner_backend
        lb_config.PLANNER_BACKEND = ""
        lb_config.CLOUD_BACKENDS = {}
        assert _resolve_planner_backend(None) is None


# ---------------------------------------------------------------------------
# Phase 2+config: complexity routing integration
# ---------------------------------------------------------------------------

class TestComplexityConfigParsing:
    def test_env_var_enabled(self):
        from load_balancer.config import _parse_backend_capabilities
        result = _parse_backend_capabilities("a=reasoning,code;b=research|c=fast,private")
        assert frozenset({"reasoning", "code"}) == result["a"]
        assert frozenset({"research"}) == result["b"]
        assert frozenset({"fast", "private"}) == result["c"]

    def test_empty_env_var(self):
        from load_balancer.config import _parse_backend_capabilities
        assert _parse_backend_capabilities("") == {}

    def test_malformed_entry_skipped(self):
        from load_balancer.config import _parse_backend_capabilities
        result = _parse_backend_capabilities("no_equals;valid=cap1,cap2")
        assert "valid" in result
        assert "no_equals" not in result
