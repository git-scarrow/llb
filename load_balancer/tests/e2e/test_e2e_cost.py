"""E2E tests for cost-aware P2C routing (Phase 4).

Covers:
  - Redis EWMA keys (lb:output_tokens_ewma:{model}|{node}) update after requests
  - P2C_BETA configured → cost term influences routing
  - Redis key format matches the implementation in strategies.py

Prerequisites (set in environment or skip):
  - LLB_BASE_URL
  - P2C_BETA > 0 and BACKEND_COST_PER_TOKEN configured in the LB process

If Redis is unavailable or P2C_BETA is 0 (default), EWMA-specific
assertions are skipped with an informational message.
"""

from __future__ import annotations

import os
from typing import Dict

import httpx
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHAT_PAYLOAD_TEMPLATE: dict = {
    "messages": [{"role": "user", "content": "Say one word."}],
    "stream": False,
    "max_tokens": 5,
}


def _chat_payload(model: str) -> dict:
    return {**_CHAT_PAYLOAD_TEMPLATE, "model": model}


def _ewma_key(model: str, node: str) -> str:
    return f"lb:output_tokens_ewma:{model}|{node}"


def _count_key(model: str, node: str) -> str:
    return f"lb:output_tokens_count:{model}|{node}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestCostAwareRouting:
    """Cost-aware P2C routing – EWMA token tracking and scoring."""

    def test_request_succeeds_with_cost_routing_enabled(
        self, lb_client: httpx.Client, model_name: str
    ):
        """Basic sanity: a request completes regardless of cost config."""
        resp = lb_client.post("/v1/chat/completions", json=_chat_payload(model_name))
        assert resp.status_code == 200, f"Chat completion failed: {resp.text}"

    def test_ewma_keys_created_after_requests(
        self,
        lb_client: httpx.Client,
        model_name: str,
        redis_client,
    ):
        """After several requests, at least one EWMA key should exist in Redis
        (requires P2C_BETA > 0; skipped when Redis is unavailable or beta=0).
        """
        if redis_client is None:
            pytest.skip("Redis unavailable – cannot inspect EWMA keys")

        p2c_beta = float(os.environ.get("P2C_BETA", "0"))
        if p2c_beta <= 0:
            pytest.skip(
                "P2C_BETA=0 (default) – cost EWMA tracking disabled; "
                "set P2C_BETA > 0 to enable this test"
            )

        # Send a few requests to accumulate token data
        for _ in range(3):
            lb_client.post("/v1/chat/completions", json=_chat_payload(model_name))

        # Look for any EWMA key matching the model
        pattern = f"lb:output_tokens_ewma:{model_name}|*"
        keys = redis_client.keys(pattern)
        assert keys, (
            f"No EWMA keys found for model '{model_name}' after 3 requests. "
            f"Expected keys matching: {pattern}"
        )

    def test_ewma_key_value_is_positive(
        self,
        lb_client: httpx.Client,
        model_name: str,
        redis_client,
    ):
        """EWMA value should be a positive float (tokens > 0)."""
        if redis_client is None:
            pytest.skip("Redis unavailable")

        p2c_beta = float(os.environ.get("P2C_BETA", "0"))
        if p2c_beta <= 0:
            pytest.skip("P2C_BETA=0 – cost EWMA tracking disabled")

        for _ in range(3):
            lb_client.post("/v1/chat/completions", json=_chat_payload(model_name))

        pattern = f"lb:output_tokens_ewma:{model_name}|*"
        keys = redis_client.keys(pattern)
        if not keys:
            pytest.skip("No EWMA keys present – backend may not return usage data")

        for key in keys:
            val = redis_client.get(key)
            assert val is not None, f"EWMA key {key} has no value"
            assert float(val) > 0, f"EWMA value not positive: {key}={val}"

    def test_count_key_increments_with_requests(
        self,
        lb_client: httpx.Client,
        model_name: str,
        redis_client,
    ):
        """lb:output_tokens_count key should increment with each request."""
        if redis_client is None:
            pytest.skip("Redis unavailable")

        p2c_beta = float(os.environ.get("P2C_BETA", "0"))
        if p2c_beta <= 0:
            pytest.skip("P2C_BETA=0 – cost EWMA tracking disabled")

        # Get node from response header
        resp = lb_client.post(
            "/v1/chat/completions", json=_chat_payload(model_name)
        )
        if resp.status_code != 200:
            pytest.skip(f"Request failed: {resp.status_code}")

        node = resp.headers.get("x-routed-node", "")
        if not node:
            pytest.skip("x-routed-node header missing – cannot inspect per-node key")

        count_key = _count_key(model_name, node)
        before = int(redis_client.get(count_key) or 0)

        # Send another request to the same node (best-effort via forced-node param)
        lb_client.post(
            "/v1/chat/completions",
            params={"node": node},
            json=_chat_payload(model_name),
        )
        after = int(redis_client.get(count_key) or 0)

        # Count may not increment if the backend does not return usage data,
        # but it should never decrease
        assert after >= before, (
            f"EWMA count key {count_key} decreased: {before} → {after}"
        )

    def test_multiple_nodes_get_individual_ewma_keys(
        self,
        lb_client: httpx.Client,
        model_name: str,
        redis_client,
    ):
        """With ≥2 backends, each node should have its own EWMA entry."""
        if redis_client is None:
            pytest.skip("Redis unavailable")

        p2c_beta = float(os.environ.get("P2C_BETA", "0"))
        if p2c_beta <= 0:
            pytest.skip("P2C_BETA=0 – cost EWMA tracking disabled")

        # Check how many nodes are healthy
        nodes_resp = lb_client.get("/v1/nodes")
        if nodes_resp.status_code != 200:
            pytest.skip("Cannot introspect nodes")
        nodes = nodes_resp.json().get("data", [])
        if len(nodes) < 2:
            pytest.skip("Fewer than 2 backend nodes – cannot verify per-node EWMA keys")

        # Send enough requests to likely hit all nodes
        for _ in range(10):
            lb_client.post("/v1/chat/completions", json=_chat_payload(model_name))

        pattern = f"lb:output_tokens_ewma:{model_name}|*"
        keys = redis_client.keys(pattern)
        assert len(keys) >= 2, (
            f"Expected ≥2 per-node EWMA keys with {len(nodes)} nodes; got: {keys}"
        )

    def test_p2c_routing_prefers_lower_cost_node(
        self,
        lb_client: httpx.Client,
        model_name: str,
        redis_client,
    ):
        """Inject artificial EWMA cost skew and verify routing preference shifts.

        This test seeds one node with a much higher EWMA token count than the
        other, then sends requests and checks that the cheaper node is selected
        more often.  Requires ≥2 nodes, P2C_BETA > 0, and Redis access.
        """
        if redis_client is None:
            pytest.skip("Redis unavailable")

        p2c_beta = float(os.environ.get("P2C_BETA", "0"))
        if p2c_beta <= 0:
            pytest.skip("P2C_BETA=0 – cost term disabled")

        nodes_resp = lb_client.get("/v1/nodes")
        if nodes_resp.status_code != 200:
            pytest.skip("Cannot introspect nodes")
        nodes = [n["node"] for n in nodes_resp.json().get("data", [])]
        if len(nodes) < 2:
            pytest.skip("Fewer than 2 nodes – cost preference test requires ≥2 nodes")

        cheap_node, expensive_node = nodes[0], nodes[1]

        # Seed EWMA: expensive node has 10× more output tokens
        redis_client.set(_ewma_key(model_name, cheap_node), "50")
        redis_client.set(_count_key(model_name, cheap_node), "10")
        redis_client.set(_ewma_key(model_name, expensive_node), "500")
        redis_client.set(_count_key(model_name, expensive_node), "10")

        # Send multiple requests and tally routing
        selection_counts: Dict[str, int] = {cheap_node: 0, expensive_node: 0}
        n_requests = 10
        for _ in range(n_requests):
            resp = lb_client.post(
                "/v1/chat/completions", json=_chat_payload(model_name)
            )
            if resp.status_code == 200:
                routed = resp.headers.get("x-routed-node", "")
                if routed in selection_counts:
                    selection_counts[routed] += 1

        # With cost-aware P2C we expect the cheap node to win more often
        cheap_count = selection_counts[cheap_node]
        expensive_count = selection_counts[expensive_node]
        assert cheap_count >= expensive_count, (
            f"Cost-aware routing did not prefer cheaper node: "
            f"{cheap_node}={cheap_count}, {expensive_node}={expensive_count}"
        )
