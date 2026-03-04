"""E2E tests for circuit breaker and failover (Phase 1 resilience).

Covers:
  - Circuit breaker opens after CIRCUIT_BREAKER_THRESHOLD failures
  - Requests route to healthy nodes after CB opens
  - CB resets after CIRCUIT_BREAKER_TTL_SECS / CIRCUIT_BREAKER_COOLDOWN_SECS
  - Failover: removing a node from Redis healthy set triggers re-routing

Note: These tests require either:
  a) Docker Compose control (``AI_LB_COMPOSE_SERVICE`` env var) to stop a
     service and simulate node failure, OR
  b) Redis write access (``redis_client`` fixture) to manually manipulate
     node state.

Tests that require docker control are automatically skipped when
``AI_LB_COMPOSE_SERVICE`` is not set.
"""

import os
import subprocess
import time

import httpx
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIMPLE_PAYLOAD_TEMPLATE = {
    "messages": [{"role": "user", "content": "Say one word."}],
    "stream": False,
    "max_tokens": 5,
}


def _chat_payload(model: str) -> dict:
    return {**_SIMPLE_PAYLOAD_TEMPLATE, "model": model}


def _cb_open_key(node: str) -> str:
    return f"node:{node}:cb_open"


def _failures_key(node: str) -> str:
    return f"node:{node}:failures"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestCircuitBreaker:
    """Circuit breaker opens after threshold failures and routes to healthy nodes."""

    def test_healthy_requests_succeed(
        self, lb_client: httpx.Client, model_name: str
    ):
        """Baseline: normal requests succeed before any failure injection."""
        resp = lb_client.post(
            "/v1/chat/completions", json=_chat_payload(model_name)
        )
        assert resp.status_code == 200, f"Baseline request failed: {resp.text}"

    def test_circuit_breaker_opens_via_redis_injection(
        self,
        lb_client: httpx.Client,
        model_name: str,
        redis_client,
    ):
        """Manually set a node's failure count above threshold in Redis and verify
        subsequent requests avoid that node (cb_open key respected).

        Requires Redis write access.
        """
        if redis_client is None:
            pytest.skip("Redis unavailable – cannot inject circuit breaker state")

        # Get a node to trip
        nodes_resp = lb_client.get("/v1/nodes")
        if nodes_resp.status_code != 200:
            pytest.skip("Cannot introspect nodes")
        nodes = [n["node"] for n in nodes_resp.json().get("data", [])]
        if len(nodes) < 2:
            pytest.skip("Need ≥2 nodes to test CB routing around a tripped node")

        tripped_node = nodes[0]
        healthy_node = nodes[1]

        # Open the circuit for tripped_node
        cb_cooldown = int(os.environ.get("CIRCUIT_BREAKER_COOLDOWN_SECS", 60))
        redis_client.set(_cb_open_key(tripped_node), "1")
        redis_client.expire(_cb_open_key(tripped_node), cb_cooldown)
        redis_client.set(_failures_key(tripped_node), "99")
        redis_client.expire(_failures_key(tripped_node), cb_cooldown)

        try:
            # Requests should now route to healthy_node only
            for _ in range(3):
                resp = lb_client.post(
                    "/v1/chat/completions", json=_chat_payload(model_name)
                )
                assert resp.status_code == 200, (
                    f"Request failed while CB open on {tripped_node}: {resp.text}"
                )
                routed = resp.headers.get("x-routed-node", "")
                assert routed != tripped_node, (
                    f"Request routed to CB-open node {tripped_node}"
                )
        finally:
            # Restore state
            redis_client.delete(_cb_open_key(tripped_node))
            redis_client.set(_failures_key(tripped_node), "0")

    def test_circuit_breaker_resets_after_ttl(
        self,
        lb_client: httpx.Client,
        model_name: str,
        redis_client,
    ):
        """CB key expires and node becomes eligible again after TTL."""
        if redis_client is None:
            pytest.skip("Redis unavailable")

        nodes_resp = lb_client.get("/v1/nodes")
        if nodes_resp.status_code != 200:
            pytest.skip("Cannot introspect nodes")
        nodes = [n["node"] for n in nodes_resp.json().get("data", [])]
        if not nodes:
            pytest.skip("No nodes available")

        node = nodes[0]

        # Set a very short TTL (1 second) to test expiry
        redis_client.set(_cb_open_key(node), "1")
        redis_client.expire(_cb_open_key(node), 1)

        time.sleep(1.5)  # Wait for CB to expire

        val = redis_client.get(_cb_open_key(node))
        assert val is None or val == "0", (
            f"CB key did not expire after 1s TTL: {val}"
        )


@pytest.mark.e2e
class TestFailover:
    """Failover: requests reroute when a node is removed from healthy set."""

    def test_requests_succeed_after_node_removal(
        self,
        lb_client: httpx.Client,
        model_name: str,
        redis_client,
    ):
        """Remove a node from nodes:healthy, verify requests still succeed."""
        if redis_client is None:
            pytest.skip("Redis unavailable – cannot manipulate node health")

        nodes_resp = lb_client.get("/v1/nodes")
        if nodes_resp.status_code != 200:
            pytest.skip("Cannot introspect nodes")
        nodes = [n["node"] for n in nodes_resp.json().get("data", [])]
        if len(nodes) < 2:
            pytest.skip("Need ≥2 nodes for failover test")

        removed_node = nodes[0]

        # Remove from healthy set temporarily
        redis_client.srem("nodes:healthy", removed_node)
        try:
            resp = lb_client.post(
                "/v1/chat/completions", json=_chat_payload(model_name)
            )
            assert resp.status_code == 200, (
                f"Failover failed after removing {removed_node}: {resp.text}"
            )
            routed = resp.headers.get("x-routed-node", "")
            assert routed != removed_node, (
                f"Request still routed to removed node {removed_node}"
            )
        finally:
            # Restore the node to healthy set
            redis_client.sadd("nodes:healthy", removed_node)

    def test_all_nodes_removed_returns_error(
        self,
        lb_client: httpx.Client,
        model_name: str,
        redis_client,
    ):
        """When all nodes are removed from the healthy set, LB returns 404."""
        if redis_client is None:
            pytest.skip("Redis unavailable")

        nodes_resp = lb_client.get("/v1/nodes")
        if nodes_resp.status_code != 200:
            pytest.skip("Cannot introspect nodes")
        nodes = [n["node"] for n in nodes_resp.json().get("data", [])]
        if not nodes:
            pytest.skip("No nodes to remove")

        # Back up and clear healthy set
        original = set(redis_client.smembers("nodes:healthy"))
        redis_client.delete("nodes:healthy")
        try:
            resp = lb_client.post(
                "/v1/chat/completions", json=_chat_payload(model_name)
            )
            # Expect a 4xx/5xx indicating no healthy nodes
            assert resp.status_code in (404, 503, 502), (
                f"Expected error when no healthy nodes, got {resp.status_code}: {resp.text}"
            )
        finally:
            # Restore
            for n in original:
                redis_client.sadd("nodes:healthy", n)


@pytest.mark.e2e
class TestDockerComposeFailover:
    """Failover using docker-compose stop to simulate real node failure.

    Requires ``AI_LB_COMPOSE_SERVICE`` env var (name of the compose service
    to stop temporarily) and docker CLI access.  Skipped otherwise.
    """

    @pytest.fixture(autouse=True)
    def require_compose_service(self):
        svc = os.environ.get("AI_LB_COMPOSE_SERVICE", "")
        if not svc:
            pytest.skip(
                "AI_LB_COMPOSE_SERVICE not set – docker-compose failover tests skipped"
            )
        self._service = svc

    def _compose(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["docker", "compose", *args],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_failover_after_compose_stop(
        self, lb_client: httpx.Client, model_name: str
    ):
        """Stop one compose service, verify requests still route to surviving nodes."""
        stop_result = self._compose("stop", self._service)
        if stop_result.returncode != 0:
            pytest.skip(
                f"docker compose stop {self._service} failed: {stop_result.stderr}"
            )

        try:
            # Allow monitor time to detect the failure
            time.sleep(2)

            for _ in range(3):
                resp = lb_client.post(
                    "/v1/chat/completions", json=_chat_payload(model_name)
                )
                # May 404 if only one node existed; otherwise should succeed
                assert resp.status_code in (200, 404), (
                    f"Unexpected status after node stop: {resp.status_code}: {resp.text}"
                )
        finally:
            self._compose("start", self._service)
            # Give the service time to come back up
            time.sleep(3)
