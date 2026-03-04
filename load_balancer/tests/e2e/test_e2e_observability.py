"""E2E tests for observability endpoints (Phase 1).

Covers:
  - GET /health  – returns 200 when ≥1 backend healthy; 503 otherwise
  - GET /metrics – Prometheus text format; key counters present
  - Counter increments: requests_total, inflight
"""

import httpx
import pytest


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestHealthEndpoint:
    """GET /health – load balancer liveness and node readiness."""

    def test_health_returns_200_when_healthy(self, lb_client: httpx.Client):
        """With at least one live backend, /health must return 200."""
        resp = lb_client.get("/health")
        assert resp.status_code == 200, (
            f"Expected 200 from /health, got {resp.status_code}: {resp.text}"
        )

    def test_health_body_has_status_field(self, lb_client: httpx.Client):
        resp = lb_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body, f"Missing 'status' in /health body: {body}"
        assert body["status"] == "healthy", (
            f"Expected status='healthy', got: {body['status']}"
        )

    def test_health_body_has_nodes_found(self, lb_client: httpx.Client):
        resp = lb_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "nodes_found" in body, f"Missing 'nodes_found': {body}"
        assert int(body["nodes_found"]) >= 1, (
            f"nodes_found should be ≥1 with live backend, got: {body['nodes_found']}"
        )

    def test_health_body_has_minimum_required(self, lb_client: httpx.Client):
        resp = lb_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "minimum_required" in body, f"Missing 'minimum_required': {body}"

    def test_health_content_type_is_json(self, lb_client: httpx.Client):
        resp = lb_client.get("/health")
        ct = resp.headers.get("content-type", "")
        assert "application/json" in ct, f"Unexpected content-type: {ct}"


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestMetricsEndpoint:
    """GET /metrics – Prometheus text format exposition."""

    def test_metrics_returns_200(self, lb_client: httpx.Client):
        resp = lb_client.get("/metrics")
        assert resp.status_code == 200, (
            f"GET /metrics failed: {resp.status_code}: {resp.text}"
        )

    def test_metrics_content_type_text(self, lb_client: httpx.Client):
        resp = lb_client.get("/metrics")
        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "")
        assert "text/plain" in ct, f"Expected text/plain content-type, got: {ct}"

    def test_metrics_contains_requests_total(self, lb_client: httpx.Client):
        resp = lb_client.get("/metrics")
        assert resp.status_code == 200
        assert "ai_lb_requests_total" in resp.text, (
            "Missing 'ai_lb_requests_total' counter in /metrics"
        )

    def test_metrics_contains_inflight_gauge(self, lb_client: httpx.Client):
        resp = lb_client.get("/metrics")
        assert resp.status_code == 200
        assert "ai_lb_inflight" in resp.text, (
            "Missing 'ai_lb_inflight' gauge in /metrics"
        )

    def test_metrics_contains_failures_gauge(self, lb_client: httpx.Client):
        resp = lb_client.get("/metrics")
        assert resp.status_code == 200
        assert "ai_lb_failures" in resp.text, (
            "Missing 'ai_lb_failures' gauge in /metrics"
        )

    def test_metrics_contains_up_gauge(self, lb_client: httpx.Client):
        resp = lb_client.get("/metrics")
        assert resp.status_code == 200
        assert "ai_lb_up" in resp.text, (
            "Missing 'ai_lb_up' gauge in /metrics"
        )

    def test_metrics_has_help_and_type_annotations(self, lb_client: httpx.Client):
        resp = lb_client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "# HELP" in text, "Missing # HELP lines in /metrics output"
        assert "# TYPE" in text, "Missing # TYPE lines in /metrics output"

    def test_metrics_requests_total_increments_after_request(
        self, lb_client: httpx.Client, model_name: str
    ):
        """Send a chat request, then verify requests_total counter increased."""

        def _parse_counter(text: str) -> int:
            for line in text.splitlines():
                if line.startswith("ai_lb_requests_total ") and "{" not in line:
                    try:
                        return int(line.split()[-1])
                    except (ValueError, IndexError):
                        pass
            return -1

        before_resp = lb_client.get("/metrics")
        assert before_resp.status_code == 200
        before = _parse_counter(before_resp.text)

        # Trigger a request
        lb_client.post(
            "/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "max_tokens": 5,
            },
        )

        after_resp = lb_client.get("/metrics")
        assert after_resp.status_code == 200
        after = _parse_counter(after_resp.text)

        if before >= 0 and after >= 0:
            assert after > before, (
                f"ai_lb_requests_total did not increment: before={before}, after={after}"
            )

    def test_metrics_up_node_present_for_healthy_nodes(
        self, lb_client: httpx.Client
    ):
        """Each healthy node should appear in the ai_lb_up metric."""
        nodes_resp = lb_client.get("/v1/nodes")
        if nodes_resp.status_code != 200:
            pytest.skip("Cannot introspect nodes")
        nodes = [n["node"] for n in nodes_resp.json().get("data", [])]

        metrics_resp = lb_client.get("/metrics")
        assert metrics_resp.status_code == 200
        metrics_text = metrics_resp.text

        for node in nodes:
            assert f'node="{node}"' in metrics_text, (
                f"Node '{node}' missing from /metrics output"
            )


# ---------------------------------------------------------------------------
# Observability integration
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestObservabilityIntegration:
    """Cross-endpoint observability checks."""

    def test_health_and_metrics_agree_on_node_count(
        self, lb_client: httpx.Client
    ):
        """nodes_found in /health should match node entries in /metrics."""
        health_resp = lb_client.get("/health")
        metrics_resp = lb_client.get("/metrics")
        assert health_resp.status_code == 200
        assert metrics_resp.status_code == 200

        nodes_found = int(health_resp.json().get("nodes_found", 0))
        # Count ai_lb_up lines with value 1 in metrics
        up_count = sum(
            1
            for line in metrics_resp.text.splitlines()
            if line.startswith("ai_lb_up{") and line.endswith(" 1")
        )
        assert up_count == nodes_found, (
            f"/health reports {nodes_found} nodes but /metrics shows {up_count} up"
        )

    def test_embeddings_endpoint_reachable(self, lb_client: httpx.Client):
        """GET /v1/embeddings with a valid payload should not return 500."""
        # Embeddings require a model that supports it; we just verify the endpoint exists
        resp = lb_client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-3-small",
                "input": "Hello world",
            },
        )
        # 200 = success, 404 = no nodes/model not found, 400 = validation error
        # Any of these are acceptable; 500 is not
        assert resp.status_code != 500, (
            f"Embeddings endpoint crashed with 500: {resp.text}"
        )
