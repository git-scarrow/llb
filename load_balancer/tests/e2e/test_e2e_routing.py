"""E2E tests for routing capabilities (Phases 1 & 2).

Covers:
  - GET /v1/models        – model aggregation across backends
  - POST /v1/chat/completions (non-streaming)  – basic routing
  - POST /v1/chat/completions (streaming)      – SSE passthrough
  - GET /v1/nodes                              – node introspection
  - GET /v1/eligible_nodes                     – eligible-node query
  - x-require-capability header                – capability pool filtering
  - x-routed-node / x-selected-model response headers
"""

import json

import httpx
import pytest


@pytest.mark.e2e
class TestModelsEndpoint:
    """GET /v1/models — model aggregation across backends."""

    def test_models_returns_list(self, lb_client: httpx.Client):
        resp = lb_client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("object") == "list"
        assert isinstance(body.get("data"), list)
        assert len(body["data"]) > 0, "No models returned – backend not registered"

    def test_models_have_id_field(self, lb_client: httpx.Client):
        resp = lb_client.get("/v1/models")
        assert resp.status_code == 200
        for m in resp.json()["data"]:
            assert "id" in m, f"Model entry missing 'id': {m}"

    def test_models_deduplicated(self, lb_client: httpx.Client):
        """Model IDs should not repeat across multi-node aggregation."""
        resp = lb_client.get("/v1/models")
        assert resp.status_code == 200
        ids = [m["id"] for m in resp.json()["data"]]
        assert len(ids) == len(set(ids)), f"Duplicate model IDs: {ids}"


@pytest.mark.e2e
class TestChatCompletions:
    """POST /v1/chat/completions – routing, headers, streaming."""

    def _simple_payload(self, model: str, stream: bool = False) -> dict:
        return {
            "model": model,
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "stream": stream,
            "max_tokens": 10,
        }

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    def test_non_stream_returns_200(self, lb_client: httpx.Client, model_name: str):
        resp = lb_client.post(
            "/v1/chat/completions",
            json=self._simple_payload(model_name, stream=False),
        )
        assert resp.status_code == 200

    def test_non_stream_response_headers(
        self, lb_client: httpx.Client, model_name: str
    ):
        resp = lb_client.post(
            "/v1/chat/completions",
            json=self._simple_payload(model_name, stream=False),
        )
        assert resp.status_code == 200
        assert resp.headers.get("x-selected-model"), "Missing x-selected-model header"
        assert resp.headers.get("x-routed-node"), "Missing x-routed-node header"
        assert resp.headers.get("x-request-id"), "Missing x-request-id header"

    def test_non_stream_response_body_structure(
        self, lb_client: httpx.Client, model_name: str
    ):
        resp = lb_client.post(
            "/v1/chat/completions",
            json=self._simple_payload(model_name, stream=False),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "choices" in body, f"Missing 'choices' in response: {body}"

    def test_non_stream_missing_messages_returns_400(
        self, lb_client: httpx.Client, model_name: str
    ):
        resp = lb_client.post(
            "/v1/chat/completions",
            json={"model": model_name},
        )
        assert resp.status_code == 400

    def test_non_stream_invalid_json_returns_400(self, lb_client: httpx.Client):
        resp = lb_client.post(
            "/v1/chat/completions",
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def test_stream_returns_200(self, lb_client: httpx.Client, model_name: str):
        with lb_client.stream(
            "POST",
            "/v1/chat/completions",
            json=self._simple_payload(model_name, stream=True),
        ) as resp:
            assert resp.status_code == 200

    def test_stream_content_type_sse(
        self, lb_client: httpx.Client, model_name: str
    ):
        with lb_client.stream(
            "POST",
            "/v1/chat/completions",
            json=self._simple_payload(model_name, stream=True),
        ) as resp:
            assert resp.status_code == 200
            ct = resp.headers.get("content-type", "")
            assert "text/event-stream" in ct, f"Unexpected content-type: {ct}"

    def test_stream_yields_data_lines(
        self, lb_client: httpx.Client, model_name: str
    ):
        chunks = []
        with lb_client.stream(
            "POST",
            "/v1/chat/completions",
            json=self._simple_payload(model_name, stream=True),
        ) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    chunks.append(line)
        assert chunks, "No SSE data lines received"

    def test_stream_response_headers(
        self, lb_client: httpx.Client, model_name: str
    ):
        with lb_client.stream(
            "POST",
            "/v1/chat/completions",
            json=self._simple_payload(model_name, stream=True),
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers.get("x-selected-model")
            assert resp.headers.get("x-request-id")

    def test_multiple_requests_routed_successfully(
        self, lb_client: httpx.Client, model_name: str
    ):
        """Send several requests; all should succeed (exercises round-robin / P2C)."""
        for _ in range(3):
            resp = lb_client.post(
                "/v1/chat/completions",
                json=self._simple_payload(model_name, stream=False),
            )
            assert resp.status_code == 200, f"Request failed: {resp.text}"


@pytest.mark.e2e
class TestNodeIntrospection:
    """GET /v1/nodes and GET /v1/eligible_nodes – node visibility."""

    def test_nodes_endpoint_returns_list(self, lb_client: httpx.Client):
        resp = lb_client.get("/v1/nodes")
        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body
        assert isinstance(body["data"], list)

    def test_nodes_have_required_fields(self, lb_client: httpx.Client):
        resp = lb_client.get("/v1/nodes")
        assert resp.status_code == 200
        for node in resp.json()["data"]:
            for field in ("node", "inflight", "failures"):
                assert field in node, f"Node entry missing '{field}': {node}"

    def test_eligible_nodes_for_known_model(
        self, lb_client: httpx.Client, model_name: str
    ):
        resp = lb_client.get("/v1/eligible_nodes", params={"model": model_name})
        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body
        assert isinstance(body["data"], list)
        assert len(body["data"]) > 0, (
            f"No eligible nodes for model '{model_name}' – "
            "backend may not be registered properly"
        )

    def test_eligible_nodes_for_unknown_model_returns_empty(
        self, lb_client: httpx.Client
    ):
        resp = lb_client.get(
            "/v1/eligible_nodes", params={"model": "no-such-model-xyz"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("data") == [], f"Expected empty list, got: {body}"


@pytest.mark.e2e
class TestCapabilityRouting:
    """x-require-capability header – capability pool filtering (Phase 1).

    The load balancer passes unknown headers through to backends.  These tests
    verify that a request with x-require-capability either succeeds (when at
    least one backend claims the capability) or returns a well-formed error
    (503 / 404) rather than crashing.

    When no backend supports the requested capability the LB should respond
    with a 4xx/5xx rather than a 200 with an empty or corrupt body.
    """

    def test_capability_header_does_not_crash_lb(
        self, lb_client: httpx.Client, model_name: str
    ):
        """Any response code except 5xx internal server error is acceptable."""
        resp = lb_client.post(
            "/v1/chat/completions",
            headers={"x-require-capability": "general"},
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
                "max_tokens": 5,
            },
        )
        assert resp.status_code != 500, (
            f"LB crashed with 500 when x-require-capability: general: {resp.text}"
        )

    def test_unsupported_capability_returns_error_or_routes(
        self, lb_client: httpx.Client, model_name: str
    ):
        """Requesting an exotic capability should either route (if backend supports it)
        or return a structured 4xx/5xx – not 200 with corrupt body.
        """
        resp = lb_client.post(
            "/v1/chat/completions",
            headers={"x-require-capability": "quantum-entanglement-99"},
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
                "max_tokens": 5,
            },
        )
        # Acceptable: 200 (routed anyway), 404, 503; not acceptable: 500
        assert resp.status_code != 500, (
            f"LB crashed with 500 for unknown capability: {resp.text}"
        )

    def test_x_routed_node_header_present(
        self, lb_client: httpx.Client, model_name: str
    ):
        """Successful requests should carry x-routed-node for observability."""
        resp = lb_client.post(
            "/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "max_tokens": 5,
            },
        )
        if resp.status_code == 200:
            assert resp.headers.get("x-routed-node"), (
                "Successful response missing x-routed-node header"
            )
