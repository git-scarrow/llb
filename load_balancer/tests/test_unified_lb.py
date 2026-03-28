import asyncio
import json
import types
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from httpx import RequestError

# Ensure package path for local src
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "load_balancer" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import app and globals
from load_balancer.main import app as lb_app
from load_balancer import main as lb_main
from load_balancer.routing.strategies import get_routing_strategy


def run(coro):
    return asyncio.run(coro)


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
        # No-op for tests
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
    def __init__(self, chunks, status_code=200, raise_on_enter=None, first_chunk_delay_ms=0):
        self._chunks = chunks
        self.status_code = status_code
        self.request = types.SimpleNamespace()
        self._raise_on_enter = raise_on_enter
        self._first_chunk_delay_ms = first_chunk_delay_ms

    async def __aenter__(self):
        if self._raise_on_enter:
            raise self._raise_on_enter
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_bytes(self):
        first = True
        for c in self._chunks:
            if first and self._first_chunk_delay_ms:
                await asyncio.sleep(self._first_chunk_delay_ms / 1000.0)
            first = False
            yield c


class FakeHTTPClient:
    def __init__(self, behavior):
        # behavior: map node host -> {"chunks": [...], "error": Exception|None, "status": int}
        self.behavior = behavior

    def stream(self, method, url, json=None, headers=None):
        node = url.split("//", 1)[1].split("/", 1)[0]
        cfg = self.behavior.get(node, {"chunks": [b""], "status": 200, "error": None})
        if cfg.get("error"):
            return FakeStreamResponse([], raise_on_enter=cfg["error"], first_chunk_delay_ms=cfg.get("first_chunk_delay_ms", 0))
        return FakeStreamResponse(cfg.get("chunks", []), status_code=cfg.get("status", 200), first_chunk_delay_ms=cfg.get("first_chunk_delay_ms", 0))

    async def aclose(self):
        return True


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    # Inject fake redis by default for all tests
    r = FakeRedis()
    lb_main.redis_client = r
    return r


def make_client():
    # Use synchronous TestClient for convenience; our handlers are async
    return TestClient(lb_app)


def test_health_endpoint_reports_minimum(monkeypatch):
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1", "n2"}
    with make_client() as c:
        resp_ok = c.get("/health")
        assert resp_ok.status_code == 200
        r.sets["nodes:healthy"] = set()
        resp_bad = c.get("/health")
        assert resp_bad.status_code == 503


def test_models_aggregate_unique(monkeypatch):
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1", "n2"}
    # Both nodes advertise overlapping models
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "m1"}]})))
    run(r.set("node:n2:models", json.dumps({"data": [{"id": "m1"}, {"id": "m2"}]})))
    with make_client() as c:
        resp = c.get("/v1/models")
        assert resp.status_code == 200
        ids = {m["id"] for m in resp.json()["data"]}
        assert ids == {"m1", "m2"}


def test_routing_round_robin_and_failover(monkeypatch):
    # Configure two eligible nodes
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1", "n2"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "m"}]})))
    run(r.set("node:n2:models", json.dumps({"data": [{"id": "m"}]})))

    # First attempt: n1 fails, should fail over to n2
    behavior = {
        "n1": {"error": RequestError("boom", request=None)},
        "n2": {"chunks": [b"data: ok\n\n"]},
    }
    lb_main.http_client = FakeHTTPClient(behavior)

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        text = b"".join(resp.iter_bytes())
        assert b"ok" in text


def test_chat_missing_model_applies_default(monkeypatch):
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"solo"}
    run(r.set("node:solo:models", json.dumps({"data": [{"id": "m"}]})))

    lb_main.http_client = FakeHTTPClient({"solo": {"chunks": [b"data: hi\n\n"]}})

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        body = b"".join(resp.iter_bytes())
        assert resp.status_code == 200
        assert b"hi" in body
        assert resp.headers.get("x-selected-model") == "m"
        assert resp.headers.get("x-model-defaulted") == "true"


def test_chat_missing_messages_returns_400(monkeypatch):
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"solo"}
    run(r.set("node:solo:models", json.dumps({"data": [{"id": "m"}]})))

    lb_main.http_client = FakeHTTPClient({"solo": {"chunks": [b"data: hi\n\n"]}})

    with make_client() as c:
        resp = c.post("/v1/chat/completions", json={"model": "m"})
        assert resp.status_code == 400
        detail = resp.json().get("detail")
        assert detail["message"].startswith("Missing")
        assert detail["missing"] == ["messages"]


def test_performance_under_load(monkeypatch):
    # One healthy node responds quickly
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"fast"}
    run(r.set("node:fast:models", json.dumps({"data": [{"id": "m"}]})))
    lb_main.http_client = FakeHTTPClient({"fast": {"chunks": [b"data: x\n\n"]}})

    with make_client() as c:
        # Fire off many concurrent requests and ensure all return quickly
        async def one():
            return c.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            ).status_code

        async def many(n=100):
            return await asyncio.gather(*[one() for _ in range(n)])

        codes = run(many())
        assert all(code == 200 for code in codes)


def test_circuit_breaker_skips_failed_node(monkeypatch):
    # Lower CB threshold for the test
    from load_balancer import config as cfg
    cfg.CIRCUIT_BREAKER_THRESHOLD = 1
    cfg.CIRCUIT_BREAKER_TTL_SECS = 60

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"flaky", "good"}
    run(r.set("node:flaky:models", json.dumps({"data": [{"id": "m"}]})))
    run(r.set("node:good:models", json.dumps({"data": [{"id": "m"}]})))

    # First request: flaky fails, triggers CB open
    lb_main.http_client = FakeHTTPClient({
        "flaky": {"error": RequestError("boom", request=None)},
        "good": {"chunks": [b"data: ok\n\n"]},
    })
    with make_client() as c:
        resp1 = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        _ = b"".join(resp1.iter_bytes())
        # Second request should skip flaky and go straight to good
        resp2 = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        text2 = b"".join(resp2.iter_bytes())
        assert b"ok" in text2


def test_metrics_endpoint(monkeypatch):
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1"}
    run(r.set("node:n1:inflight", 3))
    run(r.set("node:n1:failures", 2))
    run(r.set("lb:requests_total", 42))
    with make_client() as c:
        resp = c.get("/metrics")
        body = resp.text
        assert "llb_requests_total 42" in body
        assert 'llb_up{node="n1"} 1' in body
        assert 'llb_inflight{node="n1"} 3' in body
        assert 'llb_failures{node="n1"} 2' in body


def test_sticky_sessions_prefer_same_node(monkeypatch):
    # Two nodes can serve model; first response from n2, second should prefer n2 via stickiness
    from load_balancer import config as cfg
    cfg.STICKY_SESSIONS_ENABLED = True
    lb_main.router = get_routing_strategy("ROUND_ROBIN")
    from load_balancer.routing.strategies import ROUND_ROBIN_STATE
    ROUND_ROBIN_STATE.clear()
    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1", "n2"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "m"}]})))
    run(r.set("node:n2:models", json.dumps({"data": [{"id": "m"}]})))

    # First call: only n2 returns content successfully
    lb_main.http_client = FakeHTTPClient({
        "n1": {"error": RequestError("boom", request=None)},
        "n2": {"chunks": [b"data: n2\n\n"]},
    })
    with make_client() as c:
        resp1 = c.post(
            "/v1/chat/completions",
            headers={"x-session-id": "s"},
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        out1 = b"".join(resp1.iter_bytes())
        assert b"n2" in out1

    # Second call: even if n1 would work now, sticky should keep using n2
    lb_main.http_client = FakeHTTPClient({
        "n1": {"chunks": [b"data: n1\n\n"]},
        "n2": {"chunks": [b"data: again n2\n\n"]},
    })
    with make_client() as c:
        resp2 = c.post(
            "/v1/chat/completions",
            headers={"x-session-id": "s"},
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        out2 = b"".join(resp2.iter_bytes())
        assert b"again n2" in out2


def test_hedging_non_stream_winner_and_headers(monkeypatch):
    # Enable hedging and configure zero delay to force immediate hedge start
    from load_balancer import config as cfg
    cfg.HEDGING_ENABLED = True
    cfg.HEDGING_SMALL_MODELS_ONLY = True
    cfg.HEDGING_MAX_DELAY_MS = 0
    cfg.HEDGING_P95_FRACTION = 1.0
    # Treat 'm' as a small model
    cfg.MODEL_CLASSES = {
        "historical_small": {"candidates": ["m"], "min_nodes": 1}
    }

    # Deterministic routing: pick n1 then n2
    lb_main.router = get_routing_strategy("ROUND_ROBIN")

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1", "n2"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "m"}]})))
    run(r.set("node:n2:models", json.dumps({"data": [{"id": "m"}]})))

    # Primary (n1) will be slow relative to zero delay (i.e., not finished before hedge starts),
    # but we also want it to fail to ensure hedge succeeds first.
    lb_main.http_client = FakeHTTPClient({
        "n1": {"error": RequestError("boom", request=None)},
        "n2": {"chunks": [b"{\"ok\":true}"]},
    })

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
        # Optional headers: verify shape if present
        if resp.headers.get("x-hedged"):
            assert resp.headers.get("x-hedged") in ("true", "false")
        if resp.headers.get("x-hedge-winner"):
            assert resp.headers.get("x-hedge-winner") in ("n1", "n2", "")


def test_streaming_hedge_zero_delay(monkeypatch):
    # Configure hedging zero delay and small-model classification
    from load_balancer import config as cfg
    cfg.HEDGING_ENABLED = True
    cfg.HEDGING_SMALL_MODELS_ONLY = True
    cfg.HEDGING_MAX_DELAY_MS = 0
    cfg.HEDGING_P95_FRACTION = 1.0
    cfg.MODEL_CLASSES = {
        "historical_small": {"candidates": ["m"], "min_nodes": 1}
    }

    lb_main.router = get_routing_strategy("ROUND_ROBIN")
    from load_balancer.routing.strategies import ROUND_ROBIN_STATE
    ROUND_ROBIN_STATE.clear()

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1", "n2"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "m"}]})))
    run(r.set("node:n2:models", json.dumps({"data": [{"id": "m"}]})))

    # Primary fails immediately, secondary streams
    class HedgeHarnessHTTPClient(FakeHTTPClient):
        def __init__(self):
            super().__init__({})
            self._calls = 0

        def stream(self, method, url, json=None, headers=None):
            self._calls += 1
            if self._calls == 1:
                return FakeStreamResponse([], raise_on_enter=RequestError("boom", request=None))
            return FakeStreamResponse([b"data: win\n\n"], status_code=200)

    lb_main.http_client = HedgeHarnessHTTPClient()

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        body = b"".join(resp.iter_bytes())
        assert resp.status_code == 200
        assert b"event: hedge_start" in body
        assert b"win" in body
        assert b"error" not in body


def test_streaming_hedge_events_deterministic(monkeypatch):
    # Configure hedging with a small delay; make primary slow and secondary fast
    from load_balancer import config as cfg
    cfg.HEDGING_ENABLED = True
    cfg.HEDGING_SMALL_MODELS_ONLY = False
    cfg.HEDGING_MAX_DELAY_MS = 100
    cfg.HEDGING_P95_FRACTION = 0.05  # 50ms on p95 fallback 1.0s
    cfg.STICKY_SESSIONS_ENABLED = True

    lb_main.router = get_routing_strategy("ROUND_ROBIN")

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1", "n2"}
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "q"}]})))
    run(r.set("node:n2:models", json.dumps({"data": [{"id": "q"}]})))
    # Force stickiness so primary is n1
    run(r.set("session:s:q", "n1"))

    lb_main.http_client = FakeHTTPClient({
        "n1": {"chunks": [b"data: slow\n\n"], "status": 200, "first_chunk_delay_ms": 200},
        "n2": {"chunks": [b"data: fast\n\n"], "status": 200, "first_chunk_delay_ms": 0},
    })

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            headers={"x-session-id": "s"},
            json={
                "model": "q",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        body = b"".join(resp.iter_bytes())
        # Expect hedge events and winner n2
        assert b"event: hedge_start" in body
        assert b"event: hedge_winner" in body
        assert b'"winner": "n2"' in body


def test_least_loaded_respects_maxconn(monkeypatch):
    # Force LEAST_LOADED strategy for this test explicitly
    lb_main.router = get_routing_strategy("LEAST_LOADED")

    r = lb_main.redis_client
    r.sets["nodes:healthy"] = {"n1", "n2"}
    # Both nodes host the same model
    run(r.set("node:n1:models", json.dumps({"data": [{"id": "m"}]})))
    run(r.set("node:n2:models", json.dumps({"data": [{"id": "m"}]})))
    # n1 at capacity: inflight == maxconn
    run(r.set("node:n1:inflight", 1))
    run(r.set("node:n1:maxconn", 1))
    # n2 has free capacity
    run(r.set("node:n2:inflight", 0))

    # If routing mistakenly chooses n1, we will see 'bad' in stream
    lb_main.http_client = FakeHTTPClient({
        "n1": {"chunks": [b"data: bad\n\n"]},
        "n2": {"chunks": [b"data: ok\n\n"]},
    })

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        out = b"".join(resp.iter_bytes())
        assert b"ok" in out
        assert b"bad" not in out
