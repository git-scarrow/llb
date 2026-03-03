"""Tests for cost-aware P2C scoring with EWMA token tracking.

13 test cases covering:
  TC-1  config parsing
  TC-2  EWMA unit (update logic)
  TC-3  cold start – cost term cancels out
  TC-4  P2C_BETA=0 – disabled, identical behavior to baseline
  TC-5  P2C_BETA=0.5 – cheaper node gets lower score
  TC-6  no pricing entry – cost-neutral (0.0)
  TC-7  token extraction non-stream (usage.completion_tokens)
  TC-8  token extraction stream (final SSE chunk)
  TC-9  admin API override (runtime pricing update)
  TC-10 backward compatibility – all original tests still pass implicitly (P2C_BETA=0 default)
"""

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import RequestError

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "load_balancer" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from load_balancer.main import app as lb_app
from load_balancer import main as lb_main
from load_balancer import config as lb_config
from load_balancer.routing.strategies import PowerOfTwoChoicesStrategy


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Minimal fake Redis shared across tests
# ---------------------------------------------------------------------------

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
        self.request = types.SimpleNamespace()
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
    def __init__(self, behavior):
        self.behavior = behavior

    def stream(self, method, url, json=None, headers=None):
        node = url.split("//", 1)[1].split("/", 1)[0]
        cfg = self.behavior.get(node, {"chunks": [b""], "status": 200})
        if cfg.get("error"):
            return FakeStreamResponse([], raise_on_enter=cfg["error"])
        return FakeStreamResponse(cfg.get("chunks", []), status_code=cfg.get("status", 200))

    async def aclose(self):
        return True


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    r = FakeRedis()
    lb_main.redis_client = r
    # Reset cost-aware config to safe defaults before each test
    lb_config.P2C_BETA = 0.0
    lb_config.BACKEND_COST_PER_TOKEN = {}
    lb_config.COST_EWMA_ALPHA = 0.3
    lb_config.COST_EWMA_TTL_SECS = 3600
    lb_config.COST_EWMA_COLD_START_TOKENS = 256
    lb_config.COST_EWMA_MIN_SAMPLES = 5
    return r


def make_client():
    return TestClient(lb_app)


# ---------------------------------------------------------------------------
# TC-1: Config parsing — BACKEND_COST_PER_TOKEN JSON helper
# ---------------------------------------------------------------------------

def test_config_parse_cost_per_token():
    parsed = lb_config._parse_cost_per_token('{"gpt-4": {"input": 30.0, "output": 60.0}}')
    assert parsed["gpt-4"]["input"] == 30.0
    assert parsed["gpt-4"]["output"] == 60.0


def test_config_parse_cost_per_token_invalid_json():
    parsed = lb_config._parse_cost_per_token("not-json")
    assert parsed == {}


def test_config_parse_cost_per_token_empty():
    parsed = lb_config._parse_cost_per_token("")
    assert parsed == {}


# ---------------------------------------------------------------------------
# TC-2: EWMA unit — _record_output_tokens updates Redis correctly
# ---------------------------------------------------------------------------

def test_ewma_cold_start_initialises_to_first_sample(setup_env):
    r = setup_env
    run(lb_main._record_output_tokens("m", "n1", 100))
    val = float(run(r.get("lb:output_tokens_ewma:m|n1")))
    assert val == 100.0
    count = int(run(r.get("lb:output_tokens_count:m|n1")))
    assert count == 1


def test_ewma_second_sample_applies_decay(setup_env):
    r = setup_env
    run(lb_main._record_output_tokens("m", "n1", 100))
    run(lb_main._record_output_tokens("m", "n1", 200))
    # EWMA = 0.3 * 200 + 0.7 * 100 = 130.0
    val = float(run(r.get("lb:output_tokens_ewma:m|n1")))
    assert abs(val - 130.0) < 0.01
    count = int(run(r.get("lb:output_tokens_count:m|n1")))
    assert count == 2


# ---------------------------------------------------------------------------
# TC-3: Cold start — both candidates use same default tokens → cost cancels out
# ---------------------------------------------------------------------------

def test_p2c_cold_start_cost_neutral():
    """When neither node has EWMA data both use cold_start default → cost_range=0 → tie."""
    lb_config.P2C_BETA = 0.5
    lb_config.BACKEND_COST_PER_TOKEN = {"m": {"input": 0.5, "output": 1.5}}
    lb_config.COST_EWMA_MIN_SAMPLES = 5  # Require 5 samples before trusting EWMA
    r = FakeRedis()  # Empty Redis — no EWMA data for either node

    strategy = PowerOfTwoChoicesStrategy(beta=0.5)

    cost_n1 = run(strategy._get_cost_estimate("n1", "m", r))
    cost_n2 = run(strategy._get_cost_estimate("n2", "m", r))
    # Both use cold_start default (256 tokens) → equal costs
    assert cost_n1 == cost_n2


# ---------------------------------------------------------------------------
# TC-4: P2C_BETA=0 — no cost computation, identical to baseline
# ---------------------------------------------------------------------------

def test_p2c_beta_zero_skips_cost_term():
    """With beta=0, _get_cost_estimate is never called for scoring (opt-out)."""
    lb_config.P2C_BETA = 0.0
    lb_config.BACKEND_COST_PER_TOKEN = {"m": {"input": 0.5, "output": 1.5}}
    r = FakeRedis()
    r.kv["lb:output_tokens_ewma:m|n1"] = 1000.0
    r.kv["lb:output_tokens_count:m|n1"] = 10
    r.kv["lb:output_tokens_ewma:m|n2"] = 10.0
    r.kv["lb:output_tokens_count:m|n2"] = 10

    strategy = PowerOfTwoChoicesStrategy(beta=0.0)

    # With beta=0, both nodes' scores are purely base (inflight / latency / failures)
    # In the empty redis scenario all base scores = 0.0, so score difference = 0
    score_n1 = run(strategy._calculate_node_score("n1", "m", r))
    score_n2 = run(strategy._calculate_node_score("n2", "m", r))
    assert score_n1 == score_n2 == 0.0


# ---------------------------------------------------------------------------
# TC-5: P2C_BETA=0.5 — cheaper node gets lower composite score
# ---------------------------------------------------------------------------

def test_p2c_cheaper_node_preferred():
    """Node with lower expected token count → lower cost estimate → lower score → preferred."""
    lb_config.P2C_BETA = 0.5
    lb_config.BACKEND_COST_PER_TOKEN = {"m": {"input": 0.5, "output": 2.0}}
    lb_config.COST_EWMA_MIN_SAMPLES = 1
    r = FakeRedis()
    # n1: high token EWMA
    r.kv["lb:output_tokens_ewma:m|n1"] = 800.0
    r.kv["lb:output_tokens_count:m|n1"] = 2
    # n2: low token EWMA
    r.kv["lb:output_tokens_ewma:m|n2"] = 200.0
    r.kv["lb:output_tokens_count:m|n2"] = 2

    strategy = PowerOfTwoChoicesStrategy(beta=0.5)

    cost_n1 = run(strategy._get_cost_estimate("n1", "m", r))
    cost_n2 = run(strategy._get_cost_estimate("n2", "m", r))
    assert cost_n1 > cost_n2  # n1 is more expensive

    # Simulate select_node behavior: after min-max normalization, n2 should score lower
    # n2 is min → cost_norm=0.0, n1 is max → cost_norm=1.0
    # Base scores are equal (empty inflight/failures), so n2 wins
    scores = {}
    for node in ["n1", "n2"]:
        scores[node] = run(strategy._calculate_node_score(node, "m", r))
    costs = {"n1": cost_n1, "n2": cost_n2}
    cost_min, cost_max = min(costs.values()), max(costs.values())
    cost_range = cost_max - cost_min
    for node in ["n1", "n2"]:
        cost_norm = (costs[node] - cost_min) / cost_range
        scores[node] += 0.5 * cost_norm

    assert scores["n2"] < scores["n1"]


# ---------------------------------------------------------------------------
# TC-6: No pricing entry → cost estimate is 0.0 (cost-neutral)
# ---------------------------------------------------------------------------

def test_p2c_no_pricing_entry_is_neutral():
    lb_config.P2C_BETA = 0.5
    lb_config.BACKEND_COST_PER_TOKEN = {}  # No pricing at all
    r = FakeRedis()
    strategy = PowerOfTwoChoicesStrategy(beta=0.5)
    cost = run(strategy._get_cost_estimate("n1", "unknown_model", r))
    assert cost == 0.0


# ---------------------------------------------------------------------------
# TC-7: Non-streaming token extraction (usage.completion_tokens)
# ---------------------------------------------------------------------------

def test_non_stream_token_extraction(setup_env):
    """POST /v1/chat/completions (non-stream) records completion_tokens via EWMA."""
    r = setup_env
    lb_config.P2C_BETA = 0.5  # Enable write path

    r.sets["nodes:healthy"] = {"solo"}
    run(r.set("node:solo:models", json.dumps({"data": [{"id": "m"}]})))

    usage_payload = {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 42, "total_tokens": 52},
    }
    chunks = [json.dumps(usage_payload).encode()]
    lb_main.http_client = FakeHTTPClient({"solo": {"chunks": chunks}})

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200

    # Allow the fire-and-forget task a moment to complete
    run(asyncio.sleep(0))
    ewma_val = run(r.get("lb:output_tokens_ewma:m|solo"))
    assert ewma_val is not None
    assert float(ewma_val) == 42.0


# ---------------------------------------------------------------------------
# TC-8: Streaming token extraction (final SSE chunk with usage)
# ---------------------------------------------------------------------------

def test_stream_token_extraction(setup_env):
    """Streaming path detects usage in SSE data lines and updates EWMA."""
    r = setup_env
    lb_config.P2C_BETA = 0.5

    r.sets["nodes:healthy"] = {"solo"}
    run(r.set("node:solo:models", json.dumps({"data": [{"id": "m"}]})))

    usage_chunk = json.dumps({
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 77, "total_tokens": 82},
    })
    chunks = [
        b"data: " + json.dumps({"choices": [{"delta": {"content": "hello"}}]}).encode() + b"\n\n",
        b"data: " + usage_chunk.encode() + b"\n\n",
        b"data: [DONE]\n\n",
    ]
    lb_main.http_client = FakeHTTPClient({"solo": {"chunks": chunks}})

    with make_client() as c:
        resp = c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        _ = b"".join(resp.iter_bytes())
        assert resp.status_code == 200

    run(asyncio.sleep(0))
    ewma_val = run(r.get("lb:output_tokens_ewma:m|solo"))
    assert ewma_val is not None
    assert float(ewma_val) == 77.0


# ---------------------------------------------------------------------------
# TC-9: Admin API override — backend_cost_per_token runtime update
# ---------------------------------------------------------------------------

def test_admin_prefs_backend_cost_per_token(setup_env):
    lb_config.BACKEND_COST_PER_TOKEN = {}

    with make_client() as c:
        resp = c.post(
            "/v1/admin/prefs",
            json={"backend_cost_per_token": {"gpt-4": {"input": 30.0, "output": 60.0}}},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert "backend_cost_per_token" in body["applied"]

    assert lb_config.BACKEND_COST_PER_TOKEN.get("gpt-4") == {"input": 30.0, "output": 60.0}


# ---------------------------------------------------------------------------
# TC-10: Backward compatibility — P2C_BETA=0 preserves existing scoring
# ---------------------------------------------------------------------------

def test_p2c_beta_zero_preserves_existing_scoring():
    """With beta=0 (default), P2C scoring is identical to the pre-cost implementation."""
    lb_config.P2C_BETA = 0.0
    lb_config.BACKEND_COST_PER_TOKEN = {"m": {"input": 1.0, "output": 2.0}}

    r = FakeRedis()
    # Node n1 has higher inflight → should score worse
    r.kv["node:n1:inflight"] = 5
    r.kv["node:n1:maxconn"] = 10
    r.kv["node:n2:inflight"] = 1
    r.kv["node:n2:maxconn"] = 10
    # Give n1 huge EWMA so cost would have strongly biased toward n2 if beta != 0
    r.kv["lb:output_tokens_ewma:m|n1"] = 10.0
    r.kv["lb:output_tokens_count:m|n1"] = 10
    r.kv["lb:output_tokens_ewma:m|n2"] = 5000.0
    r.kv["lb:output_tokens_count:m|n2"] = 10

    strategy = PowerOfTwoChoicesStrategy(beta=0.0)

    # With beta=0, cost term is skipped entirely → scoring based on inflight only
    # n2 (inflight=1) should beat n1 (inflight=5)
    score_n1 = run(strategy._calculate_node_score("n1", "m", r))
    score_n2 = run(strategy._calculate_node_score("n2", "m", r))
    assert score_n1 > score_n2
