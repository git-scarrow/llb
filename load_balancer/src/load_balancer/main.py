import httpx
import math
import random
import redis.asyncio as redis
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uuid
import asyncio
from contextlib import asynccontextmanager

from . import config
from .constants import METRIC_PREFIX
from .request_validation import sanitize_chat_request, sanitize_embeddings_request
from .routing.strategies import get_routing_strategy, ComplexityRoutingStrategy
from .execution import (
    ExecutionMode,
    ExecutionConfig,
    BackendResult,
    ConsensusResult,
    ExecutionEngine,
    PlanTask,
    PlanResult,
)
from .execution.modes import (
    execute_plan_stream,
    collect_plan_result,
    plan_result_to_openai_response,
    PlanEvent,
)
from .providers import get_adapter
from .config_validation import validate_config

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, http_client, router
    validate_config()
    logger.info("Load balancer starting up...")
    if redis_client is None:
        # Create Redis client with reasonable timeouts
        redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=2,  # 2 second connection timeout
            socket_timeout=5,  # 5 second socket timeout
            retry_on_timeout=True,
        )
        # Try quick connection test, but don't block startup if it fails
        try:
            await asyncio.wait_for(redis_client.ping(), timeout=2.0)
            logger.info("Successfully connected to Redis")
        except Exception as e:
            logger.warning(f"Redis not immediately available: {e}. Will retry in background...")
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=httpx.Timeout(config.REQUEST_TIMEOUT_SECS))
    if router is None:
        try:
            router = get_routing_strategy(config.ROUTING_STRATEGY)
            logger.info("Using routing strategy: %s", config.ROUTING_STRATEGY)
        except ValueError as e:
            logger.error("Error selecting strategy: %s", e)
            router = get_routing_strategy("ROUND_ROBIN")
            logger.info("Defaulting to routing strategy: ROUND_ROBIN")
    try:
        yield
    finally:
        if redis_client is not None:
            try:
                await redis_client.close()
            except Exception:
                pass
        if http_client is not None:
            try:
                await http_client.aclose()
            except Exception:
                pass
        logger.info("Load balancer shut down.")


app = FastAPI(title="Large Language Balancer", lifespan=lifespan)
redis_client = None
http_client = None
router = None
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# OpenAI-compatible error responses
# ---------------------------------------------------------------------------

_ERROR_TYPE_MAP = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
    500: "server_error",
    502: "server_error",
    503: "service_unavailable",
}


@app.exception_handler(HTTPException)
async def _openai_error_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict):
        message = detail.get("message", str(detail))
    else:
        message = str(detail)

    body: dict = {
        "error": {
            "message": message,
            "type": _ERROR_TYPE_MAP.get(exc.status_code, "api_error"),
            "param": None,
            "code": None,
        }
    }
    if isinstance(detail, dict):
        for k, v in detail.items():
            if k != "message":
                body["error"][k] = v

    return JSONResponse(
        status_code=exc.status_code,
        content=body,
        headers=dict(exc.headers) if exc.headers else None,
    )

# Helper to resolve circuit breaker cooldown consistently across legacy and new config
def _cb_cooldown_secs() -> int:
    try:
        return int(getattr(config, "CIRCUIT_BREAKER_COOLDOWN_SECS", config.CIRCUIT_BREAKER_TTL_SECS))
    except Exception:
        return 60

async def _series_p95(model: str, node: str) -> float:
    """Approximate p95 from cumulative histogram buckets stored in Redis.
    Returns 0.0 if no data.
    """
    try:
        buckets = _LAT_BUCKETS
        series_key = f"{model}|{node}"
        cumulative_counts = []
        total_count = 0
        for le in buckets:
            val = await redis_client.get(f"lb:latency_bucket:{series_key}:{le}")
            count = int(val) if val else 0
            cumulative_counts.append(count)
            if le == float("inf"):
                total_count = count
        if total_count == 0:
            return 0.0
        p95_target = total_count * 0.95
        for i, count in enumerate(cumulative_counts):
            if count >= p95_target:
                if i == 0:
                    return buckets[i] * 0.5
                lower = buckets[i-1]
                upper = buckets[i]
                prev = cumulative_counts[i-1]
                rng = max(1, count - prev)
                pos = (p95_target - prev) / rng
                if upper == float("inf"):
                    return lower * 1.1  # Slight bump if in inf bucket
                return lower + pos * (upper - lower)
        return 0.0
    except Exception:
        return 0.0

async def _series_count(model: str, node: str) -> int:
    try:
        series_key = f"{model}|{node}"
        val = await redis_client.get(f"lb:latency_count:{series_key}")
        return int(val) if val else 0
    except Exception:
        return 0

async def _estimate_pool_p95(model: str, nodes: list[str]) -> float:
    """Estimate p95 latency for a model across a pool using median of node p95s
    that have sufficient samples; fallback to 1.0s if unknown."""
    try:
        vals = []
        for n in nodes:
            cnt = await _series_count(model, n)
            if cnt >= getattr(config, "ELIGIBILITY_MIN_P95_SAMPLES", 20):
                p = await _series_p95(model, n)
                if p:
                    vals.append(p)
        if not vals:
            return 1.0
        vals.sort()
        m = len(vals) // 2
        return vals[m] if len(vals) % 2 else (vals[m-1] + vals[m]) / 2.0
    except Exception:
        return 1.0

def _is_small_model(model: str) -> bool:
    try:
        classes = getattr(config, "MODEL_CLASSES", {}) or {}
        small = classes.get("historical_small", {})
        return model in set(small.get("candidates", []))
    except Exception:
        return False

async def get_eligible_nodes(model_name: str, include_cloud: bool = True):
    """Find healthy nodes that advertise the model and meet eligibility: not tripped, not rate-limited, and under p95 threshold.

    When include_cloud=True, also includes cloud backends that support the model.
    """
    healthy_nodes = sorted(await redis_client.smembers("nodes:healthy"))
    eligible_nodes = []
    for node in healthy_nodes:
        # Skip nodes on open circuit for this model (or node-wide)
        if await _is_circuit_open(node, model_name):
            continue
        # Skip nodes that are currently rate-limited (429 backoff)
        if await _is_rate_limited(node):
            continue
        models_json = await redis_client.get(f"node:{node}:models")
        if not models_json:
            continue
        try:
            models = json.loads(models_json).get("data", [])
        except Exception:
            models = []
        # Either the node advertises the model in /v1/models OR monitor warmed it and marked supports:{model}
        supports_key = f"node:{node}:supports:{model_name}"
        supports = await redis_client.get(supports_key)
        if any(m.get("id") == model_name for m in models) or (supports is not None and supports != "0"):
            eligible_nodes.append(node)

    # Enforce p95 threshold only when multiple nodes serve this model;
    # with a single node, filtering it out just guarantees failure.
    if len(eligible_nodes) >= 2:
        try:
            max_p95 = float(getattr(config, "MAX_P95_LATENCY_SECS", 0) or 0)
        except Exception:
            max_p95 = 0.0
        if max_p95 > 0:
            filtered = []
            for node in eligible_nodes:
                cnt = await _series_count(model_name, node)
                if cnt >= getattr(config, "ELIGIBILITY_MIN_P95_SAMPLES", 20):
                    p95 = await _series_p95(model_name, node)
                    if p95 and p95 > max_p95:
                        continue
                filtered.append(node)
            # Keep at least one node — don't filter all of them out
            if filtered:
                eligible_nodes = filtered

    # Add eligible cloud backends
    if include_cloud:
        cloud_nodes = await _get_eligible_cloud_backends(model_name)
        eligible_nodes.extend(cloud_nodes)

    return eligible_nodes

async def _aggregate_models() -> list:
    """Return a list of unique model dicts aggregated across healthy nodes."""
    healthy_nodes = sorted(await redis_client.smembers("nodes:healthy"))
    all_models = {}
    for node in healthy_nodes:
        models_json = await redis_client.get(f"node:{node}:models")
        if models_json:
            try:
                models = json.loads(models_json).get("data", [])
            except Exception:
                models = []
            for model in models:
                mid = model.get("id")
                if mid and mid not in all_models:
                    all_models[mid] = model
    return list(all_models.values())

async def _aggregate_models_by_node() -> dict:
    """Return mapping node -> list of model ids for healthy nodes."""
    healthy_nodes = sorted(await redis_client.smembers("nodes:healthy"))
    out = {}
    for node in healthy_nodes:
        models_json = await redis_client.get(f"node:{node}:models")
        ids = []
        if models_json:
            try:
                models = json.loads(models_json).get("data", [])
                ids = [m.get("id") for m in models if m.get("id")]
            except Exception:
                ids = []
        out[node] = ids
    return out

async def _model_availability(model_id: str) -> dict:
    """Return availability diagnostics for a single model across healthy nodes.

    Returns dict with keys: eligible_nodes, total_nodes, status, unavailable_reasons.
    """
    healthy_nodes = sorted(await redis_client.smembers("nodes:healthy"))
    total_nodes = 0
    eligible_nodes = 0
    reasons: list[str] = []

    for node in healthy_nodes:
        models_json = await redis_client.get(f"node:{node}:models")
        if not models_json:
            continue
        try:
            models = json.loads(models_json).get("data", [])
        except Exception:
            continue
        supports_key = f"node:{node}:supports:{model_id}"
        supports = await redis_client.get(supports_key)
        has_model = any(m.get("id") == model_id for m in models) or (supports is not None and supports != "0")
        if not has_model:
            continue
        total_nodes += 1

        # Check circuit breaker (model-scoped then node-wide)
        if await _is_circuit_open(node, model_id):
            reasons.append(f"circuit_open:{node}")
            continue
        # Check rate limit
        if await _is_rate_limited(node):
            reasons.append(f"rate_limited:{node}")
            continue
        eligible_nodes += 1

    if total_nodes == 0:
        status = "unavailable"
    elif eligible_nodes == 0:
        status = "unavailable"
    elif eligible_nodes < total_nodes:
        status = "degraded"
    else:
        status = "available"

    return {
        "eligible_nodes": eligible_nodes,
        "total_nodes": total_nodes,
        "status": status,
        "unavailable_reasons": reasons,
    }


async def _routing_headers(model_name: str, eligible_count: int) -> dict:
    """Build x-llb-* routing diagnostic headers."""
    avail = await _model_availability(model_name)
    return {
        "x-llb-eligible-nodes": str(eligible_count),
        "x-llb-total-nodes": str(avail["total_nodes"]),
        "x-llb-routing-strategy": getattr(config, "ROUTING_STRATEGY", "P2C"),
    }


def _is_model_sentinel(name: Optional[str]) -> bool:
    if not name:
        return False
    low = str(name).strip().lower()
    return low in config.MODEL_SENTINELS or low == config.LM_MODEL_SENTINEL

async def _resolve_auto_model(prefer_intersection: bool = False) -> Optional[str]:
    """Choose a model for auto/default using class fallbacks and min_nodes.
    Order classes by LM_PREFER_SMALL; within each class, pick the first candidate with
    ≥ AUTO_MIN_NODES eligible nodes. If none and STRICT_AUTO_MODE is false, allow ≥1.
    Falls back to previous union/intersection strategy if needed.
    """
    by_node = await _aggregate_models_by_node()
    if not by_node:
        return None

    async def _eligible_count(mid: str) -> int:
        nodes = await get_eligible_nodes(mid, include_cloud=False)
        return len(nodes)

    classes_cfg = getattr(config, "MODEL_CLASSES", {}) or {}
    if classes_cfg:
        # Determine class order
        prefer_small = bool(getattr(config, "LB_PREFER_SMALL", False))
        ordered = []
        if prefer_small:
            for name in ("historical_small", "historical_medium"):
                if name in classes_cfg:
                    ordered.append(name)
        else:
            for name in ("historical_medium", "historical_small"):
                if name in classes_cfg:
                    ordered.append(name)
        # Add any remaining classes not explicitly ordered
        for cname in classes_cfg.keys():
            if cname not in ordered:
                ordered.append(cname)

        # Try classes in order
        for cname in ordered:
            c = classes_cfg.get(cname) or {}
            candidates = c.get("candidates", [])
            min_nodes = int(c.get("min_nodes") or getattr(config, "AUTO_MIN_NODES", 2))
            # Strict pass
            for mid in candidates:
                if (await _eligible_count(mid)) >= min_nodes:
                    return mid
            # Soft pass if not strict
            if not getattr(config, "STRICT_AUTO_MODE", False):
                for mid in candidates:
                    if (await _eligible_count(mid)) >= 1:
                        return mid

    # Back-compat: previous behavior (preferred models → weights → union/intersection)
    ids: List[str] = []
    if prefer_intersection or config.AUTO_MODEL_STRATEGY.lower() == "intersection_first":
        sets = [set(v) for v in by_node.values() if v]
        inter = set.intersection(*sets) if sets else set()
        ids = list(inter) if inter else []
    if not ids:
        seen = set()
        for v in by_node.values():
            for mid in v:
                if mid not in seen:
                    seen.add(mid)
                    ids.append(mid)
    # Filter union by availability (≥1 eligible node) to avoid mapping to dead models
    filtered: List[str] = []
    for mid in ids:
        try:
            if (await get_eligible_nodes(mid)):
                filtered.append(mid)
        except Exception:
            continue
    ids = filtered
    # Prefer configured priorities
    for want in getattr(config, "PREFERRED_MODELS", []):
        if want in ids:
            return want
    try:
        weights = dict(getattr(config, "MODEL_WEIGHTS", {}))
        wjson = await redis_client.get("lb:model_weights")
        if wjson:
            weights.update(json.loads(wjson))
        if weights:
            best_id = None
            best_w = None
            for mid in ids:
                w = weights.get(mid)
                if w is not None and (best_w is None or w > best_w):
                    best_w = w
                    best_id = mid
            if best_id is not None:
                return best_id
    except Exception:
        pass
    return ids[0] if ids else None

@app.get("/v1/nodes")
async def list_nodes():
    healthy_nodes = await redis_client.smembers("nodes:healthy")
    out = []
    for n in healthy_nodes:
        inflight = await redis_client.get(f"node:{n}:inflight")
        failures = await redis_client.get(f"node:{n}:failures")
        maxconn = await redis_client.get(f"node:{n}:maxconn")
        out.append({
            "node": n,
            "inflight": int(inflight or 0),
            "failures": int(failures or 0),
            "maxconn": int(maxconn or 0),
        })
    return {"data": out}

@app.get("/v1/eligible_nodes")
async def list_eligible_nodes(model: str):
    return {"data": await get_eligible_nodes(model)}

@app.get("/v1/debug/eligible")
async def debug_eligible(model: str):
    """Debug endpoint: shows how eligibility is computed per node."""
    healthy_nodes = sorted(await redis_client.smembers("nodes:healthy"))
    details = []
    for node in healthy_nodes:
        item = {"node": node, "has_model": False, "cb_open": False, "cb_model_open": False, "p95": None, "skipped": False, "reason": ""}
        try:
            models_json = await redis_client.get(f"node:{node}:models")
            models = json.loads(models_json).get("data", []) if models_json else []
            has = any(m.get("id") == model for m in models)
            item["has_model"] = has
            # Node-wide CB
            try:
                nv = await redis_client.get(f"node:{node}:cb_open")
                item["cb_open"] = nv is not None and nv != "0"
            except Exception:
                pass
            # Model-scoped CB
            try:
                mv = await redis_client.get(f"node:{node}:model:{model}:cb_open")
                item["cb_model_open"] = mv is not None and mv != "0"
            except Exception:
                pass
            p95 = await _series_p95(model, node)
            item["p95"] = p95
            # Apply same logic as get_eligible_nodes
            if not has:
                item["skipped"] = True
                item["reason"] = "model_missing"
            elif item["cb_open"] or item["cb_model_open"]:
                item["skipped"] = True
                item["reason"] = "circuit_open"
            else:
                max_p95 = float(getattr(config, "MAX_P95_LATENCY_SECS", 0) or 0)
                if max_p95 > 0 and p95 and p95 > max_p95:
                    item["skipped"] = True
                    item["reason"] = f"p95>{max_p95}"
        except Exception as e:
            item["skipped"] = True
            item["reason"] = f"error:{e}"
        details.append(item)
    eligible = [d["node"] for d in details if not d["skipped"] and d["has_model"] and not d["cb_open"] and not d["cb_model_open"]]
    return {"healthy": healthy_nodes, "eligible": eligible, "details": details}

async def _inc_inflight(node: str):
    try:
        await redis_client.incrby(f"node:{node}:inflight", 1)
    except Exception:
        pass

async def _dec_inflight(node: str):
    try:
        await redis_client.incrby(f"node:{node}:inflight", -1)
    except Exception:
        pass

def _describe_upstream_error(e: Exception) -> str:
    """Extract actionable detail from upstream exceptions for diagnostics."""
    parts = [type(e).__name__]
    if isinstance(e, httpx.HTTPStatusError):
        resp = getattr(e, "response", None)
        if resp is not None:
            parts.append(f"status={resp.status_code}")
            try:
                body = resp.text[:300]
            except Exception:
                body = ""
            if body:
                parts.append(f"body={body!r}")
    elif isinstance(e, httpx.RequestError):
        # ConnectError, TimeoutException, etc.
        cause = e.__cause__ or e.__context__
        if cause:
            parts.append(str(cause))
        elif str(e):
            parts.append(str(e))
    else:
        msg = str(e)
        if msg:
            parts.append(msg)
    return " | ".join(parts)


class CapacityError(Exception):
    def __init__(self, scope: str = "node"):
        self.scope = scope
        super().__init__(f"{scope} at capacity")


class RateLimitError(Exception):
    """Raised when a backend returns HTTP 429 rate limit response."""
    def __init__(self, backend: str, retry_after: Optional[float] = None):
        self.backend = backend
        self.retry_after = retry_after
        super().__init__(f"Rate limited by {backend}")

async def _acquire_slot(node: str) -> bool:
    """Atomically increment inflight and enforce optional maxconn.
    If maxconn is set and would be exceeded, revert and return False.
    """
    key = f"node:{node}:inflight"
    try:
        new_val = await redis_client.incrby(key, 1)
        max_val = await redis_client.get(f"node:{node}:maxconn")
        if max_val not in (None, "", "0"):
            try:
                m = int(max_val)
            except Exception:
                m = None
            if m is not None and int(new_val) > m:
                # revert and signal capacity reached
                await redis_client.incrby(key, -1)
                return False
        return True
    except Exception:
        # On any Redis issue, allow (fail-open) to avoid total outage
        return True

async def _acquire_model_slot(model: str) -> bool:
    key = f"model:{model}:inflight"
    try:
        new_val = await redis_client.incrby(key, 1)
        max_val = await redis_client.get(f"model:{model}:maxconn")
        if max_val not in (None, "", "0"):
            try:
                m = int(max_val)
            except Exception:
                m = None
            if m is not None and int(new_val) > m:
                await redis_client.incrby(key, -1)
                return False
        return True
    except Exception:
        return True

async def _dec_model(model: str):
    try:
        await redis_client.incrby(f"model:{model}:inflight", -1)
    except Exception:
        pass

async def _penalize_failure(node: str):
    try:
        key = f"node:{node}:failures"
        await redis_client.incrby(key, 1)
        await redis_client.expire(key, config.FAILURE_PENALTY_TTL_SECS)
    except Exception:
        pass

async def _record_failure(node: str, model: str = ""):
    cooldown = _cb_cooldown_secs()
    try:
        # Always increment node-wide failures
        key = f"node:{node}:failures"
        failures = await redis_client.incrby(key, 1)
        await redis_client.expire(key, cooldown)
        if failures >= config.CIRCUIT_BREAKER_THRESHOLD:
            await redis_client.set(f"node:{node}:cb_open", "1")
            await redis_client.expire(f"node:{node}:cb_open", cooldown)
    except Exception:
        pass
    # Per-(node, model) circuit breaker
    if model and getattr(config, "CB_SCOPE", "model") == "model":
        try:
            mkey = f"node:{node}:model:{model}:failures"
            mfailures = await redis_client.incrby(mkey, 1)
            await redis_client.expire(mkey, cooldown)
            if mfailures >= config.CIRCUIT_BREAKER_THRESHOLD:
                await redis_client.set(f"node:{node}:model:{model}:cb_open", "1")
                await redis_client.expire(f"node:{node}:model:{model}:cb_open", cooldown)
        except Exception:
            pass
    # Inject penalty RTT into EWMA so P2C scoring naturally degrades this node
    if model:
        await _record_failure_penalty(model, node)

async def _record_success(node: str, model: str = ""):
    try:
        await redis_client.set(f"node:{node}:failures", 0)
        await redis_client.expire(f"node:{node}:failures", _cb_cooldown_secs())
    except Exception:
        pass
    if model and getattr(config, "CB_SCOPE", "model") == "model":
        try:
            await redis_client.set(f"node:{node}:model:{model}:failures", 0)
            await redis_client.expire(f"node:{node}:model:{model}:failures", _cb_cooldown_secs())
        except Exception:
            pass

async def _is_circuit_open(node: str, model: str = "") -> bool:
    # Check model-scoped breaker first (more specific)
    if model and getattr(config, "CB_SCOPE", "model") == "model":
        try:
            val = await redis_client.get(f"node:{node}:model:{model}:cb_open")
            if val is not None and val != "0":
                return True
        except Exception:
            pass
    # Fall back to node-wide breaker
    try:
        val = await redis_client.get(f"node:{node}:cb_open")
        return val is not None and val != "0"
    except Exception:
        return False


# ---------------------- Rate Limit Backoff Tracking ----------------------


async def _record_rate_limit(node: str, retry_after_secs: Optional[float] = None):
    """Record a rate limit (429) response from a backend.

    Marks the backend as temporarily unhealthy with exponential backoff.
    Uses Retry-After header value if available, otherwise exponential backoff.
    """
    try:
        # Increment rate limit counter for this node
        count_key = f"node:{node}:rate_limit_count"
        count = await redis_client.incrby(count_key, 1)
        await redis_client.expire(count_key, 300)  # Reset after 5 minutes

        # Calculate backoff duration
        if retry_after_secs and retry_after_secs > 0:
            # Use Retry-After from response
            backoff_secs = min(retry_after_secs, config.RATE_LIMIT_BACKOFF_MAX_SECS)
        else:
            # Exponential backoff: base * 2^(count-1)
            base = config.RATE_LIMIT_BACKOFF_BASE_SECS
            max_backoff = config.RATE_LIMIT_BACKOFF_MAX_SECS
            backoff_secs = min(base * (2 ** (count - 1)), max_backoff)

        # Add jitter (±jitter%)
        jitter_range = backoff_secs * config.RATE_LIMIT_BACKOFF_JITTER
        jitter = random.uniform(-jitter_range, jitter_range)
        backoff_secs = max(0.1, backoff_secs + jitter)

        # Mark backend as rate-limited (temporary unhealthy state)
        limit_key = f"node:{node}:rate_limited"
        await redis_client.set(limit_key, "1")
        await redis_client.expire(limit_key, int(backoff_secs) + 1)

        # Track metrics
        await redis_client.incrby("lb:rate_limits_total", 1)
        await redis_client.incrby(f"lb:rate_limits:{node}", 1)

        logger.warning(
            "[rate_limit] Backend %s rate limited, backing off for %.1f seconds (count: %d)",
            node, backoff_secs, count
        )
    except Exception as e:
        logger.warning("[rate_limit] Failed to record rate limit for %s: %s", node, e)


async def _is_rate_limited(node: str) -> bool:
    """Check if a backend is currently rate-limited and in backoff."""
    try:
        val = await redis_client.get(f"node:{node}:rate_limited")
        return val is not None and val != "0"
    except Exception:
        return False


async def _clear_rate_limit(node: str):
    """Clear rate limit state for a backend after successful request."""
    try:
        # Clear the rate-limited flag
        await redis_client.delete(f"node:{node}:rate_limited")
        # Reset the count (on success, we reset exponential backoff)
        await redis_client.delete(f"node:{node}:rate_limit_count")
    except Exception:
        pass


async def _inc_requests_total():
    try:
        await redis_client.incrby("lb:requests_total", 1)
    except Exception:
        pass

async def get_eligible_nodes_for_model(model_name: str):
    return await get_eligible_nodes(model_name)

async def _find_fallback_model(current: str) -> Optional[str]:
    """Pick the next candidate model from the same class list that meets min_nodes.
    Returns None if no suitable fallback is found or if fallback disabled.
    """
    if not getattr(config, "CROSS_MODEL_FALLBACK", False):
        return None
    classes = getattr(config, "MODEL_CLASSES", {}) or {}
    for cname, cfg in classes.items():
        cands = cfg.get("candidates", [])
        if current in cands:
            min_nodes = int(cfg.get("min_nodes") or getattr(config, "AUTO_MIN_NODES", 2))
            for mid in cands:
                if mid == current:
                    continue
                nodes = await get_eligible_nodes(mid, include_cloud=False)
                if len(nodes) >= min_nodes:
                    return mid
    return None

async def _warm_model_on_nodes(model: str, nodes: list[str]) -> None:
    """Best-effort tiny POST to prompt backend to load the model on selected nodes."""
    if not nodes:
        return
    payload = {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 4, "stream": False}
    timeout = httpx.Timeout(float(getattr(config, "ON_DEMAND_WARM_TIMEOUT_SECS", 30)))
    async def one(node: str):
        url = f"http://{node}/v1/chat/completions"
        try:
            r = await http_client.post(url, json=payload, timeout=timeout)
            # Mark support key to speed up eligibility on success
            if r.status_code == 200:
                try:
                    await redis_client.set(f"node:{node}:supports:{model}", 1)
                    await redis_client.expire(f"node:{node}:supports:{model}", 300)
                except Exception:
                    pass
        except Exception:
            return
    # Launch concurrently
    await asyncio.gather(*[one(n) for n in nodes])

async def _diagnose_unroutable(model_name: str) -> dict:
    """Diagnose why a model has no eligible nodes. Returns structured diagnosis."""
    healthy_nodes = sorted(await redis_client.smembers("nodes:healthy"))
    if not healthy_nodes:
        return {"reason": "no_healthy_nodes", "nodes_with_model": 0, "cb_open_nodes": [], "rate_limited_nodes": [], "retry_after_secs": 0}

    nodes_with_model = []
    cb_open = []
    rate_limited = []
    max_ttl = 0

    for node in healthy_nodes:
        models_json = await redis_client.get(f"node:{node}:models")
        if not models_json:
            continue
        try:
            models = json.loads(models_json).get("data", [])
        except Exception:
            continue
        supports_key = f"node:{node}:supports:{model_name}"
        supports = await redis_client.get(supports_key)
        has_model = any(m.get("id") == model_name for m in models) or (supports is not None and supports != "0")
        if not has_model:
            continue
        nodes_with_model.append(node)

        if await _is_circuit_open(node, model_name):
            cb_open.append(node)
            # Estimate TTL remaining on CB key
            try:
                ttl = await redis_client.ttl(f"node:{node}:model:{model_name}:cb_open")
                if ttl and ttl > 0:
                    max_ttl = max(max_ttl, ttl)
                else:
                    ttl = await redis_client.ttl(f"node:{node}:cb_open")
                    if ttl and ttl > 0:
                        max_ttl = max(max_ttl, ttl)
            except Exception:
                pass
        elif await _is_rate_limited(node):
            rate_limited.append(node)

    if not nodes_with_model:
        return {"reason": "model_unknown", "nodes_with_model": 0, "cb_open_nodes": [], "rate_limited_nodes": [], "retry_after_secs": 0}

    if cb_open and len(cb_open) == len(nodes_with_model):
        return {"reason": "circuit_open", "nodes_with_model": len(nodes_with_model), "cb_open_nodes": cb_open, "rate_limited_nodes": [], "retry_after_secs": max_ttl or _cb_cooldown_secs()}

    if rate_limited and len(rate_limited) + len(cb_open) == len(nodes_with_model):
        return {"reason": "rate_limited", "nodes_with_model": len(nodes_with_model), "cb_open_nodes": cb_open, "rate_limited_nodes": rate_limited, "retry_after_secs": getattr(config, "RETRY_AFTER_SECS", 2)}

    return {"reason": "unknown", "nodes_with_model": len(nodes_with_model), "cb_open_nodes": cb_open, "rate_limited_nodes": rate_limited, "retry_after_secs": 0}


async def _on_demand_wait_for_model(model: str) -> list[str]:
    """If no nodes are eligible, probe a few healthy nodes to load the model and poll until any become eligible or grace window elapses."""
    if not getattr(config, "ON_DEMAND_WAIT_ENABLED", True):
        return []
    try:
        healthy = list(await redis_client.smembers("nodes:healthy"))
    except Exception:
        healthy = []
    if not healthy:
        return []
    # Choose up to N nodes to probe
    fanout = max(1, int(getattr(config, "ON_DEMAND_WARM_FANOUT", 2)))
    targets = healthy if len(healthy) <= fanout else random.sample(healthy, fanout)
    await _warm_model_on_nodes(model, targets)

    # Poll for eligibility up to grace window
    grace = float(getattr(config, "ON_DEMAND_WARM_GRACE_SECS", 30))
    poll_ms = max(100, int(getattr(config, "ON_DEMAND_WARM_POLL_MS", 750)))
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        nodes = await get_eligible_nodes(model)
        if nodes:
            return nodes
        await asyncio.sleep(poll_ms / 1000.0)
    return []

async def _get_sticky_node(session_id: Optional[str], model_name: str) -> Optional[str]:
    # Stickiness requires feature flag enabled and a provided session id
    if not getattr(config, "STICKY_SESSIONS_ENABLED", False) or not session_id:
        return None
    try:
        return await redis_client.get(f"session:{session_id}:{model_name}")
    except Exception:
        return None

async def _set_sticky_node(session_id: Optional[str], model_name: str, node: str):
    # Stickiness requires feature flag enabled and a provided session id
    if not getattr(config, "STICKY_SESSIONS_ENABLED", False) or not session_id:
        return
    try:
        key = f"session:{session_id}:{model_name}"
        await redis_client.set(key, node)
        await redis_client.expire(key, config.STICKY_TTL_SECS)
    except Exception:
        pass


# ---------------------- Multi-Backend Execution Helpers ----------------------


@dataclass
class ExtendedExecutionConfig:
    """Extended execution config with oracle, fallback chain, and capability support."""
    base: ExecutionConfig
    # Oracle configuration for consensus mode
    oracle_backend: Optional[str] = None  # Validated oracle backend (e.g., "cloud:openai")
    oracle_valid: bool = True  # Whether the oracle header was valid
    # Capability filtering — only route to backends with these capability tags
    required_capabilities: Optional[frozenset] = None  # e.g. frozenset({"research", "code"})
    # Fallback chain configuration
    fallback_chain: Optional[str] = None  # Name of fallback chain to use


def _parse_execution_mode(request: Request) -> Optional[ExtendedExecutionConfig]:
    """Parse execution mode from request headers or query params.

    Headers (preferred):
        X-Execution-Mode: race | all | sequence | consensus
        X-Target-Backends: alias1,alias2,... (optional)
        X-Max-Backends: N (optional, default 3)
        X-Consensus-Oracle: openai | cloud:openai (optional, for consensus mode)
        X-Fallback-Chain: chain_name (optional, takes precedence over X-Execution-Mode)

    Query params (fallback):
        ?mode=race|all|sequence|consensus
        &backends=alias1,alias2,...
        &max_backends=N

    Precedence rules:
    - X-Fallback-Chain takes precedence over X-Execution-Mode
    - If both are set, X-Execution-Mode is ignored (logged as warning)
    """
    if not getattr(config, "MULTI_EXEC_ENABLED", True):
        return None

    # Check for fallback chain first (takes precedence)
    fallback_chain = request.headers.get("x-fallback-chain")
    if fallback_chain:
        fallback_chain = fallback_chain.strip()
        chains = getattr(config, "FALLBACK_CHAINS", {}) or {}
        if fallback_chain not in chains:
            # Invalid chain name - return None to use standard routing
            logger.warning(f"Unknown fallback chain: {fallback_chain}")
            return None
        # Log if execution mode was also set
        if request.headers.get("x-execution-mode"):
            logger.warning(
                f"Both X-Fallback-Chain ({fallback_chain}) and X-Execution-Mode set. "
                "Fallback chain takes precedence."
            )
        # Return a config that signals fallback chain execution
        return ExtendedExecutionConfig(
            base=ExecutionConfig(
                mode=ExecutionMode.SEQUENCE,  # Fallback is sequential failover
                target_backends=None,
                max_backends=1,
                timeout_secs=float(getattr(config, "FALLBACK_TOTAL_TIMEOUT_SECS", 120))
            ),
            fallback_chain=fallback_chain
        )

    # Check header first, then query param
    mode_str = request.headers.get("x-execution-mode") or request.query_params.get("mode")
    if not mode_str:
        return None

    mode_str = mode_str.lower().strip()
    try:
        mode = ExecutionMode(mode_str)
    except ValueError:
        return None

    # Parse target backends
    targets_str = request.headers.get("x-target-backends") or request.query_params.get("backends")
    target_backends = None
    if targets_str:
        target_backends = [t.strip() for t in targets_str.split(",") if t.strip()]

    # Parse max backends
    max_str = request.headers.get("x-max-backends") or request.query_params.get("max_backends")
    max_backends = int(getattr(config, "MULTI_EXEC_MAX_BACKENDS", 3))
    if max_str:
        try:
            max_backends = int(max_str)
        except ValueError:
            pass

    # Parse timeout
    timeout_str = request.headers.get("x-execution-timeout")
    timeout_secs = float(getattr(config, "MULTI_EXEC_TIMEOUT_SECS", 60.0))
    if timeout_str:
        try:
            timeout_secs = float(timeout_str)
        except ValueError:
            pass

    # Parse and validate oracle header (for consensus mode)
    oracle_raw = request.headers.get("x-consensus-oracle")
    oracle = None
    oracle_valid = True

    if oracle_raw and mode == ExecutionMode.CONSENSUS:
        # Normalize: "openai" -> "cloud:openai", "cloud:openai" -> "cloud:openai"
        oracle_name = oracle_raw.strip().replace("cloud:", "")
        cloud_backends = getattr(config, "CLOUD_BACKENDS", {}) or {}

        if oracle_name in cloud_backends:
            oracle = f"cloud:{oracle_name}"
        else:
            oracle_valid = False  # Unknown backend key
            logger.warning(f"Invalid oracle backend: {oracle_raw}")

    # Parse capability requirements (x-require-capability: research,code)
    # Inspired by Perplexity Computer capability-specialized routing
    # Case-sensitive — no lowercasing so that tags like "GPU" vs "gpu" are distinct
    cap_raw = request.headers.get("x-require-capability")
    required_capabilities = None
    if cap_raw:
        required_capabilities = frozenset(
            c.strip() for c in cap_raw.split(",") if c.strip()
        )

    return ExtendedExecutionConfig(
        base=ExecutionConfig(
            mode=mode,
            target_backends=target_backends,
            max_backends=max_backends,
            timeout_secs=timeout_secs
        ),
        oracle_backend=oracle,
        oracle_valid=oracle_valid,
        required_capabilities=required_capabilities,
    )


def _resolve_backend_alias(alias_or_host: str) -> str:
    """Resolve a backend alias to host:port, or return as-is if not an alias."""
    aliases = getattr(config, "BACKEND_ALIASES", {}) or {}
    return aliases.get(alias_or_host, alias_or_host)


def _get_backend_alias(host_port: str) -> Optional[str]:
    """Get human-friendly alias for a host:port, or None if not aliased."""
    reverse = getattr(config, "BACKEND_ALIASES_REVERSE", {}) or {}
    return reverse.get(host_port)


def _is_cloud_backend(node: str) -> bool:
    """Check if a node is a cloud backend (node address starts with 'cloud:')."""
    return node.startswith("cloud:")


def _backend_has_capabilities(node: str, required: frozenset) -> bool:
    """Return True if backend node has all required capability tags.

    Cloud backends are looked up by their name (strip "cloud:" prefix).
    Local backends are looked up by host:port.

    Inspired by Perplexity Computer model specialization — capability-typed pools.
    """
    if not required:
        return True
    caps_map = getattr(config, "BACKEND_CAPABILITIES", {}) or {}
    key = node.replace("cloud:", "") if node.startswith("cloud:") else node
    backend_caps = caps_map.get(key, frozenset())
    return required.issubset(backend_caps)


def _resolve_planner_backend(oracle_backend: Optional[str] = None) -> Optional[str]:
    """Resolve which backend to use as the PLAN mode orchestrator.

    Priority:
      1. oracle_backend from request (x-consensus-oracle header reused for PLAN)
      2. PLANNER_BACKEND config value
      3. First configured cloud backend (last resort)
    """
    if oracle_backend:
        return oracle_backend
    planner = getattr(config, "PLANNER_BACKEND", "") or ""
    if planner:
        # Normalise: "claude" → "cloud:claude", "cloud:claude" → "cloud:claude"
        if not planner.startswith("cloud:") and planner in (getattr(config, "CLOUD_BACKENDS", {}) or {}):
            return f"cloud:{planner}"
        return planner
    # Fall back to first cloud backend
    cloud_backends = getattr(config, "CLOUD_BACKENDS", {}) or {}
    if cloud_backends:
        return f"cloud:{next(iter(cloud_backends))}"
    return None


def _filter_nodes_by_capability(nodes: List[str], required: Optional[frozenset]) -> List[str]:
    """Filter node list to only those matching required capabilities.

    Falls back to the full list if no nodes match (capability tags are advisory).
    """
    if not required:
        return nodes
    filtered = [n for n in nodes if _backend_has_capabilities(n, required)]
    if not filtered:
        logger.warning(
            "No backends match required capabilities %s — using all eligible nodes",
            sorted(required),
        )
        return nodes
    return filtered


def _get_cloud_backend_name(node: str) -> Optional[str]:
    """Extract cloud backend name from node address (e.g., 'cloud:openai' -> 'openai')."""
    if not _is_cloud_backend(node):
        return None
    return node.split(":", 1)[1] if ":" in node else None


def _get_cloud_config(node: str) -> Optional[Dict[str, Any]]:
    """Get cloud backend configuration for a node."""
    name = _get_cloud_backend_name(node)
    if not name:
        return None
    cloud_backends = getattr(config, "CLOUD_BACKENDS", {}) or {}
    return cloud_backends.get(name)


def _get_cloud_url(node: str, endpoint: str = "/chat/completions") -> Optional[str]:
    """Get the full URL for a cloud backend endpoint.

    Uses provider adapters to determine the correct endpoint path.

    Args:
        node: Cloud node address (e.g., 'cloud:openai')
        endpoint: API endpoint path (e.g., '/chat/completions')

    Returns:
        Full URL (e.g., 'https://api.openai.com/v1/chat/completions') or None
    """
    cfg = _get_cloud_config(node)
    if not cfg:
        return None
    base_url = cfg.get("url", "").rstrip("/")
    if not base_url:
        return None

    # Use provider adapter for endpoint path
    provider_type = cfg.get("provider_type", "openai")
    adapter = get_adapter(provider_type)

    # Map generic endpoint to provider-specific endpoint
    if endpoint == "/chat/completions":
        endpoint = adapter.get_chat_endpoint()
    elif endpoint == "/embeddings":
        endpoint = adapter.get_embeddings_endpoint()

    return f"{base_url}{endpoint}"


def _get_cloud_headers(node: str, headers: Dict[str, str]) -> Dict[str, str]:
    """Inject cloud API auth into headers for a cloud backend request.

    Uses provider adapters for provider-specific auth (Bearer vs x-api-key).

    Returns updated headers dict with authentication set.
    """
    cfg = _get_cloud_config(node)
    if not cfg:
        return headers

    api_key = cfg.get("api_key", "")
    if not api_key:
        return headers

    # Use provider adapter for headers
    provider_type = cfg.get("provider_type", "openai")
    adapter = get_adapter(provider_type)

    return adapter.get_headers(api_key, headers)


def _get_equivalent_models(model_name: str) -> List[str]:
    """Get all equivalent model names for a given model.

    If model_name is a canonical name in MODEL_EQUIVALENTS, returns the list.
    If model_name is in MODEL_EQUIVALENTS_REVERSE, returns all equivalents for that group.
    Otherwise returns [model_name].
    """
    equivalents = getattr(config, "MODEL_EQUIVALENTS", {}) or {}
    reverse = getattr(config, "MODEL_EQUIVALENTS_REVERSE", {}) or {}

    # Check if it's a canonical name
    if model_name in equivalents:
        return equivalents[model_name]

    # Check if it's in an equivalence group
    if model_name in reverse:
        canonical = reverse[model_name]
        return equivalents.get(canonical, [model_name])

    return [model_name]


async def _get_backend_model(backend: str, requested_model: str) -> Optional[str]:
    """Find which model name a backend actually has that matches the requested model.

    Returns the actual model ID to use for this backend, or None if no match.
    """
    equivalent_models = _get_equivalent_models(requested_model)
    # Cloud backends: choose an appropriate cloud model for the requested model.
    if _is_cloud_backend(backend):
        cloud_name = _get_cloud_backend_name(backend)
        cloud_models = getattr(config, "CLOUD_MODELS", {}) or {}
        allowed = cloud_models.get(cloud_name, []) if cloud_name else None

        # If there is no explicit CLOUD_MODELS entry, fall back to requested model.
        if not allowed:
            return requested_model

        # Exact/equivalent match first
        if requested_model in allowed:
            return requested_model
        for equiv in equivalent_models:
            if equiv in allowed:
                return equiv

        # Cross-model: pick by class
        if bool(getattr(config, "CROSS_MODEL_FALLBACK", 0)):
            cls = _get_model_class(requested_model)
            picked = _pick_cloud_model_for_class(cls, cloud_name)
            return picked

        return None


    try:
        models_json = await redis_client.get(f"node:{backend}:models")
        if not models_json:
            return None
        models = json.loads(models_json).get("data", [])
        backend_model_ids = {m.get("id") for m in models if m.get("id")}

        # Check for exact match first
        if requested_model in backend_model_ids:
            return requested_model

        # Check for equivalent matches
        for equiv in equivalent_models:
            if equiv in backend_model_ids:
                return equiv

        return None
    except Exception:
        return None


async def _get_eligible_nodes_for_equivalents(model_name: str) -> List[str]:
    """Get eligible nodes that have the requested model OR any equivalent model."""
    equivalent_models = _get_equivalent_models(model_name)

    # If no equivalents defined, use standard eligibility
    if len(equivalent_models) <= 1 and model_name not in getattr(config, "MODEL_EQUIVALENTS", {}):
        return await get_eligible_nodes(model_name)

    # Gather eligible nodes for all equivalent models
    all_eligible = set()
    for equiv_model in equivalent_models:
        nodes = await get_eligible_nodes(equiv_model)
        all_eligible.update(nodes)

    # Also check the canonical name if different
    if model_name not in equivalent_models:
        nodes = await get_eligible_nodes(model_name)
        all_eligible.update(nodes)

    return list(all_eligible)


async def _backend_supports_model(backend: str, model_name: str) -> bool:
    """Check if backend supports the requested model.

    For cloud backends, checks CLOUD_MODELS configuration.
    For local backends, checks Redis model list.
    """
    if _is_cloud_backend(backend):
        # Check cloud models config
        cloud_name = _get_cloud_backend_name(backend)
        cloud_models = getattr(config, "CLOUD_MODELS", {}) or {}
        if cloud_name and cloud_name in cloud_models:
            model_list = cloud_models[cloud_name]
            # Check exact match or equivalents
            equivalent_models = _get_equivalent_models(model_name)
            result = model_name in model_list or any(m in model_list for m in equivalent_models)

            # Cross-model fallback: if enabled, allow cloud backend to participate in consensus
            # by selecting an appropriate cloud model from the same model class.
            if not result and bool(getattr(config, "CROSS_MODEL_FALLBACK", 0)):
                cls = _get_model_class(model_name)
                picked = _pick_cloud_model_for_class(cls, cloud_name)
                if picked:
                    result = True

            logger.info(
                "cloud_support_check: backend=%s model=%s cloud_name=%s models=%s equivalents=%s result=%s",
                backend,
                model_name,
                cloud_name,
                model_list,
                equivalent_models,
                result,
            )
            return result
        # If no config, assume cloud backends support requested model
        return True

    # For local backends, check Redis
    try:
        models_json = await redis_client.get(f"node:{backend}:models")
        if not models_json:
            return False
        models = json.loads(models_json).get("data", [])
        backend_model_ids = {m.get("id") for m in models if m.get("id")}
        # Check exact match or equivalents
        equivalent_models = _get_equivalent_models(model_name)
        return model_name in backend_model_ids or any(m in backend_model_ids for m in equivalent_models)
    except Exception:
        return False


async def _get_eligible_cloud_backends(model_name: str) -> List[str]:
    """Get list of cloud backends that support the requested model and are not rate-limited."""
    cloud_backends = getattr(config, "CLOUD_BACKENDS", {}) or {}
    eligible = []
    for name in cloud_backends.keys():
        cloud_node = f"cloud:{name}"
        # Check rate limit status
        if await _is_rate_limited(cloud_node):
            continue
        # Check model support
        if await _backend_supports_model(cloud_node, model_name):
            eligible.append(cloud_node)
    logger.info("eligible_cloud_backends: model=%s eligible=%s", model_name, eligible)
    return eligible


async def _select_backends_for_consensus(
    exec_config: ExecutionConfig,
    model_name: str,
    eligible_local_nodes: List[str],
    oracle_backend: Optional[str] = None
) -> Tuple[List[str], bool]:
    """Select backends ensuring mix of local + cloud for consensus.

    Mix Policy:
    - Minimum: 1 local + 1 cloud (if both available)
    - Target: 2 local + 1 cloud for 3-way consensus
    - Fallback: if only one side exists, use what's available

    Args:
        exec_config: Execution configuration
        model_name: The requested model name
        eligible_local_nodes: List of eligible local nodes
        oracle_backend: Optional oracle backend to force-include

    Returns:
        (selected_backends, oracle_present)
    """
    # Get eligible cloud backends
    cloud_capable = await _get_eligible_cloud_backends(model_name)
    local_capable = [n for n in eligible_local_nodes if not _is_cloud_backend(n)]
    logger.info(
        "consensus_select: model=%s local_capable=%s cloud_capable=%s cross_model=%s",
        model_name,
        local_capable,
        cloud_capable,
        getattr(config, "CROSS_MODEL_FALLBACK", 0),
    )

    # Force-include oracle if specified and capable
    oracle_present = False
    selected = []

    if oracle_backend and oracle_backend in cloud_capable:
        selected.append(oracle_backend)
        cloud_capable.remove(oracle_backend)
        oracle_present = True
    elif oracle_backend and oracle_backend not in cloud_capable:
        # Oracle was requested but not available
        logger.warning(f"Oracle backend {oracle_backend} not available for model {model_name}")

    # Apply mix policy
    max_b = exec_config.max_backends
    remaining = max_b - len(selected)

    if local_capable and cloud_capable:
        # Both sides available: aim for 2 local + 1 cloud (or 1+1 minimum)
        local_target = min(2, remaining - 1) if remaining > 1 else 1
        cloud_target = 1 if not oracle_present else 0
    elif local_capable:
        # Only local available
        local_target = remaining
        cloud_target = 0
    else:
        # Only cloud available
        local_target = 0
        cloud_target = remaining

    # Select local backends
    local_pool = list(local_capable)
    for _ in range(min(local_target, len(local_pool))):
        if not local_pool:
            break
        # Use router to pick best candidate from remaining pool
        candidate = await router.select_node(local_pool, model_name, redis_client)
        if candidate:
            selected.append(candidate)
            local_pool.remove(candidate)

    # Select additional cloud backends
    for _ in range(min(cloud_target, len(cloud_capable))):
        if not cloud_capable:
            break
        candidate = cloud_capable.pop(0)  # Take first available
        selected.append(candidate)

    return selected, oracle_present


async def _select_backends_for_execution(
    exec_config: ExecutionConfig,
    model_name: str,
    eligible_nodes: List[str]
) -> List[str]:
    """Select backends for multi-exec based on config.

    If target_backends specified, resolve aliases and filter to eligible.
    Otherwise, select up to max_backends from eligible pool.
    """
    if exec_config.target_backends:
        # Resolve aliases and filter to eligible
        resolved = [_resolve_backend_alias(t) for t in exec_config.target_backends]
        # Keep order but filter to eligible
        return [b for b in resolved if b in eligible_nodes][:exec_config.max_backends]

    # Auto-select from eligible pool
    # Use router to select diverse backends
    selected = []
    pool = list(eligible_nodes)

    for _ in range(min(exec_config.max_backends, len(pool))):
        if not pool:
            break
        # Use router to pick best candidate from remaining pool
        candidate = await router.select_node(pool, model_name, redis_client)
        if candidate:
            selected.append(candidate)
            pool.remove(candidate)

    return selected


async def _make_backend_request(
    backend: str,
    body: Dict[str, Any],
    headers: Dict[str, str],
    model_name: str,
    request_id: str,
    start_time: float,
    backend_model_override: Optional[str] = None
) -> BackendResult:
    """Make a single non-streaming request to a backend and return result.

    For cloud backends, uses provider adapters for request/response transformation.
    """
    # Determine URL and adapter based on backend type (cloud vs local)
    adapter = None
    if _is_cloud_backend(backend):
        url = _get_cloud_url(backend, "/chat/completions")
        if not url:
            return BackendResult(
                backend=backend,
                alias=_get_cloud_backend_name(backend),
                error="Cloud backend URL not configured"
            )
        # Inject API key for cloud backends
        headers = _get_cloud_headers(backend, headers)
        alias = _get_cloud_backend_name(backend)
        # Get provider adapter for request/response transformation
        cfg = _get_cloud_config(backend)
        if cfg:
            provider_type = cfg.get("provider_type", "openai")
            adapter = get_adapter(provider_type)
    else:
        url = f"http://{backend}/v1/chat/completions"
        alias = _get_backend_alias(backend)

    result = BackendResult(backend=backend, alias=alias)
    t0 = time.monotonic()

    # Use override model if provided (for model equivalence support)
    actual_model = backend_model_override or model_name
    request_body = dict(body)
    if backend_model_override:
        request_body["model"] = backend_model_override

    # Transform request for provider-specific format
    if adapter:
        request_body = adapter.transform_request(request_body)

    acquired = await _acquire_slot(backend)
    if not acquired:
        result.error = "Backend at capacity"
        result.latency_ms = (time.monotonic() - t0) * 1000
        return result

    model_ok = await _acquire_model_slot(actual_model)
    if not model_ok:
        await _dec_inflight(backend)
        result.error = "Model at capacity"
        result.latency_ms = (time.monotonic() - t0) * 1000
        return result

    try:
        try:
            resp = await http_client.post(url, json=request_body, headers=headers)
            result.status_code = resp.status_code
            result.first_byte_ms = (time.monotonic() - t0) * 1000

            # Handle rate limiting (429)
            if resp.status_code == 429:
                # Extract Retry-After header if present
                retry_after = resp.headers.get("retry-after")
                retry_after_secs = None
                if retry_after:
                    try:
                        retry_after_secs = float(retry_after)
                    except ValueError:
                        pass
                await _record_rate_limit(backend, retry_after_secs)
                result.error = f"HTTP 429 Rate Limited"
            elif resp.status_code >= 500 or resp.status_code == 404:
                await _record_failure(backend, model_name)
                result.error = f"HTTP {resp.status_code}"
            else:
                await _record_success(backend, model_name)
                # Clear any previous rate limit state on success
                await _clear_rate_limit(backend)
                result.success = True
                try:
                    response_body = resp.json()
                    # Transform response to OpenAI format if using provider adapter
                    if adapter:
                        response_body = adapter.transform_response(response_body)
                    result.response_body = response_body
                except Exception:
                    result.response_body = {"raw": resp.text}
        except AttributeError:
            # Compatibility for test fakes that only implement .stream()
            async with http_client.stream("POST", url, json=request_body, headers=headers) as response:
                result.status_code = getattr(response, "status_code", 200)
                result.first_byte_ms = (time.monotonic() - t0) * 1000

                # Handle rate limiting (429)
                if result.status_code == 429:
                    retry_after = getattr(response, "headers", {}).get("retry-after")
                    retry_after_secs = None
                    if retry_after:
                        try:
                            retry_after_secs = float(retry_after)
                        except ValueError:
                            pass
                    await _record_rate_limit(backend, retry_after_secs)
                    result.error = f"HTTP 429 Rate Limited"
                elif result.status_code >= 500 or result.status_code == 404:
                    await _record_failure(backend, model_name)
                    result.error = f"HTTP {result.status_code}"
                else:
                    chunks = bytearray()
                    async for chunk in response.aiter_bytes():
                        chunks.extend(chunk)
                    await _record_success(backend, model_name)
                    await _clear_rate_limit(backend)
                    result.success = True
                    try:
                        response_body = json.loads(bytes(chunks))
                        # Transform response to OpenAI format if using provider adapter
                        if adapter:
                            response_body = adapter.transform_response(response_body)
                        result.response_body = response_body
                    except Exception:
                        result.response_body = {"raw": bytes(chunks).decode(errors="replace")}

    except httpx.RequestError as e:
        await _record_failure(backend, model_name)
        result.error = str(e)
    except Exception as e:
        await _record_failure(backend, model_name)
        result.error = str(e)
    finally:
        await _dec_inflight(backend)
        await _dec_model(model_name)
        result.latency_ms = (time.monotonic() - t0) * 1000

    return result


async def _record_multi_exec_metrics(mode: str, backends_attempted: int, backends_succeeded: int):
    """Record metrics for multi-backend execution."""
    try:
        await redis_client.incrby(f"lb:multi_exec_total:{mode}", 1)
        await redis_client.incrby("lb:multi_exec_total", 1)
        await redis_client.incrby(f"lb:multi_exec_backends_sum:{mode}", backends_attempted)
        await redis_client.incrby(f"lb:multi_exec_backends_count:{mode}", 1)
        await redis_client.incrby(f"lb:multi_exec_succeeded_sum:{mode}", backends_succeeded)
    except Exception:
        pass


async def _record_consensus_metrics(
    model_name: str,
    consensus: ConsensusResult,
    backends: List[str],
    oracle_valid: bool = True
):
    """Record detailed metrics for consensus operations.

    Tracks:
    - Agreement counts (how many backends agreed)
    - Disagreement rate (when not unanimous)
    - Comparison type distribution (hash, text, tool_calls)
    - Per-model consensus quality
    - Backend combinations that disagree
    - Local vs cloud agreement
    - Oracle tracking
    - Per-backend error tracking
    """
    try:
        # Determine model class for per-class tracking
        model_class = _get_model_class(model_name)

        # Total consensus requests
        await redis_client.incrby("lb:consensus_total", 1)
        await redis_client.incrby(f"lb:consensus_total:{model_name}", 1)

        # Agreement count histogram
        agreement_count = consensus.agreement_count
        await redis_client.incrby(f"lb:consensus_agreement:{agreement_count}", 1)
        await redis_client.incrby(f"lb:consensus_agreement:{model_name}:{agreement_count}", 1)

        # Disagreement tracking
        if consensus.disagreement:
            await redis_client.incrby("lb:consensus_disagreements", 1)
            await redis_client.incrby(f"lb:consensus_disagreements:{model_name}", 1)

            # Track which backend combination disagreed (for debugging)
            backend_key = "|".join(sorted(backends))
            await redis_client.incrby(f"lb:consensus_disagreements:backends:{backend_key}", 1)
        else:
            await redis_client.incrby("lb:consensus_agreements", 1)
            await redis_client.incrby(f"lb:consensus_agreements:{model_name}", 1)

        # Comparison type distribution
        comp_type = consensus.comparison_type
        await redis_client.incrby(f"lb:consensus_comparison:{comp_type}", 1)
        await redis_client.incrby(f"lb:consensus_comparison:{model_name}:{comp_type}", 1)

        # Add to series for easier metric export
        await redis_client.sadd("lb:consensus_models", model_name)

        # Local vs cloud agreement tracking
        local_backends = [b for b in backends if not _is_cloud_backend(b)]
        cloud_backends = [b for b in backends if _is_cloud_backend(b)]

        if local_backends and cloud_backends:
            await redis_client.incrby("lb:consensus_local_cloud_total", 1)
            if model_class:
                await redis_client.incrby(f"lb:consensus_local_cloud_total:{model_class}", 1)

            if consensus.local_cloud_agreement is True:
                await redis_client.incrby("lb:consensus_local_cloud_agreed", 1)
                if model_class:
                    await redis_client.incrby(f"lb:consensus_local_cloud_agreed:{model_class}", 1)
            elif consensus.local_cloud_agreement is False:
                await redis_client.incrby("lb:consensus_local_cloud_disagreed", 1)

            # Track similarity score histogram (buckets: 0.0, 0.1, ..., 1.0)
            if consensus.local_cloud_similarity is not None:
                bucket = int(consensus.local_cloud_similarity * 10) / 10  # Round to nearest 0.1
                await redis_client.incrby(f"lb:consensus_similarity_bucket:{bucket:.1f}", 1)

        # Oracle tracking
        if consensus.oracle_backend:
            await redis_client.incrby("lb:consensus_oracle_requested", 1)
            if consensus.oracle_present:
                await redis_client.incrby("lb:consensus_oracle_present", 1)
                if consensus.oracle_agreed is True:
                    await redis_client.incrby("lb:consensus_oracle_agreed", 1)
                elif consensus.oracle_agreed is False:
                    await redis_client.incrby("lb:consensus_oracle_disagreed", 1)
            else:
                await redis_client.incrby("lb:consensus_oracle_missing", 1)

        if not oracle_valid:
            await redis_client.incrby("lb:consensus_oracle_invalid", 1)

        # Per-backend error code tracking
        for result in consensus.all_responses:
            if result.status_code == 429:
                await redis_client.incrby(f"lb:backend_errors:429:{result.backend}", 1)
            elif result.status_code >= 500:
                await redis_client.incrby(f"lb:backend_errors:5xx:{result.backend}", 1)

    except Exception:
        pass


def _get_model_class(model_name: str) -> Optional[str]:
    """Get the model class (e.g., 'small', 'medium', 'large') for a model."""
    classes = getattr(config, "MODEL_CLASSES", {}) or {}
    for class_name, class_cfg in classes.items():
        candidates = class_cfg.get("candidates", [])
        if model_name in candidates:
            return class_name
    return None



def _pick_cloud_model_for_class(model_class: Optional[str], cloud_backend_name: str) -> Optional[str]:
    """Pick a cloud model id for a given model class and cloud backend.

    Returns the first candidate in MODEL_CLASSES[model_class].candidates that is listed
    for this cloud backend. Returns None if no model_class is provided or no matching
    candidate exists.

    Note: We intentionally do NOT fall back to allowed[0] as that would cause
    every model to be considered supported by every cloud backend.
    """
    if not model_class:
        return None

    cloud_models = getattr(config, "CLOUD_MODELS", {}) or {}
    allowed = cloud_models.get(cloud_backend_name, []) or []
    if not allowed:
        return None

    classes_cfg = getattr(config, "MODEL_CLASSES", {}) or {}
    candidates = ((classes_cfg.get(model_class) or {}).get("candidates", []) or [])
    for m in candidates:
        if m in allowed:
            return m

    return None


async def _handle_multi_backend_execution(
    request: Request,
    body: Dict[str, Any],
    ext_config: ExtendedExecutionConfig,
    model_name: str,
    eligible_nodes: List[str],
    request_id: str,
    start_time: float,
    is_stream: bool,
    on_demand_wait: bool,
    warm_wait_ms: int,
    model_defaulted: bool
) -> Response:
    """Handle multi-backend execution modes (race, all, sequence, consensus).

    Supports extended execution config with oracle and fallback chain features.
    """
    exec_config = ext_config.base
    oracle_backend = ext_config.oracle_backend
    oracle_valid = ext_config.oracle_valid

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in ('host', 'content-length', 'x-execution-mode', 'x-target-backends',
                               'x-max-backends', 'x-consensus-oracle', 'x-fallback-chain',
                               'x-require-capability')
    }
    headers["x-request-id"] = request_id

    # Expand eligible nodes to include those with equivalent models
    expanded_eligible = await _get_eligible_nodes_for_equivalents(model_name)
    if expanded_eligible:
        # Merge with original eligible_nodes (union)
        all_eligible = list(set(eligible_nodes) | set(expanded_eligible))
    else:
        all_eligible = eligible_nodes

    # Apply capability filtering if requested (Perplexity Computer-style specialization)
    all_eligible = _filter_nodes_by_capability(all_eligible, ext_config.required_capabilities)

    # Select backends - use consensus-specific selection for consensus mode
    oracle_present = False
    if exec_config.mode == ExecutionMode.CONSENSUS:
        backends, oracle_present = await _select_backends_for_consensus(
            exec_config, model_name, all_eligible, oracle_backend
        )
    else:
        backends = await _select_backends_for_execution(exec_config, model_name, all_eligible)

    if not backends:
        raise HTTPException(status_code=404, detail="No eligible backends for multi-exec")

    # Build mapping of backend -> actual model name to use
    backend_models: Dict[str, str] = {}
    for backend in backends:
        actual_model = await _get_backend_model(backend, model_name)
        backend_models[backend] = actual_model or model_name

    # Create request factory with per-backend model override
    async def make_request(backend: str) -> BackendResult:
        backend_model = backend_models.get(backend, model_name)
        return await _make_backend_request(
            backend, body, headers, model_name, request_id, start_time,
            backend_model_override=backend_model if backend_model != model_name else None
        )

    engine = ExecutionEngine(
        similarity_threshold=float(getattr(config, "MULTI_EXEC_CONSENSUS_THRESHOLD", 0.9))
    )

    await _inc_requests_total()

    # Non-streaming multi-exec
    if exec_config.mode == ExecutionMode.RACE:
        result = await engine.execute_race(backends, make_request, exec_config.timeout_secs)
        await _record_multi_exec_metrics("race", len(backends), 1 if result.success else 0)

        if not result.success:
            raise HTTPException(status_code=502, detail=result.error or "All backends failed")

        resp_body = result.response_body or {}
        out = JSONResponse(content=resp_body)

    elif exec_config.mode == ExecutionMode.ALL:
        results = await engine.execute_all(backends, make_request, exec_config.timeout_secs)
        succeeded = sum(1 for r in results if r.success)
        await _record_multi_exec_metrics("all", len(backends), succeeded)

        # Build response with all results
        resp_body = {
            "mode": "all",
            "backends_attempted": len(backends),
            "backends_succeeded": succeeded,
            "responses": [
                {
                    "backend": r.backend,
                    "alias": r.alias,
                    "success": r.success,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                    "response": r.response_body
                }
                for r in results
            ]
        }
        out = JSONResponse(content=resp_body)

    elif exec_config.mode == ExecutionMode.SEQUENCE:
        results = await engine.execute_sequence(backends, make_request, exec_config.timeout_secs, stop_on_success=False)
        succeeded = sum(1 for r in results if r.success)
        await _record_multi_exec_metrics("sequence", len(results), succeeded)

        resp_body = {
            "mode": "sequence",
            "backends_attempted": len(results),
            "backends_succeeded": succeeded,
            "responses": [
                {
                    "backend": r.backend,
                    "alias": r.alias,
                    "success": r.success,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                    "response": r.response_body
                }
                for r in results
            ]
        }
        out = JSONResponse(content=resp_body)

    elif exec_config.mode == ExecutionMode.PLAN:
        # Perplexity Computer-style multi-step task decomposition + specialization
        planner_node = _resolve_planner_backend(oracle_backend)
        if not planner_node:
            # Fall back to first eligible node when no explicit planner is configured
            if all_eligible:
                planner_node = all_eligible[0]
            else:
                raise HTTPException(
                    status_code=503,
                    detail="PLAN mode requires PLANNER_BACKEND to be configured (or x-consensus-oracle set)"
                )

        # Build capability → eligible nodes mapping from all eligible nodes
        caps_map = getattr(config, "BACKEND_CAPABILITIES", {}) or {}
        capability_nodes: Dict[str, List[str]] = {}
        for node in all_eligible:
            key = node.replace("cloud:", "") if node.startswith("cloud:") else node
            for cap in caps_map.get(key, []):
                capability_nodes.setdefault(cap, []).append(node)

        # SSE streaming path: Accept: text/event-stream → execute_plan_stream
        accept_header = request.headers.get("accept", "")
        if "text/event-stream" in accept_header:
            async def _sse_call_backend(node: str, messages: list) -> str:
                plan_body = {**body, "messages": messages, "stream": False}
                resp = await http_client.post(
                    f"http://{node}/v1/chat/completions",
                    json=plan_body,
                    headers={k: v for k, v in headers.items() if k.lower() not in ("host", "content-length", "accept")},
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

            async def _sse_stream_backend(node: str, messages: list):
                async for chunk in stream_backend(node, messages, model_name, request_id):
                    yield chunk

            async def _plan_sse_generator():
                async for event in execute_plan_stream(
                    messages=body.get("messages", []),
                    call_backend=_sse_call_backend,
                    stream_backend=_sse_stream_backend,
                    planner_backend=planner_node,
                    capability_nodes=capability_nodes,
                    default_nodes=list(all_eligible),
                    max_subtasks=int(getattr(config, "PLAN_MAX_SUBTASKS", 5)),
                    subtask_timeout=float(getattr(config, "PLAN_SUBTASK_TIMEOUT_SECS", 90.0)),
                ):
                    if event.event_type == "token":
                        # chunk is already SSE-formatted bytes from stream_backend (e.g. b"data: {...}\n\n")
                        chunk = event.data["chunk"]
                        yield chunk if isinstance(chunk, bytes) else chunk.encode()
                    elif event.event_type == "error":
                        yield f"event: error\ndata: {json.dumps(event.data)}\n\n".encode()
                    else:
                        yield f"event: {event.event_type}\ndata: {json.dumps(event.data)}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                _plan_sse_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "x-request-id": request_id,
                    "x-execution-mode": "plan",
                },
            )

        # Non-streaming path (existing)

        # call_backend: make an HTTP call to a specific backend with specific messages
        async def call_backend_for_plan(backend_node: str, plan_messages: List[Dict]) -> BackendResult:
            plan_body = dict(body)
            plan_body["messages"] = plan_messages
            plan_body.pop("stream", None)  # planner calls are non-streaming
            return await _make_backend_request(
                backend_node, plan_body, headers, model_name, request_id, start_time,
            )

        plan_result = await engine.execute_plan(
            messages=body.get("messages", []),
            call_backend=call_backend_for_plan,
            planner_backend=planner_node,
            capability_nodes=capability_nodes,
            default_nodes=all_eligible,
            max_subtasks=int(getattr(config, "PLAN_MAX_SUBTASKS", 5)),
            subtask_timeout=float(getattr(config, "PLAN_SUBTASK_TIMEOUT_SECS", 30.0)),
            overall_timeout=exec_config.timeout_secs,
        )
        await _record_multi_exec_metrics("plan", len(plan_result.task_results), 1 if plan_result.final_response and plan_result.final_response.success else 0)

        if plan_result.error and not plan_result.final_response:
            raise HTTPException(status_code=502, detail=f"PLAN execution failed: {plan_result.error}")

        # Use final assembled response as the output — format as OpenAI chat.completion
        final = plan_result.final_response
        if final and final.success:
            resp_body = plan_result_to_openai_response(plan_result)
        else:
            resp_body = {
                "error": plan_result.error or "Plan assembly failed",
                "goal": plan_result.goal,
                "tasks_completed": len([r for r in plan_result.task_results.values() if r.success]),
            }
        out = JSONResponse(content=resp_body)
        # Expose plan metadata in response headers
        out.headers["x-plan-goal"] = (plan_result.goal or "")[:200]
        out.headers["x-plan-tasks"] = str(len(plan_result.tasks))
        out.headers["x-plan-planner"] = planner_node
        backends = list(plan_result.task_results.keys())  # reuse for common header below

    elif exec_config.mode == ExecutionMode.CONSENSUS:
        # Execute consensus with oracle and local/cloud tracking
        consensus = await engine.execute_consensus(
            backends, make_request, exec_config.timeout_secs,
            oracle_backend=oracle_backend,
            oracle_present=oracle_present,
            is_cloud_fn=_is_cloud_backend
        )
        succeeded = sum(1 for r in consensus.all_responses if r.success)
        await _record_multi_exec_metrics("consensus", len(backends), succeeded)
        await _record_consensus_metrics(model_name, consensus, backends, oracle_valid)

        # Build consensus response
        resp_body = {
            "mode": "consensus",
            "winner": {
                "backend": consensus.winner.backend,
                "alias": consensus.winner.alias,
                "response": consensus.winner.response_body,
                "latency_ms": consensus.winner.latency_ms
            },
            "agreement_count": consensus.agreement_count,
            "disagreement": consensus.disagreement,
            "comparison_type": consensus.comparison_type,
            "oracle_backend": consensus.oracle_backend,
            "oracle_present": consensus.oracle_present,
            "oracle_agreed": consensus.oracle_agreed,
            "local_cloud_agreement": consensus.local_cloud_agreement,
            "local_cloud_similarity": consensus.local_cloud_similarity,
            "all_responses": [
                {
                    "backend": r.backend,
                    "alias": r.alias,
                    "success": r.success,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                    "response": r.response_body
                }
                for r in consensus.all_responses
            ]
        }
        out = JSONResponse(content=resp_body)

        # Add consensus-specific headers
        if consensus.disagreement:
            out.headers["x-disagreement"] = "true"

        # Oracle validation headers
        if not oracle_valid:
            out.headers["x-consensus-oracle"] = "invalid"
        elif consensus.oracle_backend:
            out.headers["x-consensus-oracle"] = _get_cloud_backend_name(consensus.oracle_backend) or consensus.oracle_backend
            out.headers["x-oracle-present"] = "true" if consensus.oracle_present else "false"
            if consensus.oracle_present and consensus.oracle_agreed is not None:
                out.headers["x-oracle-agreed"] = "true" if consensus.oracle_agreed else "false"

        # Local vs cloud agreement headers
        if consensus.local_cloud_agreement is not None:
            out.headers["x-local-cloud-agreement"] = "true" if consensus.local_cloud_agreement else "false"
        if consensus.local_cloud_similarity is not None:
            out.headers["x-local-cloud-similarity"] = f"{consensus.local_cloud_similarity:.2f}"

    else:
        raise HTTPException(status_code=400, detail=f"Unknown execution mode: {exec_config.mode}")

    # Common headers
    out.headers["x-execution-mode"] = exec_config.mode.value
    out.headers["x-backends-attempted"] = ",".join(backends)
    out.headers["x-backends-count"] = str(len(backends))
    out.headers["x-selected-model"] = model_name
    out.headers["x-request-id"] = request_id
    out.headers["x-on-demand-wait"] = "true" if on_demand_wait else "false"
    out.headers["x-warm-wait-ms"] = str(warm_wait_ms)
    out.headers["x-model-defaulted"] = "true" if model_defaulted else "false"

    # Find fastest backend
    if exec_config.mode in (ExecutionMode.ALL, ExecutionMode.SEQUENCE, ExecutionMode.CONSENSUS):
        try:
            if exec_config.mode == ExecutionMode.CONSENSUS:
                all_results = consensus.all_responses
            else:
                all_results = results
            successful = [r for r in all_results if r.success]
            if successful:
                fastest = min(successful, key=lambda r: r.latency_ms)
                out.headers["x-fastest-backend"] = fastest.alias or fastest.backend
        except Exception:
            pass

    return out


async def _execute_with_fallback_chain(
    chain_name: str,
    body: Dict[str, Any],
    headers: Dict[str, str],
    request_id: str,
    start_time: float,
    model_name: str
) -> Response:
    """Execute request following a fallback chain until success.

    Each backend in the chain has its own timeout and retry budget.
    Total execution respects FALLBACK_TOTAL_TIMEOUT_SECS.

    Args:
        chain_name: Name of the fallback chain from config
        body: Request body
        headers: Request headers
        request_id: Request ID for tracking
        start_time: Request start time
        model_name: The requested model name

    Returns:
        JSONResponse with the successful result

    Raises:
        HTTPException if all backends in chain fail
    """
    chains = getattr(config, "FALLBACK_CHAINS", {}) or {}
    chain = chains.get(chain_name, [])

    if not chain:
        raise HTTPException(status_code=400, detail=f"Unknown fallback chain: {chain_name}")

    total_timeout = float(getattr(config, "FALLBACK_TOTAL_TIMEOUT_SECS", 120))
    chain_start = time.monotonic()
    last_error = None
    attempted_backends: List[str] = []
    last_status_code = 502

    await _inc_requests_total()

    for fb in chain:
        # Check total timeout
        elapsed = time.monotonic() - chain_start
        if elapsed >= total_timeout:
            logger.warning(f"[req={request_id}] Fallback chain {chain_name} exceeded total timeout")
            break

        backend = fb.backend
        remaining_timeout = min(fb.timeout_secs, total_timeout - elapsed)

        # Resolve "local:auto" to best available local backend
        if backend == "local:auto":
            local_nodes = await get_eligible_nodes(model_name)
            local_nodes = [n for n in local_nodes if not _is_cloud_backend(n)]
            if not local_nodes:
                logger.info(f"[req={request_id}] No local backends available, skipping local:auto")
                continue
            backend = await router.select_node(local_nodes, model_name, redis_client)
            if not backend:
                continue

        # Normalize cloud backend names
        if not _is_cloud_backend(backend) and not ":" in backend:
            # Might be a cloud backend name without prefix
            cloud_backends = getattr(config, "CLOUD_BACKENDS", {}) or {}
            if backend in cloud_backends:
                backend = f"cloud:{backend}"

        # Per-backend retry loop
        for attempt in range(fb.max_retries + 1):
            try:
                attempted_backends.append(backend)
                logger.info(f"[req={request_id}] Fallback chain {chain_name}: trying {backend} (attempt {attempt + 1})")

                result = await asyncio.wait_for(
                    _make_backend_request(
                        backend, body, headers, model_name, request_id, start_time
                    ),
                    timeout=remaining_timeout
                )

                if result.success:
                    logger.info(f"[req={request_id}] Fallback chain {chain_name}: success on {backend}")
                    # Record metrics
                    await redis_client.incrby(f"lb:fallback_chain_success:{chain_name}", 1)
                    await redis_client.incrby(f"lb:fallback_chain_success:{chain_name}:{backend}", 1)

                    out = JSONResponse(content=result.response_body)
                    out.headers["x-fallback-chain"] = chain_name
                    out.headers["x-fallback-backend"] = backend
                    out.headers["x-fallback-attempts"] = str(len(attempted_backends))
                    out.headers["x-request-id"] = request_id
                    out.headers["x-selected-model"] = model_name
                    return out

                last_error = result.error
                last_status_code = result.status_code or 502

                # Don't retry on 4xx (client errors) except 429
                if result.status_code and 400 <= result.status_code < 500:
                    if result.status_code == 429:
                        # Check if we have Retry-After that's within our budget
                        # The rate limit is already recorded, so this backend will be skipped
                        # on next iteration. Move to next backend in chain.
                        logger.info(f"[req={request_id}] Backend {backend} rate limited (429), moving to next")
                        break
                    # Other 4xx - client error, don't retry
                    logger.info(f"[req={request_id}] Backend {backend} returned {result.status_code}, moving to next")
                    break

            except asyncio.TimeoutError:
                last_error = f"Timeout after {remaining_timeout}s"
                logger.info(f"[req={request_id}] Backend {backend} timed out")
                break
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[req={request_id}] Backend {backend} error: {e}")

            # Backoff between retries (only for 5xx/network errors)
            if attempt < fb.max_retries:
                backoff = 0.5 * (attempt + 1)
                await asyncio.sleep(backoff)

    # All backends failed
    logger.warning(f"[req={request_id}] Fallback chain {chain_name} exhausted. Last error: {last_error}")
    await redis_client.incrby(f"lb:fallback_chain_exhausted:{chain_name}", 1)

    raise HTTPException(
        status_code=last_status_code,
        detail={
            "message": f"All backends in chain '{chain_name}' failed",
            "last_error": last_error,
            "attempted": attempted_backends
        },
        headers={"x-fallback-chain": chain_name}
    )


@app.get("/v1/models")
async def get_all_models(request: Request):
    """Aggregates and de-duplicates model lists from all healthy nodes.

    Query params:
      detail=true  — include all models (even unavailable) with x_llb_* availability metadata.
      (default)    — only return models that are currently routable.
    """
    detail = request.query_params.get("detail", "false").lower() in ("1", "true", "yes")
    healthy_nodes = await redis_client.smembers("nodes:healthy")
    all_models = {}
    for node in healthy_nodes:
        models_json = await redis_client.get(f"node:{node}:models")
        if models_json:
            models = json.loads(models_json).get("data", [])
            for model in models:
                mid = model.get("id")
                if mid and mid not in all_models:
                    all_models[mid] = model
    # Ensure every model object has the OpenAI-required fields and check availability
    data = []
    for m in all_models.values():
        avail = await _model_availability(m["id"])
        # Default: filter out models with no eligible nodes
        if not detail and avail["status"] == "unavailable":
            continue
        entry = {
            "id": m["id"],
            "object": m.get("object", "model"),
            "created": m.get("created", 0),
            "owned_by": m.get("owned_by", "local"),
            **{k: v for k, v in m.items() if k not in ("id", "object", "created", "owned_by")},
        }
        if detail:
            entry["x_llb_status"] = avail["status"]
            entry["x_llb_eligible_nodes"] = avail["eligible_nodes"]
            entry["x_llb_total_nodes"] = avail["total_nodes"]
            if avail["unavailable_reasons"]:
                entry["x_llb_unavailable_reasons"] = avail["unavailable_reasons"]
        data.append(entry)
    return {"object": "list", "data": data}

# Latency histogram buckets in seconds
_LAT_BUCKETS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]

async def _record_latency(model: str, node: str, elapsed_secs: float):
    try:
        series_key = f"{model}|{node}"
        await redis_client.sadd("lb:latency_series", series_key)
        # Increment sum and count
        await redis_client.incrbyfloat(f"lb:latency_sum:{series_key}", float(elapsed_secs))
        await redis_client.incrby(f"lb:latency_count:{series_key}", 1)
        # Cumulative buckets: increment every bucket >= observed
        for le in _LAT_BUCKETS:
            if elapsed_secs <= le:
                key = f"lb:latency_bucket:{series_key}:{le}"
                await redis_client.incrby(key, 1)
    except Exception:
        pass
    # Also update Peak EWMA RTT for P2C scoring
    await _record_ewma_rtt(model, node, elapsed_secs)

async def _record_ewma_rtt(model: str, node: str, elapsed_secs: float):
    """Update Peak EWMA RTT for a (model, node) pair.

    Inspired by Finagle/Envoy Peak EWMA (Apache-2.0):
      alpha = 1 - exp(-(now - last_update) / decay_time)
      new_ewma = alpha * observed + (1 - alpha) * old_ewma
    """
    try:
        key_rtt = f"lb:ewma_rtt:{model}|{node}"
        key_ts = f"lb:ewma_ts:{model}|{node}"
        now = time.time()
        decay = config.PEAK_EWMA_DECAY_SECS

        old_rtt_raw = await redis_client.get(key_rtt)
        old_ts_raw = await redis_client.get(key_ts)

        if old_rtt_raw is None or old_ts_raw is None:
            # Cold start: seed with observed value
            new_rtt = elapsed_secs
        else:
            old_rtt = float(old_rtt_raw)
            old_ts = float(old_ts_raw)
            dt = max(now - old_ts, 0.0)
            alpha = 1.0 - math.exp(-dt / decay) if decay > 0 else 1.0
            new_rtt = alpha * elapsed_secs + (1.0 - alpha) * old_rtt

        await redis_client.set(key_rtt, str(new_rtt))
        await redis_client.set(key_ts, str(now))
        # TTL so stale entries clean up (10x decay time)
        ttl = int(decay * 10) or 120
        await redis_client.expire(key_rtt, ttl)
        await redis_client.expire(key_ts, ttl)
    except Exception:
        pass

async def _record_failure_penalty(model: str, node: str):
    """Record a failure as a penalty RTT in the EWMA.

    Instead of a binary circuit breaker, failures inject a large RTT value
    (PEAK_EWMA_PENALTY_RTT, default 30s) into the EWMA, which naturally
    decays over time — graduated degradation instead of hard cutoff.
    """
    await _record_ewma_rtt(model, node, config.PEAK_EWMA_PENALTY_RTT)

async def _record_stream_ttfb(model: str, node: str, elapsed_secs: float):
    try:
        series_key = f"{model}|{node}"
        await redis_client.sadd("lb:stream_ttfb_series", series_key)
        await redis_client.incrbyfloat(f"lb:stream_ttfb_sum:{series_key}", float(elapsed_secs))
        await redis_client.incrby(f"lb:stream_ttfb_count:{series_key}", 1)
        for le in _LAT_BUCKETS:
            if elapsed_secs <= le:
                await redis_client.incrby(f"lb:stream_ttfb_bucket:{series_key}:{le}", 1)
    except Exception:
        pass

async def _record_stream_duration(model: str, node: str, elapsed_secs: float):
    try:
        series_key = f"{model}|{node}"
        await redis_client.sadd("lb:stream_duration_series", series_key)
        await redis_client.incrbyfloat(f"lb:stream_duration_sum:{series_key}", float(elapsed_secs))
        await redis_client.incrby(f"lb:stream_duration_count:{series_key}", 1)
        for le in _LAT_BUCKETS:
            if elapsed_secs <= le:
                await redis_client.incrby(f"lb:stream_duration_bucket:{series_key}:{le}", 1)
    except Exception:
        pass

async def stream_backend(node: str, messages: list, model: str, request_id: str):
    """Stream a custom messages payload to a backend node.

    Used by the PLAN assembly phase to stream synthesized responses with
    different messages than the original request body.
    """
    url = f"http://{node}/v1/chat/completions"
    req_body = {"model": model, "messages": messages, "stream": True}
    req_headers = {"x-request-id": request_id}
    async with http_client.stream("POST", url, json=req_body, headers=req_headers) as response:
        async for chunk in response.aiter_bytes():
            yield chunk


async def _record_output_tokens(model: str, node: str, tokens: int):
    """Update EWMA of output tokens per model+node for cost-aware P2C scoring."""
    try:
        alpha = float(getattr(config, "COST_EWMA_ALPHA", 0.3))
        ttl = int(getattr(config, "COST_EWMA_TTL_SECS", 3600))
        key_ewma = f"lb:output_tokens_ewma:{model}|{node}"
        key_count = f"lb:output_tokens_count:{model}|{node}"
        current = await redis_client.get(key_ewma)
        new_ewma = float(tokens) if current is None else (alpha * float(tokens) + (1.0 - alpha) * float(current))
        await redis_client.set(key_ewma, new_ewma)
        await redis_client.expire(key_ewma, ttl)
        await redis_client.incrby(key_count, 1)
        await redis_client.expire(key_count, ttl)
    except Exception:
        pass


@app.api_route("/v1/chat/completions", methods=["POST"])
async def chat_completions(request: Request):
    """Receives a chat completion request, routes it, and streams the response."""
    try:
        raw_body = await request.json()
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON payload for /v1/chat/completions: %s", exc)
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.")
    except Exception as exc:
        logger.warning("Failed to parse JSON payload for /v1/chat/completions: %s", exc)
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.")

    sanitized = sanitize_chat_request(raw_body)
    if sanitized.default_applied:
        logger.info(
            "Chat request missing model; applied default '%s'", sanitized.default_applied
        )
    if sanitized.missing_fields:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required fields.",
                "missing": sanitized.missing_fields,
            },
        )

    body = sanitized.payload
    model_name = sanitized.model_name
    is_stream = bool(body.get("stream", False))
    model_defaulted = sanitized.default_applied is not None
    start_time = time.monotonic()

    # AILB-MT-1: complexity routing telemetry context (populated below if COMPLEXITY_ROUTING_ENABLED)
    _complexity_score: Optional[float] = None
    _complexity_tier: Optional[str] = None  # "small" | "medium" | "large" | None

    # Resolve auto/default sentinel to a concrete model id
    if _is_model_sentinel(model_name):
        prefer_intersection = request.query_params.get("require_all", "false").lower() in ("1", "true", "yes")
        resolved = None

        # Complexity-based tier selection (RouteLLM-inspired, Apache-2.0 — lm-sys/RouteLLM)
        if getattr(config, "COMPLEXITY_ROUTING_ENABLED", False):
            messages = body.get("messages", [])
            if messages:
                _complexity_router = ComplexityRoutingStrategy()
                score = _complexity_router.score_prompt_complexity(messages)
                _complexity_score = score
                model_classes = getattr(config, "MODEL_CLASSES", {}) or {}
                tier_candidates = _complexity_router.get_complexity_model(score, model_classes)
                # Determine tier label for telemetry
                if score < ComplexityRoutingStrategy.LOW_THRESHOLD:
                    _complexity_tier = "small"
                elif score <= ComplexityRoutingStrategy.HIGH_THRESHOLD:
                    _complexity_tier = "medium"
                else:
                    _complexity_tier = "large"
                if tier_candidates:
                    # Try each candidate in tier order until we find a healthy one
                    for candidate in tier_candidates:
                        candidate_nodes = await get_eligible_nodes(candidate)
                        if candidate_nodes:
                            resolved = candidate
                            logger.info(
                                "Complexity routing: score=%.2f tier=%s selected tier candidate '%s'",
                                score, _complexity_tier, candidate,
                            )
                            break

        if not resolved:
            resolved = await _resolve_auto_model(prefer_intersection=prefer_intersection)
        if not resolved:
            raise HTTPException(status_code=404, detail="No models available for auto selection.")
        body["model"] = resolved
        model_name = resolved

    eligible_nodes = await get_eligible_nodes(model_name)

    # Track on-demand warm/wait for observability headers
    on_demand_wait = False
    warm_wait_ms = 0

    if not eligible_nodes:
        # Diagnose why before deciding whether to warm or fail fast
        diagnosis = await _diagnose_unroutable(model_name)
        if diagnosis["reason"] in ("circuit_open", "rate_limited"):
            # Model is known but all nodes are temporarily unavailable — fail fast
            retry_after = diagnosis["retry_after_secs"] or _cb_cooldown_secs()
            raise HTTPException(
                status_code=503,
                detail={
                    "message": f"Model '{model_name}' is temporarily unavailable: {diagnosis['reason'].replace('_', ' ')} on all nodes",
                    "code": "model_unavailable",
                    "x_llb_diagnosis": diagnosis,
                },
                headers={"Retry-After": str(retry_after), "x-request-id": request.headers.get("x-request-id", "")},
            )
        # Model is cold/unknown — try on-demand warm
        try:
            t0_warm = time.monotonic()
            waited = await _on_demand_wait_for_model(model_name)
            warm_wait_ms = int((time.monotonic() - t0_warm) * 1000)
            if waited:
                eligible_nodes = waited
                on_demand_wait = True
        except Exception:
            pass
    if not eligible_nodes:
        # Try cross-model fallback if enabled
        fb_model: Optional[str] = None
        if getattr(config, "CROSS_MODEL_FALLBACK", False):
            # If original is auto/sentinel, re-resolve; else try class-sibling then auto
            if _is_model_sentinel(model_name):
                fb_model = await _resolve_auto_model(prefer_intersection=False)
            else:
                fb_model = await _find_fallback_model(model_name)
                if not fb_model:
                    fb_model = await _resolve_auto_model(prefer_intersection=False)
            if fb_model:
                fb_nodes = await get_eligible_nodes(fb_model)
                if fb_nodes:
                    logger.warning("No healthy nodes for '%s'; falling back to '%s'", model_name, fb_model)
                    body["model"] = fb_model
                    model_name = fb_model
                    eligible_nodes = fb_nodes
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    _llb_hdrs = await _routing_headers(model_name, len(eligible_nodes))

    # Check for multi-backend execution mode or fallback chain
    exec_config = _parse_execution_mode(request)

    # Handle fallback chain execution (takes precedence over other modes)
    if exec_config and exec_config.fallback_chain:
        headers = {
            key: value for key, value in request.headers.items()
            if key.lower() not in ('host', 'content-length', 'x-fallback-chain', 'x-execution-mode')
        }
        headers["x-request-id"] = request_id
        return await _execute_with_fallback_chain(
            chain_name=exec_config.fallback_chain,
            body=body,
            headers=headers,
            request_id=request_id,
            start_time=start_time,
            model_name=model_name
        )

    # For multi-exec, try model equivalents if no direct match
    if not eligible_nodes and exec_config:
        equiv_nodes = await _get_eligible_nodes_for_equivalents(model_name)
        if equiv_nodes:
            eligible_nodes = equiv_nodes

    if not eligible_nodes:
        raise HTTPException(status_code=404, detail=f"No healthy nodes found for model '{model_name}'.")
    if exec_config:
        return await _handle_multi_backend_execution(
            request=request,
            body=body,
            ext_config=exec_config,  # ExtendedExecutionConfig with oracle/fallback support
            model_name=model_name,
            eligible_nodes=eligible_nodes,
            request_id=request_id,
            start_time=start_time,
            is_stream=is_stream,
            on_demand_wait=on_demand_wait,
            warm_wait_ms=warm_wait_ms,
            model_defaulted=model_defaulted
        )

    # Standard single-backend path continues below
    headers = {key: value for key, value in request.headers.items() if key.lower() not in ('host', 'content-length')}
    headers["x-request-id"] = request_id
    forced_node = request.query_params.get("node")
    session_id = request.headers.get("x-session-id")

    async def attempt_stream(node: str):
        # Determine URL, headers, and request body based on backend type (cloud vs local)
        adapter = None
        req_body = body
        if _is_cloud_backend(node):
            url = _get_cloud_url(node, "/chat/completions")
            if not url:
                raise httpx.RequestError(f"Cloud backend URL not configured for {node}")
            req_headers = _get_cloud_headers(node, headers)
            # Get provider adapter for request/response transformation
            cfg = _get_cloud_config(node)
            if cfg:
                provider_type = cfg.get("provider_type", "openai")
                adapter = get_adapter(provider_type)
                req_body = adapter.transform_request(body)
        else:
            url = f"http://{node}/v1/chat/completions"
            req_headers = headers

        logger.info("[req=%s] Routing request for model '%s' to %s", request_id, model_name, node)
        acquired = await _acquire_slot(node)
        if not acquired:
            logger.info("[req=%s] Node %s at capacity; skipping", request_id, node)
            raise CapacityError("node")
        model_ok = await _acquire_model_slot(model_name)
        if not model_ok:
            await _dec_inflight(node)
            raise CapacityError("model")
        try:
            async with http_client.stream("POST", url, json=req_body, headers=req_headers) as response:
                # Handle rate limiting (429)
                if response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    retry_after_secs = None
                    if retry_after:
                        try:
                            retry_after_secs = float(retry_after)
                        except ValueError:
                            pass
                    await _record_rate_limit(node, retry_after_secs)
                    raise RateLimitError(node, retry_after_secs)
                # Treat 5xx and 404 as retryable failure across nodes
                if response.status_code and (response.status_code >= 500 or response.status_code == 404):
                    await _record_failure(node, model_name)
                    raise httpx.HTTPStatusError("Upstream retryable error", request=response.request, response=response)

                # Stream with optional SSE transformation for cloud backends
                if adapter:
                    # Transform SSE stream using provider adapter
                    buffer = ""
                    current_event_type = None
                    async for chunk in response.aiter_bytes():
                        buffer += chunk.decode("utf-8", errors="replace")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.rstrip("\r")
                            if not line:
                                # Empty line - SSE event boundary
                                continue
                            # Parse the line
                            parsed = adapter.parse_sse_line(line)
                            if parsed is None:
                                continue
                            # Check for event type marker (Anthropic format)
                            if "_event_type" in parsed:
                                current_event_type = parsed["_event_type"]
                                continue
                            # Check for done signal
                            if parsed.get("done"):
                                yield b"data: [DONE]\n\n"
                                continue
                            # Merge event type into chunk if present
                            if current_event_type and "type" not in parsed:
                                parsed["type"] = current_event_type
                            current_event_type = None
                            # Transform to OpenAI format
                            transformed = adapter.transform_stream_chunk(parsed)
                            if transformed:
                                yield f"data: {json.dumps(transformed)}\n\n".encode()
                else:
                    # Passthrough for local/OpenAI-compatible backends
                    # If cost-aware P2C is enabled, parse usage from stream chunks
                    _beta = float(getattr(config, "P2C_BETA", 0.0))
                    _last_completion_tokens = 0
                    async for chunk in response.aiter_bytes():
                        if _beta > 0:
                            try:
                                text = chunk.decode("utf-8", errors="ignore")
                                for line in text.split("\n"):
                                    if line.startswith("data: ") and "[DONE]" not in line:
                                        payload = json.loads(line[6:])
                                        ct = int((payload.get("usage") or {}).get("completion_tokens", 0))
                                        if ct > 0:
                                            _last_completion_tokens = ct
                            except Exception:
                                pass
                        yield chunk
                    if _beta > 0 and _last_completion_tokens > 0:
                        await _record_output_tokens(model_name, node, _last_completion_tokens)
            await _record_success(node, model_name)
            await _clear_rate_limit(node)
        finally:
            await _dec_inflight(node)
            await _dec_model(model_name)

    async def stream_with_failover(nodes: List[str]):
        attempts = 0
        tried = set()
        tried_order = []
        last_error = None
        all_capacity = True
        first_byte_recorded = False
        same_node_retries = 0
        # We allow at most first try + MAX_RETRIES additional
        budget = min(len(nodes), 1 + config.MAX_RETRIES)
        # When only 1 node exists, allow same-node retries with delay
        if len(nodes) == 1 and config.SAME_NODE_RETRY_ENABLED:
            budget = 1 + config.SAME_NODE_RETRY_MAX
        while attempts < budget:
            # prefer nodes without open circuit
            remaining = [n for n in nodes if n not in tried]
            # Same-node retry: if no untried nodes remain, clear tried and delay
            if not remaining and len(nodes) == 1 and config.SAME_NODE_RETRY_ENABLED and same_node_retries < config.SAME_NODE_RETRY_MAX:
                same_node_retries += 1
                logger.info("[req=stream] Same-node retry %d/%d for %s after %.1fs delay",
                            same_node_retries, config.SAME_NODE_RETRY_MAX, nodes[0], config.SAME_NODE_RETRY_DELAY_SECS)
                await asyncio.sleep(config.SAME_NODE_RETRY_DELAY_SECS)
                tried.clear()
                remaining = list(nodes)
            closed = []
            for n in remaining:
                if not await _is_circuit_open(n):
                    closed.append(n)
            pool = closed if closed else remaining  # half-open if all open
            # Sticky preference on first pick
            candidate = None
            if attempts == 0 and not forced_node:
                sticky = await _get_sticky_node(session_id, model_name)
                if sticky and sticky in pool:
                    candidate = sticky
            if candidate is None:
                candidate = await router.select_node(pool, model_name, redis_client)
            if candidate is None:
                break
            tried.add(candidate)
            tried_order.append(candidate)
            attempts += 1
            # Emit SSE meta for initial attempt or failover
            meta = {
                "request_id": request_id,
                "model": model_name,
                "node": candidate,
                "attempts": attempts,
                "failover_count": max(0, attempts - 1),
                "event": "failover" if attempts > 1 else "meta",
            }
            yield f"event: {meta['event']}\n".encode()
            yield ("data: " + json.dumps(meta) + "\n\n").encode()
            try:
                await _inc_requests_total()
                # Streaming hedging decision
                hedging_enabled = bool(getattr(config, "HEDGING_ENABLED", True))
                small_only = bool(getattr(config, "HEDGING_SMALL_MODELS_ONLY", True))
                should_hedge = hedging_enabled and (not small_only or _is_small_model(model_name)) and len(pool) >= 2 and attempts == 1
                if not should_hedge:
                    async for data in attempt_stream(candidate):
                        if not first_byte_recorded:
                            try:
                                await _record_stream_ttfb(model_name, candidate, time.monotonic() - start_time)
                            except Exception:
                                pass
                            first_byte_recorded = True
                        yield data
                    # Success: set sticky mapping
                    await _set_sticky_node(session_id, model_name, candidate)
                    # Record failovers metric for streaming
                    if attempts > 1:
                        try:
                            await redis_client.incrby("lb:failovers_total", attempts - 1)
                            await redis_client.incrby(f"lb:model:{model_name}:failovers_total", attempts - 1)
                        except Exception:
                            pass
                    # Record total stream duration
                    try:
                        await _record_stream_duration(model_name, candidate, time.monotonic() - start_time)
                    except Exception:
                        pass
                    return
                # Hedged path for streaming
                try:
                    p95 = await _estimate_pool_p95(model_name, pool)
                except Exception:
                    p95 = 1.0
                delay_ms = min(int(getattr(config, "HEDGING_MAX_DELAY_MS", 800)), int(1000 * float(getattr(config, "HEDGING_P95_FRACTION", 0.6)) * p95))
                delay_s = max(0, delay_ms) / 1000.0
                # Pick a secondary distinct node
                second_pool = [n for n in pool if n != candidate]
                secondary = await router.select_node(second_pool, model_name, redis_client) if second_pool else None
                if secondary is None:
                    async for data in attempt_stream(candidate):
                        if not first_byte_recorded:
                            try:
                                await _record_stream_ttfb(model_name, candidate, time.monotonic() - start_time)
                            except Exception:
                                pass
                            first_byte_recorded = True
                        yield data
                    await _set_sticky_node(session_id, model_name, candidate)
                    try:
                        await _record_stream_duration(model_name, candidate, time.monotonic() - start_time)
                    except Exception:
                        pass
                    return
                # Zero-delay special case
                primary_gen = attempt_stream(candidate)
                if delay_s <= 0:
                    try:
                        first = await asyncio.wait_for(primary_gen.__anext__(), timeout=0)
                        if not first_byte_recorded:
                            try:
                                await _record_stream_ttfb(model_name, candidate, time.monotonic() - start_time)
                            except Exception:
                                pass
                            first_byte_recorded = True
                        yield first
                        async for data in primary_gen:
                            yield data
                        await _set_sticky_node(session_id, model_name, candidate)
                        try:
                            await _record_stream_duration(model_name, candidate, time.monotonic() - start_time)
                        except Exception:
                            pass
                        return
                    except (asyncio.TimeoutError, StopAsyncIteration, httpx.RequestError, httpx.HTTPStatusError, CapacityError, RateLimitError):
                        # Announce hedge start
                        evt = {
                            "request_id": request_id,
                            "model": model_name,
                            "primary": candidate,
                            "secondary": secondary,
                            "event": "hedge_start",
                        }
                        yield b"event: hedge_start\n"
                        yield ("data: " + json.dumps(evt) + "\n\n").encode()
                        try:
                            await redis_client.incrby("lb:hedges_total", 1)
                        except Exception:
                            pass
                        sgen = attempt_stream(secondary)
                        first = await sgen.__anext__()
                        if not first_byte_recorded:
                            try:
                                await _record_stream_ttfb(model_name, secondary, time.monotonic() - start_time)
                            except Exception:
                                pass
                            first_byte_recorded = True
                        yield first
                        async for data in sgen:
                            yield data
                        await _set_sticky_node(session_id, model_name, secondary)
                        try:
                            await redis_client.incrby(f"lb:hedge_wins:{model_name}|{secondary}", 1)
                            await redis_client.sadd("lb:hedge_wins_models", model_name)
                            await redis_client.incrby(f"lb:hedge_wins_model:{model_name}", 1)
                        except Exception:
                            pass
                        # Announce hedge winner
                        w_evt = {
                            "request_id": request_id,
                            "model": model_name,
                            "primary": candidate,
                            "secondary": secondary,
                            "winner": secondary,
                            "event": "hedge_winner",
                        }
                        yield b"event: hedge_winner\n"
                        yield ("data: " + json.dumps(w_evt) + "\n\n").encode()
                        try:
                            await _record_stream_duration(model_name, secondary, time.monotonic() - start_time)
                        except Exception:
                            pass
                        return
                # Delayed hedge: wait for first chunk from primary within delay, else race
                try:
                    first = await asyncio.wait_for(attempt_stream(candidate).__anext__(), timeout=delay_s)
                    if not first_byte_recorded:
                        try:
                            await _record_stream_ttfb(model_name, candidate, time.monotonic() - start_time)
                        except Exception:
                            pass
                        first_byte_recorded = True
                    yield first
                    async for data in attempt_stream(candidate):
                        yield data
                    await _set_sticky_node(session_id, model_name, candidate)
                    try:
                        await _record_stream_duration(model_name, candidate, time.monotonic() - start_time)
                    except Exception:
                        pass
                    return
                except asyncio.TimeoutError:
                    # Announce hedge start
                    evt = {
                        "request_id": request_id,
                        "model": model_name,
                        "primary": candidate,
                        "secondary": secondary,
                        "event": "hedge_start",
                    }
                    yield b"event: hedge_start\n"
                    yield ("data: " + json.dumps(evt) + "\n\n").encode()
                    try:
                        await redis_client.incrby("lb:hedges_total", 1)
                    except Exception:
                        pass
                    pgen = attempt_stream(candidate)
                    sgen = attempt_stream(secondary)
                    pfirst = asyncio.create_task(pgen.__anext__())
                    sfirst = asyncio.create_task(sgen.__anext__())
                    done, pending = await asyncio.wait({pfirst, sfirst}, return_when=asyncio.FIRST_COMPLETED)
                    dt = next(iter(done))
                    winner = candidate if dt is pfirst else secondary
                    wgen = pgen if dt is pfirst else sgen
                    # Cancel the other pending task
                    for t in pending:
                        t.cancel()
                    try:
                        first = dt.result()
                    except Exception as e:
                        # try the other if failed
                        try:
                            other = next(iter(pending))
                            first = await other
                            winner = secondary if winner == candidate else candidate
                            wgen = sgen if wgen is pgen else pgen
                        except Exception:
                            raise e
                    if not first_byte_recorded:
                        try:
                            await _record_stream_ttfb(model_name, winner, time.monotonic() - start_time)
                        except Exception:
                            pass
                        first_byte_recorded = True
                    yield first
                    async for data in wgen:
                        yield data
                    await _set_sticky_node(session_id, model_name, winner)
                    try:
                        await redis_client.incrby(f"lb:hedge_wins:{model_name}|{winner}", 1)
                    except Exception:
                        pass
                    # Announce hedge winner
                    w_evt = {
                        "request_id": request_id,
                        "model": model_name,
                        "primary": candidate,
                        "secondary": secondary,
                        "winner": winner,
                        "event": "hedge_winner",
                    }
                    yield b"event: hedge_winner\n"
                    yield ("data: " + json.dumps(w_evt) + "\n\n").encode()
                    try:
                        await _record_stream_duration(model_name, winner, time.monotonic() - start_time)
                    except Exception:
                        pass
                    return
            except (httpx.RequestError, httpx.HTTPStatusError, CapacityError, RateLimitError) as e:
                logger.warning("[req=%s] Upstream error from %s: %s (attempt %d/%d)", request_id, candidate, _describe_upstream_error(e), attempts, budget)
                await _record_failure(candidate, model_name)
                last_error = e
                if not isinstance(e, CapacityError):
                    all_capacity = False
                continue
        # If all attempts failed, return a final error message chunk
        err_msg = {
            "error": {
                "message": "All upstream nodes failed for model.",
                "model": model_name,
                "attempts": attempts,
                "nodes": tried_order
            }
        }
        if last_error:
            logger.error("All upstream attempts failed: %s", last_error)
        # Emit error as a named SSE event so clients can parse it cleanly
        yield f"event: error\ndata: {json.dumps(err_msg)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    if is_stream:
        # Streaming behavior (existing)
        if forced_node:
            if forced_node not in eligible_nodes:
                raise HTTPException(status_code=404, detail=f"Requested node '{forced_node}' is not eligible for model '{model_name}'.")
            async def stream_single():
                try:
                    await _inc_requests_total()
                    async for data in attempt_stream(forced_node):
                        yield data
                    await _set_sticky_node(session_id, model_name, forced_node)
                except (httpx.RequestError, httpx.HTTPStatusError, CapacityError, RateLimitError) as e:
                    err = {"error": {"message": f"Upstream error from {forced_node}: {e}", "type": "server_error"}}
                    yield f"event: error\ndata: {json.dumps(err)}\n\n".encode()
                    yield b"data: [DONE]\n\n"
            return StreamingResponse(stream_single(), media_type="text/event-stream", headers={
                "x-selected-model": model_name,
                "x-routed-node": forced_node,
                "x-request-id": request_id,
                "x-on-demand-wait": "true" if on_demand_wait else "false",
                "x-warm-wait-ms": str(warm_wait_ms),
                "x-capacity-state": "ok",
                "x-model-defaulted": "true" if model_defaulted else "false",
                **_llb_hdrs,
            })
        else:
            # Pre-check: if all nodes saturated (inflight >= maxconn), respond 429 immediately
            try:
                healthy = eligible_nodes
                saturated = True
                model_sat = False
                try:
                    m_inflight = await redis_client.get(f"model:{model_name}:inflight")
                    m_max = await redis_client.get(f"model:{model_name}:maxconn")
                    m_inflight = int(m_inflight) if m_inflight else 0
                    m_max = int(m_max) if m_max not in (None, "", "0") else None
                    if m_max is not None and m_inflight >= m_max:
                        model_sat = True
                except Exception:
                    pass
                for n in healthy:
                    inflight_val = await redis_client.get(f"node:{n}:inflight")
                    maxconn_val = await redis_client.get(f"node:{n}:maxconn")
                    inflight = int(inflight_val) if inflight_val is not None else 0
                    maxconn = int(maxconn_val) if maxconn_val not in (None, "", "0") else None
                    # If no maxconn configured, treat as not saturated
                    if maxconn is None or inflight < maxconn:
                        saturated = False
                        break
                if saturated:
                    raise HTTPException(status_code=429, detail="All nodes are at capacity for this model.", headers={"Retry-After": str(config.RETRY_AFTER_SECS), "x-request-id": request_id, "x-capacity-state": ("model_saturated" if model_sat else "cluster_saturated")})
            except HTTPException:
                raise
            except Exception:
                pass
            # Compute initial routed node for header (best-effort)
            try:
                remaining = list(eligible_nodes)
                closed = []
                for n in remaining:
                    if not await _is_circuit_open(n):
                        closed.append(n)
                pool = closed if closed else remaining
                first = None
                if not forced_node:
                    sticky = await _get_sticky_node(session_id, model_name)
                    if sticky and sticky in pool:
                        first = sticky
                if first is None:
                    first = await router.select_node(pool, model_name, redis_client)
            except Exception:
                first = None
            return StreamingResponse(stream_with_failover(eligible_nodes), media_type="text/event-stream", headers={
                "x-selected-model": model_name,
                "x-routed-node": first or "",
                "x-request-id": request_id,
                "x-on-demand-wait": "true" if on_demand_wait else "false",
                "x-warm-wait-ms": str(warm_wait_ms),
                "x-capacity-state": "ok",
                "x-model-defaulted": "true" if model_defaulted else "false",
                **_llb_hdrs,
            })

    # Non-streaming behavior: aggregate JSON and return a Response
    async def attempt_request(node: str):
        # Determine URL, headers, and request body based on backend type (cloud vs local)
        adapter = None
        req_body = body
        if _is_cloud_backend(node):
            url = _get_cloud_url(node, "/chat/completions")
            if not url:
                raise httpx.RequestError(f"Cloud backend URL not configured for {node}")
            req_headers = _get_cloud_headers(node, headers)
            # Get provider adapter for request/response transformation
            cfg = _get_cloud_config(node)
            if cfg:
                provider_type = cfg.get("provider_type", "openai")
                adapter = get_adapter(provider_type)
                req_body = adapter.transform_request(body)
        else:
            url = f"http://{node}/v1/chat/completions"
            req_headers = headers

        logger.info("[req=%s] Routing (non-stream) request for model '%s' to %s", request_id, model_name, node)
        acquired = await _acquire_slot(node)
        if not acquired:
            logger.info("[req=%s] Node %s at capacity; skipping", request_id, node)
            raise CapacityError("node")
        model_ok = await _acquire_model_slot(model_name)
        if not model_ok:
            await _dec_inflight(node)
            raise CapacityError("model")
        try:
            try:
                resp = await http_client.post(url, json=req_body, headers=req_headers)
                status_code = resp.status_code
                content = resp.content
                ctype = resp.headers.get("content-type", "application/json")
            except AttributeError:
                # Compatibility for test fakes that implement only `.stream(...)`
                async with http_client.stream("POST", url, json=req_body, headers=req_headers) as response:
                    status_code = getattr(response, "status_code", 200)
                    # Handle rate limiting (429)
                    if status_code == 429:
                        retry_after = getattr(response, "headers", {}).get("retry-after")
                        retry_after_secs = float(retry_after) if retry_after else None
                        await _record_rate_limit(node, retry_after_secs)
                        raise RateLimitError(node, retry_after_secs)
                    # Treat 5xx and 404 as retryable across nodes (LM Studio variants sometimes 404 chat endpoint)
                    if status_code and (status_code >= 500 or status_code == 404):
                        await _record_failure(node, model_name)
                        raise httpx.HTTPStatusError("Upstream retryable error", request=None, response=None)
                    chunks = bytearray()
                    async for chunk in response.aiter_bytes():
                        chunks.extend(chunk)
                    content = bytes(chunks)
                    ctype = "application/json"
            # Handle rate limiting (429)
            if status_code == 429:
                retry_after = resp.headers.get("retry-after")
                retry_after_secs = float(retry_after) if retry_after else None
                await _record_rate_limit(node, retry_after_secs)
                raise RateLimitError(node, retry_after_secs)
            # Evaluate retry conditions for normal client path
            if status_code and (status_code >= 500 or status_code == 404):
                await _record_failure(node, model_name)
                raise httpx.HTTPStatusError("Upstream retryable error", request=None, response=None)
            await _record_success(node, model_name)
            await _clear_rate_limit(node)
            # Cost-aware P2C: record output tokens in EWMA when beta > 0
            if float(getattr(config, "P2C_BETA", 0.0)) > 0 and status_code == 200:
                try:
                    _resp_data = json.loads(content)
                    _ct = int((_resp_data.get("usage") or {}).get("completion_tokens", 0))
                    if _ct > 0:
                        await _record_output_tokens(model_name, node, _ct)
                except Exception:
                    pass
            # Transform response for cloud backends
            if adapter and status_code == 200:
                try:
                    response_body = json.loads(content)
                    response_body = adapter.transform_response(response_body)
                    content = json.dumps(response_body).encode()
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning("[req=%s] Failed to transform cloud response: %s", request_id, e)
            out = Response(content=content, media_type=ctype, status_code=status_code)
            out.headers["x-selected-model"] = model_name
            out.headers["x-routed-node"] = node
            out.headers["x-request-id"] = request_id
            out.headers["x-on-demand-wait"] = "true" if on_demand_wait else "false"
            out.headers["x-warm-wait-ms"] = str(warm_wait_ms)
            out.headers["x-capacity-state"] = "ok"
            out.headers["x-model-defaulted"] = "true" if model_defaulted else "false"
            for hk, hv in _llb_hdrs.items():
                out.headers[hk] = hv
            # Success: set sticky mapping
            await _set_sticky_node(session_id, model_name, node)
            # AILB-MT-1: emit complexity routing telemetry record
            if _complexity_score is not None:
                try:
                    import os as _os
                    _elapsed_ms = int((time.monotonic() - start_time) * 1000)
                    _output_tokens: Optional[int] = None
                    try:
                        _resp_json = json.loads(content)
                        _output_tokens = (
                            _resp_json.get("usage", {}).get("completion_tokens")
                            or _resp_json.get("usage", {}).get("output_tokens")
                        )
                    except Exception:
                        pass
                    _log_path = _os.environ.get("COMPLEXITY_ROUTING_LOG", "/tmp/complexity_routing.jsonl")
                    _record = json.dumps({
                        "ts": time.time(),
                        "request_id": request_id,
                        "model": model_name,
                        "node": node,
                        "complexity_score": round(_complexity_score, 4),
                        "complexity_tier": _complexity_tier,
                        "elapsed_ms": _elapsed_ms,
                        "output_tokens": _output_tokens,
                        "status_code": status_code,
                    })
                    with open(_log_path, "a") as _f:
                        _f.write(_record + "\n")
                except Exception:
                    pass
            return out
        finally:
            await _dec_inflight(node)
            await _dec_model(model_name)

    if forced_node:
        if forced_node not in eligible_nodes:
            raise HTTPException(status_code=404, detail=f"Requested node '{forced_node}' is not eligible for model '{model_name}'.")
        try:
            await _inc_requests_total()
            return await attempt_request(forced_node)
        except (httpx.RequestError, httpx.HTTPStatusError, CapacityError, RateLimitError) as e:
            raise HTTPException(status_code=502, detail=f"Upstream error from {forced_node}: {e}")

    # Failover loop (non-stream)
    attempts = 0
    tried = set()
    last_error = None
    budget = min(len(eligible_nodes), 1 + config.MAX_RETRIES)
    all_capacity = True
    tried_order = []
    same_node_retries = 0
    # When only 1 node exists, allow same-node retries with delay
    if len(eligible_nodes) == 1 and config.SAME_NODE_RETRY_ENABLED:
        budget = 1 + config.SAME_NODE_RETRY_MAX
    while attempts < budget:
        remaining = [n for n in eligible_nodes if n not in tried]
        # Same-node retry: if no untried nodes remain, clear tried and delay
        if not remaining and len(eligible_nodes) == 1 and config.SAME_NODE_RETRY_ENABLED and same_node_retries < config.SAME_NODE_RETRY_MAX:
            same_node_retries += 1
            logger.info("[req=%s] Same-node retry %d/%d for %s after %.1fs delay",
                        request_id, same_node_retries, config.SAME_NODE_RETRY_MAX, eligible_nodes[0], config.SAME_NODE_RETRY_DELAY_SECS)
            await asyncio.sleep(config.SAME_NODE_RETRY_DELAY_SECS)
            tried.clear()
            remaining = list(eligible_nodes)
        closed = []
        for n in remaining:
            if not await _is_circuit_open(n):
                closed.append(n)
        pool = closed if closed else remaining
        candidate = None
        if attempts == 0 and not forced_node:
            sticky = await _get_sticky_node(session_id, model_name)
            if sticky and sticky in pool:
                candidate = sticky
        if candidate is None:
            candidate = await router.select_node(pool, model_name, redis_client)
        if candidate is None:
            break
        tried.add(candidate)
        tried_order.append(candidate)
        attempts += 1

        # Hedging: optional duplicate attempt after delay for small models
        hedging_enabled = bool(getattr(config, "HEDGING_ENABLED", True))
        small_only = bool(getattr(config, "HEDGING_SMALL_MODELS_ONLY", True))
        should_hedge = hedging_enabled and (not small_only or _is_small_model(model_name)) and len(pool) >= 2 and attempts == 1

        if should_hedge:
            # Select a distinct secondary candidate
            second_pool = [n for n in pool if n != candidate]
            secondary = await router.select_node(second_pool, model_name, redis_client) if second_pool else None
            if secondary is not None:
                # Estimate delay
                p95 = await _estimate_pool_p95(model_name, pool)
                delay_ms = min(int(getattr(config, "HEDGING_MAX_DELAY_MS", 800)), int(1000 * float(getattr(config, "HEDGING_P95_FRACTION", 0.6)) * p95))
                delay_s = max(0, delay_ms) / 1000.0
                # Special case: zero delay -> try primary, and on immediate failure hedge to secondary synchronously
                if delay_s <= 0:
                    try:
                        await _inc_requests_total()
                        resp = await attempt_request(candidate)
                        resp.headers["x-attempts"] = str(attempts)
                        resp.headers["x-failover-count"] = str(max(0, attempts - 1))
                        resp.headers["x-retry-count"] = str(max(0, attempts - 1))
                        resp.headers["x-hedged"] = "false"
                        resp.headers["x-hedge-winner"] = ""
                        try:
                            await _record_latency(model_name, candidate, time.monotonic() - start_time)
                        except Exception:
                            pass
                        return resp
                    except (httpx.RequestError, httpx.HTTPStatusError, CapacityError, RateLimitError):
                        # Immediate hedge to secondary
                        try:
                            await redis_client.incrby("lb:hedges_total", 1)
                        except Exception:
                            pass
                        await _inc_requests_total()
                        resp = await attempt_request(secondary)
                        resp.headers["x-attempts"] = str(attempts)
                        resp.headers["x-failover-count"] = str(max(0, attempts - 1))
                        resp.headers["x-retry-count"] = str(max(0, attempts - 1))
                        resp.headers["x-hedged"] = "true"
                        resp.headers["x-hedge-winner"] = secondary
                        try:
                            await redis_client.incrby(f"lb:hedge_wins:{model_name}|{secondary}", 1)
                            await redis_client.sadd("lb:hedge_wins_models", model_name)
                            await redis_client.incrby(f"lb:hedge_wins_model:{model_name}", 1)
                        except Exception:
                            pass
                        try:
                            await _record_latency(model_name, secondary, time.monotonic() - start_time)
                        except Exception:
                            pass
                        return resp

                # Launch primary task
                await _inc_requests_total()
                t1 = asyncio.create_task(attempt_request(candidate))
                winner_node = None
                try:
                    # Give primary a head start equal to delay
                    try:
                        resp = await asyncio.wait_for(t1, timeout=delay_s)
                        # Primary finished within delay; no hedging launched
                        resp.headers["x-attempts"] = str(attempts)
                        resp.headers["x-failover-count"] = str(max(0, attempts - 1))
                        resp.headers["x-retry-count"] = str(max(0, attempts - 1))
                        resp.headers["x-hedged"] = "false"
                        resp.headers["x-hedge-winner"] = ""
                        try:
                            await _record_latency(model_name, candidate, time.monotonic() - start_time)
                        except Exception:
                            pass
                        return resp
                    except asyncio.TimeoutError:
                        # Launch hedge
                        try:
                            await redis_client.incrby("lb:hedges_total", 1)
                        except Exception:
                            pass
                        t2 = asyncio.create_task(attempt_request(secondary))
                        # Wait for first to complete
                        done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
                        d = next(iter(done))
                        try:
                            resp = d.result()
                            winner_node = candidate if d is t1 else secondary
                            # Cancel the other
                            for p in pending:
                                p.cancel()
                            # Ensure exceptions/cancellations are collected
                            try:
                                await asyncio.gather(*([t1, t2]), return_exceptions=True)
                            except Exception:
                                pass
                            # Mark winner metrics
                            try:
                                await redis_client.incrby(f"lb:hedge_wins:{model_name}|{winner_node}", 1)
                                await redis_client.sadd("lb:hedge_wins_models", model_name)
                                await redis_client.incrby(f"lb:hedge_wins_model:{model_name}", 1)
                            except Exception:
                                pass
                            # Headers and latency
                            resp.headers["x-attempts"] = str(attempts)
                            resp.headers["x-failover-count"] = str(max(0, attempts - 1))
                            resp.headers["x-retry-count"] = str(max(0, attempts - 1))
                            resp.headers["x-hedged"] = "true"
                            resp.headers["x-hedge-winner"] = winner_node or ""
                            try:
                                await _record_latency(model_name, winner_node or candidate, time.monotonic() - start_time)
                            except Exception:
                                pass
                            return resp
                        except Exception as e:
                            # First completed with error; wait for the other
                            other = next(iter(pending)) if pending else None
                            if other is not None:
                                try:
                                    resp = await other
                                    winner_node = secondary if other is t2 else candidate
                                    resp.headers["x-attempts"] = str(attempts)
                                    resp.headers["x-failover-count"] = str(max(0, attempts - 1))
                                    resp.headers["x-retry-count"] = str(max(0, attempts - 1))
                                    resp.headers["x-hedged"] = "true"
                                    resp.headers["x-hedge-winner"] = winner_node or ""
                                    try:
                                        await redis_client.incrby(f"lb:hedge_wins:{model_name}|{winner_node}", 1)
                                        await redis_client.sadd("lb:hedge_wins_models", model_name)
                                        await redis_client.incrby(f"lb:hedge_wins_model:{model_name}", 1)
                                    except Exception:
                                        pass
                                    try:
                                        await _record_latency(model_name, winner_node or candidate, time.monotonic() - start_time)
                                    except Exception:
                                        pass
                                    # Ensure exceptions/cancellations are collected
                                    try:
                                        await asyncio.gather(*([t1, t2]), return_exceptions=True)
                                    except Exception:
                                        pass
                                    return resp
                                except Exception as e2:
                                    # Both failed; fall through to normal failure handling
                                    await _record_failure(candidate, model_name)
                                    await _record_failure(secondary, model_name)
                                    last_error = e2
                                    all_capacity = False
                                    tried.add(secondary)
                                    tried_order.append(secondary)
                                    continue
                except (httpx.RequestError, httpx.HTTPStatusError, CapacityError, RateLimitError) as e:
                    # Primary threw before delay elapsed; treat as normal failure (no hedge launched)
                    await _record_failure(candidate, model_name)
                    last_error = e
                    if not isinstance(e, CapacityError):
                        all_capacity = False
                    continue

        # No hedging path or not applicable: normal single attempt
        try:
            await _inc_requests_total()
            resp = await attempt_request(candidate)
            resp.headers["x-attempts"] = str(attempts)
            resp.headers["x-failover-count"] = str(max(0, attempts - 1))
            resp.headers["x-retry-count"] = str(max(0, attempts - 1))
            resp.headers["x-hedged"] = "false"
            resp.headers["x-hedge-winner"] = ""
            try:
                await _record_latency(model_name, candidate, time.monotonic() - start_time)
            except Exception:
                pass
            return resp
        except (httpx.RequestError, httpx.HTTPStatusError, CapacityError, RateLimitError) as e:
            logger.warning("[req=%s] Chat (non-stream) upstream error from %s: %s (attempt %d/%d)", request_id, candidate, _describe_upstream_error(e), attempts, budget)
            await _record_failure(candidate, model_name)
            last_error = e
            if not isinstance(e, CapacityError):
                all_capacity = False
            continue
    if all_capacity:
        # Capacity exhausted for this model; attempt cross-model fallback if enabled
        fb = await _find_fallback_model(model_name)
        if fb:
            logger.warning("Capacity exhausted for model '%s'; attempting fallback '%s'", model_name, fb)
            body["model"] = fb
            # Re-evaluate eligible nodes for fallback model
            fb_nodes = await get_eligible_nodes(fb)
            if not fb_nodes:
                # No fallback capacity after all
                raise HTTPException(status_code=429, detail={"message": "All nodes are at capacity for this model.", "model": model_name, "attempts": attempts, "nodes": tried_order}, headers={"Retry-After": str(config.RETRY_AFTER_SECS), "x-request-id": request_id, "x-attempts": str(attempts), "x-failover-count": str(max(0, attempts-1)), "x-capacity-state": "cluster_saturated"})
            # Try a single attempt on fallback pool using same logic but without recursion complexity
            candidate = await router.select_node(fb_nodes, fb, redis_client)
            if candidate is None:
                raise HTTPException(status_code=429, detail={"message": "No eligible nodes for fallback model.", "model": fb}, headers={"Retry-After": str(config.RETRY_AFTER_SECS), "x-request-id": request_id})
            await _inc_requests_total()
            resp = await attempt_request(candidate)
            resp.headers["x-selected-model"] = fb
            resp.headers["x-fallback-model"] = fb
            resp.headers["x-retry-count"] = str(max(0, attempts - 1))
            return resp
        # No fallback configured or available
        raise HTTPException(status_code=429, detail={"message": "All nodes are at capacity for this model.", "model": model_name, "attempts": attempts, "nodes": tried_order}, headers={"Retry-After": str(config.RETRY_AFTER_SECS), "x-request-id": request_id, "x-attempts": str(attempts), "x-failover-count": str(max(0, attempts-1)), "x-capacity-state": "cluster_saturated"})
    # Failures for the model; attempt fallback if configured
    fb = await _find_fallback_model(model_name)
    if fb:
        logger.warning("All upstream nodes failed for model '%s'; attempting fallback '%s'", model_name, fb)
        body["model"] = fb
        fb_nodes = await get_eligible_nodes(fb)
        if fb_nodes:
            candidate = await router.select_node(fb_nodes, fb, redis_client)
            if candidate is not None:
                await _inc_requests_total()
                resp = await attempt_request(candidate)
                resp.headers["x-selected-model"] = fb
                resp.headers["x-fallback-model"] = fb
                resp.headers["x-retry-count"] = str(max(0, attempts - 1))
                return resp
    raise HTTPException(status_code=502, detail={"message": "All upstream nodes failed for chat model.", "model": model_name, "attempts": attempts, "nodes": tried_order}, headers={"x-request-id": request_id, "x-attempts": str(attempts), "x-failover-count": str(max(0, attempts-1))})

@app.api_route("/v1/responses", methods=["POST"])
async def responses_api(request: Request):
    """OpenAI Responses API -> Chat Completions translation layer.

    Accepts the Responses API format, converts to chat completions,
    forwards to our own /v1/chat/completions endpoint, and converts
    the response back to Responses API format.
    """
    try:
        raw = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.")

    # --- Build chat completions payload ---
    messages = []
    if "instructions" in raw:
        messages.append({"role": "system", "content": raw["instructions"]})

    inp = raw.get("input", "")
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                item_type = item.get("type", "")
                role = item.get("role", "")
                # Standard chat message with role/content
                if role and "content" in item:
                    content = item["content"]
                    # Content can be string or array of content parts
                    if isinstance(content, list):
                        text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("input_text", "output_text", "text")]
                        content = "\n".join(text_parts) if text_parts else json.dumps(content)
                    messages.append({"role": role, "content": content})
                # Responses API "message" type
                elif item_type == "message":
                    content = item.get("content", "")
                    if isinstance(content, list):
                        text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("input_text", "output_text", "text")]
                        content = "\n".join(text_parts) if text_parts else ""
                    messages.append({"role": item.get("role", "user"), "content": content})
                # Function call (assistant requested a tool call)
                elif item_type == "function_call":
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": item.get("call_id", item.get("id", f"call_{uuid.uuid4().hex[:8]}")),
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", "{}"),
                            }
                        }]
                    })
                # Function call output (tool result)
                elif item_type == "function_call_output":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": item.get("output", ""),
                    })
                else:
                    # Unknown type — try to pass as user message
                    text = item.get("text", item.get("content", json.dumps(item)))
                    if isinstance(text, list):
                        text = json.dumps(text)
                    messages.append({"role": "user", "content": str(text)})

    logger.info("responses_api: translated %d input items -> %d messages for model=%s, tools=%d, stream=%s",
                len(inp) if isinstance(inp, list) else 1, len(messages), raw.get("model", "?"),
                len(raw.get("tools", [])), raw.get("stream", "unset"))
    # Debug: log tool schemas on first request (small message count = new conversation)
    if len(messages) <= 3 and raw.get("tools"):
        for t in raw["tools"][:5]:  # first 5 tools only
            name = t.get("name", t.get("function", {}).get("name", "?"))
            params = t.get("parameters", t.get("function", {}).get("parameters", {}))
            logger.info("responses_api: tool schema: name=%s params=%s", name, json.dumps(params)[:200])

    cc_body: Dict[str, Any] = {
        "model": raw.get("model", "auto"),
        "messages": messages,
        "stream": False,  # Force non-streaming; convert later if needed
    }
    if "tools" in raw:
        # Convert tools from Responses API format to Chat Completions format.
        # Responses API:      {"type":"function", "name":"…", "parameters":{…}}
        # Chat Completions:   {"type":"function", "function":{"name":"…", "parameters":{…}}}
        cc_tools = []
        for tool in raw["tools"]:
            if "function" in tool:
                cc_tools.append(tool)  # already Chat Completions format
            else:
                cc_tools.append({
                    "type": tool.get("type", "function"),
                    "function": {
                        k: v for k, v in tool.items()
                        if k not in ("type",)
                    },
                })
        cc_body["tools"] = cc_tools
    if "tool_choice" in raw:
        cc_body["tool_choice"] = raw["tool_choice"]
    if "max_output_tokens" in raw:
        cc_body["max_tokens"] = raw["max_output_tokens"]
    if "temperature" in raw:
        cc_body["temperature"] = raw["temperature"]

    # --- Forward to our own chat completions endpoint ---
    server = request.scope.get("server")  # (host, port) tuple
    port = server[1] if server else 8002
    lb_url = f"http://127.0.0.1:{port}/v1/chat/completions"

    async with httpx.AsyncClient() as client:
        resp = await client.post(lb_url, json=cc_body, timeout=120.0)

    if resp.status_code != 200:
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    cc = resp.json()

    # --- Convert to Responses API format ---
    choice = cc.get("choices", [{}])[0]
    msg = choice.get("message", {})

    output = []

    # Tool calls
    for tc in msg.get("tool_calls", []):
        fn_name = tc.get("function", {}).get("name", "")
        fn_args = tc.get("function", {}).get("arguments", "{}")
        if not fn_name:
            logger.warning("responses_api: dropping tool call with empty function name (model=%s)",
                           raw.get("model", "?"))
            continue
        logger.info("responses_api: tool_call name=%s args=%s", fn_name, repr(fn_args)[:200])
        output.append({
            "type": "function_call",
            "id": f"fc_{uuid.uuid4().hex[:24]}",
            "call_id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
            "name": fn_name,
            "arguments": fn_args if isinstance(fn_args, str) else json.dumps(fn_args),
        })

    # Text content (some models put text in "reasoning" instead of "content")
    content = msg.get("content") or msg.get("reasoning") or ""
    if content:
        output.append({
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content}],
        })

    usage = cc.get("usage", {})

    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    created_at = cc.get("created", int(time.time()))
    model_name_out = cc.get("model", raw.get("model", "unknown"))
    usage_out = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }

    response_obj = {
        "id": resp_id,
        "object": "response",
        "created_at": created_at,
        "model": model_name_out,
        "output": output,
        "usage": usage_out,
    }

    # --- Non-streaming: return JSON as before ---
    if not raw.get("stream"):
        return JSONResponse(content=response_obj)

    # --- Streaming: emit Responses API SSE events ---
    async def responses_sse_stream():
        # response.created
        yield f"event: response.created\ndata: {json.dumps(response_obj)}\n\n"

        for idx, item in enumerate(output):
            item_id = item.get("id", f"item_{uuid.uuid4().hex[:16]}")

            # response.output_item.added
            yield f"event: response.output_item.added\ndata: {json.dumps({'type': 'response.output_item.added', 'response_id': resp_id, 'output_index': idx, 'item': item})}\n\n"

            # For function_call items, emit arguments delta + done events
            if item.get("type") == "function_call":
                args_str = item.get("arguments", "{}")
                if not isinstance(args_str, str):
                    args_str = json.dumps(args_str)
                # response.function_call_arguments.delta (single chunk with full args)
                yield f"event: response.function_call_arguments.delta\ndata: {json.dumps({'type': 'response.function_call_arguments.delta', 'response_id': resp_id, 'item_id': item_id, 'output_index': idx, 'delta': args_str})}\n\n"
                # response.function_call_arguments.done
                yield f"event: response.function_call_arguments.done\ndata: {json.dumps({'type': 'response.function_call_arguments.done', 'response_id': resp_id, 'item_id': item_id, 'output_index': idx, 'arguments': args_str})}\n\n"

            # For text message items, emit text delta + done events
            elif item.get("type") == "message":
                for ci, part in enumerate(item.get("content", [])):
                    if part.get("type") == "output_text":
                        text = part.get("text", "")
                        # response.output_text.delta (single chunk with full text)
                        yield f"event: response.output_text.delta\ndata: {json.dumps({'type': 'response.output_text.delta', 'response_id': resp_id, 'item_id': item_id, 'output_index': idx, 'content_index': ci, 'delta': text})}\n\n"
                        # response.output_text.done
                        yield f"event: response.output_text.done\ndata: {json.dumps({'type': 'response.output_text.done', 'response_id': resp_id, 'item_id': item_id, 'output_index': idx, 'content_index': ci, 'text': text})}\n\n"

            # response.output_item.done
            yield f"event: response.output_item.done\ndata: {json.dumps({'type': 'response.output_item.done', 'response_id': resp_id, 'output_index': idx, 'item': item})}\n\n"

        # response.completed
        yield f"event: response.completed\ndata: {json.dumps(response_obj)}\n\n"

    return StreamingResponse(responses_sse_stream(), media_type="text/event-stream")

@app.api_route("/v1/embeddings", methods=["POST"])
async def embeddings(request: Request):
    try:
        raw_body = await request.json()
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON payload for /v1/embeddings: %s", exc)
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.")
    except Exception as exc:
        logger.warning("Failed to parse JSON payload for /v1/embeddings: %s", exc)
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.")

    sanitized = sanitize_embeddings_request(raw_body)
    if sanitized.default_applied:
        logger.info(
            "Embeddings request missing model; applied default '%s'", sanitized.default_applied
        )
    if sanitized.missing_fields:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required fields.",
                "missing": sanitized.missing_fields,
            },
        )

    body = sanitized.payload
    model_name = sanitized.model_name
    model_defaulted = sanitized.default_applied is not None

    if _is_model_sentinel(model_name):
        prefer_intersection = request.query_params.get("require_all", "false").lower() in ("1", "true", "yes")
        resolved = await _resolve_auto_model(prefer_intersection=prefer_intersection)
        if not resolved:
            raise HTTPException(status_code=404, detail="No models available for auto selection.")
        body["model"] = resolved
        model_name = resolved
    eligible_nodes = await get_eligible_nodes_for_model(model_name)
    on_demand_wait = False
    warm_wait_ms = 0
    if not eligible_nodes:
        # Diagnose why before deciding whether to warm or fail fast
        diagnosis = await _diagnose_unroutable(model_name)
        if diagnosis["reason"] in ("circuit_open", "rate_limited"):
            retry_after = diagnosis["retry_after_secs"] or _cb_cooldown_secs()
            raise HTTPException(
                status_code=503,
                detail={
                    "message": f"Model '{model_name}' is temporarily unavailable: {diagnosis['reason'].replace('_', ' ')} on all nodes",
                    "code": "model_unavailable",
                    "x_llb_diagnosis": diagnosis,
                },
                headers={"Retry-After": str(retry_after), "x-request-id": request.headers.get("x-request-id", "")},
            )
        try:
            t0_warm = time.monotonic()
            waited = await _on_demand_wait_for_model(model_name)
            warm_wait_ms = int((time.monotonic() - t0_warm) * 1000)
            if waited:
                eligible_nodes = waited
                on_demand_wait = True
        except Exception:
            pass
    if not eligible_nodes:
        raise HTTPException(status_code=404, detail=f"No healthy nodes found for model '{model_name}'.")

    start_time = time.monotonic()
    _llb_hdrs = await _routing_headers(model_name, len(eligible_nodes))
    headers = {key: value for key, value in request.headers.items() if key.lower() not in ('host', 'content-length')}
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    headers["x-request-id"] = request_id
    session_id = request.headers.get("x-session-id")

    async def attempt_request(node: str):
        # Determine URL and headers based on backend type (cloud vs local)
        if _is_cloud_backend(node):
            url = _get_cloud_url(node, "/embeddings")
            if not url:
                raise httpx.RequestError(f"Cloud backend URL not configured for {node}")
            req_headers = _get_cloud_headers(node, headers)
        else:
            url = f"http://{node}/v1/embeddings"
            req_headers = headers

        logger.info("[req=%s] Routing embeddings request for model '%s' to %s", request_id, model_name, node)
        acquired = await _acquire_slot(node)
        if not acquired:
            logger.info("[req=%s] Node %s at capacity; skipping", request_id, node)
            raise CapacityError("node")
        model_ok = await _acquire_model_slot(model_name)
        if not model_ok:
            await _dec_inflight(node)
            raise CapacityError("model")
        try:
            try:
                resp = await http_client.post(url, json=body, headers=req_headers)
                status_code = resp.status_code
                content = resp.content
                ctype = resp.headers.get("content-type", "application/json")
            except AttributeError:
                # Compatibility path for fakes
                async with http_client.stream("POST", url, json=body, headers=req_headers) as response:
                    status_code = getattr(response, "status_code", 200)
                    # Handle rate limiting (429)
                    if status_code == 429:
                        retry_after = getattr(response, "headers", {}).get("retry-after")
                        retry_after_secs = float(retry_after) if retry_after else None
                        await _record_rate_limit(node, retry_after_secs)
                        raise RateLimitError(node, retry_after_secs)
                    if status_code and status_code >= 500:
                        await _record_failure(node, model_name)
                        raise httpx.HTTPStatusError("Upstream 5xx", request=None, response=None)
                    chunks = bytearray()
                    async for chunk in response.aiter_bytes():
                        chunks.extend(chunk)
                    content = bytes(chunks)
                    ctype = "application/json"
            # Handle rate limiting (429)
            if status_code == 429:
                retry_after = resp.headers.get("retry-after")
                retry_after_secs = float(retry_after) if retry_after else None
                await _record_rate_limit(node, retry_after_secs)
                raise RateLimitError(node, retry_after_secs)
            if status_code and status_code >= 500:
                await _record_failure(node, model_name)
                raise httpx.HTTPStatusError("Upstream 5xx", request=None, response=None)
            await _record_success(node, model_name)
            await _clear_rate_limit(node)
            out = Response(content=content, media_type=ctype, status_code=status_code)
            out.headers["x-selected-model"] = model_name
            out.headers["x-routed-node"] = node
            out.headers["x-request-id"] = request_id
            out.headers["x-on-demand-wait"] = "true" if on_demand_wait else "false"
            out.headers["x-warm-wait-ms"] = str(warm_wait_ms)
            out.headers["x-capacity-state"] = "ok"
            out.headers["x-model-defaulted"] = "true" if model_defaulted else "false"
            for hk, hv in _llb_hdrs.items():
                out.headers[hk] = hv
            await _set_sticky_node(session_id, model_name, node)
            return out
        finally:
            await _dec_inflight(node)
            await _dec_model(model_name)

    attempts = 0
    tried = set()
    last_error = None
    all_capacity = True
    budget = min(len(eligible_nodes), 1 + config.MAX_RETRIES)
    tried_order = []
    same_node_retries = 0
    if len(eligible_nodes) == 1 and config.SAME_NODE_RETRY_ENABLED:
        budget = 1 + config.SAME_NODE_RETRY_MAX
    while attempts < budget:
        remaining = [n for n in eligible_nodes if n not in tried]
        if not remaining and len(eligible_nodes) == 1 and config.SAME_NODE_RETRY_ENABLED and same_node_retries < config.SAME_NODE_RETRY_MAX:
            same_node_retries += 1
            logger.info("[req=%s] Same-node retry %d/%d for %s after %.1fs delay",
                        request_id, same_node_retries, config.SAME_NODE_RETRY_MAX, eligible_nodes[0], config.SAME_NODE_RETRY_DELAY_SECS)
            await asyncio.sleep(config.SAME_NODE_RETRY_DELAY_SECS)
            tried.clear()
            remaining = list(eligible_nodes)
        closed = []
        for n in remaining:
            if not await _is_circuit_open(n):
                closed.append(n)
        pool = closed if closed else remaining
        candidate = None
        if attempts == 0:
            sticky = await _get_sticky_node(session_id, model_name)
            if sticky and sticky in pool:
                candidate = sticky
        if candidate is None:
            candidate = await router.select_node(pool, model_name, redis_client)
        if candidate is None:
            break
        tried.add(candidate)
        tried_order.append(candidate)
        attempts += 1
        try:
            await _inc_requests_total()
            resp = await attempt_request(candidate)
            # On success, add attempt headers and record metrics
            resp.headers["x-attempts"] = str(attempts)
            resp.headers["x-failover-count"] = str(max(0, attempts - 1))
            resp.headers["x-retry-count"] = str(max(0, attempts - 1))
            resp.headers["x-hedged"] = "false"
            resp.headers["x-hedge-winner"] = ""
            # Failover counters
            if attempts > 1:
                try:
                    await redis_client.incrby("lb:failovers_total", attempts - 1)
                    await redis_client.incrby(f"lb:model:{model_name}:failovers_total", attempts - 1)
                except Exception:
                    pass
            # Latency histogram (non-stream)
            try:
                # For embeddings, we measure end-to-end as well
                elapsed = time.monotonic() - start_time
                await _record_latency(model_name, candidate, elapsed)
            except Exception:
                pass
            return resp
        except (httpx.RequestError, httpx.HTTPStatusError, CapacityError, RateLimitError) as e:
            logger.warning("[req=%s] Embeddings upstream error from %s: %s (attempt %d/%d)", request_id, candidate, _describe_upstream_error(e), attempts, budget)
            await _record_failure(candidate, model_name)
            last_error = e
            if not isinstance(e, CapacityError):
                all_capacity = False
            continue
    if all_capacity:
        raise HTTPException(status_code=429, detail={"message": "All nodes are at capacity for this model.", "model": model_name, "attempts": attempts, "nodes": tried_order}, headers={"Retry-After": str(config.RETRY_AFTER_SECS), "x-request-id": request_id, "x-attempts": str(attempts), "x-failover-count": str(max(0, attempts-1))})
    raise HTTPException(status_code=502, detail={"message": "All upstream nodes failed for embeddings model.", "model": model_name, "attempts": attempts, "nodes": tried_order}, headers={"x-request-id": request_id, "x-attempts": str(attempts), "x-failover-count": str(max(0, attempts-1))})

@app.get("/health", status_code=200)
async def health_check(request: Request, response: Response):
    """Checks the health of the load balancer and its node cluster.

    Query params:
      models=true  — include per-model availability breakdown.
    """
    healthy_node_count = await redis_client.scard("nodes:healthy")
    include_models = request.query_params.get("models", "false").lower() in ("1", "true", "yes")

    if healthy_node_count >= config.MIN_HEALTHY_NODES:
        status = "healthy"
    else:
        status = "unhealthy"

    result = {
        "status": status,
        "nodes_found": healthy_node_count,
        "minimum_required": config.MIN_HEALTHY_NODES,
    }

    if include_models:
        all_models = await _aggregate_models()
        models_status = {}
        any_unavailable = False
        for m in all_models:
            mid = m.get("id")
            if not mid:
                continue
            avail = await _model_availability(mid)
            models_status[mid] = {
                "status": avail["status"],
                "eligible_nodes": avail["eligible_nodes"],
                "total_nodes": avail["total_nodes"],
            }
            if avail["status"] == "unavailable" and avail["total_nodes"] > 0:
                any_unavailable = True
        result["models"] = models_status
        # Upgrade to degraded if nodes are healthy but some known models are unroutable
        if status == "healthy" and any_unavailable and getattr(config, "HEALTH_MODEL_AWARE", False):
            result["status"] = "degraded"

    if result["status"] == "unhealthy":
        response.status_code = 503

    return result

@app.get("/metrics")
async def metrics():
    """Prometheus-style metrics exposition without external deps."""
    P = f"{METRIC_PREFIX}_"
    lines = []
    lines.append(f"# HELP {P}requests_total Total requests handled by the LB")
    lines.append(f"# TYPE {P}requests_total counter")
    total = await redis_client.get("lb:requests_total")
    lines.append(f"{P}requests_total {int(total) if total else 0}")

    healthy = await redis_client.smembers("nodes:healthy")
    lines.append(f"# HELP {P}up Up status of nodes (1 healthy, 0 otherwise)")
    lines.append(f"# TYPE {P}up gauge")
    for n in healthy:
        lines.append(f'{P}up{{node="{n}"}} 1')

    lines.append(f"# HELP {P}inflight Current in-flight requests per node")
    lines.append(f"# TYPE {P}inflight gauge")
    for n in healthy:
        inflight = await redis_client.get(f"node:{n}:inflight")
        lines.append(f'{P}inflight{{node="{n}"}} {int(inflight) if inflight else 0}')

    lines.append(f"# HELP {P}failures Recent failure count per node")
    lines.append(f"# TYPE {P}failures gauge")
    for n in healthy:
        failures = await redis_client.get(f"node:{n}:failures")
        lines.append(f'{P}failures{{node="{n}"}} {int(failures) if failures else 0}')

    # Failovers total (overall and per model)
    lines.append(f"# HELP {P}failovers_total Total failovers across all requests")
    lines.append(f"# TYPE {P}failovers_total counter")
    total_failovers = await redis_client.get("lb:failovers_total")
    lines.append(f"{P}failovers_total {int(total_failovers) if total_failovers else 0}")
    # Per-model failovers
    # We don't track model list centrally; infer from keys stored
    # by scanning models in latency series and model failover keys indirectly.
    # Keep simple: scan latency series' models
    series = await redis_client.smembers("lb:latency_series")
    models = set()
    for s in series:
        try:
            m, _ = s.split("|", 1)
            models.add(m)
        except Exception:
            continue
    for m in models:
        mf = await redis_client.get(f"lb:model:{m}:failovers_total")
        if mf:
            lines.append(f'{P}failovers_total{{model="{m}"}} {int(mf)}')

    # Rate limit metrics
    lines.append(f"# HELP {P}rate_limits_total Total rate limit (429) responses")
    lines.append(f"# TYPE {P}rate_limits_total counter")
    rate_limits_total = await redis_client.get("lb:rate_limits_total")
    lines.append(f"{P}rate_limits_total {int(rate_limits_total) if rate_limits_total else 0}")

    # Per-node rate limits
    lines.append(f"# HELP {P}rate_limits Rate limit counts per node")
    lines.append(f"# TYPE {P}rate_limits counter")
    for n in healthy:
        rl = await redis_client.get(f"lb:rate_limits:{n}")
        if rl:
            lines.append(f'{P}rate_limits{{node="{n}"}} {int(rl)}')

    # Hedging metrics
    lines.append(f"# HELP {P}hedges_total Total hedged duplicate attempts")
    lines.append(f"# TYPE {P}hedges_total counter")
    hedges_total = await redis_client.get("lb:hedges_total")
    lines.append(f"{P}hedges_total {int(hedges_total) if hedges_total else 0}")

    lines.append(f"# HELP {P}hedge_wins Hedge wins per model/node")
    lines.append(f"# TYPE {P}hedge_wins counter")
    # Infer winners by scanning known latency series for keys
    if series:
        for s in series:
            try:
                m, n = s.split("|", 1)
            except Exception:
                continue
            w = await redis_client.get(f"lb:hedge_wins:{s}")
            if w:
                lines.append(f'{P}hedge_wins{{model="{m}",node="{n}"}} {int(w)}')

    # Latency histogram per model and node
    lines.append(f"# HELP {P}latency_seconds Request latency histogram per model/node")
    lines.append(f"# TYPE {P}latency_seconds histogram")
    # Iterate known series
    for s in series:
        try:
            m, n = s.split("|", 1)
        except Exception:
            continue
        cumulative = 0
        for le in _LAT_BUCKETS:
            val = await redis_client.get(f"lb:latency_bucket:{s}:{le}")
            v = int(val) if val else 0
            cumulative = v  # already stored as cumulative
            le_str = "+Inf" if le == float("inf") else ("%.2f" % le).rstrip('0').rstrip('.')
            lines.append(f'{P}latency_seconds_bucket{{model="{m}",node="{n}",le="{le_str}"}} {cumulative}')
        s_sum = await redis_client.get(f"lb:latency_sum:{s}")
        s_cnt = await redis_client.get(f"lb:latency_count:{s}")
        lines.append(f'{P}latency_seconds_sum{{model="{m}",node="{n}"}} {float(s_sum) if s_sum else 0.0}')
        lines.append(f'{P}latency_seconds_count{{model="{m}",node="{n}"}} {int(s_cnt) if s_cnt else 0}')

    # Stream TTFB histogram
    ttfb_series = await redis_client.smembers("lb:stream_ttfb_series")
    if ttfb_series:
        lines.append(f"# HELP {P}stream_ttfb_seconds Time-to-first-byte for streaming requests")
        lines.append(f"# TYPE {P}stream_ttfb_seconds histogram")
        for s in ttfb_series:
            try:
                m, n = s.split("|", 1)
            except Exception:
                continue
            for le in _LAT_BUCKETS:
                val = await redis_client.get(f"lb:stream_ttfb_bucket:{s}:{le}")
                v = int(val) if val else 0
                le_str = "+Inf" if le == float("inf") else ("%.2f" % le).rstrip('0').rstrip('.')
                lines.append(f'{P}stream_ttfb_seconds_bucket{{model="{m}",node="{n}",le="{le_str}"}} {v}')
            s_sum = await redis_client.get(f"lb:stream_ttfb_sum:{s}")
            s_cnt = await redis_client.get(f"lb:stream_ttfb_count:{s}")
            lines.append(f'{P}stream_ttfb_seconds_sum{{model="{m}",node="{n}"}} {float(s_sum) if s_sum else 0.0}')
            lines.append(f'{P}stream_ttfb_seconds_count{{model="{m}",node="{n}"}} {int(s_cnt) if s_cnt else 0}')

    # Stream duration histogram
    dur_series = await redis_client.smembers("lb:stream_duration_series")
    if dur_series:
        lines.append(f"# HELP {P}stream_duration_seconds Total stream duration for streaming requests")
        lines.append(f"# TYPE {P}stream_duration_seconds histogram")
        for s in dur_series:
            try:
                m, n = s.split("|", 1)
            except Exception:
                continue
            for le in _LAT_BUCKETS:
                val = await redis_client.get(f"lb:stream_duration_bucket:{s}:{le}")
                v = int(val) if val else 0
                le_str = "+Inf" if le == float("inf") else ("%.2f" % le).rstrip('0').rstrip('.')
                lines.append(f'{P}stream_duration_seconds_bucket{{model="{m}",node="{n}",le="{le_str}"}} {v}')
            s_sum = await redis_client.get(f"lb:stream_duration_sum:{s}")
            s_cnt = await redis_client.get(f"lb:stream_duration_count:{s}")
            lines.append(f'{P}stream_duration_seconds_sum{{model="{m}",node="{n}"}} {float(s_sum) if s_sum else 0.0}')
            lines.append(f'{P}stream_duration_seconds_count{{model="{m}",node="{n}"}} {int(s_cnt) if s_cnt else 0}')

    # Aggregate hedge wins per model
    try:
        models = await redis_client.smembers("lb:hedge_wins_models")
        lines.append(f"# HELP {P}hedge_wins_total Hedge wins per model")
        lines.append(f"# TYPE {P}hedge_wins_total counter")
        for m in models:
            val = await redis_client.get(f"lb:hedge_wins_model:{m}")
            v = int(val) if val else 0
            lines.append(f'{P}hedge_wins_total{{model="{m}"}} {v}')
    except Exception:
        pass

    # Multi-backend execution metrics
    lines.append(f"# HELP {P}multi_exec_total Total multi-backend execution requests by mode")
    lines.append(f"# TYPE {P}multi_exec_total counter")
    multi_total = await redis_client.get("lb:multi_exec_total")
    lines.append(f"{P}multi_exec_total {int(multi_total) if multi_total else 0}")
    for mode in ("race", "all", "sequence", "consensus", "plan"):
        val = await redis_client.get(f"lb:multi_exec_total:{mode}")
        if val:
            lines.append(f'{P}multi_exec_total{{mode="{mode}"}} {int(val)}')

    lines.append(f"# HELP {P}multi_exec_backends Backends attempted per multi-exec request by mode")
    lines.append(f"# TYPE {P}multi_exec_backends summary")
    for mode in ("race", "all", "sequence", "consensus", "plan"):
        s_sum = await redis_client.get(f"lb:multi_exec_backends_sum:{mode}")
        s_cnt = await redis_client.get(f"lb:multi_exec_backends_count:{mode}")
        if s_cnt and int(s_cnt) > 0:
            lines.append(f'{P}multi_exec_backends_sum{{mode="{mode}"}} {int(s_sum) if s_sum else 0}')
            lines.append(f'{P}multi_exec_backends_count{{mode="{mode}"}} {int(s_cnt)}')

    lines.append(f"# HELP {P}multi_exec_succeeded Backends that succeeded per multi-exec request by mode")
    lines.append(f"# TYPE {P}multi_exec_succeeded counter")
    for mode in ("race", "all", "sequence", "consensus", "plan"):
        val = await redis_client.get(f"lb:multi_exec_succeeded_sum:{mode}")
        if val:
            lines.append(f'{P}multi_exec_succeeded{{mode="{mode}"}} {int(val)}')

    # Consensus-specific metrics
    lines.append(f"# HELP {P}consensus_total Total consensus requests")
    lines.append(f"# TYPE {P}consensus_total counter")
    consensus_total = await redis_client.get("lb:consensus_total")
    lines.append(f"{P}consensus_total {int(consensus_total) if consensus_total else 0}")

    # Per-model consensus totals
    consensus_models = await redis_client.smembers("lb:consensus_models")
    for model in consensus_models:
        val = await redis_client.get(f"lb:consensus_total:{model}")
        if val:
            lines.append(f'{P}consensus_total{{model="{model}"}} {int(val)}')

    # Consensus agreements/disagreements
    lines.append(f"# HELP {P}consensus_agreements Total consensus agreements (unanimous)")
    lines.append(f"# TYPE {P}consensus_agreements counter")
    agreements = await redis_client.get("lb:consensus_agreements")
    lines.append(f"{P}consensus_agreements {int(agreements) if agreements else 0}")

    lines.append(f"# HELP {P}consensus_disagreements Total consensus disagreements (not unanimous)")
    lines.append(f"# TYPE {P}consensus_disagreements counter")
    disagreements = await redis_client.get("lb:consensus_disagreements")
    lines.append(f"{P}consensus_disagreements {int(disagreements) if disagreements else 0}")

    # Per-model agreements/disagreements
    for model in consensus_models:
        agree_val = await redis_client.get(f"lb:consensus_agreements:{model}")
        if agree_val:
            lines.append(f'{P}consensus_agreements{{model="{model}"}} {int(agree_val)}')
        disagree_val = await redis_client.get(f"lb:consensus_disagreements:{model}")
        if disagree_val:
            lines.append(f'{P}consensus_disagreements{{model="{model}"}} {int(disagree_val)}')

    # Agreement count distribution (histogram of how many backends agreed)
    lines.append(f"# HELP {P}consensus_agreement_count Agreement count distribution")
    lines.append(f"# TYPE {P}consensus_agreement_count counter")
    for count in (0, 1, 2, 3):
        val = await redis_client.get(f"lb:consensus_agreement:{count}")
        if val:
            lines.append(f'{P}consensus_agreement_count{{count="{count}"}} {int(val)}')

    # Per-model agreement counts
    for model in consensus_models:
        for count in (0, 1, 2, 3):
            val = await redis_client.get(f"lb:consensus_agreement:{model}:{count}")
            if val:
                lines.append(f'{P}consensus_agreement_count{{model="{model}",count="{count}"}} {int(val)}')

    # Comparison type distribution
    lines.append(f"# HELP {P}consensus_comparison_type Comparison type used for consensus")
    lines.append(f"# TYPE {P}consensus_comparison_type counter")
    for comp_type in ("hash", "text", "tool_calls", "single", "none"):
        val = await redis_client.get(f"lb:consensus_comparison:{comp_type}")
        if val:
            lines.append(f'{P}consensus_comparison_type{{type="{comp_type}"}} {int(val)}')

    # Per-model comparison types
    for model in consensus_models:
        for comp_type in ("hash", "text", "tool_calls", "single", "none"):
            val = await redis_client.get(f"lb:consensus_comparison:{model}:{comp_type}")
            if val:
                lines.append(f'{P}consensus_comparison_type{{model="{model}",type="{comp_type}"}} {int(val)}')

    content = "\n".join(lines) + "\n"
    return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")

@app.post("/v1/admin/prefs")
async def set_prefs(request: Request):
    """Hot update preferences: preferred models, weights, caps.
    Body keys (all optional):
      - preferred_models: [str]
      - model_weights: {model_id: float}
      - model_caps: {model_id: int}
      - node_caps: {"host:port": int}
      - auto_model_strategy: "any_first" | "intersection_first"
    """
    body = await request.json()
    applied = {}
    pm = body.get("preferred_models")
    if isinstance(pm, list):
        config.PREFERRED_MODELS[:] = [str(x) for x in pm]
        try:
            await redis_client.set("lb:prefs:preferred_models", json.dumps(config.PREFERRED_MODELS))
        except Exception:
            pass
        applied["preferred_models"] = config.PREFERRED_MODELS
    mw = body.get("model_weights")
    if isinstance(mw, dict):
        try:
            for k, v in mw.items():
                config.MODEL_WEIGHTS[str(k)] = float(v)
        except Exception:
            pass
        try:
            await redis_client.set("lb:model_weights", json.dumps({k: float(v) for k, v in mw.items()}))
        except Exception:
            pass
        applied["model_weights"] = {k: float(v) for k, v in mw.items()}
    mc = body.get("model_caps")
    if isinstance(mc, dict):
        for mid, cap in mc.items():
            try:
                await redis_client.set(f"model:{mid}:maxconn", int(cap))
            except Exception:
                continue
        applied["model_caps"] = {k: int(v) for k, v in mc.items()}
    nc = body.get("node_caps")
    if isinstance(nc, dict):
        for node, cap in nc.items():
            try:
                await redis_client.set(f"node:{node}:maxconn", int(cap))
            except Exception:
                continue
        applied["node_caps"] = {k: int(v) for k, v in nc.items()}
    strat = body.get("auto_model_strategy")
    if isinstance(strat, str) and strat.lower() in ("any_first", "intersection_first"):
        config.AUTO_MODEL_STRATEGY = strat
        applied["auto_model_strategy"] = strat
    bcp = body.get("backend_cost_per_token")
    if isinstance(bcp, dict):
        try:
            config.BACKEND_COST_PER_TOKEN = {str(k): v for k, v in bcp.items()}
            applied["backend_cost_per_token"] = config.BACKEND_COST_PER_TOKEN
        except Exception:
            pass
    return {"ok": True, "applied": applied}


@app.post("/v1/admin/reset_histograms")
async def reset_histograms(request: Request):
    """Admin: reset/delete histogram and series metrics for a given model.

    Body (JSON):
      - model: str (required)
      - nodes: [str] (optional). If omitted, autodetect nodes from existing series sets.
      - include: ["latency", "stream_ttfb", "stream_duration"] (optional). Default: all three.
      - dry_run: bool (optional, default false). If true, return keys that would be deleted without mutating.

    Returns JSON with counts and details.
    """
    body = await request.json()
    model = (body.get("model") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="Field 'model' is required.")

    nodes = body.get("nodes")
    include = body.get("include") or ["latency", "stream_ttfb", "stream_duration"]
    dry_run = bool(body.get("dry_run", False))

    # Autodetect nodes from series sets if not provided
    if not nodes:
        nodes_detected = set()
        try:
            lat_series = await redis_client.smembers("lb:latency_series")
            for s in lat_series:
                try:
                    m, n = s.split("|", 1)
                    if m == model:
                        nodes_detected.add(n)
                except Exception:
                    continue
        except Exception:
            pass
        try:
            ttfb_series = await redis_client.smembers("lb:stream_ttfb_series")
            for s in ttfb_series:
                try:
                    m, n = s.split("|", 1)
                    if m == model:
                        nodes_detected.add(n)
                except Exception:
                    continue
        except Exception:
            pass
        try:
            dur_series = await redis_client.smembers("lb:stream_duration_series")
            for s in dur_series:
                try:
                    m, n = s.split("|", 1)
                    if m == model:
                        nodes_detected.add(n)
                except Exception:
                    continue
        except Exception:
            pass
        nodes = sorted(nodes_detected)

    # Build key lists to delete per node
    deleted = 0
    keys_preview = []
    series_removed = 0
    details = []

    async def _collect_keys(series_key: str) -> list[str]:
        out = []
        if "latency" in include:
            out.append(f"lb:latency_sum:{series_key}")
            out.append(f"lb:latency_count:{series_key}")
            try:
                bucket_keys = await redis_client.keys(f"lb:latency_bucket:{series_key}:*")
                out.extend(bucket_keys or [])
            except Exception:
                pass
        if "stream_ttfb" in include:
            out.append(f"lb:stream_ttfb_sum:{series_key}")
            out.append(f"lb:stream_ttfb_count:{series_key}")
            try:
                ttfb_keys = await redis_client.keys(f"lb:stream_ttfb_bucket:{series_key}:*")
                out.extend(ttfb_keys or [])
            except Exception:
                pass
        if "stream_duration" in include:
            out.append(f"lb:stream_duration_sum:{series_key}")
            out.append(f"lb:stream_duration_count:{series_key}")
            try:
                dur_keys = await redis_client.keys(f"lb:stream_duration_bucket:{series_key}:*")
                out.extend(dur_keys or [])
            except Exception:
                pass
        return out

    for n in nodes or []:
        s = f"{model}|{n}"
        keys = await _collect_keys(s)
        keys_preview.extend(keys)
        if not dry_run:
            try:
                if "latency" in include:
                    try:
                        removed = await redis_client.srem("lb:latency_series", s)
                        series_removed += int(removed or 0)
                    except Exception:
                        pass
                if "stream_ttfb" in include:
                    try:
                        removed = await redis_client.srem("lb:stream_ttfb_series", s)
                        series_removed += int(removed or 0)
                    except Exception:
                        pass
                if "stream_duration" in include:
                    try:
                        removed = await redis_client.srem("lb:stream_duration_series", s)
                        series_removed += int(removed or 0)
                    except Exception:
                        pass
                # Delete keys in batches
                for i in range(0, len(keys), 256):
                    chunk = keys[i:i+256]
                    if chunk:
                        deleted += await redis_client.delete(*chunk)
                details.append({"node": n, "series": s, "keys_deleted": len(keys)})
            except Exception as e:
                details.append({"node": n, "series": s, "error": str(e)})
        else:
            details.append({"node": n, "series": s, "keys_preview": len(keys)})

    return {
        "ok": True,
        "model": model,
        "nodes": nodes or [],
        "include": include,
        "dry_run": dry_run,
        "keys_preview": len(keys_preview) if dry_run else None,
        "keys_deleted": deleted if not dry_run else None,
        "series_removed": series_removed if not dry_run else None,
        "details": details,
    }
