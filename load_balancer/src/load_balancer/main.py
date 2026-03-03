import httpx
import redis.asyncio as redis
import json
import logging
import time
from typing import List, Optional
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uuid
import asyncio
from contextlib import asynccontextmanager

from . import config
from .request_validation import sanitize_chat_request, sanitize_embeddings_request
from .routing.strategies import get_routing_strategy
from .execution.modes import (
    execute_plan_stream,
    collect_plan_result,
    plan_result_to_openai_response,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, http_client, router
    logger.info("Load balancer starting up...")
    if redis_client is None:
        redis_client = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, decode_responses=True)
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


app = FastAPI(title="AI Load Balancer", lifespan=lifespan)
redis_client = None
http_client = None
router = None
logger = logging.getLogger("ai_lb")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


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
                lower = buckets[i-1] if i > 0 else 0
                upper = buckets[i]
                prev = cumulative_counts[i-1] if i > 0 else 0
                rng = max(1, count - prev)
                pos = (p95_target - prev) / rng
                return lower + pos * (upper - lower) if upper != float("inf") else 10.0
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

async def get_eligible_nodes(model_name: str):
    """Find healthy nodes that advertise the model and meet eligibility: not tripped and under p95 threshold."""
    healthy_nodes = sorted(await redis_client.smembers("nodes:healthy"))
    eligible_nodes = []
    for node in healthy_nodes:
        # Skip nodes on open circuit for this period
        if await _is_circuit_open(node):
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
            # Enforce p95 threshold if configured (>0)
            try:
                max_p95 = float(getattr(config, "MAX_P95_LATENCY_SECS", 0) or 0)
            except Exception:
                max_p95 = 0.0
            if max_p95 > 0:
                # Require a minimum number of samples before enforcing p95 threshold
                cnt = await _series_count(model_name, node)
                if cnt >= getattr(config, "ELIGIBILITY_MIN_P95_SAMPLES", 20):
                    p95 = await _series_p95(model_name, node)
                    if p95 and p95 > max_p95:
                        # Defer: skip node until it recovers
                        continue
            eligible_nodes.append(node)
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
        nodes = await get_eligible_nodes(mid)
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
        item = {"node": node, "has_model": False, "cb_open": False, "p95": None, "skipped": False, "reason": ""}
        try:
            models_json = await redis_client.get(f"node:{node}:models")
            models = json.loads(models_json).get("data", []) if models_json else []
            has = any(m.get("id") == model for m in models)
            item["has_model"] = has
            item["cb_open"] = await _is_circuit_open(node)
            p95 = await _series_p95(model, node)
            item["p95"] = p95
            # Apply same logic as get_eligible_nodes
            if not has:
                item["skipped"] = True
                item["reason"] = "model_missing"
            elif item["cb_open"]:
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
    eligible = [d["node"] for d in details if not d["skipped"] and d["has_model"] and not d["cb_open"]]
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

class CapacityError(Exception):
    def __init__(self, scope: str = "node"):
        self.scope = scope
        super().__init__(f"{scope} at capacity")

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

async def _record_failure(node: str):
    try:
        key = f"node:{node}:failures"
        failures = await redis_client.incrby(key, 1)
        await redis_client.expire(key, _cb_cooldown_secs())
        if failures >= config.CIRCUIT_BREAKER_THRESHOLD:
            await redis_client.set(f"node:{node}:cb_open", "1")
            await redis_client.expire(f"node:{node}:cb_open", _cb_cooldown_secs())
    except Exception:
        pass

async def _record_success(node: str):
    try:
        await redis_client.set(f"node:{node}:failures", 0)
        await redis_client.expire(f"node:{node}:failures", _cb_cooldown_secs())
        # closing circuit simply by letting cb_open expire; do nothing here
    except Exception:
        pass

async def _is_circuit_open(node: str) -> bool:
    try:
        val = await redis_client.get(f"node:{node}:cb_open")
        return val is not None and val != "0"
    except Exception:
        return False

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
                nodes = await get_eligible_nodes(mid)
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

@app.get("/v1/models")
async def get_all_models():
    """Aggregates and de-duplicates model lists from all healthy nodes."""
    healthy_nodes = await redis_client.smembers("nodes:healthy")
    all_models = {}
    for node in healthy_nodes:
        models_json = await redis_client.get(f"node:{node}:models")
        if models_json:
            models = json.loads(models_json).get("data", [])
            for model in models:
                if model['id'] not in all_models:
                    all_models[model['id']] = model
    return {"object": "list", "data": list(all_models.values())}

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
        # Always increment +Inf bucket explicitly (le == inf)
        await redis_client.incrby(f"lb:latency_bucket:{series_key}:{float('inf')}", 1)
    except Exception:
        pass

async def _record_stream_ttfb(model: str, node: str, elapsed_secs: float):
    try:
        series_key = f"{model}|{node}"
        await redis_client.sadd("lb:stream_ttfb_series", series_key)
        await redis_client.incrbyfloat(f"lb:stream_ttfb_sum:{series_key}", float(elapsed_secs))
        await redis_client.incrby(f"lb:stream_ttfb_count:{series_key}", 1)
        for le in _LAT_BUCKETS:
            if elapsed_secs <= le:
                await redis_client.incrby(f"lb:stream_ttfb_bucket:{series_key}:{le}", 1)
        await redis_client.incrby(f"lb:stream_ttfb_bucket:{series_key}:{float('inf')}", 1)
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
        await redis_client.incrby(f"lb:stream_duration_bucket:{series_key}:{float('inf')}", 1)
    except Exception:
        pass

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

    # Resolve auto/default sentinel to a concrete model id
    if _is_model_sentinel(model_name):
        prefer_intersection = request.query_params.get("require_all", "false").lower() in ("1", "true", "yes")
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
        # On-demand warm/wait: try to load the model on a few healthy nodes, then poll briefly
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
    if not eligible_nodes:
        raise HTTPException(status_code=404, detail=f"No healthy nodes found for model '{model_name}'.")

    headers = {key: value for key, value in request.headers.items() if key.lower() not in ('host', 'content-length')}
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    headers["x-request-id"] = request_id
    forced_node = request.query_params.get("node")
    session_id = request.headers.get("x-session-id")

    # PLAN execution mode: decompose → dispatch → assemble with SSE passthrough
    exec_mode = request.headers.get("x-execution-mode", "").strip().lower()
    if exec_mode == "plan":
        planner_backend = eligible_nodes[0]
        accept = request.headers.get("accept", "")

        async def _plan_call_backend(node: str, messages: list) -> str:
            url = f"http://{node}/v1/chat/completions"
            req_body = {"model": model_name, "messages": messages, "stream": False}
            resp = await http_client.post(url, json=req_body, headers={"x-request-id": request_id})
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

        async def _plan_stream_backend(node: str, messages: list):
            async for chunk in stream_backend(node, messages, model_name, request_id):
                yield chunk

        if "text/event-stream" in accept:
            async def plan_sse_generator():
                async for event in execute_plan_stream(
                    messages=body.get("messages", []),
                    call_backend=_plan_call_backend,
                    stream_backend=_plan_stream_backend,
                    planner_backend=planner_backend,
                    capability_nodes={},
                    default_nodes=list(eligible_nodes),
                ):
                    if event.event_type == "token":
                        yield f"data: {event.data['chunk']}\n\n".encode()
                    elif event.event_type == "error":
                        yield f"event: error\ndata: {json.dumps(event.data)}\n\n".encode()
                    else:
                        yield f"event: {event.event_type}\ndata: {json.dumps(event.data)}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                plan_sse_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "x-request-id": request_id,
                },
            )
        else:
            result = await collect_plan_result(
                execute_plan_stream(
                    messages=body.get("messages", []),
                    call_backend=_plan_call_backend,
                    stream_backend=_plan_stream_backend,
                    planner_backend=planner_backend,
                    capability_nodes={},
                    default_nodes=list(eligible_nodes),
                )
            )
            return JSONResponse(plan_result_to_openai_response(result))

    async def attempt_stream(node: str):
        url = f"http://{node}/v1/chat/completions"
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
            _beta = float(getattr(config, "P2C_BETA", 0.0))
            _last_usage_tokens = 0
            async with http_client.stream("POST", url, json=body, headers=headers) as response:
                # Treat 5xx and 404 as retryable failure across nodes
                if response.status_code and (response.status_code >= 500 or response.status_code == 404):
                    await _record_failure(node)
                    raise httpx.HTTPStatusError("Upstream retryable error", request=response.request, response=response)
                async for chunk in response.aiter_bytes():
                    if _beta > 0:
                        try:
                            text = chunk.decode("utf-8", errors="ignore")
                            for line in text.split("\n"):
                                if line.startswith("data: ") and "[DONE]" not in line:
                                    payload = json.loads(line[6:])
                                    ct = int((payload.get("usage") or {}).get("completion_tokens", 0))
                                    if ct > 0:
                                        _last_usage_tokens = ct
                        except Exception:
                            pass
                    yield chunk
            await _record_success(node)
            if _beta > 0 and _last_usage_tokens > 0:
                asyncio.create_task(_record_output_tokens(model_name, node, _last_usage_tokens))
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
        # We allow at most first try + MAX_RETRIES additional
        budget = min(len(nodes), 1 + config.MAX_RETRIES)
        while attempts < budget:
            # prefer nodes without open circuit
            remaining = [n for n in nodes if n not in tried]
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
                    except (asyncio.TimeoutError, StopAsyncIteration, httpx.RequestError, httpx.HTTPStatusError, CapacityError):
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
            except (httpx.RequestError, httpx.HTTPStatusError, CapacityError) as e:
                logger.warning("[req=%s] Upstream error from %s: %s (attempt %d/%d)", request_id, candidate, e, attempts, budget)
                await _record_failure(candidate)
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
        # For streaming, we cannot change status mid-stream; yield error body
        yield json.dumps(err_msg).encode()

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
                except (httpx.RequestError, httpx.HTTPStatusError, CapacityError) as e:
                    err = {"error": {"message": f"Upstream error from {forced_node}: {e}"}}
                    yield json.dumps(err).encode()
            return StreamingResponse(stream_single(), media_type="text/event-stream", headers={
                "x-selected-model": model_name,
                "x-routed-node": forced_node,
                "x-request-id": request_id,
                "x-on-demand-wait": "true" if on_demand_wait else "false",
                "x-warm-wait-ms": str(warm_wait_ms),
                "x-capacity-state": "ok",
                "x-model-defaulted": "true" if model_defaulted else "false",
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
            })

    # Non-streaming behavior: aggregate JSON and return a Response
    async def attempt_request(node: str):
        url = f"http://{node}/v1/chat/completions"
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
                resp = await http_client.post(url, json=body, headers=headers)
                status_code = resp.status_code
                content = resp.content
                ctype = resp.headers.get("content-type", "application/json")
            except AttributeError:
                # Compatibility for test fakes that implement only `.stream(...)`
                async with http_client.stream("POST", url, json=body, headers=headers) as response:
                    status_code = getattr(response, "status_code", 200)
                    # Treat 5xx and 404 as retryable across nodes (LM Studio variants sometimes 404 chat endpoint)
                    if status_code and (status_code >= 500 or status_code == 404):
                        await _record_failure(node)
                        raise httpx.HTTPStatusError("Upstream retryable error", request=None, response=None)
                    chunks = bytearray()
                    async for chunk in response.aiter_bytes():
                        chunks.extend(chunk)
                    content = bytes(chunks)
                    ctype = "application/json"
            # Evaluate retry conditions for normal client path
            if status_code and (status_code >= 500 or status_code == 404):
                await _record_failure(node)
                raise httpx.HTTPStatusError("Upstream retryable error", request=None, response=None)
            await _record_success(node)
            # Token accounting for cost-aware routing (fire-and-forget)
            if float(getattr(config, "P2C_BETA", 0.0)) > 0:
                try:
                    resp_data = json.loads(content)
                    ct = int((resp_data.get("usage") or {}).get("completion_tokens", 0))
                    if ct > 0:
                        asyncio.create_task(_record_output_tokens(model_name, node, ct))
                except Exception as _te:
                    logger.debug("[req=%s] cost-routing: token extraction failed for %s: %s", request_id, node, _te)
            out = Response(content=content, media_type=ctype, status_code=status_code)
            out.headers["x-selected-model"] = model_name
            out.headers["x-routed-node"] = node
            out.headers["x-request-id"] = request_id
            out.headers["x-on-demand-wait"] = "true" if on_demand_wait else "false"
            out.headers["x-warm-wait-ms"] = str(warm_wait_ms)
            out.headers["x-capacity-state"] = "ok"
            out.headers["x-model-defaulted"] = "true" if model_defaulted else "false"
            # Success: set sticky mapping
            await _set_sticky_node(session_id, model_name, node)
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
        except (httpx.RequestError, httpx.HTTPStatusError, CapacityError) as e:
            raise HTTPException(status_code=502, detail=f"Upstream error from {forced_node}: {e}")

    # Failover loop (non-stream)
    attempts = 0
    tried = set()
    last_error = None
    budget = min(len(eligible_nodes), 1 + config.MAX_RETRIES)
    all_capacity = True
    tried_order = []
    while attempts < budget:
        remaining = [n for n in eligible_nodes if n not in tried]
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
                    except (httpx.RequestError, httpx.HTTPStatusError, CapacityError):
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
                                    await _record_failure(candidate)
                                    await _record_failure(secondary)
                                    last_error = e2
                                    all_capacity = False
                                    tried.add(secondary)
                                    tried_order.append(secondary)
                                    continue
                except (httpx.RequestError, httpx.HTTPStatusError, CapacityError) as e:
                    # Primary threw before delay elapsed; treat as normal failure (no hedge launched)
                    await _record_failure(candidate)
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
        except (httpx.RequestError, httpx.HTTPStatusError, CapacityError) as e:
            logger.warning("[req=%s] Chat (non-stream) upstream error from %s: %s (attempt %d/%d)", request_id, candidate, e, attempts, budget)
            await _record_failure(candidate)
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

    headers = {key: value for key, value in request.headers.items() if key.lower() not in ('host', 'content-length')}
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    headers["x-request-id"] = request_id
    session_id = request.headers.get("x-session-id")

    async def attempt_request(node: str):
        url = f"http://{node}/v1/embeddings"
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
                resp = await http_client.post(url, json=body, headers=headers)
                status_code = resp.status_code
                content = resp.content
                ctype = resp.headers.get("content-type", "application/json")
            except AttributeError:
                # Compatibility path for fakes
                async with http_client.stream("POST", url, json=body, headers=headers) as response:
                    status_code = getattr(response, "status_code", 200)
                    if status_code and status_code >= 500:
                        await _record_failure(node)
                        raise httpx.HTTPStatusError("Upstream 5xx", request=None, response=None)
                    chunks = bytearray()
                    async for chunk in response.aiter_bytes():
                        chunks.extend(chunk)
                    content = bytes(chunks)
                    ctype = "application/json"
            if status_code and status_code >= 500:
                await _record_failure(node)
                raise httpx.HTTPStatusError("Upstream 5xx", request=None, response=None)
            await _record_success(node)
            out = Response(content=content, media_type=ctype, status_code=status_code)
            out.headers["x-selected-model"] = model_name
            out.headers["x-routed-node"] = node
            out.headers["x-request-id"] = request_id
            out.headers["x-on-demand-wait"] = "true" if on_demand_wait else "false"
            out.headers["x-warm-wait-ms"] = str(warm_wait_ms)
            out.headers["x-capacity-state"] = "ok"
            out.headers["x-model-defaulted"] = "true" if model_defaulted else "false"
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
    while attempts < budget:
        remaining = [n for n in eligible_nodes if n not in tried]
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
        except (httpx.RequestError, httpx.HTTPStatusError, CapacityError) as e:
            logger.warning("[req=%s] Embeddings upstream error from %s: %s (attempt %d/%d)", request_id, candidate, e, attempts, budget)
            await _record_failure(candidate)
            last_error = e
            if not isinstance(e, CapacityError):
                all_capacity = False
            continue
    if all_capacity:
        raise HTTPException(status_code=429, detail={"message": "All nodes are at capacity for this model.", "model": model_name, "attempts": attempts, "nodes": tried_order}, headers={"Retry-After": str(config.RETRY_AFTER_SECS), "x-request-id": request_id, "x-attempts": str(attempts), "x-failover-count": str(max(0, attempts-1))})
    raise HTTPException(status_code=502, detail={"message": "All upstream nodes failed for embeddings model.", "model": model_name, "attempts": attempts, "nodes": tried_order}, headers={"x-request-id": request_id, "x-attempts": str(attempts), "x-failover-count": str(max(0, attempts-1))})

@app.get("/health", status_code=200)
async def health_check(response: Response):
    """Checks the health of the load balancer and its node cluster."""
    healthy_node_count = await redis_client.scard("nodes:healthy")
    
    if healthy_node_count >= config.MIN_HEALTHY_NODES:
        return {
            "status": "healthy",
            "nodes_found": healthy_node_count,
            "minimum_required": config.MIN_HEALTHY_NODES
        }
    else:
        response.status_code = 503
        return {
            "status": "unhealthy",
            "nodes_found": healthy_node_count,
            "minimum_required": config.MIN_HEALTHY_NODES
        }

@app.get("/metrics")
async def metrics():
    """Prometheus-style metrics exposition without external deps."""
    lines = []
    lines.append("# HELP ai_lb_requests_total Total requests handled by the LB")
    lines.append("# TYPE ai_lb_requests_total counter")
    total = await redis_client.get("lb:requests_total")
    lines.append(f"ai_lb_requests_total {int(total) if total else 0}")

    healthy = await redis_client.smembers("nodes:healthy")
    lines.append("# HELP ai_lb_up Up status of nodes (1 healthy, 0 otherwise)")
    lines.append("# TYPE ai_lb_up gauge")
    for n in healthy:
        lines.append(f'ai_lb_up{{node="{n}"}} 1')

    lines.append("# HELP ai_lb_inflight Current in-flight requests per node")
    lines.append("# TYPE ai_lb_inflight gauge")
    for n in healthy:
        inflight = await redis_client.get(f"node:{n}:inflight")
        lines.append(f'ai_lb_inflight{{node="{n}"}} {int(inflight) if inflight else 0}')

    lines.append("# HELP ai_lb_failures Recent failure count per node")
    lines.append("# TYPE ai_lb_failures gauge")
    for n in healthy:
        failures = await redis_client.get(f"node:{n}:failures")
        lines.append(f'ai_lb_failures{{node="{n}"}} {int(failures) if failures else 0}')

    # Failovers total (overall and per model)
    lines.append("# HELP ai_lb_failovers_total Total failovers across all requests")
    lines.append("# TYPE ai_lb_failovers_total counter")
    total_failovers = await redis_client.get("lb:failovers_total")
    lines.append(f"ai_lb_failovers_total {int(total_failovers) if total_failovers else 0}")
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
            lines.append(f'ai_lb_failovers_total{{model="{m}"}} {int(mf)}')

    # Hedging metrics
    lines.append("# HELP ai_lb_hedges_total Total hedged duplicate attempts")
    lines.append("# TYPE ai_lb_hedges_total counter")
    hedges_total = await redis_client.get("lb:hedges_total")
    lines.append(f"ai_lb_hedges_total {int(hedges_total) if hedges_total else 0}")

    lines.append("# HELP ai_lb_hedge_wins Hedge wins per model/node")
    lines.append("# TYPE ai_lb_hedge_wins counter")
    # Infer winners by scanning known latency series for keys
    if series:
        for s in series:
            try:
                m, n = s.split("|", 1)
            except Exception:
                continue
            w = await redis_client.get(f"lb:hedge_wins:{s}")
            if w:
                lines.append(f'ai_lb_hedge_wins{{model="{m}",node="{n}"}} {int(w)}')

    # Latency histogram per model and node
    lines.append("# HELP ai_lb_latency_seconds Request latency histogram per model/node")
    lines.append("# TYPE ai_lb_latency_seconds histogram")
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
            lines.append(f'ai_lb_latency_seconds_bucket{{model="{m}",node="{n}",le="{le_str}"}} {cumulative}')
        s_sum = await redis_client.get(f"lb:latency_sum:{s}")
        s_cnt = await redis_client.get(f"lb:latency_count:{s}")
        lines.append(f'ai_lb_latency_seconds_sum{{model="{m}",node="{n}"}} {float(s_sum) if s_sum else 0.0}')
        lines.append(f'ai_lb_latency_seconds_count{{model="{m}",node="{n}"}} {int(s_cnt) if s_cnt else 0}')

    # Stream TTFB histogram
    ttfb_series = await redis_client.smembers("lb:stream_ttfb_series")
    if ttfb_series:
        lines.append("# HELP ai_lb_stream_ttfb_seconds Time-to-first-byte for streaming requests")
        lines.append("# TYPE ai_lb_stream_ttfb_seconds histogram")
        for s in ttfb_series:
            try:
                m, n = s.split("|", 1)
            except Exception:
                continue
            for le in _LAT_BUCKETS:
                val = await redis_client.get(f"lb:stream_ttfb_bucket:{s}:{le}")
                v = int(val) if val else 0
                le_str = "+Inf" if le == float("inf") else ("%.2f" % le).rstrip('0').rstrip('.')
                lines.append(f'ai_lb_stream_ttfb_seconds_bucket{{model="{m}",node="{n}",le="{le_str}"}} {v}')
            s_sum = await redis_client.get(f"lb:stream_ttfb_sum:{s}")
            s_cnt = await redis_client.get(f"lb:stream_ttfb_count:{s}")
            lines.append(f'ai_lb_stream_ttfb_seconds_sum{{model="{m}",node="{n}"}} {float(s_sum) if s_sum else 0.0}')
            lines.append(f'ai_lb_stream_ttfb_seconds_count{{model="{m}",node="{n}"}} {int(s_cnt) if s_cnt else 0}')

    # Stream duration histogram
    dur_series = await redis_client.smembers("lb:stream_duration_series")
    if dur_series:
        lines.append("# HELP ai_lb_stream_duration_seconds Total stream duration for streaming requests")
        lines.append("# TYPE ai_lb_stream_duration_seconds histogram")
        for s in dur_series:
            try:
                m, n = s.split("|", 1)
            except Exception:
                continue
            for le in _LAT_BUCKETS:
                val = await redis_client.get(f"lb:stream_duration_bucket:{s}:{le}")
                v = int(val) if val else 0
                le_str = "+Inf" if le == float("inf") else ("%.2f" % le).rstrip('0').rstrip('.')
                lines.append(f'ai_lb_stream_duration_seconds_bucket{{model="{m}",node="{n}",le="{le_str}"}} {v}')
            s_sum = await redis_client.get(f"lb:stream_duration_sum:{s}")
            s_cnt = await redis_client.get(f"lb:stream_duration_count:{s}")
            lines.append(f'ai_lb_stream_duration_seconds_sum{{model="{m}",node="{n}"}} {float(s_sum) if s_sum else 0.0}')
            lines.append(f'ai_lb_stream_duration_seconds_count{{model="{m}",node="{n}"}} {int(s_cnt) if s_cnt else 0}')

    # Aggregate hedge wins per model
    try:
        models = await redis_client.smembers("lb:hedge_wins_models")
        lines.append("# HELP ai_lb_hedge_wins_total Hedge wins per model")
        lines.append("# TYPE ai_lb_hedge_wins_total counter")
        for m in models:
            val = await redis_client.get(f"lb:hedge_wins_model:{m}")
            v = int(val) if val else 0
            lines.append(f'ai_lb_hedge_wins_total{{model="{m}"}} {v}')
    except Exception:
        pass

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
    cpt = body.get("backend_cost_per_token")
    if isinstance(cpt, dict):
        config.BACKEND_COST_PER_TOKEN.update({str(k): v for k, v in cpt.items()})
        applied["backend_cost_per_token"] = {str(k): v for k, v in cpt.items()}
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
