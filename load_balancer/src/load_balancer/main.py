import json
import logging
import redis.asyncio as redis
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse

from . import config
from .routing.strategies import get_routing_strategy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

redis_client = None
router = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, router
    redis_client = redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=5,
        retry_on_timeout=True,
    )
    try:
        await redis_client.ping()
        logger.info("Redis connected")
    except Exception as e:
        logger.warning("Redis not available at startup: %s", e)
    router = get_routing_strategy(config.ROUTING_STRATEGY)
    logger.info("Routing strategy: %s", config.ROUTING_STRATEGY)
    try:
        yield
    finally:
        if redis_client:
            await redis_client.aclose()


app = FastAPI(title="LLB", lifespan=lifespan)


async def _eligible_nodes(model_name: str) -> list[str]:
    """Return healthy nodes that advertise model_name."""
    try:
        healthy = sorted(await redis_client.smembers("nodes:healthy"))
    except Exception:
        return []
    eligible = []
    for node in healthy:
        models_json = await redis_client.get(f"node:{node}:models")
        if not models_json:
            continue
        try:
            models = json.loads(models_json).get("data", [])
        except Exception:
            models = []
        supports = await redis_client.get(f"node:{node}:supports:{model_name}")
        if any(m.get("id") == model_name for m in models) or (supports and supports != "0"):
            eligible.append(node)
    return eligible


async def _pick_node(model_name: str) -> str:
    """Select a backend node for model_name, or raise 503."""
    nodes = await _eligible_nodes(model_name)
    if not nodes:
        raise HTTPException(status_code=503, detail={"error": f"no eligible nodes for model '{model_name}'"})
    node = await router.select_node(nodes, model_name, redis_client)
    if not node:
        raise HTTPException(status_code=503, detail={"error": "router returned no node"})
    try:
        await redis_client.incrby("lb:requests_total", 1)
    except Exception:
        pass
    return node


async def _model_from_request(request: Request) -> str:
    try:
        body = await request.json()
        return body.get("model") or config.DEFAULT_CHAT_MODEL or ""
    except Exception:
        return config.DEFAULT_CHAT_MODEL or ""


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    try:
        healthy = sorted(await redis_client.smembers("nodes:healthy"))
    except Exception:
        return JSONResponse({"object": "list", "data": []})
    seen: dict = {}
    for node in healthy:
        models_json = await redis_client.get(f"node:{node}:models")
        if not models_json:
            continue
        try:
            for m in json.loads(models_json).get("data", []):
                mid = m.get("id")
                if mid and mid not in seen:
                    seen[mid] = m
        except Exception:
            pass
    return {"object": "list", "data": list(seen.values())}


@app.get("/v1/nodes")
async def list_nodes():
    try:
        healthy = sorted(await redis_client.smembers("nodes:healthy"))
        return {"nodes": list(healthy)}
    except Exception:
        return {"nodes": []}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    model = await _model_from_request(request)
    node = await _pick_node(model)
    return RedirectResponse(
        url=f"http://{node}/v1/chat/completions",
        status_code=307,
        headers={"X-Routed-Node": node},
    )


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    model = await _model_from_request(request)
    node = await _pick_node(model)
    return RedirectResponse(
        url=f"http://{node}/v1/embeddings",
        status_code=307,
        headers={"X-Routed-Node": node},
    )


@app.post("/v1/responses")
async def responses(request: Request):
    model = await _model_from_request(request)
    node = await _pick_node(model)
    return RedirectResponse(
        url=f"http://{node}/v1/responses",
        status_code=307,
        headers={"X-Routed-Node": node},
    )
