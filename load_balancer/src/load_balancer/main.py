import asyncio
import logging
import os
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse

from .routing.strategies import get_routing_strategy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCAN_HOSTS = os.getenv("SCAN_HOSTS", "")
SCAN_PORTS = os.getenv("SCAN_PORTS", "11434")
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "60"))
ROUTING_STRATEGY = os.getenv("ROUTING_STRATEGY", "ROUND_ROBIN")
DEFAULT_CHAT_MODEL = (os.getenv("DEFAULT_CHAT_MODEL", "") or "").strip() or None

# ---------------------------------------------------------------------------
# In-process state (constant size, never grows with request volume)
# ---------------------------------------------------------------------------

# node -> list of model-id strings
_healthy: dict[str, list[str]] = {}
_router = None
_scan_client: httpx.AsyncClient | None = None


def _parse_scan_targets() -> list[tuple[str, int]]:
    raw = SCAN_HOSTS.replace(";", ",").replace(" ", ",")
    hosts_raw = [h.strip() for h in raw.split(",") if h.strip()]
    ports = [int(p.strip()) for p in SCAN_PORTS.replace(";", ",").split(",") if p.strip()]
    pairs: list[tuple[str, int]] = []
    for h in hosts_raw:
        if ":" in h:
            host, port_str = h.rsplit(":", 1)
            pairs.append((host, int(port_str)))
        else:
            for p in ports:
                pairs.append((h, p))
    return pairs


async def _probe_node(session: httpx.AsyncClient, host: str, port: int) -> tuple[str, list[str]] | None:
    addr = f"{host}:{port}"
    try:
        r = await session.get(f"http://{addr}/v1/models", timeout=10)
        r.raise_for_status()
        loaded = [m["id"] for m in r.json().get("data", []) if m.get("id")]
        # merge cold models from /api/tags
        try:
            t = await session.get(f"http://{addr}/api/tags", timeout=10)
            if t.status_code == 200:
                loaded_set = set(loaded)
                for m in t.json().get("models", []):
                    mid = m.get("name")
                    if mid and mid not in loaded_set:
                        loaded.append(mid)
        except Exception:
            pass
        return addr, loaded
    except Exception:
        return None


async def _scan_loop():
    global _scan_client
    targets = _parse_scan_targets()
    if not targets:
        logger.warning("SCAN_HOSTS is empty — no nodes will be discovered")
        return
    logger.info("Scanning targets every %ds: %s", SCAN_INTERVAL, targets)
    _scan_client = httpx.AsyncClient()
    try:
        while True:
            results = await asyncio.gather(*[_probe_node(_scan_client, h, p) for h, p in targets])
            new_state: dict[str, list[str]] = {}
            for result in results:
                if result:
                    addr, models = result
                    new_state[addr] = models
                    if addr not in _healthy:
                        logger.info("Node up: %s (%d models)", addr, len(models))
            for addr in list(_healthy):
                if addr not in new_state:
                    logger.info("Node down: %s", addr)
            _healthy.clear()  # atomic enough for CPython's GIL; asyncio never preempts between these two lines
            _healthy.update(new_state)
            await asyncio.sleep(SCAN_INTERVAL)
    finally:
        if _scan_client:
            await _scan_client.aclose()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _router
    _router = get_routing_strategy(ROUTING_STRATEGY)
    logger.info("Routing strategy: %s", ROUTING_STRATEGY)
    task = asyncio.create_task(_scan_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="LLB", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

async def _pick_node(model: str) -> str:
    nodes = [addr for addr, models in _healthy.items() if model in models]
    if not nodes:
        # Fall back to all healthy nodes if model not yet discovered
        nodes = list(_healthy)
    if not nodes:
        raise HTTPException(status_code=503, detail={"error": f"no healthy nodes for model '{model}'"})
    node = await _router.select_node(nodes, model, None)
    if not node:
        raise HTTPException(status_code=503, detail={"error": "router returned no node"})
    return node


async def _model_from_body(request: Request) -> str:
    try:
        return (await request.json()).get("model") or DEFAULT_CHAT_MODEL or ""
    except Exception:
        return DEFAULT_CHAT_MODEL or ""

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "nodes": len(_healthy)}


@app.get("/v1/models")
async def list_models():
    seen: dict = {}
    for models in _healthy.values():
        for mid in models:
            if mid not in seen:
                seen[mid] = {"id": mid, "object": "model", "owned_by": "ollama"}
    return {"object": "list", "data": list(seen.values())}


@app.get("/v1/nodes")
async def list_nodes():
    return {"nodes": list(_healthy)}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    node = await _pick_node(await _model_from_body(request))
    return RedirectResponse(url=f"http://{node}/v1/chat/completions", status_code=307,
                            headers={"X-Routed-Node": node})


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    node = await _pick_node(await _model_from_body(request))
    return RedirectResponse(url=f"http://{node}/v1/embeddings", status_code=307,
                            headers={"X-Routed-Node": node})


@app.post("/v1/responses")
async def responses(request: Request):
    node = await _pick_node(await _model_from_body(request))
    return RedirectResponse(url=f"http://{node}/v1/responses", status_code=307,
                            headers={"X-Routed-Node": node})
