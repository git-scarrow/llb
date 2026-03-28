import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import httpx
import redis.asyncio as redis
from mcp import Server
from mcp.types import (
    Tool,
    CallToolRequest,
    CallToolResult,
    ListToolsResult,
    Prompt,
    ListPromptsResult,
    Resource,
    ListResourcesResult,
    ReadResourceRequest,
    ReadResourceResult,
)


LB_URL = os.getenv("LLB_URL", os.getenv("AI_LB_URL", f"http://localhost:{os.getenv('LLB_PORT', os.getenv('AI_LB_PORT', '8002'))}"))  # COMPAT: AI_LB_* fallback remove after 2026-06-01
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

server = Server("llb")


async def _get_http() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(60.0))


async def _get_redis() -> redis.Redis:
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


@server.list_tools()
async def list_tools() -> ListToolsResult:
    return ListToolsResult(
        tools=[
            Tool(name="lb.list_models", description="List models via LB /v1/models", inputSchema={"type": "object", "properties": {}}),
            Tool(name="lb.chat_completion", description="Non-stream chat completion via LB", inputSchema={"type": "object", "required": ["model", "messages"], "properties": {"model": {"type": "string"}, "messages": {"type": "array"}, "temperature": {"type": "number"}}}),
            Tool(name="lb.embeddings", description="Embeddings via LB", inputSchema={"type": "object", "required": ["model", "input"], "properties": {"model": {"type": "string"}, "input": {}}}),
            Tool(name="lb.health", description="LB health JSON", inputSchema={"type": "object", "properties": {}}),
            Tool(name="lb.metrics", description="Prometheus metrics (optionally filtered)", inputSchema={"type": "object", "properties": {"contains": {"type": "string"}}}),
            Tool(name="lb.get_nodes", description="Healthy nodes with inflight/failures/maxconn", inputSchema={"type": "object", "properties": {"includeStats": {"type": "boolean"}}}),
            Tool(name="lb.set_maxconn", description="Set per-node concurrency cap (Redis)", inputSchema={"type": "object", "required": ["node", "maxconn"], "properties": {"node": {"type": "string"}, "maxconn": {"type": "integer"}, "ttl_secs": {"type": "integer"}}}),
            Tool(name="lb.common_models", description="Exact ID intersection across nodes", inputSchema={"type": "object", "properties": {"nodes": {"type": "array", "items": {"type": "string"}}}}),
        ]
    )


@server.call_tool()
async def call_tool(request: CallToolRequest) -> CallToolResult:
    name = request.name
    params: Dict[str, Any] = request.arguments or {}
    async with await _get_http() as http:
        if name == "lb.list_models":
            resp = await http.get(f"{LB_URL}/v1/models")
            resp.raise_for_status()
            data = resp.json()
            ids = [m.get("id") for m in data.get("data", [])]
            return CallToolResult(content=[{"type": "text", "text": json.dumps(ids)}])

        if name == "lb.chat_completion":
            payload = {k: v for k, v in params.items() if k in ("model", "messages", "temperature")}
            payload["stream"] = False
            resp = await http.post(f"{LB_URL}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            # Best-effort aggregate from OpenAI-compatible response
            text = []
            for ch in data.get("choices", []):
                msg = ch.get("message", {})
                if "content" in msg and msg["content"]:
                    text.append(msg["content"]) 
            return CallToolResult(content=[{"type": "text", "text": "".join(text) or json.dumps(data)}])

        if name == "lb.embeddings":
            payload = {k: v for k, v in params.items() if k in ("model", "input")} 
            resp = await http.post(f"{LB_URL}/v1/embeddings", json=payload)
            resp.raise_for_status()
            return CallToolResult(content=[{"type": "text", "text": resp.text}])

        if name == "lb.health":
            resp = await http.get(f"{LB_URL}/health")
            resp.raise_for_status()
            return CallToolResult(content=[{"type": "text", "text": resp.text}])

        if name == "lb.metrics":
            contains: Optional[str] = params.get("contains")
            txt = (await http.get(f"{LB_URL}/metrics")).text
            if contains:
                lines = [ln for ln in txt.splitlines() if contains in ln]
                txt = "\n".join(lines)
            return CallToolResult(content=[{"type": "text", "text": txt}])

    # Redis-backed tools
    r = await _get_redis()
    try:
        if name == "lb.get_nodes":
            include = bool(params.get("includeStats", True))
            nodes = list(await r.smembers("nodes:healthy"))
            if not include:
                return CallToolResult(content=[{"type": "text", "text": json.dumps(nodes)}])
            out = []
            for n in nodes:
                inflight = await r.get(f"node:{n}:inflight")
                failures = await r.get(f"node:{n}:failures")
                maxconn = await r.get(f"node:{n}:maxconn")
                out.append({"node": n, "inflight": int(inflight or 0), "failures": int(failures or 0), "maxconn": int(maxconn or 0)})
            return CallToolResult(content=[{"type": "text", "text": json.dumps(out)}])

        if name == "lb.set_maxconn":
            node = params["node"]
            maxconn = int(params["maxconn"])
            ttl = int(params.get("ttl_secs")) if params.get("ttl_secs") is not None else None
            await r.set(f"node:{node}:maxconn", maxconn)
            if ttl and ttl > 0:
                await r.expire(f"node:{node}:maxconn", ttl)
            return CallToolResult(content=[{"type": "text", "text": json.dumps({"node": node, "maxconn": maxconn, "ttl": ttl or 0})}])

        if name == "lb.common_models":
            nodes: Optional[List[str]] = params.get("nodes")
            if not nodes:
                nodes = sorted(list(await r.smembers("nodes:healthy")))
            sets: List[set] = []
            for n in nodes:
                raw = await r.get(f"node:{n}:models")
                if not raw:
                    sets.append(set())
                else:
                    data = json.loads(raw)
                    ids = {m.get("id") for m in data.get("data", []) if m.get("id")}
                    sets.append(ids)
            common = set.intersection(*sets) if sets else set()
            return CallToolResult(content=[{"type": "text", "text": json.dumps(sorted(list(common)))}])
    finally:
        await r.close()

    return CallToolResult(content=[{"type": "text", "text": json.dumps({"error": f"unknown tool {name}"})}])


@server.list_resources()
async def list_resources() -> ListResourcesResult:
    return ListResourcesResult(resources=[
        Resource(uri="llb/metrics", mimeType="text/plain"),
        Resource(uri="llb/health", mimeType="application/json"),
    ])


@server.read_resource()
async def read_resource(req: ReadResourceRequest) -> ReadResourceResult:
    uri = req.uri
    async with await _get_http() as http:
        if uri == "llb/metrics":
            txt = (await http.get(f"{LB_URL}/metrics")).text
            return ReadResourceResult(contents=[{"uri": uri, "mimeType": "text/plain", "text": txt}])
        if uri == "llb/health":
            txt = (await http.get(f"{LB_URL}/health")).text
            return ReadResourceResult(contents=[{"uri": uri, "mimeType": "application/json", "text": txt}])
    return ReadResourceResult(contents=[{"uri": uri, "mimeType": "text/plain", "text": "unknown resource"}])


def main() -> None:
    # stdio transport
    asyncio.run(server.run_stdio())


if __name__ == "__main__":
    main()

