MCP Server for LLB

Overview
- Exposes the LLB (load balancer + monitor) via the Model Context Protocol (MCP) so any compliant agent can call tools/resources without bespoke adapters.
- Provides tools for listing models, creating chat completions, embeddings, querying health/metrics, inspecting nodes, setting per‑node concurrency caps, and finding common models across nodes.

Quick Start (CLI / Agents)
- Requirements: Python 3.11+, pip
- Install deps: `pip install -r mcp_server/requirements.txt`
- Run server (stdio): `python mcp_server/server.py`
  - Env vars:
    - `LB_URL` (default `http://localhost:8000`)
    - `REDIS_HOST` (default `localhost`) and `REDIS_PORT` (default `6379`)

Claude Desktop (example)
- Add to `claude_desktop_config.json` (or UI):
  {
    "mcpServers": {
      "llb": {
        "command": "python",
        "args": ["mcp_server/server.py"],
        "env": {"LB_URL": "http://localhost:8000"}
      }
    }
  }

Exposed Tools
- `lb.list_models()` → List of model IDs from `GET /v1/models`.
- `lb.chat_completion(model, messages, temperature?)` → Calls `POST /v1/chat/completions` (non‑stream), returns aggregated text.
- `lb.embeddings(model, input)` → Calls `POST /v1/embeddings` and returns JSON.
- `lb.health()` → Returns `GET /health`.
- `lb.metrics(filter?)` → Returns selected Prometheus lines (optionally filter by prefix/substring).
- `lb.get_nodes(include_stats?)` → Returns healthy nodes with `inflight`, `failures`, `maxconn` (Redis).
- `lb.set_maxconn(node, maxconn, ttl_secs?)` → Writes `node:{node}:maxconn` with optional TTL (Redis).
- `lb.common_models(nodes?)` → Returns exact ID intersection across either `nodes:healthy` or explicit list.

Resources
- `resourceId: llb/metrics` → Fetch returns the full Prometheus text.
- `resourceId: llb/health` → Fetch returns JSON for LB health.

Notes
- Chat completions are returned as aggregate text for simplicity; streaming could be added later via incremental MCP content.
- `set_maxconn` changes take effect immediately in the LB thanks to atomic enforcement.

