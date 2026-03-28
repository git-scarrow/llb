# LLB: Large Language Balancer

A modular, containerized load balancer for self-hosted Large Language Models (LLMs).

## Overview

This project provides a suite of tools to discover, monitor, and route requests to various LLM providers running on your network. It is designed to be extensible and configurable.

- **Load Balancer**: The main entry point that routes user requests.
- **Monitor**: A background service that discovers and health-checks LLM nodes.
- **Node Agent**: A lightweight agent for LLM hosts to report detailed hardware metrics.
- **Redis**: A state store for sharing information between services.

See the README in each service's directory for more details.

## Deploy

- Minimal (default): starts `redis` + `load_balancer` only.
  - `docker compose up --build`
  - Env (optional): set in `.env` or inline
    - `LOAD_BALANCER_PORT` (default 8000)
    - `ROUTING_STRATEGY` (`ROUND_ROBIN`|`RANDOM`|`LEAST_LOADED`|`P2C`, default `P2C`)
    - `REQUEST_TIMEOUT_SECS` (default 60)
    - `ATTEMPTS_PER_MODEL` (default 3) — total attempts budget (1 initial + retries)
    - `RETRY_BACKOFF_MS` (default `50,100`)
    - `CIRCUIT_BREAKER_THRESHOLD` (default 3)
    - `CIRCUIT_BREAKER_COOLDOWN_SECS` (default 60)
    - `MODEL_SENTINELS` (comma list; default `auto,default`) — treat these model names as “auto-select”
    - `LM_MODEL` (optional; default `auto`) — also treated as a sentinel value if used as `model`
    - `PREFERRED_MODELS` (optional) — priority order for auto selection
    - `AUTO_MODEL_STRATEGY` (`any_first`|`intersection_first`)
    - `AUTO_MIN_NODES` (default 2) — prefer models with at least N eligible nodes for `auto`
    - `LM_PREFER_SMALL` (default 1) — prefer the “small” class first for `auto`
    - `CROSS_MODEL_FALLBACK` (default 0) — when enabled, LB retries with a viable sibling model if none of the requested model’s nodes work
    - `STICKINESS`: `STICKY_SESSIONS_ENABLED` (default `false`), `STICKY_TTL_SECS` (default `600`)
    - `RETRY_AFTER_SECS` (default `2`) — seconds returned in `Retry-After` when saturated (429)
    - `MODEL_WEIGHTS` (e.g., `qwen14b=1.0,mistral=0.9`) — biases auto model choice; overridable via admin API

- Full (with discovery/health): enables the `monitor` via Compose profile.
  - `docker compose --profile full up --build`
  - Additional env for monitor (optional):
    - `SCAN_HOSTS` — accepts hostnames or `host:port` pairs. If a host includes a port, it overrides `SCAN_PORTS` for that host.
    - `SCAN_PORTS` (default `1234`) — applied to hosts without an inline port.
    - `SCAN_INTERVAL` (seconds, default `60`)
    - `DEFAULT_MAXCONN` (integer cap applied to all nodes; optional)
    - `MAXCONN_MAP` (e.g., `pc-sam.ts.net:1234=2,macbook:1234=4`)

#### Auto-unload and model warming

- Some backends (e.g., LM Studio) auto-unload models. To keep critical models resident and ready, enable warming in the monitor.
  - Set in `.env`:
    - `WARM_ENABLED=true`
    - `WARM_MODELS=qwen/qwen3-8b,mistralai/mistral-small-3.2`
    - `WARM_TIMEOUT_SECS=120` (time allowed for initial load)
    - `WARM_RETRIES=1`
  - The monitor posts a tiny non-stream chat to each configured model on every healthy node. On success it sets `node:{host:port}:supports:{model}=1` in Redis.
  - The load balancer considers a node eligible for a model if either `/v1/models` advertises it or `supports:{model}` is set by the monitor.
  - For best results, use `ROUTING_STRATEGY=P2C` (default).

### Networking tips (Docker/WSL/Tailscale)

- Linux/WSL: `host.docker.internal` is not guaranteed to reach services inside WSL or behind Tailscale. Prefer explicit FQDNs/IPs (Tailscale or LAN).
  - Set `SCAN_HOSTS` to values like `pc-name.ts.net,192.168.x.y` and keep `SCAN_PORTS=1234`.
  - If you must expose via Windows, you can add a Windows portproxy from 1234 → `<WSL_IP>:1234` and then scan the Windows IP directly.
- `SCAN_HOSTS` accepts both plain hosts and `host:port`. Examples:
  - Same port for all: `SCAN_HOSTS=pc-sam.ts.net,macbook` + `SCAN_PORTS=1234`.
  - Different ports: `SCAN_HOSTS=pc-sam.ts.net:1234,macbook:8080`.

## Auto Model Selection and Fallback

- `auto` resolves to a concrete model id only among models that currently have eligible nodes (no dead mappings).
- Prefer classes with multiple nodes:
  - `AUTO_MIN_NODES` (default 2) and `LM_PREFER_SMALL` bias selection.
  - Provide additional classes via `LB_MODEL_CLASSES` (JSON) if desired.
- Optional cross-model fallback:
  - If all nodes for a requested model fail or none are eligible, and `CROSS_MODEL_FALLBACK=1`, retry once with a viable sibling model.

## Eligibility and p95 Gating

- Nodes are eligible when they:
  - are in `nodes:healthy`,
  - advertise the model in `node:{host:port}:models`, and
  - are not “tripped” by the circuit breaker.
- Optional p95 gate: exclude nodes whose recent p95 exceeds a threshold.
  - `MAX_P95_LATENCY_SECS` (default 5.0; set 0 to disable)
  - `ELIGIBILITY_MIN_P95_SAMPLES` (default 20) — require at least N samples before applying the gate
- Debug endpoint to inspect decisions:
  - `GET /v1/debug/eligible?model=<id>` → returns `healthy`, `eligible`, and per-node reasons.

## Operator TUI (Live Multi-Node Streaming)

- A local TUI streams the same prompt through all eligible nodes concurrently and shows per-node output and live throughput.
- Usage:
  - `. .venv/bin/activate && pip install rich httpx`
  - `python tools/tui_stream.py --lb http://localhost:8000 --pick-model --require-all-nodes --prompt "Say hello from each node"`
- Options:
  - `--pick-model`: interactively choose a model (falls back to first if no TTY)
  - `--require-all-nodes`: list only models available on all selected nodes
  - `--pick-nodes`: interactively choose which healthy nodes to stream to
  - `--include/--exclude`: comma-separated filters for node names
  - `--filter-models`: substring filter for model list
- The TUI uses `/v1/nodes` and `/v1/eligible_nodes` and forces streaming to each node via `?node=<host:port>`.

## Observability (LB)

- Response headers: `x-request-id`, `x-selected-model`, `x-routed-node`, `x-attempts`, `x-failover-count` (non-streaming), `x-fallback-model` (if cross-model fallback used).
- Streaming SSE prelude: `event: meta` (initial), `event: failover` (on reroute) with JSON payload including request_id, attempts, and node.
- `/metrics` includes failover counters and latency histograms per model/node.
  - Also exposes streaming histograms: `ai_lb_stream_ttfb_seconds` and `ai_lb_stream_duration_seconds`.
  - Hedge counters scaffolded: `ai_lb_hedges_total`, `ai_lb_hedge_wins{model,node}`.

## Admin API

- `POST /v1/admin/prefs` to hot-update runtime behavior.
  - Body fields (all optional): `preferred_models`, `model_weights`, `model_caps`, `node_caps`, `auto_model_strategy`.
  - Updates take effect immediately and are stored in Redis.

- `POST /v1/admin/reset_histograms` to delete histogram/series metrics for a model.
  - Body:
    - `model` (string, required)
    - `nodes` (array of `host:port`, optional; autodetected if omitted)
    - `include` (array: `latency`, `stream_ttfb`, `stream_duration`; default all)
    - `dry_run` (bool, default false)
  - Example (dry-run):
    ```sh
    curl -s -X POST http://127.0.0.1:8000/v1/admin/reset_histograms \
      -H 'Content-Type: application/json' \
      -d '{"model":"qwen/qwen3-8b","nodes":["macbook:1234","sams-macbook-pro:1234"],"dry_run":true}' | jq
    ```

## MCP Server (for Any Agent)

- An optional stdio MCP server exposes LLB tools/resources to any compliant agent.
- See `mcp_server/README.md` for instructions and available tools/resources.
