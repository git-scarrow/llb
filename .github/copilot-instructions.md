# Copilot Instructions for LLB

## Project Overview

**LLB** (Large Language Balancer) is a modular, containerized load balancer for self-hosted Large Language Models (LLMs). It routes OpenAI-compatible API requests (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`) across multiple backend nodes with health-aware selection, circuit breaking, retries, and Prometheus metrics.

## Repository Structure

```
load_balancer/   # Core FastAPI routing service (main component)
  src/load_balancer/
    main.py           # FastAPI app, all HTTP endpoints
    config.py         # All env-var configuration (dataclass)
    routing/
      strategies.py   # ROUND_ROBIN, RANDOM, LEAST_LOADED, P2C routing logic
    execution/        # Request execution and retry logic
    request_validation.py
  tests/
    test_unified_lb.py
    test_cost_aware_routing.py
    test_embeddings_and_forced_node.py
    test_plan_stream.py
    e2e/              # End-to-end tests (require live stack)
monitor/         # Background service: node discovery and health checks
node_agent/      # Lightweight agent running on LLM hosts (reports metrics)
mcp_server/      # stdio MCP server exposing LB tools to AI agents
tools/           # CLI utilities (e.g., tui_stream.py for live multi-node streaming)
scripts/         # CI and operational scripts
```

## Tech Stack

- **Language**: Python 3.12+ (root package); sub-packages declare `>=3.9`
- **Web Framework**: FastAPI with async httpx for upstream requests
- **State Store**: Redis (inflight counters, EWMA keys, circuit breaker keys, node membership)
- **Package Manager**: `uv` (lockfile: `uv.lock`); use `uv run` or activate `.venv`
- **Linter/Formatter**: `ruff` (check and format)
- **Testing**: `pytest` with `pytest-asyncio`
- **Containerization**: Docker + Docker Compose (`docker-compose.yml`, `docker-compose.override.yml`)

## Key Conventions

### Configuration

All runtime configuration lives in `load_balancer/src/load_balancer/config.py` as a dataclass populated from environment variables. When adding new config:
- Add the field to the `Config` dataclass with a sensible default.
- Document the env var name in `README.md` under the Deploy section.

### Routing Strategies

Routing strategies are in `load_balancer/src/load_balancer/routing/strategies.py`. The active strategy is selected via the `ROUTING_STRATEGY` env var (`ROUND_ROBIN`, `RANDOM`, `LEAST_LOADED`, `P2C`).

- **P2C** (default): Power-of-Two-Choices; cost-aware when `P2C_BETA > 0`.
- Cost-aware P2C uses Redis EWMA keys: `lb:output_tokens_ewma:{model}|{node}` and `lb:output_tokens_count:{model}|{node}`.
- `BACKEND_COST_PER_TOKEN` is a JSON env var mapping `model_id` → `{input, output}` USD per 1M tokens.

### Redis Key Conventions

| Key pattern | Purpose |
|---|---|
| `nodes:healthy` | Set of healthy node addresses |
| `node:{host:port}:models` | Set of models advertised by a node |
| `node:{host:port}:maxconn` | Optional concurrency cap |
| `node:{host:port}:failures` | Failure counter (circuit breaker) |
| `node:{host:port}:cb_open` | Circuit breaker open flag |
| `node:{host:port}:inflight` | Current inflight request count |
| `node:{host:port}:supports:{model}` | Model warmed/supported flag (set by monitor) |
| `lb:output_tokens_ewma:{model}|{node}` | EWMA of output tokens (P2C cost routing) |

### Testing

- **Unit tests**: `python3 -m pytest load_balancer/tests/ --tb=short -q`
- **E2E tests**: Gated by `AI_LB_BASE_URL` env var; run with `pytest -m e2e`. Skipped (60 tests) when env var is unset.
- Tests use `pytest-asyncio` and mock Redis/httpx. Keep new tests consistent with the existing mock patterns in `test_unified_lb.py`.

### Code Style

- Follow `ruff` rules (configured in `pyproject.toml`).
- Use `async`/`await` throughout; avoid blocking calls in request handlers.
- Type-annotate all new functions and classes.
- Keep request handler logic thin; push business logic into `routing/` or `execution/`.

### Building & Running Locally

```bash
# Start minimal stack (Redis + load balancer)
docker compose up --build

# Start full stack (adds monitor)
docker compose --profile full up --build

# Run unit tests
cd load_balancer && python3 -m pytest tests/ --tb=short -q

# Lint
ruff check load_balancer/
ruff format --check load_balancer/
```

## Important Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /v1/models` | Aggregated model list from all healthy nodes |
| `POST /v1/chat/completions` | Routed chat completion (supports streaming) |
| `POST /v1/embeddings` | Routed embeddings request |
| `GET /v1/nodes` | List all known nodes |
| `GET /v1/eligible_nodes` | List eligible nodes for a model |
| `GET /v1/debug/eligible?model=<id>` | Per-node eligibility reasons |
| `GET /health` | Health check |
| `GET /metrics` | Prometheus metrics |
| `POST /v1/admin/prefs` | Hot-update runtime preferences |
| `POST /v1/admin/reset_histograms` | Reset latency histograms for a model |

## Response Headers (Observability)

Non-streaming responses include: `x-request-id`, `x-selected-model`, `x-routed-node`, `x-attempts`, `x-failover-count`, `x-fallback-model`.
Streaming SSE includes `event: meta` (initial) and `event: failover` (on reroute) JSON payloads.
