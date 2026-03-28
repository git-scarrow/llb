LLB — Large Language Balancer

Overview
- Consolidates two implementations into a single FastAPI service with Redis‑backed state.
- Preserves existing config defaults (ROUND_ROBIN) while adding health‑aware routing and failover.

Design Choices (HAProxy vs. LLB)
- Balancing: adopt LEAST_LOADED (LLB) with per‑node `maxconn` (HAProxy semantics via Redis keys).
- Health checks: keep `/v1/models` probing (HAProxy style) via the Monitor service; dynamic membership in `nodes:healthy`.
- Failover: retry on 5xx/network errors with bounded `MAX_RETRIES` (LLB), plus a simple circuit breaker.
- Streaming: maintain byte stream passthrough for chat completions with accurate inflight accounting.
- Metrics: expose minimal Prometheus text at `/metrics` (LLB) including requests total, up, inflight, failures.
- Config compatibility: defaults match old behavior; new envs are optional with safe fallbacks.

Key Features
- Routing strategies: ROUND_ROBIN, RANDOM, LEAST_LOADED (select via `ROUTING_STRATEGY`).
- Health‑aware selection using Redis counters and optional per‑node `node:{host:port}:maxconn` caps.
- Bounded failover with circuit breaker keys `node:{n}:failures` and `node:{n}:cb_open`.
- Endpoints: `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/nodes`, `/v1/eligible_nodes`, `/health`, `/metrics`.

Config
- `ROUTING_STRATEGY`: `ROUND_ROBIN` (default), `RANDOM`, `LEAST_LOADED`.
- `MAX_RETRIES` (default 2): number of additional nodes to try after the first.
- `REQUEST_TIMEOUT_SECS` (default 60): upstream request timeout.
- `FAILURE_PENALTY_TTL_SECS` (default 30): TTL for failure counters.
- `CIRCUIT_BREAKER_THRESHOLD` (default 3), `CIRCUIT_BREAKER_TTL_SECS` (default 30).
- `node:{node}:maxconn` (Redis key): optional cap mirroring HAProxy `maxconn`.

Compatibility & Migration
- Existing HAProxy deployments remain valid; this service can be deployed alongside and gradually take over.
- Health check behavior (`GET /v1/models`) is preserved, so existing node agents continue to work.
- Defaults (ROUND_ROBIN) ensure unchanged routing unless `ROUTING_STRATEGY` is set.

Testing
- Tests validate routing correctness, health check reliability, failover/circuit breaker, and concurrent load.

Performance Notes
- Async httpx streaming with minimal overhead; inflight counters around the full request lifecycle.
