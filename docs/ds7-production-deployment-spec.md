# LLB Production Deployment Specification (AILB-DS-7)

**Phase:** 5
**Status:** Design Spec
**Dependencies:** AILB-DS-6 (E2E Validation — PASS 7/8), AILB-MT-1 (Adaptive Thresholds)

---

## 1. Containerization

### 1.1 Multi-Stage Dockerfile

**Assumption:** The production image replaces `load_balancer/Dockerfile`. The build stage installs
deps; the runtime stage is minimal. The image listens on port **8000**.

```dockerfile
# load_balancer/Dockerfile.prod
FROM python:3.12-slim AS builder
WORKDIR /build

# Install build dependencies
RUN pip install --no-cache-dir uv

COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY ./src /app

# Non-root user for security
RUN useradd -r -u 1001 -g root ailb && chown -R ailb:root /app
USER ailb

# Health check: poll /health every 10s; allow 30s startup grace
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

CMD ["uvicorn", "load_balancer.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "warning", \
     "--no-access-log"]
```

### 1.2 nix-docker-configs Integration

**Assumption:** `nix-docker-configs` follows the convention of one directory per service with an
`image.nix` file that provides `dockerImage` and a `docker-compose.override.nix` for compose
fragments.

Create `services/llb/image.nix`:

```nix
# services/llb/image.nix
{ pkgs, ... }:
{
  # Build reference — used by CI/CD to push to local registry
  imageName = "localhost:5000/llb";
  imageTag = "latest";

  # Build command to run in llb repo root
  buildCmd = "docker build -f load_balancer/Dockerfile.prod -t localhost:5000/llb:latest ./load_balancer";

  # Health probe used by nix-docker-configs health collector
  healthProbe = "http://localhost:8000/health";
}
```

Create `services/llb/compose.nix`:

```nix
# services/llb/compose.nix — fragment merged by nix-docker-configs
{ ... }:
{
  services.llb = {
    image = "localhost:5000/llb:latest";
    restart = "unless-stopped";
    network_mode = "host";
    env_file = [ "/etc/llb/prod.env" ];
    secrets = [ "anthropic_api_key" "openai_api_key" ];
    depends_on = { redis.condition = "service_healthy"; };
  };
}
```

### 1.3 docker-compose Production Profile

The production profile uses a separate env file, named volumes with retention policy, and no
build context (pulls pre-built images from the local registry).

```yaml
# docker-compose.prod.yml
# Usage: docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile full up -d

x-redis-env: &redis-env
  REDIS_HOST: localhost
  REDIS_PORT: 6379

secrets:
  anthropic_api_key:
    file: /run/secrets/anthropic_api_key
  openai_api_key:
    file: /run/secrets/openai_api_key

services:
  redis:
    image: "redis:7-alpine"
    container_name: ai_lb_redis
    restart: always
    network_mode: host
    volumes:
      - redis_data:/data
      - ./config/redis.prod.conf:/usr/local/etc/redis/redis.conf:ro
    command: ["redis-server", "/usr/local/etc/redis/redis.conf"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 10
      start_period: 10s

  monitor:
    image: "localhost:5000/llb-monitor:latest"
    container_name: ai_lb_monitor
    network_mode: host
    restart: always
    env_file: /etc/llb/prod.env
    environment:
      <<: *redis-env
    depends_on:
      redis:
        condition: service_healthy
    profiles: ["full"]

  load_balancer:
    image: "localhost:5000/llb:latest"
    container_name: ai_lb_load_balancer
    network_mode: host
    restart: always
    env_file: /etc/llb/prod.env
    environment:
      <<: *redis-env
      # Secret injection: read from mounted files at runtime
      # See Section 2.3 — secrets are NOT passed as env vars
    secrets:
      - anthropic_api_key
      - openai_api_key
    depends_on:
      redis:
        condition: service_healthy
    logging:
      driver: "json-file"
      options:
        max-size: "50m"
        max-file: "5"

volumes:
  redis_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /var/lib/llb/redis
```

### 1.4 Redis Production Configuration

```conf
# config/redis.prod.conf

# ── Persistence ──────────────────────────────────────────────────────────────
# AOF (Append-Only File) for durability. RDB snapshots as backup.
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec          # fsync once per second (balance durability vs I/O)
no-appendfsync-on-rewrite yes # don't block AOF writes during BGREWRITE
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# RDB snapshot as secondary safety net
save 900 1
save 300 10
save 60 10000
dbfilename dump.rdb

# ── Memory ───────────────────────────────────────────────────────────────────
# llb stores metrics counters, node state, EWMA series, and circuit breaker
# flags. Peak usage under 100 nodes with 100 models: ~128 MB.
maxmemory 256mb

# Evict LRU keys with TTL first, then any LRU key.
# Metric counters (lb:*) have no TTL and must NOT be evicted.
# Node penalty keys (penalty:*) have short TTLs and are safe to evict.
maxmemory-policy allkeys-lru

# ── Networking ───────────────────────────────────────────────────────────────
bind 127.0.0.1          # localhost only; LB connects via host network
protected-mode yes
timeout 300             # close idle connections after 5 minutes
tcp-keepalive 60

# ── Logging ──────────────────────────────────────────────────────────────────
loglevel notice
logfile /var/log/redis/redis.log

# ── Slow log ─────────────────────────────────────────────────────────────────
slowlog-log-slower-than 10000  # 10ms
slowlog-max-len 128
```

---

## 2. Config Hardening

### 2.1 Env Var Validation at Boot

Add a `validate_config()` function called at application startup in `load_balancer/src/load_balancer/main.py`
(inside the `lifespan` context manager, before the app starts serving).

```python
# load_balancer/src/load_balancer/config_validation.py
"""Boot-time config validation. Fails fast with actionable error messages."""

import os
import sys
import json
import logging

log = logging.getLogger("ai_lb.config")


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


def _require(name: str, description: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        raise ConfigError(
            f"[MISSING REQUIRED CONFIG] {name} is not set.\n"
            f"  What it does: {description}\n"
            f"  Fix: export {name}=<value>  or add it to /etc/llb/prod.env"
        )
    return val


def _require_if(condition: bool, name: str, description: str) -> str | None:
    if not condition:
        return None
    return _require(name, description)


def validate_config() -> None:
    """
    Validate all required and conditionally-required env vars.
    Raises ConfigError with a full diagnostic message on first failure.
    Called once at startup inside lifespan().
    """
    errors: list[str] = []

    # ── Always required ───────────────────────────────────────────────────────
    # CLOUD_BACKENDS is required when no local scan hosts are configured.
    cloud_backends = os.getenv("CLOUD_BACKENDS", "").strip()
    scan_hosts = os.getenv("SCAN_HOSTS", "").strip()
    if not cloud_backends and not scan_hosts:
        errors.append(
            "[MISSING REQUIRED CONFIG] At least one of CLOUD_BACKENDS or SCAN_HOSTS must be set.\n"
            "  What it does: Defines the backend LLM nodes to route requests to.\n"
            "  Fix: Set CLOUD_BACKENDS=name=url|api_key|provider  or SCAN_HOSTS=host:port"
        )

    # ── Conditionally required ────────────────────────────────────────────────
    planner_backend = os.getenv("PLANNER_BACKEND", "").strip()
    plan_enabled = os.getenv("MULTI_EXEC_ENABLED", "true").lower() in ("1", "true", "yes")
    if plan_enabled and not planner_backend and not cloud_backends:
        errors.append(
            "[CONFIG WARNING] PLANNER_BACKEND is not set and no CLOUD_BACKENDS are defined.\n"
            "  What it does: PLAN execution mode requires a planner backend for task decomposition.\n"
            "  Fix: Set PLANNER_BACKEND=<cloud_backend_name> or a host:port"
        )

    # ── Numeric range validation ──────────────────────────────────────────────
    int_ranges = {
        "CIRCUIT_BREAKER_THRESHOLD": (1, 100),
        "CIRCUIT_BREAKER_COOLDOWN_SECS": (1, 3600),
        "MIN_HEALTHY_NODES": (1, 1000),
        "REQUEST_TIMEOUT_SECS": (1, 600),
        "REDIS_PORT": (1, 65535),
    }
    for name, (lo, hi) in int_ranges.items():
        raw = os.getenv(name)
        if raw is not None:
            try:
                val = int(raw)
                if not lo <= val <= hi:
                    errors.append(
                        f"[INVALID CONFIG] {name}={raw} is out of valid range [{lo}, {hi}]."
                    )
            except ValueError:
                errors.append(
                    f"[INVALID CONFIG] {name}={raw!r} is not a valid integer."
                )

    # ── JSON format validation ────────────────────────────────────────────────
    lb_model_classes = os.getenv("LB_MODEL_CLASSES", "").strip()
    if lb_model_classes:
        try:
            json.loads(lb_model_classes)
        except json.JSONDecodeError as e:
            errors.append(
                f"[INVALID CONFIG] LB_MODEL_CLASSES is not valid JSON: {e}\n"
                f"  Fix: Ensure LB_MODEL_CLASSES is a valid JSON object."
            )

    backend_cost = os.getenv("BACKEND_COST_PER_TOKEN", "").strip()
    if backend_cost:
        try:
            json.loads(backend_cost)
        except json.JSONDecodeError as e:
            errors.append(
                f"[INVALID CONFIG] BACKEND_COST_PER_TOKEN is not valid JSON: {e}"
            )

    # ── Routing strategy validation ───────────────────────────────────────────
    strategy = os.getenv("ROUTING_STRATEGY", "P2C").upper()
    valid_strategies = {"ROUND_ROBIN", "RANDOM", "LEAST_LOADED", "P2C"}
    if strategy not in valid_strategies:
        errors.append(
            f"[INVALID CONFIG] ROUTING_STRATEGY={strategy!r} is not valid.\n"
            f"  Valid values: {', '.join(sorted(valid_strategies))}"
        )

    # ── Fail fast ────────────────────────────────────────────────────────────
    if errors:
        msg = "\n\n".join(errors)
        log.critical(
            "llb failed config validation at startup. Fix the following:\n\n%s\n\n"
            "Exiting.",
            msg
        )
        sys.exit(1)

    log.info("Config validation passed.")
```

Wire into `main.py` lifespan:

```python
# In load_balancer/src/load_balancer/main.py — lifespan context manager
from contextlib import asynccontextmanager
from load_balancer.config_validation import validate_config

@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_config()          # ← add this line FIRST, before Redis init
    # ... existing startup code ...
    yield
    # ... existing shutdown code ...
```

### 2.2 Secrets Management

**Mechanism: Docker secrets (mounted files).** API keys are never passed as environment
variables. They are written to files by the Docker daemon and read at runtime.

**Secret file locations (inside container):**

| Secret | Mounted Path | Used By |
|--------|-------------|---------|
| `anthropic_api_key` | `/run/secrets/anthropic_api_key` | CLOUD_BACKENDS Anthropic entry |
| `openai_api_key` | `/run/secrets/openai_api_key` | CLOUD_BACKENDS OpenAI entry |

**Secret reading pattern** — add to `config.py`:

```python
def _read_secret(env_var: str, secret_file: str, default: str = "") -> str:
    """Read a secret from a mounted file, falling back to env var (dev only)."""
    import pathlib
    p = pathlib.Path(secret_file)
    if p.exists():
        return p.read_text().strip()
    return os.getenv(env_var, default)

ANTHROPIC_API_KEY = _read_secret(
    "ANTHROPIC_API_KEY",
    "/run/secrets/anthropic_api_key"
)
OPENAI_API_KEY = _read_secret(
    "OPENAI_API_KEY",
    "/run/secrets/openai_api_key"
)
```

**Creating secrets on the host:**

```bash
# On the deployment host
sudo mkdir -p /run/secrets
echo "sk-ant-api03-XXXX" | sudo tee /run/secrets/anthropic_api_key > /dev/null
echo "sk-XXXX" | sudo tee /run/secrets/openai_api_key > /dev/null
sudo chmod 600 /run/secrets/anthropic_api_key /run/secrets/openai_api_key
```

### 2.3 Environment Variable Reference

All variables read by `config.py`. Required = must be set in prod; all others have safe defaults.

**Prefix convention:** `AILB_` prefix intentionally absent — existing codebase uses bare names.

| Name | Required | Default | Description |
|------|----------|---------|-------------|
| `CLOUD_BACKENDS` | YES (if no local nodes) | — | `name=url\|api_key\|provider,...` Cloud backend specs |
| `CLOUD_MODELS` | NO | — | `backend=model1,model2;...` Models per backend |
| `REDIS_HOST` | NO | `localhost` | Redis hostname |
| `REDIS_PORT` | NO | `6379` | Redis port |
| `ROUTING_STRATEGY` | NO | `P2C` | `ROUND_ROBIN`, `RANDOM`, `LEAST_LOADED`, `P2C` |
| `MIN_HEALTHY_NODES` | NO | `1` | Minimum healthy nodes before returning 503 |
| `REQUEST_TIMEOUT_SECS` | NO | `60` | Per-backend request timeout in seconds |
| `ATTEMPTS_PER_MODEL` | NO | `3` | Total attempts budget (1 initial + retries) |
| `RETRY_BACKOFF_MS` | NO | `50,100` | Retry backoff delays in ms (comma-separated) |
| `FAILURE_PENALTY_TTL_SECS` | NO | `30` | How long a failure penalty persists in Redis |
| `CIRCUIT_BREAKER_THRESHOLD` | NO | `3` | Failures before CB opens |
| `CIRCUIT_BREAKER_COOLDOWN_SECS` | NO | `60` | Seconds CB stays open before half-open probe |
| `CIRCUIT_BREAKER_SUSPECT_THRESHOLD` | NO | `1` | Failures before node is suspected |
| `CIRCUIT_BREAKER_SUSPECT_WEIGHT` | NO | `0.5` | Scoring penalty weight for suspected nodes |
| `HEDGING_ENABLED` | NO | `true` | Enable speculative hedged requests |
| `HEDGING_SMALL_MODELS_ONLY` | NO | `true` | Only hedge small model requests |
| `HEDGING_MAX_DELAY_MS` | NO | `800` | Max hedge delay (P95-derived cap) |
| `HEDGING_P95_FRACTION` | NO | `0.6` | Fraction of P95 latency to use as hedge delay |
| `P2C_ALPHA` | NO | `0.5` | P2C scoring weight for P95 latency |
| `P2C_PENALTY_WEIGHT` | NO | `2.0` | P2C scoring weight for recent 5xx rate |
| `MAX_P95_LATENCY_SECS` | NO | `5.0` | Eligibility cap: exclude nodes above this P95 |
| `HEALTH_LATENCY_WINDOW_SECS` | NO | `120` | Rolling window for latency percentile stats |
| `ELIGIBILITY_MIN_P95_SAMPLES` | NO | `20` | Min samples before P95 eligibility check applies |
| `MULTI_EXEC_ENABLED` | NO | `true` | Enable RACE/ALL/SEQUENCE/PLAN/CONSENSUS modes |
| `MULTI_EXEC_MAX_BACKENDS` | NO | `3` | Max backends for multi-exec dispatch |
| `MULTI_EXEC_TIMEOUT_SECS` | NO | `60` | Timeout for multi-exec requests |
| `MULTI_EXEC_CONSENSUS_THRESHOLD` | NO | `0.9` | Similarity threshold for consensus agreement |
| `PLANNER_BACKEND` | NO | — | Backend name or host:port for PLAN decomposition |
| `PLAN_MAX_SUBTASKS` | NO | `5` | Max subtasks per PLAN decomposition |
| `PLAN_SUBTASK_TIMEOUT_SECS` | NO | `30` | Timeout per PLAN subtask |
| `COMPLEXITY_ROUTING_ENABLED` | NO | `false` | Enable complexity-based tier routing |
| `COMPLEXITY_ROUTING_LOG` | NO | `/tmp/complexity_routing.jsonl` | JSONL telemetry file path |
| `STICKY_SESSIONS_ENABLED` | NO | `false` | Enable session affinity via x-session-id |
| `STICKY_TTL_SECS` | NO | `600` | Session sticky TTL in seconds |
| `CROSS_MODEL_FALLBACK` | NO | `false` | Allow fallback to different model families |
| `FALLBACK_CHAINS` | NO | — | `name=backend1>backend2>backend3,...` |
| `FALLBACK_TOTAL_TIMEOUT_SECS` | NO | `120` | Total timeout across fallback chain |
| `DEFAULT_FALLBACK_CHAIN` | NO | — | Default chain name when request fails |
| `BACKEND_COST_PER_TOKEN` | NO | — | JSON: `{"backend_name": 0.000001}` cost per output token |
| `MODEL_SENTINELS` | NO | `auto,default` | Model names that trigger auto-selection |
| `PREFERRED_MODELS` | NO | — | Comma-separated preferred model names |
| `AUTO_MODEL_STRATEGY` | NO | `any_first` | `any_first` or `intersection_first` |
| `AUTO_MIN_NODES` | NO | `2` | Min nodes required for auto-selection |
| `STRICT_AUTO_MODE` | NO | `false` | Fail request if auto-selection finds no candidate |
| `BACKEND_ALIASES` | NO | — | `alias=host:port,...` |
| `BACKEND_CAPABILITIES` | NO | — | `backend=cap1,cap2;...` |
| `SCAN_HOSTS` | NO | — | `host:port,...` for monitor to scan |
| `SCAN_INTERVAL` | NO | `10` | Monitor scan interval in seconds |
| `TOKENS_PER_SEC_LIMIT` | NO | `0` | Global token rate limit (0 = disabled) |
| `FAIRNESS_ENABLED` | NO | `true` | Enable fair scheduling across clients |
| `RATE_LIMIT_BACKOFF_BASE_SECS` | NO | `1.0` | Base backoff for 429 responses |
| `RATE_LIMIT_BACKOFF_MAX_SECS` | NO | `60.0` | Max backoff for 429 responses |
| `RATE_LIMIT_BACKOFF_JITTER` | NO | `0.3` | Backoff jitter factor (0.0–1.0) |

### 2.4 Production env file template

```bash
# /etc/llb/prod.env — managed by host, not committed to git

# Required
CLOUD_BACKENDS=claude=https://api.anthropic.com/v1|/run/secrets/anthropic_api_key|anthropic

# Recommended production values (override defaults)
ROUTING_STRATEGY=P2C
CIRCUIT_BREAKER_THRESHOLD=5
CIRCUIT_BREAKER_COOLDOWN_SECS=90
MIN_HEALTHY_NODES=1
REQUEST_TIMEOUT_SECS=90
HEDGING_ENABLED=true
HEDGING_SMALL_MODELS_ONLY=true
MULTI_EXEC_ENABLED=true
COMPLEXITY_ROUTING_ENABLED=true
COMPLEXITY_ROUTING_LOG=/var/log/llb/complexity_routing.jsonl
PLANNER_BACKEND=claude
CROSS_MODEL_FALLBACK=true
FALLBACK_CHAINS=reliable=cloud:claude(timeout=45)>local:auto(timeout=60)
FALLBACK_TOTAL_TIMEOUT_SECS=120
DEFAULT_FALLBACK_CHAIN=reliable

# Redis (must match redis.prod.conf)
REDIS_HOST=localhost
REDIS_PORT=6379
```

---

## 3. Observability

### 3.1 Prometheus Metrics Inventory

All metrics are stored in Redis and exposed at `GET /metrics` in Prometheus text exposition
format (no `prometheus_client` library — custom renderer in `main.py`).

**Counters** (monotonically increasing, reset only on restart):

| Metric Name | Labels | Description |
|-------------|--------|-------------|
| `ai_lb_requests_total` | — | Total requests handled |
| `ai_lb_failovers_total` | `model`, `node` | Total failover events |
| `ai_lb_rate_limits_total` | `node` | Total 429 responses from backends |
| `ai_lb_hedges_total` | — | Total hedged duplicate attempts |
| `ai_lb_hedge_wins` | `model`, `node` | Hedge wins by model+node |
| `ai_lb_hedge_wins_total` | `model` | Aggregate hedge wins by model |
| `ai_lb_multi_exec_total` | `mode` | Multi-exec requests by mode (race/all/sequence/consensus/plan) |
| `ai_lb_multi_exec_succeeded` | `mode` | Backends that succeeded per multi-exec |
| `ai_lb_consensus_total` | `model` | Consensus requests by model |
| `ai_lb_consensus_agreements` | — | Unanimous consensus count |
| `ai_lb_consensus_disagreements` | — | Non-unanimous consensus count |

**Gauges** (point-in-time values):

| Metric Name | Labels | Description |
|-------------|--------|-------------|
| `ai_lb_up` | `node` | Node health: 1 = healthy, 0 = unhealthy |
| `ai_lb_inflight` | `node` | Current in-flight requests per node |
| `ai_lb_failures` | `node` | Recent failure count per node (within penalty TTL) |
| `ai_lb_rate_limits` | `node` | Rate limit count per node (within window) |

**Histograms** (with `_bucket`, `_sum`, `_count` suffixes):

| Metric Name | Labels | Buckets (seconds) | Description |
|-------------|--------|-------------------|-------------|
| `ai_lb_latency_seconds` | `model`, `node` | 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, +Inf | End-to-end request latency |
| `ai_lb_stream_ttfb_seconds` | `model`, `node` | 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, +Inf | Streaming time-to-first-byte |
| `ai_lb_stream_duration_seconds` | `model`, `node` | 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, +Inf | Total streaming request duration |

**Summary** (from multi-exec backend count distribution):

| Metric Name | Labels | Description |
|-------------|--------|-------------|
| `ai_lb_multi_exec_backends` | `mode` | Backends attempted per multi-exec request |
| `ai_lb_consensus_agreement_count` | — | Distribution of agreement counts |

**Additional metric to add** (assumption: not yet exported; add to metrics renderer):

| Metric Name | Labels | Type | Description |
|-------------|--------|------|-------------|
| `ai_lb_ewma_output_tokens` | `node`, `model` | Gauge | Current EWMA output token count from Redis key `node:{node}:ewma_output_tokens` |
| `ai_lb_circuit_breaker_open` | `node` | Gauge | 1 if CB open (Redis key `lb:circuit:{node}` exists), 0 otherwise |

### 3.2 Prometheus Scrape Config

```yaml
# prometheus.yml — add to scrape_configs
scrape_configs:
  - job_name: 'llb'
    scrape_interval: 15s
    scrape_timeout: 10s
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
    scheme: http
```

### 3.3 Grafana Dashboard Specification

**Dashboard title:** `llb — Production Overview`
**Refresh:** 30s
**Time range default:** Last 1 hour

#### Panels (row order)

**Row 1 — Traffic**

| Panel | Type | PromQL | Legend |
|-------|------|--------|--------|
| Request Rate | Time series | `rate(ai_lb_requests_total[1m])` | `req/s` |
| Error Rate % | Time series | `100 * rate(ai_lb_failovers_total[1m]) / rate(ai_lb_requests_total[1m])` | `error %` |
| In-flight Requests | Time series | `sum by (node) (ai_lb_inflight)` | `{{node}}` |
| Active Nodes | Stat | `count(ai_lb_up == 1)` | `healthy nodes` |

**Row 2 — Latency**

| Panel | Type | PromQL | Legend |
|-------|------|--------|--------|
| Latency P50 | Time series | `histogram_quantile(0.50, sum by (le) (rate(ai_lb_latency_seconds_bucket[5m])))` | `P50` |
| Latency P95 | Time series | `histogram_quantile(0.95, sum by (le) (rate(ai_lb_latency_seconds_bucket[5m])))` | `P95` |
| Latency P99 | Time series | `histogram_quantile(0.99, sum by (le) (rate(ai_lb_latency_seconds_bucket[5m])))` | `P99` |
| TTFB P95 (streaming) | Time series | `histogram_quantile(0.95, sum by (le) (rate(ai_lb_stream_ttfb_seconds_bucket[5m])))` | `TTFB P95` |

**Row 3 — Circuit Breaker & Node Health**

| Panel | Type | PromQL | Legend |
|-------|------|--------|--------|
| CB Open (per node) | State timeline | `ai_lb_circuit_breaker_open` | `{{node}}` |
| Node Up/Down | State timeline | `ai_lb_up` | `{{node}}` |
| Failure Count (per node) | Time series | `ai_lb_failures` | `{{node}}` |
| Rate Limits (per node) | Time series | `rate(ai_lb_rate_limits_total[5m])` | `{{node}}` |

**Row 4 — Cost & EWMA**

| Panel | Type | PromQL | Legend |
|-------|------|--------|--------|
| EWMA Output Tokens | Time series | `ai_lb_ewma_output_tokens` | `{{node}} / {{model}}` |
| Hedge Rate | Time series | `rate(ai_lb_hedges_total[5m])` | `hedges/s` |
| Hedge Win Rate | Time series | `rate(ai_lb_hedge_wins_total[5m])` | `{{model}}` |
| Multi-exec by Mode | Time series | `rate(ai_lb_multi_exec_total[5m])` | `{{mode}}` |

### 3.4 Alerting Rules

```yaml
# alerting/llb.yml
groups:
  - name: llb
    interval: 30s
    rules:

      # ── Error rate ────────────────────────────────────────────────────────
      - alert: AILBHighErrorRate
        expr: |
          (
            rate(ai_lb_failovers_total[2m]) /
            rate(ai_lb_requests_total[2m])
          ) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "llb error rate above 5% for 2+ minutes"
          description: >
            Error rate is {{ printf "%.1f" (mul 100 $value) }}% over the last 2 minutes.
            Check backend health and circuit breaker state.
          runbook: "Section 4.5 — Circuit Breaker Manual Reset"

      # ── Circuit breaker ───────────────────────────────────────────────────
      - alert: AILBCircuitBreakerOpen
        expr: ai_lb_circuit_breaker_open == 1
        for: 60s
        labels:
          severity: warning
        annotations:
          summary: "Circuit breaker open on node {{ $labels.node }} for >60s"
          description: >
            Node {{ $labels.node }} circuit breaker has been open for over 60 seconds.
            The node may be permanently degraded or has not recovered within cooldown period.
          runbook: "Section 4.5 — Circuit Breaker Manual Reset"

      # ── All nodes down ────────────────────────────────────────────────────
      - alert: AILBAllNodesUnhealthy
        expr: count(ai_lb_up == 1) == 0
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "CRITICAL: All llb backend nodes are unhealthy"
          description: >
            No healthy nodes available. llb is unable to serve requests.
            All requests will return 503.
          runbook: "Section 4.1 — Cold Start"

      # ── EWMA token cost anomaly ────────────────────────────────────────────
      # Anomaly: a node's EWMA output tokens exceeds 3x the median across nodes.
      # This detects runaway verbose responses or model misbehavior on a specific node.
      - alert: AILBEwmaTokenCostAnomaly
        expr: |
          (
            ai_lb_ewma_output_tokens
            /
            quantile(0.5, ai_lb_ewma_output_tokens)
          ) > 3
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "EWMA token output anomaly on node {{ $labels.node }}"
          description: >
            Node {{ $labels.node }} EWMA output tokens is {{ printf "%.0f" $value }}x
            the median across nodes for model {{ $labels.model }}.
            May indicate verbose backend responses inflating cost estimates.

      # ── No traffic (dead man's switch) ───────────────────────────────────
      - alert: AILBNoTraffic
        expr: rate(ai_lb_requests_total[5m]) == 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "llb has received no requests in 5 minutes"
          description: >
            No requests have been routed through llb in the last 5 minutes.
            Check if the service is reachable and if upstream clients are healthy.

      # ── High P99 latency ─────────────────────────────────────────────────
      - alert: AILBHighP99Latency
        expr: |
          histogram_quantile(0.99,
            sum by (le) (rate(ai_lb_latency_seconds_bucket[5m]))
          ) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "llb P99 latency above 10 seconds"
          description: >
            P99 request latency is {{ printf "%.1f" $value }}s, exceeding 10s threshold.
            Check backend node latencies and REQUEST_TIMEOUT_SECS config.
```

---

## 4. Operational Runbook

### 4.1 Cold Start (Full Startup from Zero)

**Prerequisites:** Docker installed, `/etc/llb/prod.env` exists, `/run/secrets/` populated,
`/var/lib/llb/redis/` directory exists.

1. Create required directories:
   ```bash
   sudo mkdir -p /var/lib/llb/redis /var/log/llb /etc/llb
   sudo chown -R 1001:root /var/log/llb
   ```
2. Populate secrets:
   ```bash
   echo "sk-ant-api03-XXXX" | sudo tee /run/secrets/anthropic_api_key > /dev/null
   sudo chmod 600 /run/secrets/anthropic_api_key
   ```
3. Write `/etc/llb/prod.env` from the template in Section 2.4.
4. Copy Redis config:
   ```bash
   sudo cp config/redis.prod.conf /etc/llb/redis.prod.conf
   ```
5. Pull/build images:
   ```bash
   docker build -f load_balancer/Dockerfile.prod -t localhost:5000/llb:latest ./load_balancer
   docker build -f monitor/Dockerfile -t localhost:5000/llb-monitor:latest ./monitor
   ```
6. Start Redis first and wait for healthy:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d redis
   docker compose -f docker-compose.yml -f docker-compose.prod.yml wait redis
   ```
7. Start the load balancer:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d load_balancer
   ```
8. Verify health (must succeed within 30s):
   ```bash
   for i in $(seq 1 30); do
     resp=$(curl -sf http://localhost:8000/health) && echo "Healthy: $resp" && break
     echo "Attempt $i: waiting..."; sleep 1
   done
   ```
9. Start monitor (optional, for local node scanning):
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile full up -d monitor
   ```
10. Verify metrics endpoint:
    ```bash
    curl -s http://localhost:8000/metrics | grep ai_lb_requests_total
    ```
11. Confirm config validation passed (check logs for errors):
    ```bash
    docker logs ai_lb_load_balancer 2>&1 | grep -E "Config validation|ConfigError|MISSING"
    ```

### 4.2 Rolling Update (Zero-Downtime)

**Precondition:** At least one healthy backend node is available. New image is built and tagged.

1. Build the new image:
   ```bash
   docker build -f load_balancer/Dockerfile.prod \
     -t localhost:5000/llb:candidate \
     ./load_balancer
   ```
2. Smoke-test the candidate image locally (non-prod port):
   ```bash
   docker run --rm -d --name ai_lb_candidate \
     --env-file /etc/llb/prod.env \
     -p 18000:8000 \
     localhost:5000/llb:candidate
   curl -sf http://localhost:18000/health && echo "Candidate OK"
   docker stop ai_lb_candidate
   ```
3. Tag candidate as latest:
   ```bash
   docker tag localhost:5000/llb:candidate localhost:5000/llb:latest
   ```
4. Pull the new image into the compose stack (no restart yet):
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.prod.yml pull load_balancer
   ```
5. Drain in-flight requests (wait for inflight gauge to reach 0 or timeout 60s):
   ```bash
   for i in $(seq 1 60); do
     inflight=$(curl -s http://localhost:8000/metrics | grep "^ai_lb_inflight" | awk '{sum+=$2} END {print sum}')
     [ "$inflight" = "0" ] && echo "Drained" && break
     echo "Inflight: $inflight — waiting..."; sleep 1
   done
   ```
6. Restart the load_balancer container:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.prod.yml \
     up -d --no-deps --force-recreate load_balancer
   ```
7. Verify health within 30s:
   ```bash
   for i in $(seq 1 30); do
     curl -sf http://localhost:8000/health && echo "Healthy" && break
     echo "Attempt $i..."; sleep 1
   done
   ```
8. Verify no error spike in Prometheus (check `AILBHighErrorRate` alert is not firing).
9. Remove the candidate tag:
   ```bash
   docker rmi localhost:5000/llb:candidate
   ```

### 4.3 Rollback Procedure

**Trigger:** New version fails health check or error rate spikes above 5% after rolling update.

1. Identify the previous image digest:
   ```bash
   docker images localhost:5000/llb --digests
   # Note the sha256 of the previous working image
   ```
2. Tag the previous image as latest:
   ```bash
   docker tag localhost:5000/llb@sha256:<previous-digest> localhost:5000/llb:latest
   ```
3. Force-recreate the container with the rolled-back image:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.prod.yml \
     up -d --no-deps --force-recreate load_balancer
   ```
4. Verify health:
   ```bash
   curl -sf http://localhost:8000/health
   ```
5. Verify error rate has returned to baseline:
   ```bash
   curl -s http://localhost:8000/metrics | grep ai_lb_failovers_total
   ```
6. Document the rollback reason and failed image tag in the incident log.

### 4.4 Node Add/Remove (Dynamic Backend Membership)

**Context:** Backend nodes are discovered via the monitor service or explicitly declared via
`CLOUD_BACKENDS`. Node state is tracked in Redis sets (`nodes:healthy`, `node:{addr}:*`).

**Adding a node:**

1. Start the new LLM backend server on the target host (e.g., LMStudio on port 1234).
2. If using monitor (local scan): ensure `SCAN_HOSTS` includes the new host:port.
   Update `/etc/llb/prod.env` and restart monitor:
   ```bash
   # Edit /etc/llb/prod.env: append new host to SCAN_HOSTS=...,newhost:1234
   docker compose -f docker-compose.yml -f docker-compose.prod.yml \
     --profile full restart monitor
   ```
3. If using cloud backends: add the new backend to `CLOUD_BACKENDS` in `/etc/llb/prod.env`:
   ```bash
   # Append: ,newbackend=https://newhost/v1|/run/secrets/newbackend_key|openai
   docker compose -f docker-compose.yml -f docker-compose.prod.yml \
     up -d --no-deps --force-recreate load_balancer
   ```
4. Verify the new node appears as healthy:
   ```bash
   docker exec ai_lb_redis redis-cli smembers nodes:healthy
   curl -s http://localhost:8000/v1/nodes | python3 -m json.tool
   ```
5. Verify `ai_lb_up{node="newhost:port"}` gauge equals 1 in `/metrics`.

**Removing a node:**

1. Stop the backend server on the target host.
2. Remove the node from `SCAN_HOSTS` or `CLOUD_BACKENDS` in `/etc/llb/prod.env`.
3. Manually evict the node from Redis to prevent stale routing (do this before restart):
   ```bash
   NODE="oldhost:1234"
   docker exec ai_lb_redis redis-cli srem nodes:healthy "$NODE"
   docker exec ai_lb_redis redis-cli del "node:${NODE}:inflight" \
     "node:${NODE}:failures" "lb:circuit:${NODE}" "penalty:${NODE}"
   ```
4. Restart load_balancer to pick up config change:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.prod.yml \
     up -d --no-deps --force-recreate load_balancer
   ```
5. Verify removed node no longer appears in `/v1/nodes`.

### 4.5 Circuit Breaker Manual Reset

**When to use:** A backend node has recovered but the CB cooldown has not yet elapsed, or
automatic reset is not occurring due to a bug.

1. Identify the open circuit breaker:
   ```bash
   docker exec ai_lb_redis redis-cli keys "lb:circuit:*"
   # Output: lb:circuit:hostname:port
   ```
2. Check when it expires:
   ```bash
   docker exec ai_lb_redis redis-cli ttl "lb:circuit:hostname:port"
   # -1 = no TTL (stuck); positive = seconds remaining
   ```
3. Verify the backend is actually healthy before resetting:
   ```bash
   curl -sf http://hostname:port/health || curl -sf http://hostname:port/v1/models
   ```
4. If confirmed healthy, delete the CB key:
   ```bash
   docker exec ai_lb_redis redis-cli del "lb:circuit:hostname:port"
   ```
5. Also clear the penalty key if present:
   ```bash
   docker exec ai_lb_redis redis-cli del "penalty:hostname:port"
   ```
6. Verify the node returns to healthy set:
   ```bash
   docker exec ai_lb_redis redis-cli sismember nodes:healthy "hostname:port"
   # Expected: 1
   ```
7. Verify `ai_lb_circuit_breaker_open{node="hostname:port"}` returns 0 in `/metrics`.
8. Monitor for 2 minutes to confirm the node stays healthy and CB does not re-open.

### 4.6 Horizontal Scaling (Multiple llb Instances Behind HAProxy)

**Architecture:** Multiple llb containers share the same Redis instance. All state (node
health, metrics, circuit breakers, EWMA) is Redis-backed, so instances are stateless.

**HAProxy configuration:**

```haiku
# haproxy.cfg
global
    log 127.0.0.1 local0
    maxconn 10000

defaults
    mode http
    timeout connect 5s
    timeout client 90s
    timeout server 90s
    option httplog
    option forwardfor
    option http-server-close

frontend ai_lb_frontend
    bind :80
    default_backend ai_lb_pool

backend ai_lb_pool
    balance leastconn
    option httpchk GET /health
    http-check expect status 200
    server ailb1 127.0.0.1:8000 check inter 10s fall 3 rise 2
    server ailb2 127.0.0.1:8001 check inter 10s fall 3 rise 2
    server ailb3 127.0.0.1:8002 check inter 10s fall 3 rise 2
```

**Procedure to add a second llb instance:**

1. Start a second load_balancer container on port 8001 (same Redis, same prod.env):
   ```bash
   docker run -d \
     --name ai_lb_load_balancer_2 \
     --network host \
     --env-file /etc/llb/prod.env \
     -e REDIS_HOST=localhost \
     -e REDIS_PORT=6379 \
     --restart always \
     localhost:5000/llb:latest \
     uvicorn load_balancer.main:app --host 0.0.0.0 --port 8001 --workers 2
   ```
2. Verify the new instance is healthy:
   ```bash
   curl -sf http://localhost:8001/health
   ```
3. Add the new instance to HAProxy `ai_lb_pool` and reload HAProxy:
   ```bash
   sudo haproxy -f /etc/haproxy/haproxy.cfg -c   # validate config
   sudo systemctl reload haproxy
   ```
4. Verify HAProxy is routing to both instances:
   ```bash
   echo "show info" | sudo socat stdio /run/haproxy/admin.sock | grep "CurrConns"
   ```

**Note on state sharing:** All llb instances read/write the same Redis keys. Circuit breaker
state, node health, EWMA tokens, and metric counters are visible across all instances. No
instance-local state is maintained beyond in-process async tasks (hedging, plan execution).

---

## Acceptance Criteria

- [ ] **AC1 — Container health check within 30s:** `docker run` + `HEALTHCHECK` probe returns
  healthy status within 30 seconds of container start. Verified by `docker inspect
  ai_lb_load_balancer --format='{{.State.Health.Status}}'` returning `healthy`.

- [ ] **AC2 — Boot validation with clear errors:** Remove a required env var (e.g. unset both
  `CLOUD_BACKENDS` and `SCAN_HOSTS`), start the container, and observe `sys.exit(1)` with a
  message matching `[MISSING REQUIRED CONFIG]` in container logs within 5 seconds.

- [ ] **AC3 — Prometheus scrape returns all defined metrics:** `curl http://localhost:8000/metrics`
  returns all metrics from Section 3.1, including `ai_lb_circuit_breaker_open` and
  `ai_lb_ewma_output_tokens`. Validated by checking each metric name is present in the response.

- [ ] **AC4 — Alerting rules fire on simulated failures:**
  - Inject 10 errors into Redis counter (`docker exec ai_lb_redis redis-cli incrby lb:failovers_total 100`)
    and verify `AILBHighErrorRate` alert fires in Prometheus Alertmanager within 2 minutes.
  - Set a CB key with no TTL (`docker exec ai_lb_redis redis-cli set lb:circuit:testnode:1234 1`)
    and verify `AILBCircuitBreakerOpen` alert fires within 90 seconds.
  - Delete all members from `nodes:healthy` and verify `AILBAllNodesUnhealthy` fires immediately.

- [ ] **AC5 — Runbook completeness:** Each scenario in Section 4 has been executed in a test
  environment (staging or local docker-compose) and the numbered steps produce the expected
  outcome without requiring improvisation.
