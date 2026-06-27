# Load Balancer Service

This service is the main user-facing entry point. It receives requests and routes them to the appropriate LLM node based on the configured routing strategy.

## Features

- Auto model resolution for `chat.completions` and `embeddings`:
  - Use `model: "auto"` (or any sentinel in `MODEL_SENTINELS`).
  - `?require_all=true` or `AUTO_MODEL_STRATEGY=intersection_first` prefers models available on all healthy nodes.
  - Honors `PREFERRED_MODELS` ordering.
- Sticky sessions:
  - Provide `x-session-id` to prefer the same node for subsequent requests of the same model.
  - TTL controlled by `STICKY_TTL_SECS`.
- Backpressure:
  - Returns `429` with `Retry-After` when all eligible nodes are saturated.
- Observability headers (non-stream and stream initial response):
  - `x-request-id`, `x-selected-model`, `x-routed-node`, `x-attempts`, `x-failover-count`.
  - `x-capacity-state`: `ok` | `model_saturated` | `cluster_saturated`.
  - `x-model-defaulted`: `true` when the balancer substitutes a configured default model.
- Streaming SSE meta events:
  - `event: meta` on first routing; `event: failover` on each failover.
  - Hedging signals (streaming only):
    - `event: hedge_start` with `{request_id, model, primary, secondary}` when a duplicate attempt is launched.
    - `event: hedge_winner` with `{request_id, model, primary, secondary, winner}` when one stream wins the race.
- Metrics at `/metrics` (Prometheus text):
  - `llb_requests_total` counter.
  - `llb_up{node}` gauge; `llb_inflight{node}` gauge; `llb_failures{node}` gauge.
  - `llb_failovers_total` and `llb_failovers_total{model}` counters.
  - `llb_latency_seconds_*{model,node}` histogram with buckets: 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, +Inf.
  - Streaming histograms: `llb_stream_ttfb_seconds_*`, `llb_stream_duration_seconds_*` per model/node.
  - Hedging counters: `llb_hedges_total`, `llb_hedge_wins{model,node}`, and aggregate `llb_hedge_wins_total{model}`.

## Headers and SSE

- Request
  - `x-request-id` (optional): propagated end-to-end; generated if absent.
  - `x-session-id` (optional): enables sticky routing for that session+model.
- Response
  - Always includes `x-request-id`. On success: `x-selected-model`, `x-routed-node`.
  - Non-stream: also `x-attempts`, `x-failover-count`.
  - `x-model-defaulted` flags when `DEFAULT_*_MODEL` filled a missing `model` field.
  - 429 includes `Retry-After` seconds.
- Streaming SSE prelude
  - `event: meta` with `{request_id, model, node, attempts: 1, failover_count: 0}`.
  - On failover: `event: failover` with updated `{attempts, failover_count, node}`.

## Environment

- Routing/auto
  - `ROUTING_STRATEGY`: `ROUND_ROBIN` | `RANDOM` | `LEAST_LOADED`.
  - `MODEL_SENTINELS`: comma list, default `auto,default`.
  - `LM_MODEL`: default `auto`.
  - `PREFERRED_MODELS`: comma/semicolon list, optional.
  - `AUTO_MODEL_STRATEGY`: `any_first` (default) | `intersection_first`.
  - `DEFAULT_CHAT_MODEL`: optional concrete model used when chat requests omit `model`.
  - `DEFAULT_EMBEDDINGS_MODEL`: optional concrete model used when embeddings requests omit `model`.
- Timeouts/retries
  - `REQUEST_TIMEOUT_SECS` (default 60), `MAX_RETRIES` (default 2).
  - `RETRY_AFTER_SECS` (default 2) for 429.
- Circuit/backpressure
  - `CIRCUIT_BREAKER_THRESHOLD` (default 3), `CIRCUIT_BREAKER_TTL_SECS` (default 30).
  - Optional per-node `maxconn` via Redis key `node:{host}:maxconn`.
- Sticky sessions
  - `STICKY_SESSIONS_ENABLED` (default false), `STICKY_TTL_SECS` (default 600).
  
- Hedging (latency tail mitigation)
  - `HEDGING_ENABLED` (default true): enable duplicate attempt hedging.
  - `HEDGING_SMALL_MODELS_ONLY` (default true): restrict hedging to models in `MODEL_CLASSES.historical_small.candidates`.
  - `HEDGING_P95_FRACTION` (default 0.6): start hedge after `fraction * p95(model|pool)` delay.
  - `HEDGING_MAX_DELAY_MS` (default 800): cap hedge delay; 0 forces immediate hedging (useful for tests).
  - Non-streaming: winner reflected via headers `x-hedged=true` and `x-hedge-winner`.
  - Streaming: winners announced via SSE events `hedge_start`/`hedge_winner` (headers are static at start).

### Quick smoke: forced hedging

1) Bring up the stack with two fake nodes and aggressive hedging.
   - `ROUTING_STRATEGY=RANDOM HEDGING_MAX_DELAY_MS=0 HEDGING_SMALL_MODELS_ONLY=false docker compose --profile full up -d`
2) Seed Redis to advertise the model on both nodes and mark both healthy.
   - `redis-cli SADD nodes:healthy llb_node1:9999 llb_node2:9999`
   - `redis-cli SET node:llb_node1:9999:models '{"object":"list","data":[{"id":"m","object":"model"}]}'`
   - `redis-cli SET node:llb_node2:9999:models '{"object":"list","data":[{"id":"m","object":"model"}]}'`
3) Saturate node1 only to force hedging to node2:
   - `redis-cli SET node:llb_node1:9999:maxconn 1`
   - `redis-cli SET node:llb_node1:9999:inflight 1`
4) Stream a request and observe SSE events:
   - `curl -N -s http://localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{"model":"m","messages":[{"role":"user","content":"ping"}],"stream":true}' | grep -E 'event: hedge_|data:'`
5) Check metrics:
   - `/metrics` should show `llb_hedges_total > 0` and `llb_hedge_wins{model="m",node="llb_node2:9999"} > 0`.

## Admin API

- `POST /v1/admin/prefs` body (all fields optional):
  - `preferred_models`: `["m1","m2",...]` — sets runtime priority for auto selection.
  - `model_weights`: `{ "m1": 1.0, "m2": 0.9 }` — biases auto selection when not using preferred list.
  - `model_caps`: `{ "m1": 4 }` — per-model concurrency caps across the cluster.
  - `node_caps`: `{ "host:port": 2 }` — per-node concurrency caps.
  - `auto_model_strategy`: `"any_first" | "intersection_first"` — runtime toggle.

Notes:
- Admin updates persist in Redis and take effect immediately; environment defaults remain fallback.
- Per-model caps drive 429 `x-capacity-state: model_saturated` when hit.

## Notes

- Streaming headers reflect the initial routed node only; if a failover occurs mid-stream, headers won’t change (see SSE events for failovers).
- Request validation
  - `chat.completions` requires either `messages`, `prompt`, or `input`; otherwise a `400` is returned with `{"missing": [...]}`.
  - `embeddings` requires `input`.
  - When defaults are configured, missing `model` fields are filled automatically and logged for observability.
