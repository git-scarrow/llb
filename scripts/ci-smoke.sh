#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

# Source .env for LLB_PORT if not already set
[[ -f "$ROOT_DIR/.env" ]] && set -a && source "$ROOT_DIR/.env" && set +a
LLB_PORT="${LLB_PORT:-${AI_LB_PORT:-8002}}"  # COMPAT: AI_LB_PORT fallback remove after 2026-06-01

export COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1

MODEL_ID="m"
LB_URL="http://localhost:${LLB_PORT}"
STUB_PORT=9999
STUB_PID=""

function cleanup() {
  [[ -n "$STUB_PID" ]] && kill "$STUB_PID" 2>/dev/null || true
  docker compose down -v || true
}
trap cleanup EXIT

echo "[ci-smoke] Bringing up stack..."
ROUTING_STRATEGY=ROUND_ROBIN \
STICKY_SESSIONS_ENABLED=true \
SCAN_HOSTS='' \
HEDGING_SMALL_MODELS_ONLY=false \
HEDGING_MAX_DELAY_MS=0 \
docker compose up -d --build

echo "[ci-smoke] Waiting for LB to respond..."
for i in {1..120}; do
  if curl -sf "$LB_URL/metrics" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "[ci-smoke] Starting stub backend on port $STUB_PORT..."
python3 - <<'PYEOF' &
import http.server, json, sys

RESP = json.dumps({
    "id": "chatcmpl-stub", "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
}).encode()

MODELS = json.dumps({"object": "list", "data": [{"id": "m", "object": "model"}]}).encode()

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(MODELS)
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        self.rfile.read(n)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(RESP)

http.server.HTTPServer(("0.0.0.0", 9999), H).serve_forever()
PYEOF
STUB_PID=$!
# Wait for stub to be ready
for i in {1..10}; do
  if curl -sf "http://localhost:$STUB_PORT/v1/models" >/dev/null 2>&1; then break; fi
  sleep 1
done

# node1 is a fake address — it only needs to hit the capacity check (Redis),
# never an actual HTTP connection. node2 is the stub on localhost:9999.
NODE1="node1:9998"
NODE2="localhost:9999"

echo "[ci-smoke] Seeding Redis model/node state..."
docker compose exec -T redis redis-cli SADD nodes:healthy "$NODE1" "$NODE2" >/dev/null
docker compose exec -T redis redis-cli SET "node:$NODE1:models" '{"object":"list","data":[{"id":"'"$MODEL_ID"'","object":"model"}]}' >/dev/null
docker compose exec -T redis redis-cli SET "node:$NODE2:models" '{"object":"list","data":[{"id":"'"$MODEL_ID"'","object":"model"}]}' >/dev/null
docker compose exec -T redis redis-cli SET "session:s:$MODEL_ID" "$NODE1" >/dev/null

echo "[ci-smoke] Forcing capacity on node1; freeing node2..."
docker compose exec -T redis redis-cli SET "node:$NODE1:maxconn" 1 >/dev/null
docker compose exec -T redis redis-cli SET "node:$NODE1:inflight" 1 >/dev/null
docker compose exec -T redis redis-cli DEL "node:$NODE2:maxconn" >/dev/null
docker compose exec -T redis redis-cli SET "node:$NODE2:inflight" 0 >/dev/null

round=1
ok=0
while (( round <= 3 )); do
  echo "[ci-smoke] Round $round: firing 10 non-stream requests to trigger hedging..."
  for i in $(seq 1 10); do
    curl -s -o /dev/null -X POST "$LB_URL/v1/chat/completions" \
      -H 'content-type: application/json' \
      -H 'x-session-id: s' \
      --data '{"model":"'"$MODEL_ID"'","messages":[{"role":"user","content":"hedge attempt '"$i"' (round '"$round"')"}],"max_tokens":8,"stream":false}' || true
  done
  sleep 2
  echo "[ci-smoke] Reading metrics..."
  # Poll metrics up to 10s for counters to appear
  METRICS=""
  for j in {1..10}; do
    METRICS="$(curl -s "$LB_URL/metrics" || true)"
    if [[ -n "$METRICS" ]]; then
      break
    fi
    sleep 1
  done
  REQS=$(echo "$METRICS" | awk '/^llb_requests_total /{print $2}')
  HEDGES=$(echo "$METRICS" | awk '/^llb_hedges_total /{print $2}')
  WINS_MODEL=$(echo "$METRICS" | awk -v m="$MODEL_ID" '$1 ~ /^llb_hedge_wins_total/ && $0 ~ "model=\""m"\"" {print $2}')

  echo "[ci-smoke] llb_requests_total=${REQS:-NA}"
  echo "[ci-smoke] llb_hedges_total=${HEDGES:-0}"
  echo "[ci-smoke] llb_hedge_wins_total{model=\"$MODEL_ID\"}=${WINS_MODEL:-0}"

  if [[ -n "$HEDGES" ]] && (( ${HEDGES%%.*} >= 1 )) && [[ -n "${WINS_MODEL:-}" ]] && (( ${WINS_MODEL%%.*} >= 1 )); then
    ok=1
    break
  fi
  round=$((round+1))
done

if (( ok == 1 )); then
  echo "[ci-smoke] OK: Hedging counters present and > 0"
else
  echo "[ci-smoke] FAIL: Hedging counters not observed after retries" >&2
  exit 1
fi
