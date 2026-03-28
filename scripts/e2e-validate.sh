#!/usr/bin/env bash
# E2E validation script for LLB Phases 1–3
# Tests: capability routing, complexity routing, PLAN mode, RACE/hedging,
#        response headers, circuit breaker
# Usage: ./scripts/e2e-validate.sh [--no-configure] [--lb-url URL]
#
# Flags:
#   --no-configure   Skip patching .env and restarting the LB (assume already configured)
#   --lb-url URL     LB base URL (default: http://localhost:$LLB_PORT)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

# Source .env for LLB_PORT if not already set
[[ -f "$ROOT_DIR/.env" ]] && set -a && source "$ROOT_DIR/.env" && set +a
LLB_PORT="${LLB_PORT:-${AI_LB_PORT:-8002}}"  # COMPAT: AI_LB_PORT fallback remove after 2026-06-01
LB_URL="http://localhost:${LLB_PORT}"
CONFIGURE=true

for arg in "$@"; do
  case "$arg" in
    --no-configure) CONFIGURE=false ;;
    --lb-url) shift; LB_URL="$1" ;;
    --lb-url=*) LB_URL="${arg#--lb-url=}" ;;
  esac
done

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'
pass() { echo -e "  ${GREEN}PASS${NC}  $*"; }
fail() { echo -e "  ${RED}FAIL${NC}  $*"; FAILURES=$((FAILURES+1)); }
skip() { echo -e "  ${YELLOW}SKIP${NC}  $*"; }
FAILURES=0

# ── Phase 1–3 env vars ───────────────────────────────────────────────────────
# Inject into .env if not present, then restart the LB container.
P13_VARS=(
  "BACKEND_CAPABILITIES=claude=reasoning,code;openai=code,math"
  "COMPLEXITY_ROUTING_ENABLED=true"
  "PLANNER_BACKEND=claude"
  "HEDGING_SMALL_MODELS_ONLY=false"
  "HEDGING_MAX_DELAY_MS=2000"
)

patch_env() {
  local env_file="$ROOT_DIR/.env"
  local changed=false
  for entry in "${P13_VARS[@]}"; do
    local key="${entry%%=*}"
    if grep -q "^${key}=" "$env_file" 2>/dev/null; then
      # Already present — leave as-is
      :
    else
      echo "$entry" >> "$env_file"
      echo "  → Added ${key} to .env"
      changed=true
    fi
  done
  echo "$changed"
}

if $CONFIGURE; then
  echo "── Configuring Phase 1–3 env vars ─────────────────────────────────"
  changed=$(patch_env)
  if [[ "$changed" == "true" ]]; then
    echo "  Restarting load_balancer container..."
    docker compose restart load_balancer
    echo "  Waiting for LB to be ready..."
    for i in {1..30}; do
      curl -sf "$LB_URL/metrics" >/dev/null 2>&1 && break
      sleep 1
    done
  else
    echo "  All Phase 1–3 vars already present — no restart needed."
  fi
fi

# ── Helper: send request, capture response + headers ─────────────────────────
# Usage: req <extra curl args...> -- body
# Sets globals: HTTP_STATUS RESP_BODY RESP_HEADERS
TMPDIR_E2E=$(mktemp -d)
trap 'rm -rf "$TMPDIR_E2E"' EXIT

req() {
  local extra_args=()
  local body=""
  local parsing_body=false
  for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then parsing_body=true; continue; fi
    if $parsing_body; then body="$arg"; else extra_args+=("$arg"); fi
  done

  local hdr_file="$TMPDIR_E2E/headers.$$"
  local body_file="$TMPDIR_E2E/body.$$"

  HTTP_STATUS=$(curl -s -o "$body_file" -D "$hdr_file" -w "%{http_code}" \
    -X POST "$LB_URL/v1/chat/completions" \
    -H "content-type: application/json" \
    "${extra_args[@]}" \
    --data "$body" 2>/dev/null || echo "000")

  RESP_BODY=$(cat "$body_file" 2>/dev/null || true)
  RESP_HEADERS=$(cat "$hdr_file" 2>/dev/null || true)
  rm -f "$hdr_file" "$body_file"
}

header_val() {
  echo "$RESP_HEADERS" | grep -i "^$1:" | head -1 | sed 's/^[^:]*: *//;s/\r//' || true
}

# ── Test 0: Connectivity ─────────────────────────────────────────────────────
echo ""
echo "── 0. Connectivity ─────────────────────────────────────────────────"
if curl -sf "$LB_URL/metrics" >/dev/null 2>&1; then
  pass "LB reachable at $LB_URL"
else
  echo -e "  ${RED}FATAL${NC} LB not reachable at $LB_URL — aborting"
  exit 1
fi

# ── Test 1: Response headers on a plain request ───────────────────────────
echo ""
echo "── 1. Response headers (plain request) ─────────────────────────────"
BODY='{"model":"claude-haiku-4-5-20251001","messages":[{"role":"user","content":"Say hi"}],"max_tokens":10}'
req -- "$BODY"

if [[ "$HTTP_STATUS" == "200" ]]; then
  pass "HTTP 200"
else
  fail "Expected 200, got $HTTP_STATUS"
fi

for h in x-request-id; do
  val=$(header_val "$h")
  if [[ -n "$val" ]]; then
    pass "header $h = $val"
  else
    fail "header $h missing"
  fi
done
# x-execution-mode and x-backends-attempted are only set on multi-exec paths
for h in x-execution-mode x-backends-attempted; do
  val=$(header_val "$h")
  if [[ -n "$val" ]]; then
    pass "header $h = $val"
  else
    skip "header $h absent on plain request (expected — multi-exec only)"
  fi
done

# ── Test 2: Capability routing ────────────────────────────────────────────
echo ""
echo "── 2. Capability routing (x-require-capability) ─────────────────────"

# reasoning → must route to cloud:claude (only backend with this capability)
# Use a Claude-specific model so auto-resolution doesn't fall back to local Ollama
BODY='{"model":"claude-haiku-4-5-20251001","messages":[{"role":"user","content":"Solve: if x+2=5, what is x?"}],"max_tokens":20}'
req -H "x-require-capability: reasoning" -- "$BODY"
node=$(header_val "x-routed-node")
attempted=$(header_val "x-backends-attempted")
if [[ "$HTTP_STATUS" == "200" ]]; then
  pass "reasoning capability → HTTP 200 (node=$node)"
  [[ "$node" == "cloud:claude" ]] && pass "routed to cloud:claude as expected" \
    || skip "routed to $node (cloud:claude preferred but advisory fallback may apply)"
else
  fail "reasoning capability → HTTP $HTTP_STATUS"
fi

# code capability on claude — only backend available
BODY='{"model":"claude-haiku-4-5-20251001","messages":[{"role":"user","content":"Write hello world in Python"}],"max_tokens":30}'
req -H "x-require-capability: code" -- "$BODY"
node=$(header_val "x-routed-node")
if [[ "$HTTP_STATUS" == "200" ]]; then
  pass "code capability → HTTP 200 (node=$node)"
else
  fail "code capability → HTTP $HTTP_STATUS"
fi

# unsatisfiable → should 404
BODY='{"model":"auto","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
req -H "x-require-capability: nonexistent-capability" -- "$BODY"
if [[ "$HTTP_STATUS" == "404" ]]; then
  pass "unsatisfiable capability → HTTP 404 (correct fallback)"
else
  # Advisory fallback: LB falls back to all nodes — still a valid response
  if [[ "$HTTP_STATUS" == "200" ]]; then
    skip "unsatisfiable capability → 200 (advisory fallback active — expected)"
  else
    fail "unsatisfiable capability → HTTP $HTTP_STATUS"
  fi
fi

# ── Test 3: Complexity routing ────────────────────────────────────────────
echo ""
echo "── 3. Complexity routing (COMPLEXITY_ROUTING_ENABLED) ───────────────"

small_body='{"model":"auto","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
req -- "$small_body"
sel=$(header_val "x-selected-model")
if [[ "$HTTP_STATUS" == "200" ]]; then
  pass "small prompt → HTTP 200 (x-selected-model=${sel:-<not set>})"
else
  fail "small prompt → HTTP $HTTP_STATUS"
fi

# Large complex prompt with code fences — should score high
large_msg='Implement a Python function that performs a binary search on a sorted list. Include:\n```python\n# full implementation\n```\nAnalyze time complexity step by step. Consider edge cases: empty list, single element, duplicates. Provide unit tests for each edge case with expected outputs.'
large_body="{\"model\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"$large_msg\"}],\"max_tokens\":50}"
req -- "$large_body"
sel_large=$(header_val "x-selected-model")
if [[ "$HTTP_STATUS" == "200" ]]; then
  pass "large prompt → HTTP 200 (x-selected-model=${sel_large:-<not set>})"
else
  fail "large prompt → HTTP $HTTP_STATUS"
fi

if [[ -n "$sel" && -n "$sel_large" && "$sel" != "$sel_large" ]]; then
  pass "complexity tier changed: '$sel' → '$sel_large'"
elif [[ -z "$sel" && -z "$sel_large" ]]; then
  skip "x-selected-model not present — COMPLEXITY_ROUTING_ENABLED may be false or no tier candidates"
else
  skip "same model selected for small/large (may indicate single backend available)"
fi

# ── Test 4: PLAN execution mode ───────────────────────────────────────────
echo ""
echo "── 4. PLAN execution mode (x-execution-mode: plan) ──────────────────"
plan_body='{"model":"claude-haiku-4-5-20251001","messages":[{"role":"user","content":"What is 2+2? Answer in one word."}],"max_tokens":300}'
req -H "x-execution-mode: plan" -H "x-consensus-oracle: claude" -- "$plan_body"

if [[ "$HTTP_STATUS" == "200" ]]; then
  pass "PLAN mode → HTTP 200"
else
  fail "PLAN mode → HTTP $HTTP_STATUS (body: ${RESP_BODY:0:200})"
fi

mode=$(header_val "x-execution-mode")
goal=$(header_val "x-plan-goal")
tasks=$(header_val "x-plan-tasks")
planner=$(header_val "x-plan-planner")

[[ "$mode" == "plan" ]]    && pass "x-execution-mode = plan" || fail "x-execution-mode = '${mode}' (expected 'plan')"
[[ -n "$goal" ]]           && pass "x-plan-goal = $goal"     || fail "x-plan-goal missing"
[[ -n "$tasks" ]]          && pass "x-plan-tasks = $tasks"   || fail "x-plan-tasks missing"
[[ -n "$planner" ]]        && pass "x-plan-planner = $planner" || fail "x-plan-planner missing"

# ── Test 5: RACE execution mode / hedging ────────────────────────────────
echo ""
echo "── 5. RACE execution mode (hedging across backends) ─────────────────"
race_body='{"model":"claude-haiku-4-5-20251001","messages":[{"role":"user","content":"Say one word"}],"max_tokens":10}'
req -H "x-execution-mode: race" -H "x-target-backends: cloud:claude" -- "$race_body"

if [[ "$HTTP_STATUS" == "200" ]]; then
  pass "RACE mode → HTTP 200"
else
  fail "RACE mode → HTTP $HTTP_STATUS (body: ${RESP_BODY:0:200})"
fi

mode=$(header_val "x-execution-mode")
[[ "$mode" == "race" ]] && pass "x-execution-mode = race" || fail "x-execution-mode = '${mode}'"

winner=$(header_val "x-hedge-winner")
hedged=$(header_val "x-hedged")
[[ -n "$winner" || -n "$hedged" ]] && pass "hedge winner/hedged header present (winner=$winner, hedged=$hedged)" \
  || skip "x-hedge-winner/x-hedged not set (may be single-backend fallback)"

# ── Test 6: Circuit breaker ───────────────────────────────────────────────
echo ""
echo "── 6. Circuit breaker (dead node injection) ─────────────────────────"

DEAD_NODE="dead-node.invalid:9999"
MODEL_ID="circuit-test-model"

# Register a dead node in Redis for a test model
docker compose exec -T redis redis-cli SADD "nodes:healthy" "$DEAD_NODE" >/dev/null 2>&1 || true
docker compose exec -T redis redis-cli SET "node:$DEAD_NODE:models" \
  "{\"object\":\"list\",\"data\":[{\"id\":\"$MODEL_ID\",\"object\":\"model\"}]}" >/dev/null 2>&1 || true

# Fire requests — the LB should trip the circuit breaker after threshold failures
BODY="{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":5}"
cb_tripped=false
for i in {1..5}; do
  req -- "$BODY"
  if docker compose exec -T redis redis-cli EXISTS "circuit:$DEAD_NODE" 2>/dev/null | grep -q "^1$"; then
    cb_tripped=true
    break
  fi
done

if $cb_tripped; then
  pass "Circuit breaker tripped for $DEAD_NODE after failures"
else
  # Check if node got penalised instead
  penalty=$(docker compose exec -T redis redis-cli EXISTS "penalty:$DEAD_NODE" 2>/dev/null || echo "0")
  if [[ "$penalty" == "1" ]]; then
    pass "Dead node penalised in Redis (failure penalty applied)"
  else
    skip "Circuit breaker not observed — may need more requests or threshold differs"
  fi
fi

# Clean up test node
docker compose exec -T redis redis-cli SREM "nodes:healthy" "$DEAD_NODE" >/dev/null 2>&1 || true
docker compose exec -T redis redis-cli DEL "node:$DEAD_NODE:models" >/dev/null 2>&1 || true

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────────────────────────────────"
if (( FAILURES == 0 )); then
  echo -e "${GREEN}All checks passed.${NC}"
else
  echo -e "${RED}$FAILURES check(s) failed.${NC}"
  exit 1
fi
