import os

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
# Routing strategy
# Supports ROUND_ROBIN, RANDOM, LEAST_LOADED, P2C (Power of Two Choices)
# Default to P2C to improve load spread and tail behavior
ROUTING_STRATEGY = os.getenv("ROUTING_STRATEGY", "P2C")
MIN_HEALTHY_NODES = int(os.getenv("MIN_HEALTHY_NODES", 1))

# Unified LB enhancements
REQUEST_TIMEOUT_SECS = float(os.getenv("REQUEST_TIMEOUT_SECS", 60))
# Harmonize attempts across the codebase: ATTEMPTS_PER_MODEL is the total attempts budget
# (1 initial + retries). MAX_RETRIES is derived for compatibility with existing code paths.
FAILURE_PENALTY_TTL_SECS = int(os.getenv("FAILURE_PENALTY_TTL_SECS", 30))
RETRY_AFTER_SECS = int(os.getenv("RETRY_AFTER_SECS", 2))

# Circuit breaker
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", 3))
# Deprecated name retained for backward compatibility; prefer CIRCUIT_BREAKER_COOLDOWN_SECS
CIRCUIT_BREAKER_TTL_SECS = int(os.getenv("CIRCUIT_BREAKER_TTL_SECS", 30))

# Model auto-selection configuration
# A model name matching any sentinel will trigger auto-selection.
_RAW = os.getenv("MODEL_SENTINELS", "auto,default")
MODEL_SENTINELS = {s.strip().lower() for s in _RAW.replace(";", ",").split(",") if s.strip()}
LM_MODEL_SENTINEL = os.getenv("LM_MODEL", "auto").strip().lower()
DEFAULT_CHAT_MODEL = (os.getenv("DEFAULT_CHAT_MODEL", "") or "").strip() or None
DEFAULT_EMBEDDINGS_MODEL = (os.getenv("DEFAULT_EMBEDDINGS_MODEL", "") or "").strip() or None
PREFERRED_MODELS = [s.strip() for s in os.getenv("PREFERRED_MODELS", "").replace(";", ",").split(",") if s.strip()]
AUTO_MODEL_STRATEGY = os.getenv("AUTO_MODEL_STRATEGY", "any_first")  # any_first | intersection_first

# Optional static model weights for auto selection (can be overridden via admin API)
def _parse_weights(s: str):
    out = {}
    for part in s.replace(";", ",").split(","):
        if not part.strip():
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip()
            try:
                out[k] = float(v)
            except Exception:
                continue
    return out

MODEL_WEIGHTS = _parse_weights(os.getenv("MODEL_WEIGHTS", ""))

# Enhanced response headers configuration
RESPONSE_HEADERS = [
    "x-selected-model", "x-routed-node", "x-request-id", 
    "x-hedged", "x-hedge-winner", "x-retry-count", "x-fallback-model"
]

# Sticky sessions
STICKY_SESSIONS_ENABLED = os.getenv("STICKY_SESSIONS_ENABLED", "false").lower() in ("1", "true", "yes")
STICKY_TTL_SECS = int(os.getenv("STICKY_TTL_SECS", 600))

# Model classes and fallback configuration
LB_PREFER_SMALL = os.getenv("LM_PREFER_SMALL", "false").lower() in ("1", "true", "yes")
AUTO_MIN_NODES = int(os.getenv("AUTO_MIN_NODES", 2))
STRICT_AUTO_MODE = os.getenv("STRICT_AUTO_MODE", "false").lower() in ("1", "true", "yes")

# Model classes for fallback (can be extended via config)
MODEL_CLASSES = {
    "historical_small": {
        "candidates": ["qwen/qwen3-4b-2507", "liquid/lfm2-1.2b", "mistralai/mistral-small-3.2"],
        "min_nodes": 2
    },
    "historical_medium": {
        "candidates": ["qwen/qwen3-8b", "mlx-community/gpt-oss-20b"],
        "min_nodes": 2
    }
}

# Load model classes from environment if provided
def _parse_model_classes(env_var: str):
    import json
    try:
        return json.loads(os.getenv(env_var, "{}"))
    except Exception:
        return {}

MODEL_CLASSES.update(_parse_model_classes("LB_MODEL_CLASSES"))

# Power of Two Choices routing configuration
P2C_ROUTING_ENABLED = os.getenv("P2C_ROUTING_ENABLED", "true").lower() in ("1", "true", "yes")
P2C_ALPHA = float(os.getenv("P2C_ALPHA", 0.5))  # Weight for p95 latency in scoring
P2C_PENALTY_WEIGHT = float(os.getenv("P2C_PENALTY_WEIGHT", 2.0))  # Weight for recent 5xx rate

# Cost-aware P2C routing
def _parse_cost_per_token(s: str) -> dict:
    import json as _json
    try:
        return _json.loads(s)
    except Exception:
        return {}

BACKEND_COST_PER_TOKEN: dict = _parse_cost_per_token(os.getenv("BACKEND_COST_PER_TOKEN", ""))
P2C_BETA = float(os.getenv("P2C_BETA", 0.0))          # Cost weight in P2C scoring (0 = disabled)
COST_EWMA_ALPHA = float(os.getenv("COST_EWMA_ALPHA", 0.3))         # EWMA decay factor
COST_EWMA_TTL_SECS = int(os.getenv("COST_EWMA_TTL_SECS", 3600))    # TTL for EWMA Redis keys
COST_EWMA_COLD_START_TOKENS = int(os.getenv("COST_EWMA_COLD_START_TOKENS", 256))  # Default estimate during cold start
COST_EWMA_MIN_SAMPLES = int(os.getenv("COST_EWMA_MIN_SAMPLES", 5))  # Min observations before trusting EWMA

# Enhanced retry and failover configuration
ATTEMPTS_PER_MODEL = int(os.getenv("ATTEMPTS_PER_MODEL", 3))
CROSS_MODEL_FALLBACK = os.getenv("CROSS_MODEL_FALLBACK", "false").lower() in ("1", "true", "yes")
RETRY_BACKOFF_MS = [int(x) for x in os.getenv("RETRY_BACKOFF_MS", "50,100").split(",")]

# Derive MAX_RETRIES used in existing flows (attempts after the first)
MAX_RETRIES = max(0, ATTEMPTS_PER_MODEL - 1)

# Circuit breaker enhancements
CIRCUIT_BREAKER_COOLDOWN_SECS = int(os.getenv("CIRCUIT_BREAKER_COOLDOWN_SECS", 60))
CIRCUIT_BREAKER_SUSPECT_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_SUSPECT_THRESHOLD", 1))
CIRCUIT_BREAKER_SUSPECT_WEIGHT = float(os.getenv("CIRCUIT_BREAKER_SUSPECT_WEIGHT", 0.5))

# Hedging configuration
HEDGING_ENABLED = os.getenv("HEDGING_ENABLED", "true").lower() in ("1", "true", "yes")
HEDGING_SMALL_MODELS_ONLY = os.getenv("HEDGING_SMALL_MODELS_ONLY", "true").lower() in ("1", "true", "yes")
HEDGING_MAX_DELAY_MS = int(os.getenv("HEDGING_MAX_DELAY_MS", 800))
HEDGING_P95_FRACTION = float(os.getenv("HEDGING_P95_FRACTION", 0.6))

# Health and eligibility thresholds
MAX_P95_LATENCY_SECS = float(os.getenv("MAX_P95_LATENCY_SECS", 5.0))
HEALTH_LATENCY_WINDOW_SECS = int(os.getenv("HEALTH_LATENCY_WINDOW_SECS", 120))
ELIGIBILITY_MIN_P95_SAMPLES = int(os.getenv("ELIGIBILITY_MIN_P95_SAMPLES", 20))

# Capacity and fairness
TOKENS_PER_SEC_LIMIT = int(os.getenv("TOKENS_PER_SEC_LIMIT", 0))  # 0 = disabled
FAIRNESS_ENABLED = os.getenv("FAIRNESS_ENABLED", "true").lower() in ("1", "true", "yes")

# On-demand warm/wait when a model has zero eligible nodes
ON_DEMAND_WAIT_ENABLED = os.getenv("ON_DEMAND_WAIT_ENABLED", "true").lower() in ("1", "true", "yes")
ON_DEMAND_WARM_TIMEOUT_SECS = int(os.getenv("ON_DEMAND_WARM_TIMEOUT_SECS", 30))
ON_DEMAND_WARM_GRACE_SECS = int(os.getenv("ON_DEMAND_WARM_GRACE_SECS", 30))
ON_DEMAND_WARM_POLL_MS = int(os.getenv("ON_DEMAND_WARM_POLL_MS", 750))
ON_DEMAND_WARM_FANOUT = int(os.getenv("ON_DEMAND_WARM_FANOUT", 2))  # probe up to N healthy nodes concurrently
