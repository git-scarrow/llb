import os
import pathlib


def _read_secret(env_var: str, secret_file: str, default: str = "") -> str:
    """Read a secret from a Docker-mounted file, falling back to env var (dev only)."""
    p = pathlib.Path(secret_file)
    if p.exists():
        return p.read_text().strip()
    return os.getenv(env_var, default)


ANTHROPIC_API_KEY = _read_secret(
    "ANTHROPIC_API_KEY",
    "/run/secrets/anthropic_api_key",
)
OPENAI_API_KEY = _read_secret(
    "OPENAI_API_KEY",
    "/run/secrets/openai_api_key",
)

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
P2C_ALPHA = float(os.getenv("P2C_ALPHA", 0.5))  # Weight for p95 latency in scoring (legacy additive)
P2C_PENALTY_WEIGHT = float(os.getenv("P2C_PENALTY_WEIGHT", 2.0))  # Weight for recent 5xx rate (legacy additive)

# Peak EWMA scoring (Finagle/Envoy-inspired, replaces additive scoring)
# score = ewma_rtt * (pending + 1)  -- dimensionally consistent, no tunable weights
P2C_PEAK_EWMA = os.getenv("P2C_PEAK_EWMA", "true").lower() in ("1", "true", "yes")
PEAK_EWMA_DECAY_SECS = float(os.getenv("PEAK_EWMA_DECAY_SECS", 10.0))  # Half-life for RTT decay
PEAK_EWMA_PENALTY_RTT = float(os.getenv("PEAK_EWMA_PENALTY_RTT", 30.0))  # Penalty RTT for failures (secs)
PEAK_EWMA_DEFAULT_RTT = float(os.getenv("PEAK_EWMA_DEFAULT_RTT", 0.5))  # Cold-start RTT estimate (secs)

# Cost-aware P2C routing (Phase 4)
def _parse_cost_per_token(s: str) -> dict:
    import json as _json
    try:
        return _json.loads(s)
    except Exception:
        return {}

BACKEND_COST_PER_TOKEN: dict = _parse_cost_per_token(os.getenv("BACKEND_COST_PER_TOKEN", ""))
P2C_BETA = float(os.getenv("P2C_BETA", 0.0))                              # Cost weight in P2C scoring (0 = disabled)
COST_EWMA_ALPHA = float(os.getenv("COST_EWMA_ALPHA", 0.3))                # EWMA decay factor
COST_EWMA_TTL_SECS = int(os.getenv("COST_EWMA_TTL_SECS", 3600))           # TTL for EWMA Redis keys
COST_EWMA_COLD_START_TOKENS = int(os.getenv("COST_EWMA_COLD_START_TOKENS", 256))  # Estimate during cold start
COST_EWMA_MIN_SAMPLES = int(os.getenv("COST_EWMA_MIN_SAMPLES", 5))        # Min obs before trusting EWMA

# Enhanced retry and failover configuration
ATTEMPTS_PER_MODEL = int(os.getenv("ATTEMPTS_PER_MODEL", 3))
CROSS_MODEL_FALLBACK = os.getenv("CROSS_MODEL_FALLBACK", "false").lower() in ("1", "true", "yes")
RETRY_BACKOFF_MS = [int(x) for x in os.getenv("RETRY_BACKOFF_MS", "50,100").split(",")]

# Derive MAX_RETRIES used in existing flows (attempts after the first)
MAX_RETRIES = max(0, ATTEMPTS_PER_MODEL - 1)

# Same-node retry: when only 1 node exists, retry it after a delay instead of
# failing immediately.  Covers Ollama model-swap disconnects (~10-30s).
SAME_NODE_RETRY_ENABLED = os.getenv("SAME_NODE_RETRY_ENABLED", "true").lower() in ("1", "true", "yes")
SAME_NODE_RETRY_MAX = int(os.getenv("SAME_NODE_RETRY_MAX", 2))
SAME_NODE_RETRY_DELAY_SECS = float(os.getenv("SAME_NODE_RETRY_DELAY_SECS", 3.0))

# Circuit breaker enhancements
CIRCUIT_BREAKER_COOLDOWN_SECS = int(os.getenv("CIRCUIT_BREAKER_COOLDOWN_SECS", 60))
CIRCUIT_BREAKER_SUSPECT_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_SUSPECT_THRESHOLD", 1))
CIRCUIT_BREAKER_SUSPECT_WEIGHT = float(os.getenv("CIRCUIT_BREAKER_SUSPECT_WEIGHT", 0.5))
# CB scope: "model" = per-(node,model) keys + node-wide aggregate; "node" = legacy per-node only
CB_SCOPE = os.getenv("LLB_CB_SCOPE", "model").lower()
# When true, /health returns "degraded" (HTTP 200) if any known model has zero eligible nodes
HEALTH_MODEL_AWARE = os.getenv("LLB_HEALTH_MODEL_AWARE", "false").lower() in ("1", "true", "yes")

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

# Multi-backend execution configuration
# Enables parallel/sequential execution across multiple API endpoints
MULTI_EXEC_ENABLED = os.getenv("MULTI_EXEC_ENABLED", "true").lower() in ("1", "true", "yes")
MULTI_EXEC_MAX_BACKENDS = int(os.getenv("MULTI_EXEC_MAX_BACKENDS", 3))
MULTI_EXEC_TIMEOUT_SECS = float(os.getenv("MULTI_EXEC_TIMEOUT_SECS", 60.0))
# Consensus similarity threshold for text responses (0.0-1.0)
MULTI_EXEC_CONSENSUS_THRESHOLD = float(os.getenv("MULTI_EXEC_CONSENSUS_THRESHOLD", 0.9))

# Backend aliases for human-friendly names
# Format: "alias1=host1:port1,alias2=host2:port2" or "alias1=host1:port1;alias2=host2:port2"
# Example: "m2=macbook-m2.local:1234,m4=macbook-m4.scarrow.tailnet:1234,gentoo=localhost:11434"
def _parse_backend_aliases(s: str) -> dict:
    """Parse BACKEND_ALIASES env var into {alias: host:port} mapping."""
    mapping = {}
    for part in [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]:
        if "=" not in part:
            continue
        alias, target = part.split("=", 1)
        alias = alias.strip()
        target = target.strip()
        if alias and target:
            mapping[alias] = target
    return mapping

BACKEND_ALIASES = _parse_backend_aliases(os.getenv("BACKEND_ALIASES", ""))

# Reverse mapping: host:port -> alias (for display purposes)
BACKEND_ALIASES_REVERSE = {v: k for k, v in BACKEND_ALIASES.items()}

# Model equivalence groups for multi-backend consensus
# Format: "canonical=model1,model2,model3;another=modelA,modelB"
# Example: "qwen2.5-instruct=qwen2.5-7b-instruct-mlx@4bit,qwen2.5:7b-instruct,qwen2.5-7b-instruct:latest"
# When requesting "qwen2.5-instruct", backends with any of the listed models are eligible
def _parse_model_equivalents(s: str) -> dict:
    """Parse MODEL_EQUIVALENTS into {canonical: [model1, model2, ...]}."""
    mapping = {}
    for group in [g.strip() for g in s.split(";") if g.strip()]:
        if "=" not in group:
            continue
        canonical, models = group.split("=", 1)
        canonical = canonical.strip()
        model_list = [m.strip() for m in models.split(",") if m.strip()]
        if canonical and model_list:
            mapping[canonical] = model_list
    return mapping

MODEL_EQUIVALENTS = _parse_model_equivalents(os.getenv("MODEL_EQUIVALENTS", ""))

# Build reverse lookup: model_name -> canonical name
MODEL_EQUIVALENTS_REVERSE = {}
for canonical, models in MODEL_EQUIVALENTS.items():
    for m in models:
        MODEL_EQUIVALENTS_REVERSE[m] = canonical

# Cloud backend configuration
# Format: "name=url|api_key[|provider_type],name2=url2|api_key2[|provider_type]" or semicolon-separated
# provider_type is optional, defaults to "openai"
# Example: "openai=https://api.openai.com/v1|sk-xxx|openai,claude=https://api.anthropic.com/v1|sk-ant-xxx|anthropic"
# URL should include /v1 path for OpenAI-compatible endpoints
def _parse_cloud_backends(s: str) -> dict:
    """Parse CLOUD_BACKENDS env var into {name: {url, api_key, provider_type, is_cloud}}."""
    mapping = {}
    for part in [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]:
        if "=" not in part:
            continue
        name, rest = part.split("=", 1)
        name = name.strip()
        rest = rest.strip()
        if "|" not in rest:
            continue
        parts = rest.split("|")
        if len(parts) < 2:
            continue
        url = parts[0].strip()
        api_key = parts[1].strip()
        # Provider type is optional, defaults to "openai"
        provider_type = parts[2].strip() if len(parts) > 2 else "openai"
        if name and url and api_key:
            mapping[name] = {
                "url": url,
                "api_key": api_key,
                "provider_type": provider_type,
                "is_cloud": True,
            }
    return mapping

CLOUD_BACKENDS = _parse_cloud_backends(os.getenv("CLOUD_BACKENDS", ""))

# Rate limiting configuration for cloud backends
RATE_LIMIT_BACKOFF_BASE_SECS = float(os.getenv("RATE_LIMIT_BACKOFF_BASE_SECS", 1.0))
RATE_LIMIT_BACKOFF_MAX_SECS = float(os.getenv("RATE_LIMIT_BACKOFF_MAX_SECS", 60.0))
RATE_LIMIT_BACKOFF_JITTER = float(os.getenv("RATE_LIMIT_BACKOFF_JITTER", 0.3))  # 30% jitter

# Cloud models - which models are available on which cloud backends
# Format: "backend_name=model1,model2;backend2=model3,model4"
# Example: "openai=gpt-4o,gpt-4-turbo;claude=claude-sonnet-4-20250514,claude-3-haiku"
def _parse_cloud_models(s: str) -> dict:
    """Parse CLOUD_MODELS env var into {backend_name: [model1, model2, ...]}."""
    mapping = {}
    for group in [g.strip() for g in s.split(";") if g.strip()]:
        if "=" not in group:
            continue
        name, models = group.split("=", 1)
        name = name.strip()
        model_list = [m.strip() for m in models.split(",") if m.strip()]
        if name and model_list:
            mapping[name] = model_list
    return mapping

CLOUD_MODELS = _parse_cloud_models(os.getenv("CLOUD_MODELS", ""))


# Fallback chain configuration
# Format: "chain_name=backend1>backend2>backend3;chain2=..."
# Extended format with per-backend options: "backend1(timeout=30,retries=2)>backend2"
# Example: "default=cloud:openai(timeout=30)>cloud:anthropic(timeout=45)>local:auto"
from dataclasses import dataclass as dc
from typing import List as ListType


@dc
class FallbackBackend:
    """Configuration for a single backend in a fallback chain."""
    backend: str
    timeout_secs: float = 30.0
    max_retries: int = 1


def _parse_fallback_chains(s: str) -> dict:
    """Parse FALLBACK_CHAINS env var into {chain_name: [FallbackBackend, ...]}."""
    chains = {}
    for group in [g.strip() for g in s.split(";") if g.strip()]:
        if "=" not in group:
            continue
        name, chain_str = group.split("=", 1)
        backends = []
        for part in chain_str.split(">"):
            part = part.strip()
            if not part:
                continue
            # Parse backend(option=value,...)
            if "(" in part and part.endswith(")"):
                backend_name, opts_str = part[:-1].split("(", 1)
                opts = {}
                for opt in opts_str.split(","):
                    if "=" in opt:
                        k, v = opt.split("=", 1)
                        opts[k.strip()] = v.strip()
                backends.append(FallbackBackend(
                    backend=backend_name.strip(),
                    timeout_secs=float(opts.get("timeout", 30)),
                    max_retries=int(opts.get("retries", 1))
                ))
            else:
                backends.append(FallbackBackend(backend=part))
        if name.strip() and backends:
            chains[name.strip()] = backends
    return chains


FALLBACK_CHAINS = _parse_fallback_chains(os.getenv("FALLBACK_CHAINS", ""))
DEFAULT_FALLBACK_CHAIN = os.getenv("DEFAULT_FALLBACK_CHAIN", "")
FALLBACK_TOTAL_TIMEOUT_SECS = float(os.getenv("FALLBACK_TOTAL_TIMEOUT_SECS", 120))


# ---------------------------------------------------------------------------
# Backend capability tags (inspired by Perplexity Computer model specialization)
# Maps backend name → frozenset of capability strings.
# Cloud backends: use the cloud name ("openai", "claude", "gemini").
# Local backends: use host:port.
# Format: "openai=reasoning,code;gemini=research,multimodal;localhost:11434=fast,private"
# Available capability tags (convention, not enforced):
#   reasoning, code, research, multimodal, creative, math, fast, private, long-context
# Inspiration: Perplexity Computer (multi-model specialization), Apache-2.0
# ---------------------------------------------------------------------------
def _parse_backend_capabilities(s: str) -> dict:
    """Parse BACKEND_CAPABILITIES env var into {backend_name: frozenset of caps}.

    Accepts semicolons or pipes as entry separators; commas separate capabilities.
    Cloud backend names with a "cloud:" prefix are normalized (prefix stripped)
    so that lookups from main.py work consistently.
    Capability matching is case-sensitive — no lowercasing is applied.
    """
    mapping: dict = {}
    # Pipes are accepted as an alternative separator to semicolons
    for group in [g.strip() for g in s.replace("|", ";").split(";") if g.strip()]:
        if "=" not in group:
            continue
        name, caps_str = group.split("=", 1)
        name = name.strip()
        # Strip "cloud:" prefix so keys always match the bare backend name
        if name.startswith("cloud:"):
            name = name[len("cloud:"):]
        # Case-sensitive: no .lower() applied to capability strings
        cap_list = frozenset(c.strip() for c in caps_str.split(",") if c.strip())
        if name and cap_list:
            mapping[name] = cap_list
    return mapping


BACKEND_CAPABILITIES: dict = _parse_backend_capabilities(os.getenv("BACKEND_CAPABILITIES", ""))

# ---------------------------------------------------------------------------
# PLAN execution mode (Perplexity Computer-style multi-step orchestration)
# PLANNER_BACKEND: cloud backend name or host:port used to decompose tasks.
# ---------------------------------------------------------------------------
PLANNER_BACKEND: str = os.getenv("PLANNER_BACKEND", "")
PLAN_MAX_SUBTASKS: int = int(os.getenv("PLAN_MAX_SUBTASKS", "5"))
PLAN_SUBTASK_TIMEOUT_SECS: float = float(os.getenv("PLAN_SUBTASK_TIMEOUT_SECS", "90.0"))

# ---------------------------------------------------------------------------
# Complexity-based routing (inspired by RouteLLM, Apache-2.0 — lm-sys/RouteLLM)
# When enabled, prompt complexity is scored and MODEL_CLASSES tiers are used
# to automatically select small/medium/large model pools for sentinel requests.
# ---------------------------------------------------------------------------
COMPLEXITY_ROUTING_ENABLED: bool = os.getenv("COMPLEXITY_ROUTING_ENABLED", "false").lower() in ("1", "true", "yes")
