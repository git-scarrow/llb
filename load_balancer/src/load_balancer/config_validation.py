"""Boot-time config validation. Fails fast with actionable error messages."""

import os
import sys
import json
import logging

log = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


def validate_config() -> None:
    """
    Validate all required and conditionally-required env vars.
    Logs a critical error with a full diagnostic message and exits on failure.
    Called once at startup inside lifespan() before Redis init.
    """
    errors: list[str] = []

    # ── Always required ───────────────────────────────────────────────────────
    # At least one of CLOUD_BACKENDS or SCAN_HOSTS must be set.
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
            "LB failed config validation at startup. Fix the following:\n\n%s\n\n"
            "Exiting.",
            msg
        )
        sys.exit(1)

    log.info("Config validation passed.")
