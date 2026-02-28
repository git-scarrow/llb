"""Multi-backend execution engine for parallel/sequential LLM calls."""

from .modes import (
    ExecutionMode,
    ExecutionConfig,
    BackendResult,
    ConsensusResult,
    ExecutionEngine,
    PlanTask,
    PlanResult,
    compute_consensus,
)

__all__ = [
    "ExecutionMode",
    "ExecutionConfig",
    "BackendResult",
    "ConsensusResult",
    "ExecutionEngine",
    "PlanTask",
    "PlanResult",
    "compute_consensus",
]
