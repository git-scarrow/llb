# FOSS Inspirations & Attribution

This document tracks open-source projects whose ideas, patterns, or code
influenced LLB's implementation. Attribution is included inline in source
comments as well.

---

## Perplexity Computer (2026-02-27)
- **Source**: https://venturebeat.com/technology/perplexity-launches-computer-ai-agent-that-coordinates-19-models-priced-at
- **Concept borrowed**: Hierarchical multi-model orchestration — a master model
  decomposes a user goal into a JSON subtask graph, each subtask routed to the
  best capability-specialized backend, results assembled by a final synthesis call.
- **Implemented as**: `ExecutionMode.PLAN` in `execution/modes.py`,
  `_resolve_planner_backend()` and PLAN handler in `main.py`,
  `BACKEND_CAPABILITIES` config for capability-typed pools.
- **Improvement over original**: Runs as a transparent OpenAI-compatible proxy;
  callers receive a single standard `/v1/chat/completions` response.

---

## RouteLLM — lm-sys/RouteLLM (Apache-2.0)
- **Source**: https://github.com/lm-sys/RouteLLM
- **Concept borrowed**: Complexity-scoring interface — classify prompt difficulty
  and route to strong/weak model accordingly.
- **Implemented as**: `ComplexityRoutingStrategy.score_prompt_complexity()` and
  `get_complexity_model()` in `routing/strategies.py`, wired into sentinel
  resolution in `main.py` when `COMPLEXITY_ROUTING_ENABLED=true`.
- **Improvement**: Pure heuristic scorer (no extra model call); wired into
  existing `MODEL_CLASSES` tier system.

---

## LiteLLM — BerriAI/litellm (MIT)
- **Source**: https://docs.litellm.ai/docs/proxy/auto_routing
- **Concept borrowed**: Semantic/keyword-based routing rules per model; the idea
  that backend selection can be driven by request content, not just load metrics.
- **Implemented as**: `x-require-capability` request header + `BACKEND_CAPABILITIES`
  config that tags each backend with capability strings (`research`, `code`,
  `reasoning`, `fast`, etc.), filtered in `_filter_nodes_by_capability()`.

---

## Swarms — kyegomez/swarms (Apache-2.0)
- **Source**: https://github.com/kyegomez/swarms
- **Concept borrowed**: DAG-based multi-agent task decomposition; fan-out to
  specialist agents, fan-in to assembler.
- **Implemented as**: Topological batch execution inside `ExecutionEngine.execute_plan()`
  (Kahn's algorithm for dependency ordering, parallel dispatch within each batch).

---

## OpenAI Agents SDK (MIT)
- **Source**: https://openai.github.io/openai-agents-python/multi_agent/
- **Concept borrowed**: Triage + handoff pattern — one orchestrator classifies
  intent and delegates to specialists.
- **Implemented as**: The planner/assembler roles in `execute_plan()`, and the
  `_resolve_planner_backend()` helper that picks the orchestration model.
