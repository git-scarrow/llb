"""Execution mode implementations for the AI load balancer.

This module provides the PLAN execution mode, which decomposes a user request
into a DAG of subtasks, dispatches them to capable backends, and streams the
assembled response.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PlanEvent:
    """A single event emitted by the PLAN execution pipeline."""

    event_type: str  # plan_decomposed | task_started | task_finished | assembly_started | token | error
    data: dict
    timestamp: float = field(default_factory=time.time)


@dataclass
class PlanTask:
    """A single task in the decomposed plan DAG."""

    id: str
    capability: str
    prompt: str
    depends_on: list


@dataclass
class BackendResult:
    """Result from a single backend call."""

    backend: str
    success: bool
    response_body: dict


@dataclass
class PlanResult:
    """Aggregated result of a completed PLAN execution."""

    goal: str
    task_results: dict
    final_response: Optional[BackendResult]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = (
    "You are a task planning assistant. "
    "Decompose the user's request into a minimal set of subtasks (at most 5). "
    "Return ONLY valid JSON in this exact format:\n"
    '{"goal": "<short goal summary>", "tasks": ['
    '{"id": "t1", "capability": "general", "prompt": "<subtask instruction>", "depends_on": []},'
    '{"id": "t2", "capability": "general", "prompt": "<subtask instruction>", "depends_on": ["t1"]}'
    "]}\n"
    "Do not include any text outside the JSON object."
)


def _parse_plan_json(text: str) -> dict:
    """Parse JSON from a planner response, stripping markdown code fences if present."""
    text = text.strip()
    # Remove markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()
    # Extract outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    return json.loads(text)


# ---------------------------------------------------------------------------
# Core async generator
# ---------------------------------------------------------------------------

async def execute_plan_stream(
    messages: list,
    call_backend,
    stream_backend,
    planner_backend: str,
    capability_nodes: dict,
    default_nodes: list,
    max_subtasks: int = 5,
    subtask_timeout: float = 30.0,
    overall_timeout: float = 120.0,
) -> AsyncGenerator[PlanEvent, None]:
    """Async generator that drives the PLAN pipeline and yields PlanEvents.

    Step 1 — Decompose: call the planner backend (blocking; full JSON needed).
    Step 2 — Dispatch:  execute subtasks in topological batches (Kahn's algorithm).
    Step 3 — Assemble:  stream assembly tokens directly from the planner backend.
    """

    # ------------------------------------------------------------------
    # Step 1: Decompose
    # ------------------------------------------------------------------
    try:
        planner_msgs = [
            {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
            *messages,
        ]
        decomp = await asyncio.wait_for(
            call_backend(planner_backend, planner_msgs),
            timeout=subtask_timeout,
        )
        plan_data = _parse_plan_json(decomp)
        raw_tasks = plan_data.get("tasks", [])[:max_subtasks]
        tasks = [
            PlanTask(
                id=str(t["id"]),
                capability=str(t.get("capability", "general")),
                prompt=str(t.get("prompt", "")),
                depends_on=[str(d) for d in t.get("depends_on", [])],
            )
            for t in raw_tasks
        ]
        goal = str(plan_data.get("goal", ""))
    except Exception as exc:
        yield PlanEvent("error", {"message": f"Plan decompose failed: {exc}"})
        return

    yield PlanEvent(
        "plan_decomposed",
        {
            "goal": goal,
            "task_count": len(tasks),
            "tasks": [{"id": t.id, "capability": t.capability} for t in tasks],
        },
    )

    # ------------------------------------------------------------------
    # Step 2: Dispatch (Kahn's algorithm)
    # ------------------------------------------------------------------
    task_results: dict[str, Any] = {}
    completed: set[str] = set()
    remaining = list(tasks)

    async def _run_subtask(task: PlanTask) -> tuple:
        t_start = time.time()
        try:
            node = capability_nodes.get(task.capability) or (
                default_nodes[0] if default_nodes else planner_backend
            )
            content = await asyncio.wait_for(
                call_backend(node, [{"role": "user", "content": task.prompt}]),
                timeout=subtask_timeout,
            )
            return task, {
                "success": True,
                "content": content,
                "latency_ms": int((time.time() - t_start) * 1000),
            }
        except Exception as exc:
            return task, {
                "success": False,
                "error": str(exc),
                "latency_ms": int((time.time() - t_start) * 1000),
            }

    while remaining:
        ready = [
            t for t in remaining if all(dep in completed for dep in t.depends_on)
        ]
        if not ready:
            yield PlanEvent(
                "error",
                {"message": "Unresolvable dependency: possible cycle in task graph"},
            )
            return

        for t in ready:
            yield PlanEvent(
                "task_started", {"task_id": t.id, "capability": t.capability}
            )

        pending = {asyncio.ensure_future(_run_subtask(t)) for t in ready}
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for fut in done:
                task, result = fut.result()
                task_results[task.id] = result
                completed.add(task.id)
                yield PlanEvent(
                    "task_finished",
                    {
                        "task_id": task.id,
                        "success": result["success"],
                        "latency_ms": result["latency_ms"],
                    },
                )
                if not result["success"]:
                    yield PlanEvent(
                        "error",
                        {
                            "task_id": task.id,
                            "message": result.get("error", "unknown error"),
                        },
                    )

        remaining = [t for t in remaining if t.id not in completed]

    # ------------------------------------------------------------------
    # Step 3: Assembly — pipe stream_backend tokens directly
    # ------------------------------------------------------------------
    yield PlanEvent("assembly_started", {"input_task_count": len(tasks)})

    task_context = "\n".join(
        f"Task {tid}: {r.get('content', r.get('error', ''))}"
        for tid, r in task_results.items()
    )
    assembly_msgs = [
        {
            "role": "system",
            "content": "Synthesize the following task results into a comprehensive response.",
        },
        {
            "role": "user",
            "content": f"Goal: {goal}\n\nTask results:\n{task_context}",
        },
    ]

    try:
        async for chunk in stream_backend(planner_backend, assembly_msgs):
            if chunk:
                text = (
                    chunk.decode("utf-8", errors="replace")
                    if isinstance(chunk, bytes)
                    else str(chunk)
                )
                yield PlanEvent("token", {"chunk": text})
    except Exception as exc:
        yield PlanEvent("error", {"message": f"Assembly streaming failed: {exc}"})


# ---------------------------------------------------------------------------
# Backward-compat adapter
# ---------------------------------------------------------------------------

async def collect_plan_result(gen: AsyncGenerator[PlanEvent, None]) -> PlanResult:
    """Consume an execute_plan_stream generator and return a PlanResult.

    This is the backward-compat adapter so callers that expect a blocking
    PlanResult can remain unchanged.
    """
    goal = ""
    task_results: dict = {}
    tokens: list[str] = []

    async for event in gen:
        if event.event_type == "plan_decomposed":
            goal = event.data.get("goal", "")
        elif event.event_type == "task_finished":
            task_results[event.data["task_id"]] = event.data
        elif event.event_type == "token":
            tokens.append(event.data.get("chunk", ""))

    assembled = "".join(tokens)
    return PlanResult(
        goal=goal,
        task_results=task_results,
        final_response=BackendResult(
            backend="",
            success=True,
            response_body={
                "choices": [{"message": {"content": assembled}}]
            },
        ),
    )


async def execute_plan(
    messages: list,
    call_backend,
    stream_backend,
    planner_backend: str,
    capability_nodes: dict,
    default_nodes: list,
    **kwargs,
) -> PlanResult:
    """Thin wrapper that runs execute_plan_stream and returns a PlanResult.

    Maintains backward compatibility for callers that expect a blocking call.
    """
    return await collect_plan_result(
        execute_plan_stream(
            messages=messages,
            call_backend=call_backend,
            stream_backend=stream_backend,
            planner_backend=planner_backend,
            capability_nodes=capability_nodes,
            default_nodes=default_nodes,
            **kwargs,
        )
    )


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def plan_result_to_openai_response(result: PlanResult) -> dict:
    """Convert a PlanResult to an OpenAI-compatible chat completion response dict."""
    content = ""
    if result.final_response and result.final_response.response_body:
        try:
            content = result.final_response.response_body["choices"][0]["message"][
                "content"
            ]
        except (KeyError, IndexError, TypeError):
            pass
    return {
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "model": "plan",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
