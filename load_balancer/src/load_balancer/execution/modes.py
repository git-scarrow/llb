"""Multi-backend execution modes and consensus algorithms."""

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for multi-backend requests."""
    RACE = "race"           # N-way hedging, first response wins
    ALL = "all"             # Collect all responses in parallel
    SEQUENCE = "sequence"   # Serial chain through backends
    CONSENSUS = "consensus" # Majority vote - 2 of 3 agree wins
    PLAN = "plan"           # Perplexity Computer-style task decomposition + specialization


@dataclass
class ExecutionConfig:
    """Configuration for a multi-backend execution request."""
    mode: ExecutionMode
    target_backends: Optional[List[str]] = None  # Specific backends to use (aliases or host:port)
    max_backends: int = 3
    timeout_secs: float = 60.0


@dataclass
class BackendResult:
    """Result from a single backend execution."""
    backend: str                        # host:port of the backend
    alias: Optional[str] = None         # Human-friendly alias if configured
    success: bool = False
    status_code: int = 0
    response_body: Optional[Dict] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    first_byte_ms: float = 0.0          # Time to first byte for streaming


@dataclass
class ConsensusResult:
    """Result of consensus computation across multiple backends."""
    winner: BackendResult               # The majority-agreed response
    all_responses: List[BackendResult] = field(default_factory=list)
    agreement_count: int = 0            # How many backends agreed (2 or 3)
    disagreement: bool = False          # True if not unanimous
    comparison_type: str = "unknown"    # "tool_calls" | "text" | "hash"
    # Oracle support: designated authoritative backend
    oracle_backend: Optional[str] = None      # Which backend was designated oracle
    oracle_present: bool = False              # Was oracle in selected backends?
    oracle_agreed: Optional[bool] = None      # Did oracle agree with majority?
    # Local vs cloud agreement tracking
    local_cloud_agreement: Optional[bool] = None  # Did best local vs best cloud agree?
    local_cloud_similarity: Optional[float] = None  # Similarity score (0.0-1.0)


@dataclass
class PlanTask:
    """A single subtask within a PLAN execution."""
    id: str
    description: str
    prompt: str
    capability: str = "general"
    depends_on: List[str] = field(default_factory=list)


@dataclass
class PlanResult:
    """Result of a PLAN execution across decomposed subtasks."""
    goal: str
    tasks: List[PlanTask] = field(default_factory=list)
    task_results: Dict[str, BackendResult] = field(default_factory=dict)
    final_response: Optional[BackendResult] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# PLAN mode planner prompt — Perplexity Computer-style task decomposition
# Inspiration: Perplexity Computer (2026-02-27 launch), Swarms (Apache-2.0)
# ---------------------------------------------------------------------------
_PLANNER_SYSTEM_PROMPT = """\
You are a task orchestrator. Decompose the user request into a minimal set of subtasks, \
each handled by a specialized AI capability. Respond ONLY with valid JSON — no prose, \
no markdown fences, just the JSON object.

Format:
{{
  "goal": "<brief restatement of what the user wants>",
  "tasks": [
    {{
      "id": "t1",
      "description": "<what this step produces>",
      "capability": "<research|code|reasoning|creative|math|general>",
      "depends_on": [],
      "prompt": "<exact prompt for this subtask>"
    }}
  ]
}}

Rules:
- Maximum {max_tasks} tasks. If the goal is atomic, return exactly one task with id "t1".
- depends_on lists task IDs that must complete before this task starts.
- capability must be one of: research, code, reasoning, creative, math, general.
- Each prompt must be self-contained (do not assume the executor has context from other tasks).
"""

_ASSEMBLER_SYSTEM_PROMPT = """\
You are synthesizing the outputs of multiple parallel AI subtasks into a single coherent response.
Answer the user's original goal directly and concisely, incorporating all relevant subtask results.
"""


def _normalize_tool_calls(tool_calls: List[Dict]) -> str:
    """Normalize tool calls for comparison by sorting and normalizing JSON."""
    if not tool_calls:
        return ""
    normalized = []
    for tc in tool_calls:
        call = {
            "name": tc.get("function", {}).get("name", ""),
            "arguments": tc.get("function", {}).get("arguments", "")
        }
        # Parse and re-serialize arguments to normalize JSON formatting
        try:
            args = json.loads(call["arguments"]) if call["arguments"] else {}
            call["arguments"] = json.dumps(args, sort_keys=True, separators=(",", ":"))
        except (json.JSONDecodeError, TypeError):
            pass
        normalized.append(call)
    # Sort by name for consistent ordering
    normalized.sort(key=lambda x: x["name"])
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _extract_tool_calls(response: Dict) -> Optional[List[Dict]]:
    """Extract tool_calls from an OpenAI-format response."""
    choices = response.get("choices", [])
    if not choices:
        return None
    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls")
    if tool_calls:
        return tool_calls
    # Also check for function_call (older format)
    function_call = message.get("function_call")
    if function_call:
        return [{"function": function_call}]
    return None


def _extract_text_content(response: Dict) -> str:
    """Extract text content from an OpenAI-format response."""
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return message.get("content", "") or ""


_THINKING_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove common reasoning wrappers like <thinking>...</thinking>."""
    if not text:
        return ""
    return _THINKING_RE.sub("", text)


def _strip_json_fences(text: str) -> str:
    """If content is fenced as ```json ...```, unwrap it."""
    if not text:
        return ""
    m = _JSON_FENCE_RE.match(text.strip())
    return m.group(1) if m else text


def _try_canonicalize_json(text: str) -> tuple[str, bool]:
    """If text appears to be JSON, parse and re-dump in canonical form.

    Returns (normalized_text, was_json).
    """
    if not text:
        return "", False
    candidate = text.strip()
    if not (candidate.startswith("{") or candidate.startswith("[")):
        return text, False
    try:
        obj = json.loads(candidate)
    except Exception:
        # Looks like JSON but isn't parseable
        return "__INVALID_JSON__:" + candidate, True
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":")), True
    except Exception:
        return candidate, True


def _normalize_text_for_compare(text: str) -> str:
    """Normalize assistant text for hashing/similarity.

    - Strip <thinking> blocks
    - Unwrap fenced JSON
    - Collapse whitespace
    - Canonicalize JSON when possible
    """
    if not text:
        return ""
    t = _strip_thinking(text)
    t = _strip_json_fences(t)
    t = " ".join(t.split())
    t2, _ = _try_canonicalize_json(t)
    return t2


def _compute_text_similarity(text1: str, text2: str) -> float:
    """Compute similarity ratio between two text strings using SequenceMatcher."""
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    t1 = _normalize_text_for_compare(text1)
    t2 = _normalize_text_for_compare(text2)
    return SequenceMatcher(None, t1, t2).ratio()


def _content_hash(response: Dict) -> str:
    """Compute a hash of the response content for exact comparison."""
    text = _extract_text_content(response)
    tool_calls = _extract_tool_calls(response)
    content = {
        "text": _normalize_text_for_compare(text) if text else "",
        "tool_calls": _normalize_tool_calls(tool_calls) if tool_calls else ""
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def compute_consensus(
    results: List[BackendResult],
    similarity_threshold: float = 0.9,
    oracle_backend: Optional[str] = None,
    oracle_present: bool = False,
    is_cloud_fn: Optional[Callable[[str], bool]] = None
) -> ConsensusResult:
    """
    Compute consensus from multiple backend results.

    For tool calls: Compare normalized action name + parameters.
    For text: Compare content hash, fall back to similarity ratio.

    Oracle handling:
    - If oracle_present=True and oracle_backend in results:
        - Compare oracle response to majority winner
        - If oracle disagrees, prefer oracle as winner (it's authoritative)
    - If oracle_present=False (oracle requested but couldn't be included):
        - Don't override majority decision
        - Set oracle_agreed = None (unknown)

    Args:
        results: List of BackendResult from different backends
        similarity_threshold: Minimum similarity for text agreement (0.0-1.0)
        oracle_backend: Which backend is designated oracle (authoritative)
        oracle_present: Whether the oracle backend was included in the request
        is_cloud_fn: Optional function to check if a backend is a cloud backend

    Returns:
        ConsensusResult with winner and agreement info
    """
    successful = [r for r in results if r.success and r.response_body]

    if not successful:
        # All failed - return first result with error info
        return ConsensusResult(
            winner=results[0] if results else BackendResult(backend="unknown", error="No results"),
            all_responses=results,
            agreement_count=0,
            disagreement=True,
            comparison_type="none",
            oracle_backend=oracle_backend,
            oracle_present=oracle_present,
            oracle_agreed=None
        )

    if len(successful) == 1:
        # Only one success - it wins by default
        oracle_agreed = None
        if oracle_present and oracle_backend:
            oracle_agreed = (successful[0].backend == oracle_backend)

        return ConsensusResult(
            winner=successful[0],
            all_responses=results,
            agreement_count=1,
            disagreement=False,
            comparison_type="single",
            oracle_backend=oracle_backend,
            oracle_present=oracle_present,
            oracle_agreed=oracle_agreed
        )

    # Check if responses have tool calls
    tool_call_responses = []
    for r in successful:
        tc = _extract_tool_calls(r.response_body)
        if tc:
            tool_call_responses.append((r, _normalize_tool_calls(tc)))

    # Prefer tool call comparison if most responses have tool calls
    if len(tool_call_responses) >= len(successful) // 2 + 1:
        consensus = _consensus_by_tool_calls(successful, tool_call_responses, results, similarity_threshold)
    else:
        # Fall back to text comparison
        consensus = _consensus_by_text(successful, results, similarity_threshold)

    # Apply oracle handling and local/cloud agreement
    consensus = _apply_oracle_and_local_cloud(
        consensus, successful, similarity_threshold, oracle_backend, oracle_present, is_cloud_fn
    )

    return consensus


def _apply_oracle_and_local_cloud(
    consensus: ConsensusResult,
    successful: List[BackendResult],
    similarity_threshold: float,
    oracle_backend: Optional[str],
    oracle_present: bool,
    is_cloud_fn: Optional[Callable[[str], bool]]
) -> ConsensusResult:
    """Apply oracle handling and compute local vs cloud agreement.

    Args:
        consensus: The base consensus result from majority voting
        successful: List of successful results
        similarity_threshold: Threshold for text similarity comparison
        oracle_backend: Designated oracle backend
        oracle_present: Whether oracle was included
        is_cloud_fn: Function to check if backend is cloud

    Returns:
        Updated ConsensusResult with oracle and local/cloud fields filled
    """
    consensus.oracle_backend = oracle_backend
    consensus.oracle_present = oracle_present

    # Handle oracle logic
    if oracle_present and oracle_backend:
        oracle_result = next((r for r in successful if r.backend == oracle_backend), None)
        if oracle_result:
            # Check if oracle agrees with the winner
            winner_text = _extract_text_content(consensus.winner.response_body) if consensus.winner.response_body else ""
            oracle_text = _extract_text_content(oracle_result.response_body) if oracle_result.response_body else ""

            similarity = _compute_text_similarity(winner_text, oracle_text)
            consensus.oracle_agreed = similarity >= similarity_threshold

            # If oracle disagrees, it becomes the winner (authoritative)
            if not consensus.oracle_agreed:
                logger.info(
                    f"Oracle {oracle_backend} disagrees with majority (similarity={similarity:.2f}). "
                    "Overriding winner with oracle response."
                )
                consensus.winner = oracle_result
                # Recalculate agreement count with oracle as reference
                oracle_normalized_text = _normalize_text_for_compare(oracle_text) if oracle_text else ""
                new_agreement = 1  # Oracle agrees with itself
                for r in successful:
                    if r.backend != oracle_backend:
                        r_text = _extract_text_content(r.response_body) if r.response_body else ""
                        if _compute_text_similarity(oracle_normalized_text, r_text) >= similarity_threshold:
                            new_agreement += 1
                consensus.agreement_count = new_agreement
                consensus.disagreement = new_agreement < len(successful)
        else:
            # Oracle was supposed to be present but didn't return a successful response
            consensus.oracle_agreed = None
    else:
        consensus.oracle_agreed = None

    # Compute local vs cloud agreement
    if is_cloud_fn:
        local_results = [r for r in successful if not is_cloud_fn(r.backend)]
        cloud_results = [r for r in successful if is_cloud_fn(r.backend)]

        if local_results and cloud_results:
            # Best = fastest successful response
            best_local = min(local_results, key=lambda r: r.latency_ms)
            best_cloud = min(cloud_results, key=lambda r: r.latency_ms)

            local_text = _extract_text_content(best_local.response_body) if best_local.response_body else ""
            cloud_text = _extract_text_content(best_cloud.response_body) if best_cloud.response_body else ""

            similarity = _compute_text_similarity(local_text, cloud_text)
            consensus.local_cloud_similarity = similarity
            consensus.local_cloud_agreement = similarity >= similarity_threshold

    return consensus


def _consensus_by_tool_calls(
    successful: List[BackendResult],
    tool_call_data: List[Tuple[BackendResult, str]],
    all_results: List[BackendResult],
    similarity_threshold: float
) -> ConsensusResult:
    """Compute consensus based on tool call comparison."""
    # Group by normalized tool calls
    groups: Dict[str, List[BackendResult]] = {}
    for result, normalized in tool_call_data:
        if normalized not in groups:
            groups[normalized] = []
        groups[normalized].append(result)

    # Find largest group
    largest_group = max(groups.values(), key=len)
    winner = min(largest_group, key=lambda r: r.latency_ms)  # Pick fastest in consensus group

    return ConsensusResult(
        winner=winner,
        all_responses=all_results,
        agreement_count=len(largest_group),
        disagreement=len(largest_group) < len(successful),
        comparison_type="tool_calls"
    )


def _consensus_by_text(
    successful: List[BackendResult],
    all_results: List[BackendResult],
    similarity_threshold: float
) -> ConsensusResult:
    """Compute consensus based on text content comparison."""
    # First try exact hash match
    hashes: Dict[str, List[BackendResult]] = {}
    for r in successful:
        h = _content_hash(r.response_body)
        if h not in hashes:
            hashes[h] = []
        hashes[h].append(r)

    # Check if we have hash-based majority
    largest_hash_group = max(hashes.values(), key=len)
    if len(largest_hash_group) > len(successful) // 2:
        winner = min(largest_hash_group, key=lambda r: r.latency_ms)
        return ConsensusResult(
            winner=winner,
            all_responses=all_results,
            agreement_count=len(largest_hash_group),
            disagreement=len(largest_hash_group) < len(successful),
            comparison_type="hash"
        )

    # Fall back to similarity-based grouping
    texts = [(r, _normalize_text_for_compare(_extract_text_content(r.response_body))) for r in successful]

    # Build similarity matrix and group by threshold
    groups: List[List[BackendResult]] = []
    used = set()

    for i, (r1, t1) in enumerate(texts):
        if i in used:
            continue
        group = [r1]
        used.add(i)
        for j, (r2, t2) in enumerate(texts):
            if j in used:
                continue
            if _compute_text_similarity(t1, t2) >= similarity_threshold:
                group.append(r2)
                used.add(j)
        groups.append(group)

    largest_group = max(groups, key=len)
    winner = min(largest_group, key=lambda r: r.latency_ms)

    return ConsensusResult(
        winner=winner,
        all_responses=all_results,
        agreement_count=len(largest_group),
        disagreement=len(largest_group) < len(successful),
        comparison_type="text"
    )


class ExecutionEngine:
    """Engine for executing requests across multiple backends."""

    def __init__(self, similarity_threshold: float = 0.9):
        self.similarity_threshold = similarity_threshold

    async def execute_race(
        self,
        backends: List[str],
        request_fn: Callable[[str], Awaitable[BackendResult]],
        timeout: float = 60.0
    ) -> BackendResult:
        """
        Race N backends, return first successful response.

        Args:
            backends: List of backend host:port strings
            request_fn: Async function that takes backend and returns BackendResult
            timeout: Overall timeout in seconds

        Returns:
            First successful BackendResult, or first error if all fail
        """
        if not backends:
            return BackendResult(backend="none", error="No backends provided")

        tasks = {
            asyncio.create_task(request_fn(b)): b
            for b in backends
        }

        winner = None
        errors = []

        try:
            while tasks and not winner:
                done, pending = await asyncio.wait(
                    tasks.keys(),
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED
                )

                if not done:
                    # Timeout
                    break

                for task in done:
                    try:
                        result = task.result()
                        if result.success:
                            winner = result
                            break
                        else:
                            errors.append(result)
                    except Exception as e:
                        backend = tasks.get(task, "unknown")
                        errors.append(BackendResult(
                            backend=backend,
                            error=str(e)
                        ))
                    del tasks[task]

        finally:
            # Cancel remaining tasks
            for task in tasks:
                task.cancel()

        if winner:
            return winner
        if errors:
            return errors[0]
        return BackendResult(backend="timeout", error="All backends timed out")

    async def execute_all(
        self,
        backends: List[str],
        request_fn: Callable[[str], Awaitable[BackendResult]],
        timeout: float = 60.0
    ) -> List[BackendResult]:
        """
        Execute request on all backends in parallel, collect all responses.

        Args:
            backends: List of backend host:port strings
            request_fn: Async function that takes backend and returns BackendResult
            timeout: Overall timeout in seconds

        Returns:
            List of BackendResult from all backends
        """
        if not backends:
            return []

        async def wrapped_request(backend: str) -> BackendResult:
            try:
                return await asyncio.wait_for(request_fn(backend), timeout=timeout)
            except asyncio.TimeoutError:
                return BackendResult(backend=backend, error="Timeout")
            except Exception as e:
                return BackendResult(backend=backend, error=str(e))

        tasks = [wrapped_request(b) for b in backends]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results

    async def execute_sequence(
        self,
        backends: List[str],
        request_fn: Callable[[str], Awaitable[BackendResult]],
        timeout: float = 60.0,
        stop_on_success: bool = False
    ) -> List[BackendResult]:
        """
        Execute request on backends sequentially.

        Args:
            backends: List of backend host:port strings (in order)
            request_fn: Async function that takes backend and returns BackendResult
            timeout: Per-backend timeout in seconds
            stop_on_success: If True, stop after first successful response

        Returns:
            List of BackendResult from attempted backends
        """
        results = []
        for backend in backends:
            try:
                result = await asyncio.wait_for(request_fn(backend), timeout=timeout)
                results.append(result)
                if stop_on_success and result.success:
                    break
            except asyncio.TimeoutError:
                results.append(BackendResult(backend=backend, error="Timeout"))
            except Exception as e:
                results.append(BackendResult(backend=backend, error=str(e)))
        return results

    async def execute_plan(
        self,
        messages: List[Dict],
        call_backend: Callable[[str, List[Dict]], Awaitable[BackendResult]],
        planner_backend: str,
        capability_nodes: Dict[str, List[str]],
        default_nodes: List[str],
        max_subtasks: int = 5,
        subtask_timeout: float = 30.0,
        overall_timeout: float = 120.0,
    ) -> PlanResult:
        """
        Perplexity Computer-style multi-step task orchestration.

        Steps:
          1. Decompose: send the user prompt to planner_backend with a structured
             system prompt requesting a JSON subtask graph.
          2. Dispatch: execute independent tasks in parallel, sequential batches
             for dependent tasks. Each subtask routed to the best capability-matched
             backend node.
          3. Assemble: send all subtask results back to planner_backend for synthesis
             into a single final response.

        Args:
            messages: Original OpenAI-format user messages.
            call_backend: async fn(backend_node, messages) → BackendResult.
            planner_backend: Node to use for decomposition and assembly.
            capability_nodes: maps capability string → list of eligible nodes.
            default_nodes: fallback nodes when no capability match.
            max_subtasks: Maximum subtasks allowed.
            subtask_timeout: Per-subtask timeout in seconds.
            overall_timeout: Wall-clock budget for the entire plan.

        Inspiration: Perplexity Computer (2026-02-27), Swarms (Apache-2.0 kyegomez/swarms)
        """
        deadline = asyncio.get_event_loop().time() + overall_timeout

        # --- Step 1: Decompose ---
        system_prompt = _PLANNER_SYSTEM_PROMPT.format(max_tasks=max_subtasks)
        planner_messages = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]

        try:
            decomp_result = await asyncio.wait_for(
                call_backend(planner_backend, planner_messages),
                timeout=subtask_timeout,
            )
        except asyncio.TimeoutError:
            return PlanResult(goal="", error="Planner timed out during decomposition")

        if not decomp_result.success or not decomp_result.response_body:
            return PlanResult(
                goal="",
                error=f"Planner failed: {decomp_result.error or 'empty response'}",
            )

        # Parse the JSON task graph from the planner response
        raw_text = _extract_text_content(decomp_result.response_body)
        raw_text = _strip_thinking(raw_text)
        raw_text = _strip_json_fences(raw_text).strip()
        try:
            plan_data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.warning("PLAN: could not parse planner JSON: %s — raw: %.200s", exc, raw_text)
            return PlanResult(goal="", error=f"Planner returned invalid JSON: {exc}")

        goal = plan_data.get("goal", "")
        raw_tasks = plan_data.get("tasks", [])[:max_subtasks]
        if not raw_tasks:
            return PlanResult(goal=goal, error="Planner returned no tasks")

        tasks = [
            PlanTask(
                id=t.get("id", f"t{i+1}"),
                description=t.get("description", ""),
                prompt=t.get("prompt", ""),
                capability=t.get("capability", "general").lower(),
                depends_on=t.get("depends_on", []),
            )
            for i, t in enumerate(raw_tasks)
        ]
        logger.info("PLAN: decomposed into %d subtasks for goal: %.80s", len(tasks), goal)

        # --- Step 2: Dispatch (topological batch execution) ---
        task_map = {t.id: t for t in tasks}
        task_results: Dict[str, BackendResult] = {}

        async def run_subtask(task: PlanTask) -> BackendResult:
            cap_nodes = capability_nodes.get(task.capability) or default_nodes
            backend = cap_nodes[0] if cap_nodes else planner_backend
            subtask_messages = [{"role": "user", "content": task.prompt}]
            try:
                remaining = deadline - asyncio.get_event_loop().time()
                t = min(subtask_timeout, max(1.0, remaining))
                return await asyncio.wait_for(call_backend(backend, subtask_messages), timeout=t)
            except asyncio.TimeoutError:
                return BackendResult(backend=backend, error="Subtask timed out")

        # Kahn's algorithm: process in dependency order
        completed: set = set()
        remaining_tasks = list(tasks)
        while remaining_tasks:
            # Find tasks whose dependencies are all completed
            ready = [t for t in remaining_tasks if set(t.depends_on).issubset(completed)]
            if not ready:
                logger.warning("PLAN: dependency cycle or unsatisfied dep; forcing remaining tasks")
                ready = remaining_tasks  # break cycle

            batch_results = await asyncio.gather(*[run_subtask(t) for t in ready])
            for task, result in zip(ready, batch_results):
                task_results[task.id] = result
                completed.add(task.id)
            remaining_tasks = [t for t in remaining_tasks if t.id not in completed]

        # --- Step 3: Assemble ---
        results_text = "\n\n".join(
            f"[{task_map[tid].capability.upper()} — {task_map[tid].description}]\n"
            + (_extract_text_content(res.response_body) if res.success and res.response_body else f"(failed: {res.error})")
            for tid, res in task_results.items()
            if tid in task_map
        )
        assembly_messages = [
            {"role": "system", "content": _ASSEMBLER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Original goal: {goal}\n\nSubtask results:\n{results_text}"},
        ]
        try:
            remaining = deadline - asyncio.get_event_loop().time()
            final = await asyncio.wait_for(
                call_backend(planner_backend, assembly_messages),
                timeout=max(5.0, remaining),
            )
        except asyncio.TimeoutError:
            final = BackendResult(backend=planner_backend, error="Assembler timed out")

        return PlanResult(
            goal=goal,
            tasks=tasks,
            task_results=task_results,
            final_response=final,
        )

    async def execute_consensus(
        self,
        backends: List[str],
        request_fn: Callable[[str], Awaitable[BackendResult]],
        timeout: float = 60.0,
        oracle_backend: Optional[str] = None,
        oracle_present: bool = False,
        is_cloud_fn: Optional[Callable[[str], bool]] = None
    ) -> ConsensusResult:
        """
        Execute request on all backends and compute consensus.

        Args:
            backends: List of backend host:port strings
            request_fn: Async function that takes backend and returns BackendResult
            timeout: Overall timeout in seconds
            oracle_backend: Which backend is designated oracle (authoritative)
            oracle_present: Whether the oracle backend was included in the request
            is_cloud_fn: Optional function to check if a backend is a cloud backend

        Returns:
            ConsensusResult with winner and agreement info
        """
        results = await self.execute_all(backends, request_fn, timeout)
        return compute_consensus(
            results,
            self.similarity_threshold,
            oracle_backend=oracle_backend,
            oracle_present=oracle_present,
            is_cloud_fn=is_cloud_fn
        )
