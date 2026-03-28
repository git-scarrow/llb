from __future__ import annotations

import json
import os
import uuid
import subprocess
import shlex
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Protocol

from .config import OrchestratorConfig
from .types import OrchestrationResult, Plan, Task, TaskResult
from .worker import Worker, task_result_as_dict

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None

try:
    from google import genai  # type: ignore
except ImportError:
    genai = None

try:
    import anthropic
except ImportError:
    anthropic = None


class Planner(Protocol):
    def plan(self, request: str) -> Plan:
        ...


class Verifier(Protocol):
    def verify(self, task: Task, result: TaskResult) -> bool:
        ...


class HeuristicPlanner:
    def __init__(self, max_tasks: int = 16) -> None:
        self._max_tasks = max_tasks

    def plan(self, request: str) -> Plan:
        return Plan(tasks=[Task(id="task-1", description=request)])


class CLIPlanner:
    def __init__(self, model: str = "claude-cli", max_tasks: int = 16) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package not available")
        # Use the LB URL for the client
        lb_url = os.getenv("LLB_URL", os.getenv("AI_LB_URL", f"http://localhost:{os.getenv('LLB_PORT', os.getenv('AI_LB_PORT', '8002'))}"))  # COMPAT: AI_LB_* fallback remove after 2026-06-01
        self._client = OpenAI(base_url=f"{lb_url}/v1", api_key="sk-bridge")
        self._model = model
        self._max_tasks = max_tasks

    def _plan_via_subprocess(self, request: str) -> Plan:
        # Fallback to direct CLI call if bridge is down
        prompt = "\n".join(
            [
                "You are the Orchestrator planner. Break the request into atomic tasks.",
                "Return ONLY valid JSON in this schema:",
                '{"tasks":[{"id":"t1","description":"...", "depends_on":["t0"], "payload":{}}]}',
                "Keep dependencies explicit and keep tasks small.",
                "Here is the request:",
                request
            ]
        )
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tf:
            tf.write(prompt)
            temp_path = tf.name

        # Try Claude first
        cmd_claude = ["claude", "-p", temp_path, "--output-format", "json"]
        # Try Gemini second
        cmd_gemini = ["gemini", "-p", temp_path, "-o", "json"]
        
        for cmd in [cmd_claude, cmd_gemini]:
            try:
                print(f"Executing: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
                return _plan_from_json(result.stdout, max_tasks=self._max_tasks, fallback_request=request)
            except Exception as e:
                print(f"Subprocess Fallback Error ({cmd[0]}): {e}")
                continue

        try:
            os.remove(temp_path)
        except Exception:
            pass
        return Plan(tasks=[Task(id="fallback-1", description=request)])

    def plan(self, request: str) -> Plan:
        prompt = "\n".join(
            [
                "You are the Orchestrator planner. Break the request into atomic tasks.",
                "Return ONLY valid JSON in this schema:",
                '{"tasks":[{"id":"t1","description":"...", "depends_on":["t0"], "payload":{}}]}',
                "Keep dependencies explicit and keep tasks small.",
            ]
        )

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": request},
                ],
                temperature=0.2,
            )
            content = resp.choices[0].message.content or ""
            return _plan_from_json(content, max_tasks=self._max_tasks, fallback_request=request)
        except Exception as e:
            print(f"CLI Bridge Planner Error: {e}. Trying direct subprocess fallback...")
            return self._plan_via_subprocess(request)


class OpenAIPlanner:
    def __init__(self, model: str, max_tasks: int = 16) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package not available")
        self._client = OpenAI()
        self._model = model
        self._max_tasks = max_tasks

    def plan(self, request: str) -> Plan:
        prompt = "\n".join(
            [
                "You are the Orchestrator planner. Break the request into atomic tasks.",
                "Return ONLY valid JSON in this schema:",
                '{"tasks":[{"id":"t1","description":"...", "depends_on":["t0"], "payload":{}}]}',
                "Keep dependencies explicit and keep tasks small.",
            ]
        )

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": request},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            content = resp.choices[0].message.content or ""
            return _plan_from_json(content, max_tasks=self._max_tasks, fallback_request=request)
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return Plan(tasks=[Task(id="fallback-1", description=request)])


class GeminiPlanner:
    def __init__(self, model: str, max_tasks: int = 16) -> None:
        if genai is None:
            raise RuntimeError("google-genai package not available")

        # Initialize client (it reads GOOGLE_API_KEY from env automatically)
        self._client = genai.Client()

        # FIX: Ensure we don't double-prefix or use an invalid alias
        # If the user passed "models/gemini-1.5-flash", strip it.
        # The new SDK often prefers just "gemini-1.5-flash"
        self._model = model.replace("models/", "")
        self._max_tasks = max_tasks

    def plan(self, request: str) -> Plan:
        prompt = "\n".join(
            [
                "You are the Orchestrator planner. Break the request into atomic tasks.",
                "Return ONLY valid JSON in this schema:",
                '{"tasks":[{"id":"t1","description":"...", "depends_on":["t0"], "payload":{}}]}',
                "Keep dependencies explicit and keep tasks small.",
            ]
        )

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=[prompt, request],
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.2,
                },
            )
            return _plan_from_json(response.text, max_tasks=self._max_tasks, fallback_request=request)
        except Exception as e:
            # Fallback: Print the error to help debug API permission issues
            print(f"Gemini API Error: {e}")
            # Return a single-task plan so the agent doesn't crash completely
            return Plan(tasks=[Task(id="fallback-1", description=request)])


class AnthropicPlanner:
    def __init__(self, model: str, max_tasks: int = 16) -> None:
        if anthropic is None:
            raise RuntimeError("anthropic package not available")
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tasks = max_tasks

    def plan(self, request: str) -> Plan:
        prompt = "\n".join(
            [
                "You are the Orchestrator planner. Break the request into atomic tasks.",
                "Return ONLY valid JSON in this schema:",
                '{"tasks":[{"id":"t1","description":"...", "depends_on":["t0"], "payload":{}}]}',
                "Keep dependencies explicit and keep tasks small.",
            ]
        )

        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=prompt,
                messages=[
                    {"role": "user", "content": request},
                ],
                temperature=0.2,
            )
            content = resp.content[0].text if resp.content else ""
            return _plan_from_json(content, max_tasks=self._max_tasks, fallback_request=request)
        except Exception as e:
            print(f"Anthropic API Error: {e}")
            return Plan(tasks=[Task(id="fallback-1", description=request)])


class SimpleVerifier:
    def verify(self, task: Task, result: TaskResult) -> bool:
        return result.success and bool(result.output.strip())


class OpenAIVerifier:
    def __init__(self, model: str) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package not available")
        self._client = OpenAI()
        self._model = model

    def verify(self, task: Task, result: TaskResult) -> bool:
        prompt = "\n".join(
            [
                "You are verifying a Worker output.",
                "Return ONLY JSON: {\"ok\": true|false, \"reason\": \"...\"}",
            ]
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps({"task": task.description, "output": result.output})},
            ],
            temperature=0.0,
        )
        content = resp.choices[0].message.content or ""
        payload = _safe_json(content)
        return bool(payload.get("ok", False))


class Orchestrator:
    def __init__(
        self,
        worker: Optional[Worker] = None,
        config: Optional[OrchestratorConfig] = None,
        planner: Optional[Planner] = None,
        verifier: Optional[Verifier] = None,
    ) -> None:
        self._config = config or OrchestratorConfig()
        self._worker = worker or Worker()
        self._planner = planner or _select_planner(self._config)
        self._verifier = verifier or _select_verifier(self._config)

    def run(self, request: str) -> OrchestrationResult:
        plan = self._planner.plan(request)
        ordered_tasks = _topological_sort(plan.tasks)
        results: Dict[str, TaskResult] = {}
        state: Dict[str, Any] = {"request": request, "plan": [asdict(t) for t in plan.tasks], "results": {}}

        for task in ordered_tasks:
            if task.depends_on:
                deps = {dep: results[dep].output for dep in task.depends_on if dep in results}
                task.payload = {**task.payload, "dependencies": deps}

            result = self._worker.run(task)
            results[task.id] = result
            state["results"][task.id] = task_result_as_dict(result)

            if not self._verifier.verify(task, result):
                break

        return OrchestrationResult(plan=plan, results=results, state=state)


def _select_planner(config: OrchestratorConfig) -> Planner:
    if config.provider == "cli":
        return CLIPlanner(model=config.model or "claude-cli", max_tasks=config.max_tasks)

    if config.provider == "openai" and os.getenv("OPENAI_API_KEY") and OpenAI is not None:
        return OpenAIPlanner(model=config.model, max_tasks=config.max_tasks)

    if config.provider == "gemini" and os.getenv("GOOGLE_API_KEY") and genai is not None:
        return GeminiPlanner(model=config.model, max_tasks=config.max_tasks)

    if config.provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY") and anthropic is not None:
        return AnthropicPlanner(model=config.model, max_tasks=config.max_tasks)

    return HeuristicPlanner(max_tasks=config.max_tasks)


def _select_verifier(config: OrchestratorConfig) -> Verifier:
    if config.verify_with_llm and os.getenv("OPENAI_API_KEY") and OpenAI is not None:
        return OpenAIVerifier(model=config.model)
    return SimpleVerifier()


def _topological_sort(tasks: Iterable[Task]) -> List[Task]:
    task_map = {task.id: task for task in tasks}
    deps = {task.id: set(task.depends_on) for task in tasks}
    ordered: List[Task] = []

    while deps:
        ready = [task_id for task_id, task_deps in deps.items() if not task_deps]
        if not ready:
            raise ValueError("Dependency cycle detected in plan")
        for task_id in sorted(ready):
            deps.pop(task_id, None)
            for remaining in deps.values():
                remaining.discard(task_id)
            ordered.append(task_map[task_id])
    return ordered


def _safe_json(content: str) -> Dict[str, Any]:
    cleaned = _strip_code_fence(content)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {}


def _plan_from_json(content: str, max_tasks: int, fallback_request: str) -> Plan:
    payload = _safe_json(content)
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return Plan(tasks=[Task(id="task-1", description=fallback_request)])

    tasks: List[Task] = []
    for raw in raw_tasks[:max_tasks]:
        if not isinstance(raw, dict):
            continue
        task_id = str(raw.get("id") or f"task-{uuid.uuid4().hex[:8]}")
        desc = str(raw.get("description") or fallback_request)
        depends = raw.get("depends_on") or []
        if isinstance(depends, list):
            depends_on = [str(item) for item in depends]
        else:
            depends_on = []
        payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
        tasks.append(Task(id=task_id, description=desc, depends_on=depends_on, payload=payload))

    return Plan(tasks=tasks or [Task(id="task-1", description=fallback_request)])


def _strip_code_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return text
