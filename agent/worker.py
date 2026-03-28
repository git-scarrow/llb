from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover - optional in minimal test envs
    httpx = None

from .config import WorkerConfig
from .tools.mcp_client import MCPToolRouter, ToolSpec
from .types import LbCallMetadata, Task, TaskResult, ToolCallRecord


class Worker:
    def __init__(self, config: Optional[WorkerConfig] = None, tools: Optional[MCPToolRouter] = None) -> None:
        self._config = config or WorkerConfig()
        if tools is not None:
            self._tools = tools
        elif self._config.mcp_filesystem_cmd or self._config.mcp_shell_cmd:
            self._tools = MCPToolRouter.from_env(startup_timeout_secs=self._config.mcp_startup_timeout_secs)
        else:
            self._tools = None

    def run(self, task: Task) -> TaskResult:
        tool_router = self._tools
        tool_specs: List[ToolSpec] = []
        if tool_router:
            tool_router.start()
            tool_specs = tool_router.tool_catalog()

        messages = self._build_messages(task, tool_specs)
        lb_calls: List[LbCallMetadata] = []
        tool_calls: List[ToolCallRecord] = []
        output = ""
        error: Optional[str] = None
        success = False

        try:
            for _turn in range(self._config.max_turns):
                resp = self._post_chat(messages, task)
                lb_calls.append(self._extract_lb_metadata(resp))
                if resp.status_code >= 400:
                    error = f"LB error {resp.status_code}: {resp.text}"
                    output = error
                    break

                content = self._extract_content(resp)
                action, payload = self._parse_action(content)

                if action == "tool":
                    if not tool_router:
                        error = f"Tool requested but no MCP tools configured: {payload.get('tool')}"
                        output = error
                        break
                    tool_name = payload.get("tool")
                    args = payload.get("arguments", {}) if isinstance(payload.get("arguments"), dict) else {}
                    tool_result = self._invoke_tool(tool_router, tool_name, args)
                    tool_calls.append(tool_result)
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": json.dumps(
                                {"tool_result": {"tool": tool_result.tool, "output": tool_result.output, "error": tool_result.error}}
                            ),
                        }
                    )
                    continue

                output = payload.get("output") if action == "final" else content
                success = True
                break

            if not success and error is None:
                error = f"Turn limit exceeded ({self._config.max_turns})"
                output = error
        finally:
            if tool_router:
                tool_router.close()

        return TaskResult(
            task_id=task.id,
            output=output,
            success=success,
            lb_calls=lb_calls,
            tool_calls=tool_calls,
            error=error,
        )

    def _build_messages(self, task: Task, tool_specs: List[ToolSpec]) -> List[Dict[str, Any]]:
        system_prompt = self._system_prompt(tool_specs)
        user_payload = {"task": task.description, "payload": task.payload}
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload)},
        ]

    def _system_prompt(self, tool_specs: List[ToolSpec]) -> str:
        tool_lines = []
        for tool in tool_specs:
            tool_lines.append(
                json.dumps(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema,
                    }
                )
            )
        tool_section = "\n".join(tool_lines) if tool_lines else "NO_TOOLS_AVAILABLE"
        fs_root = os.getenv("MCP_FILESYSTEM_CMD", "").split(" ")[-1] if os.getenv("MCP_FILESYSTEM_CMD") else "unknown"
        return "\n".join(
            [
                "You are the Worker. Execute the task using reasoning and available tools.",
                f"The allowed filesystem root is: {fs_root}",
                "If you don't know the directory structure or file names, use list_directory first.",
                "Respond ONLY with JSON using one of the following schemas:",
                '{"action": "tool", "tool": "<name>", "arguments": {...}}',
                '{"action": "final", "output": "<result>"}',
                "If tools are unavailable or unnecessary, respond with action=final.",
                "Available tools (JSON per line):",
                tool_section,
            ]
        )

    def _post_chat(self, messages: List[Dict[str, Any]], task: Task) -> httpx.Response:
        if httpx is None:
            raise RuntimeError("httpx is required to call the load balancer")
        payload: Dict[str, Any] = {
            "model": task.payload.get("model", self._config.model),
            "messages": messages,
            "stream": False,
        }
        if "temperature" in task.payload:
            payload["temperature"] = task.payload["temperature"]
        return httpx.post(
            f"{self._config.lb_url}/v1/chat/completions",
            json=payload,
            timeout=httpx.Timeout(self._config.timeout_secs),
        )

    def _extract_content(self, resp: httpx.Response) -> str:
        try:
            data = resp.json()
        except json.JSONDecodeError:
            return resp.text
        choices = data.get("choices", [])
        if not choices:
            return json.dumps(data)
        message = choices[0].get("message", {}) or {}
        content = message.get("content")
        if content is None:
            return json.dumps(message)
        return content

    def _parse_action(self, content: str) -> Tuple[str, Dict[str, Any]]:
        cleaned = self._strip_code_fence(content)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return "final", {"output": content}
        
        if not isinstance(payload, dict):
            return "final", {"output": content}

        action = payload.get("action")
        # Robust check: if 'tool' is present, it's likely a tool call
        if action == "tool" or ("tool" in payload and isinstance(payload["tool"], str)):
            return "tool", payload
        
        if action == "final":
            return "final", payload
            
        return "final", {"output": content}

    def _invoke_tool(self, router: MCPToolRouter, name: Optional[str], args: Dict[str, Any]) -> ToolCallRecord:
        if not name:
            return ToolCallRecord(tool="", arguments=args, output="", error="Missing tool name")
        try:
            result = router.call_tool(name, arguments=args)
            output = self._tool_result_to_text(result)
            return ToolCallRecord(tool=name, arguments=args, output=output)
        except Exception as exc:
            return ToolCallRecord(tool=name, arguments=args, output="", error=str(exc))

    def _tool_result_to_text(self, result: Dict[str, Any]) -> str:
        content = result.get("content")
        if isinstance(content, list):
            parts = []
            for entry in content:
                if isinstance(entry, dict) and entry.get("type") == "text":
                    parts.append(str(entry.get("text", "")))
            if parts:
                return "".join(parts)
        return json.dumps(result)

    def _extract_lb_metadata(self, resp: httpx.Response) -> LbCallMetadata:
        headers = resp.headers
        return LbCallMetadata(
            request_id=headers.get("x-request-id"),
            routed_node=headers.get("x-routed-node"),
            selected_model=headers.get("x-selected-model"),
            attempts=headers.get("x-attempts"),
            failover_count=headers.get("x-failover-count"),
            fallback_model=headers.get("x-fallback-model"),
            status_code=resp.status_code,
        )

    def _strip_code_fence(self, content: str) -> str:
        text = content.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                return "\n".join(lines[1:-1]).strip()
        return text


def task_result_as_dict(result: TaskResult) -> Dict[str, Any]:
    return {
        "task_id": result.task_id,
        "output": result.output,
        "success": result.success,
        "error": result.error,
        "lb_calls": [asdict(call) for call in result.lb_calls],
        "tool_calls": [asdict(call) for call in result.tool_calls],
    }
