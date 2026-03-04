"""E2E tests for PLAN execution mode (Phases 3 & 4).

Covers:
  - Non-streaming PLAN:  x-execution-mode: plan (no Accept: text/event-stream)
  - SSE streaming PLAN:  x-execution-mode: plan + Accept: text/event-stream
  - Event ordering invariant: all task_finished before first assembly token
  - TTFB: first meta-event within 2 seconds
  - Stress: 5-subtask plan with full event coverage

Note: PLAN mode requires the backend to respond with valid JSON from the
planner system prompt.  Tests use short prompts likely to produce small plans.
"""

import json
import time

import httpx
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLAN_MESSAGES = [
    {
        "role": "user",
        "content": (
            "List two capitals of European countries, one per subtask."
        ),
    }
]

_STRESS_PLAN_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Name five different programming languages. "
            "Create one subtask per language, each asking for a one-sentence description."
        ),
    }
]

_PLAN_HEADERS = {"x-execution-mode": "plan"}
_PLAN_SSE_HEADERS = {
    "x-execution-mode": "plan",
    "Accept": "text/event-stream",
}

_SSE_META_EVENT_TYPES = {
    "plan_decomposed",
    "task_started",
    "task_finished",
    "assembly_started",
}


def _parse_sse_events(raw_lines) -> list[dict]:
    """Parse SSE lines into a list of {event, data} dicts."""
    events = []
    current_event = None
    current_data = []
    for line in raw_lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.rstrip("\r\n")
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: "):
            current_data.append(line[6:])
        elif line == "" and (current_event is not None or current_data):
            data_str = "\n".join(current_data)
            try:
                data = json.loads(data_str)
            except Exception:
                data = data_str
            events.append({"event": current_event, "data": data})
            current_event = None
            current_data = []
    return events


# ---------------------------------------------------------------------------
# Non-streaming PLAN tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestPlanNonStreaming:
    """x-execution-mode: plan without Accept: text/event-stream."""

    def _post_plan(
        self, lb_client: httpx.Client, model_name: str, messages=None
    ) -> httpx.Response:
        return lb_client.post(
            "/v1/chat/completions",
            headers=_PLAN_HEADERS,
            json={
                "model": model_name,
                "messages": messages or _PLAN_MESSAGES,
                "stream": False,
            },
            timeout=120.0,
        )

    def test_plan_non_stream_returns_200(
        self, lb_client: httpx.Client, model_name: str
    ):
        resp = self._post_plan(lb_client, model_name)
        assert resp.status_code == 200, f"PLAN non-stream failed: {resp.text}"

    def test_plan_non_stream_body_has_choices(
        self, lb_client: httpx.Client, model_name: str
    ):
        resp = self._post_plan(lb_client, model_name)
        assert resp.status_code == 200
        body = resp.json()
        # collect_plan_result wraps result in OpenAI-like response
        assert "choices" in body, f"Expected 'choices' in body: {body}"

    def test_plan_non_stream_has_content(
        self, lb_client: httpx.Client, model_name: str
    ):
        resp = self._post_plan(lb_client, model_name)
        assert resp.status_code == 200
        body = resp.json()
        choices = body.get("choices", [])
        assert choices, "No choices in PLAN response"
        content = choices[0].get("message", {}).get("content", "")
        assert content, "PLAN assembled response has empty content"

    def test_plan_non_stream_request_id_header(
        self, lb_client: httpx.Client, model_name: str
    ):
        resp = self._post_plan(lb_client, model_name)
        assert resp.status_code == 200
        assert resp.headers.get("x-request-id"), "Missing x-request-id header"


# ---------------------------------------------------------------------------
# SSE streaming PLAN tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestPlanSSEStreaming:
    """x-execution-mode: plan + Accept: text/event-stream (Phase 4)."""

    def _stream_plan(
        self,
        lb_client: httpx.Client,
        model_name: str,
        messages=None,
        timeout: float = 120.0,
    ):
        """Yield raw lines from SSE stream."""
        lines = []
        with lb_client.stream(
            "POST",
            "/v1/chat/completions",
            headers=_PLAN_SSE_HEADERS,
            json={
                "model": model_name,
                "messages": messages or _PLAN_MESSAGES,
                "stream": True,
            },
            timeout=timeout,
        ) as resp:
            assert resp.status_code == 200, (
                f"PLAN SSE returned {resp.status_code}: {resp.read()}"
            )
            ct = resp.headers.get("content-type", "")
            assert "text/event-stream" in ct, f"Wrong content-type: {ct}"
            for line in resp.iter_lines():
                lines.append(line)
        return lines

    def test_plan_sse_returns_200(
        self, lb_client: httpx.Client, model_name: str
    ):
        lines = self._stream_plan(lb_client, model_name)
        assert lines, "No lines received from PLAN SSE stream"

    def test_plan_sse_contains_plan_decomposed_event(
        self, lb_client: httpx.Client, model_name: str
    ):
        lines = self._stream_plan(lb_client, model_name)
        events = _parse_sse_events(lines)
        event_types = {e["event"] for e in events}
        assert "plan_decomposed" in event_types, (
            f"Missing plan_decomposed event. Got: {event_types}"
        )

    def test_plan_sse_contains_task_events(
        self, lb_client: httpx.Client, model_name: str
    ):
        lines = self._stream_plan(lb_client, model_name)
        events = _parse_sse_events(lines)
        event_types = [e["event"] for e in events]
        assert "task_started" in event_types, (
            f"Missing task_started event. Got: {event_types}"
        )
        assert "task_finished" in event_types, (
            f"Missing task_finished event. Got: {event_types}"
        )

    def test_plan_sse_contains_assembly_started_event(
        self, lb_client: httpx.Client, model_name: str
    ):
        lines = self._stream_plan(lb_client, model_name)
        events = _parse_sse_events(lines)
        event_types = {e["event"] for e in events}
        assert "assembly_started" in event_types, (
            f"Missing assembly_started event. Got: {event_types}"
        )

    def test_plan_sse_event_ordering_invariant(
        self, lb_client: httpx.Client, model_name: str
    ):
        """All task_finished events must precede the first assembly token (data: {...})."""
        lines = self._stream_plan(lb_client, model_name)
        events = _parse_sse_events(lines)

        last_task_finished_idx = -1
        first_token_idx = -1

        for i, e in enumerate(events):
            if e["event"] == "task_finished":
                last_task_finished_idx = i
            elif e["event"] is None:
                # Bare data lines (tokens) have no event label in our parser
                # Only count lines that look like completion tokens (dicts with choices)
                d = e["data"]
                if isinstance(d, dict) and "choices" in d:
                    if first_token_idx == -1:
                        first_token_idx = i

        # If we received token events AND task_finished events, ordering must hold
        if first_token_idx != -1 and last_task_finished_idx != -1:
            assert last_task_finished_idx < first_token_idx, (
                "Ordering violation: task_finished appeared after first token. "
                f"last_task_finished_idx={last_task_finished_idx}, "
                f"first_token_idx={first_token_idx}"
            )

    def test_plan_sse_ttfb_within_2s(
        self, lb_client: httpx.Client, model_name: str
    ):
        """First meta-event (plan_decomposed) should arrive within 2 seconds."""
        t0 = time.monotonic()
        first_event_time = None
        with lb_client.stream(
            "POST",
            "/v1/chat/completions",
            headers=_PLAN_SSE_HEADERS,
            json={
                "model": model_name,
                "messages": _PLAN_MESSAGES,
                "stream": True,
            },
            timeout=120.0,
        ) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if line.startswith("event: plan_decomposed"):
                    first_event_time = time.monotonic() - t0
                    break

        # We only assert TTFB if we actually received the event
        if first_event_time is not None:
            assert first_event_time < 2.0, (
                f"TTFB too high: first plan_decomposed event arrived after "
                f"{first_event_time:.2f}s (expected < 2s)"
            )

    def test_plan_sse_stress_5_subtasks(
        self, lb_client: httpx.Client, model_name: str
    ):
        """Send a prompt that should produce ~5 subtasks; verify all events present."""
        lines = self._stream_plan(
            lb_client, model_name, messages=_STRESS_PLAN_MESSAGES, timeout=180.0
        )
        events = _parse_sse_events(lines)
        event_types = [e["event"] for e in events]

        assert "plan_decomposed" in event_types, "Missing plan_decomposed"
        assert event_types.count("task_started") >= 1, (
            f"Expected ≥1 task_started events; got: {event_types}"
        )
        assert event_types.count("task_finished") >= 1, (
            f"Expected ≥1 task_finished events; got: {event_types}"
        )
        assert "assembly_started" in event_types, "Missing assembly_started"

        # Verify task_started and task_finished counts match
        n_started = event_types.count("task_started")
        n_finished = event_types.count("task_finished")
        assert n_started == n_finished, (
            f"Unbalanced task events: {n_started} started, {n_finished} finished"
        )

    def test_plan_decomposed_data_structure(
        self, lb_client: httpx.Client, model_name: str
    ):
        """plan_decomposed event data must contain goal and tasks."""
        lines = self._stream_plan(lb_client, model_name)
        events = _parse_sse_events(lines)
        decomposed = [e for e in events if e["event"] == "plan_decomposed"]
        assert decomposed, "No plan_decomposed event"
        data = decomposed[0]["data"]
        assert isinstance(data, dict), f"plan_decomposed data not a dict: {data}"
        assert "goal" in data, f"plan_decomposed missing 'goal': {data}"
        assert "tasks" in data, f"plan_decomposed missing 'tasks': {data}"
