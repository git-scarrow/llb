"""E2E: the /v1/chat/completions SSE stream is OpenAI-spec clean.

Regression guard for the LiteLLM `MidStreamFallbackError: ... list index out of
range` incident (2026-06-25). LLB used to emit its per-attempt observability
(node / failover / hedge_start / hedge_winner) as ``data:`` chunks with NO
``choices`` array. Strict OpenAI stream parsers (LiteLLM's CustomStreamWrapper,
the OpenAI SDK) do ``chunk["choices"][0]`` on every ``data:`` line and raise
``IndexError`` on the empty list -- hard-failing any model with no fallback
(e.g. llama3.1:8b), even though the model itself answered fine.

The fix moves that metadata onto SSE *comment* lines (``: llb-obs {...}``), which
the spec says every data-only client must ignore. These tests assert:

  1. every ``data:`` line is ``[DONE]`` or JSON with a non-empty ``choices`` list
     (or a recognised ``{"error": ...}`` terminal chunk), and
  2. the observability signal is still on the wire as a ``: `` comment, so the
     feature was relocated, not silently dropped.

Gated on LLB_BASE_URL like the rest of the e2e suite (see conftest.py).
"""

import json

import httpx
import pytest


def _stream_lines(lb_client: httpx.Client, model: str):
    """Yield raw SSE lines (data:, event:, and ':' comments) for a short stream."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "stream": True,
        "max_tokens": 10,
    }
    with lb_client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200, f"stream returned {resp.status_code}"
        ct = resp.headers.get("content-type", "")
        assert "text/event-stream" in ct, f"Unexpected content-type: {ct}"
        for line in resp.iter_lines():
            yield line


@pytest.mark.e2e
class TestSSESpecCompliance:
    """Strict-OpenAI-client safety of the streaming chat endpoint."""

    def test_no_choices_less_data_chunk(
        self, lb_client: httpx.Client, model_name: str
    ):
        """Every data: chunk parses and exposes choices[0] -- the exact op a
        strict parser performs. A choices-less non-error chunk is the bug."""
        data_chunks = 0
        for line in _stream_lines(lb_client, model_name):
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                continue
            data_chunks += 1
            obj = json.loads(payload)  # must be valid JSON
            if isinstance(obj.get("error"), dict):
                continue  # recognised terminal error chunk -- clients raise, don't index
            choices = obj.get("choices")
            assert isinstance(choices, list) and len(choices) >= 1, (
                "data: chunk has no usable choices[0] -- strict OpenAI parsers "
                f"(LiteLLM) raise 'list index out of range' here: {payload[:200]}"
            )
            _ = choices[0]  # the operation that used to throw
        assert data_chunks >= 1, "stream produced no content data: chunks"

    def test_observability_metadata_relocated_to_comment(
        self, lb_client: httpx.Client, model_name: str
    ):
        """The node/failover metadata must still be observable -- now as an SSE
        comment (': llb-obs ...'), never as a choices-less data: chunk."""
        saw_obs_comment = False
        for line in _stream_lines(lb_client, model_name):
            if line.startswith(": llb-obs"):
                saw_obs_comment = True
                meta = json.loads(line[len(": llb-obs"):].strip())
                assert "node" in meta and "event" in meta, (
                    f"obs comment missing node/event: {line[:200]}"
                )
        assert saw_obs_comment, (
            "no ': llb-obs' comment seen -- observability was dropped, not relocated"
        )
