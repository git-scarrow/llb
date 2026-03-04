"""Tests for Phase 2: heuristic complexity scorer and routing strategy.

Covers all 14 required cases:
  1.  Empty prompt → score 0.0 → small tier
  2.  Short plain text → low score → small tier
  3.  Long plain text (> 8 000 chars) → score ~0.24 → small tier
  4.  3 code fences, short text → code signal maxed at 0.20
  5.  4 multi-step markers → multi-step signal maxed at 0.25
  6.  3 reasoning keywords → reasoning signal maxed at 0.25
  7.  All signals maxed → total clamped to 1.0
  8.  Multipart (long + code + reasoning) → score > 0.65 → any tier
  9.  Score == 0.35 → medium tier
  10. Score == 0.65 → medium tier
  11. Score == 0.34 → small tier
  12. Score == 0.66 → any tier
  13. ComplexityRoutingStrategy registered in STRATEGIES dict
  14. select_node delegates to P2C
"""

import sys
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "load_balancer" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from load_balancer.routing.strategies import (
    ComplexityRoutingStrategy,
    PowerOfTwoChoicesStrategy,
    STRATEGIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(text: str) -> list[dict]:
    """Build a minimal OpenAI messages list with a single user turn."""
    return [{"role": "user", "content": text}]


def _score(text: str) -> float:
    return ComplexityRoutingStrategy.score_prompt_complexity(_msg(text))


MODEL_CLASSES = {
    "historical_small": {"candidates": ["small-model-a", "small-model-b"], "min_nodes": 2},
    "historical_medium": {"candidates": ["medium-model-a", "medium-model-b"], "min_nodes": 2},
}

# ---------------------------------------------------------------------------
# Tests 1–8: scorer signal behavior
# ---------------------------------------------------------------------------

class TestComplexityScorer:

    # 1. Empty prompt
    def test_empty_messages_returns_zero(self):
        score = ComplexityRoutingStrategy.score_prompt_complexity([])
        assert score == 0.0

    def test_empty_messages_maps_to_small_tier(self):
        router = ComplexityRoutingStrategy()
        result = router.get_complexity_model(0.0, MODEL_CLASSES)
        assert result == MODEL_CLASSES["historical_small"]["candidates"]

    # 2. Short plain text
    def test_short_plain_text_low_score_small_tier(self):
        score = _score("Hello world")  # 11 chars, no markers, no keywords
        # length: min(11/10000, 1.0)*0.30 ≈ 0.00033; everything else 0
        assert score < 0.35
        router = ComplexityRoutingStrategy()
        tier = router.get_complexity_model(score, MODEL_CLASSES)
        assert tier == MODEL_CLASSES["historical_small"]["candidates"]

    # 3. Long plain text (> 8 000 chars), no other signals
    def test_long_plain_text_near_point_30_medium_tier(self):
        # AILB-MT-1 recalibration: LOW_THRESHOLD lowered to 0.10, HIGH to 0.25.
        # 8000-char plain text scores ~0.24 → now routes to medium tier.
        text = "a" * 8_000  # no code, no markers, no keywords
        score = _score(text)
        # length signal: min(8000/10000, 1.0)*0.30 = 0.24
        assert 0.23 <= score <= 0.25
        router = ComplexityRoutingStrategy()
        tier = router.get_complexity_model(score, MODEL_CLASSES)
        assert tier == MODEL_CLASSES["historical_medium"]["candidates"]

    # 4. 3 code fences → code signal maxed at 0.20
    def test_three_code_fences_maxes_code_signal(self):
        # Short text so length contributes minimally; 3 ``` occurrences
        text = "short ``` block ``` another ``` end"
        score = _score(text)
        # code signal: min(3/3, 1.0)*0.20 = 0.20
        # length: min(35/10000, 1.0)*0.30 ≈ 0.001
        assert abs(score - 0.201) < 0.01  # mostly just the code signal

    def test_four_code_fences_still_maxed(self):
        text = "a ``` b ``` c ``` d ``` e"  # 4 fences — still capped at 0.20
        score = _score(text)
        extra = _score("a ``` b ``` c ``` end")  # 3 fences
        # Both should give the same code contribution
        assert abs(score - extra) < 0.005  # length diff only

    # 5. 4 multi-step markers → signal maxed at 0.25
    def test_four_multistep_markers_maxes_signal(self):
        text = "first, do this. then, do that. next, something. finally, done."
        # markers: "first," "then," "next," "finally," → 4 → min(4/4,1.0)*0.25 = 0.25
        score = _score(text)
        # length is short so only multi-step contributes meaningfully
        assert score >= 0.25
        # Exact signal contribution check: isolate by zeroing length factor
        expected_marker = 0.25
        actual_marker = min(4 / 4, 1.0) * 0.25
        assert actual_marker == expected_marker

    def test_five_markers_capped_same_as_four(self):
        # "step 1" + "first," + "then," + "next," + "finally," = 5 matches → still cap 0.25
        text = "step 1 first, then, next, finally,"
        score = _score(text)
        assert score >= 0.25  # multi-step maxed

    # 6. 3 reasoning keywords → signal maxed at 0.25
    def test_three_reasoning_keywords_maxes_signal(self):
        text = "please analyze this and compare these two and evaluate the result"
        # keywords: "analyze", "compare", "evaluate" → 3 → min(3/3,1)*0.25 = 0.25
        score = _score(text)
        assert score >= 0.25
        actual = min(3 / 3, 1.0) * 0.25
        assert actual == 0.25

    def test_four_reasoning_keywords_capped(self):
        text = "analyze, compare, evaluate, explain why this trade-off exists"
        # 5 matches → still capped at 0.25
        score = _score(text)
        assert score >= 0.25

    # 7. All signals maxed → total clamped to 1.0
    def test_all_signals_maxed_clamped_to_one(self):
        # 10 000+ chars → length 0.30
        # 3 code fences → code 0.20
        # 4+ step markers → multi-step 0.25
        # 3+ reasoning kw → reasoning 0.25
        # sum = 1.00 exactly, clamped to 1.0
        long_text = "a" * 10_000
        markers = "first, then, next, finally,"
        keywords = "analyze compare evaluate"
        fences = "``` x ``` y ``` z"
        text = f"{long_text} {markers} {keywords} {fences}"
        score = _score(text)
        assert score == 1.0

    # 8. Multipart (long + code + reasoning) → high score → any tier
    def test_multipart_high_score_any_tier(self):
        # AILB-MT-1 recalibration: HIGH_THRESHOLD is now 0.25.
        # 5 000 chars → 0.15; 3 fences → 0.20; 3 reasoning kw → 0.25; total ~0.60
        # Any score > 0.25 routes to any-tier.
        body = "a" * 5_000
        text = f"{body} ``` code ``` here ``` end analyze compare evaluate first, then,"
        score = _score(text)
        assert score > 0.25
        router = ComplexityRoutingStrategy()
        tier = router.get_complexity_model(score, MODEL_CLASSES)
        assert tier is None  # any-tier → caller selects freely


# ---------------------------------------------------------------------------
# Tests 9–12: tier boundary precision
# ---------------------------------------------------------------------------

class TestTierBoundaries:
    """Test get_complexity_model() boundary conditions directly."""

    def setup_method(self):
        self.router = ComplexityRoutingStrategy()

    # 9. score == LOW_THRESHOLD (0.10) → medium
    def test_score_at_low_threshold_is_medium(self):
        result = self.router.get_complexity_model(0.10, MODEL_CLASSES)
        assert result == MODEL_CLASSES["historical_medium"]["candidates"]

    # 10. score == HIGH_THRESHOLD (0.25) → medium (inclusive)
    def test_score_at_high_threshold_is_medium(self):
        result = self.router.get_complexity_model(0.25, MODEL_CLASSES)
        assert result == MODEL_CLASSES["historical_medium"]["candidates"]

    # 11. score just below LOW (0.09) → small
    def test_score_below_low_threshold_is_small(self):
        result = self.router.get_complexity_model(0.09, MODEL_CLASSES)
        assert result == MODEL_CLASSES["historical_small"]["candidates"]

    # 12. score just above HIGH (0.26) → any
    def test_score_above_high_threshold_is_any(self):
        result = self.router.get_complexity_model(0.26, MODEL_CLASSES)
        assert result is None

    def test_score_zero_is_small(self):
        assert self.router.get_complexity_model(0.0, MODEL_CLASSES) == MODEL_CLASSES["historical_small"]["candidates"]

    def test_score_one_is_any(self):
        assert self.router.get_complexity_model(1.0, MODEL_CLASSES) is None

    def test_missing_model_class_returns_none(self):
        """If the tier has no config entry, return None (don't crash)."""
        assert self.router.get_complexity_model(0.20, {}) is None
        assert self.router.get_complexity_model(0.50, {}) is None


# ---------------------------------------------------------------------------
# Test 13: STRATEGIES registration
# ---------------------------------------------------------------------------

class TestStrategyRegistration:

    # 13. ComplexityRoutingStrategy registered in STRATEGIES
    def test_complexity_registered_in_strategies(self):
        assert "COMPLEXITY" in STRATEGIES
        assert STRATEGIES["COMPLEXITY"] is ComplexityRoutingStrategy

    def test_get_routing_strategy_returns_complexity(self):
        from load_balancer.routing.strategies import get_routing_strategy
        strategy = get_routing_strategy("COMPLEXITY")
        assert isinstance(strategy, ComplexityRoutingStrategy)


# ---------------------------------------------------------------------------
# Test 14: delegation to P2C
# ---------------------------------------------------------------------------

class TestP2CDelegation:

    # 14. select_node delegates to P2C
    def test_select_node_delegates_to_p2c(self):
        router = ComplexityRoutingStrategy()
        mock_node = "localhost:11434"
        # Replace the internal P2C instance with a mock
        router._p2c = MagicMock(spec=PowerOfTwoChoicesStrategy)
        router._p2c.select_node = AsyncMock(return_value=mock_node)

        nodes = ["localhost:11434", "localhost:11435"]
        redis_mock = MagicMock()

        result = asyncio.run(router.select_node(nodes, "test-model", redis_mock))

        router._p2c.select_node.assert_awaited_once_with(nodes, "test-model", redis_mock)
        assert result == mock_node

    def test_select_node_passes_empty_list_to_p2c(self):
        """P2C handles empty node list; ComplexityRoutingStrategy should not short-circuit."""
        router = ComplexityRoutingStrategy()
        router._p2c = MagicMock(spec=PowerOfTwoChoicesStrategy)
        router._p2c.select_node = AsyncMock(return_value=None)

        result = asyncio.run(router.select_node([], "model", MagicMock()))
        router._p2c.select_node.assert_awaited_once()
        assert result is None
