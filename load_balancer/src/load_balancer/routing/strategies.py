import abc
import time
import json
import random
import math
from typing import List, Optional, Dict, Any
from itertools import cycle

from .. import config as _config

# In-memory state for round-robin cycling
# A dictionary to hold an iterator for each model
ROUND_ROBIN_STATE = {}

class RoutingStrategy(abc.ABC):
    """Abstract base class for all routing strategies."""

    @abc.abstractmethod
    async def select_node(self, nodes: List[str], model_name: str, redis_client) -> str:
        """Selects a node from the available list. Returns a node address (host:port)."""
        pass

class RoundRobinStrategy(RoutingStrategy):
    """Cycles through available nodes for a given model in a round-robin fashion."""

    async def select_node(self, nodes: List[str], model_name: str, redis_client) -> str:
        if not nodes:
            return None

        # Get or create the cycle iterator for this model
        if model_name not in ROUND_ROBIN_STATE or set(ROUND_ROBIN_STATE[model_name]['nodes']) != set(nodes):
            ROUND_ROBIN_STATE[model_name] = {
                'nodes': nodes,
                'iterator': cycle(nodes)
            }
        
        # Get the next node from the cycle
        return next(ROUND_ROBIN_STATE[model_name]['iterator'])

class RandomStrategy(RoutingStrategy):
    """Selects a node randomly from the available list."""

    async def select_node(self, nodes: List[str], model_name: str, redis_client) -> str:
        if not nodes:
            return None
        return random.choice(nodes)

class LeastLoadedStrategy(RoutingStrategy):
    """Selects the node with the fewest in-flight requests (from Redis counters)."""

    async def select_node(self, nodes: List[str], model_name: str, redis_client) -> str:
        if not nodes:
            return None
        best_node = None
        best_load = None
        # Use integer counters: key=node:{node}:inflight
        # Respect optional per-node concurrency caps via key node:{node}:maxconn
        eligible = []
        for n in nodes:
            try:
                inflight_val = await redis_client.get(f"node:{n}:inflight")
                maxconn_val = await redis_client.get(f"node:{n}:maxconn")
                load = int(inflight_val) if inflight_val is not None else 0
                maxconn = int(maxconn_val) if maxconn_val not in (None, "", "0") else None
            except Exception:
                load = 0
                maxconn = None
            # Skip nodes that are at/over their max concurrency
            if maxconn is not None and load >= maxconn:
                continue
            eligible.append((n, load))

        # If all nodes are saturated, return None to signal no capacity
        if not eligible:
            return None

        for n, load in eligible:
            if best_load is None or load < best_load:
                best_load = load
                best_node = n
        return best_node


class PowerOfTwoChoicesStrategy(RoutingStrategy):
    """Power of Two Choices routing: sample 2 eligible nodes, pick the one with lower score.
    Score = inflight_normalized + alpha * p95_latency_ewma + penalty_for_recent_5xx + beta * cost_normalized
    beta=0 (default) disables cost term, preserving existing behavior exactly.
    """

    def __init__(self, alpha: float = 0.5, penalty_weight: float = 2.0, beta: float = 0.0):
        self.alpha = alpha
        self.penalty_weight = penalty_weight
        self.beta = beta

    async def select_node(self, nodes: List[str], model_name: str, redis_client) -> Optional[str]:
        if not nodes:
            return None
        if len(nodes) == 1:
            return nodes[0]

        # Sample 2 nodes randomly
        candidates = random.sample(nodes, min(2, len(nodes)))

        # Calculate base scores for each candidate
        scores = {}
        for node in candidates:
            scores[node] = await self._calculate_node_score(node, model_name, redis_client)

        # Add cost term with min-max normalization across the two candidates (opt-in via beta > 0)
        if self.beta > 0:
            costs = {}
            for node in candidates:
                costs[node] = await self._get_cost_estimate(node, model_name, redis_client)
            cost_vals = list(costs.values())
            cost_min, cost_max = min(cost_vals), max(cost_vals)
            cost_range = cost_max - cost_min
            for node in candidates:
                cost_norm = (costs[node] - cost_min) / cost_range if cost_range > 0 else 0.0
                scores[node] += self.beta * cost_norm

        return min(scores, key=lambda n: scores[n])

    async def _calculate_node_score(self, node: str, model_name: str, redis_client) -> float:
        """Calculate node score based on inflight requests, latency, and failure rate."""
        try:
            # Get inflight requests and normalize
            inflight_val = await redis_client.get(f"node:{node}:inflight")
            maxconn_val = await redis_client.get(f"node:{node}:maxconn")
            inflight = int(inflight_val) if inflight_val is not None else 0
            maxconn = int(maxconn_val) if maxconn_val not in (None, "", "0") else 100  # Default cap
            
            inflight_normalized = inflight / maxconn
            
            # Get p95 latency EWMA for this model+node
            series_key = f"{model_name}|{node}"
            p95_latency = await self._get_p95_latency(series_key, redis_client)
            
            # Get recent 5xx failure rate
            failure_rate = await self._get_failure_rate(node, redis_client)
            
            # Calculate composite score
            score = inflight_normalized + (self.alpha * p95_latency) + (self.penalty_weight * failure_rate)
            
            return score
            
        except Exception:
            # On any error, return high score to deprioritize this node
            return float('inf')

    async def _get_p95_latency(self, series_key: str, redis_client) -> float:
        """Calculate approximate p95 latency from histogram buckets."""
        try:
            buckets = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
            total_count = 0
            cumulative_counts = []
            
            # Get cumulative counts for each bucket
            for le in buckets:
                val = await redis_client.get(f"lb:latency_bucket:{series_key}:{le}")
                count = int(val) if val else 0
                cumulative_counts.append(count)
                if le == float("inf"):
                    total_count = count
            
            if total_count == 0:
                return 0.0
            
            # Find p95 bucket (95th percentile)
            p95_target = total_count * 0.95
            
            for i, count in enumerate(cumulative_counts):
                if count >= p95_target:
                    # Linear interpolation within bucket
                    if i == 0:
                        return buckets[i] * 0.5  # Assume uniform distribution in first bucket
                    
                    lower_bound = buckets[i-1] if i > 0 else 0
                    upper_bound = buckets[i]
                    
                    if i < len(cumulative_counts) - 1:  # Not the infinity bucket
                        prev_count = cumulative_counts[i-1] if i > 0 else 0
                        bucket_range = count - prev_count
                        if bucket_range > 0:
                            position = (p95_target - prev_count) / bucket_range
                            return lower_bound + position * (upper_bound - lower_bound)
                    
                    return upper_bound if upper_bound != float("inf") else 10.0
            
            return 0.0
            
        except Exception:
            return 0.0

    async def _get_failure_rate(self, node: str, redis_client) -> float:
        """Get recent failure rate for the node."""
        try:
            failures = await redis_client.get(f"node:{node}:failures")
            failure_count = int(failures) if failures else 0
            
            # Normalize failure count to a rate (simple approach)
            # This could be enhanced with time-window tracking
            return min(failure_count / 10.0, 1.0)  # Cap at 1.0
            
        except Exception:
            return 0.0

    async def _get_cost_estimate(self, node: str, model_name: str, redis_client) -> float:
        """Return estimated cost (USD) for one request to node based on EWMA output tokens."""
        try:
            pricing = getattr(_config, "BACKEND_COST_PER_TOKEN", {}).get(model_name)
            if not pricing:
                return 0.0
            output_price = float(pricing.get("output", 0.0))  # USD per 1M tokens
            if output_price <= 0:
                return 0.0
            cold_start = float(getattr(_config, "COST_EWMA_COLD_START_TOKENS", 256))
            min_samples = int(getattr(_config, "COST_EWMA_MIN_SAMPLES", 5))
            key_ewma = f"lb:output_tokens_ewma:{model_name}|{node}"
            key_count = f"lb:output_tokens_count:{model_name}|{node}"
            ewma_val = await redis_client.get(key_ewma)
            count_val = await redis_client.get(key_count)
            count = int(count_val) if count_val else 0
            tokens = float(ewma_val) if (ewma_val and count >= min_samples) else cold_start
            return tokens * output_price / 1_000_000.0
        except Exception:
            return 0.0


class ConsistentHashingStrategy(RoutingStrategy):
    """Consistent hashing for sticky sessions."""
    
    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self._hash_ring: Dict[int, str] = {}
        self._nodes: List[str] = []
        
    async def select_node(self, nodes: List[str], model_name: str, redis_client, session_id: Optional[str] = None) -> Optional[str]:
        if not nodes:
            return None
            
        # If no session_id, fall back to random selection
        if not session_id:
            return random.choice(nodes)
            
        # Rebuild hash ring if nodes changed
        if set(nodes) != set(self._nodes):
            self._rebuild_hash_ring(nodes)
            
        # Find node for session
        return self._get_node_for_session(session_id)
        
    def _rebuild_hash_ring(self, nodes: List[str]):
        """Rebuild the consistent hash ring."""
        self._hash_ring.clear()
        self._nodes = nodes[:]
        
        for node in nodes:
            for i in range(self.virtual_nodes):
                virtual_key = f"{node}:{i}"
                hash_value = hash(virtual_key) % (2**32)
                self._hash_ring[hash_value] = node
                
    def _get_node_for_session(self, session_id: str) -> Optional[str]:
        """Get the node responsible for a session."""
        if not self._hash_ring:
            return None
            
        session_hash = hash(session_id) % (2**32)
        
        # Find the first node clockwise from the session hash
        sorted_hashes = sorted(self._hash_ring.keys())
        for hash_value in sorted_hashes:
            if hash_value >= session_hash:
                return self._hash_ring[hash_value]
                
        # Wrap around to the first node
        return self._hash_ring[sorted_hashes[0]]


# Factory to get the strategy instance based on configuration
STRATEGIES = {
    "ROUND_ROBIN": RoundRobinStrategy,
    "RANDOM": RandomStrategy,
    "LEAST_LOADED": LeastLoadedStrategy,
    "P2C": PowerOfTwoChoicesStrategy,
    "POWER_OF_TWO": PowerOfTwoChoicesStrategy,
    "CONSISTENT_HASH": ConsistentHashingStrategy,
}

def get_routing_strategy(strategy_name: str, **kwargs) -> RoutingStrategy:
    strategy_class = STRATEGIES.get(strategy_name.upper())
    if not strategy_class:
        raise ValueError(f"Unknown routing strategy: {strategy_name}")
    
    # Pass configuration parameters for strategies that support them
    if strategy_name.upper() in ("P2C", "POWER_OF_TWO"):
        return strategy_class(
            alpha=kwargs.get("alpha", _config.P2C_ALPHA),
            penalty_weight=kwargs.get("penalty_weight", _config.P2C_PENALTY_WEIGHT),
            beta=kwargs.get("beta", _config.P2C_BETA),
        )
    elif strategy_name.upper() == "CONSISTENT_HASH":
        return strategy_class(virtual_nodes=kwargs.get("virtual_nodes", 150))
    else:
        return strategy_class()
