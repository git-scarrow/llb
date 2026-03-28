#!/usr/bin/env python3
"""AILB-MT-1 Phase B — Workload generator for complexity routing telemetry.

Sends N requests spread across small/medium/large complexity tiers to the live
LB stack. Progress and per-request results are printed to stdout. The LB writes
JSONL records to COMPLEXITY_ROUTING_LOG (default /tmp/complexity_routing.jsonl).

Usage:
    python scripts/mt1_collect.py [--url http://localhost:8000] [--n 600] [--concurrency 8]
"""
import argparse
import asyncio
import json
import os
import sys
import time
import random

try:
    import httpx
except ImportError:
    sys.exit("httpx is required: pip install httpx")

# ---------------------------------------------------------------------------
# Prompt corpus — spans all three complexity tiers
# ---------------------------------------------------------------------------
SMALL_PROMPTS = [
    "What is 7 times 8?",
    "Say hi.",
    "What color is the sky?",
    "Name a planet in our solar system.",
    "What is the capital of France?",
    "How many days in a week?",
    "Is Python a programming language?",
    "What does HTTP stand for?",
    "What year did World War II end?",
    "Name one primary color.",
    "What is 100 divided by 4?",
    "How many continents are there?",
    "What is the boiling point of water in Celsius?",
    "Who wrote Hamlet?",
    "What is 2 to the power of 10?",
    "Translate 'hello' to Spanish.",
    "What does CPU stand for?",
    "Name the largest ocean.",
    "What is the square root of 144?",
    "Is JSON a markup language?",
]

MEDIUM_PROMPTS = [
    "Explain the difference between TCP and UDP. First, describe TCP, then UDP, next compare them, finally give a use case for each.",
    "Step 1: Define recursion. Step 2: Give a Python example. Step 3: Explain the base case.",
    "Compare SQL and NoSQL databases. First, explain relational models, then document stores, next analyze trade-offs.",
    "Explain gradient descent. First describe the algorithm, then walk through a simple example, finally discuss learning rate.",
    "What is Docker? First explain containers, then images, next volumes, finally give a use case.",
    "Describe the OSI model. Step 1: list the 7 layers. Step 2: describe each layer briefly.",
    "Explain how HTTPS works. First describe TLS handshake, then certificate validation, finally data encryption.",
    "Compare REST and GraphQL APIs. First describe REST constraints, then GraphQL query model, next discuss trade-offs.",
    "What is a load balancer? First, explain horizontal scaling, then describe round-robin routing, finally mention health checks.",
    "Explain CAP theorem. First define consistency, then availability, then partition tolerance, finally give examples.",
    "Step 1: What is a hash function? Step 2: What makes a good hash? Step 3: Give an example use case.",
    "Describe the MVC pattern. First explain Model, then View, then Controller, next describe data flow.",
    "What is eventual consistency? First, explain distributed systems, then describe consistency models, finally give an example.",
    "Compare async and sync programming. First describe blocking I/O, then event loops, next async/await syntax.",
    "Explain the difference between authentication and authorization. First, define each term, then give examples.",
]

LARGE_PROMPTS = [
    "Analyze the trade-offs between microservices and monolithic architectures. Compare them across: deployment complexity, team autonomy, data consistency, observability, and operational cost. Evaluate which suits a 5-person startup vs a 500-person enterprise. Provide a recommendation with reasoning.",
    "Explain why transformer models revolutionized NLP. Analyze: self-attention mechanism, positional encoding, feed-forward layers, and training at scale. Compare to RNNs and CNNs. Evaluate the trade-offs of large model sizes for inference cost vs accuracy.",
    "Design a distributed rate limiter for an API gateway. Analyze: token bucket vs sliding window algorithms, Redis vs local state trade-offs, consistency under network partitions, and failure modes. Evaluate correctness guarantees. Provide pseudocode for the core algorithm.",
    "Compare and evaluate three database indexing strategies: B-tree, LSM-tree, and hash indexes. Analyze write amplification, read performance, space overhead, and compaction costs. Explain why RocksDB chose LSM-tree for write-heavy workloads.",
    "Analyze the pros and cons of GitOps for infrastructure management. Compare pull-based vs push-based deployment models. Evaluate drift detection, secret management, and rollback strategies. Provide a recommendation for a team deploying to Kubernetes.",
    "Explain how Raft consensus algorithm achieves fault tolerance. Analyze: leader election, log replication, safety properties, and split-brain prevention. Compare to Paxos. Evaluate performance trade-offs under network partitions.",
    "Design a content delivery network (CDN) cache invalidation strategy. Analyze: TTL-based expiry, event-driven purge, and surrogate keys. Evaluate consistency vs availability trade-offs. Explain why Fastly chose the surrogate key approach.",
    "Analyze the efficiency trade-offs of different attention mechanisms in transformers: full attention, sparse attention (Longformer), linear attention (Performer), and Flash Attention. Evaluate memory complexity, throughput, and accuracy loss for each approach.",
]

def build_prompt_pool(n: int):
    """Return n (prompt, tier) tuples with roughly equal tier distribution."""
    pool = []
    tiers = [
        (SMALL_PROMPTS, "small"),
        (MEDIUM_PROMPTS, "medium"),
        (LARGE_PROMPTS, "large"),
    ]
    per_tier = n // 3
    remainder = n % 3
    for i, (prompts, tier) in enumerate(tiers):
        count = per_tier + (1 if i < remainder else 0)
        for j in range(count):
            pool.append((prompts[j % len(prompts)], tier))
    random.shuffle(pool)
    return pool


async def send_one(client: "httpx.AsyncClient", url: str, prompt: str, tier: str, sem: asyncio.Semaphore, idx: int, total: int):
    payload = {
        "model": "auto",
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
    }
    t0 = time.monotonic()
    status = "?"
    output_tokens = None
    try:
        async with sem:
            resp = await client.post(f"{url}/v1/chat/completions", json=payload, timeout=60)
        status = resp.status_code
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        if resp.status_code == 200:
            try:
                body = resp.json()
                usage = body.get("usage", {})
                output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
            except Exception:
                pass
        print(f"[{idx+1:4d}/{total}] tier={tier:6s} status={status} elapsed={elapsed_ms}ms tokens={output_tokens}", flush=True)
    except Exception as e:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        print(f"[{idx+1:4d}/{total}] tier={tier:6s} ERROR {e} elapsed={elapsed_ms}ms", flush=True)


async def main():
    parser = argparse.ArgumentParser(description="AILB-MT-1 workload generator")
    parser.add_argument("--url", default=os.getenv("LLB_URL", os.getenv("AI_LB_URL", f"http://localhost:{os.getenv('LLB_PORT', os.getenv('AI_LB_PORT', '8002'))}")), help="LB base URL")  # COMPAT: AI_LB_* fallback remove after 2026-06-01
    parser.add_argument("--n", type=int, default=600, help="Total requests to send")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent requests")
    args = parser.parse_args()

    pool = build_prompt_pool(args.n)
    sem = asyncio.Semaphore(args.concurrency)

    print(f"AILB-MT-1 Phase B: sending {args.n} requests to {args.url}")
    print(f"Tier distribution: ~{args.n//3} small / ~{args.n//3} medium / ~{args.n//3} large")
    print(f"Concurrency: {args.concurrency}")
    print(f"Log output: check COMPLEXITY_ROUTING_LOG on the LB (default /tmp/complexity_routing.jsonl)")
    print()

    t_start = time.monotonic()
    async with httpx.AsyncClient() as client:
        tasks = [
            send_one(client, args.url, prompt, tier, sem, i, args.n)
            for i, (prompt, tier) in enumerate(pool)
        ]
        await asyncio.gather(*tasks)

    elapsed = time.monotonic() - t_start
    print(f"\nDone. {args.n} requests in {elapsed:.1f}s ({args.n/elapsed:.1f} req/s)")
    print("Next: copy /tmp/complexity_routing.jsonl from the LB and run scripts/mt1_analyze.py")


if __name__ == "__main__":
    asyncio.run(main())
