#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from typing import Dict, Optional

import httpx


def now() -> float:
    return time.perf_counter()


async def one_nonstream(client: httpx.AsyncClient, lb: str, model: str, prompt: str, node: Optional[str] = None):
    url = f"{lb}/v1/chat/completions"
    if node:
        url += f"?node={node}"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False}
    t0 = now()
    try:
        resp = await client.post(url, json=body)
        dt = now() - t0
        node_hdr = resp.headers.get("x-routed-node", "")
        attempts = int(resp.headers.get("x-attempts", "1") or 1)
        failovers = int(resp.headers.get("x-failover-count", "0") or 0)
        cap = resp.headers.get("x-capacity-state", "")
        ok = resp.status_code < 500
        return {
            "ok": ok,
            "status": resp.status_code,
            "latency": dt,
            "node": node_hdr,
            "attempts": attempts,
            "failovers": failovers,
            "capacity": cap,
        }
    except Exception as e:
        dt = now() - t0
        return {"ok": False, "status": 0, "latency": dt, "node": "", "attempts": 0, "failovers": 0, "capacity": "", "error": str(e)}


async def one_stream(client: httpx.AsyncClient, lb: str, model: str, prompt: str, node: Optional[str] = None):
    url = f"{lb}/v1/chat/completions"
    if node:
        url += f"?node={node}"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True}
    t0 = now()
    ttfb: Optional[float] = None
    bytes_total = 0
    node_hdr = ""
    attempts = 0
    failovers = 0
    cap = ""
    try:
        async with client.stream("POST", url, json=body) as resp:
            node_hdr = resp.headers.get("x-routed-node", "")
            attempts = int(resp.headers.get("x-attempts", "1") or 1)
            failovers = int(resp.headers.get("x-failover-count", "0") or 0)
            cap = resp.headers.get("x-capacity-state", "")
            async for chunk in resp.aiter_bytes():
                if ttfb is None:
                    ttfb = now() - t0
                bytes_total += len(chunk)
        total = now() - t0
        return {
            "ok": True,
            "status": 200,
            "ttfb": ttfb or 0.0,
            "duration": total,
            "bytes": bytes_total,
            "node": node_hdr,
            "attempts": attempts,
            "failovers": failovers,
            "capacity": cap,
        }
    except httpx.HTTPStatusError as e:
        total = now() - t0
        return {"ok": False, "status": e.response.status_code, "duration": total, "node": node_hdr, "attempts": attempts, "failovers": failovers, "capacity": cap, "error": str(e)}
    except Exception as e:
        total = now() - t0
        return {"ok": False, "status": 0, "duration": total, "node": node_hdr, "attempts": attempts, "failovers": failovers, "capacity": cap, "error": str(e)}


def pct(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


async def run(lb: str, model: str, prompt: str, total: int, concurrency: int, streaming: bool):
    results = []
    sem = asyncio.Semaphore(concurrency)

    async def worker(i):
        async with sem:
            fn = one_stream if streaming else one_nonstream
            return await fn(client, lb, model, prompt)

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        start = now()
        tasks = [asyncio.create_task(worker(i)) for i in range(total)]
        for t in asyncio.as_completed(tasks):
            res = await t
            results.append(res)
        elapsed = now() - start

    # Aggregate
    ok = [r for r in results if r.get("ok")]
    err = [r for r in results if not r.get("ok")]
    by_node = Counter(r.get("node", "") for r in ok)
    by_status = Counter(r.get("status", 0) for r in results)
    failover_counts = Counter(r.get("failovers", 0) for r in results)
    attempts = Counter(r.get("attempts", 0) for r in results)
    capacity_states = Counter(r.get("capacity", "") for r in results)

    latencies = [r["latency"] for r in ok if "latency" in r]
    ttfbs = [r["ttfb"] for r in ok if "ttfb" in r and r["ttfb"]]
    durs = [r["duration"] for r in ok if "duration" in r]

    print("\n== Stress Summary ==")
    print(f"LB: {lb}  model: {model}  mode: {'stream' if streaming else 'non-stream'}")
    print(f"Requests: {total}  Concurrency: {concurrency}  Elapsed: {elapsed:.2f}s  RPS: {total/elapsed:.1f}")
    print(f"Success: {len(ok)}  Errors: {len(err)}  Statuses: {dict(by_status)}")
    if latencies:
        print(f"Latency (s) p50={pct(latencies,50):.3f} p90={pct(latencies,90):.3f} p95={pct(latencies,95):.3f} p99={pct(latencies,99):.3f} max={max(latencies):.3f}")
    if ttfbs:
        print(f"TTFB (s)    p50={pct(ttfbs,50):.3f} p90={pct(ttfbs,90):.3f} p95={pct(ttfbs,95):.3f} p99={pct(ttfbs,99):.3f} max={max(ttfbs):.3f}")
    if durs:
        print(f"Duration (s) p50={pct(durs,50):.3f} p90={pct(durs,90):.3f} p95={pct(durs,95):.3f} p99={pct(durs,99):.3f} max={max(durs):.3f}")
    if by_node:
        print(f"Routed per node: {dict(by_node)}")
    if attempts:
        print(f"Attempts histogram: {dict(attempts)}  Failovers: {dict(failover_counts)}")
    if capacity_states:
        print(f"Capacity states: {dict(capacity_states)}")

    if err:
        sample = err[:5]
        print("\nSample errors:")
        for e in sample:
            print(json.dumps(e, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description="Simple async stress tool for LLB")
    ap.add_argument("--lb", default=os.getenv("LLB_URL", os.getenv("AI_LB_URL", f"http://localhost:{os.getenv('LLB_PORT', os.getenv('AI_LB_PORT', '8002'))}")), help="Load balancer base URL")  # COMPAT: AI_LB_* fallback remove after 2026-06-01
    ap.add_argument("--model", default="auto", help="Model id or 'auto'")
    ap.add_argument("--prompt", default="Say hello", help="Prompt to send")
    ap.add_argument("--requests", type=int, default=100, help="Total number of requests")
    ap.add_argument("--concurrency", type=int, default=20, help="Number of concurrent requests")
    ap.add_argument("--stream", action="store_true", help="Use streaming mode")
    args = ap.parse_args()

    asyncio.run(run(args.lb, args.model, args.prompt, args.requests, args.concurrency, args.stream))


if __name__ == "__main__":
    main()

