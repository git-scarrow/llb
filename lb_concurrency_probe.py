import json
import os
import threading
import time
from typing import List

import requests

LB = os.getenv("LLB_URL", os.getenv("AI_LB_URL", f"http://localhost:{os.getenv('LLB_PORT', os.getenv('AI_LB_PORT', '8002'))}"))  # COMPAT: AI_LB_* fallback remove after 2026-06-01


def get_model() -> str:
    resp = requests.get(f"{LB}/v1/models", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    # Prefer a known-good model id if present
    preferred = ["qwen/qwen3-8b", "mistralai/devstral-small-2507", "mistralai/mistral-small-3.2"]
    ids = [m["id"] for m in data.get("data", [])]
    for p in preferred:
        if p in ids:
            return p
    return ids[0]


def stream_once(model: str, idx: int):
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": f"Say hello from worker {idx} in 3 words"}],
            "stream": True,
        }
        with requests.post(f"{LB}/v1/chat/completions", json=payload, stream=True, timeout=60) as r:
            # Read a few chunks slowly to keep the stream open
            it = r.iter_lines()
            for _ in range(5):
                try:
                    next(it)
                except StopIteration:
                    break
                time.sleep(0.3)
            # Keep it open briefly to observe inflight
            time.sleep(2.0)
    except Exception as e:
        print(f"worker {idx} error: {e}")


def scrape_inflight(samples: int = 8, delay: float = 0.5) -> List[str]:
    out = []
    for _ in range(samples):
        try:
            text = requests.get(f"{LB}/metrics", timeout=5).text
            lines = [ln for ln in text.splitlines() if ln.startswith("llb_inflight") or ln.startswith("llb_up")]
            out.append("\n".join(lines))
        except Exception as e:
            out.append(f"metrics error: {e}")
        time.sleep(delay)
    return out


def main():
    import os, sys
    model = None
    if len(sys.argv) > 1:
        model = sys.argv[1]
    if not model:
        model = get_model()
    print("Using model:", model)

    # Start N workers (env override LB_PROBE_N)
    N = int(os.getenv("LB_PROBE_N", "6"))
    threads = [threading.Thread(target=stream_once, args=(model, i), daemon=True) for i in range(N)]
    for t in threads:
        t.start()

    # Sample metrics while requests are in-flight
    snapshots = scrape_inflight()
    print("\nMetrics snapshots (llb_up + llb_inflight):")
    for s in snapshots:
        print("---\n" + s)

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
