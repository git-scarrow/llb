import os
import threading
import time
import requests

LB = os.getenv("LLB_URL", os.getenv("AI_LB_URL", f"http://localhost:{os.getenv('LLB_PORT', os.getenv('AI_LB_PORT', '8002'))}"))  # COMPAT: AI_LB_* fallback remove after 2026-06-01
MODEL = os.getenv("EMB_MODEL", "text-embedding-nomic-embed-text-v1.5")
N = int(os.getenv("BURST_N", "60"))
SAMPLE_MS = int(os.getenv("SAMPLE_MS", "100"))
SAMPLES = int(os.getenv("SAMPLES", "40"))


def one(i: int):
    payload = {
        "model": MODEL,
        "input": f"hello from worker {i}",
    }
    try:
        r = requests.post(f"{LB}/v1/embeddings", json=payload, timeout=20)
        _ = r.status_code
    except Exception as e:
        print("worker", i, "error:", e)


def scrape():
    try:
        return requests.get(f"{LB}/metrics", timeout=5).text
    except Exception as e:
        return f"metrics error: {e}"


def main():
    threads = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(N)]
    for t in threads:
        t.start()

    # sample metrics while in-flight
    for _ in range(SAMPLES):
        txt = scrape()
        lines = [ln for ln in txt.splitlines() if ln.startswith("llb_inflight") or ln.startswith("llb_up")]
        print("---\n" + "\n".join(lines))
        time.sleep(SAMPLE_MS / 1000.0)

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()

