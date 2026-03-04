#!/usr/bin/env python3
"""AILB-MT-1 Phase C — Complexity routing telemetry analysis.

Reads the JSONL log produced by the instrumented LB and reports:
1. Tier distribution and score histograms
2. Latency by tier (p50/p95)
3. Output token distribution by tier (proxy for complexity signal)
4. Threshold sensitivity sweep (0.20–0.80 in 0.05 steps)
5. Recommendation: keep / tune / remove complexity routing

Usage:
    python scripts/mt1_analyze.py [--log /tmp/complexity_routing.jsonl] [--out report.md]
"""
import argparse
import json
import sys
import math
from pathlib import Path
from collections import defaultdict


def load_records(path: str):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def mean(values):
    return sum(values) / len(values) if values else None


def score_to_tier(score, low=0.35, high=0.65):
    if score < low:
        return "small"
    if score <= high:
        return "medium"
    return "large"


def analyze(records, out):
    n = len(records)
    print(f"# AILB-MT-1 Phase C — Complexity Routing Analysis", file=out)
    print(f"\n**Total records:** {n}", file=out)
    if n < 50:
        print("\n⚠️  Fewer than 50 records — results may not be statistically meaningful.", file=out)

    # Split by tier
    by_tier = defaultdict(list)
    for r in records:
        by_tier[r.get("complexity_tier", "unknown")].append(r)

    # --- 1. Tier distribution ---
    print("\n## 1. Tier Distribution\n", file=out)
    print(f"| Tier   | Count | % of total |", file=out)
    print(f"|--------|-------|------------|", file=out)
    for tier in ["small", "medium", "large"]:
        c = len(by_tier[tier])
        pct = 100 * c / n if n else 0
        print(f"| {tier:6s} | {c:5d} | {pct:9.1f}% |", file=out)

    # --- 2. Score histograms ---
    print("\n## 2. Complexity Score Distribution\n", file=out)
    buckets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bucket_counts = defaultdict(int)
    for r in records:
        s = r.get("complexity_score", 0)
        for b in buckets:
            if s <= b:
                bucket_counts[b] += 1
                break
        else:
            bucket_counts[1.0] += 1
    print("| Score range | Count |", file=out)
    print("|-------------|-------|", file=out)
    prev = 0.0
    for b in buckets:
        c = bucket_counts[b]
        print(f"| {prev:.1f}–{b:.1f}      | {c:5d} |", file=out)
        prev = b

    # --- 3. Latency by tier ---
    print("\n## 3. Latency by Tier (ms)\n", file=out)
    print("| Tier   | p50   | p95   | mean  | count |", file=out)
    print("|--------|-------|-------|-------|-------|", file=out)
    for tier in ["small", "medium", "large"]:
        lats = [r["elapsed_ms"] for r in by_tier[tier] if r.get("elapsed_ms") is not None and r.get("status_code") == 200]
        if not lats:
            print(f"| {tier:6s} | —     | —     | —     | 0     |", file=out)
            continue
        p50 = percentile(lats, 50)
        p95 = percentile(lats, 95)
        avg = mean(lats)
        print(f"| {tier:6s} | {p50:5.0f} | {p95:5.0f} | {avg:5.0f} | {len(lats):5d} |", file=out)

    # --- 4. Output tokens by tier ---
    print("\n## 4. Output Tokens by Tier\n", file=out)
    print("| Tier   | p50   | p95   | mean  | count |", file=out)
    print("|--------|-------|-------|-------|-------|", file=out)
    for tier in ["small", "medium", "large"]:
        toks = [r["output_tokens"] for r in by_tier[tier]
                if r.get("output_tokens") is not None and r.get("status_code") == 200]
        if not toks:
            print(f"| {tier:6s} | —     | —     | —     | 0     |", file=out)
            continue
        p50 = percentile(toks, 50)
        p95 = percentile(toks, 95)
        avg = mean(toks)
        print(f"| {tier:6s} | {p50:5.0f} | {p95:5.0f} | {avg:5.0f} | {len(toks):5d} |", file=out)

    # --- 5. Threshold sensitivity sweep ---
    print("\n## 5. Threshold Sensitivity Sweep\n", file=out)
    print("How tier assignment changes as LOW/HIGH thresholds vary:\n", file=out)
    thresholds = [(round(lo, 2), round(hi, 2))
                  for lo in [x/100 for x in range(20, 55, 5)]
                  for hi in [x/100 for x in range(50, 85, 5)]
                  if lo < hi]
    scores = [r["complexity_score"] for r in records if r.get("complexity_score") is not None]

    print("| LOW  | HIGH | small% | medium% | large% |", file=out)
    print("|------|------|--------|---------|--------|", file=out)
    for lo, hi in thresholds:
        if abs(lo - 0.35) < 0.01 and abs(hi - 0.65) < 0.01:
            marker = " ← current"
        else:
            marker = ""
        small_n = sum(1 for s in scores if s < lo)
        large_n = sum(1 for s in scores if s > hi)
        med_n = len(scores) - small_n - large_n
        total = len(scores) or 1
        print(
            f"| {lo:.2f} | {hi:.2f} | {100*small_n/total:6.1f}% | {100*med_n/total:7.1f}% | {100*large_n/total:6.1f}% |{marker}",
            file=out,
        )

    # --- 6. Latency delta: does tier matter? ---
    print("\n## 6. Misroute Signal — Cross-tier Latency Delta\n", file=out)
    all_lats = {tier: [r["elapsed_ms"] for r in by_tier[tier]
                       if r.get("elapsed_ms") is not None and r.get("status_code") == 200]
                for tier in ["small", "medium", "large"]}
    means = {t: mean(v) for t, v in all_lats.items() if v}

    if len(means) >= 2:
        tiers_sorted = sorted(means, key=lambda t: means[t])
        fastest = tiers_sorted[0]
        slowest = tiers_sorted[-1]
        delta_ms = means[slowest] - means[fastest]
        delta_pct = 100 * delta_ms / means[fastest] if means[fastest] else 0
        print(f"Fastest tier: **{fastest}** ({means[fastest]:.0f} ms mean)", file=out)
        print(f"Slowest tier: **{slowest}** ({means[slowest]:.0f} ms mean)", file=out)
        print(f"Cross-tier latency spread: **{delta_ms:.0f} ms** ({delta_pct:.0f}%)\n", file=out)
        if delta_pct < 10:
            verdict = "⚠️  <10% spread — tiers may not matter. Consider removing complexity routing."
        elif delta_pct < 30:
            verdict = "✅  10–30% spread — moderate signal. Threshold tuning likely beneficial."
        else:
            verdict = "✅  >30% spread — strong signal. Complexity routing is earning its keep."
        print(verdict, file=out)
    else:
        print("Not enough tier coverage to compute latency delta.", file=out)

    # --- 7. Recommendation ---
    print("\n## 7. Recommendation\n", file=out)
    ok_records = [r for r in records if r.get("status_code") == 200]
    ok_pct = 100 * len(ok_records) / n if n else 0
    print(f"- **Success rate:** {ok_pct:.1f}% ({len(ok_records)}/{n})", file=out)

    token_coverage = sum(1 for r in ok_records if r.get("output_tokens") is not None)
    print(f"- **Token coverage:** {token_coverage}/{len(ok_records)} responses included usage data", file=out)

    if token_coverage < len(ok_records) * 0.5:
        print("- ⚠️  Low token coverage — backends may not return `usage` field. "
              "Consider forcing `include_usage: true` in requests.", file=out)

    print("\n---\n*Generated by scripts/mt1_analyze.py (AILB-MT-1 Phase C)*", file=out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="/tmp/complexity_routing.jsonl")
    parser.add_argument("--out", default="-", help="Output file path, or - for stdout")
    args = parser.parse_args()

    if not Path(args.log).exists():
        sys.exit(f"Log file not found: {args.log}\nRun scripts/mt1_collect.py first.")

    records = load_records(args.log)
    if not records:
        sys.exit("No records found in log file.")

    print(f"Loaded {len(records)} records from {args.log}", file=sys.stderr)

    if args.out == "-":
        analyze(records, sys.stdout)
    else:
        with open(args.out, "w") as f:
            analyze(records, f)
        print(f"Report written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
