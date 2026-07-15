#!/usr/bin/env python3
"""Analyze the composition of SLO-violating requests under SRPF at each rate.

Key question: are ALL p99 tail requests truly cold first-turns, or are some
evicted continuations (cached_len > 0 but still slow)?

If evicted continuations appear in the tail, a cache-residency mechanism could help.
If 100% are cold first-turns, the design space is truly closed for cache mechanisms.
"""

import re
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict


def parse_time_stats(server_log_path, skip_warmup=0):
    pattern = re.compile(
        r'ReqTimeStats\('
        r'rid=([^,]+),\s*'
        r'input_len=(\d+),\s*'
        r'cached_input_len=(\d+),\s*'
        r'output_len=(\d+),\s*'
        r'type=(\w+)\):\s*'
        r'queue_duration=([0-9.]+)(ms|s|us),\s*'
        r'forward_duration=([0-9.]+)(ms|s|us)'
    )

    def to_ms(val, unit):
        if unit == 's': return val * 1000
        if unit == 'us': return val / 1000
        return val

    records = []
    with open(server_log_path, 'r', errors='ignore') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                input_len = int(m.group(2))
                cached_len = int(m.group(3))
                records.append({
                    'rid': m.group(1),
                    'input_len': input_len,
                    'cached_len': cached_len,
                    'remaining': input_len - cached_len,
                    'output_len': int(m.group(4)),
                    'type': m.group(5),
                    'queue_ms': to_ms(float(m.group(6)), m.group(7)),
                    'fwd_ms': to_ms(float(m.group(8)), m.group(9)),
                    'ttft_ms': to_ms(float(m.group(6)), m.group(7)) + to_ms(float(m.group(8)), m.group(9)),
                    'is_cached': cached_len > 0,
                    'cache_frac': cached_len / max(input_len, 1),
                })

    return records[skip_warmup:]


def analyze_tail(records, rate_label, slo_ms=8000):
    """Analyze the composition of SLO-violating requests."""
    n = len(records)
    if n == 0:
        return

    # Sort by TTFT descending
    by_ttft = sorted(records, key=lambda r: r['ttft_ms'], reverse=True)
    p99_idx = max(0, int(0.01 * n))  # Number of requests in the p99 tail

    # SLO violators
    violations = [r for r in records if r['ttft_ms'] > slo_ms]
    tail_requests = by_ttft[:p99_idx]  # the actual p99 tail

    print(f"\n{'='*90}")
    print(f"Rate λ={rate_label}: {n} requests, {len(violations)} SLO violations (>{slo_ms}ms)")
    print(f"{'='*90}")

    if not violations:
        print(f"  No SLO violations — all requests under {slo_ms}ms")
        return

    # Classify violations
    cold_violations = [r for r in violations if r['cached_len'] == 0]
    partial_cache_violations = [r for r in violations if 0 < r['cached_len'] < r['input_len']]
    full_cache_violations = [r for r in violations if r['cached_len'] == r['input_len']]

    print(f"\n  SLO-violating requests ({len(violations)} total):")
    print(f"    Cold first-turn (cached=0):           {len(cold_violations):>4d} ({100*len(cold_violations)/len(violations):5.1f}%)")
    print(f"    Partial cache (0 < cached < input):   {len(partial_cache_violations):>4d} ({100*len(partial_cache_violations)/len(violations):5.1f}%)")
    print(f"    Full cache (cached == input):          {len(full_cache_violations):>4d} ({100*len(full_cache_violations)/len(violations):5.1f}%)")

    if partial_cache_violations:
        print(f"\n  ★ Partial-cache violations (potential cache-residency opportunity):")
        for r in sorted(partial_cache_violations, key=lambda r: r['ttft_ms'], reverse=True)[:20]:
            print(f"    rid={r['rid'][:30]:30s}  input={r['input_len']:>7d}  cached={r['cached_len']:>7d}  "
                  f"remaining={r['remaining']:>7d}  cache%={100*r['cache_frac']:5.1f}%  "
                  f"TTFT={r['ttft_ms']:>8.0f}ms  queue={r['queue_ms']:>7.0f}ms  fwd={r['fwd_ms']:>8.0f}ms")

    if cold_violations:
        cold_remaining = [r['remaining'] for r in cold_violations]
        cold_ttft = [r['ttft_ms'] for r in cold_violations]
        print(f"\n  Cold violation stats:")
        print(f"    Remaining prefill: median={np.median(cold_remaining):.0f}, "
              f"mean={np.mean(cold_remaining):.0f}, max={np.max(cold_remaining):.0f}")
        print(f"    TTFT: median={np.median(cold_ttft):.0f}ms, mean={np.mean(cold_ttft):.0f}ms, "
              f"max={np.max(cold_ttft):.0f}ms")
        print(f"    Queue time: median={np.median([r['queue_ms'] for r in cold_violations]):.0f}ms, "
              f"mean={np.mean([r['queue_ms'] for r in cold_violations]):.0f}ms")

    # How many violations would be prevented by keeping cached continuations resident?
    if partial_cache_violations:
        # If these had remained cached, their remaining prefill would have been near-zero
        # TTFT would have been ~queue_ms + ~200ms (cached continuation compute)
        would_pass = sum(1 for r in partial_cache_violations
                        if r['queue_ms'] + 200 < slo_ms)
        print(f"\n  ★ If evicted continuations had been cache-hits:")
        print(f"    {would_pass}/{len(partial_cache_violations)} would pass SLO")
        print(f"    Violation reduction: {len(violations)} → {len(violations) - would_pass} "
              f"({100*would_pass/len(violations):.1f}% fewer)")

    # Distribution of ALL requests by type
    all_cold = [r for r in records if r['cached_len'] == 0]
    all_partial = [r for r in records if 0 < r['cached_len'] < r['input_len']]
    all_full = [r for r in records if r['cached_len'] == r['input_len']]
    print(f"\n  All requests breakdown:")
    print(f"    Cold (cached=0):     {len(all_cold):>5d} ({100*len(all_cold)/n:5.1f}%)")
    print(f"    Partial cache:       {len(all_partial):>5d} ({100*len(all_partial)/n:5.1f}%)")
    print(f"    Full cache:          {len(all_full):>5d} ({100*len(all_full)/n:5.1f}%)")

    # What's in the p99 tail (top 1%)?
    print(f"\n  p99 tail (worst {p99_idx} requests):")
    tail_cold = [r for r in tail_requests if r['cached_len'] == 0]
    tail_partial = [r for r in tail_requests if 0 < r['cached_len'] < r['input_len']]
    tail_full = [r for r in tail_requests if r['cached_len'] == r['input_len']]
    print(f"    Cold:     {len(tail_cold):>4d} ({100*len(tail_cold)/max(p99_idx,1):5.1f}%)")
    print(f"    Partial:  {len(tail_partial):>4d} ({100*len(tail_partial)/max(p99_idx,1):5.1f}%)")
    print(f"    Full:     {len(tail_full):>4d} ({100*len(tail_full)/max(p99_idx,1):5.1f}%)")


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_tail_composition.py <srpf_run_dir> [warmup_count]")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 1464

    log_path = run_dir / 'server.log'
    print(f"Parsing: {log_path}")
    all_records = parse_time_stats(str(log_path), skip_warmup=warmup)
    print(f"Total sweep requests: {len(all_records)}")

    rates = [3, 5, 7, 10]
    per_rate = 7037

    for i, rate in enumerate(rates):
        start = i * per_rate
        end = start + per_rate
        rate_records = all_records[start:end] if end <= len(all_records) else all_records[start:]
        analyze_tail(rate_records, str(rate))

    # Aggregate across all rates
    print(f"\n{'='*90}")
    print(f"AGGREGATE SUMMARY")
    print(f"{'='*90}")
    all_violations = [r for r in all_records if r['ttft_ms'] > 8000]
    all_cold_viol = [r for r in all_violations if r['cached_len'] == 0]
    all_partial_viol = [r for r in all_violations if 0 < r['cached_len'] < r['input_len']]
    print(f"Total SLO violations: {len(all_violations)}")
    print(f"  Cold first-turn:   {len(all_cold_viol)} ({100*len(all_cold_viol)/max(len(all_violations),1):.1f}%)")
    print(f"  Evicted cont:      {len(all_partial_viol)} ({100*len(all_partial_viol)/max(len(all_violations),1):.1f}%)")
    if all_partial_viol:
        print(f"\n  ★★ CACHE-ADDRESSABLE FRACTION: {100*len(all_partial_viol)/max(len(all_violations),1):.1f}%")
        print(f"     → A cache-residency mechanism COULD reduce violations by up to ~{len(all_partial_viol)} requests")
    else:
        print(f"\n  ALL violations are cold first-turns — cache management CANNOT address the tail")
        print(f"  Design space for cache mechanisms is TRULY CLOSED at this eval")


if __name__ == '__main__':
    main()
