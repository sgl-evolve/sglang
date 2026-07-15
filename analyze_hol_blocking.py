#!/usr/bin/env python3
"""Analyze head-of-line blocking: per-rate SRPF vs FCFS queue time comparison.

Usage: analyze_hol_blocking.py <srpf_run_dir> <fcfs_run_dir> [warmup_count]

Produces per-rate, per-type queue time comparison tables suitable for the paper.
"""

import re
import sys
import numpy as np
from pathlib import Path


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
                records.append({
                    'input_len': int(m.group(2)),
                    'cached_len': int(m.group(3)),
                    'remaining': int(m.group(2)) - int(m.group(3)),
                    'queue_ms': to_ms(float(m.group(6)), m.group(7)),
                    'fwd_ms': to_ms(float(m.group(8)), m.group(9)),
                    'ttft_ms': to_ms(float(m.group(6)), m.group(7)) + to_ms(float(m.group(8)), m.group(9)),
                    'is_cached': int(m.group(3)) > 0,
                })

    return records[skip_warmup:]


def stats(values):
    a = np.array(values)
    return {
        'n': len(a),
        'median': np.median(a),
        'mean': np.mean(a),
        'p90': np.percentile(a, 90),
        'p99': np.percentile(a, 99),
        'max': np.max(a),
        'gt1s_pct': 100 * sum(1 for x in a if x > 1000) / len(a),
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: analyze_hol_blocking.py <srpf_run_dir> <fcfs_run_dir> [warmup_count]")
        sys.exit(1)

    srpf_dir = Path(sys.argv[1])
    fcfs_dir = Path(sys.argv[2])
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 1464

    print(f"Parsing SRPF: {srpf_dir / 'server.log'}")
    srpf_all = parse_time_stats(str(srpf_dir / 'server.log'), skip_warmup=warmup)
    print(f"  {len(srpf_all)} sweep requests")

    print(f"Parsing FCFS: {fcfs_dir / 'server.log'}")
    fcfs_all = parse_time_stats(str(fcfs_dir / 'server.log'), skip_warmup=warmup)
    print(f"  {len(fcfs_all)} sweep requests")

    rates = [3, 5, 7, 10]
    per_rate = 7037

    # === Table 1: Per-rate cached queue comparison ===
    print(f"\n{'='*100}")
    print(f"TABLE 1: Cached Continuation Queue Time — SRPF vs FCFS")
    print(f"{'='*100}")
    print(f"{'Rate':>5s}  {'SRPF n':>7s}  {'SRPF med':>9s}  {'SRPF >1s':>9s}  {'FCFS n':>7s}  {'FCFS med':>9s}  {'FCFS >1s':>9s}  {'Ratio':>7s}")
    print("-" * 100)

    for i, rate in enumerate(rates):
        start = i * per_rate
        end = start + per_rate

        srpf_rate = srpf_all[start:end] if end <= len(srpf_all) else srpf_all[start:]
        fcfs_rate = fcfs_all[start:end] if end <= len(fcfs_all) else fcfs_all[start:]

        srpf_cached = [r for r in srpf_rate if r['is_cached']]
        fcfs_cached = [r for r in fcfs_rate if r['is_cached']]

        if not srpf_cached or not fcfs_cached:
            print(f"  r{rate:>2d}  (insufficient data)")
            continue

        sq = stats([r['queue_ms'] for r in srpf_cached])
        fq = stats([r['queue_ms'] for r in fcfs_cached])
        ratio = fq['median'] / max(sq['median'], 0.001)

        print(f"  r{rate:>2d}  {sq['n']:>7d}  {sq['median']:>7.1f}ms  {sq['gt1s_pct']:>7.1f}%  {fq['n']:>7d}  {fq['median']:>7.1f}ms  {fq['gt1s_pct']:>7.1f}%  {ratio:>5.0f}×")

    # === Table 2: Cold document queue comparison ===
    print(f"\n{'='*100}")
    print(f"TABLE 2: Cold Document Queue Time — SRPF vs FCFS")
    print(f"{'='*100}")
    print(f"{'Rate':>5s}  {'SRPF n':>7s}  {'SRPF med':>9s}  {'SRPF p99':>10s}  {'FCFS n':>7s}  {'FCFS med':>9s}  {'FCFS p99':>10s}  {'Med ratio':>10s}")
    print("-" * 100)

    for i, rate in enumerate(rates):
        start = i * per_rate
        end = start + per_rate

        srpf_rate = srpf_all[start:end] if end <= len(srpf_all) else srpf_all[start:]
        fcfs_rate = fcfs_all[start:end] if end <= len(fcfs_all) else fcfs_all[start:]

        srpf_cold = [r for r in srpf_rate if not r['is_cached']]
        fcfs_cold = [r for r in fcfs_rate if not r['is_cached']]

        if not srpf_cold or not fcfs_cold:
            print(f"  r{rate:>2d}  (insufficient data)")
            continue

        sq = stats([r['queue_ms'] for r in srpf_cold])
        fq = stats([r['queue_ms'] for r in fcfs_cold])
        ratio = fq['median'] / max(sq['median'], 0.001)

        print(f"  r{rate:>2d}  {sq['n']:>7d}  {sq['median']:>7.1f}ms  {sq['p99']:>8.1f}ms  {fq['n']:>7d}  {fq['median']:>7.1f}ms  {fq['p99']:>8.1f}ms  {ratio:>8.1f}×")

    # === Table 3: Forward compute comparison (should be similar) ===
    print(f"\n{'='*100}")
    print(f"TABLE 3: Cold Forward Compute — SRPF vs FCFS (control: should be similar)")
    print(f"{'='*100}")
    print(f"{'Rate':>5s}  {'SRPF med fwd':>13s}  {'FCFS med fwd':>13s}  {'Ratio':>7s}")
    print("-" * 60)

    for i, rate in enumerate(rates):
        start = i * per_rate
        end = start + per_rate

        srpf_rate = srpf_all[start:end] if end <= len(srpf_all) else srpf_all[start:]
        fcfs_rate = fcfs_all[start:end] if end <= len(fcfs_all) else fcfs_all[start:]

        srpf_cold = [r for r in srpf_rate if not r['is_cached']]
        fcfs_cold = [r for r in fcfs_rate if not r['is_cached']]

        if not srpf_cold or not fcfs_cold:
            continue

        sf = np.median([r['fwd_ms'] for r in srpf_cold])
        ff = np.median([r['fwd_ms'] for r in fcfs_cold])
        ratio = ff / max(sf, 0.001)

        print(f"  r{rate:>2d}  {sf:>11.0f}ms  {ff:>11.0f}ms  {ratio:>5.2f}×")

    # === Queue distribution comparison (for paper text) ===
    print(f"\n{'='*100}")
    print(f"QUEUE DISTRIBUTION: % of cached requests in each queue bucket")
    print(f"{'='*100}")
    buckets = [(0, 10, '<10ms'), (10, 100, '10-100ms'), (100, 1000, '100ms-1s'),
               (1000, 5000, '1-5s'), (5000, float('inf'), '>5s')]

    for i, rate in enumerate(rates):
        start = i * per_rate
        end = start + per_rate
        srpf_rate = srpf_all[start:end] if end <= len(srpf_all) else srpf_all[start:]
        fcfs_rate = fcfs_all[start:end] if end <= len(fcfs_all) else fcfs_all[start:]
        srpf_cached_q = [r['queue_ms'] for r in srpf_rate if r['is_cached']]
        fcfs_cached_q = [r['queue_ms'] for r in fcfs_rate if r['is_cached']]

        if not srpf_cached_q or not fcfs_cached_q:
            continue

        print(f"\n  Rate λ={rate}:")
        for lo, hi, label in buckets:
            sp = 100 * sum(1 for q in srpf_cached_q if lo <= q < hi) / len(srpf_cached_q)
            fp = 100 * sum(1 for q in fcfs_cached_q if lo <= q < hi) / len(fcfs_cached_q)
            print(f"    {label:>10s}: SRPF {sp:5.1f}%  FCFS {fp:5.1f}%")

    # === Summary for paper ===
    print(f"\n{'='*100}")
    print(f"SUMMARY FOR PAPER")
    print(f"{'='*100}")

    srpf_cached_all = [r for r in srpf_all if r['is_cached']]
    fcfs_cached_all = [r for r in fcfs_all if r['is_cached']]
    if srpf_cached_all and fcfs_cached_all:
        sq_all = np.array([r['queue_ms'] for r in srpf_cached_all])
        fq_all = np.array([r['queue_ms'] for r in fcfs_cached_all])
        print(f"Aggregate cached queue median: SRPF {np.median(sq_all):.1f}ms, FCFS {np.median(fq_all):.1f}ms, ratio {np.median(fq_all)/max(np.median(sq_all),0.001):.0f}×")
        print(f"Aggregate cached >1s: SRPF {100*sum(1 for x in sq_all if x>1000)/len(sq_all):.1f}%, FCFS {100*sum(1 for x in fq_all if x>1000)/len(fq_all):.1f}%")


if __name__ == '__main__':
    main()
