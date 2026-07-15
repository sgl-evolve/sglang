#!/usr/bin/env python3
"""Analyze per-request TTFT decomposition from server logs with --enable-request-time-stats-logging.

Parses ReqTimeStats lines to extract queue_duration and forward_duration for each request,
then produces:
1. Per-rate (r3/r5/r7/r10) breakdown: median/p90/p99/max of queue vs forward
2. Cached vs cold request comparison (using cached_input_len > 0 as proxy)
3. Identify which component dominates the p99 tail
4. Show that under SRPF, p99 is forward_compute-dominated (not queue-wait-dominated)
"""

import re
import sys
import json
import numpy as np
from collections import defaultdict
from pathlib import Path


def parse_time_stats(server_log_path):
    """Parse ReqTimeStats lines from server.log.

    Format: ReqTimeStats(rid=..., input_len=..., cached_input_len=..., output_len=..., type=unified):
            queue_duration=..., forward_duration=..., entry_time=...
    """
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

    records = []
    with open(server_log_path, 'r', errors='ignore') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                rid = m.group(1)
                input_len = int(m.group(2))
                cached_input_len = int(m.group(3))
                output_len = int(m.group(4))
                req_type = m.group(5)

                queue_val = float(m.group(6))
                queue_unit = m.group(7)
                fwd_val = float(m.group(8))
                fwd_unit = m.group(9)

                # Normalize to ms
                def to_ms(val, unit):
                    if unit == 's':
                        return val * 1000
                    elif unit == 'us':
                        return val / 1000
                    return val  # already ms

                queue_ms = to_ms(queue_val, queue_unit)
                fwd_ms = to_ms(fwd_val, fwd_unit)

                new_tokens = input_len - cached_input_len
                is_cached = cached_input_len > 0

                records.append({
                    'rid': rid,
                    'input_len': input_len,
                    'cached_input_len': cached_input_len,
                    'new_tokens': new_tokens,
                    'output_len': output_len,
                    'is_cached': is_cached,
                    'queue_ms': queue_ms,
                    'forward_ms': fwd_ms,
                    'ttft_ms': queue_ms + fwd_ms,
                })

    return records


def find_rate_boundaries(bench_dir):
    """Determine approximate request count boundaries per rate from bench files."""
    boundaries = {}
    for r in [3, 5, 7, 10]:
        bench_file = bench_dir / f'bench_r{r}.json'
        if bench_file.exists():
            try:
                data = json.load(open(bench_file))
                if isinstance(data, list):
                    boundaries[r] = len(data)
                elif isinstance(data, dict) and 'inputs' in data:
                    boundaries[r] = len(data['inputs'])
            except:
                pass
    return boundaries


def analyze_records(records, label=""):
    """Analyze a set of records and print statistics."""
    if not records:
        print(f"  {label}: no records")
        return

    queue = np.array([r['queue_ms'] for r in records])
    fwd = np.array([r['forward_ms'] for r in records])
    ttft = np.array([r['ttft_ms'] for r in records])
    new_tok = np.array([r['new_tokens'] for r in records])

    n = len(records)

    def pctile_str(arr, name):
        return (f"    {name}: median={np.median(arr):.1f}ms, "
                f"p90={np.percentile(arr, 90):.1f}ms, "
                f"p99={np.percentile(arr, 99):.1f}ms, "
                f"max={np.max(arr):.1f}ms, "
                f"mean={np.mean(arr):.1f}ms")

    print(f"\n  {label} (n={n}):")
    print(pctile_str(queue, "queue_wait "))
    print(pctile_str(fwd, "fwd_compute"))
    print(pctile_str(ttft, "total_ttft "))
    print(pctile_str(new_tok, "new_tokens "))

    # What fraction of p99 TTFT is queue vs forward?
    p99_idx = int(0.99 * n)
    sorted_ttft_idx = np.argsort(ttft)
    tail_indices = sorted_ttft_idx[p99_idx:]

    tail_queue = queue[tail_indices]
    tail_fwd = fwd[tail_indices]
    tail_ttft = ttft[tail_indices]
    tail_new = new_tok[tail_indices]
    tail_cached = np.array([records[i]['is_cached'] for i in tail_indices])

    print(f"    --- p99 tail requests (n={len(tail_indices)}) ---")
    if len(tail_indices) > 0:
        avg_queue_frac = np.mean(tail_queue / (tail_ttft + 1e-6)) * 100
        avg_fwd_frac = np.mean(tail_fwd / (tail_ttft + 1e-6)) * 100
        print(f"    tail composition: queue={avg_queue_frac:.1f}%, forward={avg_fwd_frac:.1f}%")
        print(f"    tail cached fraction: {np.mean(tail_cached) * 100:.1f}% (vs overall {np.mean([r['is_cached'] for r in records]) * 100:.1f}%)")
        print(f"    tail new_tokens: median={np.median(tail_new):.0f}, mean={np.mean(tail_new):.0f}")
        print(f"    tail queue_ms: median={np.median(tail_queue):.1f}, mean={np.mean(tail_queue):.1f}")
        print(f"    tail forward_ms: median={np.median(tail_fwd):.1f}, mean={np.mean(tail_fwd):.1f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_ttft_decomposition.py <run_dir>")
        print("  run_dir should contain server.log with ReqTimeStats lines")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    server_log = run_dir / 'server.log'

    if not server_log.exists():
        print(f"ERROR: {server_log} not found")
        sys.exit(1)

    print(f"Parsing {server_log}...")
    records = parse_time_stats(str(server_log))
    print(f"Found {len(records)} ReqTimeStats entries")

    if not records:
        print("No ReqTimeStats entries found. Is --enable-request-time-stats-logging set?")
        sys.exit(1)

    # Overall analysis
    print("\n=== OVERALL ===")
    analyze_records(records, "all requests")

    # Split by cached vs cold
    cached = [r for r in records if r['is_cached']]
    cold = [r for r in records if not r['is_cached']]

    print("\n=== BY CACHE STATUS ===")
    analyze_records(cached, "cached (continuation)")
    analyze_records(cold, "cold (first-turn)")

    # Split cold by size
    if cold:
        cold_small = [r for r in cold if r['new_tokens'] < 10000]
        cold_medium = [r for r in cold if 10000 <= r['new_tokens'] < 50000]
        cold_large = [r for r in cold if r['new_tokens'] >= 50000]

        print("\n=== COLD REQUESTS BY SIZE ===")
        analyze_records(cold_small, "cold <10K tokens")
        analyze_records(cold_medium, "cold 10K-50K tokens")
        analyze_records(cold_large, "cold >=50K tokens (mega-docs)")

    # Try to split by rate phase (approximate: based on chronological order)
    # With 1553 requests per rate + 300 warmup, we can estimate boundaries
    # Warmup: first ~300 convs worth of requests (variable turns)
    # But since we don't have precise boundaries, use the bench output files

    # Summary insight
    print("\n=== KEY INSIGHT ===")
    if records:
        all_ttft = np.array([r['ttft_ms'] for r in records])
        all_queue = np.array([r['queue_ms'] for r in records])
        all_fwd = np.array([r['forward_ms'] for r in records])

        p99_ttft = np.percentile(all_ttft, 99)

        # For requests near the p99 TTFT
        near_p99 = [r for r in records if r['ttft_ms'] >= p99_ttft * 0.9]
        if near_p99:
            queue_in_tail = np.mean([r['queue_ms'] / (r['ttft_ms'] + 1e-6) for r in near_p99]) * 100
            fwd_in_tail = np.mean([r['forward_ms'] / (r['ttft_ms'] + 1e-6) for r in near_p99]) * 100
            cached_in_tail = np.mean([r['is_cached'] for r in near_p99]) * 100
            cold_in_tail = 100 - cached_in_tail

            print(f"  p99 TTFT = {p99_ttft:.1f}ms")
            print(f"  Requests near p99 (>={p99_ttft*0.9:.0f}ms): n={len(near_p99)}")
            print(f"  Tail TTFT composition: {queue_in_tail:.1f}% queue + {fwd_in_tail:.1f}% forward")
            print(f"  Tail request types: {cold_in_tail:.1f}% cold, {cached_in_tail:.1f}% cached")

            if fwd_in_tail > 70:
                print(f"  CONCLUSION: p99 is FORWARD-COMPUTE DOMINATED ({fwd_in_tail:.0f}%)")
                print(f"  → Confirms: under SRPF, the tail is cold-doc own-prefill, not queue wait")
            elif queue_in_tail > 70:
                print(f"  CONCLUSION: p99 is QUEUE-WAIT DOMINATED ({queue_in_tail:.0f}%)")
            else:
                print(f"  CONCLUSION: p99 is MIXED (queue {queue_in_tail:.0f}% + fwd {fwd_in_tail:.0f}%)")


if __name__ == '__main__':
    main()
