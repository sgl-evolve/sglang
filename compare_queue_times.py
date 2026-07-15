#!/usr/bin/env python3
"""Compare per-request queue times between SRPF and FCFS runs.

Usage: compare_queue_times.py <srpf_run_dir> <fcfs_run_dir> [warmup_count]

Produces a side-by-side comparison of queue wait times under SRPF vs FCFS,
demonstrating the head-of-line blocking elimination.
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
                    'cached_input_len': int(m.group(3)),
                    'new_tokens': int(m.group(2)) - int(m.group(3)),
                    'queue_ms': to_ms(float(m.group(6)), m.group(7)),
                    'fwd_ms': to_ms(float(m.group(8)), m.group(9)),
                    'ttft_ms': to_ms(float(m.group(6)), m.group(7)) + to_ms(float(m.group(8)), m.group(9)),
                    'is_cached': int(m.group(3)) > 0,
                })

    return records[skip_warmup:]


def analyze(records, label):
    if not records:
        return {}
    cached = [r for r in records if r['is_cached']]
    cold = [r for r in records if not r['is_cached']]

    result = {'label': label, 'n': len(records)}

    for subset, name in [(records, 'all'), (cached, 'cached'), (cold, 'cold')]:
        if not subset:
            continue
        q = np.array([r['queue_ms'] for r in subset])
        f = np.array([r['fwd_ms'] for r in subset])
        t = np.array([r['ttft_ms'] for r in subset])

        result[f'{name}_n'] = len(subset)
        result[f'{name}_queue_median'] = np.median(q)
        result[f'{name}_queue_mean'] = np.mean(q)
        result[f'{name}_queue_p90'] = np.percentile(q, 90)
        result[f'{name}_queue_p99'] = np.percentile(q, 99)
        result[f'{name}_queue_max'] = np.max(q)
        result[f'{name}_fwd_median'] = np.median(f)
        result[f'{name}_ttft_p99'] = np.percentile(t, 99)

        p99_idx = int(0.99 * len(subset))
        sorted_idx = np.argsort(t)
        tail = sorted_idx[p99_idx:]
        tail_q = q[tail]
        tail_f = f[tail]
        tail_t = t[tail]
        if len(tail) > 0:
            result[f'{name}_tail_queue_frac'] = np.mean(tail_q / (tail_t + 1e-6)) * 100
            result[f'{name}_tail_cached_frac'] = np.mean([records[i]['is_cached'] for i in tail]) * 100

    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: compare_queue_times.py <srpf_run_dir> <fcfs_run_dir> [warmup_count]")
        sys.exit(1)

    srpf_dir = Path(sys.argv[1])
    fcfs_dir = Path(sys.argv[2])
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 1464

    srpf_log = srpf_dir / 'server.log'
    fcfs_log = fcfs_dir / 'server.log'

    print(f"Parsing SRPF: {srpf_log}")
    srpf_records = parse_time_stats(str(srpf_log), skip_warmup=warmup)
    print(f"  {len(srpf_records)} sweep requests")

    print(f"Parsing FCFS: {fcfs_log}")
    fcfs_records = parse_time_stats(str(fcfs_log), skip_warmup=warmup)
    print(f"  {len(fcfs_records)} sweep requests")

    srpf = analyze(srpf_records, 'SRPF')
    fcfs = analyze(fcfs_records, 'FCFS')

    print(f"\n{'='*80}")
    print(f"SRPF vs FCFS Queue Time Comparison (same node, same capacity config)")
    print(f"{'='*80}")

    header = f"{'Metric':40s} {'SRPF':>15s} {'FCFS':>15s} {'Ratio':>10s}"
    print(f"\n{header}")
    print("-" * len(header))

    for subset in ['all', 'cached', 'cold']:
        sn = srpf.get(f'{subset}_n', 0)
        fn = fcfs.get(f'{subset}_n', 0)
        if sn == 0 or fn == 0:
            continue

        label = {'all': 'All requests', 'cached': 'Cached continuations', 'cold': 'Cold first-turns'}[subset]
        print(f"\n  {label} (SRPF n={sn}, FCFS n={fn})")

        for metric, fmt in [
            ('queue_median', 'ms'),
            ('queue_mean', 'ms'),
            ('queue_p90', 'ms'),
            ('queue_p99', 'ms'),
            ('queue_max', 'ms'),
            ('fwd_median', 'ms'),
            ('ttft_p99', 'ms'),
        ]:
            sk = f'{subset}_{metric}'
            sv = srpf.get(sk)
            fv = fcfs.get(sk)
            if sv is not None and fv is not None:
                ratio = fv / max(sv, 0.001)
                mname = metric.replace('_', ' ')
                print(f"    {mname:36s} {sv:>12.1f}{fmt} {fv:>12.1f}{fmt} {ratio:>8.1f}×")

    # Tail composition comparison
    print(f"\n  p99 tail composition:")
    for subset in ['all', 'cached', 'cold']:
        sq = srpf.get(f'{subset}_tail_queue_frac')
        fq = fcfs.get(f'{subset}_tail_queue_frac')
        if sq is not None and fq is not None:
            label = {'all': 'All', 'cached': 'Cached', 'cold': 'Cold'}[subset]
            print(f"    {label:12s}: SRPF queue={sq:.1f}% | FCFS queue={fq:.1f}%")

    # Per-rate breakdown
    per_rate = 7037
    rates = [3, 5, 7, 10]
    print(f"\n{'='*80}")
    print(f"Per-Rate Queue Time Comparison")
    print(f"{'='*80}")

    for i, rate in enumerate(rates):
        start = i * per_rate
        end = start + per_rate
        srpf_rate = srpf_records[start:end] if end <= len(srpf_records) else srpf_records[start:]
        fcfs_rate = fcfs_records[start:end] if end <= len(fcfs_records) else fcfs_records[start:]

        if len(srpf_rate) < 100 or len(fcfs_rate) < 100:
            continue

        print(f"\n  Rate λ={rate} (SRPF n={len(srpf_rate)}, FCFS n={len(fcfs_rate)})")

        for typ, filter_fn in [('Cached', lambda r: r['is_cached']), ('Cold', lambda r: not r['is_cached'])]:
            s_sub = [r for r in srpf_rate if filter_fn(r)]
            f_sub = [r for r in fcfs_rate if filter_fn(r)]
            if not s_sub or not f_sub:
                continue
            sq = np.array([r['queue_ms'] for r in s_sub])
            fq = np.array([r['queue_ms'] for r in f_sub])
            ratio_med = np.median(fq) / max(np.median(sq), 0.001)
            s_gt1s = 100 * sum(1 for x in sq if x > 1000) / len(sq)
            f_gt1s = 100 * sum(1 for x in fq if x > 1000) / len(fq)
            print(f"    {typ:8s}: SRPF med={np.median(sq):7.1f}ms  FCFS med={np.median(fq):7.1f}ms  ratio={ratio_med:6.0f}×  |  SRPF >1s={s_gt1s:4.1f}%  FCFS >1s={f_gt1s:4.1f}%")


if __name__ == '__main__':
    main()
