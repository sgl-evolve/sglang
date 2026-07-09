#!/usr/bin/env python3
"""Per-lambda scheduler-pressure analysis from existing rate-sweep server logs.

Segments a sweep's server.log into its 4 per-lambda bench windows (delimited by the
`Cache flushed successfully!` markers that ratesweep.sh emits before each bench), then
aggregates the server's INTERNAL scheduler counters per window: mean queue depth,
mean pending-token backlog, and total prefilled new-tokens. These are deterministic
cache/scheduler counts (NOT wall-clock latency), so they are IMMUNE to the shared-NFS
parallel-eval contamination that makes p99 unreliable -> a clean way to test whether
cost-aware eviction relieves queue pressure per-lambda (i.e. shifts the goodput curve),
using data already on disk (no new eval).
"""
import re, sys, glob
from datetime import datetime

TS = re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
FLUSH = re.compile(r'Cache flushed successfully')
QUE = re.compile(r'#queue-req: (\d+)')
PEND = re.compile(r'#pending-token: (\d+)')
NEWT = re.compile(r'#new-token: (\d+)')
PREFILL = re.compile(r'Prefill batch')

def parse_ts(line):
    m = TS.search(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')

def analyze(path):
    lines = open(path, errors='ignore').readlines()
    # flush timestamps
    flushes = [parse_ts(l) for l in lines if FLUSH.search(l)]
    flushes = [f for f in flushes if f]
    # bench windows = long gaps (> 20 min) between consecutive flush events
    # (each phase is [flush][flush ~1min later][bench ~30-40min])
    marks = sorted(set(flushes))
    windows = []  # (start, end)
    for i in range(len(marks) - 1):
        gap = (marks[i+1] - marks[i]).total_seconds()
        if gap > 1200:  # >20min => a bench ran in this gap
            windows.append((marks[i], marks[i+1]))
    # last window: from last flush to end of log
    if marks:
        end = parse_ts(lines[-1]) or marks[-1]
        if (end - marks[-1]).total_seconds() > 1200:
            windows.append((marks[-1], end))
    return lines, windows

def window_stats(lines, start, end):
    q_sum=q_n=0; p_sum=p_n=0; newt=0; nbatch=0
    for l in lines:
        if 'Prefill batch' not in l and 'Decode batch' not in l:
            continue
        t = parse_ts(l)
        if t is None or t < start or t > end:
            continue
        mq = QUE.search(l)
        if mq: q_sum += int(mq.group(1)); q_n += 1
        if PREFILL.search(l):
            nbatch += 1
            mp = PEND.search(l);  p_sum += int(mp.group(1)) if mp else 0; p_n += 1
            mn = NEWT.search(l);  newt += int(mn.group(1)) if mn else 0
    dur = (end-start).total_seconds()
    return dict(dur_s=round(dur), mean_queue=round(q_sum/q_n,1) if q_n else 0,
                mean_pending=round(p_sum/p_n) if p_n else 0,
                total_new_tok=newt, prefill_batches=nbatch)

# map window duration -> lambda (known per-λ durations, seconds)
DUR2LAM = [(2327,'3'),(2005,'4'),(1810,'5'),(1797,'6')]
def guess_lambda(dur):
    return min(DUR2LAM, key=lambda x: abs(x[0]-dur))[1]

for tag in sys.argv[1:]:
    path = f'runs/{tag}/server.log'
    if not glob.glob(path):
        print(f'{tag}: no server.log'); continue
    lines, windows = analyze(path)
    print(f'\n===== {tag}: {len(windows)} bench windows (temporal order = λ order 3,4,5,6) =====')
    print(f"{'λ*':>4} {'dur_s':>6} {'mean_queue':>11} {'mean_pending':>13} {'total_new_tok':>14} {'prefill_batches':>16}")
    # windows are emitted in the fixed sweep order λ=3,4,5,6; label by position when 4 present.
    for i,(s,e) in enumerate(windows):
        st = window_stats(lines, s, e)
        lam = str(3+i) if len(windows)==4 else f'~{guess_lambda(st["dur_s"])}'
        print(f"{lam:>4} {st['dur_s']:>6} {st['mean_queue']:>11} {st['mean_pending']:>13} {st['total_new_tok']:>14,} {st['prefill_batches']:>16,}")
