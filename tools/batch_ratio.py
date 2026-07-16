#!/usr/bin/env python3
"""kleinrock — Direction 4/5 (decode tail): quantify prefill-vs-decode scheduling from a server.log.

Counts Prefill vs Decode batch forward passes and the distribution of consecutive-prefill-run lengths
(the number of forward passes an in-flight generation waits without advancing a token = the decode-starvation
depth). Use to (a) show the stock prefill-first starvation (high prefill:decode ratio, long runs), and (b)
CONFIRM the decode-QoS mechanism worked (--decode-starvation-bound K should cap the run length at ~K and
raise the decode-batch share).

Usage:
  python3 tools/batch_ratio.py runs/v0-stock/server.log [runs/v15-decodeqos-k4/server.log ...]
GPU-free (parses the log). NOTE: a server.log spans the whole sweep incl. warmup + all rates; the run-length
distribution is a scheduling-structure summary, not a per-rate metric.
"""
import sys, re


def analyze(path):
    seq = []
    pat = re.compile(r"\] (Decode|Prefill) batch")
    try:
        for ln in open(path, errors="ignore"):
            m = pat.search(ln)
            if m:
                seq.append(m.group(1))
    except FileNotFoundError:
        print(f"  (no file {path})")
        return
    npf = seq.count("Prefill")
    ndc = seq.count("Decode")
    # consecutive-prefill-run lengths (between two decode batches)
    runs = []
    cur = 0
    for k in seq:
        if k == "Prefill":
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    runs.sort()
    pc = lambda a, q: a[min(len(a) - 1, int(q * len(a)))] if a else None
    ratio = (npf / ndc) if ndc else float("inf")
    print(f"\n=== {path} ===")
    print(f"  prefill batches = {npf}   decode batches = {ndc}   prefill:decode = {ratio:.1f}:1")
    print(
        f"  consecutive-prefill-run length: n={len(runs)} "
        f"p50={pc(runs,.5)} p90={pc(runs,.9)} p99={pc(runs,.99)} max={runs[-1] if runs else None}"
    )
    print(f"  decode-batch share = {100*ndc/max(1,npf+ndc):.1f}%")


if __name__ == "__main__":
    logs = sys.argv[1:] or ["runs/v0-stock/server.log"]
    for lg in logs:
        analyze(lg)
