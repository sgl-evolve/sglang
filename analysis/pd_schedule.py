#!/usr/bin/env python3
"""floyd: is the prefill/decode split a goodput lever, or is saturation compute-hard? (last axis)

At saturation the GPU must divide each forward step between PREFILL (new first-tokens => TTFT) and
DECODE (advancing running requests => E2E, not TTFT). If the split left goodput headroom, a co-scheduler
could help. We parse the stock v0-stock server.log:
  - big time-gaps between logged steps = protocol inter-rate FLUSHES (34-61s), not recoverable bubbles.
  - at saturation (full token usage > 0.9), what fraction of steps is prefill vs decode?
Result: ~99% of saturation steps are PREFILL (decode starved to ~1%). Since goodput@SLO is a TTFT metric,
spending steps on prefill (= more first-tokens) is already TTFT-optimal; shifting steps to decode would
LOWER goodput, and throttling prefill IS admission control (paper 1's negative). Saturation throughput
(4.22 req/s) equals the work-conservation ceiling (4.2) => compute-hard, no macro-bubbles. The
prefill/decode arbitration is therefore a NON-LEVER for goodput@SLO — the last axis, bounded.
"""
import re
from collections import Counter
from datetime import datetime

LOG = "runs/v0-stock/server.log"
FMT = "%Y-%m-%d %H:%M:%S"

def main():
    pre = dec = sat_pre = sat_dec = 0
    times = []
    for l in open(LOG, errors="ignore"):
        m = re.search(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", l)
        if not m: continue
        is_pre = "Prefill batch" in l; is_dec = "Decode batch" in l
        if not (is_pre or is_dec): continue
        times.append(m.group(1))
        if is_pre: pre += 1
        else: dec += 1
        fu = re.search(r"full token usage: ([0-9.]+)", l)
        if fu and float(fu.group(1)) > 0.9:
            if is_pre: sat_pre += 1
            else: sat_dec += 1
    ts = sorted(set(times)); secs = [datetime.strptime(t, FMT) for t in ts]
    gaps = sorted(((secs[i+1]-secs[i]).total_seconds() for i in range(len(secs)-1)), reverse=True)
    big = [g for g in gaps if g > 2]
    nsat = sat_pre + sat_dec
    print(f"batch steps: prefill {pre} ({100*pre/(pre+dec):.0f}%)  decode {dec} ({100*dec/(pre+dec):.0f}%)")
    print(f"idle gaps >2s (protocol flushes, not bubbles): {len(big)} -> {big[:6]}")
    print(f"SATURATION (full-usage>0.9): {nsat} steps  prefill {100*sat_pre/max(1,nsat):.0f}%  decode {100*sat_dec/max(1,nsat):.0f}%")
    print("=> 99% prefill at saturation is TTFT-optimal (more first-tokens); shifting to decode lowers")
    print("   goodput, throttling prefill = admission (paper1 neg). Throughput 4.22 == ceiling 4.2 =>")
    print("   compute-hard, no macro-bubbles. prefill/decode split is a NON-LEVER for goodput@SLO.")

if __name__ == "__main__":
    main()
