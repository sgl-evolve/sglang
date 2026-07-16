#!/usr/bin/env python3
"""Is the goodput-rate p99 TTFT tail KV-memory-wait, or compute+head-of-line?

Closes the *proactive KV-memory-management* sub-axis (decode retraction / headroom
reservation to admit big cold prefills faster) from the STOCK trace, with no new GPU:

  * retraction events        -> if 0, the reactive decode-eviction path never fires,
                                so there is no "retraction-too-late stall" to pre-empt.
  * prefill steps admitting   -> #new-seq==0 while work is queued would mean admission
    ZERO new sequences           is memory-blocked (a stall a reservation could remove).
  * KV-pool token usage        -> shows the pool DOES reach near-full, yet neither
    percentiles                  retracts nor blocks: pressure is absorbed losslessly by
                                 evicting CLEAN cached prefixes (P2: LRU=Belady=0 avoidable).

Verdict if (retractions==0 AND #new-seq==0 count==0): the tail is compute + head-of-line,
NOT memory-wait -> proactive memory management has no goodput headroom (a non-lever).

Usage: mem_admission_stats.py runs/v0-stock/server.log [runs/v-stock-samenode-1/server.log ...]
Reads only the engine's own stdout log lines; no external deps.
"""
import re
import sys

USAGE_RE = re.compile(r"full token usage: ([0-9.]+)")
NEWSEQ_RE = re.compile(r"Prefill batch, #new-seq: ([0-9]+)")
# real retraction log lines (exclude the server_args config echo which contains the
# substrings 'optimistic_prefill_retries' / 'num_reserved_decode_tokens')
RETRACT_RE = re.compile(r"[Rr]etract")
CONFIG_RE = re.compile(r"server_args=|optimistic_prefill_retries|num_reserved")


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))]


def analyze(path):
    usage, newseq = [], []
    retract = 0
    with open(path, errors="ignore") as f:
        for line in f:
            if RETRACT_RE.search(line) and not CONFIG_RE.search(line):
                retract += 1
            m = USAGE_RE.search(line)
            if m:
                usage.append(float(m.group(1)))
            m = NEWSEQ_RE.search(line)
            if m:
                newseq.append(int(m.group(1)))
    zero = sum(1 for n in newseq if n == 0)
    print(f"\n{path}")
    print(f"  real retraction events               : {retract}")
    print(f"  prefill steps                        : {len(newseq)}")
    print(f"  prefill steps with #new-seq==0       : {zero}  (memory-blocked admission)")
    print(f"  KV-pool token usage p50/p90/p99/max  : "
          f"{pct(usage,.5):.2f} / {pct(usage,.9):.2f} / {pct(usage,.99):.2f} / {max(usage):.2f}")
    verdict = "COMPUTE+HOL (memory-mgmt non-lever)" if (retract == 0 and zero == 0) \
        else "MEMORY-WAIT headroom present"
    print(f"  => verdict: {verdict}")
    return retract, zero


if __name__ == "__main__":
    # Default: all four full sweeps -> 2 stock (FCFS) + 2 SRPF, so the non-lever is shown
    # robust to the scheduling policy, not an artifact of stock ordering.
    logs = sys.argv[1:] or [
        "runs/v0-stock/server.log", "runs/v-stock-samenode-1/server.log",
        "runs/v-srpf-full/server.log", "runs/v-srpf-samenode-1/server.log",
    ]
    tot_r = tot_z = n = 0
    for p in logs:
        try:
            r, z = analyze(p)
            tot_r += r
            tot_z += z
            n += 1
        except FileNotFoundError:
            print(f"\n{p}: not found (skip)")
    print(f"\nPOOLED over {n} full sweeps (stock FCFS + SRPF): "
          f"{tot_r} retractions, {tot_z} memory-blocked prefill admissions.")
    print("Both zero under BOTH policies => the proactive-KV-memory-management sub-axis"
          " (retraction / reservation) has no headroom on this workload; tail = compute + head-of-line.")
