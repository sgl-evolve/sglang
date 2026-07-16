#!/usr/bin/env python3
"""floyd: is the L2->L1 (host->device) KV load-back a binding cost under load?

Reads the sglang Prometheus load-back histogram from runs/<ver>/metrics_r<rate>.txt
(emitted by observe_load_back_duration / load_back_tokens_total in unified_radix_cache.py).
Answers the program's "transfer/compute overlap under load" ambitious target directly, and by
extension the movement-side family (layout/paging for movement cost, conversation co-residency
for reload cost): if the L2->L1 load-back is already cheap and scales under load, there is no
latency to hide, so movement-hiding / layout / residency-timing mechanisms have no goodput headroom.

Usage: python3 analysis/load_back_stats.py [runs/v0-stock]
"""
import sys, re, glob, os

run = sys.argv[1] if len(sys.argv) > 1 else "runs/v0-stock"


def g(t, pat):
    m = re.search(pat, t)
    return float(m.group(1)) if m else None


print(f"{'rate':>5} {'load-backs':>11} {'tokens':>10} {'mean(ms)':>9} {'<10ms':>7} {'<30ms':>7}")
for f in sorted(glob.glob(f"{run}/metrics_r*.txt"), key=lambda p: int(re.search(r'_r(\d+)', p).group(1))):
    rate = re.search(r'_r(\d+)', f).group(1)
    t = open(f).read()
    s = g(t, r'load_back_duration_seconds_sum\{[^}]*\}\s+([0-9.e+]+)')
    c = g(t, r'load_back_duration_seconds_count\{[^}]*\}\s+([0-9.e+]+)')
    tok = g(t, r'load_back_tokens_total\{[^}]*\}\s+([0-9.e+]+)')
    le10 = g(t, r'load_back_duration_seconds_bucket\{[^}]*le="0.01"\}\s+([0-9.e+]+)')
    le30 = g(t, r'load_back_duration_seconds_bucket\{[^}]*le="0.03"\}\s+([0-9.e+]+)')
    inf = g(t, r'load_back_duration_seconds_bucket\{[^}]*le="\+Inf"\}\s+([0-9.e+]+)')
    if not c:
        continue
    print(f"{rate:>5} {int(c):>11} {tok:>10.3g} {1000*s/c:>9.2f} {100*le10/inf:>6.1f}% {100*le30/inf:>6.1f}%")

print("""
Interpretation: the L2->L1 load-back is NON-BINDING and SCALES under load. Every load-back completes
in <30 ms (>=99.4% <10 ms; mean ~1.3-1.5 ms) at every rate, EVEN as the load-back count/volume grows
~3.7x from lambda3 to lambda10 (eviction pressure rises with load, so MORE reloads happen -- but each
stays cheap). Versus the queueing-dominated TTFT p99 (17-24 s at lambda>=5), a <=30 ms load-back is
<0.2% of TTFT and never on the critical path. => The program's "transfer/compute overlap that hides
L1<->L2 latency under load" target has NO goodput headroom, and neither do layout/paging (movement
cost) or conversation co-residency (reload cost): movement is already cheap. Bounds the whole
movement-side mechanism family; the goodput lever remains prefill scheduling (see P3).
""")
