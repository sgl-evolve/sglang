#!/usr/bin/env python3
"""floyd: extract KV-pool-usage distribution from a run's server.log.

The stock scheduler's device-pool usage is bimodal (idle <-> saturated) -- the
metastable signature CCA targets. This prints a histogram + key fractions so we
can show CCA collapses the saturation peak into the mid-range.

Usage: python analysis/pool_hist.py runs/<version>/server.log
"""
import collections, re, sys

def main(path):
    vals = []
    for line in open(path, errors="ignore"):
        m = re.search(r"full token usage: ([0-9.]+)", line)
        if m:
            vals.append(float(m.group(1)))
    if not vals:
        print(f"no pool-usage samples in {path}")
        return
    tot = len(vals)
    h = collections.Counter(min(9, int(v * 10)) for v in vals)
    mx = max(h.values())
    print(f"{path}: n={tot}")
    for b in range(10):
        c = h.get(b, 0)
        bar = "#" * int(60 * c / mx)
        print(f"  [{b/10:.1f},{(b+1)/10:.1f}) {c:6d} ({100*c/tot:4.1f}%) {bar}")
    sat = sum(1 for v in vals if v >= 0.90)
    lull = sum(1 for v in vals if v <= 0.05)
    mid = sum(1 for v in vals if 0.3 <= v <= 0.7)
    p100 = sum(1 for v in vals if v >= 0.999)
    print(f"  saturated>=0.90: {100*sat/tot:.1f}%  | idle<=0.05: {100*lull/tot:.1f}%  "
          f"| mid 0.3-0.7: {100*mid/tot:.1f}%  | at 1.00: {p100} events")

if __name__ == "__main__":
    for p in (sys.argv[1:] or ["runs/v0-stock/server.log"]):
        main(p)
