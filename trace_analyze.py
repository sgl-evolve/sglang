#!/usr/bin/env python3
"""Analyze a per-prefill trace (SGLANG_WILKES_TRACE .rank0), dedup by rid (chunked prefills log multiple
entries per request; the FIRST entry per rid carries the true match at admission). Reports the per-request
uncached-prefill (recompute) structure = the goodput@SLO miss anatomy.

Usage: python3 trace_analyze.py runs/<ver>/trace.rank0"""
import sys, json, numpy as np

def main(path):
    first = {}
    order = []
    for l in open(path, errors="ignore"):
        l = l.strip()
        if not l: continue
        try: r = json.loads(l)
        except Exception: continue
        rid = r.get("rid")
        if rid not in first:
            first[rid] = r; order.append(rid)
    rows = [first[r] for r in order]
    plen = np.array([r["plen"] for r in rows], float)
    unc = np.array([r["uncached"] for r in rows], float)
    dev = np.array([r["dev"] for r in rows], float)
    host = np.array([r["host"] for r in rows], float)
    wq = np.array([r.get("wq", 0) for r in rows], float)
    n = len(rows)
    print(f"unique requests (rid-dedup): {n}")
    q = lambda a, p: np.percentile(a, p)
    print(f"plen (prompt tokens):    p50={q(plen,50):.0f} p90={q(plen,90):.0f} p99={q(plen,99):.0f} max={plen.max():.0f}")
    print(f"uncached (recompute):    p50={q(unc,50):.0f} p90={q(unc,90):.0f} p99={q(unc,99):.0f} max={unc.max():.0f}")
    print(f"aggregate hit-rate = {1 - unc.sum()/max(1,plen.sum()):.4f}  (uncached={unc.sum():,.0f} / prompt={plen.sum():,.0f})")
    print(f"device-hit tok={dev.sum():,.0f}  host-hit tok={host.sum():,.0f}  (host/dev={host.sum()/max(1,dev.sum()):.2f})")
    # cold turn-0 (dev+host ~0, big plen) vs warm-turn miss (some hit but big uncached) vs warm hit (small uncached)
    cold = (dev + host) < 0.05 * plen  # ~no prefix match => cold turn-0 (or fully-evicted conv)
    warm_hit = unc < 0.1 * plen        # mostly cached
    warm_miss = (~cold) & (~warm_hit)  # had partial match but still big uncached (evicted-mid-conv)
    print(f"\nrequest classes (by prefix-match fraction):")
    print(f"  COLD (match<5%):     {100*cold.mean():.1f}% of reqs, {100*unc[cold].sum()/max(1,unc.sum()):.1f}% of uncached work")
    print(f"  WARM-HIT (unc<10%):  {100*warm_hit.mean():.1f}% of reqs, {100*unc[warm_hit].sum()/max(1,unc.sum()):.1f}% of uncached work")
    print(f"  WARM-MISS (partial): {100*warm_miss.mean():.1f}% of reqs, {100*unc[warm_miss].sum()/max(1,unc.sum()):.1f}% of uncached work")
    print(f"\ntop-1% largest uncached hold {100*np.sort(unc)[::-1][:max(1,n//100)].sum()/max(1,unc.sum()):.1f}% of uncached work")
    print(f"congestion at admission: waiting-queue p50={q(wq,50):.0f} p90={q(wq,90):.0f} max={wq.max():.0f}")
    print("\nKEY: WARM-MISS work = AVOIDABLE recompute (conv prefix evicted mid-conversation). High WARM-MISS%")
    print("under pressure => residency headroom; if COLD dominates => irreducible cold-doc prefill (context-bound).")

if __name__ == "__main__":
    if len(sys.argv) < 2: print(__doc__); sys.exit(2)
    main(sys.argv[1])
