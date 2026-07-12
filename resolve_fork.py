#!/usr/bin/env python3
"""One-shot fork resolver for a screen/eval run: combines curve.csv + server.log + trace.rank0 into the
two-bound verdict goodput@SLO = min(R_thru, R_p99).

Usage: python3 resolve_fork.py runs/<ver>
Prints: goodput@SLO, R_p99 (rate where p99 crosses 8s), P, avg_uncached, R_thru, warm-miss% of prefill,
and the class (cold-turn0 vs warm-miss) of the p99-tail requests => cache-affectable vs cold-doc-bound.
"""
import sys, os, re, json, csv
import numpy as np

SLO = 8000.0

def read_curve(d):
    p = os.path.join(d, "curve.csv")
    if not os.path.exists(p): return []
    return [r for r in csv.DictReader(open(p)) if r.get("req_throughput")]

def P_from_serverlog(d):
    p = os.path.join(d, "server.log")
    if not os.path.exists(p): return None
    m = open(p, errors="ignore").read()
    thr = [float(x) for x in re.findall(r"input throughput \(token/s\):\s*([0-9.]+)", m)]
    # steady-state: use the upper half (under-load batches), median
    thr = [t for t in thr if t > 1000]
    return float(np.median(thr)) if thr else None

def trace_stats(d):
    # find trace.rank0 (or .rank*); dedup by rid (first)
    cand = [os.path.join(d, f) for f in os.listdir(d) if f.startswith("trace")] if os.path.isdir(d) else []
    r0 = [c for c in cand if c.endswith("rank0")] or cand
    if not r0: return None
    first = {}
    for l in open(r0[0], errors="ignore"):
        l = l.strip()
        if not l: continue
        try: r = json.loads(l)
        except Exception: continue
        if r.get("rid") not in first: first[r["rid"]] = r
    rows = list(first.values())
    if not rows: return None
    plen = np.array([r["plen"] for r in rows], float)
    unc = np.array([r["uncached"] for r in rows], float)
    dev = np.array([r["dev"] for r in rows], float); host = np.array([r["host"] for r in rows], float)
    cold = (dev + host) < 0.05 * np.maximum(plen, 1)
    warm_hit = unc < 0.1 * np.maximum(plen, 1)
    warm_miss = (~cold) & (~warm_hit)
    # p99-tail requests: top 1% by uncached -> what class?
    thr = np.percentile(unc, 99)
    tail = unc >= thr
    return dict(n=len(rows), avg_unc=float(unc.mean()), hit=1 - unc.sum()/max(1, plen.sum()),
                cold_frac=float(cold.mean()), cold_work=float(unc[cold].sum()/max(1,unc.sum())),
                wmiss_frac=float(warm_miss.mean()), wmiss_work=float(unc[warm_miss].sum()/max(1,unc.sum())),
                tail_cold_frac=float(cold[tail].mean()), tail_wmiss_frac=float(warm_miss[tail].mean()))

def main(d):
    rows = read_curve(d)
    print(f"=== {d} ===")
    if rows:
        print(f"{'rate':>4} {'req/s':>7} {'p50':>8} {'p99':>10} {'e2e_p99':>10} {'hit':>6}")
        good = 0.0
        for r in rows:
            f = lambda k: (float(r[k]) if r.get(k) not in (None, "") else None)
            p99 = f("ttft_p99_ms"); req = f("req_throughput")
            print(f"{r['rate']:>4} {req or 0:>7.2f} {f('ttft_p50_ms') or 0:>8.0f} {p99 or 0:>10.0f} "
                  f"{f('e2e_p99_ms') or 0:>10.0f} {r.get('hit_rate',''):>6}")
            if p99 and p99 <= SLO and req: good = max(good, req)
        print(f"GOODPUT@SLO (R_p99-bound, max req/s w/ p99<=8s) = {good:.2f} req/s")
    else:
        print("(no curve rows yet)")
    P = P_from_serverlog(d)
    ts = trace_stats(d)
    if P: print(f"P (prefill tok/s, steady-state median) = {P:.0f}")
    if ts:
        Rthru = P / ts["avg_unc"] if (P and ts["avg_unc"]) else None
        print(f"trace: n={ts['n']} hit={ts['hit']:.3f} avg_uncached={ts['avg_unc']:.0f}")
        print(f"  COLD {100*ts['cold_frac']:.0f}%reqs/{100*ts['cold_work']:.0f}%work | "
              f"WARM-MISS {100*ts['wmiss_frac']:.0f}%reqs/{100*ts['wmiss_work']:.0f}%work")
        print(f"  p99-tail requests: {100*ts['tail_cold_frac']:.0f}% COLD, {100*ts['tail_wmiss_frac']:.0f}% WARM-MISS")
        if Rthru: print(f"  R_thru = P/avg_uncached = {Rthru:.2f} req/s")
        print("\nVERDICT:")
        print("  if p99-tail is mostly COLD & warm-miss%work small => cold-doc-bound (BOUNDED-NEGATIVE: cache can't move goodput)")
        print("  if p99-tail is mostly WARM-MISS & warm-miss%work large => CACHE-AFFECTABLE (screen residency policies)")

if __name__ == "__main__":
    if len(sys.argv) < 2: print(__doc__); sys.exit(2)
    main(sys.argv[1])
