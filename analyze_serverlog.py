#!/usr/bin/env python3
"""Resolve the FORK criterion from a baseline server.log (+ conv_trace.json).

sglang logs per prefill batch:
  "Prefill batch, #new-seq: N, #new-token: X, #cached-token: Z, ... #running-req: R, #queue-req: Q,
   #pending-token: P, ..., input throughput (token/s): T"
We extract: prefill throughput P (input tok/s), per-batch new/cached (hit rate + work), congestion
(queue/running), and the #new-token distribution (cold-whale batches). Then apply the p99-irreducibility
criterion: %turns whose PERFECT-cache cold-prefill exceeds the 8s SLO at the measured P.

Usage: python3 analyze_serverlog.py runs/<version>/server.log
"""
import sys, os, re, json, numpy as np

def _ts(line):
    m = re.search(r"\[(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)", line)
    if not m: return None
    Y, Mo, D, h, mi, s = map(int, m.groups())
    return ((D * 24 + h) * 60 + mi) * 60 + s  # seconds within the month (fine for deltas)

def main(path):
    txt = open(path, errors="ignore").read()
    rows = []
    # timestamp-based prefill rate: pair each prefill-batch line with its ts; Δt to next prefill line
    pf_lines = [(l, _ts(l)) for l in txt.splitlines() if "Prefill batch" in l]
    for m in re.finditer(
        r"#new-seq:\s*(\d+),\s*#new-token:\s*(\d+),\s*#cached-token:\s*(\d+)"
        r".*?#running-req:\s*(\d+),\s*#queue-req:\s*(\d+),\s*#pending-token:\s*(\d+)"
        r".*?input throughput \(token/s\):\s*([0-9.]+)", txt):
        ns, nt, ct, rr, qr, pt, thr = m.groups()
        rows.append(dict(new_seq=int(ns), new_tok=int(nt), cached_tok=int(ct),
                         run=int(rr), queue=int(qr), pending=int(pt), thr=float(thr)))
    if not rows:
        print("no prefill-batch lines found; check log path/format"); return
    # robust P: for consecutive LARGE prefill batches (#new-token>=2000, back-to-back), tok/s = newtok/Δt
    rates = []
    ntoks = [int(re.search(r"#new-token:\s*(\d+)", l).group(1)) for l, _ in pf_lines if re.search(r"#new-token:\s*(\d+)", l)]
    tss = [t for _, t in pf_lines]
    for i in range(len(pf_lines) - 1):
        nt_i = ntoks[i] if i < len(ntoks) else 0
        if nt_i >= 2000 and tss[i] is not None and tss[i+1] is not None:
            dt = tss[i+1] - tss[i]
            if 0 < dt <= 30: rates.append(nt_i / dt)
    if rates:
        rates = np.array(rates)
        print(f"** timestamp-based prefill rate on large chunks (#new>=2000): n={len(rates)} "
              f"p50={np.median(rates):.0f} p90={np.percentile(rates,90):.0f} tok/s  (ROBUST P estimate) **")
    else:
        print("** no large (#new>=2000) back-to-back prefill batches yet -> P estimate not ready (warmup only) **")
    thr = np.array([r["thr"] for r in rows if r["thr"] > 0])
    newt = np.array([r["new_tok"] for r in rows])
    cach = np.array([r["cached_tok"] for r in rows])
    q = np.array([r["queue"] for r in rows]); run = np.array([r["run"] for r in rows])
    print(f"prefill batches: {len(rows)}")
    print(f"input throughput tok/s (=P): p50={np.percentile(thr,50):.0f} p90={np.percentile(thr,90):.0f} "
          f"max={thr.max():.0f} mean={thr.mean():.0f}")
    tot_new, tot_cached = newt.sum(), cach.sum()
    print(f"aggregate hit-rate (cached/(new+cached)) = {tot_cached/max(1,tot_new+tot_cached):.3f}  "
          f"(new={tot_new:,} cached={tot_cached:,})")
    print(f"#new-token/batch: p50={np.percentile(newt,50):.0f} p90={np.percentile(newt,90):.0f} max={newt.max()}")
    print(f"congestion: queue p50={np.percentile(q,50):.0f} p90={np.percentile(q,90):.0f} max={q.max()} | "
          f"running p50={np.percentile(run,50):.0f} p90={np.percentile(run,90):.0f} max={run.max()}")
    # fork criterion using measured P
    tr = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim", "conv_trace.json")
    if os.path.exists(tr):
        convs = json.load(open(tr))
        perfect = np.array([inp for c in convs for inp, _ in [c[0]] ] + [inp for c in convs for inp, _ in c[1:]])
        # rebuild per-turn perfect uncached properly:
        perfect = []
        for c in convs:
            for i, (inp, out) in enumerate(c):
                perfect.append(inp)
        perfect = np.array(perfect)
        for P in sorted(set([int(np.percentile(thr, 50)), int(np.percentile(thr, 25)), int(thr.mean())])):
            if P <= 0: continue
            b = 8 * P
            frac = 100 * np.mean(perfect > b)
            print(f"  @P={P}: 8s budget={b:,} tok -> %turns(perfect-cold-prefill>8s)={frac:.2f}% "
                  f"-> {'IRREDUCIBLE p99 (bounded-negative)' if frac > 1.0 else 'cache CAN affect p99 (mechanism viable)'}")
    print("\nNOTE: measured P here is AGGREGATE prefill throughput (shared across concurrent prefills under")
    print("chunked prefill); a single whale's effective rate is lower, so this is an OPTIMISTIC P (lower")
    print("bound on irreducibility). Cross-check with curve.csv p99 vs lambda + hit-rate vs lambda.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    main(sys.argv[1])
