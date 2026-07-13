#!/usr/bin/env python3
"""Deterministic prefill-contention analysis from the per-prefill trace (trace.rank0).

Tests the PREMISE of admission-staggering (§9d): does capping concurrent large cold
prefills remove real contention? Uses MEASURED in-flight intervals (no service model):
each chunked-prefill emits lines with decreasing `uncached`; a request is in-flight
while uncached>0, so its interval is [first t, last t with uncached>0] and its effective
prefill rate is unc_full / duration. When prefills overlap they share the chunk budget,
so contention would show as a rate drop with concurrency.

Result (whale-full, cert-lru-full @ λ=3): big(>=20K) prefills almost never co-execute
(peak concurrency 2, ~15-17% overlap, overlapped rate ~= solo rate, <=1.04x), and prefill
throughput is near-constant ~35K tok/s (mild <=17% penalty only at the rare 3-way overlap).
=> a concurrent-count cap has nothing to bite on; the tail is admission-WAIT, not prefill
execution contention. Refutes the earlier "admission-staggering is the lever" conjecture.

Usage: python3 sim/prefill_contention.py runs/whale-full/trace.rank0 [BIG]
"""
import json, collections, statistics as st, sys

def load(path):
    byrid = collections.defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            byrid[d["rid"]].append(d)
    recs = []
    for r, ls in byrid.items():
        unc_full = max(d["uncached"] for d in ls)
        active = [d["t"] for d in ls if d["uncached"] > 0]
        if not active:
            continue
        t0 = min(d["t"] for d in ls)
        t1 = max(max(active), t0)
        recs.append(dict(unc=unc_full, t0=t0, t1=t1))
    return recs

def concurrency_stats(recs, thr):
    xs = [x for x in recs if x["unc"] >= thr]
    xs.sort(key=lambda x: x["t0"])
    for x in xs:
        x["conc"] = sum(1 for y in xs if y["t0"] <= x["t0"] < y["t1"] and y is not x)
        dur = x["t1"] - x["t0"]
        x["rate"] = x["unc"] / dur if dur > 1e-3 else None
    ev = []
    for x in xs:
        ev += [(x["t0"], 1), (x["t1"], -1)]
    ev.sort()
    cur = peak = 0
    for _, d in ev:
        cur += d
        peak = max(peak, cur)
    return xs, peak

def report(path, BIG=20000):
    recs = load(path)
    print("=== %s ===" % path)
    # big-vs-big
    bigs, peak = concurrency_stats(recs, BIG)
    ovl = [x for x in bigs if x["conc"] >= 1 or any(
        y["t0"] < x["t1"] and x["t0"] < y["t1"] and y is not x for y in bigs)]
    # simpler overlap flag via interval intersection
    bigs.sort(key=lambda x: x["t0"])
    for x in bigs:
        x["ovl"] = False
    for i in range(len(bigs)):
        for j in range(i + 1, len(bigs)):
            if bigs[j]["t0"] >= bigs[i]["t1"]:
                break
            bigs[i]["ovl"] = bigs[j]["ovl"] = True
    solo = [x for x in bigs if not x["ovl"]]
    ovl = [x for x in bigs if x["ovl"]]
    def med_rate(xs):
        rs = [x["rate"] for x in xs if x["rate"]]
        return (st.median(rs), len(rs)) if rs else (0, 0)
    sr, sn = med_rate(solo); orr, on = med_rate(ovl)
    print(" big(>=%d): n=%d peak_concurrent=%d overlap=%.0f%% rate solo=%.0f ovl=%.0f (%.2fx)" % (
        BIG, len(bigs), peak, 100 * len(ovl) / max(1, len(bigs)), sr, orr, sr / orr if orr else 0))
    # general contention by concurrency bucket (>=2000 tok)
    gen, gpeak = concurrency_stats(recs, 2000)
    buckets = collections.defaultdict(list)
    for x in gen:
        if x["rate"]:
            buckets[min(x["conc"], 3)].append(x["rate"])
    print(" all(>=2000): n=%d peak_concurrent=%d" % (len(gen), gpeak))
    for c in sorted(buckets):
        print("   conc=%d: %6.0f tok/s (n=%d)" % (c, st.median(buckets[c]), len(buckets[c])))

if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "runs/whale-full/trace.rank0"
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    report(p, b)
