#!/usr/bin/env python3
"""Trace-driven M/G/1 queueing model of the prefill server (Pollaczek-Khinchine).
Tests the thesis: cache lowers utilization rho (mean prefill work) but the p99 TTFT
tail is set by the heavy-tailed COLD turn-0 doc prefills, which are cache-immune ->
goodput@SLO is tail-limited, not hit-rate-limited. Parameterized by prefill rate P (tok/s);
calibrate P from the real baseline curve later. CPU-only, no engine deps."""
import json, statistics as st

S = json.load(open("tools/trace_stats.json"))
doc = S["doc_toks"]; nturns = S["nturns"]
Qm, Am = S["q_p50"], S["a_p50"]   # median follow-up Q and answer(output) tokens
SLO = 8.0

def prefill_tokens_stream(cache):
    """Per-request prefill token demand across all turns, under a cache model."""
    xs = []
    for d, n in zip(doc, nturns):
        # turn 0: doc + Q0 (Q0 ~ Qm). ALWAYS cold (unique doc, flushed).
        xs.append(d + Qm)
        acc = d + Qm + Am  # accumulated context after turn 0
        for t in range(1, n):
            if cache == "perfect":      # prefix fully resident -> only new Q
                xs.append(Qm)
            elif cache == "none":       # recompute full accumulated context every turn
                xs.append(acc)
            acc += Qm + Am
    return xs

def moments(xs):
    n=len(xs); m1=sum(xs)/n; m2=sum(x*x for x in xs)/n
    xs_s=sorted(xs); p99=xs_s[min(n-1,int(0.99*n))]; p50=xs_s[n//2]
    return n,m1,m2,p50,p99

def analyze(cache, P):
    xs = prefill_tokens_stream(cache)
    n,m1,m2,p50,p99 = moments(xs)
    print(f"\n[{cache} cache, P={P} tok/s]  requests={n}  E[tok]={m1:.0f} p50={p50} p99={p99}  E[tok^2]^.5={m2**.5:.0f}")
    print(f"  {'lam':>5} {'rho':>6} {'Wq_s':>8} {'ttft_p50_s':>10} {'ttft_p99_s~':>11} {'<=SLO?':>7}")
    good=0
    for lam in [3,5,7,10]:
        rho = lam*m1/P
        if rho>=1: 
            print(f"  {lam:>5} {rho:>6.2f} {'UNSTABLE':>8} {'-':>10} {'-':>11} {'no':>7}"); continue
        Wq = lam*(m2/P**2)/(2*(1-rho))        # P-K mean waiting time (s)
        # tail: for heavy-tailed service, p99 TTFT ~ wait + own p99 service time
        ttft_p50 = Wq + p50/P
        ttft_p99 = Wq + p99/P
        ok = ttft_p99 <= SLO
        if ok: good=lam
        print(f"  {lam:>5} {rho:>6.2f} {Wq:>8.2f} {ttft_p50:>10.2f} {ttft_p99:>11.2f} {('yes' if ok else 'no'):>7}")
    print(f"  => predicted goodput@SLO = {good} req... (lam units)")
    return good

print("="*70)
print("Thesis check: does cache (perfect vs none) move goodput@SLO, or just rho?")
print("p99 TTFT ~ Wq + (own p99 prefill tokens)/P. Turn-0 docs are cold in BOTH models.")
for P in [8000, 15000, 25000, 40000]:
    print("\n"+"#"*40+f" P={P} tok/s "+"#"*40)
    for cache in ["perfect","none"]:
        analyze(cache, P)
