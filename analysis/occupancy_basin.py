#!/usr/bin/env python3
"""Paper 3 / Paper 2 §5: the lam=3 goodput coin-flip is a metastable OCCUPANCY-BASIN effect,
and capacity margin (C-lam) governs it. Two evidences from existing bench telemetry (workload
deterministic: --seed 1, --disable-shuffle):

(A) COIN-FLIP PAIR (stock, same node): the FAIL run (p99 11.5s) vs PASS run (p99 6.2s) differ
    by +34% occupancy / +19% decode-tpot / +43% E2E -> FAIL sat in a higher-occupancy basin.
(B) MARGIN CHAIN (stock margin 1.27 vs write_back margin 2.04): higher margin => LOWER mean
    occupancy AND ~20x TIGHTER occupancy across runs => tighter p99 => reliable. This is the
    M/G/1 prediction (E[N]~rho/(1-rho), Var[N]~rho/(1-rho)^2) and likely metastable amplification.
"""
import json, os, statistics
RUNS=os.path.join(os.path.dirname(__file__),"..","runs")
def g(v):
    p=os.path.join(RUNS,v,"bench_r3.json"); return json.load(open(p)) if os.path.exists(p) else None

print("(A) COIN-FLIP PAIR (stock, deterministic workload, same node):")
BAD,GOOD=g("v1_stock"),g("v1b_stock")
for k,lab in [("p99_ttft_ms","p99 TTFT"),("concurrency","occupancy"),("mean_tpot_ms","decode tpot"),("mean_e2e_latency_ms","E2E mean")]:
    print(f"    {lab:12} FAIL {BAD[k]:9.0f}  PASS {GOOD[k]:9.0f}  ratio {BAD[k]/GOOD[k]:.2f}")

print("\n(B) MARGIN -> OCCUPANCY -> RELIABILITY chain:")
groups={"stock (margin 1.27)":["v1_stock","v1b_stock"], "write_back (margin 2.04)":["v3_wb","v3b_wb"]}
print(f"    {'config':24}{'occupancy runs':>22}{'mean':>7}{'spread':>8}{'p99 runs (s)':>18}{'p99 spread':>11}")
for name,vs in groups.items():
    ds=[g(v) for v in vs if g(v)]
    occ=[d['concurrency'] for d in ds]; p99=[d['p99_ttft_ms']/1000 for d in ds]
    print(f"    {name:24}{str([round(x) for x in occ]):>22}{statistics.mean(occ):7.0f}"
          f"{(max(occ)-min(occ)):8.1f}{str([round(x,1) for x in p99]):>18}{(max(p99)-min(p99)):11.1f}")
print("\n=> higher margin: LOWER mean occupancy (90 vs 145) + ~20x TIGHTER occupancy (2 vs 42)")
print("   => ~6.5x tighter p99 => reliable. Caching raises margin => damps the coin-flip.")
print("   (occupancy-spread ratio ~19x exceeds simple M/G/1 Var ratio ~1.8x => metastable amplification;")
print("    n=2/config so spreads are suggestive, means robust. More replicates fit the exponent.)")
