#!/usr/bin/env python3
"""Comprehensive analysis of ablation results.

Computes summary statistics, effect sizes, and rankings across all versions.
Usage: python3 analyze_results.py [--group] [--brief]
"""
import json
import os
import sys
import statistics
from collections import defaultdict

WS = "/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/sgl_mech/researchers/sgl_mech/runs"

# Version categorization
CONTROLS = {"v0-ctl", "v0-ctl2", "ctlC", "ctlD", "v0-ctl3", "v0-ctl4"}
COST_AWARE = {"v1-t2048", "t2048C", "t2048D"}

def load_all():
    results = {}
    for d in sorted(os.listdir(WS)):
        sf = os.path.join(WS, d, "summary.json")
        if os.path.isfile(sf):
            data = json.load(open(sf))
            results[d] = data["panel"]
    return results

def classify(v):
    if v in CONTROLS: return "CONTROL"
    if v in COST_AWARE: return "COST-AWARE-REF"
    if "gdsf" in v: return "GDSF"
    if "lfu" in v and "slfu" not in v: return "LFU"
    if "slru" in v: return "SLRU"
    if "costfreq" in v: return "COST-FREQ"
    if "contcost" in v: return "CONT-COST"
    if "writeadmit" in v: return "WRITE-ADMIT"
    if "sjf" in v: return "SJF"
    if "warmfirst" in v: return "WARM-FIRST"
    if "freqdecay" in v: return "FREQ-DECAY"
    if "sizelru" in v: return "SIZE-LRU"
    if "loadback" in v: return "LOADBACK"
    if "valuegate" in v: return "VALUE-GATE"
    if "wt2" in v or "wt3" in v: return "WT-THRESHOLD"
    if "fullstack" in v: return "FULL-STACK"
    if "backupcost" in v: return "BACKUP-AWARE"
    if "freqcost" in v: return "FREQ-COST"
    if "reuse" in v: return "REUSE-GATE"
    if "3tier" in v: return "3-TIER"
    if "depth" in v: return "DEPTH"
    if "t1024" in v or "t4096" in v or "t8192" in v: return "THRESHOLD"
    if v.startswith("v5") or v.startswith("v6"): return "COST-AWARE-VAR"
    if v.startswith("v7"): return "3-TIER"
    if v.startswith("v8"): return "DEPTH"
    if "sweep" in v or "knee" in v: return "SWEEP"
    return "OTHER"

def main():
    brief = "--brief" in sys.argv
    results = load_all()
    if not results:
        print("No results found.")
        return

    # Compute control baselines
    ctl_hr = [results[v]["overall/hit_rate"] for v in CONTROLS if v in results]
    ctl_p50 = [results[v]["overall/ttft_p50_ms"] for v in CONTROLS if v in results]
    ctl_p99 = [results[v]["overall/ttft_p99_ms"] for v in CONTROLS if v in results]
    ctl_rps = [results[v]["overall/req_throughput"] for v in CONTROLS if v in results]

    if ctl_hr:
        base_hr = statistics.mean(ctl_hr)
        base_p50 = statistics.mean(ctl_p50)
        base_p99 = statistics.mean(ctl_p99)
        base_rps = statistics.mean(ctl_rps)
        hr_std = statistics.stdev(ctl_hr) if len(ctl_hr) > 1 else 0.01
        p99_std = statistics.stdev(ctl_p99) if len(ctl_p99) > 1 else 500
    else:
        base_hr = base_p50 = base_p99 = base_rps = hr_std = p99_std = 0

    print("=" * 100)
    print("ABLATION RESULTS ANALYSIS")
    print("=" * 100)

    if ctl_hr:
        print(f"\nControl baselines (n={len(ctl_hr)}):")
        print(f"  hit_rate: {base_hr:.3f} ± {hr_std:.3f}")
        print(f"  p50:      {base_p50:.0f} ± {statistics.stdev(ctl_p50) if len(ctl_p50)>1 else 0:.0f} ms")
        print(f"  p99:      {base_p99:.0f} ± {p99_std:.0f} ms")
        print(f"  rps:      {base_rps:.1f}")

    # CostAware reference
    ca_hr = [results[v]["overall/hit_rate"] for v in COST_AWARE if v in results]
    if ca_hr:
        ca_mean = statistics.mean(ca_hr)
        print(f"\nCostAware@2048 reference (n={len(ca_hr)}):")
        print(f"  hit_rate: {ca_mean:.3f} ± {statistics.stdev(ca_hr) if len(ca_hr)>1 else 0:.3f}")
        print(f"  Δhit_rate vs control: +{ca_mean - base_hr:.3f}")

    # All versions ranked by hit_rate
    print(f"\n{'='*100}")
    print("ALL VERSIONS (ranked by hit_rate)")
    print(f"{'='*100}")
    print(f"{'rank':>4s}  {'version':>20s}  {'category':>15s}  {'hr':>6s}  {'Δhr':>6s}  {'σ':>4s}  {'p50':>6s}  {'p99':>6s}  {'rps':>4s}  {'verdict'}")
    print("-" * 100)

    sorted_versions = sorted(results.items(), key=lambda x: x[1].get("overall/hit_rate", 0), reverse=True)
    for rank, (v, p) in enumerate(sorted_versions, 1):
        hr = p.get("overall/hit_rate", 0)
        p50 = p.get("overall/ttft_p50_ms", 0)
        p99 = p.get("overall/ttft_p99_ms", 0)
        rps = p.get("overall/req_throughput", 0)
        cat = classify(v)
        dhr = hr - base_hr if base_hr else 0
        sigma = dhr / hr_std if hr_std > 0 else 0

        if dhr > 2 * hr_std:
            verdict = "WIN"
        elif dhr < -2 * hr_std:
            verdict = "LOSS"
        elif abs(dhr) <= hr_std:
            verdict = "NEUTRAL"
        else:
            verdict = "MARGINAL"

        print(f"{rank:4d}  {v:>20s}  {cat:>15s}  {hr:6.3f}  {dhr:+6.3f}  {sigma:+4.1f}  {p50:6.0f}  {p99:6.0f}  {rps:4.1f}  {verdict}")

    # Category summary
    if not brief:
        print(f"\n{'='*100}")
        print("CATEGORY SUMMARY")
        print(f"{'='*100}")
        cats = defaultdict(list)
        for v, p in results.items():
            cat = classify(v)
            hr = p.get("overall/hit_rate", 0)
            p99 = p.get("overall/ttft_p99_ms", 0)
            if hr > 0.5 and p99 < 20000:
                cats[cat].append({"v": v, "hr": hr, "p99": p99})

        print(f"{'category':>15s}  {'n':>3s}  {'mean_hr':>7s}  {'Δhr':>6s}  {'mean_p99':>8s}  {'verdict'}")
        print("-" * 60)
        for cat, items in sorted(cats.items(), key=lambda x: -statistics.mean([i["hr"] for i in x[1]])):
            n = len(items)
            mhr = statistics.mean([i["hr"] for i in items])
            mp99 = statistics.mean([i["p99"] for i in items])
            dhr = mhr - base_hr
            verdict = "WIN" if dhr > 2*hr_std else ("LOSS" if dhr < -2*hr_std else "NEUTRAL")
            print(f"{cat:>15s}  {n:3d}  {mhr:7.3f}  {dhr:+6.3f}  {mp99:8.0f}  {verdict}")

    print(f"\n{'='*100}")
    print(f"Total versions with results: {len(results)}")
    print(f"Versions with hit_rate > control + 2σ (WIN): "
          f"{sum(1 for v,p in results.items() if p.get('overall/hit_rate',0) > base_hr + 2*hr_std)}")
    print(f"Versions with hit_rate < control - 2σ (LOSS): "
          f"{sum(1 for v,p in results.items() if p.get('overall/hit_rate',0) < base_hr - 2*hr_std)}")

if __name__ == "__main__":
    main()
