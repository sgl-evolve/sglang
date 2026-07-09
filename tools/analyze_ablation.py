#!/usr/bin/env python3
"""Analyze the eviction-policy ablation results. Prints a summary table
comparing all completed runs against the LRU reference, with z-scores.
Usage: .venv/bin/python tools/analyze_ablation.py
"""
import json, os, glob, statistics, math

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_run(ver):
    p = os.path.join(WS, "runs", ver, "summary.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        s = json.load(f)
    m = s.get("mix", {})
    if not m or m.get("hit_rate") is None:
        return None
    def g(k, d=0):
        v = m.get(k)
        return v if v is not None else d
    return {
        "version": ver,
        "hit": g("hit_rate"),
        "p50": g("ttft_median_ms"),
        "p99": g("ttft_p99_ms"),
        "mean": g("ttft_mean_ms"),
        "req_s": g("req_throughput"),
        "evict_ms": g("hicache_eviction_mean_ms"),
        "load_ms": g("hicache_load_back_mean_ms"),
        "host_util": g("hicache_host_util"),
    }

# LRU exclusive reference runs
LRU_VERS = ["v1_exclusive", "v2_exclusive_solo", "v2c_exclusive_rep",
            "v_exclusive_node03", "v_ab_exclusive", "v_ab2_exclusive"]

# Queued/new eviction policy runs
EVICT_VERS = {
    "random": "v_random_excl",
    "size_lru": "v_size_lru_excl",
    "fifo": "v_fifo_excl",
    "mru": "v_mru_excl",
    "filo": "v_filo_excl",
    "gdsf": "v_gdsf_excl",
    "2q": "v_2q_excl",
}

# Other experiments
OTHER = {
    "sjf_sched": "v_sjf_excl",
    "discard128": "v_discard128_excl",
    "discard512": "v_discard512_excl",
    "excl_rep3": "v_exclusive_rep3",
    "random_wb": "v_random_wb",
    "random_base": "v_random_base",
}

# Previously tested policies
PREV = {
    "lfu": "v_lfu_excl",
    "slru": "v_slru_excl",
    "queue_aware": "v_queue_aware_excl",
    "cost_lru": "v_cost_lru_t4096",
}

def main():
    # Load LRU reference
    lru_runs = [load_run(v) for v in LRU_VERS]
    lru_runs = [r for r in lru_runs if r]
    if not lru_runs:
        print("ERROR: no LRU reference runs found"); return
    lru_hits = [r["hit"] for r in lru_runs]
    lru_mean = statistics.mean(lru_hits)
    lru_std = statistics.stdev(lru_hits)
    lru_n = len(lru_hits)

    print(f"LRU reference: n={lru_n}, mean={lru_mean:.4f}, std={lru_std:.5f}")
    print()

    hdr = f"{'policy':>20s} {'version':>25s} {'hit':>7s} {'z':>7s} {'sig':>6s} {'Δpp':>7s} {'p50':>6s} {'p99':>6s} {'mean':>6s}"
    print(hdr)
    print("-" * len(hdr))

    all_policies = {}
    all_policies.update(EVICT_VERS)
    all_policies.update(PREV)
    all_policies.update(OTHER)

    for name, ver in sorted(all_policies.items(), key=lambda x: x[0]):
        r = load_run(ver)
        if not r:
            print(f"{name:>20s} {ver:>25s} {'—':>7s} {'—':>7s} {'PEND':>6s}")
            continue
        z = (r["hit"] - lru_mean) / lru_std if lru_std > 0 else 0
        sig = "NS" if abs(z) < 2 else ("WORSE" if z < 0 else "BETTER")
        delta = (r["hit"] - lru_mean) * 100
        print(f"{name:>20s} {ver:>25s} {r['hit']:7.4f} {z:7.2f} {sig:>6s} {delta:+7.2f} {r['p50']:6.0f} {r['p99']:6.0f} {r['mean']:6.0f}")

    # Add exclusive_rep3 to LRU reference if available
    rep3 = load_run("v_exclusive_rep3")
    if rep3:
        all_lru = lru_hits + [rep3["hit"]]
        print(f"\nUpdated LRU reference (with rep3): n={len(all_lru)}, mean={statistics.mean(all_lru):.4f}, std={statistics.stdev(all_lru):.5f}")

    # Cross-tiering comparison
    wb_lru = load_run("s_writeback")
    wb_rand = load_run("v_random_wb")
    base_lru = [load_run(v) for v in ["v_baseline_03", "v_baseline_node03", "v_baseline_ondem3", "v_ab_baseline", "v_ab2_baseline"]]
    base_lru = [r for r in base_lru if r]
    base_rand = load_run("v_random_base")

    if wb_rand and wb_lru:
        print(f"\n=== Cross-tiering random vs LRU ===")
        print(f"Write_back LRU:    hit={wb_lru['hit']:.4f}")
        print(f"Write_back random: hit={wb_rand['hit']:.4f}  Δ={((wb_rand['hit']-wb_lru['hit'])*100):+.2f}pp")
    if base_rand and base_lru:
        bl_mean = statistics.mean([r["hit"] for r in base_lru])
        print(f"Baseline LRU (n={len(base_lru)}): hit={bl_mean:.4f}")
        print(f"Baseline random:   hit={base_rand['hit']:.4f}  Δ={((base_rand['hit']-bl_mean)*100):+.2f}pp")

    # Summary statistics
    excl_hits = []
    for name, ver in {**EVICT_VERS, **PREV}.items():
        r = load_run(ver)
        if r and r["hit"] > 0.7:
            excl_hits.append(r["hit"])
    excl_hits.extend(lru_hits)
    if len(excl_hits) > 1:
        print(f"\n=== Overall eviction-policy spread (exclusive tier) ===")
        print(f"n={len(excl_hits)}, range=[{min(excl_hits):.4f}, {max(excl_hits):.4f}], spread={((max(excl_hits)-min(excl_hits))*100):.2f}pp")
        cap_effect = lru_mean - statistics.mean([r["hit"] for r in base_lru]) if base_lru else 0.13
        pol_spread = max(excl_hits) - min(excl_hits)
        if pol_spread > 0:
            print(f"Capacity:policy ratio = {cap_effect/pol_spread:.0f}:1")

if __name__ == "__main__":
    main()
