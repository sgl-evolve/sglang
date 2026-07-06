#!/usr/bin/env python3
"""Ready-to-fire verdict: slfu vs the held LRU baseline. Run after each v20-be-slfu* completes.

Compares the slfu eviction runs' hit-rate distribution against the 5-run best_effort+LRU baseline
(the eviction-timing finding: same fixed workload, so hit rate reflects retention policy). Tests both
predictions: (1) slfu RAISES mean hit rate, (2) slfu SHRINKS its variance. No GPU; pure analysis.

Usage: python3 compare_slfu.py
"""
import json, os, statistics

# The 5 best_effort+LRU (default) runs = the held baseline distribution.
LRU_BASELINE = ["v4-parallel-besteffort", "v4r-repeat", "v4r2-repeat", "v16-be-lfu", "v17-be-slru"]
# slfu runs (appear as they complete).
SLFU_RUNS = ["v20-be-slfu", "v20b-be-slfu"]
SLRU_RUNS = ["v18-be-slru"]   # built-in reuse-aware reference


def load(d):
    s = f"runs/{d}/summary.json"
    if not os.path.exists(s):
        return None
    j = json.load(open(s))
    m = j.get("mix", {})
    return dict(
        ttft=m.get("ttft_mean_ms"), hit=m.get("hit_rate"),
        host_util=m.get("hicache_host_util"),
        evict=m.get("hicache_evicted_tokens"),
        prompt=m.get("hicache_prompt_tokens"),
        l3=m.get("hicache_hit_storage_frac"),
        pf=j.get("resolved_args", {}).get("hicache_storage_prefetch_policy"),
        commit=j.get("commit"),
    )


def collect(names):
    out = []
    for d in names:
        r = load(d)
        if r and r["hit"] is not None:
            out.append((d, r))
    return out


def stats(vals):
    if not vals:
        return None
    m = statistics.mean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else float("nan")
    return m, sd


def main():
    base = collect(LRU_BASELINE)
    slfu = collect(SLFU_RUNS)
    slru = collect(SLRU_RUNS)

    bh = [r["hit"] for _, r in base]
    bt = [r["ttft"] for _, r in base if r["ttft"] is not None]
    bm, bsd = stats(bh)
    btm, btsd = stats(bt)
    print(f"=== LRU baseline (n={len(bh)}): hit {bm:.4f} ± {bsd:.4f} | ttft {btm:.0f} ± {btsd:.0f} ===")
    for d, r in base:
        print(f"    {d:24s} hit={r['hit']:.4f} ttft={r['ttft'] and round(r['ttft'])}")

    # Sanity: same fixed workload across baseline (the finding's premise).
    prompts = [r["prompt"] for _, r in base if r["prompt"]]
    if prompts:
        spread = (max(prompts) - min(prompts)) / statistics.mean(prompts)
        print(f"    workload-fixed check: prompt_tok spread = {spread*100:.4f}% (expect ~0)")

    for label, runs in [("slfu (NOVEL)", slfu), ("slru (ref)", slru)]:
        if not runs:
            print(f"\n=== {label}: no runs yet ===")
            continue
        hits = [r["hit"] for _, r in runs]
        m, sd = stats(hits)
        print(f"\n=== {label} (n={len(hits)}): hit {m:.4f}" + (f" ± {sd:.4f}" if len(hits) > 1 else "") + " ===")
        for d, r in runs:
            l3 = f" L3={r['l3']:.2f}" if r["l3"] is not None else ""
            print(f"    {d:24s} hit={r['hit']:.4f} ttft={r['ttft'] and round(r['ttft'])} host_util={r['host_util']}{l3} commit={r['commit']}")
        # Prediction 1: raises mean hit. z vs baseline (per-run, using baseline sd).
        if bsd and bsd == bsd:
            z = (m - bm) / bsd
            print(f"    P1 mean-hit: Δ={m-bm:+.4f} = {z:+.2f}σ of baseline sd ({bsd:.4f})")
            if len(hits) >= 2:
                # unpaired 2σ threshold at these n's
                import math
                thr = 2 * bsd * math.sqrt(1/len(hits) + 1/len(bh))
                verdict = "ABOVE noise" if (m - bm) > thr else ("BELOW noise" if abs(m-bm) < thr else "borderline")
                print(f"       2σ unpaired threshold |Δ|>{thr:.4f} -> {verdict}")
        # Prediction 2: shrinks variance.
        if len(hits) > 1 and bsd and bsd == bsd:
            print(f"    P2 variance: slfu sd={sd:.4f} vs baseline sd={bsd:.4f} -> {'SHRINKS' if sd < bsd else 'no shrink'}")

    print("\n(reminder: confirm slfu is on the ACTIVE path — grep server.log for the eviction-policy echo)")


if __name__ == "__main__":
    main()
