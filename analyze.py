#!/usr/bin/env python3
"""wilkes tail-decomposition / displacement diagnosis for a v0.31 sweep run.

Usage: python3 analyze.py runs/<version> [runs/<other> ...]
Reads curve.csv + metrics_r{R}.txt + bench_r{R}.json for each rate, prints a per-rate
decomposition and a displacement diagnosis, and computes goodput@SLO (p99 TTFT <= 8s).
Mechanism-agnostic; safe to run on any run dir the eval produced.
"""
import sys, os, re, json, csv

SLO_MS = 8000.0
RATES = [3, 5, 7, 10]

def parse_metrics(path):
    """Extract the sglang counters we care about from a /metrics dump."""
    if not os.path.exists(path):
        return {}
    m = open(path, errors="ignore").read()
    def sraw(metric):  # sum all label-sets of a counter
        vs = re.findall(r"(?m)^" + re.escape(metric) + r"\{[^}]*\}\s+([0-9.eE+]+)", m)
        return sum(float(x) for x in vs) if vs else None
    def by_label(metric, **lab):
        for l, v in re.findall(r"(?m)^" + re.escape(metric) + r"\{([^}]*)\}\s+([0-9.eE+]+)", m):
            if all(f'{k}="{val}"' in l for k, val in lab.items()):
                return float(v)
        return None
    def gauge_mean(metric):  # gauges: average across ranks
        vs = re.findall(r"(?m)^" + re.escape(metric) + r"\{[^}]*\}\s+([0-9.eE+]+)", m)
        return (sum(float(x) for x in vs) / len(vs)) if vs else None
    prm = sraw("sglang:prompt_tokens_total")
    dev = by_label("sglang:cached_tokens_total", cache_source="device") or 0.0
    hst = by_label("sglang:cached_tokens_total", cache_source="host") or 0.0
    out = {
        "prompt_tokens": prm,
        "cached_device": dev,
        "cached_host": hst,
        "hit_rate": round((dev + hst) / prm, 4) if prm else None,
        "gen_tokens": sraw("sglang:generation_tokens_total"),
        "evict_tokens": sraw("sglang:hicache_evict_tokens_total") or sraw("sglang:evict_tokens_total"),
        "load_back_tokens": sraw("sglang:hicache_load_back_tokens_total") or sraw("sglang:load_back_tokens_total"),
        "host_util": gauge_mean("sglang:hicache_host_util") or gauge_mean("sglang:host_util"),
    }
    # discover any hicache-ish counters we didn't name (so nothing is missed)
    extra = {}
    for name in sorted(set(re.findall(r"(?m)^(sglang:[a-z_]*(?:evict|load_back|backup|hicache|prefetch)[a-z_]*)\{", m))):
        v = sraw(name)
        if v:
            extra[name] = v
    out["_extra"] = extra
    return out

def analyze_run(run_dir):
    print(f"\n{'='*78}\nRUN: {run_dir}\n{'='*78}")
    # curve.csv
    curve = {}
    cc = os.path.join(run_dir, "curve.csv")
    if os.path.exists(cc):
        for r in csv.DictReader(open(cc)):
            try:
                curve[int(float(r["rate"]))] = r
            except Exception:
                pass
    # per-rate metrics deltas: counters are cumulative across the no-flush sweep, so
    # per-rate work = counter(R) - counter(prev R).
    prev = {"prompt_tokens": 0, "cached_device": 0, "cached_host": 0,
            "evict_tokens": 0, "load_back_tokens": 0, "gen_tokens": 0}
    hdr = f"{'rate':>4} {'req/s':>7} {'p50':>8} {'p99':>9} {'e2e_p99':>9} {'hit':>6} {'host_u':>6} " \
          f"{'d_prompt':>10} {'d_miss':>10} {'miss%':>6} {'d_evict':>11} {'d_loadbk':>11}"
    print(hdr); print("-" * len(hdr))
    goodput = 0.0
    for R in RATES:
        mt = parse_metrics(os.path.join(run_dir, f"metrics_r{R}.txt"))
        row = curve.get(R, {})
        req = row.get("req_throughput", "")
        p50 = row.get("ttft_p50_ms", ""); p99 = row.get("ttft_p99_ms", ""); e2e = row.get("e2e_p99_ms", "")
        hit = mt.get("hit_rate")
        hu = mt.get("host_util")
        pt, cd, ch = mt.get("prompt_tokens"), mt.get("cached_device"), mt.get("cached_host")
        ev, lb, gn = mt.get("evict_tokens"), mt.get("load_back_tokens"), mt.get("gen_tokens")
        d_prompt = (pt - prev["prompt_tokens"]) if pt is not None else None
        d_cached = ((cd + ch) - (prev["cached_device"] + prev["cached_host"])) if cd is not None else None
        d_miss = (d_prompt - d_cached) if (d_prompt is not None and d_cached is not None) else None
        missp = (100.0 * d_miss / d_prompt) if (d_miss is not None and d_prompt) else None
        d_ev = (ev - prev["evict_tokens"]) if ev is not None else None
        d_lb = (lb - prev["load_back_tokens"]) if lb is not None else None
        def fmt(x, w, d=0):
            if x is None or x == "":
                return " " * (w - 1) + "-"
            try:
                return f"{float(x):{w}.{d}f}"
            except Exception:
                return f"{str(x):>{w}}"
        print(f"{R:>4} {fmt(req,7,2)} {fmt(p50,8,0)} {fmt(p99,9,0)} {fmt(e2e,9,0)} "
              f"{fmt(hit,6,3)} {fmt(hu,6,3)} {fmt(d_prompt,10,0)} {fmt(d_miss,10,0)} "
              f"{fmt(missp,6,1)} {fmt(d_ev,11,0)} {fmt(d_lb,11,0)}")
        # goodput@SLO
        try:
            if p99 != "" and float(p99) <= SLO_MS and req != "":
                goodput = max(goodput, float(req))
        except Exception:
            pass
        if pt is not None:
            prev = {"prompt_tokens": pt, "cached_device": cd, "cached_host": ch,
                    "evict_tokens": ev or 0, "load_back_tokens": lb or 0, "gen_tokens": gn or 0}
    print("-" * len(hdr))
    print(f"GOODPUT@{int(SLO_MS)}ms = {goodput:.2f} req/s")
    print("\nDisplacement diagnosis:")
    print("  * hit% falling + d_miss/d_evict rising super-linearly as p99->SLO  => displacement/thrash (A/B)")
    print("  * hit% ~flat but p99 explodes                                      => queueing/HOL (D)")
    print("  * d_loadbk dominates and tracks p99                                => load-back on crit path (C)")
    # show any extra hicache counters at the top rate for reference
    top = parse_metrics(os.path.join(run_dir, f"metrics_r{RATES[-1]}.txt"))
    if top.get("_extra"):
        print("\n  extra hicache counters @ top rate:")
        for k, v in top["_extra"].items():
            print(f"    {k} = {v:.0f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    for d in sys.argv[1:]:
        analyze_run(d)
