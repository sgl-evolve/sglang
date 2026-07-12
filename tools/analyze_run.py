#!/usr/bin/env python3
"""Diagnostic analyzer for a kleinrock eval run dir (runs/<version>/).
Parses curve.csv + per-rate metrics_r*.txt (cumulative Prometheus dumps) and
decomposes the goodput@SLO curve + the load-back / eviction / hit dynamics per rate,
diffing consecutive cumulative dumps to get per-rate deltas. Read-only; no engine deps.

Usage: python3 tools/analyze_run.py runs/<version>
"""
import sys, os, re, csv, json

SLO_MS = 8000.0

def parse_metrics(path):
    """Return dict: counters{name: total}, hist{name: {le: cumcount, '_sum':x, '_count':n}}, gauges{name:last}."""
    counters, hist, gauges = {}, {}, {}
    if not os.path.exists(path):
        return counters, hist, gauges
    txt = open(path, errors="ignore").read()
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(sglang:[a-zA-Z0-9_]+)(\{[^}]*\})?\s+([0-9.eE+\-]+)$", line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", float(m.group(3))
        if name.endswith("_bucket"):
            base = name[:-7]
            le = re.search(r'le="([^"]+)"', labels)
            if le:
                hist.setdefault(base, {}).setdefault("_buckets", {})
                # sum across label-sets (ranks) for same le
                hist[base]["_buckets"][le.group(1)] = hist[base]["_buckets"].get(le.group(1), 0.0) + val
        elif name.endswith("_sum"):
            hist.setdefault(name[:-4], {})["_sum"] = hist.get(name[:-4], {}).get("_sum", 0.0) + val
        elif name.endswith("_count"):
            hist.setdefault(name[:-6], {})["_count"] = hist.get(name[:-6], {}).get("_count", 0.0) + val
            counters[name] = counters.get(name, 0.0) + val
        else:
            # sum counters across ranks; gauges: keep max (approx)
            counters[name] = counters.get(name, 0.0) + val
            gauges[name] = max(gauges.get(name, val), val)
    return counters, hist, gauges

def hist_pct(bucketmap, total, q):
    """Approx quantile q (0..1) from cumulative bucket counts {le: cumcount}."""
    if not bucketmap or not total:
        return None
    items = sorted(((float("inf") if le == "+Inf" else float(le)), c) for le, c in bucketmap.items())
    target = q * total
    prev_le, prev_c = 0.0, 0.0
    for le, c in items:
        if c >= target:
            if le == float("inf"):
                return prev_le
            # linear interp within bucket
            frac = (target - prev_c) / max(1e-9, c - prev_c)
            return prev_le + frac * (le - prev_le)
        prev_le, prev_c = le, c
    return items[-1][0]

def main():
    d = sys.argv[1]
    print(f"=== analyze {d} ===")
    # curve
    cc = os.path.join(d, "curve.csv")
    rows = []
    if os.path.exists(cc):
        rows = list(csv.DictReader(open(cc)))
        print("\n-- curve.csv --")
        print(f"{'rate':>4} {'req/s':>7} {'tok/s':>8} {'ttftp50':>9} {'ttftp99':>9} {'e2ep99':>9} {'hit':>6}")
        for r in rows:
            g = lambda k: r.get(k, "") or "-"
            print(f"{g('rate'):>4} {g('req_throughput'):>7} {g('out_tok_s'):>8} {g('ttft_p50_ms'):>9} {g('ttft_p99_ms'):>9} {g('e2e_p99_ms'):>9} {g('hit_rate'):>6}")
        good = max([float(r["req_throughput"]) for r in rows
                    if r.get("ttft_p99_ms") and float(r["ttft_p99_ms"]) <= SLO_MS] or [0])
        print(f"\n  GOODPUT@{int(SLO_MS)}ms = {good:.2f} req/s")

    # per-rate metric deltas
    rates = [r["rate"] for r in rows] if rows else ["3", "5", "7", "10"]
    prev_c = {}
    prev_hb = {}
    print("\n-- per-rate deltas (from cumulative /metrics) --")
    print(f"{'rate':>4} {'prompt_tok':>12} {'cache_dev':>11} {'cache_host':>11} {'hit%':>6} {'loadback_tok':>13} {'evict_tok':>12} {'lb_mean_ms':>10} {'lb_p99_ms':>10}")
    for R in rates:
        c, h, g = parse_metrics(os.path.join(d, f"metrics_r{R}.txt"))
        def dc(name):
            return c.get(name, 0.0) - prev_c.get(name, 0.0)
        prm = dc("sglang:prompt_tokens_total")
        dev = dc("sglang:cached_tokens_total") if "sglang:cached_tokens_total" in c else 0.0
        # cached by source needs per-label; fall back to overall
        lb = dc("sglang:load_back_tokens_total")
        ev = dc("sglang:eviction_tokens_total")
        # load-back duration histogram delta
        lbh = h.get("sglang:load_back_duration_seconds", {})
        buckets = lbh.get("_buckets", {})
        dbuckets = {le: buckets.get(le, 0.0) - prev_hb.get(le, 0.0) for le in buckets}
        dcount = lbh.get("_count", 0.0) - prev_hb.get("_count", 0.0)
        dsum = lbh.get("_sum", 0.0) - prev_hb.get("_sum", 0.0)
        lb_mean = (dsum / dcount * 1000) if dcount else 0.0
        lb_p99 = hist_pct(dbuckets, dcount, 0.99)
        lb_p99 = (lb_p99 * 1000) if lb_p99 else 0.0
        hitpct = 100.0 * (dev / prm) if prm else 0.0
        print(f"{R:>4} {prm:>12.0f} {dev:>11.0f} {'':>11} {hitpct:>6.1f} {lb:>13.0f} {ev:>12.0f} {lb_mean:>10.2f} {lb_p99:>10.1f}")
        prev_c = dict(c)
        prev_hb = dict(buckets); prev_hb["_count"] = lbh.get("_count", 0.0); prev_hb["_sum"] = lbh.get("_sum", 0.0)

    # dump all counter names seen (last rate) for discovery
    c, h, g = parse_metrics(os.path.join(d, f"metrics_r{rates[-1]}.txt"))
    print("\n-- available sglang counters (last dump) --")
    for k in sorted(c):
        if "cached_tokens" in k or "load_back" in k or "evict" in k or "prompt_tokens" in k or "num_" in k or "queue" in k or "token_usage" in k:
            print(f"   {k} = {c[k]:.0f}")
    print("\n-- histograms available --")
    for k in sorted(h):
        print(f"   {k}  count={h[k].get('_count',0):.0f} sum={h[k].get('_sum',0):.2f}")

if __name__ == "__main__":
    main()
