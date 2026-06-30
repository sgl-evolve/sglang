#!/usr/bin/env python3
"""Parse one version's eval outputs into summary.json.
Usage: parse_results.py <out_dir> [version_label] [commit]
Reads: loogle_result.json (bench_serving JSONL), bench_mix.out (bench_mix stdout),
       *_metrics_post.txt (prometheus /metrics snapshots).
Writes: <out_dir>/summary.json
"""
import json, os, re, sys


def parse_prom(path):
    """Return {metric_name: float} for simple gauge/counter lines (last value wins)."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|\+Inf|-Inf)$", line)
            if not m:
                continue
            name, labels, val = m.group(1), m.group(2) or "", m.group(3)
            try:
                v = float(val)
            except ValueError:
                continue
            key = name + labels
            out[key] = v
    return out


def cache_metrics(prom):
    """Extract cache-related metrics; compute a hit_rate if token counters present."""
    cm = {k: v for k, v in prom.items() if ("cache" in k.lower() or "prefix" in k.lower() or "hierarchical" in k.lower())}
    return cm


def hit_rate_from(prom):
    """Best-effort prefix-cache hit rate from token counters."""
    # sglang exposes counters like sglang:prompt_tokens_total and sglang:cached_tokens_total (names vary by ver)
    def find(substr):
        for k, v in prom.items():
            if substr in k.lower():
                return v
        return None
    cached = None
    prompt = None
    for k, v in prom.items():
        kl = k.lower()
        if "cached_token" in kl:
            cached = (cached or 0) + v
        if re.search(r"prompt_token", kl) and "cached" not in kl:
            prompt = (prompt or 0) + v
    if cached is not None and prompt:
        try:
            return cached / prompt
        except ZeroDivisionError:
            return None
    return None


def parse_loogle(out_dir):
    p = os.path.join(out_dir, "loogle_result.json")
    res = {}
    if os.path.exists(p):
        with open(p) as f:
            lines = [l for l in f if l.strip()]
        if lines:
            d = json.loads(lines[-1])  # last run
            res = d
    prom = parse_prom(os.path.join(out_dir, "loogle_metrics_post.txt"))
    out = {
        "ttft_mean_ms": res.get("mean_ttft_ms"),
        "ttft_median_ms": res.get("median_ttft_ms"),
        "ttft_p99_ms": res.get("p99_ttft_ms"),
        "ttft_std_ms": res.get("std_ttft_ms"),
        "tpot_mean_ms": res.get("mean_tpot_ms"),
        "tpot_median_ms": res.get("median_tpot_ms"),
        "tpot_p99_ms": res.get("p99_tpot_ms"),
        "itl_mean_ms": res.get("mean_itl_ms"),
        "itl_median_ms": res.get("median_itl_ms"),
        "itl_p99_ms": res.get("p99_itl_ms"),
        "e2e_mean_ms": res.get("mean_e2e_latency_ms"),
        "e2e_median_ms": res.get("median_e2e_latency_ms"),
        "e2e_p99_ms": res.get("p99_e2e_latency_ms"),
        "out_tok_s": res.get("output_throughput"),
        "in_tok_s": res.get("input_throughput"),
        "req_throughput": res.get("request_throughput"),
        "completed": res.get("completed"),
        "total_input_tokens": res.get("total_input_tokens"),
        "total_output_tokens": res.get("total_output_tokens"),
        "concurrency": res.get("concurrency"),
        "duration_s": res.get("duration"),
    }
    hr = hit_rate_from(prom)
    if hr is not None:
        out["hit_rate"] = hr
    out["cache_metrics"] = cache_metrics(prom)
    return out


def parse_sharegpt(out_dir):
    p = os.path.join(out_dir, "bench_mix.out")
    txt = ""
    if os.path.exists(p):
        with open(p, errors="ignore") as f:
            txt = f.read()

    def g(pat, cast=float):
        m = re.search(pat, txt)
        return cast(m.group(1)) if m else None

    # bench_mix prints TTFT/latency in SECONDS; convert to ms.
    ttft_s = g(r"Average TTFT:\s*([0-9.]+)")
    p90_ttft_s = g(r"P90 TTFT:\s*([0-9.]+)")
    med_ttft_s = g(r"Median TTFT:\s*([0-9.]+)")
    lat_s = g(r"Average latency:\s*([0-9.]+)")
    p90_lat_s = g(r"P90 latency:\s*([0-9.]+)")
    med_lat_s = g(r"Median latency:\s*([0-9.]+)")
    out = {
        "ttft_mean_ms": ttft_s * 1000 if ttft_s is not None else None,
        "ttft_p90_ms": p90_ttft_s * 1000 if p90_ttft_s is not None else None,
        "ttft_median_ms": med_ttft_s * 1000 if med_ttft_s is not None else None,
        "latency_mean_ms": lat_s * 1000 if lat_s is not None else None,
        "latency_p90_ms": p90_lat_s * 1000 if p90_lat_s is not None else None,
        "latency_median_ms": med_lat_s * 1000 if med_lat_s is not None else None,
        "throughput_req_s": g(r"Throughput:\s*([0-9.]+)"),
        "cache_hit_rate": g(r"Cache Hit Rate:\s*([0-9.]+)"),
        "total_requests": g(r"Total requests:\s*([0-9]+)", int),
    }
    prom = parse_prom(os.path.join(out_dir, "sharegpt_metrics_post.txt"))
    out["cache_metrics"] = cache_metrics(prom)
    return out


def main():
    out_dir = sys.argv[1]
    version = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(out_dir.rstrip("/"))
    commit = sys.argv[3] if len(sys.argv) > 3 else None
    summary = {
        "version": version,
        "commit": commit,
        "loogle": parse_loogle(out_dir),
        "sharegpt": parse_sharegpt(out_dir),
    }
    sp = os.path.join(out_dir, "summary.json")
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("WROTE", sp)


if __name__ == "__main__":
    main()
