#!/usr/bin/env python3
"""kleinrock — Direction 3 (movement axis): is L1<->L2 KV movement a goodput@SLO lever?

Consolidates the evidence that the L2->L1 restore (and D->L2 eviction) DMA is NOT on the
critical path of the p99 TTFT tail:

  (A) ENQUEUE cost      : sglang:load_back_duration_seconds / eviction_duration_seconds histograms
                          from the eval's own captured /metrics (these wrap only the enqueue).
  (B) ACTUAL DMA cost   : [KLDMA] lines from a server.log run with KL_DMA_TIMING=1
                          (CUDA-event elapsed_time of the real load_stream H->D copy).
  (C) EXPOSED cost      : per-request TTFT vs prompt_len for NON-QUEUED cache-hit reqs
                          (perreq_*.csv) — flatness => restore is not exposed on TTFT.
  (D) COST HIERARCHY    : restore tok/s (from B) vs prefill compute tok/s (from bench input tput)
                          => how many x cheaper is moving KV than recomputing it.

Usage:
  python3 tools/movement_analyze.py                      # A + C from existing runs (GPU-free)
  python3 tools/movement_analyze.py runs/dma_timing_*.out   # + B (actual DMA) from a KLDMA log
Deterministic, GPU-free (parses artifacts only).
"""
import sys, re, os, glob, csv


def parse_hist(txt, name):
    b = []
    s = c = None
    for line in txt.splitlines():
        if line.startswith(name + "_bucket"):
            m = re.search(r'le="([^"]+)".*?\}\s+([0-9.eE+]+)$', line)
            if m:
                le = float(m.group(1)) if m.group(1) != "+Inf" else float("inf")
                b.append((le, float(m.group(2))))
        elif line.startswith(name + "_sum"):
            m = re.search(r"\s+([0-9.eE+]+)$", line)
            if m:
                s = float(m.group(1))
        elif line.startswith(name + "_count"):
            m = re.search(r"\s+([0-9.eE+]+)$", line)
            if m:
                c = float(m.group(1))
    return b, s, c


def hpc(b, q):
    if not b or b[-1][1] == 0:
        return None
    t = q * b[-1][1]
    for le, cum in b:
        if cum >= t:
            return le
    return b[-1][0]


def section_A(run="runs/v0-stock"):
    print("=" * 78)
    print("(A) ENQUEUE cost of movement (load_back = restore H->D; eviction = backup D->H)")
    print("    [from the eval's OWN captured /metrics; these histograms wrap only the enqueue]")
    print("=" * 78)
    for lam in ["3", "5", "7", "10"]:
        f = f"{run}/metrics_r{lam}.txt"
        if not os.path.exists(f):
            continue
        t = open(f, errors="ignore").read()
        lb, ls, lc = parse_hist(t, "sglang:load_back_duration_seconds")
        ev, es, ec = parse_hist(t, "sglang:eviction_duration_seconds")
        ltok = re.search(r"sglang:load_back_tokens_total\S*\s+([0-9.eE+]+)", t)
        etok = re.search(r"sglang:evicted_tokens_total\S*\s+([0-9.eE+]+)", t)
        ltok = float(ltok.group(1)) if ltok else 0
        etok = float(etok.group(1)) if etok else 0
        print(
            f"  lam={lam:>2}: restore n={int(lc or 0):>6} avg={1000*(ls/lc):.2f}ms p99<={1000*hpc(lb,.99):.0f}ms "
            f"tokens={ltok/1e6:.0f}M | evict n={int(ec or 0):>6} avg={1000*(es/ec):.2f}ms p99<={1000*hpc(ev,.99):.0f}ms tokens={etok/1e6:.0f}M"
        )


def section_B(logs):
    print("=" * 78)
    print("(B) ACTUAL H->D DMA cost (CUDA-event elapsed_time, KL_DMA_TIMING=1)")
    print("=" * 78)
    last = {}
    for lg in logs:
        if not os.path.exists(lg):
            print(f"  (no log {lg})")
            continue
        txt = open(lg, errors="ignore").read()
        rows = re.findall(
            r"\[KLDMA\] load n=(\d+) actual_ms sum=([0-9.]+) avg=([0-9.]+) p50=([0-9.]+) p90=([0-9.]+) p99=([0-9.]+) max=([0-9.]+) \| tok=(\d+) tok_per_s=([0-9.]+)",
            txt,
        )
        if not rows:
            print(f"  {lg}: no [KLDMA] lines yet")
            continue
        n, s, avg, p50, p90, p99, mx, tok, tps = rows[-1]  # last = cumulative
        print(f"  {os.path.basename(lg)}: (cumulative, last of {len(rows)} windows)")
        print(
            f"     n={n} actual_ms avg={avg} p50={p50} p90={p90} p99={p99} max={mx} | tok={int(tok)/1e6:.0f}M restore_tok_per_s={float(tps):.0f}"
        )
        last = dict(p99=float(p99), avg=float(avg), tps=float(tps), mx=float(mx))
    return last


def section_C():
    print("=" * 78)
    print("(C) EXPOSED cost: TTFT vs prompt_len for NON-QUEUED cache-hit reqs (stock lam5)")
    print("    flat => even a 100K cached-context restore is not exposed on TTFT")
    print("=" * 78)
    files = sorted(glob.glob("runs/v6-shortlane-abl/perreq_reserve0_l5_r*.csv"))
    rows = []
    for f in files:
        for r in csv.DictReader(open(f)):
            try:
                rows.append(
                    (int(float(r["prompt_len"])), float(r["ttft_ms"]), int(float(r["output_len"])))
                )
            except Exception:
                pass
    if not rows:
        print("  (no perreq files)")
        return
    ttfts = sorted(x[1] for x in rows)
    pc = lambda a, q: a[min(len(a) - 1, int(q * len(a)))]
    med = pc(ttfts, 0.5)
    hits = [x for x in rows if x[1] <= med and x[2] <= 32]  # non-queued, later-turn cache hits
    bins = [(0, 1000), (1000, 5000), (5000, 15000), (15000, 40000), (40000, 100000)]
    print(f"  N={len(rows)} reqs; non-queued cache-hit subset N={len(hits)}")
    print(f"  {'prompt_len':>14} {'N':>6} {'ttft_p50':>9} {'ttft_p90':>9}")
    for lo, hi in bins:
        sub = sorted(x[1] for x in hits if lo <= x[0] < hi)
        if sub:
            print(f"  {f'{lo}-{hi}':>14} {len(sub):>6} {pc(sub,.5):>8.0f}ms {pc(sub,.9):>8.0f}ms")


def section_D(dma, run="runs/v0-stock"):
    print("=" * 78)
    print("(D) COST HIERARCHY: moving KV (restore) vs producing it (prefill recompute)")
    print("=" * 78)
    import json

    # RECOMPUTE rate = NEW (non-cached) KV tokens produced by prefill per second, aggregate.
    #   = (prompt_tokens_total - cached_tokens_total) / duration, from metrics + bench.
    for lam in ["5", "7"]:
        mf, bf = f"{run}/metrics_r{lam}.txt", f"{run}/bench_r{lam}.json"
        if not (os.path.exists(mf) and os.path.exists(bf)):
            continue
        m = open(mf, errors="ignore").read()
        prm = sum(
            float(x) for x in re.findall(r"(?m)^sglang:prompt_tokens_total\{[^}]*\}\s+([0-9.eE+]+)", m)
        )
        cac = sum(
            float(x) for x in re.findall(r"(?m)^sglang:cached_tokens_total\{[^}]*\}\s+([0-9.eE+]+)", m)
        )
        dur = json.load(open(bf)).get("duration")
        if not dur:
            continue
        recompute_tps = (prm - cac) / dur  # cumulative across run; approximate but order-correct
        print(
            f"  lam={lam}: prompt_toks={prm/1e6:.0f}M cached={cac/1e6:.0f}M new(recomputed)={ (prm-cac)/1e6:.0f}M "
            f"| recompute ~{recompute_tps:.0f} tok/s (cumulative/last-rate dur)"
        )
    restore_tps = dma.get("tps") if dma else None
    print(f"  restore (H->D DMA, measured) tok/s : {restore_tps}")
    print(
        "  NOTE: restore tok/s from (B); recompute tok/s above. ratio = how many x cheaper to MOVE than RECOMPUTE a KV token."
    )


if __name__ == "__main__":
    logs = sys.argv[1:] or sorted(glob.glob("runs/dma_timing_*.out"))
    section_A()
    dma = section_B(logs) if logs else {}
    section_C()
    section_D(dma)
