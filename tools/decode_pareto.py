#!/usr/bin/env python3
"""kleinrock — Direction 4 (the hidden decode tail): prefill-decode Pareto analysis.

Compares two arms across the rate sweep on BOTH axes:
  - TTFT tail / goodput@SLO (the headline metric) : p99 TTFT, pass = <=8000ms
  - decode tail (the hidden metric)               : p99 ITL, p99 TPOT, p99 E2E
so we can see whether a decode-friendly setting (e.g. --enable-mixed-chunk) reduces the decode tail and
at what TTFT-goodput cost = the prefill-decode Pareto.

Usage:
  python3 tools/decode_pareto.py                      # default: stock (v0-stock) vs mixedchunk (v14-mixedchunk)
  python3 tools/decode_pareto.py A=<dir> B=<dir>      # custom arm dirs (per-rate bench_r<lam>.json)
Reads runs/<dir>/bench_r<lam>.json. GPU-free (parses artifacts). SLO=8000ms.
"""
import sys, json, os, glob

SLO = 8000.0


def load_arm(d, multi=False):
    """Return {lam: [rows]} where each row is the metric dict. multi=True globs r<lam>_r* replicates."""
    out = {}
    for lam in [3, 5, 7, 10]:
        rows = []
        cands = (
            sorted(glob.glob(f"runs/{d}/bench_*l{lam}_r*.json"))
            if multi
            else [f"runs/{d}/bench_r{lam}.json"]
        )
        for f in cands:
            if os.path.exists(f):
                try:
                    rows.append(json.load(open(f)))
                except Exception:
                    pass
        if rows:
            out[lam] = rows
    return out


def med(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def summarize(name, arm):
    print(f"\n=== {name} ===")
    print(
        f"{'lam':>4} {'ttft_p99':>9} {'goodput':>8} {'itl_p99':>9} {'tpot_p99':>9} {'e2e_p99':>9} {'req/s':>7} {'reps':>4}"
    )
    res = {}
    for lam in sorted(arm):
        rows = arm[lam]
        ttft = med([r.get("p99_ttft_ms") for r in rows])
        itl = med([r.get("p99_itl_ms") for r in rows])
        tpot = med([r.get("p99_tpot_ms") for r in rows])
        e2e = med([r.get("p99_e2e_latency_ms") for r in rows])
        req = med([r.get("request_throughput") for r in rows])
        gp = "PASS" if (ttft is not None and ttft <= SLO) else "fail"
        res[lam] = dict(ttft=ttft, itl=itl, tpot=tpot, e2e=e2e, req=req, pass_=(ttft or 9e9) <= SLO)
        f = lambda x, s=0: (f"{x:>{9 if not s else s}.0f}" if x is not None else f"{'—':>{9 if not s else s}}")
        print(f"{lam:>4} {f(ttft)} {gp:>8} {f(itl)} {f(tpot)} {f(e2e)} {f(req,7)} {len(rows):>4}")
    return res


if __name__ == "__main__":
    A_dir, B_dir = "v0-stock", "v14-mixedchunk"
    for a in sys.argv[1:]:
        if a.startswith("A="):
            A_dir = a[2:]
        elif a.startswith("B="):
            B_dir = a[2:]
    A = load_arm(A_dir, multi=("l5_r" in " ".join(glob.glob(f"runs/{A_dir}/bench_*.json"))))
    B = load_arm(B_dir, multi=("l5_r" in " ".join(glob.glob(f"runs/{B_dir}/bench_*.json"))))
    rA = summarize(f"A: {A_dir}", A)
    rB = summarize(f"B: {B_dir}", B)
    print("\n=== PARETO (B vs A): decode-tail change vs TTFT-goodput cost ===")
    print(f"{'lam':>4} {'d_ttft_p99':>11} {'d_itl_p99':>10} {'d_tpot_p99':>11} {'goodput A→B':>14}")
    for lam in sorted(set(rA) & set(rB)):
        a, b = rA[lam], rB[lam]
        d = lambda k: (f"{b[k]-a[k]:+.0f}" if (a[k] is not None and b[k] is not None) else "—")
        gp = f"{'PASS' if a['pass_'] else 'fail'}→{'PASS' if b['pass_'] else 'fail'}"
        print(f"{lam:>4} {d('ttft'):>11} {d('itl'):>10} {d('tpot'):>11} {gp:>14}")
    print("\nINTERPRET: B reduces itl_p99/tpot_p99 (decode tail) => decode-friendly helps; watch d_ttft_p99 +"
          " goodput flips => the TTFT cost. Large decode-cut + goodput-preserved = free win (config, honest);"
          " decode-cut + goodput-LOST = real prefill-decode Pareto tension (the strong thesis).")
