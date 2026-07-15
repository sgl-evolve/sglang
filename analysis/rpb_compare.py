#!/usr/bin/env python3
"""floyd: compare an RPB run against its same-node SRPF baseline on the screen (λ3,5).

Reads runs/<baseline>/curve.csv and runs/<rpb>/curve.csv and reports, per rate:
  - p99 TTFT baseline vs RPB + delta + SLO pass/fail   (the goodput-setting quantity)
  - req/s and tok/s                                     (THROUGHPUT-PRESERVATION check —
                                                          RPB's reserve is wasted iff no small
                                                          turn uses it; a tok/s drop => waste)
  - hit_rate                                            (LOSSLESS indicator: cache untouched,
                                                          must match; RPB only reslices prefill)
Verdict rule (screen): RPB is a WIN iff it lowers λ5 p99 meaningfully (toward/under SLO) with
throughput preserved (tok/s within ~3%) and hit_rate matched (within ~0.01). Otherwise NEUTRAL/NEG.

Usage: python3 analysis/rpb_compare.py [baseline_ver] [rpb_ver]   (defaults v-srpf-ctl4 v-rpb25)
"""
import csv, os, sys

def read_curve(ver):
    p = f"runs/{ver}/curve.csv"
    if not os.path.exists(p): return None
    out = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            try:
                out[float(row["rate"])] = {
                    "req": float(row["req_throughput"]), "tok": float(row["out_tok_s"]),
                    "p50": float(row["ttft_p50_ms"]), "p99": float(row["ttft_p99_ms"]),
                    "hit": float(row.get("hit_rate", "nan") or "nan"),
                }
            except (KeyError, ValueError):
                continue
    return out

def main():
    base_v = sys.argv[1] if len(sys.argv) > 1 else "v-srpf-ctl4"
    rpb_v = sys.argv[2] if len(sys.argv) > 2 else "v-rpb25"
    SLO = 8000.0
    b, r = read_curve(base_v), read_curve(rpb_v)
    if not b: print(f"baseline {base_v}: no curve.csv yet"); return
    if not r: print(f"rpb {rpb_v}: no curve.csv yet"); return
    print(f"{'rate':>4} | {'base p99':>9} {'rpb p99':>9} {'Δp99':>8} {'SLO':>10} | "
          f"{'base tok/s':>10} {'rpb tok/s':>10} {'Δtok%':>7} | {'base hit':>8} {'rpb hit':>8} {'Δhit':>7}")
    win5 = None
    for rate in sorted(set(b) & set(r)):
        bb, rr = b[rate], r[rate]
        dp99 = rr["p99"] - bb["p99"]
        dtok = 100.0 * (rr["tok"] - bb["tok"]) / bb["tok"] if bb["tok"] else 0.0
        dhit = rr["hit"] - bb["hit"]
        slo = f"{'PASS' if bb['p99']<=SLO else 'fail'}->{'PASS' if rr['p99']<=SLO else 'fail'}"
        print(f"{rate:>4.0f} | {bb['p99']:>9.0f} {rr['p99']:>9.0f} {dp99:>+8.0f} {slo:>10} | "
              f"{bb['tok']:>10.1f} {rr['tok']:>10.1f} {dtok:>+6.1f}% | {bb['hit']:>8.3f} {rr['hit']:>8.3f} {dhit:>+7.3f}")
        if rate == 5.0:
            win5 = (dp99, dtok, dhit, bb["p99"], rr["p99"])
    print()
    if win5 is not None:
        dp99, dtok, dhit, bp, rp = win5
        lossless = abs(dhit) <= 0.01
        tput_ok = dtok >= -3.0
        improved = dp99 <= -300.0
        robust = rp <= SLO and bp > SLO  # turned a fail into a pass, or already passing tighter
        verdict = ("WIN" if (improved and tput_ok and lossless) else
                   "NEUTRAL/NEG" if lossless else "LOSSLESS-VIOLATION?")
        print(f"λ5 verdict: {verdict}  (Δp99={dp99:+.0f}ms, Δtok={dtok:+.1f}%, Δhit={dhit:+.3f}; "
              f"lossless={lossless}, tput_ok={tput_ok}, improved={improved})")
        print("  NOTE: throughput drop => RPB reserve wasted (all-big waiting queue) => refine to adaptive reserve.")
        print("  NOTE: hit mismatch >0.01 => investigate (RPB must be lossless; cache is untouched by design).")

if __name__ == "__main__":
    main()
