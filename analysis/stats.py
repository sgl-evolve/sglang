#!/usr/bin/env python3
"""floyd: aggregate replicate runs and compare arms with non-parametric stats.

goodput@SLO is variance-dominated (a metastable coin-flip), so single-run A/Bs
are void. This compares DISTRIBUTIONS across replicates:
  - per-rate p99 TTFT values (the sensitive continuous signal)
  - goodput@SLO (coarse; report median + pass-fraction)
  - Levene (variance equality) + Mann-Whitney U (location) where n allows.

Usage:
  python analysis/stats.py --arm stock v0-stock v0-stock-r2 ... \
                           --arm cca   v1-cca  v1-cca-r2 ...
"""
import argparse, csv, json, os, statistics as st

RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
SLO = 8000.0
RATES = ["3", "5", "7", "10"]

def load_curve(v):
    cp = os.path.join(RUNS, v, "curve.csv")
    if not os.path.exists(cp):
        return None
    return [r for r in csv.DictReader(open(cp)) if r.get("req_throughput")]

def fnum(r, k):
    try:
        return float(r.get(k))
    except (TypeError, ValueError):
        return None

def goodput(rows):
    p = [fnum(r, "req_throughput") for r in rows
         if fnum(r, "ttft_p99_ms") is not None and fnum(r, "ttft_p99_ms") <= SLO
         and fnum(r, "req_throughput") is not None]
    return max(p) if p else 0.0

def p99_at(rows, R):
    row = next((r for r in rows if r.get("rate") == R), None)
    return fnum(row, "ttft_p99_ms") if row else None

def summarize(vs):
    rows_list = [load_curve(v) for v in vs]
    rows_list = [r for r in rows_list if r]
    gps = [goodput(r) for r in rows_list]
    out = {"n": len(rows_list), "versions": vs, "goodput": gps}
    out["p99"] = {R: [p99_at(r, R) for r in rows_list if p99_at(r, R) is not None]
                  for R in RATES}
    return out

def med(x):
    return st.median(x) if x else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", action="append", metavar=("LABEL","VER"),
                    help="--arm <label> <ver1> <ver2> ...")
    a = ap.parse_args()
    if not a.arm:
        ap.error("need at least one --arm; e.g. "
                 "--arm stock v0-stock v0-stock-r2 --arm srpf v-srpf-r1 v-srpf-r2 v-srpf-r3")
    arms = {x[0]: summarize(x[1:]) for x in a.arm}
    for label, s in arms.items():
        gps = s["goodput"]
        passct = sum(1 for g in gps if g >= 3)
        print(f"\n=== arm '{label}'  n={s['n']}  versions={s['versions']}")
        print(f"  goodput@SLO: values={gps} median={med(gps):.2f} "
              f"pass(>=3)={passct}/{len(gps)}")
        for R in RATES:
            vals = s["p99"][R]
            if vals:
                print(f"  p99@lambda={R}: median={med(vals):.0f}ms "
                      f"vals={[round(v) for v in vals]}")
    # pairwise stats if exactly 2 arms and scipy present
    if len(arms) == 2:
        try:
            from scipy import stats as sps
            (la, sa), (lb, sb) = list(arms.items())
            print(f"\n=== {la} vs {lb} (per-rate p99) ===")
            for R in RATES:
                xa, xb = sa["p99"][R], sb["p99"][R]
                if len(xa) >= 2 and len(xb) >= 2:
                    lev = sps.levene(xa, xb).pvalue
                    mwu = sps.mannwhitneyu(xa, xb, alternative="two-sided").pvalue
                    print(f"  lambda={R}: Levene p={lev:.4f}  MWU p={mwu:.4f}  "
                          f"med {med(xa):.0f} vs {med(xb):.0f}")
            ga, gb = sa["goodput"], sb["goodput"]
            if len(ga) >= 2 and len(gb) >= 2:
                print(f"  goodput MWU p={sps.mannwhitneyu(ga, gb, alternative='two-sided').pvalue:.4f}")
        except ImportError:
            print("\n(scipy not installed; install for Levene/MWU)")

if __name__ == "__main__":
    main()
