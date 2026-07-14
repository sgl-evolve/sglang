#!/usr/bin/env python3
"""floyd: compare rate-sweep runs (goodput@SLO + curve) side by side.

Usage: python analysis/compare.py <version1> <version2> ...
Reads runs/<v>/summary.json and runs/<v>/curve.csv.
"""
import csv, json, os, sys

RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
SLO = 8000.0

def load(v):
    d = os.path.join(RUNS, v)
    rows = []
    cp = os.path.join(d, "curve.csv")
    if os.path.exists(cp):
        rows = [r for r in csv.DictReader(open(cp)) if r.get("req_throughput")]
    summ = {}
    sp = os.path.join(d, "summary.json")
    if os.path.exists(sp):
        summ = json.load(open(sp))
    return rows, summ

def fnum(r, k):
    v = r.get(k)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def goodput(rows):
    passing = [fnum(r, "req_throughput") for r in rows
               if fnum(r, "ttft_p99_ms") is not None and fnum(r, "ttft_p99_ms") <= SLO
               and fnum(r, "req_throughput") is not None]
    return max(passing) if passing else 0.0

def main():
    versions = sys.argv[1:] or ["v0-stock", "v1-cca"]
    print(f"{'metric':<22}" + "".join(f"{v:>16}" for v in versions))
    data = {v: load(v) for v in versions}
    # headline
    gp = {v: goodput(data[v][0]) for v in versions}
    print(f"{'goodput@8s (req/s)':<22}" + "".join(f"{gp[v]:>16.3f}" for v in versions))
    for metric, key in [("peak req/s", "req_throughput"), ("peak tok/s", "out_tok_s"),
                        ("best hit_rate", "hit_rate")]:
        line = f"{metric:<22}"
        for v in versions:
            rows = data[v][0]
            vals = [fnum(r, key) for r in rows if fnum(r, key) is not None]
            line += f"{(max(vals) if vals else 0):>16.3f}"
        print(line)
    # per-rate p99 TTFT and req/s
    print("\n--- per-rate p99 TTFT (ms) [PASS if <=8000] ---")
    rates = ["3", "5", "7", "10"]
    for R in rates:
        line = f"  lambda={R:<14}"
        for v in versions:
            rows = data[v][0]
            row = next((r for r in rows if r.get("rate") == R), None)
            p99 = fnum(row, "ttft_p99_ms") if row else None
            tag = "" if p99 is None else ("PASS" if p99 <= SLO else "FAIL")
            line += f"{(f'{p99:.0f} {tag}' if p99 is not None else '-'):>16}"
        print(line)
    print("\n--- per-rate req/s achieved ---")
    for R in rates:
        line = f"  lambda={R:<14}"
        for v in versions:
            rows = data[v][0]
            row = next((r for r in rows if r.get("rate") == R), None)
            rq = fnum(row, "req_throughput") if row else None
            line += f"{(f'{rq:.3f}' if rq is not None else '-'):>16}"
        print(line)

if __name__ == "__main__":
    main()
