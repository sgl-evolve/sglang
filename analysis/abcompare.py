#!/usr/bin/env python3
"""Quick A/B comparison across turing eval versions. Reads runs/<ver>/curve.csv +
summary.json and prints the goodput@SLO headline + per-rate curve + hit rates.
Usage: python3 analysis/abcompare.py v1_stock v2_flat2 v3_size4k
"""
import sys, json, csv, os
BASE = os.path.join(os.path.dirname(__file__), "..", "runs")

def load(ver):
    d = os.path.join(BASE, ver)
    sm = os.path.join(d, "summary.json")
    cv = os.path.join(d, "curve.csv")
    out = {"ver": ver, "rows": [], "panel": {}}
    if os.path.exists(sm):
        out["panel"] = json.load(open(sm)).get("panel", {})
    if os.path.exists(cv):
        out["rows"] = list(csv.DictReader(open(cv)))
    return out

def f(r, k):
    v = r.get(k)
    try: return float(v)
    except: return None

def main():
    vers = sys.argv[1:] or ["v1_stock"]
    data = [load(v) for v in vers]
    print(f"\n{'='*78}\nGOODPUT@SLO + peak (from summary.json)\n{'='*78}")
    print(f"{'version':<16}{'goodput@SLO':>12}{'peak_tok/s':>12}{'peak_req/s':>12}{'best_hit':>10}")
    for d in data:
        p = d["panel"]
        print(f"{d['ver']:<16}{p.get('overall/goodput_reqs_at_SLO','-'):>12}"
              f"{p.get('overall/peak_out_tok_s','-'):>12}{p.get('overall/peak_req_s','-'):>12}"
              f"{p.get('overall/hit_rate','-'):>10}")
    # per-rate curve table
    print(f"\n{'='*78}\nPER-RATE CURVE (req/s | ttft_p99_ms | hit_rate)\n{'='*78}")
    rates = ["3","5","7","10"]
    for metric,label in [("req_throughput","req/s"),("ttft_p99_ms","ttft_p99"),("hit_rate","hit")]:
        print(f"\n-- {label} --")
        print(f"{'version':<16}" + "".join(f"{'λ='+r:>12}" for r in rates))
        for d in data:
            byr = {row["rate"]: row for row in d["rows"]}
            cells = ""
            for r in rates:
                v = f(byr[r], metric) if r in byr else None
                cells += f"{(('%.3f'%v) if v is not None else '-'):>12}"
            print(f"{d['ver']:<16}{cells}")
    print()

if __name__ == "__main__":
    main()
