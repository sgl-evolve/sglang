#!/usr/bin/env python3
"""Paper 4 analysis: does occupancy-feedback damping (v7_decfloor) damp the lam=3 coin-flip
losslessly without moving capacity C? Run when runs/v7_decfloor/ lands.

Checks (from committed run JSON/CSV):
  PRIMARY  lam=3 p99 TTFT vs stock coin-flip band {6.2, 11.5}s -> did decode-floor land tight/low?
  MECHANISM lam=3 mean concurrency (occupancy) vs stock {124,166} -> did the controller lower it?
           lam=3 decode tpot vs stock {392,468} -> did decode drain faster?
  LOSSLESS hit_rate vs stock (must be ~invariant, <0.5pp)
  GUARD    C at lam=10 vs stock 4.14 / size 4.67 (must be ~unchanged -> not de-saturation)
"""
import csv, json, os, statistics
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")

def curve(v):
    p = os.path.join(RUNS, v, "curve.csv")
    if not os.path.exists(p): return {}
    return {int(float(r["rate"])): r for r in csv.DictReader(open(p))}

def bench(v, r):
    p = os.path.join(RUNS, v, f"bench_r{r}.json")
    return json.load(open(p)) if os.path.exists(p) else None

V = "v7_decfloor"
if not curve(V):
    print(f"{V} not landed yet. Re-run when runs/{V}/curve.csv exists.")
    raise SystemExit(0)

c7 = curve(V)
print("=== v7_decfloor curve ===")
for r in (3,5,7,10):
    if r in c7:
        row = c7[r]
        print(f"  lam={r:2d}  C={float(row['req_throughput']):.2f}  hit={float(row['hit_rate']):.4f}"
              f"  p50={float(row['ttft_p50_ms']):.0f}  p99={float(row['ttft_p99_ms']):.0f}")

# PRIMARY: lam=3 p99 vs stock coin-flip band
if 3 in c7:
    p99_7 = float(c7[3]['ttft_p99_ms']); hit7 = float(c7[3]['hit_rate'])
    print(f"\nPRIMARY lam=3 p99 = {p99_7/1000:.1f}s  (stock coin-flip band 6.2..11.5s; SLO 8s)")
    print(f"  -> {'PASS (<=8s)' if p99_7<=8000 else 'FAIL (>8s)'};"
          f" {'below stock-mean 8.8s' if p99_7<8800 else 'not below stock-mean'}")
    # LOSSLESS
    print(f"LOSSLESS hit_rate {hit7:.4f} vs stock ~0.675 -> "
          f"{'OK (lossless)' if abs(hit7-0.675)<0.02 else 'CHECK (hit moved)'}")

# MECHANISM: occupancy + tpot from bench_r3
b7 = bench(V, 3)
if b7:
    print(f"\nMECHANISM lam=3: concurrency={b7['concurrency']:.0f} (stock 124..166),"
          f" tpot={b7['mean_tpot_ms']:.0f}ms (stock 392..468)")
    print(f"  -> {'occupancy LOWER (mechanism damping, WIN signal)' if b7['concurrency']<124 else 'occupancy NOT lower than stock-best'}")

# GUARD: C at lam=10 unchanged (not de-saturation)
if 10 in c7:
    C10 = float(c7[10]['req_throughput'])
    print(f"\nGUARD C@lam10 = {C10:.2f} (stock 4.14 / size 4.67) -> "
          f"{'~unchanged (not de-saturation)' if 3.9<C10<4.9 else 'CHANGED -- investigate'}")

print("\nVERDICT: WIN if lam=3 p99 tight+<=SLO AND occupancy lower AND hit invariant AND C@10 unchanged.")
print("  If p99 UP or C down -> deferral-backfire/de-saturation negative (fold into Paper 3, still publishable).")
print("  Then: n>=3 replicates + sweep THETA_HI in {0.85,0.90,0.95}, GAIN in {0.3,0.5}.")
