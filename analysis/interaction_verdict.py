#!/usr/bin/env python3
"""Adjudicate the SRPF x write_back interaction (job 20306, v-srpfwb-1) against the
pre-registered outcomes (see report.md "PRE-REGISTRATION" entry, commit 0306ba3d4).

Reads curve.csv (label,rate,req_throughput,out_tok_s,ttft_p50_ms,ttft_p99_ms,e2e_p99_ms,hit_rate)
for each run, computes goodput@SLO = max sustained rate with p99 TTFT <= SLO_MS, and prints a
comparison table + the pre-registered verdict. No args: uses the canonical three runs.

Usage:  .venv/bin/python analysis/interaction_verdict.py [SLO_MS]   (default SLO_MS=8000)
"""
import csv, os, sys

RUNS = os.path.join(os.path.dirname(__file__), os.pardir, "runs")
SLO_MS = float(sys.argv[1]) if len(sys.argv) > 1 else 8000.0

# (label, dir, description)
CANON = [
    ("SRPF-alone",        "v-srpf-full",   "--schedule-policy srpf (node 1-2, cross-node ref)"),
    ("FCFS+write_back",   "v-writeback",   "--hicache-write-policy write_back"),
    ("SRPF+write_back",   "v-srpfwb-1",    "--schedule-policy srpf --hicache-write-policy write_back (ondem-2)"),
    ("SRPF-alone@ondem2", "v-srpf-ondem2", "--schedule-policy srpf (ondem-2 SAME-NODE control for 20306)"),
]


def load_curve(d):
    p = os.path.join(RUNS, d, "curve.csv")
    if not os.path.exists(p):
        return None
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                rows.append((float(r["rate"]),
                             float(r["ttft_p99_ms"]) if r.get("ttft_p99_ms") else float("nan"),
                             float(r["req_throughput"]) if r.get("req_throughput") else float("nan"),
                             float(r["hit_rate"]) if r.get("hit_rate") else float("nan")))
            except (ValueError, KeyError):
                pass
    return sorted(rows)


def goodput(rows):
    """max req_throughput over rate points whose p99 <= SLO (contiguous from low rate)."""
    g = 0.0
    last_rate = None
    for rate, p99, thpt, hit in rows:
        if p99 == p99 and p99 <= SLO_MS:   # not NaN and within SLO
            g = max(g, thpt)
            last_rate = rate
        else:
            break  # SLO tail broken; goodput is capped at the last passing rate
    return g, last_rate


def main():
    print(f"=== SRPF x write_back interaction verdict (SLO p99 TTFT <= {SLO_MS:.0f} ms) ===\n")
    results = {}
    for label, d, desc in CANON:
        rows = load_curve(d)
        if rows is None:
            print(f"[{label:16s}] runs/{d}/curve.csv  NOT PRESENT YET")
            results[label] = None
            continue
        g, lr = goodput(rows)
        results[label] = (g, lr, rows)
        print(f"[{label:16s}] {desc}")
        print(f"  {'rate':>5} {'p99_ms':>9} {'thpt':>6} {'hit':>6}  {'SLO':>4}")
        for rate, p99, thpt, hit in rows:
            ok = "PASS" if (p99 == p99 and p99 <= SLO_MS) else "FAIL"
            print(f"  {rate:>5.0f} {p99:>9.0f} {thpt:>6.2f} {hit:>6.3f}  {ok:>4}")
        print(f"  -> goodput@SLO = {g:.2f} req/s (last passing rate lambda={lr})\n")

    srpf = results.get("SRPF-alone")
    test = results.get("SRPF+write_back")
    EXPECTED_RATES = 4  # frozen sweep lambda in {3,5,7,10}
    if not test:
        print("VERDICT: v-srpfwb-1 not landed yet — curve.csv has no data rows (sweep in progress).")
        print("  (curve.csv is created at launch with only a header; rows append as each rate completes.)")
        return
    n_rows = len(test[2])
    if n_rows < EXPECTED_RATES:
        print(f"VERDICT: v-srpfwb-1 INCOMPLETE — {n_rows}/{EXPECTED_RATES} rate rows so far (sweep still running).")
        print("  Do NOT adjudicate on a partial curve (goodput needs the full sweep incl. the failing rate).")
        print("  Re-run this tool when curve.csv has all 4 rows or summary.json exists.")
        return
    # Prefer the SAME-NODE control (confound-free) when it has completed the full sweep.
    samenode = results.get("SRPF-alone@ondem2")
    if samenode and len(samenode[2]) >= EXPECTED_RATES:
        baseline_label, baseline = "SRPF-alone@ondem2 (SAME-NODE, confound-free)", samenode
        confound = "SAME-NODE"
    else:
        baseline_label, baseline = "SRPF-alone (v-srpf-full, CROSS-NODE — within +/-14% node noise)", srpf
        confound = "CROSS-NODE"
        if samenode:
            print(f"  [same-node control v-srpf-ondem2 present but incomplete "
                  f"({len(samenode[2])}/{EXPECTED_RATES} rates) — using cross-node ref for now]\n")
    g_srpf = baseline[0] if baseline else float("nan")
    g_test = test[0]
    print("=== PRE-REGISTERED ADJUDICATION ===")
    print(f"  baseline: {baseline_label}")
    print(f"  SRPF-alone goodput   = {g_srpf:.2f}")
    print(f"  SRPF+wb   goodput    = {g_test:.2f}   ({confound} comparison)")
    delta = g_test - g_srpf
    if abs(delta) < 0.24:  # < half a rate-grid step (grid ~3,5,7,10) -> same goodput bucket
        print("  OUTCOME = PRIMARY (no compound): goodput ~ SRPF-alone.")
        print("  => Ceiling non-lever for goodput CONFIRMED even with scheduling active + ceiling raised.")
        print("     write_back's E[W] saving is on reused docs, off the first-sight SLO tail. Sharpens flagship.")
    elif delta > 0:
        print(f"  OUTCOME = FALSIFIER (COMPOUND WIN): +{delta:.2f} req/s goodput over SRPF-alone.")
        print("  => write_back's freed compute lets SRPF clear a higher-rate tail. LOSSLESS, both mechanisms mine.")
        print("     ACTION: honestly revise flagship (ceiling CAN convert to goodput under scheduling);")
        print("     run a SAME-NODE paired replicate before claiming.")
    else:
        print(f"  OUTCOME = THIRD (interference): {delta:.2f} req/s goodput vs SRPF-alone.")
        print("  => write_back backup dynamics degrade SRPF's tail. Bounded negative; characterize.")


if __name__ == "__main__":
    main()
