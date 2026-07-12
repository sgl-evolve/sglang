#!/usr/bin/env python3
"""Calibrate prefill rate P from a real run dir and overlay measured vs P-K prediction.
Reads runs/<ver>/{curve.csv, bench_r<R>.json}. P_achieved(R) = total_input*(1-hit)/duration;
P := max over rates (saturation estimate). Then prints measured p99 vs perfect-cache prediction.
Usage: python3 tools/calibrate.py runs/<ver>"""
import sys, os, csv, json

def load_bench(path):
    """bench_serving appends JSONL; take the last record."""
    if not os.path.exists(path): return None
    recs=[json.loads(l) for l in open(path) if l.strip()]
    return recs[-1] if recs else None

def main():
    d=sys.argv[1]
    rows=list(csv.DictReader(open(os.path.join(d,"curve.csv"))))
    f=lambda r,k:(float(r[k]) if r.get(k) not in (None,"") else None)
    print(f"{'rate':>4} {'req/s':>7} {'hit':>5} {'in_tok':>12} {'dur_s':>8} {'P_ach(tok/s)':>13} {'p99_meas_s':>11}")
    Pests=[]
    for r in rows:
        R=r["rate"]; b=load_bench(os.path.join(d,f"bench_r{R}.json"))
        hit=f(r,"hit_rate") or 0.0; p99=(f(r,"ttft_p99_ms") or 0)/1000
        intok=b.get("total_input_tokens") if b else None
        dur=b.get("duration") if b else None
        Pach=(intok*(1-hit)/dur) if (intok and dur) else None
        if Pach: Pests.append(Pach)
        print(f"{R:>4} {f(r,'req_throughput') or 0:>7.2f} {hit:>5.2f} {str(intok or '-'):>12} "
              f"{(f'{dur:.0f}' if dur else '-'):>8} {(f'{Pach:.0f}' if Pach else '-'):>13} {p99:>11.2f}")
    if not Pests:
        print("\n(no bench json durations — cannot calibrate P; using curve only)"); 
    P = max(Pests) if Pests else None
    print(f"\nCalibrated P (peak achieved prefill tok/s) = {P:.0f}" if P else "\nP: unknown")
    # overlay: perfect-cache prediction at calibrated P
    if P:
        try:
            import importlib.util
            spec=importlib.util.spec_from_file_location("qm","tools/queue_model.py")
        except Exception: pass
        # inline P-K perfect-cache prediction using trace stats
        S=json.load(open("tools/trace_stats.json"))
        doc=S["doc_toks"]; nt=S["nturns"]; Qm,Am=S["q_p50"],S["a_p50"]
        xs=[]
        for dtok,n in zip(doc,nt):
            xs.append(dtok+Qm)
            for t in range(1,n): xs.append(Qm)
        n=len(xs); m1=sum(xs)/n; m2=sum(x*x for x in xs)/n; xs.sort(); xp99=xs[int(0.99*n)]
        print(f"\nPerfect-cache P-K prediction at P={P:.0f}:  (measured p99 in parens)")
        meas={r['rate']:(f(r,'ttft_p99_ms') or 0)/1000 for r in rows}
        for lam in [3,5,7,10]:
            rho=lam*m1/P
            if rho>=1: print(f"  lam={lam}: UNSTABLE (rho={rho:.2f})  [meas {meas.get(str(lam),'-')}]"); continue
            Wq=lam*(m2/P**2)/(2*(1-rho)); pred=Wq+xp99/P
            print(f"  lam={lam}: pred p99={pred:.2f}s (rho={rho:.2f})  [meas {meas.get(str(lam),0):.2f}s]")
        print("\nGAP: measured p99 >> perfect prediction ⇒ real cache imperfection (recompute/load-back/"
              "cold-ramp) OR model too optimistic. measured ≈ perfect ⇒ tail is fundamental (impossibility).")
if __name__=="__main__": main()
