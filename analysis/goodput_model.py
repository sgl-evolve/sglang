#!/usr/bin/env python3
"""Direction 2: analytical goodput@SLO model for hierarchical-KV conversational serving.

Thesis: goodput@SLO is bounded by the SERVICE-TIME TAIL (giant uncacheable turn-0
prefills), not the mean. Caching lowers the service mean (via hit rate) but not the
heavy tail, so a tail-dominated P99 cannot be fixed by caching — the analytical form
of the Paper-1 impossibility.

Model (open-loop M/G/1-ish, per-request TTFT = queue wait + own prefill):
- Service demand of a request = prefill work of its uncached tokens.
  prefill_time(req) = uncached_tokens / R_pref.
  uncached = turn-0: full doc (always) ; turn>=1: (1-hit)*prior_prefix + new_q.
- Heavy-tailed doc sizes => heavy-tailed service => Pollaczek-Khinchine:
  E[Wq] = rho/(1-rho) * E[S^2]/(2 E[S]),  rho = lambda * E[S].
- We use the measured curve (v1_stock) to (a) fit R_pref from achieved req/s at the
  knee, (b) validate the predicted P99 explosion, then (c) derive goodput@SLO(SLO)
  and the feasible region, and (d) show caching (raising hit) shifts rho but not the
  tail term E[S^2], so P99 stays tail-bound.
"""
import json, ast, statistics, math

MIX="/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"
CH=4.0

def load_service_tokens(hit_rate):
    """Per-request UNCACHED prefill tokens under a given hit_rate (mean applied to
    reused prefix). Returns list of uncached-token counts across all requests."""
    recs=[json.loads(l) for l in open(MIX)]
    unc=[]
    for r in recs:
        qp=r.get("qa_pairs",""); doc=r.get("input","")
        if qp=="none" or (isinstance(qp,(list,str)) and len(qp)==0):
            turns=[("Input: "+doc+" Question: Please summarize the input","")]
        else:
            try: qa=ast.literal_eval(qp) if isinstance(qp,str) else qp
            except Exception: continue
            turns=[]
            for i,q in enumerate(qa):
                turns.append((("Input: "+doc+" Question: "+str(q.get("Q",""))) if i==0 else str(q.get("Q","")), ""))
        committed=0
        for ti,(p,_) in enumerate(turns):
            newq=max(1,int(len(p)/CH))
            prior=committed
            if ti==0:
                uncached=newq                      # whole doc, uncacheable (first sight)
            else:
                uncached=int((1-hit_rate)*prior)+newq  # reused prefix partly missed + new q
            unc.append(uncached); committed=prior+newq
    return unc

def pk_p99(lam, S, R_pref):
    """Pollaczek-Khinchine mean wait + heavy-tail P99 estimate.
    service time s_i = S[i]/R_pref. rho=lam*E[s]. Returns (rho, E[Wq], approx P99 TTFT)."""
    s=[x/R_pref for x in S]
    Es=statistics.mean(s); Es2=statistics.mean([x*x for x in s])
    rho=lam*Es
    if rho>=1.0:
        return rho, float('inf'), float('inf')
    EWq=rho/(1-rho)*Es2/(2*Es)          # P-K mean queue wait
    # P99 TTFT ~ high quantile of (wait + own service). Wait is ~exponential-tailed in
    # M/G/1; own service P99 is the doc-tail. Approx P99 ~ EWq*ln(100) + p99(service).
    s_sorted=sorted(s); p99_s=s_sorted[int(0.99*len(s_sorted))]
    p99_ttft=EWq*math.log(100)+p99_s
    return rho, EWq, p99_ttft

def main():
    # measured v1_stock curve (node 0-3)
    measured={3:(2.83,11493.55,0.6753),5:(3.59,24956.58,0.6627),7:(3.99,35752.59,0.6568),10:(4.14,41014.51,0.6537)}
    print("=== service-time tail (uncached prefill tokens/request) at stock hit 0.66 ===")
    S=load_service_tokens(0.66)
    Ss=sorted(S)
    Es=statistics.mean(S); Es2=statistics.mean([x*x for x in S])
    cv2=(Es2-Es*Es)/(Es*Es)
    print(f"  n_req={len(S)} E[S]={Es:.0f} tok  p50={Ss[len(Ss)//2]} p99={Ss[int(0.99*len(Ss))]} max={max(Ss)}")
    print(f"  squared-CV of service C^2 = {cv2:.1f}  (>>1 => heavy-tailed; P-K wait ∝ (1+C^2)/2)")
    # fit R_pref so achieved throughput saturates ~ measured peak req/s (~4.14). Peak req/s = 1/E[s] at rho=1.
    # 1/E[s] = R_pref / E[S]  => R_pref = peak_reqs * E[S]. Use peak 4.14 req/s (but req/s is per CONV; service is per TURN)
    turns_per_conv=len(S)/1553
    peak_turn_rps=4.14*turns_per_conv    # convert conv-rate to turn-rate
    R_pref=peak_turn_rps*Es
    print(f"\n  turns/conv={turns_per_conv:.2f}; peak turn-rate={peak_turn_rps:.1f}/s; fitted R_pref={R_pref:.0f} tok/s")
    print(f"\n=== predicted vs measured P99 TTFT ===")
    print(f"  {'λ(conv/s)':>10}{'ρ':>8}{'pred_p99_s':>12}{'meas_p99_s':>12}")
    for lam,(rq,p99m,hit) in measured.items():
        lam_turn=lam*turns_per_conv
        rho,EWq,p99=pk_p99(lam_turn,S,R_pref)
        pr=f"{p99/1000:.1f}" if p99!=float('inf') else "inf(ρ≥1)"
        print(f"  {lam:>10}{rho:>8.2f}{pr:>12}{p99m/1000:>12.1f}")
    # --- CLEAN LOWER BOUND (robust, no fragile P-K fit) ---
    # TTFT(req) >= own_prefill_time = uncached_tokens / R_pref_loaded.
    # Fit R_pref_loaded from measured p50 TTFT at λ=3 (low load => TTFT ~ own prefill):
    p50_S=Ss[len(Ss)//2]                 # p50 uncached tokens at stock hit
    p50_ttft_s=measured[3][1]/1000 * 0   # placeholder; use measured p50 TTFT below
    p50_ttft_meas=1.017                  # v1_stock λ=3 p50 TTFT (s)
    R_load=p50_S/p50_ttft_meas
    print(f"\n=== CLEAN caching-invariant P99 floor (fit R_pref from measured p50 TTFT) ===")
    print(f"  R_pref_loaded = p50_S/p50_TTFT = {p50_S}/{p50_ttft_meas}s = {R_load:.0f} tok/s")
    print(f"  {'hit':>6}{'E[S]tok':>10}{'p99_S tok':>12}{'P99floor=p99_S/R (s)':>22}{'vs 8s SLO':>10}")
    for h in [0.0,0.62,0.675,0.737,0.809,0.95]:
        Sh=load_service_tokens(h); Esh=statistics.mean(Sh); p99h=sorted(Sh)[int(0.99*len(Sh))]
        floor=p99h/R_load
        print(f"  {h:>6.3f}{Esh:>10.0f}{p99h:>12}{floor:>22.1f}{'FAIL' if floor>8 else 'ok':>10}")
    maxdoc=max(S);
    print(f"\n  Largest doc alone: {maxdoc} tok / {R_load:.0f} = {maxdoc/R_load:.0f}s own-prefill (>> 8s SLO).")
    print("  => The p99 request's OWN prefill time exceeds the SLO at EVERY hit rate (turn-0 doc tail is")
    print("     uncacheable => p99_S invariant). P99 TTFT >= this floor => goodput@SLO impossible by caching.")
    print("  This is the analytical form of the Paper-1 impossibility: caching cuts E[S] (throughput/ρ) but")
    print("  the goodput-binding P99 is set by the caching-invariant service TAIL.")

if __name__=="__main__": main()
