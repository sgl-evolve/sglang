#!/usr/bin/env python3
"""Direction 4 free screen: does prefill TAIL-ACCELERATION cut p99 TTFT below the SLO?

Idea: the p99 TTFT tail = few giant turn-0 prefills whose effective rate R is crushed by
256-way contention (~4100 tok/s vs single-req peak ~20-40K). Dedicate compute to a giant
while it prefills (suppress decode/other-prefill) so it finishes at R_peak; the many small
requests have slack under the 8s SLO to absorb the brief monopolization.

Model (processor-sharing approximation, calibrated to baseline p99~11.5s @λ=3):
- Requests arrive as a Poisson stream at the measured achieved rate; each has uncached
  prefill work U (tokens, from the real trace at stock hit) and its TTFT = time from arrival
  until its prefill work is fully served.
- BASELINE: GPU is processor-shared across all in-flight prefills at aggregate rate
  R_agg (tok/s); a request with work U among N concurrent prefills gets R_agg/N -> its
  prefill finishes when it has accumulated U tokens of service. (Interleaved w/ decode.)
- TAIL-ACCEL: requests with U >= THRESH are 'giants' and get PRIORITY — when any giant is
  in service it takes the whole R_agg (others paused); giants served FCFS among themselves;
  small requests share R_agg only when no giant is active.
We compare p99 TTFT. Calibrate R_agg so baseline p99 ~ 11.5s.
"""
import json, ast, heapq, random, statistics
MIX="/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"; CH=4.0; random.seed(11)

def uncached_stream(hit=0.66):
    recs=[json.loads(l) for l in open(MIX)]; U=[]
    for r in recs:
        qp=r.get("qa_pairs",""); doc=r.get("input","")
        if qp=="none" or (isinstance(qp,(list,str)) and len(qp)==0): turns=[("Input: "+doc+" Q: summarize","")]
        else:
            try: qa=ast.literal_eval(qp) if isinstance(qp,str) else qp
            except: continue
            turns=[(("Input: "+doc+" Q: "+str(q.get("Q",""))) if i==0 else str(q.get("Q",""))) for i,q in enumerate(qa)]
            turns=[(t,"") for t in turns]
        committed=0
        for ti,(p,_) in enumerate(turns):
            newq=max(1,int(len(p)/CH)); prior=committed
            U.append(newq if ti==0 else int((1-hit)*prior)+newq); committed=prior+newq
    return U

def simulate(U, lam_req, R_agg, policy, thresh=30000):
    """Discrete-event processor-sharing sim. lam_req = request arrival rate (req/s).
    Returns list of TTFTs. Simplified: we process arrivals; at each 'slice' the active set
    shares R_agg (baseline) or a giant monopolizes (tail-accel)."""
    n=len(U)
    # arrival times (Poisson)
    t=0.0; arr=[]
    for i in range(n): t+=random.expovariate(lam_req); arr.append(t)
    # event sim with small time steps is too slow; use an M/G/PS-ish approximation:
    # process requests in arrival order through a virtual single server whose RATE the
    # request sees = R_agg / (effective concurrency). We approximate effective concurrency
    # by the number of arrivals within the request's own service window (self-consistent-ish).
    # Simpler + robust: treat as a single FIFO server of capacity R_agg for the PRIORITY class.
    # Baseline: single PS server rate R_agg over all work -> TTFT_i via M/G/1-PS sojourn approx.
    # We instead run an explicit event sim at the giant granularity (giants are few).
    ttft=[0.0]*n
    # explicit sim: server timeline; giants get priority (tail-accel) or FCFS (baseline).
    # Represent work as (arrival, U, idx). Server processes at R_agg tok/s, one 'job' at a time
    # for the priority class; small jobs run in a shared background pool at R_agg/Kbg.
    Kbg=64  # background sharing factor for small reqs (calibration knob ~ concurrency)
    jobs=sorted(range(n), key=lambda i: arr[i])
    # priority queue of (ready_time)
    server_free=0.0
    small_q=[]  # (arrival, U, idx) small reqs, processed in background PS
    # We do a two-class approximation:
    #  - giants: served one-at-a-time by the server at R_agg (tail-accel) or interleaved (baseline)
    #  - smalls: served in background at R_agg/Kbg each, always flowing
    for i in jobs:
        u=U[i]; a=arr[i]
        is_giant = u>=thresh
        if policy=="baseline":
            # everything is background PS at R_agg/Kbg
            ttft[i]=u/(R_agg/Kbg)
        else: # tail-accel: giants monopolize server at R_agg; smalls background
            if is_giant:
                start=max(a, server_free); server_free=start+u/R_agg
                ttft[i]=(server_free-a)
            else:
                # small: background, but paused while a giant holds the server after its arrival
                base=u/(R_agg/Kbg)
                # add expected pause = fraction of its window overlapping giant service
                ttft[i]=base  # smalls keep flowing in background pool (giant uses separate share)
    return ttft

def p99(x): xs=sorted(x); return xs[int(0.99*len(xs))]
def main():
    U=uncached_stream(); n=len(U)
    print(f"requests={n}  U: p50={sorted(U)[n//2]} p99={sorted(U)[int(0.99*n)]} max={max(U)}")
    lam=2.8  # achieved req/s at λ=3
    # calibrate R_agg/Kbg so baseline p99 ~ 11.5s: p99 TTFT_baseline = U_p99/(R_agg/Kbg)
    Up99=sorted(U)[int(0.99*n)]
    Reff=Up99/11.5   # effective per-request rate reproducing baseline p99
    print(f"calibrated effective per-req rate R_eff={Reff:.0f} tok/s (baseline p99 -> 11.5s)")
    base=simulate(U,lam,Reff*64,"baseline")  # R_agg=Reff*Kbg
    print(f"baseline p99 TTFT = {p99(base):.1f}s (check ~11.5)")
    for Rpeak_mult in [4,8,16]:
        # tail-accel: giants served at R_peak = Rpeak_mult * R_eff (dedicated)
        acc=simulate(U,lam,Reff*Rpeak_mult,"accel",thresh=30000)
        # giants' ttft in acc uses server at R_agg=Reff*Rpeak_mult; smalls background at R_eff
        # recompute smalls at R_eff, giants at accel:
        Up=sorted(U)
        print(f"  R_peak={Rpeak_mult}x: tail-accel p99 TTFT = {p99(acc):.1f}s  "
              f"(giant p99 work {Up99} tok / {Reff*Rpeak_mult:.0f} = {Up99/(Reff*Rpeak_mult):.1f}s)")
    print("\nNOTE: crude 2-class PS approximation; giants few, smalls keep flowing. Screen only.")
    print("If tail-accel p99 < 8s while baseline ~11.5s => worth the scheduler surgery to test for real.")
if __name__=="__main__": main()
