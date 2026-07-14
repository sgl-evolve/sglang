#!/usr/bin/env python3
"""Paper 2 -- discrete-event sim of the chunked-prefill server, to (a) reproduce
the measured flat-p50 / exploding-p99 signature and C(h) saturation, and (b)
compute the OFFLINE-OPTIMAL (SRPF) goodput@SLO ceiling in [3, C].

Model (continuous batching + chunked prefill, the sglang serving loop):
  - Requests arrive Poisson(lam); request i has uncached prefill work U_i tokens
    (from the real trace at hit h; turn-0 docs are the giants, follow-ups small).
  - Active set: up to N_MAX=256 requests in service. A scheduler step processes a
    global budget of B prefill tokens, split evenly across active reqs (chunked),
    and advances wall-clock by (tokens_this_step / R_raw) seconds.
  - A request completes its prefill (TTFT recorded) when its U_i tokens are done.
  - Admission order into the active set: FCFS (baseline) or SRPF (smallest U first).
Calibrate R_raw, B so FCFS reproduces stock (p50~1s, p99 6-11s @lam3 .. 41s @lam10,
C_ach saturating ~4.1). Then read off SRPF goodput@SLO.
"""
import json, ast, heapq, random, statistics, sys
MIX="/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"; CH=4.0
N_MAX=256

def uncached_work(hit):
    """Per-request uncached prefill tokens U_i, faithful to the loogle multiturn loader."""
    recs=[json.loads(l) for l in open(MIX)]; U=[]
    for r in recs:
        qp=r.get("qa_pairs",""); doc=r.get("input","")
        if qp=="none" or (isinstance(qp,(list,str)) and len(qp)==0):
            turns=["Input: "+doc+" Question: Please summarize the input"]
        else:
            try: qa=ast.literal_eval(qp) if isinstance(qp,str) else qp
            except Exception: continue
            turns=[("Input: "+doc+" Question: "+str(q.get("Q",""))) if i==0 else str(q.get("Q",""))
                   for i,q in enumerate(qa)]
        committed=0
        for ti,p in enumerate(turns):
            newq=max(1,int(len(p)/CH))
            U.append(newq if ti==0 else int((1-hit)*committed)+newq)  # turn-0 full; later reuse prior
            committed += newq
    return U

def simulate(U, lam, R_raw, B, policy, seed=0):
    rnd=random.Random(seed); n=len(U)
    # arrivals
    t=0.0; arr=[]
    for i in range(n): t+=rnd.expovariate(lam); arr.append(t)
    order=sorted(range(n), key=lambda i: arr[i])
    ttft=[None]*n
    clock=0.0; nexta=0
    active=[]   # list of [remaining, idx]
    waiting=[]  # indices admitted-eligible but no slot (arrived, waiting)
    def admit():
        # fill active set up to N_MAX from waiting, by policy
        if not waiting: return
        if policy=="srpf": waiting.sort(key=lambda i: U[i])       # smallest work first
        else: waiting.sort(key=lambda i: arr[i])                  # fcfs
        while waiting and len(active)<N_MAX:
            i=waiting.pop(0); active.append([U[i], i])
    # event loop: advance by admitting arrivals up to clock, then process one step
    while nexta<n or active or waiting:
        # bring in all arrivals whose time <= clock (or jump clock to next arrival if idle)
        if not active and not waiting and nexta<n:
            clock=max(clock, arr[order[nexta]])
        while nexta<n and arr[order[nexta]]<=clock:
            waiting.append(order[nexta]); nexta+=1
        admit()
        if not active:
            if nexta<n: clock=arr[order[nexta]]; continue
            else: break
        # one chunked-prefill step: B tokens split across active reqs
        k=len(active); share=B/k
        step_tokens=0.0
        done=[]
        for slot in active:
            take=min(slot[0], share); slot[0]-=take; step_tokens+=take
        dt=step_tokens/R_raw; clock+=dt
        # bring in arrivals during this step (they wait for a slot next admit)
        while nexta<n and arr[order[nexta]]<=clock:
            waiting.append(order[nexta]); nexta+=1
        # finish completed
        still=[]
        for slot in active:
            if slot[0]<=1e-6: ttft[slot[1]]=clock-arr[slot[1]]
            else: still.append(slot)
        active=still
        admit()
    xs=sorted(v for v in ttft if v is not None)
    return xs

def pct(xs,q): return xs[min(len(xs)-1,int(q*len(xs)))]
def run(hit, R_raw, B, lam, policy):
    U=uncached_work(hit); xs=simulate(U,lam,R_raw,B,policy)
    return pct(xs,0.50), pct(xs,0.99), len(xs)

if __name__=="__main__":
    hit=0.654
    U=uncached_work(hit); n=len(U); Us=sorted(U)
    print(f"stock hit={hit}: requests={n}  U p50={Us[n//2]} p90={Us[int(.9*n)]} "
          f"p99={Us[int(.99*n)]} max={max(U)}  mean={sum(U)/n:.0f}")
    # calibrate R_raw so C_ach(lam=10 overload) ~ 4.14 ; B so p50~1s
    # C_ach = R_raw/E[U]; E[U]=mean; -> R_raw = 4.14*mean
    R_raw = 4.14*sum(U)/n
    print(f"calibrated R_raw = {R_raw:.0f} tok/s  (=C_sat*E[U])")
    for B in (8192, 16384, 32768):
        print(f"\n B={B}:")
        print(f"   {'lam':>4} {'p50_fcfs':>9} {'p99_fcfs':>9} {'p99_srpf':>9}")
        for lam in (3,5,7,10):
            p50f,p99f,_=run(hit,R_raw,B,lam,"fcfs")
            _,p99s,_=run(hit,R_raw,B,lam,"srpf")
            print(f"   {lam:4d} {p50f*1000:9.0f} {p99f*1000:9.0f} {p99s*1000:9.0f}")
