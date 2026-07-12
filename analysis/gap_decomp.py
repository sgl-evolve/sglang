# Realizability-gap DECOMPOSITION for pending-aware KV retention.
# Reads convs.json (token lengths). Closed-loop Poisson arrival + re-enqueue (same model as cache_sim.py).
# Ladder of policies, each strictly more future-informed:
#   lru               : no future info (baseline)
#   protect_pending   : v1 -- protect convs whose NEXT turn is already enqueued (waiting), LRU within (REALIZABLE now)
#   protect_pend_bud  : v1 with a hard pin budget = frac of cache (what the engine actually enforces)
#   pred_thresh_K     : REALISTIC predictor -- protect a completed turn's conv iff current turn index < K
#                       (predict "will continue"); imperfect precision, budgeted. Proactive (does not need enqueue).
#   protect_continue  : IDEALIZED predictor -- protect ALL non-terminal convs (perfect continuation oracle), LRU within
#   belady            : full page-level future (upper bound)
# Decomposition of the Belady gap G = belady-lru:
#   A = protect_pending - lru      (captured by the realizable pending-set signal)
#   B = protect_continue - protect_pending  (extra value of protecting NOT-yet-enqueued future turns, perfect precision)
#   C = belady - protect_continue  (irreducible: page-level ordering not expressible as per-conv continue/terminal)
import json, heapq, math
convs=[[tuple(t) for t in c] for c in json.load(open("convs.json"))]
N=len(convs); PAGE=64

def gen_order(lam, maxc=256, tpot=0.27, ttft=1.0):
    q=deque=__import__("collections").deque((c,0) for c in range(N)); inflight=[]; t=0.0; dt=1.0/lam
    order=[]; issue_time={}; comp_time={}; next_pull=0.0; comp_map={}
    while q or inflight:
        can_pull=q and len(inflight)<maxc
        if can_pull and (not inflight or next_pull<=inflight[0]):
            t=max(t,next_pull); c,ti=q.popleft(); issue_time[(c,ti)]=t; order.append((c,ti))
            ctx,p,a=convs[c][ti]; svc=ttft+tpot*a; ct=t+svc
            comp_time[(c,ti)]=ct; heapq.heappush(inflight,ct); comp_map.setdefault(round(ct,6),[]).append((c,ti)); next_pull=t+dt
        else:
            if not inflight: break
            ct=heapq.heappop(inflight); t=max(t,ct)
            for (c,ti) in comp_map.get(round(ct,6),[]):
                if ti+1<len(convs[c]): q.append((c,ti+1))
            comp_map.pop(round(ct,6),None)
    return order, issue_time, comp_time

def simulate(order, issue_time, comp_time, cap_tokens, policy, K=3, budget_frac=0.5):
    cap=cap_tokens//PAGE
    r={}; la={}; nu={}; nextissue={}; curturn={}
    hit=0; denom=0; total=0
    order_idx={k:i for i,k in enumerate(order)}
    pin_budget=int(cap*budget_frac)
    for idx,(c,ti) in enumerate(order):
        ctx,p,a=convs[c][ti]
        need=math.ceil(ctx/PAGE) if ctx>0 else 0
        rc=r.get(c,0); h=min(rc,need)
        hit+=h*PAGE; denom+=(ctx+p)
        new_len=math.ceil((ctx+p+a)/PAGE); r[c]=new_len; total+=new_len-rc
        la[c]=idx
        nu[c]=order_idx.get((c,ti+1), float('inf')) if ti+1<len(convs[c]) else float('inf')
        nextissue[c]=order_idx.get((c,ti+1), float('inf')) if ti+1<len(convs[c]) else float('inf')
        curturn[c]=ti
        def protected(x):
            if policy=="protect_pending": return nextissue.get(x,float('inf'))>idx and nu[x]!=float('inf')
            if policy=="protect_pend_bud": return nextissue.get(x,float('inf'))>idx and nu[x]!=float('inf')
            if policy=="pred_thresh": return nu[x]!=float('inf') and curturn.get(x,999)<K  # predict continue for early turns
            if policy=="protect_continue": return nu[x]!=float('inf')  # perfect continuation oracle
            return False
        while total>cap:
            cand=[x for x in r if r[x]>0 and x!=c]
            if not cand: break
            if policy=="lru":
                vc=min(cand,key=lambda x: la[x])
            elif policy=="belady":
                vc=max(cand,key=lambda x: nu[x])
            else:
                # protect set (optionally budgeted), evict unprotected LRU first
                prot=[x for x in cand if protected(x)]
                if policy in ("protect_pend_bud","pred_thresh"):
                    # enforce hard pin budget: keep only the most-recently-active protected up to budget
                    if prot:
                        prot_sorted=sorted(prot,key=lambda x:-la[x]); kept=0; keepset=set()
                        for x in prot_sorted:
                            if kept+r[x]<=pin_budget: keepset.add(x); kept+=r[x]
                        prot=[x for x in prot if x in keepset]
                protset=set(prot)
                nonp=[x for x in cand if x not in protset]
                pool=nonp if nonp else cand
                vc=min(pool,key=lambda x: la[x])
            free=r[vc]; take=min(free,total-cap); r[vc]-=take; total-=take
    return hit/denom if denom else 0

POLS=["lru","protect_pending","protect_pend_bud","pred_thresh","protect_continue","belady"]
for lam in [3,5,10]:
    order,it,ct=gen_order(lam)
    for cap in [10_700_000, 8_000_000]:
        res={p:simulate(order,it,ct,cap,p) for p in POLS}
        G=res["belady"]-res["lru"]
        A=res["protect_pending"]-res["lru"]
        B=res["protect_continue"]-res["protect_pending"]
        C=res["belady"]-res["protect_continue"]
        print(f"lam={lam:2d} cap={cap//10**6:2d}M | lru={res['lru']:.3f} v1={res['protect_pending']:.3f} "
              f"v1bud={res['protect_pend_bud']:.3f} predK={res['pred_thresh']:.3f} "
              f"contOracle={res['protect_continue']:.3f} belady={res['belady']:.3f}", flush=True)
        if G>1e-9:
            print(f"        gap G={G*100:.2f}pp  A(pending)={A*100:.2f}pp={100*A/G:.0f}%  "
                  f"B(predict-future)={B*100:.2f}pp={100*B/G:.0f}%  C(page-belady)={C*100:.2f}pp={100*C/G:.0f}%", flush=True)
