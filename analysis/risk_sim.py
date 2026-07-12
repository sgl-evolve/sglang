# RIGOROUS event-driven cache sim to stress-test the "risk-first pinning ~ Belady" finding.
# Precise PINNABLE set = convs whose next turn is currently ENQUEUED-and-WAITING (realizable: what the
# engine can actually pin), NOT merely "non-terminal". Configurable pull order (FIFO vs LPM-ish).
# Policies:
#   lru                : recency, no future
#   belady             : page-level oracle (min next-use), upper bound
#   pin_all            : protect ALL pinnable convs, evict non-pinnable LRU first (idealized pending, unbudgeted)
#   pin_recent_bud     : budgeted, protect MOST-recently-active pinnable first (naive priority)  [control]
#   pin_risk_bud       : budgeted, protect LEAST-recently-active pinnable first (LRU-victim priority) [proposed]
#   pin_qwait_bud      : budgeted, protect pinnable with LONGEST queue-wait first (direct reuse-soonness signal)
import json, heapq, math
from collections import deque
convs=[[tuple(t) for t in c] for c in json.load(open("convs.json"))]
N=len(convs); PAGE=64

def gen_trace(lam, maxc=256, tpot=0.27, ttft=1.0, pull="fifo"):
    # closed-loop within conversation; returns event list of ADMISSIONS in order, each with:
    #   (c, ti, now, enq_time_of_this_turn)  and a global map of when each turn was enqueued.
    q=[]  # waiting queue of (enq_time, c, ti)
    for c in range(N): q.append((0.0,c,0))
    inflight=[]; t=0.0; dt=1.0/lam; next_pull=0.0
    admits=[]; enq_time={}; comp_time={}
    for (e,c,ti) in q: enq_time[(c,ti)]=e
    running=[]  # heap of completion times -> (ct,c,ti)
    q=deque(q)
    def pick_from_queue():
        if pull=="fifo":
            return q.popleft()
        # lpm-ish: prefer the turn whose conv has the largest already-produced context (longest prefix)
        best=0
        for i,(e,c,ti) in enumerate(q):
            if convs[c][ti][0] > convs[q[best][1]][q[best][2]][0]: best=i
        item=q[best]; del q[best]; return item
    while q or running:
        can_pull=q and len(running)<maxc
        if can_pull and (not running or next_pull<=running[0][0]):
            t=max(t,next_pull); e,c,ti=pick_from_queue()
            admits.append((c,ti,t,e)); ctx,p,a=convs[c][ti]; ct=t+ttft+tpot*a
            heapq.heappush(running,(ct,c,ti)); comp_time[(c,ti)]=ct; next_pull=t+dt
        else:
            if not running: break
            ct,c,ti=heapq.heappop(running); t=max(t,ct)
            if ti+1<len(convs[c]):
                enq_time[(c,ti+1)]=ct; q.append((ct,c,ti+1))
    return admits, enq_time, comp_time

def simulate(admits, enq_time, comp_time, cap_tok, policy, budget_frac=0.5):
    cap=cap_tok//PAGE; pin_budget=int(cap*budget_frac)
    aidx={(c,ti):i for i,(c,ti,_,_) in enumerate(admits)}
    # next-use index per (conv) for belady = admit index of this conv's next turn
    r={}; la={}; hit=0; denom=0; total=0
    nextuse={}   # conv -> admit idx of its next turn (inf if none)
    pinnable_since={}  # conv -> enq_time of its waiting next turn, if currently pinnable, else absent
    for idx,(c,ti,now,e) in enumerate(admits):
        ctx,p,a=convs[c][ti]; need=math.ceil(ctx/PAGE) if ctx>0 else 0
        rc=r.get(c,0); h=min(rc,need); hit+=h*PAGE; denom+=(ctx+p)
        new_len=math.ceil((ctx+p+a)/PAGE); r[c]=new_len; total+=new_len-rc; la[c]=idx
        # this conv's next turn (if any): it becomes pinnable at comp_time[(c,ti)], waiting until admitted
        nu = aidx.get((c,ti+1), math.inf) if ti+1<len(convs[c]) else math.inf
        nextuse[c]=nu
        # recompute pinnable set lazily at eviction time using enq/comp/now
        def is_pinnable(x):
            # x pinnable now iff its most-recent turn completed (<=now) and its next turn not yet admitted (enq exists, admit>idx)
            nxu=nextuse.get(x,math.inf)
            if nxu==math.inf: return False
            # next turn enqueued (its enq_time = comp_time of prev) and not yet admitted (nxu>idx)
            return nxu>idx
        while total>cap:
            cand=[x for x in r if r[x]>0 and x!=c]
            if not cand: break
            if policy=="lru": vc=min(cand,key=lambda x: la[x])
            elif policy=="belady": vc=max(cand,key=lambda x: nextuse.get(x,math.inf))
            else:
                pin=[x for x in cand if is_pinnable(x)]
                if policy!="pin_all" and pin:
                    if policy=="pin_recent_bud":   key=lambda x:-la[x]
                    elif policy=="pin_risk_bud":   key=lambda x: la[x]
                    elif policy=="pin_qwait_bud":  key=lambda x: -( (idx)-la[x] )  # longest since last active first (=lowest la)
                    ps=sorted(pin,key=key); kept=0; keep=set()
                    for x in ps:
                        if kept+r[x]<=pin_budget: keep.add(x); kept+=r[x]
                    pin=[x for x in pin if x in keep]
                pset=set(pin); nonp=[x for x in cand if x not in pset]
                pool=nonp if nonp else cand; vc=min(pool,key=lambda x: la[x])
            take=min(r[vc],total-cap); r[vc]-=take; total-=take
    return hit/denom if denom else 0

import sys
pull=sys.argv[1] if len(sys.argv)>1 else "fifo"
print(f"=== pull order: {pull} ===")
for lam in [3,5]:
    admits,enq,comp=gen_trace(lam,pull=pull)
    for cap in [10_700_000, 8_000_000]:
        lru=simulate(admits,enq,comp,cap,"lru")
        pinall=simulate(admits,enq,comp,cap,"pin_all")
        bel=simulate(admits,enq,comp,cap,"belady")
        g=bel-lru
        print(f"lam={lam} cap={cap//10**6}M | lru={lru:.3f} pin_all={pinall:.3f} belady={bel:.3f}  [belady gap={100*g:.2f}pp; pin_all captures {100*(pinall-lru)/g if g>0 else 0:.0f}%]")
        for bf in [0.25,0.5,0.75]:
            rec=simulate(admits,enq,comp,cap,"pin_recent_bud",bf)
            rsk=simulate(admits,enq,comp,cap,"pin_risk_bud",bf)
            print(f"     bud={bf:.2f}: recent-first +{100*(rec-lru):5.2f}pp ({100*(rec-lru)/g if g>0 else 0:3.0f}% of belady)   RISK-first +{100*(rsk-lru):5.2f}pp ({100*(rsk-lru)/g if g>0 else 0:3.0f}% of belady)")
