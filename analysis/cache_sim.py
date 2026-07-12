import json, heapq, math
from collections import deque
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-122B-A10B-FP8", trust_remote_code=True)
enc=lambda s: len(tok.encode(s, add_special_tokens=False))
rows=[json.loads(l) for l in open("/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl")]
PAGE=64
convs=[]
for d in rows:
    qp=d.get("qa_pairs"); turns=[]
    if not qp or qp=="none" or len(qp)==0:
        doc=enc("Input: "+d["input"]+" Question: Please summarize the input"); ans=enc(d["input"][:1024]); turns.append((0,doc,ans))
    else:
        try: pairs=eval(qp)
        except: pairs=[]
        if not pairs: continue
        ctx=0
        for i,qa in enumerate(pairs):
            p=enc(("Input: "+d["input"]+" Question: "+qa["Q"]) if i==0 else qa["Q"]); a=enc(qa["A"]); turns.append((ctx,p,a)); ctx+=p+a
    if turns: convs.append(turns)
N=len(convs); print(f"convs={N} turns={sum(len(c) for c in convs)}", flush=True)

def gen_order(lam, maxc=256, tpot=0.27, ttft=1.0):
    q=deque((c,0) for c in range(N)); inflight=[]; t=0.0; dt=1.0/lam
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

def simulate(order, issue_time, comp_time, cap_tokens, policy):
    cap=cap_tokens//PAGE
    r={}; la={}; nu={}; pend_until={}
    hit=0; denom=0; total=0
    order_idx={k:i for i,k in enumerate(order)}
    for idx,(c,ti) in enumerate(order):
        ctx,p,a=convs[c][ti]
        need=math.ceil(ctx/PAGE) if ctx>0 else 0
        rc=r.get(c,0); h=min(rc,need)
        hit+=h*PAGE; denom+=(ctx+p)   # real hit_rate denom = total prompt tokens
        new_len=math.ceil((ctx+p+a)/PAGE); r[c]=new_len; total+=new_len-rc
        la[c]=idx
        nu[c]=order_idx.get((c,ti+1), float('inf')) if ti+1<len(convs[c]) else float('inf')
        # pending window: conv is 'pending' between completion of this turn and issue of next
        if ti+1<len(convs[c]):
            pend_until[c]=order_idx.get((c,ti+1), float('inf'))
        else:
            pend_until[c]=-1  # terminal: never pending again
        while total>cap:
            cand=[x for x in r if r[x]>0 and x!=c]
            if not cand: break
            if policy=="lru":
                vc=min(cand,key=lambda x: la[x])
            elif policy=="belady":
                vc=max(cand,key=lambda x: nu[x])
            elif policy=="protect_pending":
                # evict non-pending (idle/terminal) first by LRU; pending only if forced
                nonp=[x for x in cand if not (pend_until.get(x,-1)>idx)]
                pool=nonp if nonp else cand
                vc=min(pool,key=lambda x: la[x])
            elif policy=="admit_no_terminal":
                # oracle admission: terminal convs (no future) evicted first, else LRU
                term=[x for x in cand if nu[x]==float('inf')]
                pool=term if term else cand
                vc=min(pool,key=lambda x: la[x])
            free=r[vc]; take=min(free,total-cap); r[vc]-=take; total-=take
    return hit/denom if denom else 0

import sys
for lam in [3,5,10]:
    order,it,ct=gen_order(lam)
    line=f"lam={lam:2d}: "
    for cap in [8_000_000,10_700_000]:
        res={p:simulate(order,it,ct,cap,p) for p in ["lru","belady","protect_pending","admit_no_terminal"]}
        line+=f" | cap{cap//10**6}M lru={res['lru']:.3f} belady={res['belady']:.3f} protPend={res['protect_pending']:.3f} noTerm={res['admit_no_terminal']:.3f}"
    print(line, flush=True)
