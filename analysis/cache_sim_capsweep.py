# Capacity sweep: LRU vs Belady vs protect-pending (realizable). Motivation figure for the paper.
import json, math
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
N=len(convs)
def gen_order(lam, maxc=256, tpot=0.27, ttft=1.0):
    q=deque((c,0) for c in range(N)); inflight=[]; t=0.0; dt=1.0/lam
    order=[]; next_pull=0.0; comp_map={}
    while q or inflight:
        can_pull=q and len(inflight)<maxc
        if can_pull and (not inflight or next_pull<=inflight[0]):
            t=max(t,next_pull); c,ti=q.popleft(); order.append((c,ti))
            ctx,p,a=convs[c][ti]; ct=t+ttft+tpot*a; inflight.append(ct); inflight.sort(); comp_map.setdefault(round(ct,6),[]).append((c,ti)); next_pull=t+dt
        else:
            if not inflight: break
            ct=inflight.pop(0); t=max(t,ct)
            for (c,ti) in comp_map.get(round(ct,6),[]):
                if ti+1<len(convs[c]): q.append((c,ti+1))
            comp_map.pop(round(ct,6),None)
    return order
def sim(order, cap_tokens, policy):
    cap=cap_tokens//PAGE; r={}; la={}; nu={}; pend={}; hit=0; denom=0; total=0
    idxof={k:i for i,k in enumerate(order)}
    for idx,(c,ti) in enumerate(order):
        ctx,p,a=convs[c][ti]; need=math.ceil(ctx/PAGE) if ctx>0 else 0; rc=r.get(c,0)
        hit+=min(rc,need)*PAGE; denom+=(ctx+p)
        nl=math.ceil((ctx+p+a)/PAGE); r[c]=nl; total+=nl-rc; la[c]=idx
        nu[c]=idxof.get((c,ti+1),float('inf')) if ti+1<len(convs[c]) else float('inf')
        pend[c]=idxof.get((c,ti+1),-1) if ti+1<len(convs[c]) else -1
        while total>cap:
            cand=[x for x in r if r[x]>0 and x!=c]
            if not cand: break
            if policy=="lru": vc=min(cand,key=lambda x:la[x])
            elif policy=="belady": vc=max(cand,key=lambda x:nu[x])
            else:  # protect_pending
                nonp=[x for x in cand if not (pend.get(x,-1)>idx)]; pool=nonp if nonp else cand; vc=min(pool,key=lambda x:la[x])
            take=min(r[vc],total-cap); r[vc]-=take; total-=take
    return hit/denom if denom else 0
order=gen_order(5)
print("cap(M) | LRU   Belady protPend | gain(pp) recompute_reduction%")
for cap in [6,7,8,9,10,10.7,12,14]:
    l=sim(order,int(cap*1e6),"lru"); b=sim(order,int(cap*1e6),"belady"); pp=sim(order,int(cap*1e6),"protect_pending")
    rr=100*(( (1-l)-(1-pp) )/(1-l)) if l<1 else 0
    print(f"{cap:5.1f}  | {l:.3f}  {b:.3f}  {pp:.3f}  | {100*(pp-l):+.1f}   {rr:+.1f}%")
