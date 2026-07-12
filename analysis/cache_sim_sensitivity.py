# Robustness of the LRU->Belady gap & protect-pending capture across arrival-model params.
import json, math
from collections import deque
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-122B-A10B-FP8", trust_remote_code=True)
enc=lambda s: len(tok.encode(s, add_special_tokens=False))
rows=[json.loads(l) for l in open("/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl")]
PAGE=64; convs=[]
for d in rows:
    qp=d.get("qa_pairs"); turns=[]
    if not qp or qp=="none" or len(qp)==0:
        turns.append((0, enc("Input: "+d["input"]+" Question: Please summarize the input"), enc(d["input"][:1024])))
    else:
        try: pairs=eval(qp)
        except: pairs=[]
        if not pairs: continue
        ctx=0
        for i,qa in enumerate(pairs):
            p=enc(("Input: "+d["input"]+" Question: "+qa["Q"]) if i==0 else qa["Q"]); a=enc(qa["A"]); turns.append((ctx,p,a)); ctx+=p+a
    if turns: convs.append(turns)
N=len(convs)
def gen_order(lam, maxc, tpot, ttft):
    q=deque((c,0) for c in range(N)); inflight=[]; t=0.0; dt=1.0/lam; order=[]; comp_map={}; nxt=0.0
    while q or inflight:
        if q and len(inflight)<maxc and (not inflight or nxt<=inflight[0]):
            t=max(t,nxt); c,ti=q.popleft(); order.append((c,ti)); ctx,p,a=convs[c][ti]
            ct=t+ttft+tpot*a; inflight.append(ct); inflight.sort(); comp_map.setdefault(round(ct,6),[]).append((c,ti)); nxt=t+dt
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
            else: nonp=[x for x in cand if not (pend.get(x,-1)>idx)]; vc=min(nonp if nonp else cand,key=lambda x:la[x])
            take=min(r[vc],total-cap); r[vc]-=take; total-=take
    return hit/denom if denom else 0
CAP=9_000_000
print(f"cap={CAP/1e6:.0f}M  |  lam maxc tpot ttft ->  LRU  Belady  protPend  (gap, capture%)")
for lam,maxc,tpot,ttft in [(5,256,0.27,1.0),(3,256,0.27,1.0),(10,256,0.27,1.0),(5,128,0.27,1.0),(5,256,0.15,0.5),(5,256,0.40,2.0),(7,256,0.20,1.0)]:
    o=gen_order(lam,maxc,tpot,ttft)
    l=sim(o,CAP,"lru"); b=sim(o,CAP,"belady"); pp=sim(o,CAP,"protect_pending")
    cap_pct=100*(pp-l)/(b-l) if b>l else 0
    print(f"           {lam:3d} {maxc:4d} {tpot:.2f} {ttft:.1f} -> {l:.3f} {b:.3f} {pp:.3f}  (+{100*(pp-l):.1f}pp, {cap_pct:.0f}% of Belady)")
