import json, ast, sys, time
from transformers import AutoTokenizer
MIX="/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"
MODEL="/rmeng_data/junyanch-data/hf_cache/hub/models--Qwen--Qwen3.5-122B-A10B-FP8/snapshots/a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9"
tok=AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
rows=[json.loads(l) for l in open(MIX)]
t0=time.time()
doc_toks=[]; q_toks=[]; a_toks=[]; conv_total=[]; nturns=[]
for r in rows:
    inp=r.get("input","") or ""
    dt=len(tok(inp, add_special_tokens=False).input_ids) if inp else 0
    doc_toks.append(dt)
    q=r.get("qa_pairs","")
    try: qa=ast.literal_eval(q) if isinstance(q,str) and q not in("","none") else []
    except: qa=[]
    nturns.append(max(1,len(qa)))
    ctotal=dt
    for i,pair in enumerate(qa):
        qt=len(tok(pair.get("Q","") or "", add_special_tokens=False).input_ids)
        at=len(tok(pair.get("A","") or "", add_special_tokens=False).input_ids)
        q_toks.append(qt); a_toks.append(at); ctotal+=qt+at
    conv_total.append(ctotal)
def summ(x,n):
    x=sorted(x); L=len(x)
    if L==0: print(f"{n}: (none)"); return
    p=lambda q:x[min(L-1,int(q*L))]
    print(f"{n}: n={L} min={x[0]} p10={p(.1)} p50={p(.5)} p90={p(.9)} p99={p(.99)} max={x[-1]} mean={sum(x)/L:.0f} sum={sum(x)}")
print(f"tokenized in {time.time()-t0:.0f}s")
summ(doc_toks,"turn0_doc_tokens (all convs)")
summ([d for d in doc_toks if d>0],"turn0_doc_tokens (non-empty)")
summ(q_toks,"followup_Q_tokens")
summ(a_toks,"answer_A_tokens (output len)")
summ(conv_total,"conv_total_tokens (full working set/conv)")
print("GRAND total tokens:",sum(conv_total))
json.dump({"doc_toks":doc_toks,"nturns":nturns,
           "q_p50":sorted(q_toks)[len(q_toks)//2] if q_toks else 0,
           "a_p50":sorted(a_toks)[len(a_toks)//2] if a_toks else 0},
          open("tools/trace_stats.json","w"))
print("wrote tools/trace_stats.json")
