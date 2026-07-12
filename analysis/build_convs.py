# Tokenize the mooncake mix ONCE -> convs.json (list of convs; each conv = list of [ctx,p,a] token counts).
# Lets the cache-policy simulator iterate in seconds without re-tokenizing.
import json
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-122B-A10B-FP8", trust_remote_code=True)
enc=lambda s: len(tok.encode(s, add_special_tokens=False))
rows=[json.loads(l) for l in open("/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl")]
convs=[]
for d in rows:
    qp=d.get("qa_pairs"); turns=[]
    if not qp or qp=="none" or len(qp)==0:
        doc=enc("Input: "+d["input"]+" Question: Please summarize the input"); ans=enc(d["input"][:1024]); turns.append([0,doc,ans])
    else:
        try: pairs=eval(qp)
        except: pairs=[]
        if not pairs: continue
        ctx=0
        for i,qa in enumerate(pairs):
            p=enc(("Input: "+d["input"]+" Question: "+qa["Q"]) if i==0 else qa["Q"]); a=enc(qa["A"])
            turns.append([ctx,p,a]); ctx+=p+a
    if turns: convs.append(turns)
json.dump(convs, open("convs.json","w"))
print(f"convs={len(convs)} turns={sum(len(c) for c in convs)} -> convs.json")
