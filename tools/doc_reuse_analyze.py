#!/usr/bin/env python3
# kleinrock: workload analysis — does the goodput@SLO p99 tail admit ANY caching lever, or is it necessarily a
# scheduling problem? The p99 tail is HOL blocking behind the few longest (cold turn-0) documents. If those
# tail-causing mega-docs are REUSED across conversations, a cross-conversation / content-addressed prefix cache
# could convert their cold prefills into hits (a caching lever for the tail). If they are UNIQUE (first-sight),
# no cache at any capacity can help them -> the tail is necessarily schedulable-only. Deterministic, GPU-free.
import json,sys,hashlib
from collections import defaultdict,Counter
MIX=sys.argv[1] if len(sys.argv)>1 else "/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"
recs=[]
for i,line in enumerate(open(MIX)):
    d=json.loads(line); inp=d.get("input","")
    recs.append((i, hashlib.md5(inp.encode()).hexdigest(), len(inp)))
n=len(recs); pos=defaultdict(list); dl={}
for i,h,L in recs: pos[h].append(i); dl[h]=L
uniq=len(pos); reused={h:p for h,p in pos.items() if len(p)>1}
print(f"records={n}  unique_docs={uniq}  records_sharing_a_doc={n-uniq} ({100*(n-uniq)/n:.0f}%)")
print(f"reused_docs={len(reused)} covering {sum(len(p) for p in reused.values())} records")
# ~4 char/token; tail = longest docs (>=100K tok ~ >=400K char)
def tok(L): return L//4
def bucket(L):
    return ">=100K tok (p99 tail)" if L>=400_000 else "25-100K tok" if L>=100_000 else "10-25K tok" if L>=40_000 else "<10K tok"
ub=Counter(); rb=Counter()
for h,p in pos.items():
    (rb if len(p)>1 else ub)[bucket(dl[h])]+=1
print("\nlen bucket             : unique(seen once) | reused(shared across convs)")
for b in [">=100K tok (p99 tail)","25-100K tok","10-25K tok","<10K tok"]:
    print(f"  {b:22}: {ub[b]:4} | {rb[b]:3}")
tail=[(h,p) for h,p in pos.items() if dl[h]>=400_000]
tail.sort(key=lambda x:-dl[x[0]])
print(f"\nEXTREME-TAIL docs (>=100K tok, the p99 HOL-causers): count={len(tail)}, reused={sum(1 for _,p in tail if len(p)>1)}")
for h,p in tail: print(f"  ~{tok(dl[h])//1000}K tok  occurrences={len(p)}  positions={p}")
avoid=sum(len(p)-1 for h,p in reused.items() if dl[h]>=100_000)
print(f"\navoidable cross-conv prefills among long(>=25K tok) docs = {avoid} / {n} = {100*avoid/n:.2f}% (negligible)")
print("VERDICT: p99-tail mega-docs are"+(" UNIQUE first-sight -> caching CANNOT move the p99 tail; scheduling-only." if all(len(p)==1 for _,p in tail) else " partly reused -> a caching lever may exist."))
