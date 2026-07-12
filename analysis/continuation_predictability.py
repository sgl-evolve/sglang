# Why the Belady gap isn't server-realizable: can turn-count predict whether a finished turn will be reused?
import json
rows=[json.loads(l) for l in open("/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl")]
def nturns(d):
    qp=d.get("qa_pairs")
    if not qp or qp=="none" or len(qp)==0: return 1
    try: return max(1,len(eval(qp)))
    except: return 1
T=[nturns(d) for d in rows]
N=len(T)
# For a conversation with n turns, turns 1..n-1 are "reused" (continue), turn n is "terminal".
# At the moment turn k completes (k=1..n), P(continue) = P(n>k | reached turn k).
from collections import Counter
cnt=Counter(T)
maxk=max(T)
print(f"convs={N} mean_turns={sum(T)/N:.2f}")
# survival: reached_k = #convs with n>=k ; continue_k = #convs with n>k
print("k  reached  P(continue|reached k)  (a finished turn-k prefix will be reused?)")
for k in [1,2,3,4,5,8,10]:
    reached=sum(1 for t in T if t>=k); cont=sum(1 for t in T if t>k)
    if reached: print(f"{k:2d}  {reached:5d}    {cont/reached:.3f}")
# If we protect ALL finished turns (pc): fraction that are terminal (wasted)
total_finished_turns=sum(T)  # each conv contributes n finished turns
terminal_turns=N             # each conv has exactly 1 terminal (last) turn
print(f"\nOf all finished-turn events: {total_finished_turns}; terminal (never-reused)={terminal_turns} = {100*terminal_turns/total_finished_turns:.0f}%")
print(f"=> protecting every finished turn (pc) wastes protection on {100*terminal_turns/total_finished_turns:.0f}% terminal turns.")
# Best turn-count threshold predictor: protect only turns k<K (predict continue). 
# precision = P(reused | protected), recall = fraction of reused turns protected
reused_total=total_finished_turns - terminal_turns
print("\nthreshold K (protect finished turn-k if k<K):  precision  recall")
for K in [2,3,4,6,10,999]:
    protected=sum(min(t,K-1) for t in T)         # turns 1..min(t,K-1) protected
    prot_reused=sum(max(0,min(t-1,K-1)) for t in T)  # of those, reused = turns 1..min(t-1,K-1)
    prec=prot_reused/protected if protected else 0
    rec=prot_reused/reused_total if reused_total else 0
    print(f"K={K:3d}: precision={prec:.3f} recall={rec:.3f}")
