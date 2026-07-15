#!/usr/bin/env python3
# kleinrock: quantify head-of-line (HOL) blocking from per-request dumps (KLEINROCK_PERREQ_DUMP).
# Direct evidence for Paper 1 §2.2/§5.6: the p99 TTFT tail is SHORT prompts (tiny own-prefill) waiting many
# seconds = queueing behind heavy cold-doc prefills, NOT their own compute. A prompt of <1000 tokens cannot have
# >~1s of prefill on Qwen3.5-122B, so a multi-second TTFT for such a request is definitionally HOL delay.
# Aggregates all stock (reserve0) per-request dumps passed as args. Deterministic; no network.
import sys, csv, statistics as st, glob
files=[]
for a in sys.argv[1:]: files += glob.glob(a)
rows=[]
for f in files:
    for r in csv.DictReader(open(f)):
        try: rows.append((float(r['prompt_len']), float(r['ttft_ms'])))
        except: pass
if not rows: print("no rows"); sys.exit(0)
rows.sort(key=lambda x:-x[1])   # by TTFT desc
n=len(rows)
ttfts=sorted(t for _,t in rows)
p99=ttfts[int(n*0.99)]; p999=ttfts[min(n-1,int(n*0.999))]
print(f"files={len(files)} reqs={n} | p99 TTFT={p99/1000:.1f}s p99.9={p999/1000:.1f}s")
# the p99 tail = requests with ttft >= p99
tail=[(pl,tt) for pl,tt in rows if tt>=p99]
SHORT=1000  # tokens; << any multi-second prefill
short_in_tail=sum(1 for pl,tt in tail if pl<SHORT)
print(f"\n=== the p99 TTFT tail (top {len(tail)} reqs, ttft>= {p99/1000:.1f}s) ===")
print(f"  fraction with prompt_len < {SHORT} tok (own-prefill negligible => TTFT is HOL): {short_in_tail}/{len(tail)} = {100*short_in_tail/len(tail):.0f}%")
tl=[pl for pl,_ in tail]
print(f"  tail prompt_len: median={st.median(tl):.0f} min={min(tl):.0f} max={max(tl):.0f}")
# by prompt-size bucket: median TTFT
def bucket(pl):
    if pl<1000: return "<1K (tiny)"
    if pl<8000: return "1K-8K"
    if pl<50000: return "8K-50K"
    return ">=50K (heavy)"
from collections import defaultdict
b=defaultdict(list)
for pl,tt in rows: b[bucket(pl)].append(tt)
print("\n=== TTFT by prompt-size bucket (HOL victims = small prompts with LARGE ttft) ===")
for k in ["<1K (tiny)","1K-8K","8K-50K",">=50K (heavy)"]:
    if k in b:
        v=sorted(b[k]); print(f"  {k:16} n={len(v):5} ttft median={st.median(v)/1000:5.1f}s  p99={v[int(len(v)*0.99)]/1000:5.1f}s  max={max(v)/1000:5.1f}s")
# headline: tiny prompts that wait > 8s (SLO) = pure HOL victims
tiny=[tt for pl,tt in rows if pl<1000]
tiny_over_slo=sum(1 for tt in tiny if tt>8000)
print(f"\n=== HEADLINE: tiny (<1K tok) requests with TTFT > 8s SLO = pure HOL victims ===")
print(f"  {tiny_over_slo} of {len(tiny)} tiny requests ({100*tiny_over_slo/len(tiny):.0f}%) exceed the 8s SLO purely from queueing")
print(f"  (a <1K-token prompt has <~0.3s of own prefill; its multi-second TTFT is HOL delay)")
