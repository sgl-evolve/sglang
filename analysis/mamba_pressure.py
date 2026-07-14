#!/usr/bin/env python3
"""floyd: is the Mamba-state pool ever the binding constraint? (bounds the hybrid-asymmetry axis)

Hybrid Qwen3.5 has TWO device pools sharing the HBM budget: attention-KV ("full token usage")
and Mamba/GDN recurrent state ("mamba usage"). If the attention-KV saturates while the Mamba
pool stays slack, then managing the Mamba state cannot move goodput (it is never full) — the
attention-KV is the sole binding tier. Parsed from the stock v0-stock server.log.
"""
import re, statistics as st
LOG = "runs/v0-stock/server.log"
def main():
    full=[]; mam=[]
    for l in open(LOG, errors="ignore"):
        if "token usage" not in l: continue
        f=re.search(r"full token usage: ([0-9.]+)", l); m=re.search(r"mamba usage: ([0-9.]+)", l)
        if f: full.append(float(f.group(1)))
        if m: mam.append(float(m.group(1)))
    def s(v,n):
        v2=sorted(v); k=len(v2)
        print(f"{n}: mean={st.mean(v2):.3f} p50={v2[k//2]:.3f} p90={v2[int(.9*k)]:.3f} p99={v2[int(.99*k)]:.3f} max={max(v2):.3f}")
    s(full,"attn-KV usage "); s(mam,"mamba usage   ")
    both=list(zip(full,mam)); n=len(both)
    aslack=sum(1 for f,m in both if f>0.95 and m<0.9)
    print(f"steps={n}  attn>0.95: {100*sum(f>0.95 for f,_ in both)/n:.1f}%  mamba>0.95: {100*sum(m>0.95 for _,m in both)/n:.1f}%")
    print(f"attn saturated WHILE mamba slack(<0.9): {100*aslack/n:.1f}% => Mamba-state NEVER binding => non-lever")
if __name__=="__main__": main()
