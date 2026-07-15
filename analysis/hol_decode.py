#!/usr/bin/env python3
"""floyd: the DECODE-SIDE of head-of-line blocking (direct trace, stock v0-stock server.log).

hol_trace.py showed the TTFT-side: the waiting queue is 9.6x deeper during big-cold-doc prefill chunks
(small turns pile up). This shows the SAME mechanism's decode-side: while a big cold doc grinds its
back-to-back 6144-tok chunks, the GPU runs NO decode steps, so every RUNNING request stalls between
tokens. We measure the inter-Decode-batch gap and split by whether a big-cold-doc prefill chunk
(#new-token>=6144, #cached-token==0) occurred in the gap.

Result: gap WITH a big-cold-doc chunk = mean 24.5s (p50 21, p90 37, max 182); gap otherwise = mean 1.8s
(p50 1, p90 4) => ~13.6x longer decode stall. This is why p99 E2E/TPOT are catastrophic (minutes) even at
lambda=3: one head-of-line mechanism (big cold docs monopolizing prefill) damages BOTH the TTFT tail
(waiting turns) AND decode (running requests). SRPF relieves the TTFT side by reordering; the decode stall
is inherent to serializing a big cold prefill and motivates the compute-side lever (context-parallel
prefill) for a full fix. Off the fixed goodput@SLO (TTFT) metric, but the same mechanism the paper diagnoses.
"""
import re, statistics as st
from datetime import datetime
LOG="runs/v0-stock/server.log"; FMT="%Y-%m-%d %H:%M:%S"

def main():
    rows=[]
    for l in open(LOG,errors="ignore"):
        m=re.search(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)",l)
        if not m: continue
        t=datetime.strptime(m.group(1),FMT)
        if "Decode batch" in l: rows.append((t,"dec"))
        elif "Prefill batch" in l:
            nt=re.search(r"#new-token: (\d+)",l); ct=re.search(r"#cached-token: (\d+)",l)
            rows.append((t,"prebig" if (nt and ct and int(nt.group(1))>=6144 and int(ct.group(1))==0) else "pre"))
    with_big=[]; without=[]; last=None; big=0
    for t,typ in rows:
        if typ=="prebig": big+=1
        if typ=="dec":
            if last is not None: (with_big if big>0 else without).append((t-last).total_seconds())
            last=t; big=0
    def s(v,n):
        v=sorted(v); k=len(v)
        print(f"{n}: n={k} mean={st.mean(v):.1f}s p50={v[k//2]:.0f}s p90={v[int(.9*k)]:.0f}s max={max(v):.0f}s")
    print("Decode inter-batch gap (how long running requests wait between tokens):")
    s(with_big,"  WITH big-cold-doc prefill chunk in the gap")
    s(without, "  no big prefill in the gap                 ")
    print(f"=> ~{st.mean(with_big)/st.mean(without):.1f}x longer decode stall during big-cold-doc prefills")
    print("   => decode-side head-of-line: one mechanism damages BOTH TTFT (waiting turns) and E2E (running decodes).")

if __name__=="__main__": main()
