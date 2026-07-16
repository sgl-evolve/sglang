#!/usr/bin/env python3
"""Direct log evidence for P6's mechanism claim ('bigger chunk -> fewer prefill STEPS'). Parses each run's
server.log 'Prefill batch ... #new-token: N' lines and reports, per config: total prefill steps for the
identical 7037-req workload, the fraction of steps that fired the boost (#new-token > chunked_prefill_size
6144), and the max chunk. ACCEL should show a mode at 2x6144=12288 and fewer total steps; STOCK caps at 6144
and never fires. No GPU; reads existing runs/*/server.log."""
import re, statistics
CPS = 6144  # frozen chunked_prefill_size
def analyze(vers):
    rows=[]
    for v in vers:
        toks=[]
        try:
            for line in open(f'runs/{v}/server.log'):
                m=re.search(r'Prefill batch, #new-seq: \d+, #new-token: (\d+)', line)
                if m: toks.append(int(m.group(1)))
        except FileNotFoundError: continue
        if not toks: continue
        n=len(toks); big=sum(1 for t in toks if t>CPS)
        rows.append(dict(v=v, steps=n, big=big, fracbig=100*big/n, maxtok=max(toks)))
    return rows
ACCEL=['v18_faccel1','v18_faccel2','v18_faccel3','v10_accel','v10b_accel','v10c_accel','v10d_accel']
STOCK=['v18_fstock1','v18_fstock2','v18_fstock3','v9_stock2','v11_stock3','v12_stock4']
for name,vers in [('ACCEL',ACCEL),('STOCK',STOCK)]:
    r=analyze(vers)
    if not r: continue
    st=[x['steps'] for x in r]; bs=[x['big'] for x in r]; fb=[x['fracbig'] for x in r]; mx=[x['maxtok'] for x in r]
    print(f"\n{name} (n={len(r)} runs):")
    for x in r: print(f"  {x['v']:14} steps={x['steps']:6d}  boost-fired={x['big']:5d} ({x['fracbig']:4.1f}%)  max_chunk={x['maxtok']}")
    print(f"  -> median steps {int(statistics.median(st))} | boost-fired {statistics.median(fb):.1f}% | max_chunk median {int(statistics.median(mx))}")
