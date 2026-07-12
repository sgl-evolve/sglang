import json, statistics as st
DS="/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"
rows=[json.loads(l) for l in open(DS)]
def turns(data):
    # replicate sample_loogle_requests (multiturn, no shared_prefix)
    t=[]
    qp=data.get("qa_pairs")
    if not qp or qp=="none" or len(qp)==0:
        prompt="Input: "+data["input"]+" Question: Please summarize the input"
        ans=data["input"][:1024]
        t.append((prompt,ans))
    else:
        try: pairs=eval(qp)
        except Exception: pairs=[]
        for i,qa in enumerate(pairs):
            if i==0:
                t.append(("Input: "+data["input"]+" Question: "+qa["Q"], qa["A"]))
            else:
                t.append((qa["Q"], qa["A"]))
    return t
nturns=[]; doc_chars=[]; q_chars=[]; a_chars=[]
nocache_pref=[]; perfcache_pref=[]
CPT=3.6  # chars/token calibration (refine below)
for d in rows:
    ts=turns(d)
    nturns.append(len(ts))
    doc_chars.append(len(d["input"]))
    # accumulated prefix char lengths
    acc=0; nocache=0; perf=0
    for i,(p,a) in enumerate(ts):
        q_chars.append(len(p) if i>0 else len(p)-len(d["input"]))  # new question chars (turn0 excl doc)
        a_chars.append(len(a))
        prefix_before=acc                # chars reused from cache
        newtok=len(p)                    # chars to prefill this turn (the whole prompt p; but prefix_before is cached)
        # turn i prefill: prefix_before (cached) + (p minus overlap). For turn0 p includes doc; acc=0.
        # Model: reused = prefix_before ; new prefill = len(p) - (overlap). Approx overlap = prefix_before for turns>0? 
        # Simpler: cumulative context after turn i = acc + len(p) + len(a).
        nocache += acc + len(p)          # no-cache: reprefill full prefix each turn
        perf    += len(p)                # perfect-cache: only new prompt tokens (prefix served from cache)
        acc += len(p) + len(a)
    nocache_pref.append(nocache); perfcache_pref.append(perf)
tot_doc=sum(doc_chars); tot_nocache=sum(nocache_pref); tot_perf=sum(perfcache_pref)
print(f"conversations: {len(rows)}")
print(f"turns/conv: mean={st.mean(nturns):.2f} median={st.median(nturns)} max={max(nturns)} min={min(nturns)} ; #1-turn convs={sum(1 for n in nturns if n==1)}")
print(f"doc chars: mean={st.mean(doc_chars):,.0f} median={st.median(doc_chars):,.0f} p90={sorted(doc_chars)[int(.9*len(doc_chars))]:,} max={max(doc_chars):,}")
print(f"  ~doc tokens (chars/{CPT}): mean={st.mean(doc_chars)/CPT:,.0f} median={st.median(doc_chars)/CPT:,.0f}")
print(f"new-question chars: mean={st.mean(q_chars):,.0f} median={st.median(q_chars):,.0f}")
print(f"answer chars: mean={st.mean(a_chars):,.0f} median={st.median(a_chars):,.0f}")
print(f"TOTAL prefill chars  no-cache={tot_nocache/1e6:.1f}M  perfect-cache={tot_perf/1e6:.1f}M  reuse-savings={100*(1-tot_perf/tot_nocache):.1f}%")
print(f"  => ~tokens no-cache={tot_nocache/CPT/1e6:.1f}M  perfect-cache={tot_perf/CPT/1e6:.1f}M")
print(f"working set (sum unique ctx chars ~ sum(doc + all Q+A)) = approx perfect-cache total = {tot_perf/1e6:.1f}M chars ~ {tot_perf/CPT/1e6:.1f}M tok")
