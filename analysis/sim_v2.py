#!/usr/bin/env python3
"""Calibrated offline KV-cache simulator (v2).

Fix vs v1: compute hit_rate exactly as eval.sh does (cached_tokens/prompt_tokens
over the FULL multiturn prompts), and CALIBRATE effective capacity so stock LRU
reproduces the measured baseline hit rate (0.6217). Only then compare policies.

Arrival model: conversations arrive Poisson(lam) in file order; a conv's turn t
arrives after turn t-1's E2E (prefill of new tokens + decode of output). The
global event stream interleaves all convs. Cache is LRU over an EFFECTIVE
capacity (CAP_eff), which is < physical L1+L2 because running-request KV and
fragmentation reduce usable prefix-cache space; we sweep CAP_eff to calibrate.

hit_rate = sum_over_turns(matched_prior_prefix) / sum_over_turns(full_prompt).
"""
import json, ast, random, collections, statistics, sys

MIX="/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"
CH=4.0
random.seed(7)

def build_convs():
    recs=[json.loads(l) for l in open(MIX)]
    convs=[]
    for r in recs:
        qp=r.get("qa_pairs",""); doc=r.get("input","")
        if qp=="none" or (isinstance(qp,(list,str)) and len(qp)==0):
            turns=[("Input: "+doc+" Question: Please summarize the input", doc[:1024])]
        else:
            try: qa=ast.literal_eval(qp) if isinstance(qp,str) else qp
            except Exception: continue
            turns=[]
            for i,q in enumerate(qa):
                if i==0: turns.append(("Input: "+doc+" Question: "+str(q.get("Q","")),str(q.get("A",""))))
                else: turns.append((str(q.get("Q","")),str(q.get("A",""))))
        pt=[max(1,int(len(p)/CH)) for p,a in turns]
        ot=[max(1,int(len(a)/CH)) for p,a in turns]
        src=r.get("title","").split("_")[0]
        src="sharegpt" if src=="sharegpt" else ("leval" if src=="leval" else "loogle")
        convs.append(dict(pt=pt,ot=ot,src=src,n=len(turns)))
    return convs

def build_stream(convs, lam, base_gap=3.0, dec=0.08, pref=0.0005):
    ev=[]; t=0.0
    for cid,c in enumerate(convs):
        t+=random.expovariate(lam); clock=t
        committed=0
        for ti in range(c["n"]):
            newq = c["pt"][ti] if ti==0 else c["pt"][ti]
            ev.append((clock,cid,ti))
            # E2E of this turn ~ prefill(new tokens)+decode(output)
            newtok = c["pt"][ti] if ti==0 else c["pt"][ti]
            clock += base_gap + newtok*pref + c["ot"][ti]*dec
            committed += newq + c["ot"][ti]
    ev.sort()
    return ev

class LRU:
    def __init__(s,cap): s.cap=cap; s.od=collections.OrderedDict(); s.used=0
    def has(s,k): return k in s.od
    def add(s,k,size):
        if k in s.od: s.used-=s.od[k]; s.od.pop(k)
        s.od[k]=size; s.od.move_to_end(k); s.used+=size
        while s.used>s.cap and s.od:
            kk,ss=s.od.popitem(last=False); s.used-=ss
    def rm(s,k):
        if k in s.od: s.used-=s.od.pop(k)

def simulate(convs, ev, cap_eff, policy="stock", prob_frac=0.08, size_gate=None):
    """policy: stock | gated | sizegate.
    gated: turn-0 goes to probation (prob_frac*cap); promoted to main on turn>=1.
    sizegate: like gated but ONLY large turn-0 prefixes (> size_gate tok) are gated;
              small ones admit straight to main (cheap, low opportunity cost)."""
    main=LRU(int(cap_eff*(1-prob_frac)) if policy!="stock" else cap_eff)
    prob=LRU(int(cap_eff*prob_frac)) if policy!="stock" else None
    depth={}; protected=set()
    tot_prompt=0; cached=0
    # committed prefix len per conv path
    committed={}
    for (at,cid,ti) in ev:
        c=convs[cid]
        prior=committed.get(cid,0)
        newq=c["pt"][ti]
        full_prompt = newq if ti==0 else prior+newq
        tot_prompt += full_prompt
        resident = (main.has(cid) or (prob is not None and prob.has(cid)))
        avail = depth.get(cid,0) if resident else 0
        matched=min(prior,avail)
        cached += matched
        # commit
        newdepth = prior+newq+c["ot"][ti]
        committed[cid]=newdepth; depth[cid]=newdepth
        if policy=="stock":
            main.add(cid,newdepth)
        else:
            gate_this = True
            if policy=="sizegate" and ti==0 and size_gate is not None:
                gate_this = (newdepth > size_gate)  # only gate big cold docs
            if cid in protected or not gate_this:
                protected.add(cid);
                if prob is not None: prob.rm(cid)
                main.add(cid,newdepth)
            elif ti>=1:
                protected.add(cid); prob.rm(cid); main.add(cid,newdepth)
            else:
                prob.add(cid,newdepth)
    return cached/tot_prompt

def main():
    convs=build_convs()
    ev=build_stream(convs, lam=3.0)
    print(f"convs={len(convs)} events={len(ev)}")
    phys=10_700_000
    print("\n=== calibrate stock LRU: sweep effective capacity ===")
    print(f"{'CAP_eff':>10}{'stock_hit':>11}")
    calib=None
    for capm in [1.0,1.5,2.0,2.5,3.0,4.0,5.0,7.0,10.7]:
        cap=int(capm*1e6)
        h=simulate(convs,ev,cap,"stock")
        print(f"{capm:>9.1f}M{h:>11.4f}")
        if calib is None and h>=0.6217: calib=cap
    # find CAP_eff whose stock hit ~ 0.6217
    best=None; bestd=9
    for capm in [x/10 for x in range(8,60)]:
        cap=int(capm*1e6); h=simulate(convs,ev,cap,"stock")
        if abs(h-0.6217)<bestd: bestd=abs(h-0.6217); best=(cap,h,capm)
    cap,h,capm=best
    print(f"\nCALIBRATED CAP_eff = {capm:.1f}M (stock hit={h:.4f} ~ baseline 0.6217)")
    print(f"(phys L1+L2=10.7M; effective is smaller due to running-KV pressure/frag)")
    print(f"\n=== policy comparison at CAP_eff={capm:.1f}M ===")
    print(f"{'policy':<22}{'hit_rate':>10}{'delta_pp':>10}")
    base=simulate(convs,ev,cap,"stock")
    print(f"{'stock':<22}{base:>10.4f}{0.0:>10.2f}")
    for pf in [0.05,0.08,0.15]:
        hh=simulate(convs,ev,cap,"gated",prob_frac=pf)
        print(f"{'gated@'+str(int(pf*100))+'%':<22}{hh:>10.4f}{(hh-base)*100:>+10.2f}")
    for sg in [8000,16000,32000]:
        hh=simulate(convs,ev,cap,"sizegate",prob_frac=0.08,size_gate=sg)
        print(f"{'sizegate>'+str(sg//1000)+'k':<22}{hh:>10.4f}{(hh-base)*100:>+10.2f}")
    # oracle
    orc=simulate(convs,ev,10**12,"stock")
    print(f"{'oracle(inf cache)':<22}{orc:>10.4f}{(orc-base)*100:>+10.2f}")

if __name__=="__main__": main()
