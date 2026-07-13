#!/usr/bin/env python3
"""First-order discrete-event sim of the v0.31 closed-loop multiturn workload.

Goal: test the DISPLACEMENT hypothesis and measure Belady (OPT) headroom over LRU in the CONCURRENT
regime. The protocol claims "LRU~=Belady, eviction is dead" -- that is for IN-ORDER reuse; here reuse
is out-of-order across the think-gap, so LRU may evict soon-reused conversation prefixes. Relative
policy comparison + headroom bound only; the GPU eval is the real proof.

RADIX-FAITHFUL cache model: a conversation's resident KV is a CONTIGUOUS prefix depth d (runs 0..d-1
resident); eviction is leaf-first (drop a conv's DEEPEST resident run). A turn t needs ancestors 0..t-1
resident; missing runs [d..t-1] are recomputed (avoidable) + its own input (always). After serving, the
conv's depth becomes t+1. node/run size = that turn's (in_tok + out_tok).

Two analyses:
  1) timing sim (LRU vs CONT): FIFO prefill server at P tok/s, decode at D tok/s, Poisson(lambda) arrivals,
     concurrency cap. Gives realizable recompute + TTFT for LRU and the continuation-aware policy CONT
     (protect a conv's prefix from eviction for GRACE s after its latest turn completes).
  2) headroom (LRU vs OPT): replay the LRU run's fixed access order through a pure Belady oracle
     (evict the conv whose next access is farthest) -> optimal recompute for that trace.
"""
import json, os, heapq, random
from collections import deque, defaultdict

TRACE = os.path.join(os.path.dirname(__file__), "conv_trace.json")
def load(): return json.load(open(TRACE))

# ----- timing sim (LRU / CONT) -----
class Sim:
    def __init__(self, convs, lam, cap, policy, P, D, grace=0.0, maxc=256, seed=1, record=False):
        self.convs=convs; self.lam=lam; self.cap=cap; self.policy=policy
        self.P=P; self.D=D; self.grace=grace; self.maxc=maxc; self.rng=random.Random(seed)
        self.record=record; self.access_order=[]
        # per-conv resident depth + bookkeeping
        self.depth=defaultdict(int)          # ci -> resident contiguous depth
        self.last=defaultdict(float)         # ci -> last access wall (for lru)
        self.protect=defaultdict(float)      # ci -> protection deadline wall (cont)
        self.used=0
        self.n_active=0; self.prefill_free_at=0.0
        self.ttfts=[]; self.prefill_tok=[]; self.recompute=0
    def run_size(self, ci, k):
        i,o=self.convs[ci][k]; return i+o
    def conv_resident_tokens(self, ci):
        return sum(self.run_size(ci,k) for k in range(self.depth[ci]))
    def evict(self, need, wall):
        # drop deepest resident runs of victim convs until `need` fits
        while self.used + need > self.cap:
            # choose victim conv with a resident run to drop
            victim=None; best=None
            for ci,d in self.depth.items():
                if d<=0: continue
                if self.policy=="lru":
                    sc=self.last[ci]
                else:  # cont: protected convs sorted after unprotected
                    prot = self.protect[ci] > wall
                    sc = self.last[ci] + (1e18 if prot else 0.0)
                if best is None or sc<best:
                    best=sc; victim=ci
            if victim is None: break
            d=self.depth[victim]
            self.used -= self.run_size(victim, d-1)
            self.depth[victim]=d-1
    def serve(self, ci, t, wall):
        # recompute missing prefix runs [depth..t-1]
        d=self.depth[ci]
        miss=0
        if d < t:
            for k in range(d, t):
                miss += self.run_size(ci,k)
        uncached = self.convs[ci][t][0] + miss   # own input + missing prefix
        self.recompute += miss
        self.prefill_tok.append(uncached)
        return uncached
    def materialize(self, ci, t, wall):
        # after serving turn t, runs 0..t are resident (depth t+1); add newly-resident tokens
        newdepth=t+1
        add=0
        for k in range(self.depth[ci], newdepth):
            add += self.run_size(ci,k)
        if add>0: self.evict(add, wall)
        self.used += add; self.depth[ci]=newdepth
        self.last[ci]=wall
        if self.policy=="cont": self.protect[ci]=wall+self.grace
    def run(self):
        convs=self.convs; evt=[]; seqc=0
        def push(w,k,d):
            nonlocal seqc; heapq.heappush(evt,(w,seqc,k,d)); seqc+=1
        queue=deque((ci,0) for ci in range(len(convs)))
        pslot=deque(); admit_t={}; pull_waiting=[False]
        def start_slots(wall):
            while pslot and self.n_active<self.maxc:
                ci,t=pslot.popleft()
                if self.record: self.access_order.append((ci,t))
                self.last[ci]=wall
                uncached=self.serve(ci,t,wall)
                start=max(wall,self.prefill_free_at); pf=uncached/self.P
                self.prefill_free_at=start+pf
                self.ttfts.append((start-admit_t[(ci,t)])+pf)
                self.n_active+=1
                push(self.prefill_free_at + convs[ci][t][1]/self.D, "done", (ci,t))
        push(0.0,"pull",None)
        while evt:
            wall,_,kind,data=heapq.heappop(evt)
            if kind=="pull":
                if queue:
                    r=queue.popleft(); admit_t[r]=wall; pslot.append(r); start_slots(wall)
                    push(wall+self.rng.expovariate(self.lam),"pull",None)
                else:
                    pull_waiting[0]=True
            else:
                ci,t=data; self.n_active-=1
                self.materialize(ci,t,wall)
                if t+1 < len(convs[ci]):
                    queue.append((ci,t+1))
                    if pull_waiting[0]: pull_waiting[0]=False; push(wall,"pull",None)
                start_slots(wall)
        self.ttfts.sort()
        p=lambda q: self.ttfts[min(len(self.ttfts)-1,int(len(self.ttfts)*q))] if self.ttfts else 0
        return {"policy":self.policy,"lam":self.lam,"recompute":self.recompute,
                "prefill":sum(self.prefill_tok),"rc%":100*self.recompute/max(1,sum(self.prefill_tok)),
                "p50":p(.5),"p90":p(.9),"p99":p(.99),"makespan":self.prefill_free_at}

# ----- fixed-trace Belady headroom (LRU vs OPT over the SAME access order) -----
def trace_headroom(access_order, convs, cap):
    def rs(ci,k): i,o=convs[ci][k]; return i+o
    # next-use position per (ci, needed-depth up to t) -> we need next access positions per conv
    nextpos=defaultdict(deque)
    for p,(ci,t) in enumerate(access_order):
        nextpos[ci].append(p)
    # W = position-window proxy for CAR's wall-time completion grace (a turn-0 is "on probation" if it was
    # last touched within W accesses = recently completed, so its first reuse is likely still near).
    W=int(os.environ.get("SIM_CARW", 300))
    def run(policy):
        depth=defaultdict(int); used=0; last=defaultdict(int); recompute=0; proven=defaultdict(bool)
        served=defaultdict(int); first_seen=defaultdict(lambda: 10**12)
        # per-conv future accesses pointer
        fut={ci:deque(v) for ci,v in nextpos.items()}
        for p,(ci,t) in enumerate(access_order):
            if fut[ci] and fut[ci][0]==p: fut[ci].popleft()
            if t>=1: proven[ci]=True   # served a later turn => proven multi-turn conv (hit_count>=1)
            served[ci]+=1
            if p<first_seen[ci]: first_seen[ci]=p
            d=depth[ci]
            if d<t:
                recompute += sum(rs(ci,k) for k in range(d,t))
            # materialize to t+1
            add=sum(rs(ci,k) for k in range(depth[ci], t+1))
            while used+add>cap:
                victim=None; best=None
                for cj,dd in depth.items():
                    if dd<=0 or cj==ci: continue
                    if policy=="lru": sc=(last[cj],)
                    elif policy=="slru":   # protect proven (hit_count>=1); unproven evicted first, then LRU
                        sc=(1 if proven[cj] else 0, last[cj])
                    elif policy=="car":    # 3-seg: 0=unproven&stale(dead whale) 1=unproven&recent(turn-0 probation) 2=proven
                        seg = 2 if proven[cj] else (1 if (p-last[cj])<=W else 0)
                        sc=(seg, last[cj])
                    elif policy=="whale":  # evict BIGGEST unproven first (size-aware liveness); protect proven, LRU tiebreak
                        size=sum(rs(cj,k) for k in range(depth[cj]))
                        sc=(1 if proven[cj] else 0, -size, last[cj])
                    elif policy=="lfu":    # least-frequently-served first, LRU tiebreak
                        sc=(served[cj], last[cj])
                    elif policy=="fifo":   # oldest-created (first seen) first
                        sc=(first_seen[cj],)
                    elif policy=="mru":    # most-recently-used first (adversarial control)
                        sc=(-last[cj],)
                    elif policy=="size_only":  # evict BIGGEST regardless of proven (no proven protection)
                        sc=(-sum(rs(cj,k) for k in range(depth[cj])), last[cj])
                    elif policy=="cost_aware":  # recompute-cost-aware: evict CHEAPEST-to-rebuild (smallest) first
                        sc=(sum(rs(cj,k) for k in range(depth[cj])), last[cj])
                    elif policy=="cost_aware_pp":  # cost-aware WITH proven-protection (protect proven, then cheapest unproven)
                        sc=(1 if proven[cj] else 0, sum(rs(cj,k) for k in range(depth[cj])), last[cj])
                    elif policy=="hit_density":  # evict lowest hits-per-byte (served/size) first — GDSF-family value density
                        size=max(1,sum(rs(cj,k) for k in range(depth[cj])))
                        sc=(served[cj]/size, last[cj])
                    elif policy=="gdsf":  # greedy-dual-size-frequency; cost=size => priority = served + recency-aging
                        sc=(served[cj], last[cj])  # (cost/size=1 so reduces to freq+recency; == LFU here, recorded for completeness)
                    else:  # opt: evict conv whose NEXT access is farthest (or none)
                        sc = (-(fut[cj][0] if fut[cj] else 10**12),)
                    if best is None or sc<best: best=sc; victim=cj
                if victim is None: break
                dd=depth[victim]; used-=rs(victim,dd-1); depth[victim]=dd-1
            used+=add; depth[ci]=t+1; last[ci]=p
        return recompute
    return {k:run(k) for k in ("lru","slru","car","whale","lfu","fifo","mru","size_only","cost_aware","cost_aware_pp","hit_density","gdsf","opt")}

if __name__=="__main__":
    convs=load()
    cap=float(os.environ.get("SIM_CAP",10.7e6))
    P=float(os.environ.get("SIM_P",25000)); D=float(os.environ.get("SIM_D",4000))
    grace=float(os.environ.get("SIM_GRACE",20.0))
    print(f"convs={len(convs)} cap={cap:.3e} P={P} D={D} grace={grace}s")
    print(f"{'lam':>4} {'policy':>6} {'recompute':>13} {'prefill':>13} {'rc%':>6} {'p50':>8} {'p90':>9} {'p99':>10}")
    for lam in [3,5,7,10]:
        base=Sim(convs,lam,cap,"lru",P,D,record=True); rb=base.run()
        rc=Sim(convs,lam,cap,"cont",P,D,grace=grace).run()
        for r in (rb,rc):
            print(f"{lam:>4} {r['policy']:>6} {r['recompute']:>13,} {r['prefill']:>13,} {r['rc%']:>5.1f} "
                  f"{r['p50']:>8.2f} {r['p90']:>9.2f} {r['p99']:>10.2f}")
        hr=trace_headroom(base.access_order, convs, cap)
        red=100*(hr['lru']-hr['opt'])/max(1,hr['lru'])
        # timing-independent victim-choice replay (same access order, different eviction policy):
        d=lambda k: 100*(hr[k]-hr['lru'])/max(1,hr['lru'])  # % recompute vs LRU (+=worse, -=better)
        print(f"     HEADROOM(replay, same order): lru {hr['lru']:,} | slru {hr['slru']:,} ({d('slru'):+.1f}%) "
              f"| car {hr['car']:,} ({d('car'):+.1f}%) | opt {hr['opt']:,} ({-red:.1f}% = Belady headroom)")
