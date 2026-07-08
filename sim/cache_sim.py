#!/usr/bin/env python3
"""Fast discrete-event simulator of the v0.25 2-tier HiCache under the fixed multiturn workload.

Locates the LOCALITY headroom cheaply (no GPU) and compares scheduling/admission policies on hit-rate.

Model (see report.md): each conversation is a LINEAR radix chain (cross-conv sharing ~=0); resident
state R_c = #resident prefix tokens (contiguous from 0). LRU (per-conv uniform access time) trims the
least-recently-served conv's TAIL first (radix leaf-first). Turn t of conv c: cumulative prefix P; new
user n_t; output o_t. hit=min(R_c,P); prefill_miss=(P-hit)+n_t; after run R_c=P+n_t+o_t (write_through).
Serving locks a conv (not evictable). Capacity C tokens. Server: M slots; service=A*prefill+B*decode.
New convs arrive Poisson(lambda); a conv's next turn re-enqueues on completion. Policies pick the next
ready turn when a slot frees; admission cap bounds the distinct active (cache-holding) conv set.
"""
import json, heapq, os, argparse
from collections import OrderedDict, deque
import numpy as np

WL = os.path.join(os.path.dirname(__file__), "workload_tokens.json")

def load_workload():
    d = json.load(open(WL))
    convs = []
    for turns in d["conversations"]:
        P, meta, cum = [], [], 0
        for t in turns:
            P.append(cum); meta.append((t["new_prompt_tok"], t["output_tok"]))
            cum += t["new_prompt_tok"] + t["output_tok"]
        convs.append((P, meta))
    return convs, d["meta"]

def simulate(convs, policy="fcfs", C=10_200_000, M=128, lam=3.0,
             A=0.0003, B=0.010, admit_cap=None, seed=0):
    rng = np.random.default_rng(seed)
    N = len(convs)
    arriv = np.cumsum(rng.exponential(1.0/lam, size=N))
    R = [0]*N
    turn_idx = [0]*N
    nturns = [len(c[0]) for c in convs]
    lru = OrderedDict()        # conv -> True, ordered LRU-first ... last = most-recently-served
    serving = set()
    resident_sum = 0
    # true active working-set size = len(lru) + len(serving)  (disjoint: serving convs are del'd from lru)

    evq = [(float(arriv[c]), 0, 'a', c) for c in range(N)]
    heapq.heapify(evq)
    ready = deque()            # fcfs order; for lpm we scan it
    ready_since = {}
    hit_tokens = prefill_tokens = total_prompt = 0
    ttft = []
    act_hist = []
    ctr = 0
    now = 0.0
    finished = 0
    busy = 0

    def in_ws(c):
        return R[c] > 0 or c in serving
    def ws_size():
        return len(lru) + len(serving)

    def evict(needed, protect):
        nonlocal resident_sum
        while resident_sum + needed > C and lru:
            c, _ = next(iter(lru.items()))
            if c == protect or c in serving:
                lru.move_to_end(c)     # can't evict; rotate (rare)
                # guard: if everything is protected, stop
                if all((cc == protect or cc in serving) for cc in lru):
                    break
                continue
            deficit = (resident_sum + needed) - C
            if R[c] <= deficit:
                resident_sum -= R[c]; R[c] = 0
                del lru[c]
            else:
                R[c] -= deficit; resident_sum -= deficit
        # active_count recomputed lazily where needed

    def pick_ready():
        """Return index in `ready` to schedule next, honoring policy + admission. None if blocked."""
        if not ready:
            return None
        if policy == "fcfs":
            if admit_cap is None:
                return 0
            for i, c in enumerate(ready):
                if in_ws(c) or ws_size() < admit_cap:
                    return i
            return None
        # lpm: max cached-prefix (min miss) first, among admissible
        best_i, best_hit = None, -1
        for i, c in enumerate(ready):
            if admit_cap is not None and not (in_ws(c) or ws_size() < admit_cap):
                continue
            t = turn_idx[c]
            h = R[c] if R[c] < convs[c][0][t] else convs[c][0][t]
            if h > best_hit:
                best_hit = h; best_i = i
        return best_i

    while finished < N:
        # fill free slots
        while busy < M:
            i = pick_ready()
            if i is None:
                break
            c = ready[i]; del ready[i]
            t = turn_idx[c]
            P = convs[c][0][t]; n_t, o_t = convs[c][1][t]
            hit = R[c] if R[c] < P else P
            miss = (P - hit) + n_t
            new_total = P + n_t + o_t
            grow = new_total - R[c]
            if grow > 0:
                evict(grow, protect=c)
            hit_tokens += hit; prefill_tokens += miss; total_prompt += (P + n_t)
            ttft.append((now - ready_since.get(c, now)) + A*miss)
            resident_sum += (new_total - R[c]); R[c] = new_total
            if c in lru: del lru[c]
            serving.add(c); busy += 1
            act_hist.append(ws_size())
            heapq.heappush(evq, (now + A*miss + B*o_t, ctr, 'f', c)); ctr += 1
        if not evq:
            break
        now, _, kind, c = heapq.heappop(evq)
        if kind == 'a':
            ready.append(c); ready_since[c] = now
        else:
            serving.discard(c); busy -= 1
            # returns to cache (write_through), becomes MRU; if R==0 it simply isn't in the working set
            if R[c] > 0:
                lru[c] = True; lru.move_to_end(c)
            turn_idx[c] += 1
            if turn_idx[c] < nturns[c]:
                ready.append(c); ready_since[c] = now
            else:
                finished += 1   # done convs keep residual cache in lru until evicted (dead weight)
    hr = hit_tokens / max(1, total_prompt)
    tp = np.array(ttft) if ttft else np.array([0.0])
    return {"policy": policy, "admit_cap": admit_cap, "hit_rate": round(hr,4),
            "prefill_tok": int(prefill_tokens), "total_prompt_tok": int(total_prompt),
            "makespan_s": round(now,1), "ttft_mean": round(float(tp.mean()),3),
            "ttft_p99": round(float(np.percentile(tp,99)),3),
            "mean_active": round(float(np.mean(act_hist)),1) if act_hist else 0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=10_200_000)
    ap.add_argument("--A", type=float, default=0.0003)
    ap.add_argument("--B", type=float, default=0.010)
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--only", default="")   # comma list to run subset
    args = ap.parse_args()
    convs, meta = load_workload()
    print("workload:", meta)
    print(f"params: C={args.C} A={args.A} B={args.B} M={args.M}\n")
    runs = [("fcfs", None, "fcfs"), ("lpm", None, "lpm"),
            ("admit40", 40, "fcfs"), ("admit25", 25, "fcfs"), ("admit15", 15, "fcfs"),
            ("lpm+ad40", 40, "lpm"), ("lpm+ad25", 25, "lpm"), ("lpm+ad15", 15, "lpm")]
    if args.only:
        keep = set(args.only.split(","))
        runs = [r for r in runs if r[0] in keep]
    import time
    print(f"{'run':<12}{'hit_rate':>9}{'makespan':>10}{'ttft_mean':>10}{'ttft_p99':>10}{'act':>7}{'sec':>7}")
    for name, cap, pol in runs:
        t0=time.time()
        r = simulate(convs, policy=pol, C=args.C, M=args.M, A=args.A, B=args.B, admit_cap=cap)
        print(f"{name:<12}{r['hit_rate']:>9}{r['makespan_s']:>10}{r['ttft_mean']:>10}{r['ttft_p99']:>10}{r['mean_active']:>7}{time.time()-t0:>7.1f}")

if __name__ == "__main__":
    main()
