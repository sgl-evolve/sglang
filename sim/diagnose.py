#!/usr/bin/env python3
"""Diagnostics + mechanism screen on top of the validated 2-tier sim.

1. Recompute breakdown: cold (unavoidable: turn0 doc + each turn's new Q) vs
   capacity-recompute (evicted previously-cached content = the recoverable loss).
2. Per-turn-index hit rate (where does the loss concentrate?).
3. GANG admission: the server bounds the active conversation set to G convs
   (admit next gang only when current gang's convs finish). Models cache-pressure
   admission. Measures hit_rate vs G. Access order within a gang = round-robin waves.
"""
import json, os, argparse

TOK = os.path.join(os.path.dirname(__file__), "workload_tokens.json")

def cumfull(turns, ovh):
    cf, s = [], 0
    for (pl, ol) in turns:
        s += pl + ol + ovh
        cf.append(s)
    return cf

def build_accesses_gang(convs, gang):
    """Partition convs into gangs of size `gang` (file order). Access order:
    for each gang sequentially, round-robin waves within the gang."""
    acc = []
    for g0 in range(0, len(convs), gang):
        gids = list(range(g0, min(g0 + gang, len(convs))))
        maxt = max(len(convs[c]) for c in gids)
        for w in range(maxt):
            for c in gids:
                if w < len(convs[c]):
                    acc.append((c, w))
    return acc

def simulate(convs, device_cap, host_cap, page, ovh, gang=None):
    def pg(x): return (x // page) * page
    cf = [cumfull(t, ovh) for t in convs]
    acc = build_accesses_gang(convs, gang) if gang else _waves(convs)

    lru, resident, total = [], {}, 0
    tot_prompt = tot_hit = tot_dev = tot_host = 0
    cold_rc = cap_rc = 0
    # per-turn-index accounting
    from collections import defaultdict
    ti_prompt = defaultdict(int); ti_hit = defaultdict(int)

    for (cid, i) in acc:
        pl, ol = convs[cid][i]
        reuse_target = cf[cid][i-1] if i >= 1 else 0
        prompt = reuse_target + pl
        tot_prompt += prompt
        ti_prompt[i] += prompt

        res = resident.get(cid, 0)
        hit = min(res, pg(reuse_target))
        idx = lru.index(cid) if cid in resident else len(lru)
        above = sum(resident[x] for x in lru[:idx])
        dev_hit = max(0, min(above + hit, device_cap) - above)
        host_hit = hit - dev_hit
        tot_hit += hit; tot_dev += dev_hit; tot_host += host_hit
        ti_hit[i] += hit
        # recompute breakdown: reuse_target - hit is capacity-recompute (was cached before);
        # pl (new Q) is cold. turn0's reuse_target=0 so its pl is cold doc.
        cap_rc += pg(reuse_target) - hit
        cold_rc += pl + (reuse_target - pg(reuse_target))

        new_full = cf[cid][i]
        if cid in resident:
            lru.remove(cid)
        total += new_full - resident.get(cid, 0)
        resident[cid] = new_full
        lru.insert(0, cid)
        j = len(lru) - 1
        while total > host_cap and j >= 0:
            vic = lru[j]
            if vic == cid:
                j -= 1; continue
            free = min(resident[vic], total - host_cap)
            resident[vic] -= free; total -= free
            if resident[vic] <= 0:
                lru.pop(j); del resident[vic]
            j -= 1

    return {
        "hit_rate": tot_hit / tot_prompt,
        "device_frac": tot_dev / max(1, tot_hit),
        "prompt": tot_prompt, "hit": tot_hit,
        "cold_recompute": cold_rc, "capacity_recompute": cap_rc,
        "ti_hitrate": {k: ti_hit[k]/ti_prompt[k] for k in sorted(ti_prompt) if ti_prompt[k]},
        "ti_prompt": dict(ti_prompt),
    }

def _waves(convs):
    maxt = max(len(c) for c in convs)
    return [(c, w) for w in range(maxt) for c in range(len(convs)) if w < len(convs[c])]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-cap", type=int, default=2_350_000)
    ap.add_argument("--host-cap", type=int, default=8_400_000)
    ap.add_argument("--page", type=int, default=64)
    ap.add_argument("--ovh", type=int, default=4)
    args = ap.parse_args()
    convs = json.load(open(TOK))["convs"]

    print("=== BASELINE (all convs concurrent, wave order) ===")
    r = simulate(convs, args.device_cap, args.host_cap, args.page, args.ovh)
    print(f"hit_rate={r['hit_rate']:.4f} device_frac={r['device_frac']:.4f}")
    print(f"cold_recompute={r['cold_recompute']:,}  capacity_recompute(recoverable)={r['capacity_recompute']:,}")
    print(f"  prompt={r['prompt']:,}  -> cold-loss={r['cold_recompute']/r['prompt']:.3f} capacity-loss={r['capacity_recompute']/r['prompt']:.3f}")
    print("per-turn-index hit rate (turn: hit_rate, prompt_share):")
    tp = r['prompt']
    for k in sorted(r['ti_hitrate'])[:8]:
        print(f"    turn {k}: hit={r['ti_hitrate'][k]:.3f}  prompt_share={r['ti_prompt'][k]/tp:.3f}")

    print("\n=== GANG ADMISSION sweep (bound active conv set to G) ===")
    print(f"{'G':>6} {'hit_rate':>9} {'device_frac':>11} {'cap_recompute':>14}")
    for G in [1553, 1000, 700, 500, 300, 200, 100, 50, 25]:
        rg = simulate(convs, args.device_cap, args.host_cap, args.page, args.ovh, gang=G)
        print(f"{G:>6} {rg['hit_rate']:>9.4f} {rg['device_frac']:>11.4f} {rg['capacity_recompute']:>14,}")

if __name__ == "__main__":
    main()
