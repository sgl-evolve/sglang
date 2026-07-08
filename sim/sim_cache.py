#!/usr/bin/env python3
"""Offline 2-tier HiCache simulator for the v0.25 multi-turn workload.

Faithfully models:
 - Access order: bench_serving FIFO re-enqueue == round-robin WAVES
   (wave w = turn w of every conv with > w turns, in file order). This is EXACT
   given completions are ~FIFO under the concurrency semaphore.
 - Radix prefix reuse: turn i reuses the conv's cumulative content through turn i-1
   (doc + all prior Q/A); cross-conv sharing = 0 (distinct docs). Eviction is
   leaf-first within a conv (radix), so a conv keeps its ROOT (doc) longest.
 - 2-tier LRU with write_through: distinct-resident bound = host_cap; device_cap is
   an inclusive hot subset (device hits are free; host hits cost a load_back H->D).
   A page is a device hit iff its global LRU rank < device_cap; host hit iff in
   [device_cap, host_cap); recompute iff >= host_cap.

Reports hit_rate, device/host split, load_back / recompute / evict tokens, host_util,
plus the infinite-cache ceiling. Validate against baseline.json (0.6217 / 0.40 / 0.60).

Usage: sim_cache.py [--device-cap N] [--host-cap N] [--page 64] [--ovh 4] [--policy lru]
"""
import json, os, argparse

TOK = os.path.join(os.path.dirname(__file__), "workload_tokens.json")

def build_accesses(convs):
    """Return list of (conv_id, turn_idx) in round-robin wave order."""
    maxturns = max(len(c) for c in convs)
    acc = []
    for w in range(maxturns):
        for cid, turns in enumerate(convs):
            if w < len(turns):
                acc.append((cid, w))
    return acc

def cumfull(turns, ovh):
    """cumulative content tokens after each turn: cf[i] = sum_{k<=i}(pl_k+ol_k+ovh)."""
    cf = []
    s = 0
    for (pl, ol) in turns:
        s += pl + ol + ovh
        cf.append(s)
    return cf

def simulate(convs, device_cap, host_cap, page, ovh, policy="lru", protect_docs=0):
    """policy: 'lru' baseline. protect_docs: if >0, pin each conv's first `protect_docs`
    tokens (its document root) so it is never evicted from host (mechanism screen)."""
    def pg(x):  # floor to page granularity for prefix matching
        return (x // page) * page
    cf = [cumfull(t, ovh) for t in convs]
    acc = build_accesses(convs)

    # LRU list of conv ids, MRU first. resident[cid] = resident root tokens.
    lru = []          # list of cids, index 0 = MRU
    pos = {}          # cid -> index in lru (kept in sync lazily)
    resident = {}     # cid -> resident root tokens (contiguous from root)
    pinned = {}       # cid -> pinned root tokens (never evicted), for protect_docs
    total = 0

    tot_prompt = tot_hit = tot_dev = tot_host = tot_recompute = 0
    tot_loadback = tot_evict = 0
    # ceiling (infinite host): hit = reuse_target (all prior content always resident)
    ceil_hit = ceil_prompt = 0

    for (cid, i) in acc:
        turns = convs[cid]
        pl, ol = turns[i]
        reuse_target = cf[cid][i-1] if i >= 1 else 0
        prompt = reuse_target + pl
        tot_prompt += prompt
        ceil_prompt += prompt
        ceil_hit += pg(reuse_target)

        res = resident.get(cid, 0)
        hit = min(res, pg(reuse_target))
        # tier split: sum of resident of convs MORE recent than cid
        if cid in pos:
            idx = lru.index(cid)
            above = sum(resident[x] for x in lru[:idx])
        else:
            above = total  # new conv: sits below everything currently resident
        dev_hit = max(0, min(above + hit, device_cap) - above)
        host_hit = hit - dev_hit
        recompute = pg(reuse_target) - hit  # evicted middle/leaf must be recomputed

        tot_hit += hit
        tot_dev += dev_hit
        tot_host += host_hit
        tot_loadback += host_hit
        tot_recompute += recompute

        # after serving: conv fully resident at cumfull[i]; move to MRU
        new_full = cf[cid][i]
        if cid in pos:
            lru.remove(cid)
        old_res = resident.get(cid, 0)
        total += new_full - old_res
        resident[cid] = new_full
        if protect_docs:
            pinned[cid] = min(protect_docs, new_full)
        lru.insert(0, cid)
        pos[cid] = 0

        # evict from LRU tail until total <= host_cap
        j = len(lru) - 1
        while total > host_cap and j >= 0:
            vic = lru[j]
            if vic == cid:
                j -= 1
                continue
            floor = pinned.get(vic, 0)
            can_free = resident[vic] - floor
            if can_free <= 0:
                j -= 1
                continue
            need = total - host_cap
            free = min(can_free, need)
            resident[vic] -= free
            total -= free
            tot_evict += free
            if resident[vic] <= floor and floor == 0:
                lru.pop(j)
                del resident[vic]
                del pos[vic]
            j -= 1
        # rebuild pos map lazily (indices shifted); cheap enough
        pos = {c: k for k, c in enumerate(lru)}

    return {
        "hit_rate": tot_hit / tot_prompt,
        "device_frac": tot_dev / max(1, tot_hit),
        "host_frac": tot_host / max(1, tot_hit),
        "prompt_tokens": tot_prompt,
        "hit_tokens": tot_hit,
        "loadback_tokens": tot_loadback,
        "recompute_tokens": tot_recompute,
        "evict_tokens": tot_evict,
        "ceiling_hit_rate": ceil_hit / ceil_prompt,
        "final_resident": total,
        "host_util": min(1.0, total / host_cap),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-cap", type=int, default=2_350_000)
    ap.add_argument("--host-cap", type=int, default=7_812_544)  # baseline host_total_tokens
    ap.add_argument("--page", type=int, default=64)
    ap.add_argument("--ovh", type=int, default=4)
    ap.add_argument("--protect-docs", type=int, default=0)
    args = ap.parse_args()
    convs = json.load(open(TOK))["convs"]
    n_turns = sum(len(c) for c in convs)
    print(f"convs={len(convs)} turns={n_turns} device_cap={args.device_cap} host_cap={args.host_cap} ovh={args.ovh} protect_docs={args.protect_docs}")
    r = simulate(convs, args.device_cap, args.host_cap, args.page, args.ovh, protect_docs=args.protect_docs)
    for k, v in r.items():
        if isinstance(v, float):
            print(f"  {k:22s} {v:.4f}")
        else:
            print(f"  {k:22s} {v:,}")
    print("\nbaseline.json: hit_rate 0.6217 device_frac 0.4017 host_frac 0.5983 host_util 0.9999")

if __name__ == "__main__":
    main()
