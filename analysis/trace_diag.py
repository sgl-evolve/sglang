#!/usr/bin/env python3
"""floyd trace-driven diagnostic (CPU-only, login-node OK).

Quantifies the *cache-adjusted prefill cost* distribution of the FIXED eval
workload under an idealized LRU radix cache of the real 2-tier capacity.

Goal: motivate whether the goodput@SLO tail is dominated by a small number of
high-cost cold/evicted prefills (=> a prefill-cost-aware control lever exists),
or is uniform (=> no scheduling lever).

Token counts use chars/CHARS_PER_TOK as a fast proxy (labelled approximate).
The SHAPE of the distribution — not absolute tokens — is what we need here.
"""
import ast, json, sys
from collections import OrderedDict

TRACE = "/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"
CHARS_PER_TOK = 4.0
# 2-tier capacity: L1 (GPU) + L2 (768GB host). ~10.7M tokens per program.md.
CAP = 10_700_000

def toks(s):
    return max(1, int(len(s) / CHARS_PER_TOK))

def build_conversations(path):
    """Reproduce the loogle multiturn loader: turn0 = 'Input: {doc} Question: {Q0}',
    turns i>=1 = '{Qi}'. context grows with full history (Q+A each turn)."""
    convs = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            doc = r.get("input", "")
            try:
                qa = ast.literal_eval(r["qa_pairs"]) if r.get("qa_pairs") else []
            except Exception:
                qa = []
            if not qa:
                continue
            turns = []
            for i, t in enumerate(qa):
                if not isinstance(t, dict):
                    continue
                Q = str(t.get("Q", "")); A = str(t.get("A", ""))
                if i == 0:
                    user = "Input: " + doc + " Question: " + Q
                else:
                    user = Q
                turns.append((toks(user), toks(A)))
            if turns:
                convs.append(turns)
    return convs

def simulate(convs, cap):
    """Replay all turns in file order (approx arrival order at fixed rate).
    LRU cache keyed by conversation id, storing the cached prefix length (tokens).
    Per turn: context = sum of prior (in+out) + this input. cache-adjusted
    prefill work = context - cached_prefix (if resident) else context (full)."""
    cache = OrderedDict()  # conv_id -> cached_prefix_tokens (the KV we hold)
    resident_tokens = 0
    per_turn = []  # (conv_id, turn_idx, context_len, adjusted_work, kind)
    # emulate closed-loop-within-conv by interleaving: file order approximates
    # the sequence of *first* turns; but real turns interleave. For the cost
    # DISTRIBUTION, we replay each conversation's turns in order, and interleave
    # conversations round-robin-ish by arrival. Simplest faithful proxy: process
    # conversations in file order, all turns of a conv consecutively is WRONG
    # (would always hit). Instead we interleave: pop one turn from each active
    # conv in rounds, matching that turns are spread over time.
    active = [(cid, turns, 0, 0) for cid, turns in enumerate(convs)]  # cid,turns,turn_idx,ctx
    # arrival: a conversation becomes active in file order, one new conv per "tick";
    # each tick we also advance one pending turn of already-active convs.
    import collections
    pending = collections.deque()
    file_iter = iter(enumerate(convs))
    def evict_to_fit(need):
        nonlocal resident_tokens
        while resident_tokens + need > cap and cache:
            _, ev = cache.popitem(last=False)  # LRU
            resident_tokens -= ev
    done = False
    ctx_of = {}   # cid -> current context tokens (grows)
    idx_of = {}   # cid -> next turn index
    turns_of = {} # cid -> turns list
    while not done or pending:
        # admit one new conversation per tick (arrival)
        try:
            cid, turns = next(file_iter)
            ctx_of[cid] = 0; idx_of[cid] = 0; turns_of[cid] = turns
            pending.append(cid)
        except StopIteration:
            done = True
        if not pending:
            break
        # process the head conversation's next turn
        cid = pending.popleft()
        turns = turns_of[cid]; ti = idx_of[cid]
        in_tok, out_tok = turns[ti]
        context = ctx_of[cid] + in_tok
        cached = cache.get(cid, 0)
        if cid in cache:
            # resident: only the new suffix (context - cached) must be prefilled
            adj = max(0, context - cached)
            kind = "reuse" if cached >= ctx_of[cid] else "evicted_partial"
        else:
            adj = context  # full cold/evicted recompute
            kind = "cold" if ti == 0 else "evicted_full"
        per_turn.append((cid, ti, context, adj, kind))
        # after prefill, KV for full context+output is resident
        new_prefix = context + out_tok
        # update residency (evict others as needed)
        old = cache.pop(cid, 0)
        resident_tokens -= old
        evict_to_fit(new_prefix)
        cache[cid] = new_prefix          # MRU
        resident_tokens += new_prefix
        ctx_of[cid] = new_prefix
        idx_of[cid] += 1
        if idx_of[cid] < len(turns):
            pending.append(cid)          # re-queue next turn (closed-loop)
    return per_turn

def pct(xs, q):
    xs = sorted(xs); n = len(xs)
    return xs[min(n-1, int(q*n))]

def main():
    convs = build_conversations(TRACE)
    total_turns = sum(len(c) for c in convs)
    print(f"conversations={len(convs)} total_turns={total_turns}")
    per_turn = simulate(convs, CAP)
    adj = [p[3] for p in per_turn]
    ctx = [p[2] for p in per_turn]
    from collections import Counter
    kinds = Counter(p[4] for p in per_turn)
    tot_adj = sum(adj); tot_ctx = sum(ctx)
    print(f"\n=== cache-adjusted prefill work (LRU cap={CAP/1e6:.1f}M tok) ===")
    print(f"turns={len(adj)}  total_adjusted_prefill={tot_adj/1e6:.1f}M  total_ctx_if_nocache={tot_ctx/1e6:.1f}M")
    print(f"effective_hit_rate (1 - adj/ctx) = {1 - tot_adj/tot_ctx:.3f}")
    print(f"adjusted work per turn: p50={pct(adj,.5)} p90={pct(adj,.9)} p99={pct(adj,.99)} max={max(adj)} mean={tot_adj/len(adj):.0f}")
    # concentration: what frac of total prefill work is in the top X% of turns?
    a = sorted(adj, reverse=True)
    for frac in (0.01, 0.05, 0.10, 0.20):
        k = max(1, int(frac*len(a)))
        print(f"  top {frac*100:.0f}% costliest turns carry {sum(a[:k])/tot_adj*100:.1f}% of all prefill work")
    print(f"\n=== breakdown by kind (turns / adjusted-tokens) ===")
    for k in ("cold","reuse","evicted_full","evicted_partial"):
        tw = sum(p[3] for p in per_turn if p[4]==k)
        print(f"  {k:16s}: {kinds.get(k,0):6d} turns ({kinds.get(k,0)/len(adj)*100:4.1f}%)  {tw/1e6:6.2f}M tok ({tw/tot_adj*100:4.1f}% of work)")
    # avoidable = evicted (was cached, got evicted, re-prefilled). cold = unavoidable floor.
    avoidable = sum(p[3] for p in per_turn if p[4] in ("evicted_full","evicted_partial"))
    cold = sum(p[3] for p in per_turn if p[4]=="cold")
    print(f"\n  AVOIDABLE (evicted recompute) = {avoidable/1e6:.2f}M ({avoidable/tot_adj*100:.1f}% of work)")
    print(f"  COLD floor (first-ever)      = {cold/1e6:.2f}M ({cold/tot_adj*100:.1f}% of work)")

if __name__ == "__main__":
    main()
