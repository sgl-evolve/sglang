#!/usr/bin/env python3
"""Offline trace-driven KV-cache simulator for the mooncake_mix_v1 loogle workload.

Goal: screen L2-admission / residency policies for document-recompute headroom
BEFORE spending a GPU rate-sweep. Faithful to the bench_serving loogle+multiturn
construction. Token lengths approximated as chars/CH (relative sizes preserved);
the qualitative conclusion is checked for STABILITY vs the arrival-timing knob.

Model
-----
- Each record -> one conversation. Turn 0 prefix = "Input: {doc} Question: {Q0}".
  Turns 1..N = follow-up questions (multiturn; prefix grows).
- Conversation c arrives (turn 0) at Poisson time A_c (rate lam, file order).
- Turn t arrives at A_c + t * gap_c, gap_c = BASE_GAP + out_tokens_c * DEC.
- Global event stream = all (conv,turn) sorted by arrival time.
- Cache = single LRU over L1+L2 capacity (CAP tokens). Residency at the
  per-conversation "deepest resident token" granularity (LRU refreshes the whole
  path on each access, which is what the radix cache does).
- On a turn: matched = min(prior_prefix_len, resident_depth_c); recompute =
  prior_prefix_len - matched  (avoidable prefill work = the TTFT driver).
  New prompt tokens (q_t) are always computed (not counted as *re*compute).

Policies
--------
- stock   : every conv admitted to the LRU on turn 0 (write_through-ish).
- gated   : a conv's prefix is only PROTECTED in the main LRU after it has been
            re-hit once (turn>=1). Before that it lives in a small probation
            region of size PROB_FRAC*CAP. One-shot convs never leave probation
            and never evict protected content.
- gated_grace : like gated, but a probationary conv is protected from eviction
            for a grace window G_GRACE (models "hold until expected next turn").
"""
import sys, json, ast, heapq, random, statistics, collections

MIX = "/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"
CH = 4.0                 # chars/token approx
CAP = 10_700_000         # L1+L2 tokens (2.35M + 8.4M)
random.seed(12345)

def build_convs():
    recs = [json.loads(l) for l in open(MIX)]
    convs = []
    for r in recs:
        qp = r.get("qa_pairs", "")
        doc = r.get("input", "")
        if qp == "none" or (isinstance(qp, (list, str)) and len(qp) == 0):
            turns = [("Input: " + doc + " Question: Please summarize the input", doc[:1024])]
        else:
            try:
                qa = ast.literal_eval(qp) if isinstance(qp, str) else qp
            except Exception:
                continue
            turns = []
            for i, q in enumerate(qa):
                if i == 0:
                    turns.append(("Input: " + doc + " Question: " + str(q.get("Q", "")), str(q.get("A", ""))))
                else:
                    turns.append((str(q.get("Q", "")), str(q.get("A", ""))))
        # token lengths
        prompt_toks = [max(1, int(len(p) / CH)) for p, a in turns]
        out_toks = [max(1, int(len(a) / CH)) for p, a in turns]
        src = r.get("title", "").split("_")[0]
        src = "sharegpt" if src == "sharegpt" else ("leval" if src == "leval" else "loogle")
        convs.append(dict(prompt=prompt_toks, out=out_toks, src=src, n=len(turns)))
    return convs

def build_stream(convs, lam, base_gap, dec):
    """Return list of events (arr_time, conv_id, turn_idx) sorted by arr_time."""
    ev = []
    t = 0.0
    for cid, c in enumerate(convs):
        # Poisson conv-arrival
        t += random.expovariate(lam)
        A = t
        clock = A
        for ti in range(c["n"]):
            ev.append((clock, cid, ti))
            gap = base_gap + c["out"][ti] * dec
            # prefill of this turn also adds latency ~ prompt tokens; small vs decode
            clock += gap
    ev.sort()
    return ev

class LRU:
    def __init__(self, cap):
        self.cap = cap
        self.od = collections.OrderedDict()  # key -> size
        self.used = 0
    def contains(self, k): return k in self.od
    def touch(self, k):
        if k in self.od: self.od.move_to_end(k)
    def add(self, k, size):
        if k in self.od:
            self.used -= self.od[k]; self.od.pop(k)
        self.od[k] = size; self.od.move_to_end(k); self.used += size
        self._evict()
    def _evict(self):
        while self.used > self.cap and self.od:
            k, s = self.od.popitem(last=False)
            self.used -= s
    def remove(self, k):
        if k in self.od:
            self.used -= self.od.pop(k)

def prefix_lens(c):
    """cumulative committed prefix length after each turn (prompt+out), and the
    'prior prefix' matchable at the start of each turn."""
    cum = []
    tot = 0
    prior = []
    for ti in range(c["n"]):
        prior.append(tot)                     # matchable prefix before turn ti
        tot += c["prompt"][ti] + c["out"][ti] # committed after turn ti
        cum.append(tot)
    return prior, cum

def run(convs, ev, policy, prob_frac=0.05, g_grace=None):
    """Simulate. Returns dict of metrics. Cache keyed per-conversation; value =
    the committed prefix length currently resident (approximation: a conv path is
    resident-as-a-unit up to its last-committed depth, LRU over convs by token mass)."""
    main = LRU(int(CAP * (1 - prob_frac))) if policy != "stock" else LRU(CAP)
    prob = LRU(int(CAP * prob_frac)) if policy != "stock" else None
    resident_depth = {}   # cid -> committed prefix len resident
    protected = set()     # cids that have earned main-cache residency
    grace_until = {}      # cid -> time (gated_grace)
    prior_of = {}; cum_of = {}
    for cid, c in enumerate(convs):
        p, cu = prefix_lens(c); prior_of[cid] = p; cum_of[cid] = cu

    recompute = 0        # avoidable recompute tokens (prior prefix that missed)
    served_prior = 0     # total prior-prefix tokens across turns (denominator)
    backup_tokens = 0    # tokens written to L2 (write traffic proxy)
    hits = 0; misses = 0; turns_ge1 = 0

    for (at, cid, ti) in ev:
        c = convs[cid]
        prior = prior_of[cid][ti]
        # ---- matched vs recompute ----
        rd = resident_depth.get(cid, 0)
        # is the conv resident in either cache?
        in_main = main.contains(cid); in_prob = prob is not None and prob.contains(cid)
        avail = rd if (in_main or in_prob) else 0
        matched = min(prior, avail)
        rc = prior - matched
        recompute += rc
        served_prior += prior
        if ti >= 1:
            turns_ge1 += 1
            if matched >= prior and prior > 0: hits += 1
            else: misses += 1
        # ---- commit turn ti: resident depth grows to cum ----
        newdepth = cum_of[cid][ti]
        resident_depth[cid] = newdepth
        # ---- admission / backup ----
        if policy == "stock":
            main.add(cid, newdepth)
            backup_tokens += (c["prompt"][ti] + c["out"][ti])  # write_through: back up all new
        else:
            if cid in protected:
                main.add(cid, newdepth)
                backup_tokens += (c["prompt"][ti] + c["out"][ti])
            else:
                if ti >= 1:
                    # demonstrated reuse -> promote to main + (now) back up whole prefix
                    protected.add(cid)
                    if prob is not None: prob.remove(cid)
                    main.add(cid, newdepth)
                    backup_tokens += newdepth   # first backup of the earned prefix
                else:
                    # probation only (turn 0): NOT backed up to L2 yet
                    prob.add(cid, newdepth)
                    if g_grace is not None:
                        grace_until[cid] = at + g_grace
        # touch on access already handled by add/move
    return dict(policy=policy,
                recompute_M=recompute/1e6,
                served_prior_M=served_prior/1e6,
                recompute_frac=recompute/served_prior if served_prior else 0,
                backup_M=backup_tokens/1e6,
                turn1_hit_rate=hits/turns_ge1 if turns_ge1 else 0,
                turns_ge1=turns_ge1)

def main_run():
    convs = build_convs()
    print(f"convs={len(convs)} total_turns={sum(c['n'] for c in convs)} "
          f"total_committed={sum(sum(c['prompt'])+sum(c['out']) for c in convs)/1e6:.1f}M tok  CAP={CAP/1e6:.1f}M")
    lam = 3.0
    print(f"\n{'gap_model':<22}{'policy':<14}{'recompute_M':>12}{'rc_frac':>9}{'backup_M':>10}{'turn1_hit':>10}")
    # sensitivity sweep over the arrival-timing knob (base_gap seconds, dec sec/tok)
    for (base_gap, dec, label) in [(2.0, 0.05, "fast(2s+.05/tok)"),
                                   (5.0, 0.10, "med(5s+.10/tok)"),
                                   (10.0, 0.20, "slow(10s+.20/tok)")]:
        ev = build_stream(convs, lam, base_gap, dec)
        for pol, kw in [("stock", {}), ("gated", {"prob_frac": 0.05}),
                        ("gated", {"prob_frac": 0.10}),
                        ("gated_grace", {"prob_frac": 0.05, "g_grace": 60.0})]:
            m = run(convs, ev, pol, **kw)
            tag = pol + (f"@{int(kw.get('prob_frac',0)*100)}%" if pol!="stock" else "") + ("+grace" if "g_grace" in kw else "")
            print(f"{label:<22}{tag:<14}{m['recompute_M']:>12.2f}{m['recompute_frac']:>9.3f}"
                  f"{m['backup_M']:>10.1f}{m['turn1_hit_rate']:>10.3f}")
    print("\nNote: tok=chars/4 (approx); recompute_M = avoidable prefill tokens (TTFT driver);"
          " backup_M = L2 write traffic proxy. Lower recompute & backup = better.")

if __name__ == "__main__":
    main_run()
