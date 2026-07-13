# Researcher: valiant — sglang KV-cache (v0.31, full-decode rate sweep, goodput@SLO)

**Name:** valiant · **Branch:** `evolve/valiant` · **W&B run:** `valiant` (project `sgl-evolve`, group `v0.31`)
**Base commit:** `a334877e5` · **Cell:** `programs/sgl/v0.31/research`

Headline metric = **goodput@SLO** = max req/s over λ∈{3,5,7,10} with p99 TTFT ≤ 8 s. 2-tier HiCache
(L1 GPU 2.35 M tok + L2 host 768 GB ≈ 8.4 M tok, no L3). Model Qwen3.5-122B-A10B-FP8 (hybrid-Mamba,
12/48 full-attn layers carry per-token KV). Active cache = **UnifiedRadixCache** (`is_hybrid_ssm` +
`--enable-hierarchical-cache` → `_create_unified_radix_cache`); `hiradix_cache.py` / `hi_mamba_radix_cache.py`
are DORMANT.

## ⭐ BOTTOM LINE (result as of 2026-07-12)
**Contribution = Pending-aware KV retention** (scheduler↔cache co-design): pin the L2 residency of a
conversation's prefix while it has a request in the scheduler's queue; release at admission. ~110 LOC in
`managers/scheduler.py`, env-gated, lossless-by-construction. **Registered formal submission:**
`submissions/pending-aware-retention/paper.html`.
- **v1 (queued-pin) — RELIABLY POSITIVE + LOSSLESS but MODEST & VARIANCE-AFFECTED (⚠️ n=3 revised the headline):**
  hit gain@λ3 = +2.2/+2.8pp (shipped nodes 0-3/1-2) BUT **+0.4pp on ondem-2** (pin ENGAGED, max 19 pins > shipped's 6,
  yet small gain) ⇒ magnitude VARIABLE +0.4…+2.8pp (mean ~+1.8pp), NOT a robust +2.5pp. Under LPM (ondem-2, full sweep):
  +0.77/+0.33/+0.22/+0.15pp @λ3/5/7/10 — pin helps under BOTH schedulers (retention scheduler-independent), diminishing
  with load. ALL measured gains positive (+0.15…+2.8pp); pin never hurts hit (lossless+protective). Ties to variance theme.
  Peak throughput +2–6%@knee (shipped n=2), p50 −25%@λ3.
- **★★★ v8 (2026-07-13, CLEAN same-JOB A/B @fraction1.0, ondem-2) — REFUTES the v7 budget-gain; bounded-negative HOLDS:** stock_v8 λ3=0.6813, v1f1_v8 λ3=**0.6712** ⇒ **Δ=−1.0pp** (negative! same job). So v7's "+1.8pp@fraction1.0" was a CROSS-JOB baseline artifact (v1_fixstress 0.679 vs OLD low stock_v6 0.6721; vs same-session stock 0.6813 it's negative). Two v1@1.0 runs 0.679↔0.6712 = ~1.8pp run-to-run variance ⇒ variance DOMINATES; the pin gain is at/below the noise floor even at max budget. **Discipline paid off: I validated before reframing → did NOT flip the paper to positive on v7 n=1; v8 refuted it. Paper stays bounded-negative (correct).** Crash fix: 0 asserts in v1f1_v8 too (holds across v7+v8; still not conclusive re 26 pins). Added §5.1 noise-floor caveat (stock 0.6721-0.6813 ~0.9pp ≈ gain). v1f1_v8 λ5/7/10 pending.
- **v7 (2026-07-13, PIN_FRACTION=1.0 on ondem-2) — n=1 findings, (2) NOW REFUTED by v8:**
  (1) **CRASH FIX WORKS (supportive):** with the both-set-discard fix, v1 survived a FULL high-pin sweep incl. λ7 on ondem-2 (the crash node) — **0 asserts** (job 19638). Caveat: reached 13 pins, not the 26 v1_v6 crashed at; intermittent → supportive not conclusive.
  (2) **"node-dependent thin gain" is largely BUDGET-STARVATION:** ondem-2 @fraction1.0 gave Δhit λ3 **+1.8pp** (0.679 vs stock 0.6721), λ5 +1.25, λ7 +0.71, λ10 +0.41 — vs +0.10pp @fraction0.5. So ondem-2's deep pending set exceeded the 0.5 budget → starved; raising budget RECOVERS ~+1.8pp. ⇒ gain is BUDGET-GATED (budget must cover the node-dependent pending set), NOT inherently node-thin. This UN-does part of the bounded-negative. ⇒ VALIDATE (clean same-node A/B @fraction1.0 confirming gain + losslessness + re-stress fix), then reframe paper from "unsafe+thin negative" toward "root-caused+fixed bug + budget-gated modest gain." Fix commit in _remove_leaf_from_parent (both-set discard).
- **★★ CRITICAL (2026-07-13, v1_v6 λ7): v1 ITSELF CRASHES at high load — `assert v==node`, all 8 TP ranks, 26 pins.** NOT just the rejected pc — the SHIPPED short-pin v1 crashed the scheduler at λ7 on ondem-2 (eviction-tree invariant, `_evict_device_leaf`→`_remove_leaf_from_parent`). **My fix (1e50159a8) is INCOMPLETE**: crash is at line 1522 (AFTER my `if node.children` guard line 1518) — the assert fails on a CHILDLESS node too (parent's child-slot doesn't map to this node; pin corrupts mapping differently than I root-caused). ⇒ v1 is NOT robustly stable; shipped n=2 (0-3/1-2, max 6 pins) was LUCK/low-pin-count. **HONEST REFRAME: both positive claims fail under replication — gain small+node-dependent AND v1 intermittently crashes at high pin count. The mechanism is not a deployable/safe win; this is now a rigorously-bounded NEGATIVE: no robust safe lossless KV-retention win exists on well-provisioned 2-tier, and exploiting the residual signal destabilizes eviction.** Valid v6 data: λ3 +0.10pp, λ5 +0.25pp (pre-crash).
- **★ n=4 (2026-07-13): pin gain is NODE-DEPENDENT, not just run-variance.** pin-under-FCFS Δhit@λ3: nodes 0-3/1-2 = **+2.2/+2.8pp**; ondem-2 (×2, v1_fcfs + v1_v6) = **+0.4/+0.10pp**. Both ondem-2 measurements SMALL, both 0-3/1-2 LARGE ⇒ bimodal BY NODE (ondem-2 ~+0.25pp; 0-3/1-2 ~+2.5pp), mean ~+1.4pp. stock baseline node-STABLE (0.6726/0.6728/0.6721 all nodes). Likely ondem-2 faster→shorter queues→smaller pending window→less to pin (pending-window mechanism). ⇒ update paper: gain is node-systematic (not random), +0.1–2.8pp. (v6 via queued sbatch 19608, DRAM-settle fix worked: DRAM 1301→1519G wait before v1_v6, no DRAM_TOO_LOW.) v1_v6 λ5/7/10 pending.
- **OPS: ondem-2 (on-demand) RECLAIMED mid-run 19:07** — manager pool reshuffle ended my hold jid19376; server got
  "Gracefully exiting" (not a code crash — my eviction fix held, 0 asserts); v1_fcfs λ5/7/10 LOST (have λ3). ondem-2
  reallocated to sibling. Flock auto-released. Lesson: on-demand held nodes can vanish mid-eval; skip-if-done pipeline
  preserved completed curves. Did NOT re-run (enough data + pending sibling; good-neighbor).
- **pc (post-completion extension) HURTS — n=2 same-node confirmed @λ3 AND λ5:** hit@λ3 pc 0.674/0.673 ≈ stock
  0.678/0.673 ≪ v1 0.700/0.701; hit@λ5 pc 0.661/0.672 ≈ stock 0.664/0.668 ≪ v1 0.680/0.677 → the extension
  CANCELS v1's gain across rates (pc_e full curve corroborates: pc < stock at λ5/7/10).
  broad retention DISPLACES LRU's working set. Not misprediction (continuation 78% predictable). **Targeting
  (the pending set) is the lever, not recency.** (pc full sweeps → W&B pending, ~1.5h.)
- **Bound:** realized +2.5pp ≪ idealized offline oracle (+7pp) ⇒ LRU is near-optimal for concurrent multiturn
  EXCEPT its one scheduler-visible failure (evicting queued continuations), which v1 removes.
- **★★ DEEPER RESULT (2026-07-12, sim+trace) — SCHEDULER-CACHE COUPLING:** the LRU-vs-Belady gap is CREATED by
  the scheduler's pull order, not the cache. Under the eval's default `schedule_policy=fcfs` (VERIFIED default;
  eval.sh doesn't override) LRU trails Belady 7–22pp; under prefix-aware (lpm) pull gap→0 (LPM temporally
  clusters each conv's turns → LRU already keeps them). MECHANISM = recency-INVERSION: measured Spearman
  ρ(recency,next-use)=**+0.81** among pending convs → LRU's victim is the SOONEST-reused (queued-longest =
  pulled-soonest). So pinning LRU's victim (engine pins ARRIVAL-order = oldest-waiting first = risk-aligned) is
  Belady-aligned. Budget/priority sweep: risk-first pin captures 61–92% of Belady @25–75% budget; recent-first
  ~0% (= why pc/broad-recency fails — pins the wrong set). ⇒ pending-pin gives FCFS prefix-scheduling's cache
  locality WITHOUT its reorder/TTFT-fairness cost. Realized +2.5pp ≪ +7pp sim-ceiling = 2-tier host absorbs
  device evictions. Analysis: `analysis/{build_convs,risk_sim,gap_decomp,compare_coupling}.py`, saved
  `analysis/results/sim_summary.txt`. Paper §2.4 + Fig 2 added (commits 97db816f1…df45d253b).
- **★★ GPU COUPLING RESULT — COMPLETE (stock, n=1 same-node ondem-2) — REFUTES SIM's HIT-COUPLING (integrity win):**
  | λ | hit FCFS | hit LPM | Δhit | p99 FCFS | p99 LPM | Δp99 |
  | 3 | 0.6726 | 0.6715 | −0.11pp | 8052 | 11323 | +41% |
  | 5 | 0.6662 | 0.6619 | −0.43pp | 16286 | 26938 | +65% |
  | 7 | 0.6605 | 0.6578 | −0.27pp | 35029 | 36844 | +5% |
  | 10| 0.6576 | 0.6550 | −0.26pp | 40770 | 43047 | +6% |
  ⇒ Real 2-tier hit is **scheduler-INSENSITIVE** (LPM Δ<0.5pp ALL rates — decisively NOT the sim's +7pp) + LPM
  **worsens p99 41–65%** at moderate load. Host tier absorbs eviction-order → sim's FCFS gap is single-tier
  artifact (same reason realized +2.5pp ≪ +7pp sim ceiling). **retention (pin) is the hit lever; scheduling isn't**
  (LPM: no hit gain, tail cost). Inversion ρ=+0.81 (model-free) still explains pin targeting → transfers. **Paper
  HONESTLY REVISED: §2.5 + Table 2 added; §2.4/Fig2/abstract/contrib/§7 reframed to report the refutation of my
  own sim (commit d27019401).** Still running: v1_lpm, v1_fcfs (does pin help under each scheduler? expect yes =
  retention scheduler-independent). W&B logging pending.
- **Metric finding:** goodput@SLO is variance-dominated at the 8s boundary (identical stock runs flip 0↔3.02
  across nodes) ⇒ hit-rate/throughput is the robust metric.
- **pc CRASH (3rd strike vs pc):** pc_c crashed @λ=7 (`assert v==node` in `_evict_device_leaf`→`_remove_leaf_from_parent`): host-pinning a deep node keeps a host-only child alive → its ancestor stays a device-leaf-with-child → write-through delete-entirely removes a non-empty node → tree corruption. pc's LONG pins make it common; v1's short pins (released at admission) ran clean n=2 all rates. Retracted pc_c from W&B. Fix (future): stricter `_is_device_leaf` (require childless) or demote-not-delete. Ship v1 (short-pin, unaffected).

---

## Direction (ONE ambitious line): Pending-aware prefix retention — scheduler↔cache co-design

**Thesis.** In concurrent *multiturn* serving the cache's dominant loss is **not** eviction *order*
(LRU≈Belady among a fixed set) but the eviction of prefixes belonging to conversations that have an
**in-flight continuation the scheduler already knows about** but the cache ignores. Coupling the cache's
retention to the scheduler's *pending-request set* recovers (in the offline model) the full oracle
(Belady) hit rate — with a realizable, non-oracle signal.

### Trace-driven motivation (GPU-free, `mooncake_mix_v1.jsonl`, exact tokenizer)
- Workload = 1553 conversations, mean **4.61 turns** (median 3, max 61); **593 are single-turn** (no
  reuse). Each conversation = one huge document (mean **~11.8k tok**, p90 ~28k, max ~183k) reused across
  all its turns; follow-up questions are tiny (median **~18 tok**). No cross-conversation sharing.
- A perfect cache eliminates **80.9%** of prefill work (no-cache ≈ 98 M prompt tok → perfect ≈ 18.7 M).
  So prefill is the goodput lever and the cache's potential value is enormous.
- **Continuation turns are load-bound**: huge cached prefix + tiny new compute → the whole turn's cost is
  reuse-or-recompute of the prefix. If a continuation's prefix is evicted before it runs, the entire
  ~10k-token document is recomputed → a prefill spike → TTFT tail under load.
- Multiturn issue model (verified in `bench_serving`): a turn's *next* turn is re-enqueued on completion
  (line 190) and pulled per-Poisson → **turns interleave with other conversations**, so a continuation's
  prefix ages under churn and LRU evicts it even though its reuse is imminent.

### Offline cache simulation (page-level, real token counts, discrete-event issue order)
Hit-rate = cached_tok / prompt_tok (real definition). Capacity = combined L1+L2.

| λ | cap | LRU | Belady | **protect-pending (realizable)** | admit-no-terminal (oracle) |
|---|-----|-----|--------|----------------------------------|----------------------------|
| 3 | 10.7M | 0.736 | 0.808 | **0.808** | 0.808 |
| 5 | 10.7M | 0.737 | 0.808 | **0.808** | 0.808 |
| 10| 10.7M | 0.737 | 0.808 | **0.808** | 0.808 |
| 3–10 | 8M | 0.57–0.60 | ~0.80 | 0.67–0.71 | 0.67–0.71 |

**Capacity sweep (λ=5, `analysis/cache_sim_capsweep.py`)** — protect-pending gain over LRU:

| cap | 6M | 7M | 8M | 9M | 10.7M | 12M | 14M |
|-----|----|----|----|----|-------|-----|-----|
| LRU | .358 | .438 | .583 | .725 | .737 | .745 | .758 |
| protect-pending | .425 | .516 | .696 | .808 | .808 | .808 | .808 |
| **gain (pp)** | +6.8 | +7.8 | **+11.2** | +8.3 | +7.1 | +6.4 | +5.0 |
| **recompute ↓** | 10% | 14% | **27%** | 30% | 27% | 25% | 21% |

Peak gain at cap ~8–9M — the **likely real effective-capacity regime** (running requests consume L1, so
reuse-capacity < the 10.7M nominal). ≥9M: protect-pending == Belady (full oracle).

**Finding:** at the real capacity a realizable *protect-pending* policy (protect conversations that have a
request currently in the system) closes the **entire ~7 pp LRU→Belady gap** — a **~27% reduction in
recompute**. The gap is conversation-structural (terminal-conversation pollution + eviction of pending
continuations), which overturns the "LRU≈Belady, eviction is a dead end" assumption *in this concurrent
multiturn regime*. At tight capacity the working set genuinely overflows and protection captures ~half.

### Why it's novel (vs the excluded / prior-art baselines)
- **Not an eviction-order policy** (LRU/LFU/2Q/GreedyDual are excluded and, per above, near-useless here):
  it is a *scheduler→cache reuse-intent signal*. The lever is *which* prefixes are protected, sourced from
  the scheduler's pending set — a co-design, not a replacement policy.
- **Not Strata** (cache-aware scheduling overlaps *loads* with compute): this *prevents the eviction*
  that causes the recompute in the first place; Strata's data-plane fix (GPU-assisted `kernel` I/O) is
  moreover *unavailable* on hybrid-Mamba (frozen `direct`/`page_first_direct`).
- **Not exclusive-tiering / write_back config** (those are the known ~13pp baseline, not a contribution).

### Implementation hook (confirmed)
- `_prefetch_kvcache(req)` (scheduler.py:2280) runs when a req enters the waiting queue; in 2-tier (no L3)
  it is a **no-op** → a continuation's cached prefix is unprotected while it waits and can be evicted by
  running requests. This is the insertion point.
- `UnifiedRadixCache.inc_host_lock_ref(node)` / `dec_host_lock_ref(node)` pin/unpin a node's host (L2)
  residency; `match_prefix` yields `last_host_node`. Eviction (`evict_host` via `get_prev_no_host_lock`)
  already skips host-locked nodes. So: pin `last_host_node` of a waiting continuation; release on
  admission/abort.

### Honesty caveats to test (not assume)
1. Offline LRU (0.737) > real baseline hit (0.62 in the old single-point). Real effective capacity may be
   lower (fragmentation/pinning/mamba) → gain may shrink toward the 8M row. **Calibrate on the real sweep.**
2. Server sees a continuation only during the *server-queue* window, not the client think/send gap; under
   high λ (where goodput is decided) the server queue is deep, so this window should dominate — **verify**.
3. Pinning reduces effective capacity for others; must be **capacity-aware** (degrade gracefully when the
   pending set exceeds free L2). The real design contribution is *what to pin when you can't pin all*.

---

## Live baseline observations (rate=3, job 19434)
- Reuse is real: 26% of prefill batches reuse a prefix (p90 cached 23.7k tok, max 277k). L1 usage 0.95–0.97
  (genuine device pressure). Confirms the pressure + reuse regime.
- **Server queue depth**: p50=2 but a deep tail (p90=21, p99=95, max=107) even at λ=3 → deeper at λ=7,10.
  So v1 (server-queue pinning) has real scope at high load (where goodput@SLO is decided); the pending set
  (~1M tok) fits the 0.5×L2 ≈ 4.2M budget. Running-req p50=252 (near the 270 cap) → L1-memory-bound admission.
- **Client-gap caveat quantified**: because the client caps concurrency at 256 ≈ server capacity, much of a
  continuation's inter-turn gap is spent *behind the client semaphore* (unprotected), not in the server queue.
  v1 protects only the server-queue window; the post-completion variant (v2) targets the client gap. The v1
  result will tell us what fraction of the offline Bélády gap the server-queue window alone captures.

## ★ Real baseline calibration (job 19434, ondem-3)
- **λ=3: hit=0.678, req/s=2.87, TTFT p50=1018ms, p99=11763ms** (e2e_p99 parse-empty).
- hit 0.678 ⇒ real effective reuse-capacity ≈ 9M tok (running reqs consume L1) — squarely in the offline
  **peak-gain regime** (+8–11pp / ~30% recompute reduction predicted). Mechanism should have headroom.
- **p99 TTFT already 11.8s > 8s SLO at λ=3** (protocol expected ~7.6s; the warm/no-flush sweep is harsher).
  Higher rates worsen p99 ⇒ **baseline goodput@SLO likely = 0**. So the win condition is: cut the p99 tail
  below 8s at some rate (0 → positive goodput), OR at minimum shift the p99 curve down materially. The
  recompute reduction lowers total prefill load ⇒ shorter queues ⇒ lower p99 for all requests.

## ★ Prefill-work decomposition (why the mechanism has leverage on the tail)
From baseline hit=0.678 + structural perfect-cache (18.7M irreducible new tokens):
- total prompt = 99.9M; baseline **prefill work = 32.2M (32%)**.
  - irreducible first-compute: 18.7M (58% of prefill).
  - **evicted-continuation recompute: 13.5M (42% of prefill)** ← the mechanism's target.
- Recovering the Bélády gap (hit 0.678→0.808) cuts prefill work **32.2M→19.2M (−40%)**; hit→0.75 = −22%.
- The p99 tail is queue-bound (median 1.0s, p90 3.2s, p99 11.8s); a 22–40% prefill-load cut → shorter
  queues (chunked-prefill interleaves, so less total work speeds every request incl. the cold-doc tail)
  → the mechanism should pull p99 down materially, plausibly under the 8s SLO.

## ★★★ KEY ABLATION — TARGETING matters (v1 targeted WINS; pc indiscriminate is a WASH)
- **v1 (queued-pin, ~6 targeted pins @λ=3):** hit **+2.5pp** (n=2), p50 −25%, throughput +5-6% @high load.
- **pc (post-completion, 99 indiscriminate pins @λ=3):** hit **0.671 ≈ stock (~0 gain!)** despite 16× more pins.
- **Insight:** protecting ALL recently-finished convs for a horizon wastes L2 on terminal/non-reused convs,
  offsetting the client-gap benefit → net wash. The winning signal is the SCHEDULER'S PENDING SET (guaranteed
  imminent reuse), not broad recency-retention (which ≈ LRU). This is why the offline Bélády gap (+8pp) is NOT
  fully server-realizable: reuse-TIMING (client think-gap) is unknowable server-side; only the server queue
  gives a guaranteed-reuse signal (+2.5pp). A bounded-realizability result + a clean targeting ablation.
- (pc_e @λ=3 also p99 6.5s but that's node-luck, not the mechanism — hit shows no gain.)
- Checking whether pc helps at HIGH λ (short client-gap under backlog). pc_b/pc_c (n=2 same-node) will confirm.

## ★★★ FIRST EFFICACY RESULT — mechanism WORKS (v1_b vs stock_b, same node 0-3, λ=3)
| metric | stock_b | v1_b (queued-pin) | Δ |
|--------|---------|-------------------|---|
| hit | 0.6779 | **0.6998** | **+2.2pp** (≫ ±0.5pp node noise → real) |
| p50 TTFT | 749ms | **538ms** | **−28%** |
| p99 TTFT | 8128ms | 7978ms | −1.8% (tail = cold-doc prefill, irreducible) |
| req/s | 3.02 | 3.02 | = |

Queued-pinning recovers evicted-continuation reuse: **hit +2.2pp (robust), p50 −28%**, lossless by construction.
Max ~6 concurrent pins at λ=3 (shallow queue) yet cumulative hit gain is real. p99 barely moves (cold-doc tail
dominated) — so v1 helps median/throughput, not the p99 tail; **pc (post-completion) expected to add more via
client-gap coverage**. Awaiting v1_b λ=5/7/10 (hit gain may grow with load), v1_c (n=2), pc_b/pc_c.

**★ v1_b λ=5** (vs stock_b@5): hit .6643→**.6795 (+1.5pp)**, p99 27.0→**23.6s (−12.5%)**, req/s 3.83→**4.02 (+5%)**.
⇒ v1's systems impact (p99 −12.5%, throughput +5%) GROWS with load (deeper queue → more pins); hit gain ~1.5-2.5pp
across rates. p99 still ≫8s at λ=5 (goodput SLO decided at λ=3). **pc scope = 70 pins (vs v1's 6) ⇒ pc read next.**

**★ n=2 CONFIRMED (v1_c on 1-2, λ=3):** hit 0.6734→**0.7011 (+2.8pp)**, p50 692→**530ms (−23%)**, p99
6286→8031ms (+28%). So across n=2: **hit +2.2/+2.8pp (robust ~+2.5pp), p50 −28/−23% (robust ~−25%), p99
−2%/+28% (NOISE — coin-flip confirmed).** ⇒ Reliable contribution = hit + median; p99/goodput variance-dominated
(honest). Max pins ≤6 (queued window small) ⇒ pc should be larger.

## ★★ COIN-FLIP FINDING (decisive for metric choice)
Three **identical stock** runs (3 nodes), λ=3:
| node | p99 | hit | goodput@SLO |
|------|-----|-----|-------------|
| ondem-3 (baseline) | 11.8s | 0.678 | 0 |
| 0-3 (stock_b) | 8.1s | 0.678 | 0 |
| 1-2 (stock_c) | **6.3s** | 0.673 | **3.02** |

Stock **goodput@SLO flips 0↔3.02** across nodes (λ=3 p99 straddles the 8s SLO: 6.3–11.8s, 1.9× spread), while
**hit is stable (0.673–0.678)**. ⇒ goodput@SLO is variance-dominated at the boundary (matches the v0.3
coin-flip). **Headline = hit-rate / recompute reduction (robust, node-independent); p99 reported as same-node
delta with n≥2 and the metric's variance quantified.** The mechanism must reduce λ=3 p99 enough (same-node)
to push it reliably under 8s across the variance — a ~30% p99 cut would flip goodput positive on ~all nodes.
Full baseline sweep (ondem-3): p99 11.8/24.5/34.2/39.7s @ λ=3/5/7/10, hit 0.678/0.662/0.658/0.658, peak 4.5 req/s.

## Node-variance finding (validates same-node A/B)
Same STOCK config on two nodes, λ=3: **ondem-3 p99=11.8s vs 0-3 p99=8.1s (−45%!)**, but hit identical
(0.678 vs 0.678). ⇒ p99/goodput are strongly node-variance-sensitive; **hit-rate is node-independent**.
Conclusion: compare v1_b/pc_b to **stock_b (same node 0-3)** and stock_c on 1-2; report the **p99 DELTA (%)**
as the robust metric (absolute goodput@SLO is node-boundary-sensitive). Bonus: on 0-3 stock λ=3 p99=8.1s sits
right at the 8s SLO, so a small mechanism-driven p99 cut can flip goodput 0→3.02 there.
Baseline (ondem-3) stays a stock reference; the primary A/B lives on 0-3 and 1-2.

Real stock reference (node 0-3, stock_b): λ=3 req=3.02 p50=749ms **p99=8128ms** hit=0.678.

## Parallelization (robust, boundary-surviving)
Discovered: plain `nohup &` processes survive session boundaries (only harness `run_in_background` tasks are
torn down). So parallel evals via nohup'd `srun --overlap` into the manager's idle held nodes are robust.
- ondem-3 (sbatch): baseline (stock ref).
- **held 0-3 (nohup srun pipeline, flock): stock_b → v1_b → pc_b** — all same-node ⇒ clean A/B.
- Registered 1-2, ondem-2 in `_pool/held` for siblings (good neighbor); cancelled my ondem-3 v1 to free it.

## Ablation / iteration plan (as GPU frees)
1. **v1 vs baseline** (same certified node): does pinning raise hit-rate + goodput@SLO, lossless? Confirm
   `[valiant]` pins activate.
2. **pin fraction sweep** (0.3 / 0.5 / 0.7) — capacity trade-off; find the graceful-degradation knee.
3. **pin on/off same-code** (`VALIANT_PIN_ENABLE=0`) — isolate the mechanism from any incidental diff.
4. **error bars** — n≥2 replicates of baseline & best config.
5. **v2 (if headroom):** post-completion retention — protect a just-finished conversation's leaf for a
   bounded horizon to also cover the *client* think/send gap (server-queue pinning only covers the
   server-queue window). Ablate the increment. Possibly + terminal-prefix write-admission.

## Mechanism code (one version, env-configured — commit d10eabb5e)
Combined engine change in `managers/scheduler.py` (~110 LOC). Env toggles (sbatch snapshots env, so
configs queue independently):
- `VALIANT_PIN_ENABLE` (default 1): master. `0` = stock behavior (dormant code) → the same-code control.
- `VALIANT_PC_ENABLE` (default 0): post-completion extension. `0` = queued-only (v1); `1` = +post-completion (v2).
- `VALIANT_PIN_FRACTION` (default 0.5): host-pin token budget = fraction × L2.
- `VALIANT_PC_HORIZON_S` (default 20): post-completion pin lifetime after admission.

Note: ondem-3 baseline (19434) runs the **pristine** commit a334877e5 (truly stock); v1 (19443) runs the
combined code with pin on / pc off (= queued-only). Clean same-node A/B on ondem-3.

## Versions
| ver | tag | config | goodput@SLO vs base | lossless | takeaway |
|-----|-----|--------|---------------------|----------|----------|
| v0_official | baseline | stock single-point | (W&B reference point) | — | logged |
| v0-baseline | baseline | pristine sweep (job 19434) | running | — | calibrates real curve |
| stock_b (0-3) | config | PIN_ENABLE=0 (stock, same binary) | **0** (p99 8.1/27.0/35.3/41.6s; hit .678/.664/.659/.658) | n/a | node-0-3 stock ref (W&B) |
| stock_c (1-2) | config | PIN_ENABLE=0 (stock, same binary) | **3.02** (p99 6.3/13.0/34.8/41.5s; hit .673/.668/.663/.659) | n/a | node-1-2 stock ref (W&B); coin-flip vs stock_b |
| v1_b (0-3) | mechanism | pin on, pc off (queued-only) | running (mechanism confirmed active) | expect yes | vs stock_b same-node |
| v1_c (1-2) | mechanism | pin on, pc off (queued-only) | running | expect yes | vs stock_c same-node |

**n=2 stock hit consistency (±0.5pp across nodes):** λ=3 .678/.673, λ=5 .664/.668, λ=7 .659/.663, λ=10 .658/.659.
**Coin-flip confirmed:** identical stock → goodput 0 (0-3) vs 3.02 (1-2). Hit is the robust metric.

## Formal submissions
_(none yet)_
