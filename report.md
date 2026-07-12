# Researcher `wilkes` — sglang KV-cache architecture (v0.31, research-tier)

**Name:** wilkes (Maurice Wilkes — inventor of cache memory / "slave memory", 1965).
**Cell:** v0.31 / research. **Base commit:** `a334877e5` (clean `main`, branch `evolve/wilkes`).
**W&B:** run `wilkes` in `sgl-evolve` (group v0.31). **Siblings (do not read):** kleinrock, valiant.

Bar = top-venue only (SOSP/OSDI/MLSys/ASPLOS/FAST). ONE deep, novel, lossless mechanism that shifts
the **goodput@SLO** curve. Not eviction tuning, not exclusive-tiering, not WSAC/PGAC/SRPF/lpm (all done
by predecessors / textbook / config-equivalent).

---

## 0. Environment & protocol (verified from the live scripts, not the stale prose)

- **eval.sh is a REAL rate sweep** (the SOP prose is stale, describes an old single-λ eval): warmup 300
  convs @λ3 → sweep **λ∈{3,5,7,10}**, **NUMP=1553**, **conc 256**, **NO flush between rates** (warm
  steady-state), SLO **p99 TTFT ≤ 8000 ms**. Headline **goodput@SLO = max req/s with p99 TTFT ≤ 8s**.
  Controls (≈flat, decode-bound): peak tok/s, peak req/s. This v0.31 protocol deliberately fixes the two
  v0.3 flaws (cold-start goodput coin-flip → warmup+no-flush; under-pressure hit 0.84 → NUMP 1553, hit ~0.62).
- **Path bug (worked around):** `eval.sh:9` hardcodes `workspace/sgl/v0.3_ablations/research`; setup writes
  `workspace/sgl/v0.31/research`. Fixed by symlink `v0.3_ablations/research -> v0.31/research` (matches the
  sibling `base` symlink already present).
- **Frozen config:** 2-tier L1 GPU (~2.35M tok) + L2 host 768 GB (~8.4M tok), no disk; TP8, ctx 262144,
  page 64, chunked-prefill 6144, io `direct`, layout `page_first_direct`, write_through. Hybrid-Mamba MoE
  (Qwen3.5-122B-A10B-FP8): full-attn every 4th layer carry KV; MambaPoolHost for SSM state (non-lever —
  O(#seq)≪O(#tok)).
- **Baseline (v0_official, old single-λ point, logged to W&B):** hit 0.6217, host_util 0.9999 (L2 FULL,
  genuine pressure), λ=3 ttft p50 750 / p99 6326 ms, req/s 2.78, out 355 tok/s, load_back 298M tok,
  evict 582M tok. My own stock **sweep** baseline = job 19437 (queued).

## 1. Workload structure (verified by reading the harness myself)

- **Closed-loop within a conversation, open-loop (Poisson λ) across conversations.** `get_requests` pulls
  exactly `num_actual_requests` (total turns) at rate λ; a conversation's **turn N+1 is re-enqueued to the
  BACK only after turn N's full response completes** (bench_serving.py:186-190). conc semaphore = 256.
- Turn 0 = long document (LEval/LooGLE, up to tens of K tok) + Q → **expensive cold prefill, no reuse yet**.
  Turns 1+ = accumulated history (grows) + short Q → **cheap IFF the prefix is still resident**.
- **Reuse distance (think-gap):** turn N's KV sits idle for `turnN E2E latency + requeue/queue delay`
  before turn N+1 reuses it. Load-dependent: under pressure the gap widens.
- **Hit 0.62 is entirely within-conversation multiturn reuse** (no cross-document reuse in the mix).

## 2. The displacement externality (the core mechanism opening — verified in code)

Admitting a prefill that needs device KV → `evict()` pops **LRU device leaves** (evict_policy.py, key =
`node.last_access_time`) → write_through **demotes** them to host → host is full (util≈1.0) → `evict_host`
**drops LRU host leaves** → their next reuse is a **full recompute (miss)**.

⇒ **A big cold "whale" prefill (turn-0 long doc) evicts soon-to-be-reused warm conversation prefixes.**
This is a negative externality that NO existing policy prices in:
- `lpm`/`dfs-weight` order by the request's OWN prefix hit; `srpf` (absent here) by OWN residual work;
  `WSAC`/`PGAC` (absent here) binary-defer cold starts (which just pushes cold TTFT into the p99 tail —
  the documented failure mode). None account for the **recompute cost inflicted on other resident KV**.
- Eviction order itself is a dead end (LRU≈Belady) **under in-order reuse** — but multiturn think-gap
  reuse is **out-of-order** and miss **cost is highly non-uniform** (a warm-turn miss recomputes the whole
  accumulated history), so the p99 tail has headroom the "eviction is dead" result does not cover.

## 3. Hypothesis (to confirm from the baseline sweep before committing)

As λ climbs 3→10, p99 TTFT crosses 8s primarily because of **avoidable warm-turn misses induced by
cold-prefill displacement** (+ HOL queueing behind long cold prefills), not (only) irreducible cold work.
If true, a **displacement-aware / reuse-protected prefill+residency co-mechanism** reduces TOTAL prefill
tokens recomputed (a real work reduction, not mere reordering/de-saturation) and shifts goodput up.

**Decision gate (from baseline `curve.csv` + `bench_r$R.json` + `metrics_r$R.txt`):**
- Decompose the p99 TTFT tail: cold turn-0 whales vs warm-turn misses vs pure queueing.
- Track evict_tokens / load_back_tokens / hit_rate vs λ. Super-linear miss growth ⇒ displacement confirmed.
- Then pick the flagship: (A) displacement-aware admission, (B) reuse-imminence residency protection,
  (C) load-back prefetch/overlap, or (D) whale chunk co-scheduling. Leaning (A)/(B).

## 4. Extension points (verified, this clone)
- Ordering/admission: `schedule_policy.py PrefillAdder.add_one_req` (~866), `SchedulePolicy.calc_priority`
  (~170); scheduler `_get_new_batch_prefill_raw` (~2758), `add_one_req` call (~2892).
- Residency protection: `unified_cache_components/full_component.py drive_eviction` heap (~136),
  `evict_policy.py` (LRU/priority), node fields `last_access_time/hit_count/priority/creation_time/backuped`.
- Cache/allocator state at admission: allocator `available_size()`, `tree_cache.evictable_size()`,
  per-req `prefix_indices/host_hit_length/num_matched_prefix_tokens`.

---

## 5. Offline simulation (first-order, sim/) — DIRECTIONAL, not proof

Tokenized the real dataset (sim/conv_trace.json): **1553 convs, 7037 turns, avg 4.53/conv (max 61),
working set 20.2M tok = 1.89× oversubscribed** vs L1+L2 (10.7M). turn-0 docs dominate conv KV (p50 7.9K,
p90 32.6K, max 192.7K) → a **warm-turn miss recomputes the whole document** (reused ~3.5×/conv avg).

Radix-faithful cache sim (sim/simulate.py). **Robust, timing-independent finding:** over the access
trace, LRU incurs ~7.2M avoidable-recompute tokens (~23–29% of prefill work) that a **Belady / evict-DEAD-
first oracle eliminates** — LRU conflates *recency* with *liveness* and re-computes live conversation
prefixes it evicted during their think-gap, while a finished conversation's DEAD KV lingers. This is
directional evidence that **"LRU≈Belady, eviction is dead" breaks in the concurrent closed-loop regime**
(the protocol's claim is for in-order reuse). CAVEAT: the sim's timing model is under-congested (single
FIFO prefill server → concurrency stays low, TTFT/λ-sensitivity not captured), so magnitudes are
model-dependent; the GPU eval + a traced diagnostic are the real proof. A naive "protect every completed
leaf" grace policy degrades to LRU (no live/dead discrimination) — the discriminator must be TEMPORAL
(grace ≈ predicted think-gap: live convs return within grace → hit; dead convs' grace expires → evicted).

## 6. Committed direction (DATA-GATED)

**Thesis:** the concurrent closed-loop multiturn regime creates eviction headroom (avoidable warm-turn
recompute) that the single-point regime hides; a **continuation-aware residency** mechanism captures it,
losslessly, shifting goodput@SLO. The novel signal = closed-loop completion-timing / predicted think-gap
(not recency/frequency), targeting the regime where LRU≠Belady. NOTE: expensive misses are those evicted
from BOTH L1 and L2 (L1-only eviction → cheap load-back); host_util≈1.0 means L2 forced-eviction is the
crux → the mechanism biases the *host* leaf eviction victim choice too.

**GATE (from baseline sweep 19437 aggregates + a traced diagnostic):** confirm warm-turn misses grow with
λ (hit% falls, evict/recompute rises super-linearly) and the LRU-vs-Belady gap is real on hardware. If
confirmed → implement + eval the mechanism. If the gap is small (e.g. L2 large enough that live prefixes
survive think-gaps) → rigorous **bounded-negative** result (also top-venue). Either outcome is honest.

## 7. PIVOTAL code finding — stock already protects server-waiting prefixes (revises the plan)

Stock default policy is **fcfs**, but `calc_priority` still calls `match_prefix_for_req` for **every**
waiting request each scheduler step (schedule_policy.py:179-185, cache-agnostic branch), and
`match_prefix` **bumps `last_access_time`** for the matched node + ancestors (unified_radix_cache.py:976-980).
⇒ Every request in the SERVER waiting_queue has its prefix recency-refreshed each step ⇒ LRU already keeps
waiting-reusers' prefixes hot ⇒ they are NOT the source of warm-turn misses. So my "WSRP / protect
waiting-reuser prefixes" idea is **largely redundant with stock** (do NOT build it as-is).

Consequences (residency windows for a conv's prefix between turn t and turn t+1):
- just-completed (fresh, top of LRU) → safe.
- turn t+1 in SERVER waiting_queue → touched each step → safe (stock).
- turn t+1 created but stuck CLIENT-side (Poisson pull backlog / concurrency-256 semaphore) → invisible to
  server, prefix ages → the ONLY realizable vulnerable window, but NOT server-capturable (no conv id).
⇒ Warm-turn misses (if they matter) come from the client-blocked/think-gap window, or the tail is
dominated by irreducible cold-whale prefill + HOL queueing, not cache misses.

**Revised gate (baseline sweep 19437 = the arbiter):**
- hit% DROPS with λ + evict/recompute grows super-linearly ⇒ warm-turn misses real ⇒ find a capturable
  mechanism (or prove the capturable part is small = bounded negative).
- hit% ~flat + p99 grows ⇒ tail = cold-whale prefill cost / HOL / prefill-decode interference ⇒ pivot to a
  prefill-cost / interference-reduction mechanism (must beat textbook SRPF/mixed-chunk, both known).
Either way the paper is honest: mechanism if capturable headroom exists, else a rigorously-bounded negative
("concurrent multiturn does NOT break LRU≈Belady in practice because stock schedule-time matching + large
L2 keep live prefixes resident; the p99 limiter is X").

## 8. Prior-art synthesis (for related work + fork menu)

Closest prior art per candidate (all cited in submissions later):
- **AttentionStore/CachedAttention (ATC'24, 2403.19708):** hierarchical KV cache + layer-wise ASYNC
  prefetch + async save for multiturn; 87% TTFT cut. Prefetch is REACTIVE (when a req enters the batch).
  ⇒ plain L2→L1 prefetch is NOT novel. Gap = prefetch for reqs STILL WAITING (queue-depth-aware).
- **Strata (2508.18572):** balanced batching (pair prefill w/ decode to hide I/O), GPU-assisted transfer.
- **Mooncake (FAST'25, 2407.00079):** KVCache-centric disagg + streaming layer-wise KV transfer (cluster).
- **PPD (2603.13358):** append-prefill (turn≥2) colocated w/ decode → 68% turn2+ TTFT — but CLUSTER-level
  routing; NOT applicable to my fixed single-node config.
- **RedKnot (2606.06256):** per-(layer,head) KV classes + SegPagedAttention (single-tier, kernel-heavy).
- **Sarathi-Serve (OSDI'24):** chunked prefill / decode-maximal batching (already in stock, textbook).
- **AsymCache (2606.02964)/Predictive-Multi-Tier (2604.26968)/Continuum (2511.02230):** cost/Bayesian/TTL
  residency — overlaps my displacement-admission idea (which stock waiting-protection already blunts).

**Refined fork menu (choose after baseline p99 anatomy):**
- p99 = warm-turn LOAD-BACK bound → **queue-aware speculative L2→L1 prefetch** (novel vs AttentionStore's
  reactive prefetch; lossless; orthogonal to eviction). ← leading candidate IF load-back is on crit path.
- p99 = cold-whale HOL / prefill-congestion → hard to beat Sarathi/SRPF novelly → likely characterization
  + modest refinement, or bounded-negative.
- p99 = thrash/miss bound → residency, but stock schedule-time matching caps it → head-aware or negative.

De-risk gate for the prefetch branch: **is load-back actually on the TTFT critical path, or already per-layer
overlapped with compute?** → **VERDICT (from code): per-layer OVERLAPPED.** `start_loading` transfers on a
separate `load_stream` with a per-layer `producer_event.complete(i)`; the model forward waits per-layer via
`layer_transfer_counter.wait_until(layer_id)` (memory_pool.py:1526/1537) right before each layer's attention.
So load-back overlaps suffix compute; prefetch-during-wait would only remove the ~layer-0 bubble (~10-50ms)
→ **marginal, not a flagship.**

## 9. Convergent assessment (pre-baseline) — the easy cache levers are already taken by stock

Three independent code findings blunt the obvious mechanisms BEFORE spending an eval:
- (§7) stock `fcfs` re-matches every server-waiting req each step → LRU already keeps at-server prefixes hot.
- (§8) load-back is per-layer overlapped → prefetch is marginal.
- residual warm-miss headroom lives in the CLIENT-blocked window (offered load > concurrency 256), which is
  invisible to the server; the only server-side predictor of client-blocked continuation is turn-count
  (=hit_count) → LFU/SLRU, which is non-novel and prior campaigns found neutral/harmful.
⇒ A NOVEL, capturable residency/transfer win looks unlikely. Honest leading outcome = a **rigorous
bounded-negative + characterization** ("in 2-tier no-disk HiCache under concurrent closed-loop multiturn,
the cache is not the goodput@SLO lever; the p99 limiter is prefill congestion / cold-whale HOL, bounded
by X; residency headroom is either already captured by stock schedule-time matching or non-capturable
because the vulnerable window is client-side"). This is charter-valid IF rigorously demonstrated.
**Still gated on baseline aggregates** (hit% vs λ, load_back vs λ, p50/p90/p99 vs λ) — if they surprise
(e.g. hit collapses under load in a server-capturable way), revisit for a mechanism. Next eval after the
reference will likely be a stock+trace DIAGNOSTIC to decompose the p99 tail and nail the limiter.

## 10. Workload reuse structure (from conv_trace.json — motivation data, holds regardless of fork)

- **39.0% of conversations are single-turn** (605/1553); avg 4.53 turns, median 3, max 61. Single-turn
  turn-0 docs (esp. the huge summarization inputs) are cached but NEVER reused = cache pollution.
- **Continuation hazard:** P(continue|reached turn1)=0.61, but **0.75–0.89 for turns ≥2** → hit_count≥1
  strongly predicts continuation. Exploiting this = SLRU/LFU/GDSF (textbook; prior campaigns: neutral/harmful).
- **Reuse ceiling = 80.6%** of prefill tokens are within-conv-reusable; **top-10% of convs hold 64.6% of
  the reuse value** (concentrated in long convs). Baseline hit 0.62 ⇒ headroom exists but the exploiting
  policies are all textbook, so a NOVEL capturable residency win is unlikely (converges with §9).
- Interpretation: the exploitable structure is real, but every lever for it is either already-in-stock,
  per-layer-overlapped, client-side-invisible, or a textbook policy ⇒ strengthens the bounded-negative
  hypothesis; the rigorous question the baseline settles is WHAT actually bounds goodput@SLO under load.

## 11. ★ SHARP FORK CRITERION — prefill-throughput vs SLO (sim/prefill_tail.py)

Under PERFECT within-conv cache, **99% of irreducible prefill work is cold turn-0 documents** (1% is later
turns). Per-turn uncached-prefill distribution: PERFECT-cache p99=39K tok (max 193K); NO-cache p99=58K tok.
So whether p99 TTFT is IRREDUCIBLY over the 8s SLO depends on the real prefill throughput P:
| P (tok/s) | 8s budget | %turns w/ PERFECT-cache prefill >8s | verdict |
|-----------|-----------|-------------------------------------|---------|
| 4000  | 32K  | 2.23% (>1%) | p99 IRREDUCIBLE → cache can't raise goodput@SLO (BOUNDED-NEGATIVE) |
| 8000  | 64K  | 0.16%       | cache CAN affect p99 → mechanism viable |
| 15000 | 120K | 0.03%       | cache CAN affect p99 → mechanism viable |

**Avoidable-recompute headroom** (if cache-affectable): baseline uncached ≈37.8M tok (hit0.62) vs perfect
≈19.3M → **~18.5M tok (~49% of baseline prefill) is avoidable recompute** = warm-turn misses a finite-cache
policy could reduce. (Belady-finite < perfect-infinite; my sim put finite headroom ~23%.)

⇒ **The FORK is decided by measuring P from the baseline** (curve.csv λ=3 + server.log per-batch prefill
timing). This resolves at the FIRST rate (~40 min into the run), not the full sweep. Plan:
- estimate P = (Σ #new-token over prefill batches)/(prefill wall time) from server.log at λ=3.
- if p99(cold-prefill) > 8s for >~1% of turns → **bounded-negative** (characterize + a couple policy
  controls showing eviction policy doesn't move goodput → cache not the lever; p99=cold-context-tail-bound).
- else → **mechanism**: target the ~18.5M avoidable recompute (finite-cache residency beating LRU under
  load) — but must beat textbook SLRU/GDSF/LFU novelly (open problem; may still be negative if only
  turn-count-predictable). Keep both live until P is known.

## 12. Sim v2 (PS + stock-protection modeled, sim/simulate2.py) — residency headroom is real but textbook

Congested processor-sharing sim that models stock's at-server protection (a conv with a req at-server keeps
its prefix recency-refreshed, mirroring fcfs per-step match). Findings (robust to P sweep 15K/40K, λ 3/5/10):
- **Stock protection cuts LRU recompute ~5.8M→~2M** vs v1 (no-protection) — confirms §3 empirically.
- **Belady eliminates the residual ~2M entirely (OPT≈0 avoidable):** the workload has ABUNDANT dead KV
  (39% single-turn whales + finished convs), so an oracle never evicts a live prefix — LRU's whole residual
  cost is DEAD/LIVE confusion (keeps recently-finished dead whales, evicts older live multi-turn convs).
- That headroom is exactly **dead-vs-live-distinguishable = hit_count/size-predictable = SLRU/GDSF**
  (textbook; prior campaigns neutral on the single-point eval). ⇒ real headroom, NOT novel-capturable.
- Whether reducing this warm-turn recompute moves **goodput@SLO** depends on §11's fork: if p99 is
  cold-turn-0-tail-bound, warm-miss reduction helps p50/p90 but NOT p99/goodput (→ negative on goodput);
  if p99 is cache-affectable, an SLRU/GDSF-style policy would move it (but that's a config/textbook win,
  not a contribution). Either way, a NOVEL top-venue mechanism looks unlikely; the contribution is the
  characterization + the P-vs-SLO criterion + (pending baseline) the measured verdict.

## 13. ★★ FORK RESOLVED (diagnostic 19452 on nodeset-0) — P≈35K tok/s → CACHE-AFFECTABLE

Off-contract diagnostic (idle non-certified nodeset-0, NUMP=400, λ=3) — steady-state over 164 real prefill
batches: **prefill throughput P ≈ 35,000 tok/s** (input-throughput p50 35.5K/p90 37.3K; the 25–119 tok/s
first-3-batch reading was warmup, idle-diluted — do NOT use it). Timestamp method floors at 6144 (1s log
granularity), so trust the input-throughput field under load.

Apply §11 criterion: 8s×35K = **280K-token budget > max document 192.7K** ⇒ **NO single request's cold
prefill exceeds the SLO** (max doc = 5.5s single-request). ⇒ p99 TTFT under load is **queue/load-driven,
not single-request-irreducible ⇒ CACHE-AFFECTABLE.** The pure "irreducible cold tail" bounded-negative is
REFUTED. Reducing total prefill work (fewer warm-turn misses) reduces queueing ⇒ can lower p99 ⇒ raise
goodput@SLO. **Mechanism branch is LIVE**; target = the ~18.5M-token (~49%) avoidable recompute.

**Revised plan:** (1) certified baseline (19437, NUMP=1553, pressured) → real hit/goodput/p99 anatomy;
(2) resolve whether the avoidable recompute is capturable ONLY by textbook LRU-variants (SLRU/GDSF — then
the contribution is the methodological/characterization result: "eviction is a dead end" is FALSE for
goodput@SLO-under-load though true at a single point) OR whether a novel residency/admission signal beats
them (stronger). Test via `--radix-eviction-policy {lru,slru,lfu}` sweeps (allowed) + a candidate mechanism.
Caveat: P from a non-certified node; certified P may differ ±, but 35K ≫ 8K threshold so the fork is robust.

## 14. Diag trace (rid-dedup) — the COLD-DOC FLOOR refinement (validates tooling; NUMP=400 preview)

Per-request trace (SGLANG_WILKES_TRACE works; chunked prefills log multiple entries → dedup by rid, first
entry = true match). Diag NUMP=400 (under-pressured), 1929 reqs, hit **0.818** (≈ the predicted under-pressured
0.84). Request-class split (by prefix-match fraction):
- **COLD (turn-0, match<5%): 24% of reqs but 93.9% of prefill WORK** — irreducible, each unique doc once.
- WARM-HIT (unc<10%): 63% of reqs, 1.5% of work.
- **WARM-MISS (partial, AVOIDABLE): 13% of reqs, 4.6% of work** (at low pressure).

**Refinement of the fork:** even though P≈35K makes p99 not single-request-irreducible, the prefill WORK has
an irreducible **cold-document floor** (Σ unique docs ≈ 19M tok = 94% of low-pressure prefill). The cache can
only reduce the *warm-miss* fraction. At P=35K, prefilling the 19M cold floor ≈ 543s of pure prefill vs a
~704s arrival window at λ=10 ⇒ cold docs ALONE near-saturate prefill at high λ. So goodput@SLO headroom is
bounded by the warm-miss fraction, which grows with pressure. **screen-v0 (NUMP=1553) quantifies the pressured
warm-miss% = the decisive number:** large ⇒ mechanism materially helps; small ⇒ cold-doc-floor-bound
(bounded-negative). This is the crisp, quantified thesis either way: *goodput@SLO is set by the cold-context
prefill floor; the cache addresses only the avoidable warm-miss remainder (X% under pressure).*

## 15. Diag λ=3 result (CONFOUNDED — no warmup) — cold-doc tail signal

Diag λ=3 (NUMP=400, nodeset-0): req/s 2.72, **median TTFT 1846ms but p99 TTFT 48,552ms** (26× median).
⚠️ CONFOUND: my diag.sh omitted the warmup burst that official eval.sh runs (WARMUP_NUMP=300) to kill
cold-start metastability → this p99 is cold-start-inflated, NOT directly comparable to official (old
baseline.json λ=3 p99=6326ms WITH warmup + conc128). So treat 48.5s as an upper bound, not the real p99.

SIGNAL (despite confound): the huge tail with only 4.6% warm-miss work ⇒ **p99 is driven by long COLD-DOC
prefills under concurrency, NOT cache misses.** If this survives warmup (screen-v0), goodput@SLO is
cold-context-bound (bounded-negative), because 1253/1553 conversations' cold turn-0 docs are never warmed and
arrive during the measured sweep. DECISIVE test = screen-v0 (eval.sh WITH warmup + NUMP=1553): does warmup
bring p99 under 8s (⇒ cache-affectable), or does the cold-doc tail persist (⇒ cold-bound negative)?
LESSON: future diagnostics must include a warmup burst.
- λ=7 diag (same confound): req/s 4.82, median 2063ms, **p99 45,686ms** — p99 is ~RATE-INDEPENDENT (48.5s@λ3, 45.7s@λ7) with low medians ⇒ tail = fixed set of extreme cold-doc prefills, NOT queue growth (which would grow p99 with λ). Signature of R_p99≪R_thru (cold-doc-bound) — but confounded by no-warmup; screen-v0 (warmup) disambiguates cold-start-spike vs warmup-resistant cold-doc-tail.

## 16. Analytical spine — the two-bound goodput model (validate w/ screen-v0)

**goodput@SLO = min(R_thru, R_p99):**
- **R_thru** = prefill-throughput bound = P / avg_uncached_prefill_per_req. Cache REDUCES avg_uncached ⇒
  RAISES R_thru. At P≈35K: baseline (hit 0.62, avg 5372 tok) ⇒ R_thru≈6.5 req/s; perfect cache (avg 2743
  tok) ⇒ R_thru≈12.8 req/s. So cache could ~2× the throughput bound.
- **R_p99** = max rate with p99 TTFT ≤ 8s = set by the cold-doc tail under concurrency. Cache does NOT
  shrink the largest cold-doc prefills ⇒ does NOT raise R_p99 (except indirectly via queue).

**The fork = which bound binds:**
- If **R_p99 < R_thru** (p99 blows past 8s BELOW throughput saturation, i.e. a few cold whales tail-out
  even at low load) ⇒ goodput is p99/cold-doc-bound ⇒ **cache can't raise goodput ⇒ BOUNDED-NEGATIVE.**
- If **R_p99 ≈ R_thru** (p99 blows up only AT saturation = queue buildup) ⇒ cache (raising R_thru) delays
  saturation ⇒ raises goodput ⇒ **MECHANISM viable** (capture the warm-miss recompute).
- Caveat: R_p99 and R_thru are coupled via the queue; the cold-doc tail is what can DECOUPLE them (blow p99
  before saturation). The diag λ=3 (req/s 2.72 ≪ R_thru 6.5, yet p99 48.5s) HINTS R_p99<R_thru — but that
  p99 is cold-start-confounded (no warmup). **screen-v0 (warmup) measures the real p99-vs-λ curve ⇒ reads
  off R_p99 and compares to R_thru ⇒ resolves the fork.** This min() model + its validation is the paper's
  quantitative core (predicts goodput from workload-context-distribution + P, cache entering only via R_thru).

## Versions (test submissions)
- **v0-baseline** (stock sweep, clean reference) — job 19437, QUEUED. [pending curve]

## Formal submissions
- (none yet)

## 17. ★ PRESSURED PREVIEW (screen-v0 live trace, warmup+8% of λ=3) — supports cold-doc-bound

From the live SGLANG_WILKES_TRACE under NUMP=1553 pressure (hit dropping 0.83→**0.57** as working set fills):
- **COLD turn-0 docs = 47% of reqs but 95% of uncached WORK**; WARM-HIT 41.6%reqs/0.6%work;
  **WARM-MISS (avoidable) = 11.4% reqs but only 4.4% of uncached work.**
- ⇒ Even under building pressure, avoidable warm-miss recompute is ~4-5% — because **stock schedule-time
  matching already protects active convs' prefixes** (empirically confirms §3). The irreducible cold-document
  floor dominates (95%). A residency/eviction mechanism can only touch ~4-5% of prefill work ⇒ minimal
  goodput headroom ⇒ **leading verdict: goodput@SLO is COLD-DOCUMENT-BOUND (bounded-negative).**
- PENDING: the λ=3 p99 (curve.csv) confirms goodput@SLO (R_p99); full sweep + resolve_fork.py finalize.
  Caveat: partial (8% of λ=3); warm-miss% may shift over the full sweep, but it's stable vs NUMP=400 (4.6%).

## 18. ★★ VERDICT CONFIRMED (screen-v0 pressured trace) — BOUNDED-NEGATIVE (cold-document-bound)

Under full NUMP=1553 pressure (hit 0.549, avg_uncached 5980 tok, P 34,854 tok/s):
- **p99-tail requests are 100% COLD turn-0 documents, 0% WARM-MISS** (same as under-pressured diag).
- COLD = 49%reqs/95%work; WARM-MISS (avoidable) = 11%reqs/**5%work**, and NOT in the p99 tail.
- R_thru = P/avg_uncached ≈ **5.83 req/s**.

**Conclusion:** eviction/residency/prefetch mechanisms can only reduce WARM-MISS recompute (≤5% of work,
absent from the tail). The goodput@SLO p99 tail is set entirely by irreducible COLD-DOCUMENT prefills under
concurrency, which no lossless cache mechanism can shrink. ⇒ **goodput@SLO is cold-document-bound; the cache
is NOT the goodput lever in this regime.** This is a rigorous, per-request-validated bounded-negative,
consistent with the two-bound model (§16): cache raises R_thru but the binding bound is R_p99, set by the
cold-context tail. Stock schedule-time matching (§3) already protects warm prefixes, so there is no residency
headroom to capture.

**Remaining to finalize (numbers, not direction):** (1) screen-v0 curve.csv λ=3..10 p99 → the goodput@SLO
value (R_p99) + the full curve; (2) OPTIONAL eviction-policy control (--radix-eviction-policy slru/lfu) to
show goodput invariant to eviction (empirically airtight, though structurally implied); (3) CERTIFIED
confirmation (job 19437) for official numbers. Then finalize the characterization/impossibility paper:
"goodput@SLO for long-context multiturn on a tiered KV cache is bounded by the cold-document prefill tail,
not cache efficiency; lossless cache mechanisms are confined to R_thru (p50/p90), the tail is R_p99."

## 19. ⚠️ RETRACTION of §18 verdict — CLASSIFICATION FLAW (integrity)

§18 ("p99-tail 100% COLD => bounded-negative") is **PREMATURE / RETRACTED**. resolve_fork.py's "COLD"
class = prefix-match <5% of prompt, which **conflates two very different things**:
(a) genuine turn-0 documents (first request of a conversation — never cacheable, irreducible), and
(b) fully-EVICTED later turns (turn-N whose entire accumulated prefix was evicted mid-conversation → the
    request recomputes its whole 40K+ history — this is AVOIDABLE, exactly what a better cache would prevent).
Both have match≈0, so my tool mislabeled (b) as COLD/irreducible.

Tell-tale: COLD was 24% at low pressure (diag NUMP=400 ≈ the 22% genuine-turn-0 fraction) but 49% under
pressure (screen-v0) — the extra ~25% are almost certainly **evicted later turns (avoidable)**, NOT cold
docs. If those dominate the pressured p99 tail, the cache CAN help ⇒ the verdict may be MECHANISM-VIABLE,
not bounded-negative. Also confound: the screen-v0 trace read was only 8-10% into λ=3 (turn-0-heavy startup),
inflating COLD%.

**FIX (in progress):** added a conversation prefix-hash `chash` (hash of first 48 origin tokens; a conv's
turns share it) to the trace (scheduler.py). Offline: first request per chash = genuine turn-0; a repeat
chash with low match = evicted later turn (avoidable). A pressured chash-enabled run + steady-state (full
λ=3, not the startup) will correctly classify the p99 tail. Verdict is RE-OPENED pending that run.
LESSON: match-fraction alone cannot distinguish irreducible-cold from avoidable-evicted; need conv identity.

## 20. Corrected analyzer validated (v0b chash trace, UNDER-pressured warmup) — nuanced signal

chash-aware trace_analyze on v0b's warmup+early trace (hit 0.63, under-pressured): overall COLD-TURN0
41.8%reqs/92%work, WARM-HIT 41.8%/0.7%, **EVICTED (avoidable) 16.5%reqs/7.4%work**. Crucially, the class
mix among BIG (SLO-breaching-size) prefills: **T≥40K = 100% COLD-TURN0, 0% EVICTED**; T≥20K = 93% COLD /
7% EVICTED. So even correctly classified, EVICTED (avoidable) turns are SMALLER than genuine cold docs and
don't reach the SLO-breaching size — they'd cut p50/p90, not the p99 tail.
⇒ This is a CAREFUL, corrected bounded-negative signal (big/tail prefills = genuine cold docs; the avoidable
recompute is real but sub-tail-size). CAVEAT: UNDER-PRESSURED (warmup). DECISIVE test = full-pressure λ=3:
do long-conversation full-evictions (accumulated 40-190K) grow into the ≥40K bucket and enter the p99 tail?
If yes ⇒ cache-affectable; if the tail stays genuine-cold-docs ⇒ corrected bounded-negative confirmed.
The correction MATTERS methodologically (separates avoidable-evicted from irreducible-cold) even if the
verdict lands negative — it's the right way to make the claim.

## 21. ★ TREND REVERSAL as pressure builds (v0b chash, early λ=3) — verdict trending CACHE-AFFECTABLE

As λ=3 pressure begins (hit 0.82→0.79, still ramping to ~0.55 steady-state), the corrected chash
classification SHIFTS toward avoidable recompute:
- warmup (under-pressure): EVICTED 8% of work, 0% of ≥40K big/tail prefills.
- early λ=3: EVICTED **23% of work**, and **20% of ≥40K big prefills are EVICTED** (avoidable) — evicted
  later-turns are STARTING to reach SLO-breaching size and enter the tail.
⇒ The flawed match-only analysis (retracted §18) would have hidden this (called them "COLD"); the chash fix
reveals avoidable recompute GROWING under pressure and entering the tail. If steady-state confirms a large
EVICTED-in-tail share ⇒ **CACHE-AFFECTABLE → mechanism-viable** (a residency policy protecting proven
multi-turn convs would cut those evictions → lower p99 → higher goodput). This REVERSES the earlier
bounded-negative lean. Waiting for λ=3 steady-state (hit~0.55, ~15 min) to confirm the tail EVICTED fraction.
PLAN if confirmed: screen --radix-eviction-policy slru (protect hit_count>=1 proven convs; no code) as the
first mechanism test; if it moves goodput, design a novel continuation/cost-aware residency to beat it.

## 22. ★★★ VERDICT: CACHE-AFFECTABLE / MECHANISM-VIABLE (corrected, chash)

As λ=3 pressure builds (hit 0.82→0.79→0.67, → ~0.55 steady-state), EVICTED(avoidable) recompute grows
MONOTONICALLY: 8%→23%→**47% of prefill work**; EVICTED share of big/tail (≥40K) prefills: 0%→20%→**38%**
(n=21). COLD-TURN0 stable ~21% (=genuine turn-0). ⇒ **~half the pressured prefill work is avoidable recompute
from evicted PROVEN multi-turn conversations, and they are a large fraction of the SLO-breaching tail.**
The cache CAN move goodput ⇒ MECHANISM-VIABLE. (The retracted match-only analysis hid this by mislabeling
evicted turns as cold — the chash fix was decisive; integrity check paid off.)

**Mechanism direction:** protect PROVEN multi-turn convs (hit_count≥1) from eviction; evict single-turn
"whale" turn-0s (hit_count=0, 39% of convs, never reused) first — freeing capacity for proven convs across
their think-gaps. This shrinks both bounds: fewer avoidable recomputes → higher R_thru; fewer big evicted
turns in the tail → lower p99 → higher R_p99.
Plan: (1) v0b (lru) finishes = baseline curve; (2) screen slru (hit_count segments) + lfu (evict fewest-
reused) — existing policies, no code, directly protect proven convs; (3) if a policy moves goodput, that
overturns "eviction is dead" for goodput@SLO-under-load; (4) design a NOVEL continuation/cost-aware residency
to beat textbook (protect proven convs weighted by recompute-cost/continuation, targeting the p99 tail);
(5) certified confirmation. Paper pivots from bounded-negative to a mechanism + the two-bound analysis.

## 23. ⚠ MAGNITUDE CORRECTION — §22's 47%/38% was a transient peak (read partial data too eagerly)

The EVICTED fraction MOVED as λ=3 progressed and pressure settled to steady-state (hit → 0.55):
  n=1865 hit0.674: EVICTED 47%work / 38% of ≥40K tail   (transient eviction burst, working set first oversubscribing)
  n=2468 hit0.550: EVICTED 29%work / **18% of ≥40K tail** (COLD-TURN0 37%)  ← steady-state, more reliable
⇒ CACHE-AFFECTABLE is CONFIRMED (evicted proven convs are a SUBSTANTIAL ~18-29% of tail/work, vs ~0% a pure
bounded-negative would show), but the magnitude is SMALLER than §22's peak and still settling. Do NOT quote
47%/38%; the settled figure (~29%work / ~18%tail so far) needs the FULL λ=3. LESSON: stop reading partial-run
fluctuations as results — wait for the full sweep. NOTE also: mid-run COLD-TURN0 (37%) exceeds the 22%
genuine-turn-0 fraction (chash first-occurrence noise from warmup/no-flush re-appearance) — another reason to
trust only the completed run. Honest current read: cache CAN help (a residency mechanism protecting proven
convs can cut ~18-29% avoidable recompute), gain to be measured by the policy screens on the FULL sweep.

## 24. ★ Think-gap distribution (v0b chash+timestamps) — reshapes the mechanism + expectations

Inter-turn gaps (time between a conv's consecutive prefill-admissions, proxy for think-gap) from 713
multiturn convs / 2298 gaps: **p50=54.6s, p75=458.6s, p90=606s, mean=223s** (only 30% <30s; 53% <60s;
60% <120s). These gaps are HUGE and QUEUE-dominated (turn N+1 waits in the re-enqueue/client backlog under
load, not client think-time).
Implications:
- CAR grace=30s protects only 30% of reuses — TOO SHORT. Raising grace toward p75 (458s) over-protects
  (holds KV minutes → evicts others, protects dead single-turn whales too long → ≈LRU). No single grace fits
  the wide gap distribution. Set CAR grace to ~90s (covers median + margin; ~55% of reuses) for the screen.
- The avoidable evicted recompute is PARTLY CAPACITY-FORCED: holding a conv's KV for 55-600s under 1.89×
  oversubscription is hard, so a residency mechanism can only capture the SHORTER-gap reuses. This tempers
  the expected goodput gain (the mechanism helps, but the long-gap tail is capacity-limited).
- SLRU (protect proven hit_count>=1 indefinitely, evict single-turn whales) may suit large gaps better than
  a grace-pin, BUT it misses the turn-0->turn-1 first reuse (turn-0 hit_count=0 evicted during the long gap;
  61% of turn-0s continue). CAR-with-adequate-grace can capture that first reuse — its distinguishing value.
The screens (lru/slru/car) will EMPIRICALLY measure which helps goodput and by how much.

## 25. ★★★ REAL BASELINE (v0b lru, FULL λ=3 completed) — goodput@SLO=0, tail 63% AVOIDABLE

v0b (stock lru) λ=3 COMPLETED (nodeset-0, warmup+NUMP1553): req/s 2.89, p50 876ms, **p99 TTFT 11,254ms
(>8s SLO)**, hit 0.679 ⇒ **GOODPUT@SLO = 0** (even the lowest swept rate breaches the 8s SLO). R_thru=7.81
req/s ≫ R_p99(=0), so p99 is the hard binding constraint.
Full-λ=3 trace (n=8517, reliable steady-state — NOT a mid-run partial): COLD-TURN0 19%reqs, WARM-HIT ~56%,
**EVICTED(avoidable) 25%reqs / 58% of prefill WORK; and of the ≥40K SLO-breaching TAIL: 63% EVICTED / 37%
COLD-TURN0.** ⇒ the p99 tail is DOMINATED by avoidable evicted-proven-conv recompute. Strong cache-affectable.
CORRECTION: my §23 "over-correction" to 29%/18% was itself a mid-run partial artifact; the FULL rate is
58%/63%. LESSON (again): only trust COMPLETED rates.
**Mechanism target (crisp):** cut the evicted recompute (63% of tail) to bring λ=3 p99 from 11.25s under 8s →
goodput 0 → 2.89+ (a categorical win). OPEN: does removing evicted turns bring p99<8s, or does the 37%
cold-doc floor keep p99>8s? The slru/car90 screens (next in pipeline) answer it directly.

## 26. ★ MECHANISM RESULT #1: SLRU HURTS (motivates CAR's turn-0 probation)

λ=3 equal-pressure comparison (nodeset-0 screening):
| policy | hit | p50 TTFT | p99 TTFT | EVICTED %work | EVICTED %of≥40K-tail | goodput@SLO |
|--------|-----|----------|----------|---------------|----------------------|-------------|
| lru    | 0.679 | 876ms  | 11254ms | 58% | 63% | 0 |
| slru   | 0.507 | 1498ms | 10091ms | **73%** | **83%** | 0 |
SLRU (protect hit_count>=1) is WORSE: hit 0.68→0.51, p50 876→1498ms, EVICTED 58→73%work / 63→83% tail.
WHY: protecting "proven" convs evicts UNPROVEN turn-0s more aggressively — but turn-0s are the ENTRY to
multi-turn reuse (61% continue), so sacrificing them creates MORE evicted turns. Confirms prior campaigns'
"SLRU/LFU neutral-or-harmful" AND directly motivates CAR: protect turn-0s with a grace PROBATION (capture the
turn-0->turn-1 first reuse) while still protecting proven convs. Both lru & slru goodput@SLO=0 (p99>8s at λ=3);
the open question is whether CAR (3-segment) can pull p99<8s. Screening car90 next.

## 27. car90 first attempt FAILED on argparse (fixed) — resubmitted same-node

car90 (job 19470) died instantly: `--radix-eviction-policy: invalid choice: 'car' (choose from lru,lfu,slru,priority)`.
CARStrategy was registered in the FACTORY (utils.py) but server_args.py's argparse `RADIX_EVICTION_POLICY_CHOICES`
(line 281) is a hardcoded curated subset that gates the CLI BEFORE the factory. Added "car" to that list (commit).
PYTHONPATH=$WORK/python resolves via the v0.3_ablations->v0.31 symlink to the edited file. Resubmitted as job 19480,
PINNED to nodeset-0 for a clean same-node A/B vs my lru (screen-v0/v0b) & slru (screen-slru) baselines there
(node variance ±14% req/s / ±30% p99 is the #1 confound for a mechanism claim — same-node is mandatory).
Cancelled redundant certified-lru job 19437 (screen lru already IS a full eval.sh run w/ summary.json; it was
competing for nodes). car90 waits behind sibling base-v1x holding nodeset-0 (~3h). grace=90 (covers think-gap p50 55s;
protects turn-0->turn-1 first reuse that slru evicts). If car90 wins at λ=3, full curve + certified confirm follow.

## 28. ★ OFFLINE BELADY HEADROOM = 100% at real cap (liveness, not capacity) + sim timing-blindness

Extended sim/simulate.py trace_headroom (fixed served-access-order replay through different eviction victims;
timing-INDEPENDENT, isolates victim choice). At real 2-tier cap 10.7M tok: LRU recompute 7.24M, **Belady
(evict-farthest-future) recompute = 0 → 100% avoidable**. Reason: 39% single-turn whales (read once) + completed
convs = always dead KV to evict; the LIVE (will-be-reused) working set FITS even though TOTAL is 1.89×. Headroom
monotone in cap: 59%@4M, 78%@6M, 95%@8M, 100%@10.7M, 100%@15M → at real cap it's a LIVENESS problem (LRU can't
tell live from dead), not capacity. Belady is offline (sees future); causal policy can't hit 0 → this is the
CEILING; GPU eval measures what CAR captures. This CORRECTS the retracted §4 claim ("headroom ≤5%/not-in-tail"
was pre-chash) — consistent now with §5.2 (avoidable recompute IS 63% of the tail).

★ SIM LIMITATION (why I did NOT put slru/car replay numbers in the paper): the fixed-order replay gives
lru==slru==car EXACTLY (same victims, same 7.24M) because the compressed access order has no wall-time think-gaps
— the oldest-touched resident conv is already an unproven whale, so LRU and SLRU pick identical victims. The GPU
SLRU-backfire is driven by wall-time gap AGING (proven convs' KV idle for MINUTES → become LRU victims), which a
timing-independent replay cannot reproduce. So the replay corroborates the HEADROOM (opt=0) but is BLIND to the
slru/car distinction; the GPU eval is the only valid test of slru vs car (as already measured: slru HURTS §26).

## 29. CAR's expected ceiling (interpretation prep for car90) — short-gap vs long-gap avoidable

Belady gets 100% of the 7.24M avoidable (§28) using the FUTURE. An online policy cannot. Reasoning about CAR's
reachable fraction of the 63%-avoidable tail:
- CAR segment-2 protects ALL proven convs indefinitely (LRU within). A proven conv that has COMPLETED is DEAD
  but still segment-2. Fine WHILE segment 0/1 (dead whales + expired turn-0s) has evictable KV; but once that's
  exhausted, segment-2 LRU evicts the OLDEST-accessed proven — which for a LONG think-gap (p75=458s) could be a
  LIVE conv mid-gap, not the completed one. Recency alone cannot separate live-old from dead-old proven.
- So CAR captures the SHORT-gap avoidable recompute (protect recent turn-0s [seg1] + recently-active proven
  [seg2 top]) but NOT the long-gap avoidable (can't hold minute-long-gap KV under 1.89x oversubscription, and
  can't tell live-old from dead-old). Expected CAR gain = short-gap fraction of the 63%, NOT the full 63%.
- ⇒ HONEST prediction for car90: p99 likely improves over lru (captures short-gap evicted turns) but may NOT
  reach <8s if the tail's biggest evicted turns are long-gap (unreachable). The result quantifies the online-
  reachable fraction. A future predictor (per-conv inter-turn-gap history → protect convs whose next turn is
  near) could reach more, but is speculative; CAR is the clean principled first mechanism. This is the paper's
  honest discussion regardless of car90's sign.

## 30. ★★★ DECISIVE ANALYSIS: the headroom is liveness-bound but ONLINE-UNRECOVERABLE (paper's core)

Trace-driven (lru screen-v0b, λ=3+warmup, 997 multi-turn convs), three measured facts that together characterize
WHY the 63%-avoidable tail resists causal capture:

1. **turn-0→turn-1 gap is HUGE: p50=460s, p75=559s, p90=614s, max=1322s** (7.7 min median!). grace-90 covers
   only **14.3%** of these gaps; grace-180 19.5%; grace-300 23.7%. (Later turn≥1 gaps are shorter: p50 92s.)
   turn-0 is a long doc → long E2E response + think + requeue before turn-1.
2. **The real-time live working set FITS: peak concurrent live KV = 9.3-9.4M tok = 0.87-0.88× cap** (10.7M),
   bracketed lower(plen_i)/upper(plen_{i+1}). 902 convs simultaneously in-gap at peak. So it is DEFINITIVELY
   LIVENESS-bound, not capacity-bound: an offline policy holding exactly the live set achieves ~0 recompute
   (Belady confirms, §28). NOT a capacity problem.
3. **But live/dead is UNOBSERVABLE at the decision point (turn-0).** A live turn-0 (will reuse in ~460s) and a
   dead single-turn whale (never reuses; 39% of convs) are BOTH unproven (hit_count=0) at turn-0 time. The only
   online continuation signal is turn-count (=hit_count), which is 0 for ALL turn-0s. So a causal policy cannot
   tell them apart.

★ THE ONLINE BARRIER (quantified): to hold a live turn-0 until reuse needs grace ≥ ~460s. But grace-460 ALSO
holds every dead whale for 460s; at λ=3 that accumulates ~8M of dead-whale KV in the window, which + the 9.3M
live set ≫ 10.7M cap → swamps cache → forces eviction of live KV → FAILS. Short grace → doesn't cover the 460s
gap (live reuse lost) → ≈lru. So the grace knob is trapped: small=no coverage, large=capacity swamp. Belady wins
ONLY because it knows the future (holds the 9.3M live, evicts dead) — knowledge no causal policy has at turn-0.

⇒ **PREDICTED VERDICT (car90 will confirm): no causal lossless residency policy moves goodput@SLO here.** CAR
(grace) and SLRU (proven) both fail, for complementary reasons. This is a RIGOROUS BOUNDED-IMPOSSIBILITY for
causal residency — DISTINCT from the textbook "eviction≈Belady, dead under in-order reuse": here reuse is
out-of-order with 100% liveness-headroom, yet still online-unrecoverable because liveness is unobservable at the
decision point + the reuse gap forces all-or-nothing under capacity. Top-venue-worthy as stated. car90 (grace 90,
=small-grace point ≈lru) confirms one end; a large-grace run (swamp) would bracket the other. HONEST + strong.

## 31. ★ GRACE-TRAP QUANTIFIED (airtight, ~5× feasible-vs-needed gap) + predicts car90≈lru

Peak grace-forced resident KV (all convs touched in trailing G-sec window, since grace can't tell live/dead):
  grace  90s:  9.0M = 0.85x cap  FITS  (covers 14% of turn0->turn1 gaps)
  grace 180s: 14.9M = 1.39x cap  SWAMP (20%)
  grace 300s: 21.1M = 1.97x cap  SWAMP (24%)
  grace 460s: 27.7M = 2.59x cap  SWAMP (50% — the median gap)
  grace 600s: 34.4M = 3.21x cap  SWAMP
Largest grace that FITS (~90s, 0.85x) covers only 14% of gaps → behaves ≈lru. Covering the p50 460s gap needs
grace≥460 → 2.6x swamp → evicts live KV → fails. ~5× gap between feasible-grace-ceiling (~90-120s) and
gap-floor (460s). NO grace both covers & fits ⇒ grace-trap is QUANTITATIVE + airtight. ★ PREDICTS car90 (grace
90, fits 0.85x) ≈ lru (goodput 0, p99~11s) — car90 is the CONFIRMING measurement of this predicted horn.

## 32. ★★ car90 λ=3 RESULT — PREDICTION CONFIRMED: CAR (grace 90) ≈ LRU (small-grace horn)

| policy (λ=3, same node nodeset-0) | req/s | p50 TTFT | p99 TTFT | hit | goodput@SLO |
|-----------------------------------|-------|----------|----------|-----|-------------|
| lru                               | 2.89  | 876ms    | 11254ms  | 0.679 | 0 |
| slru (protect proven)             | 2.78  | 1498ms   | 10091ms  | 0.507 | 0 |
| **car (grace 90)**                | 2.97  | 722ms    | **11174ms** | **0.6792** | **0** |

★ CAR grace-90 ≈ LRU EXACTLY: p99 11174 vs 11254ms (−0.7%, noise), hit 0.6792 vs 0.679 (identical), p50 slightly
better (722 vs 876). Unlike SLRU, CAR does NOT hurt (hit intact). goodput@SLO=0 (p99≫8s). EXACTLY the §4.2/§31
prediction: grace 90 FITS cache (0.85x) but covers only 14% of the 460s turn0->turn1 gaps → no measurable effect
→ degrades to LRU. This is the SMALL-GRACE horn of the grace-trap, empirically confirmed. Next: car300 (grace
300, 1.97x swamp) = the LARGE-GRACE horn (predicted to HURT). Both horns → the bounded-impossibility is airtight.

## 32b. car90 tail decomposition (chash, uncached-tail) — mechanistically ≈ lru
Consistent method (tail = uncached≥40K): lru EVICTED 61%work/65%tail; slru 73%/83%; car90 65%/69%.
car90 ≈ lru (both far below slru's 83%). CAR grace-90 does NOT reduce the evicted tail → captures ~none of the
avoidable recompute (grace 90 covers only 14% of the 460s gaps). Confirms §32 mechanistically. paper §5.3 rows
now consistent (all uncached-tail method). Next: car300 (19502, swamp horn).
