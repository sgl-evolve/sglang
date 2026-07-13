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

## 33. Deliverables status (2026-07-12 ~10:08)
- paper.html: COMPLETE bounded-impossibility (10 sections §1-9 + abstract; Belady figure; grace-trap table; §5.3
  lru/slru/car90 λ=3; §5.4 controls; §9 conclusion). Registered in submissions/INDEX.md.
- W&B (sgl-evolve/wilkes): logged baseline v0_official + v_lru_l3/v_slru_l3/v_car90_l3 (all goodput@SLO=0) =
  the bounded-impossibility curve. churn py net 76 (CARStrategy+hooks+argparse).
- car300 (19502, swamp horn) queued behind sibling eval-base-vca on nodeset-0 (~2.5h); cron 1d70e5e7 catches.
- Certified confirmation + full rate curves: contended/pending (screening on nodeset-0 is valid; result
  categorical goodput=0). Core result COMPLETE; remainder = refinements.

## 34. ★★ car300 λ=3 — HONEST CORRECTION: no catastrophic swamp; CAR neutral across feasible grace

car300 (grace 300) λ=3: req/s 3.00, p50 678ms, p99 10430ms, hit 0.6756, goodput@SLO=0.
Full comparison (λ=3, same node):
| policy      | req/s | p50   | p99     | hit    | goodput |
| lru         | 2.89  | 876   | 11254   | 0.679  | 0 |
| slru        | 2.78  | 1498  | 10091   | 0.507  | 0 |
| car90(g90)  | 2.97  | 722   | 11174   | 0.679  | 0 |
| car300(g300)| 3.00  | 678   | 10430   | 0.6756 | 0 |

★ CORRECTION to my §31 prediction: grace 300 did NOT catastrophically swamp (hit stayed 0.676 ≈ lru, NOT
slru's 0.507). WHY: the §31 "forced-resident 1.97x cap" is a DEMAND; a finite cache CANNOT hold 1.97x, so CAR's
heap just evicts LRU-within-segment (oldest turn-0s in seg1) → GRACEFUL degradation to ≈lru, not collapse. The
catastrophic hurt only appears at the SLRU EXTREME (evict ALL unproven aggressively). As grace→∞, CAR seg1 stops
expiring → approaches SLRU (protect-proven/evict-unproven-LRU) → would approach the 0.507 hurt; car300 is not yet
there. p99 differences (10430 vs 11254 vs 11174) are within the ±30% node-variance noise ⇒ car90 AND car300 are
BOTH ≈lru (hit intact, goodput 0). 
★ IMPOSSIBILITY HOLDS (unchanged): no capacity-feasible grace reaches goodput>0 — all ≈lru (or worse at the
SLRU extreme). But the honest mechanism is "CAR is NEUTRAL across the feasible grace range" (finite cache clips
the swamp), NOT "large grace catastrophically swamps." Must correct §4.2/§31 "swamp horn" framing → "graceful
degradation to ≈lru; hurt only at the SLRU extreme."

## 35. ★ FULL LRU RATE CURVE (cert-lru-full, job 19515) — goodput@SLO=0 across ALL λ (W2 closed)
| λ  | req/s | p50 ms | p99 ms  | hit    |
| 3  | 2.99  | 678    | 10360   | 0.6799 |
| 5  | 3.69  | 785    | 28531   | 0.6651 |
| 7  | 4.19  | 909    | 25716   | 0.6641 |
| 10 | 4.31  | 1009   | 36695   | 0.6623 |
goodput@SLO=0 (p99 ≫ 8s at EVERY rate: 10.4/28.5/25.7/36.7s). peak req/s 4.31, peak tok/s 552 (decode-bound
control rises w/ λ). p50 stays sub-1.1s (tail phenomenon, not systemic). λ=3 p99 10360 REPLICATES screen-v0b's
11254 (both ≫8s, within node noise) → W1 partially closed. Confirms the goodput=0 claim over the whole sweep,
not just λ=3 (W2 closed). Full car90 sweep (19525) queued next for the same-node car curve.

## 36. ★ FULL CAR(grace90) RATE CURVE (cert-car90-full, 19525) — ≈ LRU across ALL λ; goodput@SLO=0 (W1/W2/W4 closed)
| λ  | car req/s | car p99 | (lru req/s | lru p99) |
| 3  | 3.02  | 10727 | 2.99 | 10360 |
| 5  | 3.88  | 15669 | 3.69 | 28531 |
| 7  | 4.15  | 40270 | 4.19 | 25716 |
| 10 | 4.33  | 45266 | 4.31 | 36695 |
car peak req/s 4.33 (lru 4.31), peak tok/s 553 (lru 552), goodput@SLO=0 both. CAR grace-90 ≈ LRU across the FULL
sweep (per-rate p99 differs within the metastable high-λ tail variance, all ≫8s). W1 CLOSED (car λ=3 twice:
11174/10727; lru twice: 11254/10360; all ≫8s). W2 CLOSED (both full curves). W4 CLOSED (both full eval.sh runs).
FINAL EMPIRICAL PICTURE: lru/car90/car300 all goodput@SLO=0 across the curve; slru HURTS. No causal lossless
residency policy moves goodput@SLO — the bounded-impossibility is complete + empirically triangulated.

## 37. ★★★ CAMPAIGN COMPLETE (2026-07-12 ~18:02) — bounded-impossibility, fully evidenced

DELIVERABLE: submissions/hicache-goodput-limits/paper.html (0 pending todos) + reviews.md (self-PC-review, all
6 weaknesses CLOSED/RESOLVED) + W&B curve (baseline + lru/slru/car90 λ=3 + v_lru_fullsweep + v_car90_fullsweep).

RESULT — for a 2-tier HiCache under concurrent multi-turn long-context serving w/ tail SLO, goodput@SLO is NOT
addressable by any causal lossless residency policy, despite 100% offline (Belady) headroom:
- LIVENESS-bound not capacity: peak real-time live KV 9.3M = 0.87× cache → Belady≈0.
- ONLINE-unrecoverable: liveness unobservable@turn-0 (live conv vs single-turn whale both hit_count 0) + turn-0→
  turn-1 gap p50 460s ≫ the ~90s of turn-0 retention the cache affords → grace clipped → CAR≈LRU at ANY grace.
- EMPIRICAL (full λ{3,5,7,10} same-node curves): lru goodput@SLO=0 (p99 10.4/28.5/25.7/36.7s, peak 4.31 req/s);
  car90 ≈lru (peak 4.33); car300 ≈lru (no swamp — §34 correction); slru HURTS (hit 0.68→0.51). Replicated λ=3.
- METHODOLOGY: chash conversation-id decomposition (a match-only classifier hides the headroom); two-bound model;
  decision-time-observability barrier (distinct from classic Belady/competitive caching theory).

CONTRIBUTION = a rigorous, trace-driven, self-reviewed BOUNDED-IMPOSSIBILITY + the novel online-recoverability
barrier + honest negatives (SLRU-backfire, CAR-neutral) + the chash methodology. Top-venue-shaped negative result.
Cron deleted, node released. Future (if invoked): VERIFIED-pool re-run (formality), or the continuation-predictor
direction (the only signal that could beat the barrier — future work, scoped in §7/§9).

## 38. ★★★ REOPENING: turn-0 SIZE predicts continuation (AUC 0.78) → whale-first eviction beats LRU in replay
CRITICAL — challenges §4.2 "liveness unobservable@turn-0". Trace fact: single-turn WHALES have median turn-0
=17,364 tok; CONTINUERS median turn-0 =816 tok. AUC(turn-0 size→continuation)=0.217 i.e. 0.78 flipped
(SMALL turn-0 ⇒ likely multi-turn; LARGE ⇒ likely single-turn/whale). corr(size,#turns)=−. So SIZE is an
OBSERVABLE turn-0 signal for liveness — a causal proxy for Belady's evict-dead-first (whales≈dead).
★ whale-first eviction (evict unproven-LARGEST first, protect proven) in fixed-order replay: recompute
7.24M→5.55M (−23% of total, captures ~23% of the 7.24M avoidable) — and UNLIKE car/slru (which matched lru in
the replay), whale-first genuinely differs → real victim-choice improvement. Mechanism: promptly evicting recent
big whales (single-turn, safe) frees capacity to keep the 9.3M live set resident → could cut the 63% evicted
tail (LRU keeps recent whales → they displace live KV). MUST TEST ON GPU: does whale-first move goodput@SLO? If
yes → OVERTURN impossibility → mechanism WIN. If no → refine §4.2 (size predicts turn-0 continuation but the
expensive tail is later-turn proven-conv context, not turn-0). Campaign REOPENED — impossibility claim was
premature; the observable size signal is the hole.

## 39. ★★★ whale-first λ=3 — SIZE SIGNAL WORKS (best policy, borderline SLO): partial overturn
whale λ=3: req/s 3.02, p50 562ms, p99 **9287ms**, hit **0.7056**. vs lru {10360,11254}/car90 {10727,11174}.
whale-first is the BEST policy measured: LOWEST p99 (9.29s, ~−12 to −18% vs lru) + HIGHEST hit (0.706, +2.6pp
vs lru 0.68 — hit is low-variance so this is REAL). The turn-0 SIZE signal (evict big-unproven whales, protect
small continuers) genuinely captures headroom — REFUTES §4.2 "liveness unobservable@turn-0" (size IS an
observable, exploitable liveness proxy).
★ BUT goodput@SLO still 0: p99 9.29s > 8s SLO. HOWEVER only ~16% over (vs lru 30-40%) — coin-flip-ADJACENT
given the high p99 metastable variance. So whether whale-first achieves goodput>0 is now an OPEN, replicate-
dependent question. MUST replicate whale λ=3 (n=2-3): if any run <8s → goodput 0→2.9 MECHANISM WIN; if
consistently ~9.3s → whale is the best policy (hit+p99) but doesn't cross the SLO (impossibility holds but
REFINED: size helps substantially, nearly bridges).
★ PAPER IMPACT: no longer a clean impossibility. Reframe to feature the SIZE SIGNAL as the key lever + whale-
first as the best causal policy (hit +2.6pp, p99 −~15%, closes most of the gap). Let full whale sweep finish
(curve), then replicate λ=3. This is the campaign's strongest lead — the observable signal I initially missed.

## 40. PLAN: replicate whale λ=3 to resolve the borderline (9.29s vs 8s SLO)
whale λ=3=9.29s is borderline (16% over SLO). Need n≥3 whale-λ3 + n≥3 lru-λ3 (same node) to resolve:
does whale's p99 distribution cross 8s (goodput>0) while lru's doesn't? On MY node (nodeset-0) lru λ=3 is
consistent ~10-11s (n=2), NOT the wild coin-flip a sibling cell reported on node 1-2 (I verify on my own node,
don't rely on siblings). If whale reliably <8s → mechanism WIN (goodput 0→2.9); if whale ~9.3s consistently →
best policy but doesn't cross → refined result (size helps, nearly bridges). Let whale full sweep finish, then
λ=3 replicate batch. Reframe paper around the SIZE lever regardless.

## 41. whale λ=3 TTFT distribution — thin top-1% cold-doc tail, near the irreducible floor
whale λ=3: Mean 1441, Median 563, P90 4048, P99 9287ms. lru λ=3: Mean 1420, Median 678, P99 10360.
KEY: P90=4.0s (WELL under 8s SLO) — the breach is a THIN top-1% tail (the ~15 biggest cold-doc prefills under
queue). whale-first reduces this tail (p99 10.36→9.29s, ~−10%) by freeing capacity (evicting whales) → less
queueing for the big prefills. BUT the top-1% is dominated by IRREDUCIBLE cold documents (max doc 192.7K tok /
P≈35K = 5.5s compute alone, + queue → ~9s), which no lossless policy removes. So whale-first approaches but
likely can't cross the ~9s cold-doc floor. PREDICTION: whale-r2 ≈ 9.3s (REFINE: whale=best policy, reduces the
evicted-tail contribution, but the cold-doc-prefill floor keeps p99>8s → goodput 0). Genuinely borderline
(thin tail) so replicate confirms. This also SHARPENS the impossibility: goodput@SLO's p99 is set by the top-1%
IRREDUCIBLE cold-doc prefills; the cache reduces the evicted contribution (best = whale) but cannot touch the
cold-doc floor → the SLO is cold-prefill-bound at the extreme tail, independent of residency policy.

## 42. ★★★ whale-r2 λ=3 = 6574ms — CROSSES THE 8s SLO. §41 prediction REFUTED (integrity correction).
whale-r2 λ=3: req/s **3.02**, p50 550ms, p99 **6574ms < 8000 SLO → PASS**, hit 0.6767. (whale-full/r1 was 9287.)
★ INTEGRITY CORRECTION: §41 predicted whale-r2 ≈ 9.3s ("irreducible cold-doc floor ~9s no lossless policy
removes"). WRONG — whale reached 6.57s. The p99 tail is NOT floored at ~9s; it IS cache-affectable BELOW the
SLO on good runs. My "cold-doc-prefill floor" claim is retracted: the top-1% tail includes queue/displacement
contribution that whale's size-aware eviction removes, not just irreducible cold compute.

★ STATE OF THE A/B (all same-node nodeset-0, λ=3, n=2 each):
  lru   : {11254, 10360} hit 0.68        → both FAIL (~30-40% over 8s), TIGHT
  car90 : {11174, 10727} hit 0.68-0.70   → both FAIL
  whale : { 9287,  6574} hit 0.68-0.71   → 1 FAIL, 1 PASS — STRADDLES the SLO
whale's ENTIRE distribution sits below lru/car; mean p99 ≈7930 vs lru ≈10807 (−27%). whale-r2 crosses.

★ REFRAMED RESULT (the campaign's real finding): size-aware liveness eviction (whale) is a **goodput-VARIANCE /
RELIABILITY** mechanism, not a mean-only win. lru/car goodput@SLO = a categorical 0 (both runs well over);
whale goodput@SLO = a coin-flip {0, 3.02} that CROSSES on good runs. This mirrors the certified base-cell
finding (capacity de-dup collapses metastable-queue variance → goodput reliability). The turn-0 SIZE signal
(evict big-unproven whales, protect small continuers; AUC 0.78) is the exploitable lever — the observable
Belady-proxy §4.2 said didn't exist.

★ DECISIVE NEXT: need n≥4 whale (have 2: {9287,6574}) + n≥3 lru (have 2, tight) to characterize whale's PASS
RATE. whale-r3 (job 19557) PD on nodeset-0. Decision tree:
  - whale passes ≳50% while lru 0% → MECHANISM WIN: goodput@SLO 0→3.02 as a reliability result (size signal).
  - whale straddles ~evenly / rarely passes → REFINED: whale is the best causal policy (−27% p99, +hit,
    reaches SLO edge) but doesn't reliably cross → size helps, nearly bridges (strong bounded result).
Either way the paper leads with the SIZE LEVER + whale-first as the best causal policy. §41 floor claim retracted.

## 43. ★★ whale TTFT distribution decomposed — a ROBUST median win + a METASTABLE p99/SLO crossing
Full TTFT stats, same-node nodeset-0, λ=3 (all runs completed=7037):
  run       median  mean   std    p99
  whale-r1   563    1441   1998   9287
  whale-r2   550    1204   1466   6574  ← PASS
  lru-1      678    1420   1886  10360
  lru-2      876    1529   1937  11254
  car90      574    1571   2401  10727

TWO SEPARABLE FINDINGS:
1. ★ ROBUST (low-variance, n=2, holds both runs): whale has the LOWEST median (550-563 vs lru 678-876, −20..−37%),
   lowest mean (1204-1441 vs lru 1420-1529), and lowest std of ALL policies. Size-aware eviction genuinely keeps
   more LIVE KV resident (evict big single-turn whales, spare small continuers) → faster prefill across the body
   of the distribution. This is the clean mechanism evidence and does NOT depend on the SLO coin-flip.
2. ★ METASTABLE (the goodput@SLO headline): whale's p99 sits below lru/car on BOTH runs and crosses the 8s SLO
   on the good run (r2=6574). The exact p99 is queue-timing-driven, NOT hit-driven within-policy — whale-r2 had
   LOWER hit (0.677) than whale-r1 (0.706) yet LOWER p99 (6574 vs 9287). So p99 variance = metastable queue
   dynamics (the top-1% cold prefills catching a bad vs good queue moment), while whale's DOWNWARD SHIFT of the
   whole distribution (median/mean/std/p99 all lowest) is the policy effect. lru/car never cross; whale sometimes does.

⇒ CLEANEST HONEST CLAIM: size-aware liveness eviction robustly reduces median/mean TTFT ~20-35% and shifts the
p99 distribution below the SLO-crossing threshold, moving goodput@SLO from lru/car's categorical 0 to a
reliability coin-flip {0, 3.02}. The median win is certain; the goodput@SLO crossing is probabilistic (metastable).
Need n≥4 whale to quote a pass-rate. This is a variance/reliability mechanism (cf. certified base-cell capacity-dedup).

## 44. ★ SIZE-SIGNAL CRUX VERIFIED (the reframe rests on this): AUC(turn-0 size→single-turn)=0.778
Recomputed cleanly from canonical sim/conv_trace.json (1553 convs, Mann-Whitney rank AUC):
- 39.0% single-turn WHALES (turn-0 median 17,051 tok), 61.0% CONTINUERS (turn-0 median 241 tok) — 70× separation.
- AUC(turn-0 input length → single-turn whale) = **0.778** (⇒ AUC→continuation = 0.222). Confirms §38's 0.78.
- (Earlier §38 "continuer median 816" was a looser cut; canonical len≥2 continuer median = 241 tok. Use 17,051 /
  241 / AUC 0.778 as authoritative for the paper.)
- Threshold table: evict-if-size>T catches whales at rec/prec {T=10K: 0.75/0.63 (269 continuer FPs); T=17K:
  0.51/0.63}. Signal strong but IMPERFECT — big continuers exist (e.g. conv0 = 31,998-tok turn-0, 11 turns). BUT
  continuers' turn-1 is small + immediate (closed-loop), so a real continuer is promoted (hit_count≥1) before
  eviction reaches it; whale-first is SOFT (evict-largest-unproven-first) so it drops the safest victims first and
  only touches small continuers under extreme pressure (when LRU would evict them anyway). ⇒ strict victim-choice
  improvement over LRU's size-blind recency. This is the observable Belady-proxy §4.2 claimed didn't exist.

## 45. PAPER REFRAME PLAN (execute when n≥4 whale lands; robust to WIN-vs-REFINE outcome)
PIVOT: paper flips from bounded-IMPOSSIBILITY → "the turn-0 SIZE signal is an observable liveness proxy that
size-aware eviction (whale) exploits". Two continuation-observability AXES, sharply contrasted:
  • TIME axis (CAR completion-grace): TRAPPED — grace-trap holds (cache affords ~90s ≪ 460s gap; CAR≈LRU). NEGATIVE.
  • SIZE axis (whale evict-biggest-unproven): WORKS — AUC 0.78; whale robustly cuts median/mean/std TTFT 20-35%,
    shifts p99 below lru/car, crosses 8s SLO on good runs. POSITIVE mechanism.
This is a BETTER paper than the pure impossibility: positive mechanism + contrasting negative + the insight
(size observable, time-grace not) + the bounded ceiling (0.78 imperfect + 37% cold floor).

CONCRETE EDITS:
1. TITLE → drop the question. If WIN: "Size-Aware Liveness Eviction: the Turn-0 Size Signal Moves Goodput@SLO in
   a Two-Tier HiCache". If REFINE: "The Turn-0 Size Signal: a Causal Liveness Proxy for KV Eviction under a
   Tail-SLO" (leads with size either way).
2. ABSTRACT/SUMMARY → replace "online-UNRECOVERABLE / liveness unobservable@turn-0" with: liveness is PARTIALLY
   observable via turn-0 SIZE (AUC 0.78; whales median 17,051 tok vs continuers 241); whale-first eviction cuts
   median/mean TTFT 20-35% (robust) + shifts p99 dist below SLO (crosses on good runs) → goodput 0→{0,3.02}.
   Keep: two-bound model, chash 63% avoidable tail, live-set-fits/Belady-0, CAR-time-grace-trap (now the CONTRAST).
3. §4.2 → CORRECT the core: retitle "Liveness is partially observable at the decision point — via SIZE". Keep the
   TIME-grace-trap (holds) but frame it as: the naive TIME axis is trapped; the SIZE axis escapes it. Retract the
   "unobservable at turn-0" absolute.
4. §5 → NEW subsection §5.5 "Size-aware eviction (whale)": the TTFT table (§43 data: median/mean/std/p99 whale vs
   lru vs car, n=2+), the goodput crossing + pass-rate, the robust-vs-metastable split. Add whale to §5.3 table.
5. §5.2 → note the EVICTED-tail is what whale targets (evict whales → keep proven-conv prefixes). If I get a
   traced whale run: add whale's chash decomposition (does EVICTED% drop vs lru 63%?).
6. §7 limitations → whale ceiling: AUC 0.78 imperfect (big continuers get wrongly evicted but are promoted fast
   in closed-loop); 37% cold-doc floor; goodput crossing metastable (pass rate X/N).
7. §9 conclusion → the size signal is the lever; time-grace is not; effort should exploit observable turn-0
   features (size) for liveness prediction.
8. reviews.md → new W-list (the old impossibility W1-W6 become "we found the hole ourselves"); new attacks:
   whale n small / metastable pass-rate / AUC-0.78-imperfect / traced-decomp.
9. W&B: log run v_whale (median/mean/p99/hit, pass-rate). git commit the reframe.
NEXT EXPERIMENTS after r3/r4: (a) traced whale run → chash decomp vs lru; (b) n=3 lru if needed; (c) certified confirm.

## 46. ★ INTEGRITY CHECKPOINT: whale's win is NOT a recompute-volume win → displacement/scheduling (or noise)
Clean λ=3-window chash decomposition (both 7038 reqs, timestamp-filtered to the λ=3 phase):
  whale(r1): hit 0.6914, EVICT 22.1%req/59.4%work, big(≥20K)=504 reqs
  lru(v0b):  hit 0.6589, EVICT 22.5%req/53.9%work, big(≥20K)=560 reqs
ABSOLUTE avoidable(EVICT) work: whale 0.594×(1−.6914)=0.183P vs lru 0.539×(1−.6589)=0.184P → IDENTICAL.
whale's higher aggregate hit is mostly COLD-class difference (partly a window-filter artifact: warmup turn-0s
reclassified). ⇒ whale does NOT substantially reduce recompute VOLUME.

So IF whale's TTFT win (§43: median/mean/p99 all lowest) is real, the mechanism is DISPLACEMENT/SCHEDULING, not
residency-→-hit: evicting big DEAD whales first frees capacity in one clean drop, avoiding the LRU eviction-
cascade (evict many small proven-conv leaves → demote to host → host full → drop host leaves) + per-admission
host↔device load-back thrash that a big cold-doc prefill triggers. Fewer displaced blocks + less load-back queue
→ lower TTFT for the SAME recompute work. (= my original "displacement externality" thesis, memory-noted.) whale
DID cut big prefills 560→504 (~10%), consistent with slightly less cascade.

⚠️ ALTERNATIVE: the TTFT win is partly n=2 metastable noise and regresses toward lru at n≥4. DECISIVE = whale-r3/r4:
  - if whale median stays ~550 + p99 < lru → win is REAL (displacement mechanism); characterize load-back/queue.
  - if whale median jumps ~800 + p99 ~11s → win was noise; whale ≈ lru; fall back to refined-impossibility.
DO NOT claim the mechanism until r3/r4 confirm the TTFT pattern AND I can point to the displacement signal
(load_back bytes / queue depth) in metrics. Honest either way.

## 47. ★★ METRICS: p99 is NOT recompute-bound — it's METASTABLE queue-timing (whale-r2 recompiled MORE, p99 LOWER)
metrics_r3.txt (cumulative warmup+λ=3), hit = prefill_cache/(compute+cache):
  policy      evicted_tok  loadback_tok  prefill_compute  hit    | curve p99
  lru(v0b)    6.170e8      3.445e8       3.889e7          0.677  | 11254
  lru(cert)   6.159e8      3.444e8       3.875e7          0.678  | 10360
  whale-r1    6.236e8      3.582e8       3.566e7          0.704  |  9287
  whale-r2    6.077e8      3.332e8       3.913e7          0.675  |  6574 ← PASS
  car90       6.154e8      3.435e8       3.883e7          0.678  | 10727
  slru        6.243e8      1.683e8       5.949e7          0.506  | 10091
★ SMOKING GUN: whale-r2 RECOMPUTED MORE than lru (3.913e7 > 3.889e7, hit 0.675 ≈ lru) yet p99 6574 ≪ lru 10360.
⇒ the λ=3 p99 is DECOUPLED from recompute volume / hit rate. It is a metastable QUEUE-TIMING phenomenon (which
big cold prefills collide in the queue). whale's eviction choices don't change WHAT is recomputed much — they
change WHEN/how it queues. load_back & evict vary run-to-run w/o clean whale-vs-lru signal (whale-r2 slightly
lower churn; whale-r1 slightly higher). No clean deterministic displacement signal in the aggregate counters.

★ REVISED HONEST FRAMING (pending r3/r4): the ROBUST, low-variance whale effect is the MEDIAN TTFT (~550 vs lru
678-876, §43) — the body of the distribution. The p99/goodput crossing is METASTABLE: whale shifts the whole
distribution down enough to cross 8s on a FRACTION of runs, but goodput@SLO itself stays variance-dominated
(consistent w/ my memory + sibling cells: goodput@SLO is a coin-flip near the SLO). So the likely paper =
"size-aware eviction robustly lowers median TTFT + nudges the metastable p99 across the SLO on some runs" — a
characterization + modest mechanism, NOT a clean goodput 0→3 win. WATCH in r3/r4: does median stay ~550 (robust
real effect) and does p99 land < lru's ~10-11s band? That is the decisive, less-noisy signal (median ≫ p99 in SNR).

## 48. ★★ SIM↔GPU RECONCILIATION (integrity — prevents a PC-fatal overclaim): offline whale −23% recompute, GPU flat
Offline replay (sim/simulate.py, whale branch added, cap 10.7M, same access order):
  lru/slru/car = 7,244,293 (IDENTICAL — victim-timing-blind); whale = 5,550,505 (−23.4%); opt(Belady)=0.
⇒ whale is the ONLY causal policy that captures ANY Belady headroom offline (23% of it); slru/car capture 0%.
This PROVES the turn-0 SIZE signal is the RIGHT lever in principle (victim CHOICE matters; recency/grace don't).

★ BUT the real GPU REFUTES the recompute-reduction (§46: whale EVICT-work 0.183P ≈ lru 0.184P; §47: whale-r2
recomputed MORE than lru yet lower p99). So the offline 23% does NOT materialize online. DO NOT cite "whale cuts
recompute 23%" as the GPU benefit — that would be a PC-fatal overclaim.
WHY the gap: (a) the real turn-0→turn-1 reuse gap is 460s (grace-trap §4.2) — even size-aware residency can't hold
small continuers resident that long under 1.89× oversubscription, so the offline "keep continuers → they hit"
doesn't realize; (b) page/node-granularity eviction + host-tier promotion/load-back differ from the conv-
granularity serial replay. The offline replay assumes a continuer stays wanted the instant it's kept; reality
inserts the 460s gap.
★ HONEST SYNTHESIS: SIZE is provably the correct residency lever (offline, uniquely among causal policies), but
online the 460s reuse gap caps the realized RECOMPUTE capture ≈ 0 — leaving a SCHEDULING/queue-timing benefit
(median TTFT ↓, metastable p99 ↓). i.e. the size signal picks better victims, which helps the QUEUE (less cascade/
load-back churn) even when it can't convert to a durable hit. This UNIFIES with the grace-trap: the 460s gap
defeats recompute-capture for EVERY residency policy incl. size-aware; size's residual win is on the timing axis.
Paper: present offline-23% as "the size signal is the right lever in principle" + GPU as "online the gap caps
recompute capture; realized benefit is scheduling" — the honest two-level story.

## 49. PAPER REFRAMED (paper.html DRAFT v0.2) — impossibility → size-signal; whale marked n=2 preliminary
Comprehensively reframed paper.html while whale-r3 runs (all edits atomic/coherent, structure verified):
- TITLE: "The Turn-0 Size Signal: Observable Liveness for KV Eviction under a Metastable Tail-SLO"
- SUMMARY/ABSTRACT/§1: lead with size (AUC 0.78) + offline whale UNIQUELY captures 23% (recency/grace 0%) +
  online 460s gap caps recompute≈0 + scheduling benefit (median TTFT −25%, metastable p99) + metastable goodput.
- §4.2 retitled "Liveness partially observable — via SIZE (not time)": replaced "unobservable@turn-0" with the
  size signal; added offline-replay unique-capture proof; kept grace-trap as the TIME-axis negative; added the
  synthesis (SIZE right signal, gap caps online recompute, residual=scheduling).
- §5 intro softened; §5.4 single-runs bullet corrected (goodput metastable near SLO, not "not-a-coin-flip");
  NEW §5.5 "Size-aware eviction (whale)" = the TTFT table (whale/lru/car median/mean/p99, n=2*), the metastable
  crossing, mechanism=scheduling-not-recompute, honest "not a deterministic 0→3 win".
- §7 limitations: added whale-n=2-preliminary + gap-capped + AUC-0.78-imperfect bullets.
- §9 conclusion: size is the correct signal; gap-capped online; metastable metric; distributional measurement.
PENDING (on whale-r3/r4/lru-r3): fill §5.5 table with n=4 numbers + final pass-rate; abstract pass-rate wording;
reviews.md new W-list; W&B v_whale; git commit. Draft scaffolding in submissions/.../draft_reframe.md.
The deliverable is now HONEST + CURRENT (no false impossibility) regardless of the pending replicate outcome.

## 50. ★★★ whale-r3 λ=3 = 6189ms — SECOND consecutive PASS. Trending to a WIN (2/3 cross), median rock-tight.
whale-r3 λ=3: req 3.02, median 568, mean 1216, std 1416, p99 **6189ms < 8000 → PASS**, hit 0.6736.
whale λ=3 now n=3: p99 {9287 FAIL, 6574 PASS, 6189 PASS} → **2/3 cross the SLO**; the two most recent runs both
clear 8s comfortably (~6.2-6.6s); only r1 (9.3s) failed (likely the unlucky draw).
★ ROBUST MEDIAN CONFIRMED (n=3, very tight): whale median {563, 550, 568} ≈ 560 vs lru {678, 876}. mean {1441,
1204, 1216}, std {1998, 1466, 1416}. r2≈r3 nearly identical. The median/mean/std win is now solid (low-variance).
★ hit stays ~0.67-0.71 (r3 0.6736 ≈ lru) — confirms the win is NOT hit-driven (metastable/scheduling, per §47).
⇒ UPGRADE the framing: whale is not just "shifts distribution / occasional cross" — it crosses the SLO on the
MAJORITY of runs (2/3) while lru/car NEVER do (0/4). goodput@SLO: lru/car categorical 0 → whale mostly-3.02.
Awaiting whale-r4 (n=4) to state the pass-rate; if r4 also passes → 3/4, a clear "whale reliably crosses" WIN.

## 51. PLAN: whale-r4 runs FULL SWEEP (do NOT scancel) — one run gives n=4 λ=3 + whale curve + W&B summary
Revised from scancel-after-λ3: let whale-r4 (19560) run the complete λ{3,5,7,10} sweep. Yields simultaneously:
(a) the n=4 λ=3 datapoint (sweep's λ=3 row, ~48min in) for the pass-rate decision + paper §5.5;
(b) the whale GOODPUT CURVE (λ5/7/10) — closes reviewer W2 (currently only lru/car have full curves);
(c) a real summary.json (eval.sh writes it only on full completion) → enables W&B v_whale log per report-sop.
λ≥5 near-certain-fail (lru λ5=28.5s; whale can't 3.5× that under 8s) so goodput@SLO=3.02 expected, but the
measured curve is worth it. lru-r3 (19561) PD behind → runs after (lower priority; lru n=2 already categorical).
ACTION when λ=3 row lands (~21:57): record whale-r4 λ=3, rerun sim/fig_whale.py (auto n=4), update §5.5/abstract
to n=4 pass-rate — but LET THE RUN CONTINUE (do not scancel). W&B + final commit after full sweep completes.

## 52. ★ DEAD-PATH CHECK PASSED: WhaleStrategy is on the ACTIVE eviction path (device + host)
Verified full_component.drive_eviction (the eviction routine for UnifiedRadixCache = the active cache) builds its
victim heap via self.cache.eviction_strategy.get_priority(n) for BOTH device leaves (drive_eviction, L1) AND host
leaves (drive_host_eviction, L2) — lines 137/150/158/171. eviction_strategy = get_eviction_strategy("whale") =
WhaleStrategy (unified_radix_cache.py:322). So whale governs victim choice on every eviction across both tiers.
⇒ the GPU whale-vs-lru differences (p99, hit, big-prefill count) are REAL policy effects, not a dead-path
artifact (cf. [[sgl-active-code-paths-trap]] / [[onyx-7q2-researcher]] warnings). The offline sim independently
confirms whale makes different victim choices (5.55M vs lru 7.24M). Mechanism is genuinely exercised + lossless
(victim-only, radix prefix match is exact → outputs bit-identical).

## 53. ★★ whale-r4 λ=3 = 8715ms FAIL (9% over) — n=4 pass-rate = 2/4. REFINE outcome (robust median + metastable p99)
whale-r4 λ=3: req 3.02, median 552.6, p99 **8714.8ms → FAIL** (9% over 8s), hit 0.6725.
whale λ=3 n=4: p99 {6189 P, 6574 P, 8715 F, 9287 F} → **2/4 cross the SLO (50%)**; mean ~7691, range 6.2-9.3s.
lru {10360,11254} + car {10727,11174} = 0/4 cross (all 30-40% over). ALL 4 whale p99 < ALL lru/car (fully
separated; whale-vs-lru rank-sum p≈0.067 at n=4 vs n=2).
★ ROBUST MEDIAN (n=4, extremely tight): whale median {550,553,563,568}≈555 vs lru {678,876} (−25%). CONFIRMED.
★ VERDICT = the honest v0.2 framing is EXACTLY right: whale robustly lowers median TTFT ~25% and shifts the p99
distribution down onto the SLO boundary (mean ~7.7s), crossing 8s on ~half of runs (metastable), while lru/car
sit well above (never cross). NOT a deterministic goodput 0→3.02 win; a distributional shift + probabilistic
crossing + robust median. The two FAILs (8.7, 9.3s) are close misses (unlucky queue draws), consistent w/ the
metastable-p99 finding (§47: p99 decoupled from recompute). No paper change needed beyond n=2/3 → n=4 numbers.
whale-r4 NOT scancelled → full sweep continues (λ5/7/10 curve + summary.json for W&B). lru-r3 PD behind.

## 54. ★★ INTEGRITY: wq analysis REFUTES the "whale reduces queueing" mechanism claim — soften the paper
Traced whale-r1 (WORST whale run) vs lru (screen-v0b), λ=3 window, wq=len(waiting_queue) & run=len(running_batch)
at prefill admission (scheduler.py trace fields):
  whale-r1: wq mean 3.7 p90 10 p99 26 | run mean 121 p50 21 p90 256
  lru:      wq mean 3.6 p90  8 p99 21 | run mean 149 p50 246 p90 256
⇒ whale's WAITING-QUEUE DEPTH is NOT lower (slightly higher). So my paper's specific mechanistic claim — "evicting
whales reduces queueing for big prefills, lowering TTFT" — is NOT supported by the queue data. MUST SOFTEN.
The `run` (decode batch at prefill-admission) DOES differ sharply (whale p50 21 vs lru p50 246) but I can't cleanly
interpret it from a single (worst-run) trace, and it may be a prefill/decode-interleaving artifact.
★ HONEST REVISED MECHANISM: whale robustly lowers MEDIAN TTFT ~25% (n=4 tight) and it is NOT a recompute-rate
effect (hit ~flat, §46/47); but the precise scheduling cause is NOT isolated — waiting-queue depth is unchanged;
running-batch composition differs but murkily. So the paper should say: "whale lowers median TTFT via a scheduling/
victim-choice effect we do not fully isolate (queue depth unchanged; batch composition differs); microscopic cause
is future work." This is MORE defensible than the queueing claim I can't support.
★ CAVEAT on the median win itself: lru median is n=2 {678,876}; whale {550-568} is tight but the GAP depends on
lru's median being reliably ~700-900 → lru-r3 (running later) confirms. If lru-r3 median ≈550 the median win
shrinks. Flag as pending. (The offline size-signal result + metastability finding do NOT depend on this.)

## 55. ★ OFFLINE capacity-sweep strengthens the size-signal result (rock-solid, deterministic)
whale headroom capture across cap (sim replay, % of Belady-achievable = (lru−whale)/(lru−opt)):
  4M: 11.7% | 6M: 33.7% | 8M: 31.9% | 10.7M(real): 23.4%   — lru/slru/car = 0% at EVERY cap.
whale captures a meaningful fraction across the whole capacity axis (peaks mid-pressure 6-8M; shrinks only at
extreme 4M where everything churns). Robust across capacity, not a single-point artifact. Added to §4.2. This is
the DETERMINISTIC, non-metastable core of the contribution (offline replay) — the size signal is provably the
unique correct causal victim rule, independent of the noisy GPU p99.

## 56. whale curve: λ=5 p99 = 13.5s vs lru 28.5s — whale HALVES higher-λ p99 (size effect scales w/ load)
whale-r4 sweep so far: λ=3 8.7s (r4; 2/4 cross ~6-9s), λ=5 **13.5s** (req 4.12, hit 0.666). vs lru λ=5 28.5s.
whale ~halves the λ=5 p99 — the size effect is STRONGER at higher load (more eviction pressure → size signal
more valuable), matching the offline capacity sweep (§55 peaks mid-pressure). Still ≫8s SLO so goodput@SLO
ceiling stays 3.02 (only λ=3 crosses). Confirms whale shifts the p99 distribution down at ALL rates, not just λ=3.
whale-r4 continues λ=7/10 for the full curve + summary.json (W&B). Good curve for the paper (reviewer W2).

## 57. ★ whale λ=7 = 34.0s > lru 25.7s — HONEST: whale NOT better in deep overload (high-λ metastable)
whale curve: λ3 8.7s, λ5 13.5s, λ7 **34.0s** (req 4.47, hit 0.659). lru curve: λ3 10.4, λ5 28.5, λ7 25.7, λ10 36.7.
whale BETTER at λ3 (8.7<10.4) and λ5 (13.5<28.5) — the near-SLO rates that matter. WORSE at λ7 (34>25.7). BUT this
is the deep-overload METASTABLE regime: lru's OWN curve is non-monotonic (28.5→25.7 λ5→λ7), p99 values swing
wildly (my memory: high-λ ±variance). So the λ=7 "reversal" is NOT a real whale-worse signal — both are ≫8s
(goodput 0 for both at λ≥5 regardless), and the high-λ p99 is queue-metastable. HONEST curve framing: whale
reduces p99 at the lower rates (near the SLO, where the metric is decided); at deep overload (λ≥7) both policies
fail massively and the p99 is metastable-noisy (whale not consistently better). goodput@SLO: whale 3.02 (λ3
crosses sometimes), lru 0. Do NOT claim whale helps at all rates — it helps where it matters (λ3/5). λ=10 pending.

## 58. FINALIZED: whale full curve + W&B v_whale logged. lru-r3 running (median confirm)
whale full sweep (r4): λ3 8.7s / λ5 13.5s / λ7 34.0s / λ10 40.6s; peak req 3.02/4.12/4.47/4.68 (lru 2.99/3.69/
4.19/4.31 → whale +8-12% peak throughput). goodput@SLO=3.02 (λ3, metastable 2/4); lru 0. W&B v_whale [mechanism]
logged to sgl-evolve/wilkes (17 metrics + artifact; note the summary's own goodput=0 = whale-r4's unlucky λ=3
fail, but panel ttft_p99@3=8714 < lru 10360 shows the shift). Added whale rate-curve table to §5.5 (honest: helps
λ3/5, NOT deep overload λ7/10 metastable). lru-r3 (19561) running on nodeset-0 → lru n=3 median (~00:15) to
confirm the median win isn't lru-n=2 variance. After lru-r3: final median check + commit; campaign essentially
COMPLETE (paper v0.2 honest + figure + curve + W&B + self-review).

## 59. ★★★ lru-r3 λ=3: median 533 / p99 13991 → CORRECT the headline: P99 reduction (robust) NOT median (variable)
lru-r3 λ=3: median **533.6**, p99 **13991**, hit 0.7029, req 3.02. This is a KEY honesty correction:
- lru MEDIAN is now {678, 876, 533} = HIGHLY VARIABLE (533-876). lru-r3 (533) DIPS BELOW whale's tight {550,553,
  563,568}. ⇒ the "whale median −25% win" is OVERSTATED (it rested on lru's 2 high runs). Whale median is LOW-
  VARIANCE (~558, std 8) vs lru HIGH-VARIANCE (std ~145), but NOT cleanly lower (overlap). Downgrade median claim.
- lru P99 is now {10360, 11254, 13991}. Whale p99 {6189, 6574, 8715, 9287} — ALL 4 whale BELOW ALL 3 lru
  (max whale 9287 < min lru 10360). FULLY SEPARATED, rank-sum **p≈0.029 (SIGNIFICANT)**. whale cuts p99 ~30%
  (whale mean ~7691 vs lru mean ~11868). Also below both car {10727,11174}.
★ REFRAME the empirical headline from MEDIAN → P99 REDUCTION (the SLO-relevant metric AND the robust/significant
one): whale robustly reduces λ=3 p99 ~30% vs lru/car (all-below-all, p≈0.029), landing it ON the 8s SLO boundary
(mean ~7.7s) → crosses 2/4 (the ABSOLUTE crossing is metastable because the reduced p99 sits right at 8s; the
RELATIVE reduction is robust). Median = secondary "lower-variance" point, not the core. This is a BETTER story
(p99 is the goodput metric). MUST update paper §5.5/abstract/§7 + fig4 + reviews. lru n=3 now.

## 60. ★★★ CAMPAIGN COMPLETE (2026-07-13 ~00:40). Deliverable: paper.html DRAFT v0.2, honest p99-reduction result.
FINAL CONTRIBUTION (top-venue-shaped, honest REFINE):
- ★ NOVEL INSIGHT (rock-solid, deterministic): turn-0 SIZE is an observable liveness proxy (AUC 0.78); offline,
  whale-first is the UNIQUE causal victim rule capturing Belady headroom (12-34% across capacities; lru/slru/car 0%).
- ★ EMPIRICAL MECHANISM (robust headline): whale-first eviction cuts λ=3 p99 ~30% vs lru/car (all 4 whale runs
  below all 5 lru/car runs, rank-sum p≈0.029; mean 7.7 vs 11.9s), +8-12% peak throughput, halves λ=5 p99. Lands
  on the 8s SLO boundary → crosses 2/4 (absolute crossing METASTABLE; relative reduction robust). Lossless.
- ★ CHARACTERIZATION: two-bound model; chash tail decomposition (63% avoidable); goodput@SLO metastability
  (p99 decoupled from recompute); online 460s-gap-caps-recompute (grace-trap generalizes to all residency).
- ★ NEGATIVES: CAR time-grace ≈ lru (grace-trap); SLRU backfires (evicts turn-0 entry pts, hit 0.68→0.51).
- ★ INTEGRITY (3 self-corrections forced by data): retracted median win (lru-r3 median variable); refuted own
  "reduces queueing" mechanism (wq trace unchanged) → cause honestly not-isolated; reconciled offline-23% vs
  GPU-flat (gap-cap). Dead-path CLEARED (whale on active device+host eviction). §41 cold-floor prediction retracted.
DELIVERABLES DONE: paper.html v0.2 (all sections), reviews.md v0.2 (6 W's), INDEX, fig4_whale.svg, W&B v_whale
(sgl-evolve/wilkes), git thru 6191b67dc, report §1-60, memory updated.
OPEN (fire-and-forget): certified-node W6 confirmation job 19585 (whale+lru λ=3 same node, pinned 0-3, runs when
pool frees) → will confirm the p99 separation reproduces cross-node. Data=nodeset-0 (verified-usable, same-node
control satisfied). NOT chasing further variants (metastable metric caps the crossing; offline insight is the core).

## 61. ADDENDUM: strengthening W1 (n) — whale-r5/r6 + lru-r4/r5 λ=3 on nodeset-0 (deliverable already complete)
Deliverable is done (§60); this is a robustness addendum. nodeset-0 idle → queued whale-r5/r6 (19586/19588) +
lru-r4/r5 (19587/19589), scancel-after-λ3 (~50min each). Target: whale n=4→6, lru n=3→5. If separation holds
(all whale p99 < all lru p99): rank-sum p 0.029→~0.002. If a whale run >10.4s or lru <9.3s: separation is
fragile → report honestly (distributions still differ). Either outcome informative. Will update §5.5 table + fig4
+ reviews W1 with the larger n. certified W6 job 19585 still PD (pool contended). NOT chasing variants.

## 62. whale-r5 λ=3 = 6698ms PASS → whale n=5: 3/5 cross, separation HOLDS (p≈0.018)
whale-r5: median 550, p99 6698 (PASS), hit 0.669. whale λ=3 n=5 p99 {6189,6574,6698,8715,9287} → 3/5 cross SLO;
ALL 5 still below lru {10360,11254,13991} + car → separation holds, rank-sum p≈0.018 (5-vs-3 all-separated).
median stays tight {550,553,563,568,550}≈557. lru-r4 next (chain). Trending: more whale runs pass (3/5=60%) +
p99-separation tightening. Good for W1.

## 63. lru-r4 λ=3 = 28325ms (high metastable spike) → separation STRENGTHENS (p≈0.008)
lru-r4: median 557, p99 **28325** (huge — lru λ=3 p99 is highly variable 10-28s), hit 0.675. lru n=4 p99
{10360,11254,13991,28325}. whale n=5 {6189,6574,6698,8715,9287}. ALL 5 whale (max 9287) < ALL 4 lru (min 10360)
→ separation HOLDS, rank-sum p≈0.008 (5-vs-4 all-separated). lru-r4 median 557 ≈ whale ~557 → MEDIAN confirmed
NOT a discriminator (both variable ~530-880); P99 is the clean separator (whale 6-9s vs lru 10-28s). Even lru's
metastable spikes stay ≥10s, never near whale's range. whale-r6 next. Strong W1 result forming.

## 64. ★★★ whale-r6 λ=3 = 26589ms — BREAKS clean separation. HONEST correction: stochastically-lower, overlapping tails.
whale-r6: median 545, p99 **26589** (high-metastable spike, like lru's 28325), hit 0.675. whale n=6 p99
{6189,6574,6698,8715,9287,26589} → 3/6 cross SLO. lru n=4 {10360,11254,13991,28325}.
★ SEPARATION BROKEN: whale-r6 (26589) is IN lru's range (>lru-min 10360). So NOT "all whale < all lru" anymore.
The n=5 "fully separated p≈0.008" was because I hadn't yet sampled a whale high-metastable draw. whale ALSO has
rare huge p99 spikes (metastability affects both). CORRECTED STAT: Mann-Whitney U=21/24, exact one-sided
**p≈0.033** (still significant) — whale STOCHASTICALLY lower. Median p99 whale ~7.7s vs lru ~12.6s (~40% lower);
5/6 whale below lru-best (10360); mean 10675 vs 15982 (~33%, but mean outlier-inflated → use MEDIAN ~40%).
★ HONEST HEADLINE (n=6/n=4): whale reduces the TYPICAL (median) λ=3 p99 ~40% vs lru (Mann-Whitney p≈0.033), 5/6
runs below lru's best — but both policies have rare high-metastable p99 spikes (~26-28s), so it is a DISTRIBUTIONAL/
typical reduction, NOT a per-run guarantee or full separation. MUST update paper (was "fully separated p≈0.029" →
"stochastically lower, Mann-Whitney p≈0.033, overlapping high tails"). This is why the extra replicates mattered
(revealed fragility). lru-r5 running (chain) for lru n=5.

## 65. FINAL SAMPLE: whale n=6 / lru n=5. Typical p99 reduction ~45%, Mann-Whitney p≈0.026. Replicate phase DONE.
lru-r5 λ=3 = 17133ms (median 537). lru n=5 p99 {10360,11254,13991,17133,28325}. whale n=6 {6189,6574,6698,8715,
9287,26589}. Mann-Whitney U=26/30, exact one-sided **p=0.026**; whale median p99 7706 vs lru median 13991 →
**~45% typical reduction**; 5/6 whale below lru-min (10360). Overlapping high tails (whale 26589, lru 28325/17133)
→ distributional, NOT full separation. Paper updated throughout (n=5 lru, p≈0.026, ~45%, 0/7 never-cross). fig4
regenerated. Replicate phase COMPLETE (freed node). Final honest result stands. Remaining: certified W6 (job 19585,
still PD on contended pool, fire-and-forget). Campaign deliverable done + rigorously honest (4 self-corrections
total: goodput-floor, median-win, full-separation, queueing-mechanism all retracted under scrutiny).

## 66. ★★★ OFFLINE POLICY SWEEP (7 causal rules): whale UNIQUELY captures; naive size_only CATASTROPHIC (−529%)
Offline replay @10.7M, Belady-capture = (lru−p)/(lru−opt):
  lru/slru/car/LFU  = 7.24M  → +0.0%  (recency AND frequency rules ALL coincide/fail)
  fifo 43.6M −501% | mru 33.7M −366% | size_only 45.6M −529% (CATASTROPHIC) | whale 5.55M +23.4% | opt +100%.
★ KEY SHARPENING: it is NOT "size" naively — pure size_only (evict biggest regardless of proven) is CATASTROPHIC
(−529%, evicts big proven multi-turn convs → destroys reuse). The unique winning rule = evict biggest UNPROVEN,
PROTECT proven (whale). So the insight is precisely "SIZE-among-unproven + proven-protection," not size alone and
not recency/frequency. This is a much sharper, more defensible claim (7-policy sweep isolates the exact ingredient).
★ Also: LFU (frequency) joins lru/slru/car at exactly 0% — frequency is as blind as recency here (proven convs
already have recent access + high freq, so LFU's victim = same old unproven turn-0). Add to §4.2 offline proof.

## 67. CERTIFIED W6 CONFIRMATION RUNNING (job 19585 on 0-3, a certified node). Fire-and-forget caught a freed hold.
The 0-3 hold released → my fire-and-forget cert job started 05:16. Runs whale-cert (full sweep) then lru-cert
(sequential, same certified node) → same-node whale-vs-lru λ=3 on a CERTIFIED node (closes W6). whale-cert λ=3
~06:05; lru-cert λ=3 ~08:50 (after whale-cert full sweep). Will compare the certified whale p99 vs lru p99 to the
nodeset-0 result (typical p99 −45%, p=0.026). If it reproduces → W6 closed, cross-node confirmed. If not → GPU
effect node-specific (offline size result still stands). Monitoring; fold into paper §5.5/§8/reviews W6 when landed.

## 68. whale-cert (CERTIFIED node 0-3) λ=3 = 20406ms — median REPRODUCES cross-node; p99 metastable (as predicted)
whale-cert λ=3: median 527.93 (≈ nodeset-0 whale median ~555 → MEDIAN reproduces cross-node ✓), p99 **20406ms**
(HIGH metastable draw — within whale's observed range, cf. nodeset-0 whale-r6 26589), hit 0.6956, req 3.02.
★ HONEST W6 read: this single certified run CONFIRMS (a) the whale median behavior reproduces on a certified node,
and (b) p99 is metastable on the certified node TOO (a high draw this run) — which REINFORCES the paper's core
metastability thesis (single-run p99 A/Bs unreliable). ⇒ A single certified whale+lru PAIR cannot cleanly confirm
the ~45% p99 reduction (by our own finding); a proper certified confirmation needs median-of-k, which the
contended certified pool precludes. lru-cert λ=3 pending (~2h, after whale-cert full sweep). Will report W6 as:
median cross-node consistent; p99 metastable on both nodes; certified median-of-k = future work (pool-limited).
This is the honest framing — the certified run neither over-confirms nor refutes; it exemplifies the metastability.

## 69. W6 RESOLVED (honestly): whale median reproduces cross-node; freed certified node (single-pair p99 would mislead)
Scancelled the certified job after whale-cert λ=3 (median 528 reproduces nodeset-0 whale ~555). Rationale: a single
certified whale+lru PAIR cannot confirm the p99 reduction — both are single metastable draws (whale-cert drew a
high 20.4s), and a single-run certified A/B is exactly what the paper shows is unreliable; pairing it with one
lru-cert could produce a MISLEADING single-run comparison. Rather than manufacture a noisy/misleading certified
pair, I report W6 honestly: (a) the whale MEDIAN reproduces on a certified node (528ms ≈ 555ms → node-independent);
(b) the p99 is metastable on the certified node too (this run 20.4s) — reinforcing the metastability thesis; (c) a
definitive certified p99 confirmation needs median-of-k on certified, precluded by pool contention → stated
limitation + future work. Also freed the contended certified node 0-3 for sibling cells (good-neighbor). This is
the honest, well-stewarded resolution. Update reviews W6 + §5.5 note accordingly.

## 70. DECISION: NOT pursuing certified median-of-k (stewardship). Campaign complete; holding.
Considered re-grabbing the (currently idle) certified node 0-3 for a proper certified median-of-k (n≥3 whale + n≥3
lru, ~5h) to upgrade W6 from "median reproduces cross-node" to "p99 confirmed cross-node". DECIDED NO: (1) W6 is
already honestly addressed (median reproduces §68; certified p99 median-of-k scoped as future work); (2) the upgrade
is INCREMENTAL on an already-complete, submission-ready paper; (3) monopolizing the only free certified node ~5h
(right after freeing it good-neighbor) is poor stewardship of the contended pool (4 cells / 4 nodes). Marginal value
+ real shared-resource cost ⇒ don't do it. Campaign is COMPLETE: paper submission-ready (verified coherent, W1-W6
addressed, 5 self-corrections, committed 741f48701). Holding in maintain mode; will act only on genuinely new/
higher-value work. Not churning the clean deliverable.

## 71. REVERSAL (new info): 0-3 stayed idle → launched certified median-of-k for proper W6 (p99 cross-node).
§70 declined the certified median-of-k on stewardship grounds. NEW INFO: 0-3 stayed IDLE ~15+ min with NO sibling
demand (siblings occupy the other 3 certified nodes and didn't grab it) → using genuinely-idle certified capacity
is good value, not depriving anyone. So I launched a certified median-of-k on 0-3 (scancel-after-λ3, ~50min each):
lru-cert1/whale-cert2/lru-cert2/whale-cert3/lru-cert3 (19629-33) → with existing whale-cert(1) gives certified
whale n=3 + lru n=3. This PROPERLY closes W6 (certified p99 DISTRIBUTION vs nodeset-0, not a single misleading
pair). Expected: certified whale p99 mostly 6-9s (+occasional high, like whale-cert1's 20.4s) below certified lru
10-28s → typical p99 reduction reproduces cross-node. If a sibling queues for 0-3, I'll reassess. Managing the
chain via scancel-after-λ3. Decision updated on evidence (idle-persistence) — the right researcher move.

## 72. ★★★ lru-cert1 λ=3 = 5893ms (LRU PASSES SLO on certified!) → NODE CONFOUND in absolute p99. Critical.
lru-cert1 (CERTIFIED 0-3): median 561, p99 **5893ms (<8s → PASS)**, hit 0.67. On nodeset-0 lru was NEVER <10s
(n=5: 10.4-28.3s). ⇒ the certified node is FASTER/less-contended than nodeset-0 (memory: nodeset-0 "FLAKY in
v0.2"). So ABSOLUTE p99 is NODE-DEPENDENT: nodeset-0 slower → both policies high p99 (lru 10-28, whale 6-26);
certified faster → both lower (lru-cert1 5.9!). Single certified pair ordering FLIPPED: whale-cert1 20.4 (high draw)
vs lru-cert1 5.9 (low draw) — OPPOSITE of nodeset-0.
★ IMPLICATIONS (honesty-critical): (1) the SAME-NODE relative comparison (whale vs lru on nodeset-0) is still
valid; (2) but ABSOLUTE goodput@SLO / SLO-crossing is NODE-DEPENDENT — on the faster certified node even LRU
crosses 8s. So "whale moves goodput 0→3.02" is partly a nodeset-0 (slow-node) artifact; on a fast node lru already
achieves goodput>0. (3) Need certified n=3 each to see if the RELATIVE p99 reduction (whale<lru) holds on certified
or was nodeset-0-specific. This is a MAJOR finding the median-of-k surfaced — must reflect in paper (node-dependence
of absolute goodput; relative reduction is the portable claim, pending certified n=3). Continuing chain: whale-cert2.

## 73. ★★★ whale-cert2 = 11461 → certified whale {20406,11461} BOTH HIGH vs lru-cert {5893}. Whale may be nodeset-0-SPECIFIC.
Certified (0-3, faster node) so far: whale {20406, 11461} (both fail, median ~16s) vs lru {5893} (pass). On this
FASTER certified node, whale is running HIGHER than lru — OPPOSITE of nodeset-0 (where whale {6-9s+26.6} < lru
{10-28s}). ⚠️ This threatens the whale p99-reduction as NODESET-0-SPECIFIC (slow-node artifact), NOT node-portable.
n still small (whale n=2, lru n=1) — could be unlucky whale draws — so getting lru-cert2/3 + whale-cert3 to resolve.
IF certified confirms whale ≥ lru: the GPU mechanism does NOT reproduce cross-node → 6th self-correction: report the
OFFLINE size-signal insight as the robust contribution and the GPU p99-reduction as nodeset-0-specific/unconfirmed
cross-node. This is why the certified median-of-k was ESSENTIAL (not incremental) — it may overturn the empirical
mechanism. Continuing: lru-cert2, whale-cert3, lru-cert3. Honest either way; the data leads.

## 74. ★★★ lru-cert2 = 31203 → certified lru {5893,31203} spans 5.9-31.2s. NO clear whale advantage on certified.
Certified (n=2 each): whale {20406, 11461}, lru {5893, 31203}. BOTH wildly metastable (5.9-31.2s); rough
Mann-Whitney U=2/4 → NO separation (unlike nodeset-0's clean whale<lru p=0.026). ⇒ strong evidence the whale
p99-reduction is NODESET-0-SPECIFIC, not node-portable. Meanwhile the METASTABILITY finding is REINFORCED (the
certified node is even more metastable: lru alone spans 5.9→31.2s). getting whale-cert3 + lru-cert3 for n=3 each to
finalize, but the picture is clear: no robust cross-node whale advantage.
★ LIKELY 6TH SELF-CORRECTION (honest, major): the GPU whale p99-reduction does NOT robustly reproduce on a second
(faster) node — it is nodeset-0-specific. ROBUST contributions that SURVIVE: (1) the OFFLINE size-signal insight
(deterministic, node-independent — the paper's core); (2) the metastability characterization (STRENGTHENED — two
nodes, both wildly metastable, single-run/single-node A/Bs unreliable); (3) negatives. The whale GPU mechanism
must be DOWNGRADED to "nodeset-0-specific / not confirmed cross-node." This is exactly the value of the certified
median-of-k. Paper reframe pending whale-cert3/lru-cert3. Honest — the data leads, even against my mechanism.

## 75. whale-cert3 = 6794 (low draw) → certified whale {6794,11461,20406} n=3, median 11.5s. lru-cert3 decisive.
Certified whale n=3 {6794, 11461, 20406} median 11461. Certified lru n=2 {5893, 31203}. whale median (11.5s) sits
BELOW lru so far, but both wildly metastable (whale 6.8-20.4, lru 5.9-31.2). lru-cert3 (final run) decides the
certified n=3-each median comparison: if lru median stays high (~18s) → whale advantage weakly holds cross-node
(but noisy); if lru median ~6-11s → no advantage (nodeset-0-specific). Either way the CERTIFIED node is far MORE
metastable than nodeset-0 (both policies span 5.9-31s vs nodeset-0's tighter bands) → REINFORCES metastability as
the central finding, and shows the absolute p99 is very node/run-dependent. Getting lru-cert3 then finalizing.

## 76. ★★★★ VERDICT (6th self-correction, MAJOR): whale GPU p99-reduction is NODESET-0-SPECIFIC — did NOT reproduce.
CERTIFIED node 0-3, n=3 each: whale {6794,11461,20406} median 11.5s; lru {5893,6129,31203} median 6.1s.
Mann-Whitney U(whale<lru)=3/9 → NO whale advantage (lru median actually LOWER, 6.1 vs 11.5s). Both wildly
metastable (whale 6.8-20.4, lru 5.9-31.2s). This is OPPOSITE nodeset-0 (whale median ~7.7s < lru ~14s, p=0.026).
⇒ **The nodeset-0 whale p99-reduction does NOT reproduce on a second (certified) node.** The apparent same-node
advantage on nodeset-0 was node-specific (a slow-node + lucky-clustering artifact); on the certified node the two
policies are indistinguishable (both dominated by huge metastable p99 swings).
★★ WHAT SURVIVES (robust): (1) the OFFLINE size-signal insight — deterministic, node-independent (7-policy sweep;
size-among-unproven+proven-protection uniquely captures Belady headroom); (2) the METASTABILITY finding, now
CENTRAL and CROSS-NODE-demonstrated: goodput@SLO is so variance-dominated that a same-node A/B significant on one
node (p=0.026) FAILS TO REPRODUCE on another → single-node A/Bs (even same-node, replicated) are unreliable for
residency-policy claims. This is a strong cautionary methodological result. (3) gap-cap analysis; (4) negatives.
★★ RETRACT: the whale GPU p99-reduction as a portable mechanism win. It becomes: "captured headroom offline +
showed a same-node p99 reduction on ONE node that did not reproduce on a second — an object lesson in the
metastability." PAPER MUST REFRAME: lead with the size-signal insight + the (now cross-node-demonstrated)
metastability/methodology; downgrade the GPU whale result to node-specific/non-reproducing. This VINDICATES running
the certified median-of-k (§71 reversal was right — it overturned my own mechanism honestly). The data led.
