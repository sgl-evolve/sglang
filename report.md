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
