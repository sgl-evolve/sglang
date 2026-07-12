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

## Versions (test submissions)
- **v0-baseline** (stock sweep, clean reference) — job 19437, QUEUED. [pending curve]

## Formal submissions
- (none yet)
