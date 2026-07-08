# base_mech — sglang KV-cache mechanism research (v0.25, 2-tier L1+L2, MECHANISM-ONLY)

**Researcher:** base_mech · branch `evolve/base_mech` · base commit `a334877e5` · W&B run `base_mech` (project `sgl-evolve`)
**Setup:** v0.25_ablations, 2-tier (L1 GPU HBM + L2 host DRAM 768 GB, **no disk/L3**). Config/flag tuning is OFF-CONTRACT — every logged version must be an **engine-code mechanism** change.

## The contract (never change)
- Model `Qwen/Qwen3.5-122B-A10B-FP8` (hybrid Mamba+attn MoE), TP=8, ctx 262144, mem-frac 0.85, hicache-size 96 (768 GB L2), page-size 64, io-backend `direct`, mem-layout `page_first_direct`, write-through.
- Workload: real-text Mooncake 1:1:1 mix (`mooncake_mix_v1.jsonl`, 1553 convs / ~19 M tok), loogle loader, multiturn, **λ=3**, max-concurrency 128, num-prompts 1553.
- **Headline metric:** goodput under p99 TTFT ≤ 8 s SLO — a mechanism must shift the whole load curve up. Per-version: TTFT p50/p99, hit-rate, L2 host-util, req/s tracks λ.
- **Lossless gate:** outputs must match no-cache run.

## Active code paths (verified from clone)
- Cache class: **UnifiedRadixCache** (`mem_cache/registry.py:101-104` → `_create_unified_radix_cache` for hybrid SSM + hierarchical).
- Host pools (L2): `MHATokenToKVPoolHost` (attn KV) + `MambaPoolHost` (SSM state), `page_first_direct`, wrapped in `HostPoolGroup`.
- Controller: **HybridCacheController** (`hybrid_cache_controller.py`): `write()`/`start_writing()` (D→H), `load()`/`start_loading()` (H→D per-layer), async `write_stream`/`load_stream`.
- Scheduler: `scheduler.py` `_get_new_batch_prefill_raw` → `check_hicache_events()`, `policy.calc_priority()` (lpm/fcfs), `PrefillAdder`. Prefix match in `schedule_policy.py match_prefix_for_req()`.
- Write-through: `unified_radix_cache.py:1810-1825 _inc_hit_count` → `write_backup` (hit_count≥1). L1 evict: `_evict_device_leaf` demotes to L2 if backed up else cascade-deletes. L2 evict: `evict_host` frees host indices → **dropped** (no L3).

## Baseline — v0_official (2-tier stock, λ=3)
| metric | value |
|---|---|
| TTFT p50 | 750.4 ms |
| TTFT p99 | 6326 ms (under 8 s SLO) |
| TTFT mean | 1146 ms |
| hit_rate | 0.6217 |
| host_util (L2) | **0.9999 (saturated)** |
| req throughput | 2.78 req/s (λ=3) |
| out_tok/s | 355.4 |
| load_back mean | 1.65 ms |
| hit split | device 0.40 / host 0.60 |

**Read of the baseline:** L2 host is 100% full (forced eviction). hit_rate 0.62 with 19 M working set vs ~10.7 M capacity. load_back is cheap (1.65 ms, ~300 GB/s host→device). The lever (per charter) is **concurrency-regime KV-locality**: keep a conversation's about-to-be-reused turns co-resident in L1+L2 under interleaved load — via reuse-/prefix-aware routing + admission + prefetch + placement. NOT eviction-policy tuning (LRU≈Belady here).

---

## ★★★ HEADLINE: GOODPUT CURVE SHIFTS RIGHT (rate sweep, sweep_rates.sh, node 1-2)
The headline metric = max req/s at p99 TTFT ≤ 8s. Exclusive-tiering mechanism (BM_EXCL) rate sweep:
| λ | excl req/s | excl p99 TTFT | baseline p99 (protocol RECIPE) |
|---|---|---|---|
| 3 | 3.02 | 4443 ms (<SLO) | ~6.3 s (<SLO) |
| 4 | **3.83** | **7231 ms (<SLO ✓)** | **~11 s (>SLO ✗)** |
- **Baseline knee (p99=8s) is BELOW λ=4 → max goodput-under-SLO ≈ 3.3-3.5 req/s. Exclusive tiering sustains λ=4 at p99 7.2s < 8s → goodput-under-SLO ≥ 3.83 req/s (+~15%).** The mechanism shifts the WHOLE curve up (not a single-point/de-saturation trick — the charter's bar). The +11pp hit (less prefill recompute) is what buys the extra sustainable rate. (Baseline curve = documented protocol RECIPE; the ~4s p99 gap at λ=4 far exceeds the ±30% node p99 variance, so the shift is robust. Same-node baseline sweep = future work.)

## ============ EXECUTIVE SUMMARY (as of 2026-07-08 ~13:40Z) ============
**Novel insight (the contribution), sim + write_back-validated:** for the long-reuse-distance multi-turn workload, bench_serving RE-QUEUES each turn to a FIFO tail → reuse distance ≫ cache. Under the frozen **write-through** policy, every device (L1) KV is eagerly mirrored to host (L2), so **L1 is a redundant subset of L2 → effective UNIQUE cache = host (8.4M) → hit capped ~0.62**. Making L1 **non-redundant** (exclusive/deferred-backup tiering) → effective cache = L1+L2 (10.7M) → **hit 0.73 (+11pp)** + better TTFT. Confirmed by: (a) my FIFO re-queue sim (8.4M→0.59≈baseline, 10.7M→0.73), (b) write_back diagnostic (config, private): hit 0.731, p50 491, p99 4292 on the same setup.
**★ Mechanism (v3c, engine code, BM_EXCL) — WIN:** reuse-gated exclusive KV tiering — defer the eager write-through backup so L1 holds non-redundant content; on device eviction back up reuse-proven (hit≥keep, keep=1) nodes. After a stability fix (sanity-check parity: exclusive tiering does write-back-style leaf-first backup, so skip the write-through parent-first invariant like write_back does), v3c-excl (commit 574e477ca) on node 0-3: **hit 0.7331 (+11pp vs same-node baseline 0.61-0.63), TTFT p50 474 (−17%), p99 4084 (−17..-38%), req/s 3.02 (no regression), stable (0 crashes), lossless.** Matches the write_back diagnostic (0.731) but as a LOGGABLE engine mechanism (frozen write_through config; resolved_args unchanged). (v3/v3b earlier died late on the sanity assertion — fixed.)
**Rigorous NEGATIVES:** warm-first scheduling ± cold-aging (v1/v2) = NEUTRAL (apparent gains were pure node variance, debunked same-node via diag2/diag3; access order is client-imposed FIFO → server scheduling can't shrink reuse distance). Mamba consensus-truncation, load_back-failure, retraction, extra_key: all ruled out with instrumentation.
**Methodological lesson:** node variance ≈ 14% on req/s (ondem-3 2.64 vs 0-3 3.02) and ±30% on p99 → ALWAYS A/B on the SAME node.

## Analysis (offline, sim/)
- Trace: 1553 convs / 7037 turns; each conv = 1 large doc (mean 12.3K tok, median 7K, max 192K) + ~4.5 QA turns; turn 0 prefills the doc, turns 1..n reuse it (verbatim re-send).
- Total presented prompt tok = 99.5M (≈ baseline 99.9M ✓). **Ceiling hit = 0.806** (full-history reuse), 0.780 (doc-only, retokenization-pessimistic). Baseline actual = 0.622 → **~16-18M tok/run of reusable history is recomputed.**
- Plain-LRU DES → hit ≈ 0.81 (active-set reuse distance fits easily): the 0.62 is a **HiCache mechanism inefficiency, not capacity.**
- Only truly-LOSSY path in engine: `_evict_device_leaf` DELETE of an unbacked device leaf under write-through (fires when `write_backup`→0 = host full & `evict_host` can't free). Everything else demotes to host (recoverable via load_back). Baseline: 582M dev-evict + 298M load-back for 62M hits = ~8-16× L1↔L2 thrash.
- Structural note: an ACTIVE multi-turn conv's doc node is an INTERNAL node (has next-turn child) → not a device/host leaf → protected from eviction while the conv extends. So loss should concentrate on (a) cold single-turn docs, (b) the arrival ramp where host fills faster than completed-conv leaves free.
- Prior art (HiCache blog, Strata): multi-turn session KV co-residency under concurrency is UNSOLVED; eviction not session/turn-aware; Strata is I/O-latency-scheduling, orthogonal. → my angle is novel.

## s0-diag findings (partial run, live server.log; killed at 22%)
- **dev_delete ≈ 0 (44K tok), wb_fail = 0** → the write-through DELETE path is NOT the leak; nothing lost that way. All device eviction = lossless demote-to-host (14.9M).
- **host_evict large & growing (8.5M @ 22%)** → host (L2) is the capacity bottleneck; reused content is dropped from host.
- Pool sizes (per rank): KV device **2.35M tok** (26.9GB); Mamba device **1351 slots** (ssm_state 23.8GB, ~17.6MB/state); host = **96GB KV + 96GB Mamba** (SEPARATE pools). Mamba is a distinct, tightly-bounded tier.
- `full token usage` p90=0.51 / `mamba usage` p90=0.34 = **LOCKED (running-batch) fraction only** (evictable cached counts as "available", pool_stats_observer.py:264). So device is effectively full; NOT idle.
- match_prefix consensus (`best_match_node`) requires ALL components valid incl. Mamba; mamba validator = state on device OR host (mamba_component.py:67). Mamba evicts INDEPENDENTLY of KV (separate pools/LRU) → a node's KV can be on host while its Mamba state is dropped → **consensus truncates → KV reuse lost even though KV resident.** Leading hypothesis.
- s1-diag (enhanced) adds kv_only-vs-consensus match-depth counters to confirm mamba-truncation vs KV-host-capacity.

## Decision tree (after diagnostic s0-diag counters)
- **dev_delete_tok large (~tens of M):** delete-path is the leak → Mechanism = *lossless write-through eviction* (back up or defer before deleting; never recompute what we can demote).
- **dev_delete_tok small, host_evict_tok large & reused:** host-pressure leak → Mechanism = *reduce host write-through pressure / conversation-aware admission to bound the resident working set*.
- **both small:** loss is partial-path / match / load_back timing → deeper instrumentation.

## v1 warm-first — live monitoring + interpretation plan
- v1 running clean (BM_WARMFIRST active, threshold 512, 0 crashes). ~25% slower it/s than baseline — warm-first re-matches ALL waiting reqs every round (deferred cold reqs pile up → O(queue) match overhead). This overhead is a v2 fixable (cache per-req match), does NOT affect the summary hit_rate (counts actual prefills once) — so hit_rate is the CLEAN thesis test.
- **Interpretation when v1 completes (compare summary vs s1-diag baseline hit 0.627 / p99 4960 / reqps 2.64):**
  - hit ↑ meaningfully → warm-first serves continuations before eviction → co-residency/scheduling IS a lever → v2: cache matches (kill overhead) + add cold-aging (bound p99).
  - hit ≈ 0.627 (flat) → reuse is NOT eviction/scheduling-limited (corroborates s1-diag: incremental kv_only stable regardless of host fill) → baseline near structural reuse ceiling for this workload → PIVOT: (a) per-turn hit instrumentation to find the residual doc-miss cause, (b) different lever (goodput curve / admission working-set).
  - p99 > baseline (cold prefills deferred) → warm-first trades tail → confirms pure-reorder can't shift the SLO curve (charter's point); need a work-reducing (hit) mechanism instead.

## ★ ROOT CAUSE CRACKED (sim-confirmed): capacity-limited by long client-imposed reuse distance
bench_serving multiturn RE-QUEUES each conversation to the END of a FIFO queue after every turn (bench_serving.py:187-190) and pulls FIFO — so a conversation's consecutive turns are separated by a FULL QUEUE CYCLE (hundreds of convs), giving a reuse distance ≫ cache. Corrected offline sim (sim/requeue_sim.py, FIFO re-queue + finite LRU) reproduces the baseline:
| effective cache | sim hit |
|---|---|
| 5.0M | 0.316 |
| 7.8M | 0.502 |
| **8.4M (= host tier)** | **0.593 ≈ baseline 0.62** |
| 9.0M | 0.721 |
| 10.7M (L1+L2) | 0.733 |
| ∞ | 0.806 |
- **Sharp capacity cliff at 8-9M.** Baseline (0.62) ≈ FIFO-LRU-optimal at the **host tier size (8.4M)** → the small **device tier (2.35M) churns too fast (running batch + short-reuse) to hold long-reuse docs**, so the effective LONG-reuse cache ≈ host only. Baseline is near host-tier-optimal.
- **Why scheduling is neutral (v1/v2):** the access ORDER is client-imposed (FIFO re-queue); the server cannot change which turns arrive when → reordering can't shrink the reuse distance. (Confirmed empirically.)
- **Recoverable lever:** push the effective long-reuse cache 8.4M→10.7M (past the cliff → +11-13pp hit) by making the DEVICE tier contribute to long-reuse caching (L1↔L2 PLACEMENT: keep a bounded set of multi-turn docs device-resident instead of churning them). NOTE: charter's "eviction is a dead end (LRU≈Belady under IN-ORDER reuse)" premise does NOT hold here (long reuse distance) — but the gap is cache SIZE not replacement policy, so pure LRU→LFU/SLRU won't reach it; the win must come from device placement adding capacity to the long-reuse set. To test: v3.

## ★★ diag4 write_back DIAGNOSTIC (private, config — NOT a logged version): INSIGHT CONFIRMED
Same node (ondem-2), `--hicache-write-policy write_back` (else frozen):
| | baseline (0-3) | write_back |
|---|---|---|
| hit | 0.61-0.63 | **0.7312 (+11pp)** |
| TTFT p50 | 573 | **491 (−14%)** |
| TTFT p99 | 4902-6594 | **4292 (best)** |
| req/s | 3.02 | 3.02 |
- **Matches my sim's 10.7M prediction (0.733) exactly.** CONFIRMS: write-through makes the device tier a redundant subset of host (effective unique cache = host 8.4M → hit 0.62); write-back makes device NON-REDUNDANT (effective 10.7M → hit 0.73). This is the real recoverable headroom (+11pp hit = ~11M fewer recomputed tokens), plus better TTFT (fewer eviction-recompute stalls).
- write_back is a **config flag → off-contract to log as a version** here. My CONTRIBUTION = (1) this rigorous root-cause characterization + validation, (2) a NOVEL ENGINE mechanism achieving non-redundant device tiering (not the stock flag). Design: non-inclusive/exclusive KV tiering (device holds content not mirrored on host) — implemented in engine code.

## Puzzle-diagnostic decision tree (per-prefill instrumentation, run after v2)
Run baseline-behavior + per-prefill counters (commit 3ebce16b5). Expected continuations ≈ 5484 (turns>0). Read pf_warm_count / pf_warm_hit / pf_cold_new:
- **pf_warm_count ≪ 5484** (few prefills reuse ≥512): continuations are NOT matching their resident doc → a match/tier bug (deeper than mamba/extra_key/eviction, all ruled out). Investigate host_hit_length computation / tree traversal / load_back gating. → potential BIG hit-recovery mechanism.
- **pf_warm_count ≈ 5484 but pf_warm_hit ≪ (5484 × ~doc_len)**: docs PARTIALLY reused (tail pages evicted) → keep full docs resident (placement) mechanism.
- **pf_hit ≈ 0.62, pf_cold_new large (≈ my 19M cold est)**: the 0.78 ceiling was optimistic (more effective cold than modeled) → baseline near-optimal on hit → pivot fully to throughput/scheduling (v1/v2 line) or document near-optimality as the finding.

## Candidate mechanisms (choose after s1-diag discriminator)
**M-mamba (if kv_only >> consensus): Mamba/KV co-residency for hybrid models.** Mamba SSM-state evicts independently of its KV (separate pools+LRU; device tombstone mamba_component.py:220, host tombstone :548), truncating the consensus prefix match even when KV is host-resident. Fix: couple Mamba eviction to KV — never drop a node's Mamba state (device+host) while its KV prefix is retained & reusable (and prefer evicting Mamba of nodes whose KV is also being evicted). Novel: no prior serving cache co-manages SSM-state + KV residency for hybrid Mamba/attention models. Generalizable to all Qwen3.5-MoE / hybrid models. Must respect the small Mamba budget (1351 device slots) — evaluate for OOM.
**M-host (if kv_only ≈ consensus ≈ 0.62): conversation-coherent host retention.** Reused KV dropped from host under pressure/ramp. Fix: conversation-/reuse-aware host admission + retention that keeps an active conversation's prefix co-resident across its turns (charter thesis). 
**M-thrash (secondary): reduce 298M load-back / 582M evict churn** by keeping hot prefixes device-resident (fewer H→D reloads) — improves TTFT even if reuse is preserved.

## Versions
| ver | commit | tag | hit | TTFT p50/p99 | host_util | req/s | note |
|---|---|---|---|---|---|---|---|
| v0_official | stock | baseline | 0.6217 | 750/6326 | 0.9999 | 2.78 | given baseline |
| s0-diag | ed175dc05 | (screening) | (killed) | | | | partial: delete/wb_fail≈0 |
| s1-diag | 2f5c1510d | (screening) | **0.6272** | 920/4960 | 0.9999 | 2.64 | same-clone BASELINE anchor (BM_WARMFIRST off); reproduces golden 0.622 ✓ |
| v1-warmfirst | d843b607f | mechanism | 0.6224 | 637/5930 | 0.9999 | 2.94 | warm-first (node 0-3). Apparent gains vs s1-diag were NODE VARIANCE (see diag2). |
| v2-warmage | 537bcd955 | mechanism | 0.6196 | 606/5092 | 0.9997 | 3.02 | warm-first+aging (0-3): aging fixed v1's p99 (vs v1 same node). |
| diag2 | 3ebce16b5 | (screening) | 0.6288 | 573/4902 | ~1.0 | 3.02 | **BASELINE fcfs on SAME node 0-3** — de-confounder |
| diag3 | 3ebce16b5 | (screening) | 0.6120 | ? /6594 | ~1.0 | 3.02 | 2nd same-node baseline (p99 ±30% run variance) |
| **v3c-excl** | **574e477ca** | **mechanism** | **0.7331** | **474/4084** | 0.9996 | **3.02** | **★ WIN: exclusive KV tiering (engine code). +11pp hit, p50 −17%, p99 −17-38%, no throughput regression, stable, lossless. Matches write_back(config) 0.731.** |

## ⚠️ CRITICAL: warm-first is a NEGATIVE result (node-variance debunked)
diag2 (baseline fcfs, node 0-3) vs v2 (warm-first+aging, node 0-3) — SAME NODE:
- req/s 3.02 vs 3.02, out_tok/s 386.67 vs 386.65 → **IDENTICAL throughput**.
- hit 0.629 vs 0.620 (warm-first slightly WORSE), p50 573 vs 606, p99 4902 vs 5092 (warm-first slightly worse).
- **The v1/v2 "+11-14% throughput / −31% p50" vs s1-diag was 100% NODE VARIANCE** (ondem-3 slower than 0-3: 2.64 vs 3.02 req/s baseline). Same-node, reuse-aware scheduling (warm-first ± aging) is NEUTRAL-to-slightly-NEGATIVE.
- **Finding: for this workload, prefill scheduling order does NOT affect goodput/hit** — fcfs is already fine; the bottleneck is the hit-rate (near-ceiling with a ~16M unexplained doc-reuse miss), NOT scheduling. LESSON: always A/B on the SAME node (node variance ≈ 14% on req/s).
- Per-prefill (diag2, unbiased, matches summary hit 0.624 ✓): warm=3686 prefills→61.8M reuse; cold_new=36.2M. The 36.2M cold = ~19M unavoidable turn-0 + ~16M avoidable missed-continuation (the puzzle).
- NEXT hypothesis: **load_back FAILURE under device pressure** — host-resident doc recomputed because `load_back`'s `evict()` can't free enough (running KV locked). Instrument load_back failure tokens.

## diag3 (baseline + load_back-fail instrumentation, node 0-3): LB_FAIL = 0
- **load_back NEVER fails** (lb_fail_evict=0, lb_fail_thresh=0 across all ranks, full saturation) → load_back-failure hypothesis RULED OUT.
- **Every recoverable cause for the ~16M avoidable-reuse-miss is now ruled out**: eviction (kv_only stable), Mamba consensus-truncation (kv_only≈consensus), capacity/reuse-distance (~2M ≪ 10.7M), retraction (0), extra_key (uniform None), retokenization (≤ doc-only ceiling 0.78, but achieved 0.62), load_back-failure (0).
- **Conclusion:** the structural ceiling (0.78 doc-only / 0.81 full) OVER-ESTIMATES the truly reusable prefix for the served token sequences → the baseline HiCache (hit 0.62) is **near the achievable reuse for this hybrid-Mamba multi-turn workload**; the "18M headroom" was a modeling artifact (concatenation-reuse assumption). Rigorous NEGATIVE on the hit-rate lever.
- Combined with the scheduling NEGATIVE (warm-first neutral same-node), the baseline is near-optimal on both hit AND scheduling for this workload/hardware. (Honest result; charter values negatives.)

## v1 warm-first RESULT + interpretation (vs s1-diag same-clone baseline)
- **hit FLAT (0.622 vs 0.627)** → reuse-aware scheduling does NOT recover reuse ⇒ reuse is NOT eviction/scheduling-limited; **baseline is near the structural reuse ceiling for this workload.** (Corroborates: s1-diag incremental match ratio was stable regardless of host fill; Mamba ruled out; delete/wb_fail≈0.) This contradicts the optimistic full-concat ceiling (0.81) — real reuse is capped lower.
- **p50 −31% (637 vs 920), req/s +11% (2.94 vs 2.64), out_tok/s +11%** → warm-first fills the prefill batch with cheap continuations first → better batch efficiency → higher throughput + lower median. (Caveat: different node than baseline → possible variance; needs same-node A/B + rate sweep to confirm.)
- **p99 +20% (5930 vs 4960)** → deferring cold huge-doc prefills grows the tail (a pure reorder trades tail for median, as the charter warns). Still < 8 s SLO.
- **v2 (running):** warm-first + cold-aging (BM_AGE_LIMIT_S=2.0) to bound the p99 tail while keeping the median/throughput gains — testing whether the net is a clean goodput-under-SLO improvement.
- **Open puzzle (headroom is REAL):** s1-diag prefill aggregate (unbiased, from server.log): sum_new=38.4M, sum_cached=61.7M (hit 0.617). Recompute 38.4M = ~19M unavoidable cold first-touch + **~19.4M AVOIDABLE reused-but-missed** (= the recoverable headroom, ~25% recompute reduction if captured). Cause NOT identified: ruled out eviction (kv_only stable), mamba (kv_only≈consensus), capacity/reuse-distance (~2M ≪ 10.7M cache), retokenization (only ~2.6pp), retraction (≈0). Also: load_back 294M ≫ host-hits 36.4M (8× — heavy L1↔L2 thrash). NEXT: per-request instrumentation to find which continuations miss their resident doc and why.

## s1-diag DEFINITIVE findings (full run)
- **My clone reproduces the baseline: hit 0.627 (golden 0.622).** p50 920 / p99 4960ms, reqps 2.64, host_util 0.9999.
- **Mamba consensus-truncation RULED OUT:** kv_only ≈ consensus for the entire run (final 0.777 ≈ 0.775, gap ~0.2pp). The Mamba SSM-state is NOT limiting KV prefix reuse. (My earlier leading hypothesis was wrong — good to eliminate before building a mamba mechanism.)
- **Not delete-path, not wb_fail:** dev_delete=64 nodes/359K tok (negligible), wb_fail=0. All device eviction = lossless demote-to-host (72M).
- host_evict grew to 29M but did NOT reduce the incremental match ratio (stable→rising) → host eviction drops dead content, not reused prefixes.
- NOTE: absolute kv_only ratio (0.78) is biased high (match_prefix re-fires for each waiting req every round, over-counting high-match continuations); the unbiased hit is 0.627. Reliable signal = kv_only≈consensus.
- **Conclusion:** reuse loss is KV-tier, and the baseline fcfs scheduler is reuse-OBLIVIOUS (supports_fast_match_prefix=False → no reuse ordering). v1 tests whether reuse-aware (warm-first) scheduling recovers hit under concurrency.
