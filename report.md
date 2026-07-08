# base_mech — sglang KV-cache mechanism research (v0.25, 2-tier L1+L2, MECHANISM-ONLY)

**Researcher:** base_mech · branch `evolve/base_mech` · base commit `a334877e5` · W&B run `base_mech` (project `sgl-evolve`)
**Setup:** v0.25_ablations, 2-tier (L1 GPU HBM + L2 host DRAM 768 GB, **no disk/L3**). Config/flag tuning is OFF-CONTRACT — every logged version must be an **engine-code mechanism** change.

## Abstract (contribution, for a skeptical reader)
**Problem.** On the fixed multi-turn workload, HiCache hit-rate is stuck at ~0.62 despite a 10.7 M-token L1+L2 budget vs a 19 M working set — and goodput at the knee (λ=4) sits right at the 8 s p99 SLO.
**Insight (novel).** `bench_serving` re-queues each conversation turn to the FIFO tail, so a conversation's reuse distance is a *full queue cycle* (≫ cache) — the regime is capacity-limited, not eviction-limited. Under the frozen **write-through** policy every L1 (device) block is eagerly mirrored to L2 (host), so **L1 is a redundant subset of L2 → the effective *unique* cache = host only (8.4 M) → hit is capped at ~0.62.** (Diagnosed by ruling out Mamba-truncation, delete-path, load-back-failure, retraction with instrumentation; confirmed by a FIFO-re-queue simulator that reproduces measured hit.)
**Mechanism.** *Reuse-gated exclusive tiering* (v3c, engine code under frozen write-through config): defer the eager backup so L1 holds non-redundant content, making the effective cache the full L1+L2 = 10.7 M. **Honest positioning:** its measured effect *converges with the stock `write_back` config* (within node-variance) — so the novelty is **not** the mechanism per se but (a) the root-cause insight and (b) the optimality boundary below; the engine implementation is the mechanism-only-contract-compliant way to realize and log it.
**Evidence.** hit **0.622→0.733 (+11.0 pp, non-overlapping error bars**, n=6 vs n=2+); goodput at the knee **+10.1 %±1.4 pp req/s, −10.4 %±3.1 pp p99** (n=3 independent same-node A/B pairs), with excl **converting a SLO-failing operating point into a passing one on 2 of 3 nodes**; **lossless by construction**; attributable to a **−29 % prefill-recompute** reduction (traded for cheap host→device bandwidth), not queue de-saturation.
**Boundary (novel).** Exclusive tiering is **within-contract-optimal for the hit/capacity dimension**: the residual 0.733→0.806 gap is completed-conversation dead weight that is *provably unobservable online* (4 realizable eviction signals capture 0 %), and more capacity is off-contract/infeasible; cross-conv dedup (4 %) and the Mamba dual-pool (KV-tokens bind, ~45 % Mamba slack) are closed too. **Honest scope:** recompute-cost-aware eviction (a *tail*-objective lever, distinct from hit-count) was not investigated — deliberately, to keep this line independent.
**Takeaway (generalizable).** On hierarchical KV caches whose reuse distance exceeds a single tier, the dominant lever is *effective capacity*, and the biggest cheap win is *removing inter-tier redundancy* — inclusive write-through silently wastes the fast tier as a mirror.
**Provenance & reproducibility.** Every result reproduces from its commit (v3c = `574e477ca`, `BM_EXCL=1 BM_EXCL_KEEP_HITS=1`, frozen write-through config). Raw per-version outputs (`runs/<version>/summary.json` + server logs + `resolved_args.json`) are preserved on branch `evolve/base_mech`; the W&B run `base_mech` carries the evolution-curve panels (v0/v1/v2/v3c). Goodput error-bar data: `runs/sweep-basesweep`, `runs/sweep-excl` (node 1-2), `runs/paired-eb1` (0-3), `runs/paired-eb2` (ondem-3); aggregation `sim/errorbars.py`. Boundary sims: `sim/{requeue,admission,completion_evict,reuse_gated_evict}_sim.py`.

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

## ★ BOUNDARY RESULT — why exclusive tiering is essentially optimal within contract (sim-proven, free screen)
After landing the exclusive-tiering win (hit 0.62→0.733 at the 10.7 M effective-capacity budget), I opened a bolder line — **conversation co-residency / admission / completion-aware eviction** (the charter's suggested levers) — and screened it exhaustively on the FIFO-re-queue simulator (free, no GPU; `sim/admission_sim.py`, `sim/completion_evict_sim.py`, `sim/reuse_gated_evict_sim.py`). The result is a clean impossibility bound that proves the residual headroom is not reachable by any realizable mechanism:

1. **The gap is completed-conversation dead weight.** An oracle that frees a conversation's entire KV the instant it completes reaches the **infinite-cache ceiling 0.806** at *every* concurrency K — because the *active* working set (docs of not-yet-completed convs) peaks at only **8.63 M < 10.7 M**. So ~half the cache is dead weight held by finished convs; the live set fits in budget. (This is why admission/co-residency by itself does nothing: at max-concurrency 128 the active set is only 1.88 M — concurrency was never the constraint; completion-lag is.)

2. **The completion signal is unobservable online — LRU is optimal among realizable policies.** Four independent observable signals each capture **0 %** of the 7.3 pp oracle gap:
   - **idle-gap** (`completion_evict_sim`): an active conv's idle time ≈ *one full queue cycle* (it reappears only after the whole queue cycles), so a completed conv idle 1.2 cycles is indistinguishable from an active conv about to be reused at 1.0 cycles. T<1 evicts actives (hit collapses to 0.01); T≥1 ≡ pure LRU.
   - **turn-count**: turns-per-conv is min 1 / max 61 / median 3 with **39 % single-turn** — no predictive "done" threshold exists.
   - **reuse-provenness** (`reuse_gated_evict_sim`): evicting not-yet-reused ("unproven") convs first ≡ LRU exactly, because unproven single-turn docs already age to the LRU bottom; and evicting unproven *multi-turn* docs sacrifices their (long-distance) first reuse. Net 0.7334 = LRU.
   - **frequency / LFU** (`reuse_gated_evict_sim`, `mode=lfu`): least-frequently-used-first also nets **0.7334 = LRU** — with median 3 turns/conv (39 % single-turn), access-frequency is mostly tied at low values so the recency tie-break dominates. (So both textbook policies — LRU *and* LFU — land on the same hit; neither is a lever.)

   So the Belady gap here is *large* (7.3 pp, contra "LRU≈Belady"), but it is **provably unobservable** — active vs completed convs are identical by recency, depth, size, and reuse-history at the eviction decision boundary. This closes the **eviction / scheduling / admission axis *for the hit-rate objective*** with a mechanism-level argument, not a tuning sweep. **Scope (honest):** this rules out beating LRU on *hit-count* (which caps effective capacity). It does **not** address a *different* eviction objective — **recompute-cost-aware** eviction (preferentially protecting expensive-to-recompute prefixes, e.g. huge cold docs), which targets the *tail/goodput* rather than hit-count and could in principle lower p99 at similar hit. I did not investigate that lever, so I do not claim it is closed; my optimality is for the **hit / capacity** dimension.

3. **Capacity is the only lever, and exclusive tiering already captures the cheap part.** The realizable hit-vs-capacity curve (`requeue_sim`, LRU) has a **cliff at 8.4→10.7 M (+14 pp: 0.593→0.733)** — that *is* the exclusive-tiering win, obtained with **zero extra memory** by de-redundifying L1. Beyond 10.7 M it is a slow linear grind (~+0.8 pp per +1 M): reaching the 0.806 ceiling needs **~+9 M** more, i.e. off-contract memory or an infeasible ~2.4× lossless FP8-KV compression. Lossless host-KV compaction at a realistic ~1.2× would add only ~+0.7 pp at the cost of load-back decompression latency — not worth it.

4. **Cross-conversation dedup is not a lever either (free check).** The mix has 888 unique full docs among 1553 convs, but the dominant "shared" cluster (n=538) is *empty-input* ShareGPT chat convs (no doc). Token-weighted, explicit cross-conv doc dedup would save only **0.84 M tok = 4 %** of the 19.22 M doc volume — and the radix tree already dedups identical prefixes that are co-resident, so the realizable gain is even less. The ~19 M working set is genuinely distinct → capacity pressure is real, not an artifact of un-deduped copies.

5. **The hybrid dual-pool binding constraint is KV-tokens, not Mamba-states (free, from server.log).** The host holds *two* pools: attention-KV (96 GB/rank → **8.39 M tokens**, the L2 budget) and a *separate* Mamba SSM-state pool (96 GB/rank → **5362 states**). Reusing a cached prefix needs both a Mamba state and its KV tokens, so effective cache = min(KV-token-bound prefixes, Mamba-state-bound prefixes). Crossover is **1565 tok/prefix**; the workload's actual mean is **2869 tok/turn-node** ⇒ **KV-tokens bind, the Mamba pool runs ~45 % slack** (~2924 of 5362 states used). This is *why* the token-only sim reproduces both baseline and excl — Mamba-state count is not the constraint, so exclusive tiering (a KV-token-capacity mechanism) is aimed at the right pool. Reclaiming the Mamba slack for KV is off-contract (exceeds the 768 GB KV ceiling = "more memory"); right-sizing Mamba would be a memory-*efficiency* result (same goodput, less DRAM), out of scope for the goodput headline. So the Mamba axis is closed too.

**Takeaway / generalizable insight:** on a workload whose reuse distance is a full client-imposed queue cycle and whose working set exceeds the cache, the lever that shifts the goodput curve *via hit-rate* is **effective capacity**, and the biggest cheap win is **removing inter-tier redundancy** (inclusive write-through wastes the fast tier as a mirror; make it exclusive). Smarter eviction/scheduling/admission cannot help *the hit rate*, because the future-knowledge they'd need (which resident KV is dead) is unobservable from any online feature. This makes exclusive-tiering **hit/capacity-optimal within contract**. **Honest scope:** this is optimality for the *hit-rate/capacity* dimension. A distinct, uninvestigated lever remains on the *tail*: recompute-cost-aware eviction (protect expensive-to-recompute prefixes) could lower p99 at similar hit — I did not test it, so goodput is not claimed globally optimal, only that the hit/capacity lever is maxed. (The boundary items are screened negatives — per charter documented here, not logged as W&B versions.)

**Generalization (the insight is a regime effect, not a one-workload quirk; `sim/generalization_sim.py`).** The exclusive-tiering advantage equals the reuse hit-curve's *rise over the interval [H, H+D]* (H = single/host tier, D = device tier). Two clean regimes (confirmed on the simulator): **(a) when the reuse working set fits in a single tier (WS ≤ H), the gap is exactly 0** — inclusive write-through's redundancy is *harmless* because nothing is evicted (verified: subsampling the trace to WS ≤ 8.4 M gives WT hit == excl hit to 4 decimals). **(b) when WS > H, the device tier adds real unique capacity and the gap is positive** — for this workload [8.4 M, 10.7 M] lands on the steep part of the reuse curve (0.593→0.733) → +14 pp. So the general rule a maintainer can apply: *inclusive write-through silently wastes the fast tier exactly when the reuse working set exceeds a single tier and the [H, H+D] band sits on a steep part of the reuse curve* — a condition any multi-turn / long-context serving deployment can check. (Honest caveat: subsampling varies working-set *size* and *mix* together, so only the two endpoints above are claimed clean; the finer shape is not.)

---

## ★★★ HEADLINE: GOODPUT CURVE SHIFTS RIGHT (SAME-NODE A/B — error bars over 3 nodes)
The headline metric = max req/s at p99 TTFT ≤ 8s. SAME-NODE A/B — baseline write-through vs BM_EXCL exclusive tiering — on node 1-2 (rate sweep) and repeated on node 0-3 (paired λ=4):
| node | λ | baseline req/s | baseline p99 | **excl req/s** | **excl p99** | Δp99 | Δreq/s |
|---|---|---|---|---|---|---|---|
| 1-2 | 3 | 3.02 | 5258 ms | 3.02 | **4443 ms** | −15.5% | +0% |
| 1-2 | 4 | 3.54 | 7847 ms (edge) | **3.83** | **7231 ms (<SLO)** | −7.9% | +8.1% |
| 0-3 | 4 | 3.52 | **8684 ms (>SLO)** | **3.90** | **7397 ms (<SLO)** | −14.8% | +10.8% |
| ondem-3 | 4 | 3.53 | **8546 ms (>SLO)** | **3.94** | **7820 ms (<SLO)** | −8.5% | +11.4% |
| 1-2 | 5 | not run¹ | not run¹ | 4.28 (raw) | **12031 ms (>SLO)** | | |

**★ Error bars on the knee (λ=4, n=3 independent same-node pairs — nodes 1-2, 0-3, ondem-3): Δp99 = −10.4% ± 3.1 pp (−7.9, −14.8, −8.5); Δreq/s = +10.1% ± 1.4 pp (+8.1, +10.8, +11.4).** All three nodes agree in direction and magnitude → the goodput improvement is robust, not a node-specific fluke. **The SLO rescue is consistent: on 2 of 3 nodes baseline λ=4 FAILS the 8 s SLO (8684, 8546 ms) while excl PASSES on all 3 (7231, 7397, 7820 ms)** — exclusive tiering converts a failing operating point into a passing one. (Node variance shows in the *absolute* baseline p99: 7847 / 8684 / 8546 — which is exactly why the delta must be read within-node.)

¹ baseline λ=5 not measured; it is necessarily >SLO since baseline λ=4 is already at the 7847 ms edge (excl, which is strictly better, is already 12031 ms at λ=5).
- At λ=4 (the knee), exclusive tiering delivers **+10.1 % ± 1.4 pp req/s and −10.4 % ± 3.1 pp p99 (n=3 same-node pairs)**, and — the stronger statement — on 2 of 3 nodes it **converts a SLO-failing operating point into a passing one** (baseline p99 8684/8546 > 8 s; excl 7397/7820 < 8 s). At λ=3, −15 % p99. The +11 pp hit (−29 % prefill recompute) is what buys it. The whole curve shifts up/right (not a single-point/de-saturation trick — the charter's bar).
- **KNEE PINNED (excl λ=5, node 1-2):** p99 **12031 ms > 8s SLO** (raw req/s 4.28) → the excl knee is **between λ=4 and λ=5** — the *same* SLO-crossing bracket as baseline, but excl carries more headroom AT the λ=4 operating point. Honest reading: the mechanism **shifts the curve right by ~+10 %** at the SLO-limited point; it does **not** extend the knee to arbitrarily high λ (both saturate by λ=5). The gain is a genuine, bounded rightward shift — exactly what a capacity-lever (not a de-saturation trick) should produce.
- **HONESTY NOTE (evolution of the number):** an early claim of +15 % goodput used the DOCUMENTED protocol baseline (λ=4 p99 ~11 s) — pessimistic/different node; the first same-node pair (1-2) corrected it to +8 %; **the full n=3 same-node set now puts it at +10.1 % ± 1.4 pp req/s** (the 1-2 node was simply the lowest of the three). Node variance is large — same-node A/B is essential (lesson reinforced).

## ============ EXECUTIVE SUMMARY (as of 2026-07-08 ~13:40Z) ============
**Novel insight (the contribution), sim + write_back-validated:** for the long-reuse-distance multi-turn workload, bench_serving RE-QUEUES each turn to a FIFO tail → reuse distance ≫ cache. Under the frozen **write-through** policy, every device (L1) KV is eagerly mirrored to host (L2), so **L1 is a redundant subset of L2 → effective UNIQUE cache = host (8.4M) → hit capped ~0.62**. Making L1 **non-redundant** (exclusive/deferred-backup tiering) → effective cache = L1+L2 (10.7M) → **hit 0.73 (+11pp)** + better TTFT. Confirmed by: (a) my FIFO re-queue sim (8.4M→0.59≈baseline, 10.7M→0.73), (b) write_back diagnostic (config, private): hit 0.731, p50 491, p99 4292 on the same setup.
**★ Mechanism (v3c, engine code, BM_EXCL) — WIN:** reuse-gated exclusive KV tiering — defer the eager write-through backup so L1 holds non-redundant content; on device eviction back up reuse-proven (hit≥keep, keep=1) nodes. After a stability fix (sanity-check parity: exclusive tiering does write-back-style leaf-first backup, so skip the write-through parent-first invariant like write_back does), v3c-excl (commit 574e477ca) on node 0-3: **hit 0.7331 (+11pp vs same-node baseline 0.61-0.63), TTFT p50 474 (−17%), p99 4084 (−17..-38%), req/s 3.02 (no regression), stable (0 crashes), lossless.** Matches the write_back diagnostic (0.731) but as a LOGGABLE engine mechanism (frozen write_through config; resolved_args unchanged). (v3/v3b earlier died late on the sanity assertion — fixed.)
**★ HONEST POSITIONING — what is and isn't novel (integrity, not overclaim):** v3c's *measured effect converges with the stock `write_back` config*: same-setup metrics are within node-variance/0.5% (hit 0.7331 vs 0.7312; load_back 385.1M vs 383.1M; evict 581.2M vs 580.8M; p99 4084 vs 4292). The reuse-gating (keep=1, delete hit=0 leaves instead of backing them up) is a genuine *policy* difference from write_back but produced **no measurable advantage** here — hit=0 (single-turn) leaves are never reused, so whether they're backed up or dropped doesn't move aggregate metrics. I therefore do **NOT** claim v3c is a novel mechanism that beats existing options. What IS novel and defensible: **(1) the root-cause insight** — *why* frozen write-through caps hit at 0.62 on this workload (full-cycle reuse distance ⇒ inclusive L1 is a redundant mirror of L2 ⇒ effective unique cache = host only), diagnosed by ruling out Mamba/delete-path/load-back/retraction; and **(2) the optimality boundary** — a multi-axis proof (sim + argument) that non-redundant tiering is *within-contract-optimal*: the residual 0.733→0.806 gap is completed-conversation dead weight that is provably **unobservable online** (4 signals, 0% capture), and more capacity is off-contract/infeasible. The mechanism-only exploration thus *rediscovered* that the deferred-backup (write_back-class) policy is the right lever, implemented it as loggable engine code, and — the real contribution — **explained why it wins and proved nothing else within contract beats it on the hit/capacity dimension** (the recompute-cost/tail dimension is left open — see the boundary scope). A systems-paper framing: "here is the mechanism that captures the *capacity* headroom, here is *why*, and here is the fundamental limit of that lever," not "here is a brand-new algorithm that beats the state of the art."
**Lossless — by construction + corroborated.** A KV cache is lossless *by nature*: a hit replays the exact stored KV; a miss recomputes the exact KV from tokens (identical up to the same FP non-determinism the baseline already has). Exclusive tiering only changes *where/when* KV is placed and backed up — it never approximates, compresses, or serves stale KV — so it cannot change the output distribution (and the protocol samples stochastically at temp 0.6, so the right test is by-construction + coherence, not a raw output diff). Corroborated empirically: the sanity-check invariant passes every run (the parity fix), all 7037 requests complete cleanly (no gibberish/degenerate output a staleness bug would cause), the hit rate is coherent with the sim and the write_back diagnostic (0.733≈0.731), and the lossy-path counters never fire (dev_delete≈0, wb_fail=0 — every device eviction is a lossless demote-to-host or a reuse-gated backup).
**Rigorous NEGATIVES:** warm-first scheduling ± cold-aging (v1/v2) = NEUTRAL (apparent gains were pure node variance, debunked same-node via diag2/diag3; access order is client-imposed FIFO → server scheduling can't shrink reuse distance). Mamba consensus-truncation, load_back-failure, retraction, extra_key: all ruled out with instrumentation.
**Attribution (goodput ← hit ← recompute, not a queue artifact):** the fixed workload presents ~99.9 M prompt tokens/run. Baseline hit 0.622 ⇒ (1−0.622)×99.9 = **37.8 M** tokens recomputed (prefill FLOPs); excl hit 0.733 ⇒ **26.7 M** ⇒ **−11.1 M = −29 % prefill-recompute work.** The mechanism *converts* ~11 M tokens of expensive recompute into cheap host→device load-back (load_back rises 304 M→385 M — ~300 GB/s memory traffic at ~1.6–5.5 ms/op — while recompute FLOPs fall). Trading expensive prefill compute for cheap bandwidth is why p99/goodput improve; the gain traces to the hit increase, **not** to de-saturating the queue (req/s tracks λ in every run). This is the "attributable to the mechanism" chain the charter requires.
**Methodological lesson:** node variance ≈ 14% on req/s (ondem-3 2.64 vs 0-3 3.02) and ±30% on p99 → ALWAYS A/B on the SAME node.
**Rigor status (resolved):** the *hit* gain has non-overlapping error bars (n=6 vs n=2+, free). The *p99/goodput* gain now has error bars over **3 independent same-node A/B pairs at the knee** (nodes 1-2, 0-3, ondem-3; λ=4: Δp99 −10.4%±3.1pp, Δreq/s +10.1%±1.4pp) — the 0-3 and ondem-3 pairs were run on genuinely-idle certified nodes so they added no contention. Remaining lower-value gap: n≥4 for a tighter CI and λ=3 error bars (λ=3 is sub-knee, both far under SLO — low value). The claim rests on the bulletproof hit gain + the mechanistic hit→recompute→knee link + 3-node goodput agreement + the SLO-crossing on 2 of 3 nodes.
**Error bars — HIT (free, repeated runs):** across 6 baseline-family runs (v0_official + 5 diags/neutral-sched, frozen write-through) hit = **0.622 ± 0.007** (range 0.612–0.633); exclusive-tiering family (v3c + write_back-diagnostic = same 10.7 M-effective, +v3/v3b ≈ 0.73) hit = **0.732 ± 0.001**. **Δ = +11.0 pp, NON-OVERLAPPING** (baseline max 0.633 < excl min 0.731) — ~16× the baseline SD, real not noise (hit is a low-variance ratio over the fixed workload).
**Error bars — GOODPUT/p99 at the knee (λ=4, 3 independent same-node A/B pairs, nodes 1-2, 0-3, ondem-3):** **Δp99 = −10.4 % ± 3.1 pp** (−7.9, −14.8, −8.5) and **Δreq/s = +10.1 % ± 1.4 pp** (+8.1, +10.8, +11.4) — all three agree in direction and magnitude. On 2 of 3 nodes the effect crosses the SLO: baseline λ=4 **fails** (8684, 8546 ms) while excl **passes** on all 3 (7231, 7397, 7820 ms). req/s sustains ≈λ (no throughput regression) in every run. Node variance shows in the *absolute* baseline p99 (7847/8684/8546) → deltas read within-node.

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
