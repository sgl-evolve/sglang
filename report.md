# Research report — **kleinrock** (sglang KV-cache, v0.31)

**Researcher name:** `kleinrock` (Leonard Kleinrock — queueing theory; goodput@SLO under open-loop
Poisson load *is* a queueing problem). Branch `evolve/kleinrock`; W&B run `kleinrock` in `sgl-evolve`.
Clean base commit `a334877e5`.

## The bar (self-reminder)
Top-venue ONLY. Need a **genuinely novel primitive** (not eviction policy, not exclusive tiering, not
SRPF/SJF, not a config flip), a **non-obvious generalizing insight**, and **PC-proof evidence** (clean
curve shift, ablations, error bars, lossless, attributable to the mechanism not de-saturation). A
rigorous bounded **negative/impossibility** result also counts.

## The eval (v0.31, frozen)
Full-decode Poisson **rate sweep** λ∈{3,5,7,10}, WARMUP burst (300 convs) then sweep with **NO flush**
(warm steady-state), NUMP=1553 (~19M tok ≫ L1+L2 ~10.7M → real pressure, hit ~0.62), max-concurrency
256. **Headline = goodput@SLO = max req/s with p99 TTFT ≤ 8 s.** 2-tier (L1 GPU + L2 768GB host, no L3).
v0.31 FIXES the two v0.3-era eval flaws: under-pressure (now NUMP=1553) + cold-start metastability (now
warmup + warm steady-state).

Baseline.json (old single-point λ=3, cold): p99 TTFT 6326ms, p50 750ms, req/s 2.78, hit 0.6217,
host_util 0.9999, load_back_tokens 298M, evict_tokens 582M. host DRAM is SATURATED → genuine pressure.

## Landscape / prior art in this codebase (what NOT to redo)
- Stock scheduling policies present: LPM, DFS_WEIGHT, FCFS, LOF, RANDOM, ROUTING_KEY. (SRPF/WSAC/PGAC are
  NOT in my clean base — they were prior-campaign code in the dirty fork tree.)
- Known dead-ends/baselines (from the field & prior campaigns): eviction-policy tuning (LRU≈Belady),
  exclusive L1↔L2 tiering (~+13pp hit but config-equivalent = a *baseline*), admission gating (PGAC/WSAC
  reportedly catastrophic/neutral), recurrent/Mamba state (non-lever). Cache-residency positives on the
  official eval are largely closed.
- **The one charter-endorsed, un-exploited code gap:** transfer/compute overlap for **L2↔L1**. In stock
  sglang, host→device **load-back is SYNCHRONOUS + on-demand at admission** (scheduler forward path);
  **prefetch exists ONLY for the L3→L2 storage tier** (`prefetch_from_storage`), which is DISABLED in our
  2-tier config. So there is NO overlap of L2→L1 load-back with compute. `load_back_tokens`=298M at λ=3
  → a lot of data moves on-demand; under load this can convoy on the single load stream.

## Workload model (from the trace + loogle loader)
1553 docs → 1553 multiturn conversations. Each conv = **turn-0** ("Input: <full doc> Question: <Q0>",
cold, HEAVY-TAILED prefill: doc tokens p50 7.4K / p90 29K / **max 191K**) + **turns 1..N** (short Q_i,
warm — reuse the doc+prior-QA prefix, small new prefill; may need L2→L1 load-back of the doc if evicted
between turns). Total ~19.2M tok ≫ cache 10.7M. Hit ~0.62 = mostly INTRA-conversation doc reuse
(turns 1..N reuse turn-0's doc). Cross-rate re-runs are mostly COLD again (LRU over 1553 docs ≫ cache →
a conv's own doc is evicted by the time the next rate re-runs it) — so no-flush warmth helps intra-conv,
not turn-0 re-runs.

**Tail structure (the crux):** goodput@SLO = max λ with p99 TTFT ≤ 8s. p99 TTFT ≈ prefill time of a
~p99-length (≈29K-tok) doc **+ queueing delay**. Old cold λ=3 point: p99=6326ms ≈ a 29K-doc prefill → so
baseline goodput@SLO ≈ 3 (λ=3 passes), with headroom to push to 5/7 by cutting the **queueing-delay**
component of p99 at higher λ (the docs themselves don't get longer; the wait does). This is a
Pollaczek–Khinchine tail: heavy-tailed prefill service time S (docs) → large E[S²] → the tail is
queueing-delay-amplified variance, NOT mean hit rate.

## Hypothesis (leading, pending baseline data)
Under open-loop load, goodput@SLO is a **tail** (p99 TTFT) metric. The tail is set by (a) HoL blocking
from long cold prefills and (b) the **non-overlapped, on-demand L2→L1 load-back** for cache hits, which
convoys under load. A **queue-driven speculative load-back** — start the host→device transfer for
soon-to-be-admitted cache-hit requests *while they wait*, overlapped with running-batch compute, under a
memory-pressure-aware revocable budget — should hide transfer latency and shift the curve, losslessly.
**Go/no-go depends on whether load-back is a real tail cost under load** (measured in v0-stock sweep).
Fallback if load-back is negligible: rigorous characterization of what sets goodput@SLO (candidate
bounded-impossibility for lossless KV-movement).

---

## Versions
_(pending baseline sweep — job 19436 queued behind the active v0.3 campaign holding the certified pool)_

### v0-stock (config) — baseline rate sweep — QUEUED
Hypothesis/change: stock 2-tier config swept λ∈{3,5,7,10} to get the real throughput–latency curve +
goodput@SLO baseline + per-rate load-back/host-util diagnostics. This anchors all comparisons.

## Mechanism design (ready to implement, gated on baseline data)
**Primary — SLOP: Speculative Load-back Overlap under Pressure (queue-driven L2→L1 KV prefetch).**
Today load-back is layer-pipelined with the *admitting* iteration's forward (`LayerDoneCounter.wait_until`)
but STARTS only at admission — never overlapped with prior decode iterations, and the sync
alloc/evict sits on the admission path. SLOP:
- Hook `_add_request_to_queue`/`_prefetch_kvcache` (scheduler.py:2305/2280): on queue entry, run
  match_prefix; if host hit ≥ threshold AND device-prefetch budget available → start the H2D load-back NOW
  (overlaps with the running batch's decode compute).
- Admission (scheduler.py:2861 loop): mirror the L3 storage-prefetch skip pattern (line 2882) — if the
  speculative load-back is still in flight, `continue`; when done, admit with KV already resident (no sync
  load_back, no admission-path eviction).
- **Pressure-aware, revocable budget** (the hard part / novelty): prefetch only up to a reserved device-KV
  budget B so it NEVER evicts running-batch KV; revoke under pressure. Prefetch depth ≈ λ × load-back
  latency (Kleinrock: prefetch exactly the reqs admitted within the transfer window).
- Novelty vs prior art: the L2→L1 prefetch tier does not exist in stock (only L3→L2
  `prefetch_from_storage`); memory-pressure-aware revocable speculative load-back driven by the
  waiting-queue lookahead is new; ties queueing dynamics to the KV memory hierarchy. Distinct from
  Strata/Mooncake (L3/remote prefetch), LMCache/CacheGen (compression), AttentionStore (disk).
- **Go/no-go:** only if v0-stock shows load-back on the critical path under load (load_back_duration p99
  grows with λ; load-back tokens material).

**Fallback A (load-back negligible):** dual-class prefill-budget reservation — reserve a slice of the
per-iteration prefill token budget for short/warm (cached) requests so a long cold prefill can't HoL-block
them; SLO-feedback-tuned. NOT SRPF (bandwidth reservation, not global reorder).
**Fallback B (tail is irreducible cold prefill):** rigorous bounded-impossibility — goodput@SLO is set by
heavy-tailed cold-doc prefill service time; no lossless KV-movement mechanism shifts it (cache helps mean,
not the tail). Establish with the curve + tail decomposition + ablations.

## Formal submissions
_(none yet)_

## Ops notes
- Path bug: `eval.sh` computes WORK under `workspace/sgl/v0.3_ablations/research/...` (stale v0.3
  copy-paste) but setup clones under `workspace/sgl/v0.31/research/...`. Fixed WITHOUT editing frozen
  eval.sh via symlink `v0.3_ablations/research -> ../v0.31/research` (matches the already-present
  `base ->` symlink the manager/base-cell used).
- Compute contention: v0.3 research manager holds 3/4 certified nodes (0-3,1-2,ondem-2) with 24h
  sleep-infinity + runs evals into them; ondem-3 cycles v0.3 evals. v0.31 `_pool/held` is EMPTY → my
  cell's evals fall back to exclusive sbatch and QUEUE. Not fighting for resources unfairly; study while
  queued.
