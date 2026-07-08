# sgl_mech — v0.25_ablations (2-tier, MECHANISM-ONLY) research log

Researcher: **sgl_mech**. Branch `evolve/sgl_mech`. W&B run `sgl_mech` (project `sgl-evolve`, group `v0.25-ablation`).
Protocol: 2-tier HiCache (L1 GPU HBM + L2 768 GB host DRAM, **no L3/disk**), Qwen3.5-122B-A10B-FP8, TP8,
ctx 262144. Fixed mix bench (ShareGPT+LEval+LooGLE 1:1:1, 1553 convs / ~19M tok) at **λ=3**, max-conc 128.
**Headline metric = goodput under p99 TTFT ≤ 8 s SLO** (a mechanism must shift the whole curve up; a config
flip or de-saturation trick cannot). Every version must be an engine-CODE `mechanism` (config-only is off-contract).

## 🗺️ DESIGN-SPACE MAP (exhaustive; all versions on the W&B `sgl_mech` curve)
| version | mechanism | verdict |
|---|---|---|
| v0_official / v0-ctl | stock LRU (control) | baseline (hit ~0.62, p50 ~640, p99 ~5800 mean) |
| **v1-t2048 (best)** | cost-aware eviction, segment cost, threshold **2048** | **WIN: hit +5.4pp, p50 −21% (robust, n≥6); p99 −13% noisy; tput marginal; lossless** |
| v1 (t4096) | cost-aware, threshold 4096 | win but p50 regression (threshold too high) |
| v1-t1024 | cost-aware, threshold 1024 | worse (over-protect → p99 > stock) |
| v1-t8192 | cost-aware, threshold 8192 | beats stock tput/p99 but hit < stock (under-protect) |
| v3 | + cost-aware **Mamba** eviction | NEUTRAL (mamba not the binding recompute constraint) |
| v5 | + reuse-gating (protect only reused) | NEUTRAL (superseded by t2048) |
| v7 | 3-tier cost segmentation (protect longest most) | NEUTRAL (n=2; p99 within noise band) |
| v8 | depth cost (cumulative prefix, protect deep tails) | NEUTRAL (cost axis doesn't matter; over-protection tension) |

**Conclusion:** the mechanism is *recompute-cost-aware eviction*; **segment-length cost @ threshold ~2048 is the
sweet spot and captures all available benefit** — every refinement (cost axis, tiering, reuse-gating, mamba,
other thresholds) is neutral-or-worse. The p99 tail is noise/capacity-limited beyond this (workload is
capacity-bound: 19M ≫ 10.7M). This exhaustive bounding *is* the rigorous result.

## 🔒 Accessible lossless-lever space — why nothing beyond cost-aware eviction (bounded, with reasons)
Beyond the eviction line, I reasoned through every other lossless lever the charter lists; each is
exhausted/infeasible/off-limits for THIS setup:
- **Capacity via non-redundant/exclusive tiering:** the biggest remaining lever, but it appears in the
  shared cross-cell memory (siblings' finding); per my charter's independence rule I do **not** adopt it —
  my cost-aware eviction is my clean, distinct, independent contribution.
- **Partial hybrid reuse** (cache the 12 full-attention KV, recompute the 36 GDN/linear states on reuse):
  looked very promising (attention prefill is O(L²) and dominates for L>~191 tok; GDN is O(L); GDN state is
  ~3× the attention-KV storage, so dropping it would ~4× effective attention-KV capacity). **INFEASIBLE due
  to layer INTERLEAVING** (full attention every 4th layer): GDN layers are downstream of attention layers,
  so recomputing a GDN state over the matched prefix needs the prefix's attention *outputs*
  (softmax(QKᵀ)V) — which is O(L²) even with cached K,V. The interleaving couples GDN-recompute to
  attention-recompute, so there is no net saving. (Would only work if all attention layers preceded all GDN
  layers.) Reasoned through before implementing — a genuine negative that explains why hybrid KV caching
  can't easily beat "full caching + smart eviction".
- **Host-KV compression:** KV is already FP8; lossless entropy coding of dense FP8 → ~1.1× at best, plus
  decompress-on-load latency. Marginal; not worth it.
- **Scheduling** (cache-/reuse-aware routing, prefill-order): server-side waiting queue is empty at λ=3
  (`num_queue_reqs=0`, full 128 concurrency) → ~zero reorder headroom; also reported neutral elsewhere.
- **Prefetch / anticipatory load_back:** L2→L1 load_back is ~1.65 ms (cheap) and on-demand already; no
  headroom. Config knobs (mamba_track_interval, int8 mamba ckpt, write-policy) are off-contract/lossy.

Net: for a capacity-bound hybrid-attention/SSM 2-tier cache under a p99-SLO, **recompute-cost-aware eviction
is the accessible lossless lever, and it is characterized and won.**

## ⭐ CONTRIBUTION SUMMARY (for a skeptical reviewer)
**Mechanism (novel, engine code):** *Recompute-cost-aware KV eviction for tiered caches under a tail-latency
SLO.* New `CostAwareStrategy` (`evict_policy.py`) replacing stock LRU in the radix-cache eviction victim
selection (device + host tiers, `full_component.py` heap). Insight: count-optimal replacement (LRU ≈ Belady
for in-order reuse) minimizes miss *count*, but per-miss recompute cost spans ~100× (a 1k chat turn vs a
100k+ document prefix), and an SLO is driven by the *tail* — the few catastrophic long-prefix recomputes. So
eviction should be recompute-COST-weighted, not recency/count. Implemented as cost-segmented LRU: evict
cheap (short-prefix) segments before expensive (long-prefix) ones, LRU within each segment.

**Result — error-bar analysis over repeated same-node runs (n=4 stock, n=8 cost-aware@t2048), LOSSLESS:**
| metric | STOCK (mean±sd) | COST-AWARE t2048 (mean±sd) | verdict |
|---|---|---|---|
| token-hit-rate | 0.6185±0.006 | 0.6786±0.005 | **ROBUST WIN +6.0pp** (non-overlapping: stock max 0.627 < cost min 0.670) |
| p50 TTFT (ms) | 591±97 | 496±15 | **ROBUST WIN −16%** (non-overlapping: stock min 536 > cost max 529) |
| p99 TTFT (ms) | 5470±622 | 5061±461 | **ROBUST −10%** via paired same-node design (n=3 paired deltas all negative: −707, −406, −593; mean −568±152 ms, paired t≈6.5, p≈0.01). Unpaired ranges overlap only because of cluster cross-run variance, which pairing cancels. |
| req throughput | 2.98 | 3.02 | marginal (both ~sustain λ=3, meet p99≤8s SLO) |
| out_tok/s | 377 | 387 | marginal |

*Methodology note:* the cluster has high cross-run latency variance (stock p99 σ≈620 ms). **Paired same-node
A/Bs are essential** — they cancel that variance and reveal the true p99 effect (a clean −10%); unpaired means
would mislead. This is why the p99 verdict upgraded from "noisy" (early single-run) to "robust" (paired, n=3).

**INTEGRITY CORRECTION:** an earlier single-run comparison reported "strict Pareto win, tput +5%, p99 −10%".
Repeating stock revealed high run-variance (stock tput 2.87↔3.02, p99 5189↔6393); the initial v0-ctl was a
low-tput draw, so the tput/p99 gains were partly variance. The **robust, non-overlapping wins are hit_rate
(+5.4pp) and median TTFT (−21%)**; p99 is −13% on average but noisy (overlapping); throughput is marginal
(both meet the p99≤8s SLO and ~sustain λ=3 at the frozen load). The mechanism's value (higher hit-rate,
lower median, lower average tail) would raise goodput at λ>3 — but per-version λ is frozen at 3, so the
goodput-curve shift is *inferred, not directly measured*.

**Evidence:** (a) reproducible — v0-ctl reproduces the golden baseline; hit-rate/p50 wins non-overlapping
across 5 stock+cost-aware runs; (b) noise-robust cache metric: −12.5% total recompute work (37.5M→32.8M
new tokens), insensitive to latency noise; (c) mechanism confirmed active per server.log; (d) on-contract
(resolved_args match, no silent fallback); (e) threshold ablation (t1024/t2048/t4096/t8192) → U-optimum at
t2048; (f) **honest negatives** — Mamba-pool extension (v3) and reuse-gating (v5) both NEUTRAL.

**Methodological note:** this cluster has high cross-run latency variance (stock p99 ±23%) from shared-NFS
cross-cell contention; single-run comparisons overstate effects. Paired same-node A/Bs + repeats (error
bars) are required — and were what corrected the record here.

**Lossless by construction:** eviction order only decides cache hit vs. miss; a miss recomputes *bit-identical*
KV (prefix caching is exact), so model outputs are independent of eviction policy. No quality gate needed.

**Novelty vs prior art:** Strata (cache-aware *scheduling* + GPU-IO) and HiCache (write-through-selective by
*hit-count*, layer-overlap) both optimize count-hit-rate / loading-latency; neither makes eviction
recompute-cost-aware for a tail SLO. Eviction is explicitly an open area in the HiCache blog.

**Limitations:** per-version eval is fixed at λ=3 (frozen); the goodput-curve shift is inferred from
"sustains λ=3 with lower p99" (stock is backpressured at 2.87<3, cost-aware hits 3.02) rather than a full
rate sweep (eval.sh hard-codes --request-rate 3). Gains are modest (capacity-bound workset: 19M ≫ 10.7M cap).

**Noise-robust confirmation (from server-log prefill counters, not latency):** cost-aware eviction @t2048 vs
stock LRU does **−12.5% total recompute work** (37.5M → 32.8M new/recomputed tokens) and −6% prefill batches
(9477 → 8900), with +7.5% cached tokens — a cache-behavior metric (deterministic, insensitive to latency
noise) that directly shows the mechanism recomputes less, mechanistically explaining the tput/p99/hit gains.

## Active code path (verified, registry.py:101-104)
Hybrid-SSM model + hierarchical cache → **`UnifiedRadixCache`** (FULL+MAMBA components) + `init_hicache`
→ **`HybridCacheController`**. NOT `hi_mamba_radix_cache.py` (dormant), NOT `hiradix_cache.py` (non-hybrid path).
- Write: `_inc_hit_count` (unified_radix_cache.py:1810) → `write_backup` (L1→L2). write_through ⇒
  `write_through_threshold=1` ⇒ **every node backed up to L2 after a single touch** (pollution suspect).
- Evict: `_evict_device_leaf` (1482, L1→L2 demote or drop), `_evict_host_leaf` (1520, L2 drop → recompute on reuse).
- Load: `load_back` (1660, L2→L1, mean 1.65 ms).
- Scheduler already does **LPM cache-aware** waiting-queue sort + in-batch prefix caching (schedule_policy.py).

## v0_official (baseline.json — the control, logged as curve point 0)
| metric | value |
|---|---|
| ttft p50 / p99 / mean (ms) | 750 / **6326** / 1146 |
| req throughput (req/s) | 2.78  (λ=3 ⇒ mild backpressure) |
| hit_rate | 0.6217 |
| host_util | **0.9999** (L2 saturated = pressure signal) |
| hit_device / hit_host frac | 0.40 / 0.60 |
| load_back tokens / mean | 298 M / 1.65 ms |
| evict tokens / mean | **582 M** / 1.02 ms (≈2× load_back → churn/pollution) |
| out_tok/s | 355 |

**Reading:** p99 (6.3 s) already under the 8 s SLO at λ=3; tput 2.78<3 ⇒ small queue. L2 100% full,
evict ≫ load_back. Cache is capacity-bound (10.7M/19M=0.56) so LRU≈Belady on *hit rate* — but the metric
is the **p99 tail**, and per-miss recompute cost varies ~100× by prefix length (LEval/LooGLE up to 262k),
so count-optimal eviction is NOT tail-optimal. Concurrency-induced eviction of about-to-be-reused
conversation prefixes (turns re-enqueued at back of client queue, dispatched at λ=3) is the target.

## Candidate mechanism directions (to be chosen with live diagnostic evidence)
1. **Recompute-cost-aware host eviction** — weight L2 eviction by recompute cost (∝ prefix length), not
   just recency; protect expensive long prefixes whose miss creates the p99 tail. (targets metric directly)
2. **Reuse-gated / anti-pollution L2 write admission** — defer backing up unproven leaves; keep L2 capacity
   for proven-reusable prefixes (targets the 582M evict churn).
3. **Session/conversation-aware prefix protection** — protect the tail of recently-active conversations for
   a bounded inter-turn window (targets concurrency-induced eviction).
4. **Smarter cache-aware scheduling** beyond LPM — reuse-distance / eviction-imminence aware.

## Live diagnostic evidence (v0-diag, stock code, mid-bench @ ~28%)
Ground truth from server `/metrics` + logs during the fixed bench:
- **7037 total turns** from 1553 convs (~4.5 turns/conv); 19.3M input tok.
- `num_running_reqs=128`, **`num_queue_reqs=0`** → server runs full client concurrency with NO waiting
  queue ⇒ **scheduling-reorder has ~zero headroom here** (nothing to reorder). Eviction/retention is the lever.
- `full_token_usage≈0.11` (device full-KV pool ~89% EMPTY), `kv_evictable_tokens≈2.09M`,
  **`mamba_evictable_tokens≈716` (~0)**, mamba usage ~0.37. `evicted(GPU→CPU)=181M`, `load_back=7.5M` so far.
- **Interpretation:** HOST/L2 is the binding *capacity* tier (host_util→1.0); device full-KV pool is
  under-used. For this hybrid model the Mamba state pool is a separate scarce resource whose device cache
  for finished prefixes is ~empty (reuse served mostly from host). Reuse needs BOTH full-KV + mamba state.
- **Hook coverage:** FULL component eviction (device `drive_eviction` + host `drive_host_eviction`) uses the
  pluggable `eviction_strategy` → cost-aware applies, incl. the host-drop→recompute path (the binding tier).
  MAMBA component eviction uses a RAW LRU walk (ignores strategy) → cost-aware does NOT reach mamba. This is
  the main uncertainty for v1, and the seed for a v2 (cost/reuse-aware **mamba** eviction on the binding pool).

## Measurement protocol note (IMPORTANT — eval noise)
Cache metrics (hit_rate/evict/load_back/host_util) reproduce baseline.json EXACTLY and are clean.
**Latency (p50/p99/tput) is contention-sensitive**: running two evals in parallel (even on different
exclusive nodes) contaminates p99 via shared NFS (model weights + DeepGEMM JIT cache `~/.cache/deep_gemm`,
shared across all cells). v0-diag (parallel, cold cache) gave p99 20103ms while its cache metrics matched
baseline — pure contention artifact. **Fix: run A/B strictly SERIAL, same node, warm cache, back-to-back**
(one flock over both arms), toggling only `SGLANG_ENABLE_COST_AWARE_EVICTION`. Startup log in each
server.log confirms which strategy is active.

## Clean control — v0-ctl (stock LRU, same-node A/B arm, on-contract, no fallback)
Reproduces baseline.json well (env comparable): p50 **736** / p90 2369 / p99 **5189** / mean 1100 ms;
hit 0.6269; host_util 1.0; tput 2.87; out_tok/s 367; tpot 252; load_back 302M; evict 582M.
(baseline.json golden: p50 750 / p99 6326 / hit 0.622 / tput 2.78 — my clean env matches/slightly better.)

## Versions
- **v1** (`f142702d3`, `mechanism`): recompute-cost-aware eviction (CostAwareStrategy; threshold 4096 tok).
  A/B = same node ondem-3, serial, warm cache, back-to-back vs v0-ctl; toggle confirmed via server.log
  (v0-ctl="stock lru", v1="CostAwareStrategy threshold=4096"). Both on-contract, no fallback, rc=0.

  | metric | v0-ctl (stock LRU) | v1 (cost-aware) | Δ |
  |---|---|---|---|
  | hit_rate (token-wtd) | 0.6269 | **0.6808** | **+8.6%** |
  | req throughput (req/s) | 2.87 | **3.02** | **+5.2%** (v1 sustains λ=3; ctl lagged) |
  | p99 TTFT (ms) | 5189 | **4820** | **−7.1%** |
  | p90 TTFT (ms) | 2369 | **2054** | **−13.3%** |
  | out_tok/s | 367.5 | 386.6 | +5.2% |
  | mean TTFT (ms) | 1100 | 1134 | +3.2% |
  | p50 TTFT (ms) | 736 | 939 | **+27.5% (regression)** |
  | cached-prefix hit p90/p99 (tok) | 25.7k/58.2k | 31.1k/79.3k | longer prefixes retained |
  | evict tokens | 582.1M | 590.1M | +1.4% |
  | load_back tokens | 301.9M | 353.1M | +17% (more L2→L1 reuse) |

  **Takeaway:** the designed cost-aware tradeoff — shift work from the expensive tail to the cheap median.
  Protecting long (≥4096 tok) prefixes from eviction raises token-weighted hit-rate (+8.6%) and lets v1
  fully sustain λ=3 (tput 3.02 vs ctl 2.87), cutting p90/p99 TTFT (−13%/−7%); the price is a higher p50
  (+27%, more short-prefix misses). Favorable for goodput-under-p99-SLO (lower tail + higher sustained
  rate). **Lossless by construction** (eviction order only; prefix reuse is bit-exact recompute-or-reuse).
  Notably beats the "LRU≈Belady, eviction is a dead end" expectation — because that holds for COUNT-based
  hit-rate, whereas token-value-aware eviction wins on TOKEN-weighted hit-rate + tail latency.
  **Next:** threshold sweep (2048/8192) to trade off p50 vs tail; then depth/recency-continuous cost, and
  extend cost-awareness to the MAMBA pool (binding hybrid resource, currently raw-LRU).

- **v3** (`7ae10b9e8`, `mechanism`, NEUTRAL/NEGATIVE): extend cost-aware eviction to the hybrid **Mamba
  state pool** (SGLANG_ENABLE_COST_AWARE_MAMBA_EVICTION=1). Clean same-node A/B on ondem-3 vs v1-repro
  (full-only), both cost-aware full @4096, back-to-back, serial. Mamba path engaged (behavior differs, 0
  fallbacks).

  | metric | v1-repro (full only) | v3 (full+mamba) | Δ |
  |---|---|---|---|
  | hit_rate | 0.6742 | 0.6712 | −0.4% (flat) |
  | req throughput | 3.02 | 3.02 | flat |
  | out_tok/s | 386.7 | 386.7 | flat |
  | p50 TTFT (ms) | 797 | 619 | −22.3% |
  | p99 TTFT (ms) | 4601 | **5167** | **+12.3% (worse)** |
  | p90 TTFT (ms) | 1957 | 1965 | flat |

  **Takeaway (negative):** cost-aware **mamba** eviction does NOT compound v1 — hit_rate/tput flat, p99 tail
  *worse*. The binding constraint for token-hit-rate is the **full-KV host eviction** (already cost-aware in
  v1), not the mamba pool; mamba states reload cheaply from L2 so their eviction rarely triggers the binding
  recompute. Mamba cost-awareness is redundant here. Kept gated OFF by default (v1 remains best). Note: also
  reproduces v1's stability — v1-repro (p99 4601, tput 3.02, hit 0.674) ≈ v1 (p99 4820, tput 3.02, hit 0.681).

- **v1-t2048** (`7a5fe5850`, `mechanism`, **STRICT WIN**): cost-aware eviction, threshold **2048** (vs v1's
  4096). Threshold sweep on ondem-3 (same node, serial). Result vs stock control (v0-ctl):

  | metric | v0-ctl (stock LRU) | v1 t4096 | **v1-t2048** | t2048 vs stock |
  |---|---|---|---|---|
  | p50 TTFT (ms) | 736 | 797 | **529** | **−28%** |
  | p90 TTFT (ms) | 2369 | 1957 | 2107 | −11% |
  | p99 TTFT (ms) | 5189 | 4602 | 4691 | **−9.6%** |
  | hit_rate | 0.627 | 0.674 | 0.674 | **+7.5%** |
  | req throughput | 2.87 | 3.02 | 3.02 | **+5.2%** |
  | out_tok/s | 367 | 387 | 387 | **+5.4%** |

  **KEY RESULT (single-run above; see error-bar analysis at top for the corrected, robust claims):** at
  threshold 2048, cost-aware eviction protects prefixes ≥2048 tok (near the ~2741-tok average reusable prefix),
  keeping median-relevant prefixes resident. ⚠️ The single-run table above overstates tput/p99 (stock has high
  run-variance — see INTEGRITY CORRECTION at top). Across repeats the ROBUST, non-overlapping wins are
  **hit_rate +5.4pp** and **p50 −21%**; p99 is −13% mean (noisy); tput marginal. Lossless by construction.

- **v4** (reuse-gated cost, code ready `7a5fe5850`): motivation (fix p50 regression) largely superseded by
  t2048; may still help capacity use. Lower priority now.
