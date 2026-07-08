# sgl_mech — v0.25_ablations (2-tier, MECHANISM-ONLY) research log

Researcher: **sgl_mech**. Branch `evolve/sgl_mech`. W&B run `sgl_mech` (project `sgl-evolve`, group `v0.25-ablation`).
Protocol: 2-tier HiCache (L1 GPU HBM + L2 768 GB host DRAM, **no L3/disk**), Qwen3.5-122B-A10B-FP8, TP8,
ctx 262144. Fixed mix bench (ShareGPT+LEval+LooGLE 1:1:1, 1553 convs / ~19M tok) at **λ=3**, max-conc 128.
**Headline metric = goodput under p99 TTFT ≤ 8 s SLO** (a mechanism must shift the whole curve up; a config
flip or de-saturation trick cannot). Every version must be an engine-CODE `mechanism` (config-only is off-contract).

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

- **v4** (planned): **reuse-gated cost-aware eviction** — protect a long prefix only when it is BOTH
  expensive (≥ thr tokens) AND has proven reuse (hit_count ≥ 1), so one-shot long prefixes aren't protected
  at the expense of reused short ones. Aims to keep v1's tail/hit/tput gains while cutting the p50 regression.
