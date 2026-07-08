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

## Versions
- **v1** (`7bac2815d`, `mechanism`): recompute-cost-aware eviction (CostAwareStrategy; threshold 4096).
  Status: committed+pushed; eval queued behind v0-diag on the shared pool. A/B vs v0-diag (same env).
