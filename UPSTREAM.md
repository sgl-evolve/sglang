# Exclusive L1↔L2 tiering for HiCache — upstream summary

**One-liner:** Make HiCache's GPU (L1) and host (L2) tiers hold **disjoint** cached KV instead of the
default **inclusive** behavior (where `write_through` eagerly duplicates every hot device entry onto host).
This reclaims the GPU tier as *distinct* cache capacity → higher prefix-cache hit-rate under memory
pressure, lossless.

## Problem
Under KV memory pressure (long-context multiturn; working set ≫ L1+L2), the stock `write_through` HiCache
policy is **inclusive**: on reuse, `UnifiedRadixCache._inc_hit_count` eagerly calls `write_backup` (D→H)
and KEEPS the device copy. So every hot device entry is duplicated on host ⇒ the *distinct* cached set ≈
HOST capacity alone; the GPU tier is a redundant subset (mirrors host), not extra capacity. Measured on
Qwen3.5-122B-A10B (hybrid-Mamba), TP8, 2-tier (GPU ~2.35M tok + host 7.81M tok): baseline prefix hit-rate
0.62; the ~2.35M-token GPU tier is largely wasted on duplicates.

## Mechanism (engine code; env-gated, default off)
`SGLANG_HICACHE_EXCLUSIVE=1` makes an entry live on device **XOR** host, never both:
1. **No eager backup** — `_inc_hit_count` returns early (like write_back): an entry stays device-only until
   device eviction (`unified_radix_cache.py`).
2. **Back up on eviction, not eagerly** — `_evict_device_leaf` writes D→H then demotes (write_back path).
3. **Free host on promotion** — new: `_promote_free_host` in `loading_check` frees the host copy after an
   H→D load-back completes, so a re-warmed entry is device-exclusive again.
Two consensus call-sites (`evict()`'s batched `writing_check`, and the `sanity_check` validator) also treat
exclusive like write_back. A frequency-aware hybrid knob (`SGLANG_HICACHE_EXCLUSIVE_HOT_KEEP`, keep hot
nodes inclusive) was tried and is **neutral** (transfer cost isn't the limiter) — leave at 0.
- Files: `python/sglang/srt/environ.py` (+11), `python/sglang/srt/mem_cache/unified_radix_cache.py` (+68).
- Commits: `21023af6d` (free-host-on-promotion), `5496b0716` (self-contained), `feca1871e` (consensus
  call-sites), `91d611a10` (hybrid knob, optional).

## Evaluation (fixed protocol: real 1:1:1 mix, 1553 convs, λ=3, max-conc 128; SLO p99 TTFT ≤ 8s)
- **Prefix hit-rate: 0.62 → 0.75 (+13pp), ROBUST/node-independent** — exclusive = 0.7522 on 2 distinct
  nodes and 0.7509±0.002 across n=4 runs. This ≈ 13% less fresh-prefill compute.
- **Latency/goodput (grow with load; node-dependent):** mean TTFT −14…−22%; p99 TTFT lowered AND stabilized
  (~4.4–4.5s vs baseline high+variable 4.7–6.9s); on a prefill-stressed node the p99≤8s **goodput knee
  shifts ~+14–18% sustainable req/s**. The magnitude scales with how prefill-stressed the operating point is.
- **Lossless: MEASURED bit-exact.** Greedy 3-mode verify (24 long docs, fresh vs cache-hit exercising the
  exclusive host-free/load-back path): exclusive == stock outputs **24/24** on both paths. (Note: the
  hybrid-Mamba radix cache is inherently ~non-bit-exact vs *no-cache* on ~17% of long docs due to Mamba
  checkpoint reconstruction — but stock shows the IDENTICAL pattern, so this is an sglang cache property,
  not the exclusive mechanism; the baseline bears it equally.)

## When to use it (generalization)
Benefit scales monotonically with the **device/host cache ratio** (the fraction of total cache in the fast
tier): larger GPU-cache-to-host-cache ratio ⇒ more inclusive-duplication reclaimed ⇒ bigger win; never
negative. Largest under memory pressure + high load (near the SLO knee).

## Limits / not pursued
- Residual headroom to the analytic hit ceiling (~0.81) requires MORE capacity, reachable only by *lossy*
  KV quantization (`--kv-cache-dtype fp8_e4m3` doubles capacity → hit 0.808 but changes outputs) — a config
  flag, out of scope for a lossless mechanism. Lossless KV compression is dominated by it (only ~1.4× on
  bf16 → +2–3pp, and needs a new variable-size host allocator) → not worth building.
- Ordering (`--schedule-policy lpm`), admission/concurrency caps, eviction-order tuning, and
  scheduling-based "co-residency" were all tried and do NOT recover hit-rate here (capacity-bound;
  LRU≈Belady) — co-residency is served by *placement* (this mechanism), not scheduling.
