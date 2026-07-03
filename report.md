# drift-3e7 — sglang HiCache KV-cache research log

- **Researcher:** `drift-3e7` (independent replicate; own clone, own branch, own W&B run)
- **Branch:** `evolve/drift-3e7`
- **W&B:** project `sgl-evolve`, run `drift-3e7`
- **Workspace:** `workspace/sgl/researchers/drift-3e7/`
- **Eval:** FIXED protocol via `eval-on-pool.sh` on a certified held node. Never modified.

## Baselines (logged, not re-run)

| version | tag | ttft_mean_ms | ttft_median_ms | ttft_p90_ms | ttft_p99_ms | hit_rate | l3_hit_frac | out_tok_s | req_thpt |
|---|---|---|---|---|---|---|---|---|---|
| v0_official | baseline | 87615.4 | 1224.5 | 243466 | 270798 | 0.8163 | 0.2538 | 146.9 | 1.15 |
| v0_tuned | baseline | 108824.3 | 1437.9 | 295413 | 322801 | 0.8207 | 0.2585 | 119.3 | 0.93 |

**Note / anomaly:** `v0_tuned` (the nominal "bar to beat") has *worse* mean TTFT than
`v0_official`, despite identical `resolved_args`. In this overloaded regime a single run's
mean TTFT is dominated by the queue-wait tail and is high-variance. I therefore target the
**harder** bar — `v0_official`'s **87615 ms** mean TTFT. Beating it beats both.

## Bottleneck analysis (from the baselines)

The workload is **fundamentally overloaded**: arrivals λ=3.5 req/s vs measured
`req_throughput` ≈ 1.15 req/s. So the queue grows during the run and **mean TTFT is
dominated by queue wait** — median TTFT is only ~1.2 s but the mean is ~87 s (p90 ≈ 243 s).

Mechanism of the tail (from reading the code):
- Frozen prefetch policy is `wait_complete` (`can_terminate = completed`,
  `hiradix_cache.py:1355`): a request cannot be scheduled until its **entire** SSD-resident
  prefix finishes loading host-side. `get_new_batch_prefill` *skips* (does not block on) such
  requests (`scheduler.py:2882`), so they sit in the waiting queue.
- Under a saturated SSD (l3_hit_frac ≈ 0.25, disk_read ≈ 20.7 M tokens, prefetch ≈ 166 M
  tokens), prefetches serialize → many requests stay un-schedulable → the **decode batch
  starves → throughput collapses → the queue and the TTFT tail explode.**

**Implication:** the highest-leverage lossless wins reduce time spent blocked on / bandwidth
spent by SSD prefetch, and keep hot prefixes off the SSD tier. Two attack surfaces:
1. **Cache quality** — keep hot prefixes in host RAM so fewer requests fall to SSD (v1).
2. **Prefetch path** — don't block on / don't redundantly read a saturated SSD (v2+).

---

## v1 — SLRU host eviction (mechanism) — RETRACTED (dead code, never logged)

- **Hypothesis:** host-tier eviction is destructive; replacing pure-LRU host eviction with
  SLRU (protect `hit_count ≥ 2` prefixes) should keep hot multi-turn prefixes in host RAM,
  cutting SSD reads → lower TTFT tail.
- **What killed it:** I edited `mem_cache/hiradix_cache.py` (`HiRadixCache`). But this model
  (Qwen3.5-122B-A10B) is **hybrid-SSM/GDN**, and `mem_cache/registry.py:101-104` routes
  `enable_hierarchical_cache + is_hybrid_ssm` to **`UnifiedRadixCache`** (confirmed in the
  running server: "Allocating … hierarchical Mamba cache", `radix_eviction_policy='lru'`,
  `schedule_policy='fcfs'`). So my edit was **dead code**. I killed the in-flight eval before
  it wasted the node (it would have just reproduced baseline). No curve point.
- **Also found (would have crashed if it HAD been live):** the `evict_host` parent re-push
  mixed a float LRU priority into an SLRU tuple heap → `TypeError` under host pressure. Fixed
  in `7605b7e52`, but moot since the file is unused for this model.
- **Lesson:** verify the actually-instantiated class from the running server before coding.

## v2cfg-besteffort — best_effort prefetch policy (config) — RUNNING

- **Hypothesis / purpose:** bottleneck probe. `--hicache-storage-prefetch-policy best_effort`
  never blocks scheduling on SSD (schedule immediately, recompute the un-loaded tail). If it
  sharply cuts mean TTFT, SSD-prefetch blocking is confirmed as the dominant tail.
- **Change:** config only (valid — the flag reaches `UnifiedRadixCache.prefetch_stop_policy`).
- **Result:** _pending (~1 h)._  **Lossless:** recompute yields identical tokens.

## v3cfg-timeout — timeout prefetch policy (config) — DEFERRED (no free node)

- Third point on the prefetch-policy spectrum. Default timeout deadline is
  `1.0 + pages×0.25 s`, so it's a *muted* probe for the tail (huge-prefix reads still wait
  tens–hundreds of s), and the timeout knobs live in the FROZEN `--hicache-...-extra-config`
  so I can't sharpen it via config. Low priority; run if a node is idle.

## v4-adaptive-prefetch — load-adaptive prefetch stop policy (MECHANISM) — READY (commit `8e15590dc`)

- **Hypothesis:** the win is to keep `wait_complete`'s full-prefetch benefit when the SSD tier
  has headroom, but *give up early and recompute* (like best_effort) exactly when the SSD
  backlog is saturating — so a saturated SSD can't starve the decode batch and blow up the tail.
- **Change (`mem_cache/unified_radix_cache.py`, `server_args.py`):** new
  `--hicache-storage-prefetch-policy adaptive`. In `can_terminate_prefetch`, deadline =
  `base + pages×per_page × (1 − pressure)`, `pressure = prefetch_tokens_occupied /
  prefetch_capacity_limit ∈ [0,1]`. pressure→0 ⇒ the timeout-policy deadline (wait for the
  whole prefetch); pressure→1 ⇒ flat `base` (~1 s) regardless of prefix size ⇒ give up, admit,
  recompute. Self-regulates via negative feedback around the capacity limit.
- **Why robust:** under light load it *is* the wait/timeout behavior (safe); it only deviates
  under saturation. Unit-tested the deadline math offline.
- **Plan:** launch the moment a certified node frees; compare vs `v0_official` (wait_complete)
  and `v2cfg-besteffort` (the no-wait ceiling). Tag `mechanism`.
- **Lossless:** recompute yields identical tokens.
