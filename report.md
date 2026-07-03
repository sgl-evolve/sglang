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

## v1 — SLRU host eviction (mechanism) — RUNNING

- **Hypothesis:** host-tier eviction is destructive (a fully-evicted node is removed from the
  tree; its prefix must be re-fetched from SSD next time). Default host eviction is pure LRU.
  Using **SLRU** (segmented LRU: nodes with `hit_count ≥ 2` are "protected" and evicted only
  after all probationary nodes) should preserve frequently-reused multi-turn prefixes in host
  RAM, cutting SSD reads → shorter prefetch stalls → lower TTFT tail.
- **Change (`mem_cache/hiradix_cache.py`):** added `self.host_eviction_strategy =
  SLRUStrategy(protected_threshold=2)`; `evict_host()` now scores leaves (and parent re-pushes)
  with it instead of the device LRU strategy. Device (GPU) eviction unchanged.
- **Bug found & fixed pre-eval:** the parent re-push in `evict_host` still used the device
  (LRU→float) strategy while the initial heap used SLRU→tuple; mixing float and tuple
  priorities in one heap raises `TypeError` under host pressure (host_util ≈ 1.0 here → would
  crash). Fixed to use `host_eviction_strategy` in both places. Commit `7605b7e52`.
- **Result:** _pending._
- **Lossless:** eviction-policy-only; does not change computed outputs.

## v2cfg-besteffort — best_effort prefetch policy (config) — RUNNING

- **Hypothesis / purpose:** bottleneck probe. If switching `--hicache-storage-prefetch-policy`
  from `wait_complete` to `best_effort` (schedule immediately, recompute the not-yet-loaded
  tail instead of waiting on SSD) sharply cuts mean TTFT, that confirms SSD-prefetch blocking
  is the dominant tail and points v3 at an adaptive/dedup prefetch mechanism.
- **Change:** config only (extra eval flag); runs on the v1 code (host-eviction effect is
  separable from the prefetch-policy effect).
- **Result:** _pending._
- **Lossless:** recompute yields identical tokens.
