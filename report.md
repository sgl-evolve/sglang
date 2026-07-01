# drift-9e4 — sglang KV-cache evolution report

Branch: `evolve/drift-9e4`  
W&B: project `sgl-evolve`, run `drift-9e4`  
Eval protocol: Qwen3.5-397B-A17B-FP8, TP=8, ctx=262144, mem_frac=0.85, hicache_size=64, file backend on localssd, LooGLE λ=3.5, ShareGPT 60 clients

---

## V0 — Baseline (unmodified sglang)

**Commit:** baseline (unmodified)  
**Hypothesis:** Establish reference metrics.

| Metric | LooGLE | ShareGPT |
|--------|--------|----------|
| TTFT mean (ms) | 40370.89 | 35660 |
| TTFT p99 (ms) | 147140.57 | — |
| Hit rate | 0.7676 | 0.607 |
| Output tok/s | 33.83 | — |
| Req throughput | — | 1.36 |

HiCache: disk_read_tokens=1.74M, load_back_mean=1.727ms, host_util=0.9996  
Evicted ~340M tokens, loaded back ~262M, backed up ~78M.

**Takeaway:** Host memory at 99.96% utilization → constant eviction churn. Working set (4.56M tokens) exceeds host capacity (4.17M) by ~9%. TTFT dominated by queueing due to GPU pool pressure (~686K tokens).

---

## V1 — SLRU eviction + amortized host eviction (NEGATIVE)

**Commit:** `5fdbbad48`  
**Flag:** `--radix-eviction-policy slru`

**Hypothesis:** LRU eviction thrashes shared prefixes under high reuse. SLRU shields frequently-accessed nodes. Amortized eviction (evict more than minimum per call) reduces heap rebuild overhead.

**What changed:**
- `hiradix_cache.py:evict_host()` — target = `num_tokens + max(num_tokens, page_size * 64)` (evict 65x more than needed)
- `--radix-eviction-policy slru` — SLRUStrategy instead of LRU

**Result (vs V0):**

| Metric | V0 | V1 | Delta |
|--------|----|----|-------|
| LooGLE TTFT mean (ms) | 40371 | 46528 | **+15.3% worse** |
| LooGLE TTFT p99 (ms) | 147141 | 161809 | +10.0% worse |
| LooGLE out tok/s | 33.83 | 31.12 | -8.0% worse |
| LooGLE hit rate | 0.7676 | 0.8235 | +7.3% better |
| ShareGPT TTFT mean (ms) | 35660 | 32950 | -7.6% better |
| ShareGPT req throughput | 1.36 | 1.35 | -0.7% |
| ShareGPT hit rate | 0.607 | 0.616 | +1.5% better |
| Disk read tokens | 1.74M | 4.56M | **+162% worse** |
| Load back mean (ms) | 1.727 | 1.631 | -5.6% better |

**Root cause:** Amortized eviction (65x over-eviction) pushed 2.8x more data from host to disk. Despite hit rate improving (SLRU protected shared prefixes), disk reads dominated — each disk read is orders of magnitude slower than host memory. LooGLE suffered because its working set relies heavily on host-tier KV; flushing that to disk destroyed the fast-path.

**Takeaway:** Host memory is precious — evict only what's needed. SLRU improved hit rates but was overwhelmed by the amortized eviction damage. Reverted the amortized eviction.

---

## V2 — Write-stream yields PCIe to load-stream + adaptive write threshold

**Commit:** `bd5b84e1c` (V1 amortized eviction reverted; V2 write-yield + V3 adaptive threshold retained)  
**Flags:** default LRU (no SLRU)

**Hypothesis:** Two changes, both reducing unnecessary host-tier pressure:
1. **Write-yield** (`cache_controller.py:start_writing()`): Write-through DMA and load-back DMA share PCIe. When both run concurrently, loads (TTFT-critical) get only half bandwidth. Adding a CUDA sync event makes writes wait for pending loads, giving loads full PCIe bandwidth.
2. **Adaptive threshold** (`hiradix_cache.py:_inc_hit_count()`): When host is >95% full, raise write_through_threshold to 2 so only pages with proven reuse get backed up — reduces host churn from single-access pages.

**Result:** *(eval running — job 17846)*
