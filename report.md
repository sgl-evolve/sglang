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

## V1 — SLRU eviction + amortized host eviction

**Commit:** `5fdbbad48`  
**Flag:** `--radix-eviction-policy slru`

**Hypothesis:** LRU eviction thrashes shared prefixes under high reuse. SLRU (segmented LRU with protected_threshold=2) shields frequently-accessed nodes from eviction. Additionally, the per-call overhead of host eviction is high because it rebuilds the heap each time; amortized eviction (evict more than the minimum per call) reduces total calls.

**What changed:**
- `hiradix_cache.py:evict_host()` — changed from `while num_evicted < num_tokens` to `target = num_tokens + max(num_tokens, page_size * 64)`, evicting extra tokens per call to amortize heap rebuild cost.
- `--radix-eviction-policy slru` flag — uses SLRUStrategy (probation + protected segments) instead of LRU.

**Result:** *(eval running — job 17844)*

---

## V2 — Write-stream yields PCIe to load-stream

**Commit:** `1f4b0b85a`  
**Stacked on:** V1

**Hypothesis:** Write-through DMA (GPU→host, `write_stream`) and load-back DMA (host→GPU, `load_stream`) share the physical PCIe bus. When both run concurrently, each gets ~half the bandwidth, doubling per-layer load-back latency. Since load-back is on the TTFT-critical path, writes should yield.

**What changed:**
- `cache_controller.py:start_writing()` — before issuing write DMA, records a CUDA sync event on `load_stream` and makes `write_stream` wait on it. Write DMA only starts after all pending loads complete. Zero overhead when no loads are active (event fires immediately).

**Result:** *(pending — will eval after V1)*

---

## V3 — Adaptive write threshold under host memory pressure

**Commit:** `1bab5fb52`  
**Stacked on:** V1 + V2

**Hypothesis:** With write_through (threshold=1), every new page is backed up on first access. When host is >95% utilized, every backup triggers a host eviction — pure churn for pages that won't be reused (unique conversation tokens). Raising the threshold to 2 under pressure means only pages with proven reuse (2+ accesses) get backed up, preserving host memory for high-value shared prefixes.

**What changed:**
- `hiradix_cache.py:_inc_hit_count()` — when host pool has <5% free slots, effective write_through_threshold is raised to max(original, 2). Single-access pages skip backup; multi-access pages still get backed up.

**Result:** *(pending — will eval as part of combined V1+V2+V3)*
