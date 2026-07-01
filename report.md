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

**Result (vs V0) — NEGATIVE:**

| Metric | V0 | V2 | Delta |
|--------|----|----|-------|
| LooGLE TTFT mean (ms) | 40371 | 43796 | **+8.5% worse** |
| LooGLE TTFT p99 (ms) | 147141 | 138454 | -5.9% better |
| LooGLE out tok/s | 33.83 | 31.85 | -5.9% worse |
| LooGLE hit rate | 0.7676 | 0.7377 | **-3.9% worse** |
| ShareGPT TTFT mean (ms) | 35660 | 35220 | -1.2% better |
| ShareGPT req throughput | 1.36 | 1.35 | -0.7% |
| ShareGPT hit rate | 0.607 | 0.617 | +1.6% better |
| Disk read tokens | 1.74M | 1.91M | **+10.1% worse** |
| Host cached tokens | 29.2M | 27.1M | **-7.2% worse** |
| Load back mean (ms) | 1.727 | 1.677 | -2.9% better |

**Root cause:** The adaptive write threshold (V3) is the primary culprit. Under sustained host pressure (99.97% util), threshold=2 prevents first-access nodes from being backed up. When these nodes are evicted from GPU, they're completely lost — no host backup exists. Future requests must recompute from scratch or fetch from disk. This reduced host cached tokens by 7.2% and increased disk reads by 10.1%, overwhelming any benefit from the write-yield change.

The hit rate dropped from 76.8% to 73.8%, confirming fewer tokens are served from cache. The write-yield (V2) may also contribute by delaying backup completion, causing some nodes to be evicted before their write finishes.

**Takeaway:** Write-through threshold=1 is critical — backing up all first-access nodes immediately ensures no data is lost on GPU eviction. PCIe contention between loads and writes is NOT the bottleneck at this scale; data preservation is. Reverting both V2 and V3.

---

## V4 — SLRU eviction policy (flag-only, no code changes)

**Commit:** `a8730ad7b` (all V1-V3 code changes reverted — effectively baseline code)  
**Flag:** `--radix-eviction-policy slru`

**Hypothesis:** LRU evicts pages purely by recency, treating all pages equally. Under high churn (340M evictions in baseline), frequently-accessed shared prefixes can be evicted just because a burst of new single-use pages pushes them out.

SLRU (Segmented LRU) splits the eviction pool into two segments:
- **Probationary** (hit_count < 2): newly inserted pages; evicted first
- **Protected** (hit_count >= 2): pages with proven reuse; evicted last within segment, then by LRU

This protects shared document prefixes (high hit_count) from being displaced by ephemeral single-conversation tokens. V1 showed SLRU improved hit rate from 76.8% to 82.4% (+7.3%), but the effect was masked by the catastrophic amortized eviction change. This test isolates SLRU on unmodified code.

**Result:** *(eval running — job 17853)*

---

## V5 — Graduated SLRU (GSLRU) eviction

**Commit:** *(pending — code ready)*  
**Flag:** `--radix-eviction-policy gslru`

**Hypothesis:** Binary SLRU uses only two segments: probationary (hit_count < 2) and protected (hit_count >= 2). All nodes with hit_count >= 2 are treated identically — a node accessed twice has the same eviction priority as a shared system prompt accessed 50 times.

GSLRU uses `min(hit_count, 4)` as the segment index, creating 5 eviction tiers:
- Tier 0: hit_count=0 (brand new, never accessed)
- Tier 1: hit_count=1 (accessed once — ephemeral)
- Tier 2: hit_count=2 (moderate reuse)
- Tier 3: hit_count=3 (frequent reuse)
- Tier 4+: hit_count>=4 (hot shared prefixes — maximum protection, capped)

This gives finer-grained protection. Under LooGLE's shared-document workload, document prefixes touched by many questions accumulate high hit_count. Binary SLRU protects them from single-use pages but can still evict them for any node with hit_count=2. GSLRU ensures highly-shared prefixes survive longer.

**Risk:** More segments mean the protected tiers hold more data, potentially reducing the probationary pool size. If the working set is dominated by high-hit-count nodes, eviction may have fewer candidates. However, at 99.96% host utilization, the churn is so high that finer eviction ordering should help, not hurt.

**Result:** *(pending — will eval after V4)*
