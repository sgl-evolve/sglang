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

**Result (vs V0) — POSITIVE (first improvement):**

| Metric | V0 | V4 | Delta |
|--------|----|----|-------|
| LooGLE TTFT mean (ms) | 40371 | 34832 | **-13.7% better** |
| LooGLE TTFT p99 (ms) | 147141 | 135216 | **-8.1% better** |
| LooGLE out tok/s | 33.83 | 37.76 | **+11.6% better** |
| LooGLE hit rate | 0.7676 | 0.8127 | **+5.9% better** |
| LooGLE req throughput | 2.065 | 2.448 | **+18.6% better** |
| ShareGPT TTFT mean (ms) | 35660 | 35020 | **-1.8% better** |
| ShareGPT req throughput | 1.36 | 1.36 | unchanged |
| ShareGPT hit rate | 0.607 | 0.617 | +1.6% |
| Disk read tokens | 1.74M | 2.62M | +50.7% more |
| Host util | 0.9996 | 0.9955 | slightly lower |
| Load back mean (ms) | 1.727 | 1.604 | -7.1% better |

**Analysis:** SLRU isolates cleanly and delivers strong gains. The +5.9% hit rate improvement translates to -13.7% TTFT on LooGLE because:
1. Higher hit rate → less prefill compute per request (fewer tokens to recompute)
2. Less prefill → faster request completion → queue drains faster → less queueing delay
3. The effect compounds: TTFT drop (-13.7%) far exceeds hit rate gain (+5.9%)

Disk reads increased (+50.7%) — SLRU protects high-hit-count nodes in host, which means low-hit-count nodes get evicted to disk more aggressively. But the net effect is positive because the protected nodes are reused much more often.

ShareGPT shows modest improvement — its shorter, more diverse prompts have less prefix sharing, so SLRU's protected segment has fewer entries.

**Takeaway:** SLRU is a clean, flag-only win. The binary protected/probationary split successfully shields shared prefixes from ephemeral single-use pages. Next: test graduated SLRU (GSLRU) which uses finer-grained protection tiers.

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

**Result (vs V0) — MARGINAL POSITIVE (incumbent: V4 SLRU):**

| Metric | V0 | V4 (SLRU) | V5 (GSLRU) | V5 vs V4 |
|--------|----|-----------|-----------:|----------|
| LooGLE TTFT mean (ms) | 40371 | 34832 | 34637 | -0.6% |
| LooGLE TTFT p99 (ms) | 147141 | 135216 | 134893 | -0.2% |
| LooGLE out tok/s | 33.83 | 37.76 | 38.07 | +0.8% |
| LooGLE hit rate | 0.7676 | 0.8127 | 0.8145 | +0.2% |
| ShareGPT TTFT (ms) | 35660 | 35020 | 35420 | +1.1% |
| ShareGPT throughput | 1.36 | 1.36 | 1.37 | +0.7% |
| ShareGPT hit rate | 0.607 | 0.617 | 0.620 | +0.5% |
| Disk read tokens | 1.74M | 2.62M | 3.14M | +19.8% |

**Analysis:** GSLRU provides marginal improvement over binary SLRU (+0.2% hit rate, -0.6% TTFT). The finer-grained protection tiers (5 segments instead of 2) help slightly, but the dominant effect is the binary distinction between single-access (probationary) and multi-access (protected) nodes. ShareGPT TTFT regressed slightly (+1.1%) — the extra protection tiers may over-protect stale nodes in ShareGPT's diverse, less-repetitive workload.

**Takeaway:** The major eviction quality gain comes from the SLRU binary split. GSLRU adds diminishing returns. The eviction strategy space is largely explored. Future improvements should be orthogonal to eviction ordering — e.g., reducing eviction VOLUME, improving cache locality, or changing tier transition policies.

---

## V6 — GSLRU + LPM scheduling (cache-aware request ordering)

**Commit:** *(pending — code ready)*
**Flag:** `--radix-eviction-policy gslru` (via code) + LPM scheduling (via code)

**Hypothesis:** Under FCFS scheduling, requests are processed in arrival order regardless of cache state. A request whose prefix is entirely GPU-resident (fast) may wait behind a request with no cached prefix (slow, requires full prefill). This wastes cache efficiency.

LPM (Longest Prefix Match) scheduling reorders the waiting queue to process requests with the most cached data first. Benefits:
1. Requests with hot cached prefixes get served first → less prefill → faster completion
2. Cached data stays hot longer (used before eviction can displace it)
3. Requests sharing the same document prefix cluster together → better batching

Code changes:
- `scheduler.py`: Auto-switch from FCFS to LPM when hierarchical cache is enabled
- `schedule_policy.py`: Raise LPM→FCFS fallback threshold from 128 to 512 under hicache (our working queue is ~100-150)

This stacks on top of GSLRU eviction — better eviction ordering + better request ordering is complementary.

**Risk:** LPM may starve late-arriving requests if early requests keep refreshing cached prefixes. Also, O(n * match_cost) prefix matching for each scheduling round adds CPU overhead.

**Result:** *(pending)*
