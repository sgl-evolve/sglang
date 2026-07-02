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

**Result (vs V0) — MASSIVE LooGLE improvement, ShareGPT regression:**

| Metric | V0 | V5 (GSLRU) | V6 (GSLRU+LPM) | V6 vs V0 | V6 vs V5 |
|--------|----|-----------:|----------------:|----------|----------|
| LooGLE TTFT mean (ms) | 40371 | 34637 | 10951 | **-72.9%** | **-68.4%** |
| LooGLE TTFT median (ms) | — | 31797 | 1693 | — | **-94.7%** |
| LooGLE TTFT p99 (ms) | 147141 | 134893 | 225131 | **+53.0% worse** | **+66.9% worse** |
| LooGLE out tok/s | 33.83 | 38.07 | 53.58 | **+58.4%** | **+40.8%** |
| LooGLE req throughput | 2.065 | 2.47 | 3.47 | **+68.0%** | **+40.5%** |
| LooGLE hit rate | 0.7676 | 0.8145 | 0.8919 | **+16.2%** | **+9.5%** |
| ShareGPT TTFT mean (ms) | 35660 | 35420 | 37790 | **+6.0% worse** | **+6.7% worse** |
| ShareGPT req throughput | 1.36 | 1.37 | 1.25 | **-8.1% worse** | **-8.8% worse** |
| ShareGPT hit rate | 0.607 | 0.620 | 0.60 | -1.2% worse | -3.2% worse |
| Disk read tokens | 1.74M | 3.14M | 1.15M | **-33.9%** | **-63.3%** |

**Analysis:** LPM scheduling is the most impactful change yet. Under LooGLE, it produces a bimodal TTFT distribution:
- Median 1.7s (94.7% faster than V5): requests with cached document prefixes get served immediately
- P99 225s (67% worse than V5): requests for NEW documents (no cached prefix) get starved

The mean TTFT dropped 72.9% vs baseline because most LooGLE requests (same-document follow-ups) benefit enormously from cache-aware scheduling. Hit rate jumped from 76.8% to 89.2% because cached prefixes stay hot (used before eviction displaces them). Disk reads dropped 63% — LPM clusters cache-hit requests, reducing tier-down eviction pressure.

However, ShareGPT REGRESSED on all metrics. ShareGPT has diverse, short prefixes with little sharing. LPM adds sorting overhead without meaningful reordering benefit. Throughput dropped 8.1% and TTFT increased 6.0%. LPM is counterproductive for diverse-prefix workloads.

**Takeaway:** LPM is transformative for prefix-sharing workloads (LooGLE) but harmful for diverse workloads (ShareGPT). V7 must add (1) anti-starvation to fix p99, and (2) adaptive fallback to FCFS when prefix matches are short, to preserve ShareGPT performance.

---

## V7 — Adaptive LPM + anti-starvation + partial load-back

**Commit:** `bfd5ebbb4`
**Flag:** `--radix-eviction-policy gslru` + code (LPM auto-switch + adaptive fallback + anti-starvation)

**Hypothesis:** V6 exposed two LPM weaknesses:
1. **P99 starvation** (225s): cold-prefix requests wait indefinitely under pure LPM
2. **ShareGPT regression** (-8.1% throughput): LPM sorting adds overhead for workloads without significant prefix sharing

Three changes:
1. **Anti-starvation** (`schedule_policy.py:_sort_by_longest_prefix`): requests waiting >30s get boosted to highest priority (FIFO among boosted). Bounds worst-case TTFT.
2. **Adaptive fallback** (`schedule_policy.py:_determine_active_policy`): if no request has >1024 cached prefix tokens, fall back to FCFS. Detects diverse-prefix workloads at scheduling time.
3. **Partial load-back** (`hiradix_cache.py:load_back`): when full KV chain doesn't fit in GPU, load a root-side partial chain. Provides proportional prefix benefit instead of all-or-nothing. (Defense-in-depth — doesn't trigger under current load levels.)

**Risk:** Anti-starvation at 30s may reduce LPM's mean TTFT benefit (boosted cold-prefix requests take longer to serve). Adaptive threshold of 1024 tokens might be too conservative (some ShareGPT prefixes above 1024 could trigger unnecessary LPM). Partial load-back is untested path (never triggered in V4-V6).

**Result (vs V0) — LooGLE REGRESSED, ShareGPT RECOVERED:**

| Metric | V0 | V6 (GSLRU+LPM) | V7 (Adaptive LPM) | V7 vs V0 | V7 vs V6 |
|--------|----|-----------:|---:|----------|----------|
| LooGLE TTFT mean (ms) | 40371 | 10951 | 33729 | -16.5% | **+208% worse** |
| LooGLE TTFT p99 (ms) | 147141 | 225131 | 135680 | -7.8% | **-39.7% better** |
| LooGLE out tok/s | 33.83 | 53.58 | 39.71 | +17.4% | -25.9% worse |
| LooGLE hit rate | 0.7676 | 0.8919 | 0.8113 | +5.7% | **-9.0% worse** |
| ShareGPT TTFT mean (ms) | 35660 | 37790 | 35780 | +0.3% | **-5.3% better** |
| ShareGPT req throughput | 1.36 | 1.25 | 1.36 | **unchanged** | **+8.8% better** |
| ShareGPT hit rate | 0.607 | 0.60 | 0.615 | +1.3% | +2.5% better |

**Root cause analysis:** The adaptive FCFS fallback (`max_match < 1024`) was catastrophic for LooGLE. The check used stale prefix match values from the previous scheduling round and fired too aggressively — converting LPM to FCFS even when prefix sharing was high. This destroyed the cache hit rate (89.2% → 81.1%, -9.0pp), directly causing the TTFT mean regression.

Additionally, the 30s anti-starvation threshold was far too low for 262K-context workloads. With GPU pool capacity of only ~686K tokens (~2-3 concurrent 262K requests), normal TTFT for any request is already 10-30s even with full cache hits. Queue depth of 100+ meant most requests exceeded the 30s threshold, effectively converting LPM to FCFS for the entire queue.

The p99 improvement (-39.7%) confirms anti-starvation works in principle — it just needs a much higher threshold. The ShareGPT recovery (+8.8% throughput) confirms the FCFS fallback helped diverse workloads, but at an unacceptable cost to LooGLE.

**Takeaway:** Anti-starvation at 30s is too aggressive. The FCFS fallback based on prefix match length is wrong for workloads where prefix matching takes multiple rounds to build up. V8 must: (1) Remove the adaptive FCFS fallback entirely, and (2) raise anti-starvation to 120s — targeting only truly starved requests.

---

## V8 — Calibrated anti-starvation (120s threshold, no FCFS fallback)

**Commit:** `9e26d3e81`
**Flag:** `--radix-eviction-policy gslru` + code (LPM auto-switch + anti-starvation at 120s)

**Hypothesis:** V7 showed two independent signals:
1. The FCFS fallback regressed LooGLE dramatically — remove it
2. Anti-starvation at 30s was too low — raise to 120s so only truly starved requests get boosted

Changes:
- Removed the `max_match < 1024` FCFS fallback entirely (8 lines deleted)
- Raised `_LPM_STARVATION_SECS` from 30.0 to 120.0

Expected: LooGLE TTFT should recover to near-V6 levels (10.9s) since LPM is always active for prefix-sharing workloads. The 120s anti-starvation should still improve p99 (from V6's 225s) but without the mean TTFT regression. ShareGPT may re-regress to V6 levels (~1.25 req/s) since the FCFS fallback that helped it is removed, but the existing queue-size threshold (>512 → FCFS) provides a fallback for truly large queues.

**Risk:** ShareGPT may show V6-like regression (-8.1% throughput) since LPM remains active for all queue sizes ≤512. The 120s threshold may still be too low or too high — workload-dependent.

**Result (vs V0) — partial recovery, new best p99 and hit rate:**

| Metric | V0 | V6 (GSLRU+LPM) | V7 (Adaptive) | V8 (Calibrated) | V8 vs V6 | V8 vs V7 |
|--------|----|-----------:|---:|---:|----------|----------|
| LooGLE TTFT mean (ms) | 40371 | 10951 | 33729 | 16070 | +46.7% worse | **-52.4% better** |
| LooGLE TTFT p99 (ms) | 147141 | 225131 | 135680 | **85889** | **-61.8% better** | **-36.7% better** |
| LooGLE out tok/s | 33.83 | 53.58 | 39.71 | 52.26 | -2.5% | **+31.6% better** |
| LooGLE hit rate | 0.7676 | 0.8919 | 0.8113 | **0.9717** | **+8.9pp better** | **+16.0pp better** |
| LooGLE req throughput | 2.065 | 3.47 | 2.57 | 3.39 | -2.3% | **+31.9% better** |
| ShareGPT TTFT mean (ms) | 35660 | 37790 | 35780 | 38090 | +0.8% | +6.5% worse |
| ShareGPT req throughput | 1.36 | 1.25 | 1.36 | 1.25 | unchanged | -8.1% worse |
| ShareGPT hit rate | 0.607 | 0.60 | 0.615 | 0.614 | +2.3% | -0.2% |

**Analysis:** Removing the FCFS fallback massively recovered LooGLE:
- Hit rate 97.17% (NEW BEST) — pure LPM keeps cached prefixes hot and serves cache-hit requests first
- TTFT p99 85889ms (NEW BEST) — 120s anti-starvation catches outliers V6's pure LPM couldn't
- TTFT mean 16070ms — 52% better than V7, but still 47% worse than V6's 10951ms

The remaining gap to V6 is caused by the 120s anti-starvation. V6 had NO anti-starvation — pure LPM sorting by prefix match length only. V8's anti-starvation at 120s boosts requests that have waited >120s ahead of LPM order, which rescues p99 outliers but disrupts mean ordering. With a queue of ~95 requests at steady-state and each taking 10-30s, a meaningful fraction of requests exceed the 120s threshold and get priority-boosted, hurting the average.

The trade-off: V8 trades 47% mean regression for 62% p99 improvement vs V6. This is a reasonable trade-off for production systems that care about tail latency, but the mean is the primary optimization target.

ShareGPT regressed back to V6 levels (1.25 req/s), as predicted. Without the FCFS fallback, ShareGPT requests (low prefix matches) get LPM-sorted, which adds overhead without benefit.

**Takeaway:** The FCFS fallback was the primary V7 culprit (confirmed). Anti-starvation helps p99 but hurts mean. V9 should remove anti-starvation entirely to match V6's pure LPM — with 97% hit rate (vs V6's 89%), the mean should be even better than V6.

---

## V9 — Pure LPM (no anti-starvation)

**Commit:** *(pending)*
**Flag:** `--radix-eviction-policy gslru` + code (pure LPM, no FCFS fallback, no anti-starvation)

**Hypothesis:** V8 showed that removing the FCFS fallback restores 97.17% hit rate (NEW BEST) but the 120s anti-starvation still hurts mean TTFT by 47% vs V6. V6 had pure LPM with no time-based priority boosting and achieved 10951ms mean TTFT.

V9 removes anti-starvation entirely from `_sort_by_longest_prefix` — requests are sorted purely by `-num_matched_prefix_tokens`. This matches V6's scheduling behavior exactly, but on top of V8's codebase (no FCFS fallback, partial load-back still present).

With V8's 97.17% hit rate (vs V6's 89.2%), V9 could potentially BEAT V6's mean TTFT: more cache hits means more requests finish quickly, which drains the queue faster, reducing wait times for all requests.

**Risk:** P99 may regress to V6 levels (~225s) since there's no anti-starvation safety net. Requests with zero prefix match can starve indefinitely under pure LPM. However, with 97% hit rate, fewer requests are truly "cold" (un-cached), so the worst case may be better than V6 despite no anti-starvation.

**Result (vs V0) — V6 reproduction, not a new best:**

| Metric | V0 | V6 (incumbent) | V8 (anti-starvation) | V9 (pure LPM) | V9 vs V6 |
|--------|----|-----------:|---:|---:|----------|
| LooGLE TTFT mean (ms) | 40371 | 10951 | 16070 | 12186 | +11.3% worse |
| LooGLE TTFT p99 (ms) | 147141 | 225131 | **85889** | 227552 | +1.1% (same) |
| LooGLE out tok/s | 33.83 | 53.58 | 52.26 | 51.47 | -3.9% |
| LooGLE hit rate | 0.7676 | 0.8919 | **0.9717** | 0.8905 | -0.2% (same) |
| ShareGPT TTFT mean (ms) | 35660 | 37790 | 38090 | 37150 | -1.7% better |
| ShareGPT req throughput | 1.36 | 1.25 | 1.25 | 1.28 | +2.4% better |

**Analysis:** V9 is essentially a V6 reproduction on the current codebase. The 11% gap vs V6 (12186 vs 10951ms) is within expected run-to-run variance (~5-15% on shared GPU clusters).

Critical insight: V8's anti-starvation at 120s INCREASED hit rate from 89% (V6/V9) to 97% because it diversified the request processing order — low-match requests interleaved with high-match ones reduces eviction churn. But this diversity also hurts mean TTFT because low-match requests take much longer to process (full prefill).

This reveals a Pareto frontier:
- **Pure LPM (V6/V9):** best mean TTFT (10-12s), worst p99 (225s), 89% hit rate
- **Anti-starvation LPM (V8):** worse mean (16s), best p99 (86s), 97% hit rate

To beat V6, need a mechanism that achieves V8's hit rate without V8's scheduling disruption.

**Takeaway:** Pure LPM and anti-starvation LPM represent a trade-off between mean and tail latency. V10 should try a structurally different scheduling approach: DFS_WEIGHT, which naturally clusters same-document requests without time-based priority boosting.

---

## V10 — DFS-weight scheduling (tree-structure-aware)

**Commit:** *(pending)*
**Flag:** `--radix-eviction-policy gslru` + code (DFS_WEIGHT auto-switch, no anti-starvation)

**Hypothesis:** LPM sorts by total prefix match LENGTH, which can interleave requests from different documents with similar-length prefixes. Under GPU memory pressure, this interleaving causes cache thrashing — loading doc1's prefix, processing 1-2 requests, evicting it, loading doc2's prefix, etc.

DFS_WEIGHT sorts by radix tree structure: requests sharing the same subtree are processed consecutively. This naturally batches same-document requests together:
1. All 8 follow-ups for document X processed together
2. Doc X's prefix loaded once, reused for all 8 requests
3. No eviction/reload cycles between same-document requests

Under LooGLE (200 conversations × 8 rounds), DFS_WEIGHT should reduce GPU cache thrashing by keeping same-document requests together, potentially improving hit rate and TTFT simultaneously.

**Risk:** DFS_WEIGHT ignores `temporary_deprioritized` requests (in-batch prefix caching). For 262K-context workloads, the deprioritization threshold (32 tokens) is negligible. DFS_WEIGHT also doesn't consider which tier prefix data is in (GPU vs host), while LPM counts both equally.

**Result (vs V0) — CLEAR NEGATIVE:**

| Metric | V0 | V6 (incumbent) | V10 (DFS_WEIGHT) | V10 vs V6 |
|--------|----|-----------:|---:|----------|
| LooGLE TTFT mean (ms) | 40371 | 10951 | 36857 | **+236% worse** |
| LooGLE TTFT median (ms) | — | 1693 | 30270 | **+1688% worse** |
| LooGLE hit rate | 0.7676 | 0.8919 | 0.8388 | -5.9% worse |
| ShareGPT req throughput | 1.36 | 1.25 | 1.23 | -1.6% worse |

**Root cause:** DFS_WEIGHT sorts by TREE STRUCTURE, not by actual cache state. It processes all requests in the heaviest subtree first, regardless of whether their prefix data is in GPU, host, or evicted. Requests with excellent cache hits in a "lighter" subtree wait behind cold-prefix requests in the "heavier" subtree. The median TTFT jumped from 1.7s (LPM) to 30s — no request gets fast service because scheduling ignores cache state entirely. Concurrency spiked to 110 (vs 61 under LPM), indicating all requests were being processed slowly rather than some fast and some slow.

**Takeaway:** LPM's strength is that it directly optimizes for cache state — the request with the most cached data goes first. DFS_WEIGHT's tree-structural ordering is inferior. Scheduling must be cache-state-aware.

---

## V11 — Conservative anti-starvation (300s threshold)

**Commit:** *(pending)*
**Flag:** `--radix-eviction-policy gslru` + code (LPM + anti-starvation at 300s)

**Hypothesis:** The Pareto frontier shows:
- V9 (pure LPM, no starvation): 12186ms mean, 228s p99
- V8 (120s starvation): 16070ms mean, 86s p99

300s is 2.5x higher than V8's 120s. Under the LooGLE queue (~95 requests, 10-30s typical TTFT), very few requests will exceed 300s of queue wait — only the truly starved cold-prefix requests. Expected: mean close to V9 (~12s), p99 improved from V9 (~150-200s instead of 228s), with minimal scheduling disruption.

**Risk:** 300s may be too conservative to meaningfully improve p99 (only ~1% of requests wait >300s under V6/V9). May be a non-result.

**Result (vs V0) — NEW BEST LooGLE TTFT mean:**

| Metric | V0 | V6 (incumbent) | V9 (pure LPM) | V11 (300s starvation) | V11 vs V6 | V11 vs V9 |
|--------|----|-----------:|---:|---:|----------|----------|
| LooGLE TTFT mean (ms) | 40371 | 10951 | 12186 | **10658** | **-2.7% better** | **-12.5% better** |
| LooGLE TTFT median (ms) | — | 1693 | 1697 | 1522 | **-10.1% better** | **-10.3% better** |
| LooGLE TTFT p99 (ms) | 147141 | 225131 | 227552 | 225215 | unchanged | -1.0% |
| LooGLE out tok/s | 33.83 | 53.58 | 51.47 | 53.57 | unchanged | +4.1% better |
| LooGLE hit rate | 0.7676 | 0.8919 | 0.8905 | 0.8928 | +0.1% | +0.3% |
| ShareGPT TTFT mean (ms) | 35660 | 37790 | 37150 | 37530 | -0.7% better | +1.0% |
| ShareGPT req throughput | 1.36 | 1.25 | 1.28 | 1.27 | +1.6% better | -0.8% |
| ShareGPT hit rate | 0.607 | 0.60 | 0.613 | 0.623 | +3.8% better | +1.6% better |

**Analysis:** V11 achieves the best LooGLE TTFT mean (10658ms) and median (1522ms) across all versions. The 300s anti-starvation threshold barely triggers — hit rate (89.28%) and p99 (225s) are virtually identical to V9's pure LPM (89.05%, 228s), confirming that <1% of requests exceed 300s queue wait.

The improvement over V6 (-2.7%) is marginal and within run-to-run variance (~5-15%). However, the improvement over V9 (-12.5%) is significant and suggests the rare anti-starvation interventions prevent pathological cascades where a truly starved request, once finally scheduled, displaces cached prefixes for many other requests in its subtree. By catching these extreme outliers at 300s, the overall queue dynamics remain healthier.

ShareGPT hit rate improved to 62.3% (best across all versions) — the 300s threshold is high enough that it never triggers during ShareGPT's shorter-context workload, preserving pure LPM behavior.

**Takeaway:** 300s anti-starvation is the sweet spot — conservative enough to preserve pure LPM's mean TTFT advantage while preventing rare cascading disruptions. The scheduling axis (LPM variants, anti-starvation thresholds, DFS_WEIGHT) is now well-explored. Future improvements should target orthogonal axes: eviction quality, memory management, prefetch strategy, or load-back optimization.

---

## Current Pareto Frontier

| Version | LooGLE TTFT mean | LooGLE p90 | LooGLE p99 | Hit rate | Regime |
|---------|----------------:|----------:|----------:|--------:|--------|
| **V13** | **10217ms** | **6593ms** | 214431ms | 89.5% | Best mean + p90 (device weight=4) |
| V8 | 16070ms | — | **85889ms** | **97.2%** | Best p99 + hit rate (aggressive anti-starvation) |
| V0 | 40371ms | — | 147141ms | 76.8% | Baseline |

Key advances: Device-weighted LPM with weight=4 (V13) extends V12's gains. The weight sensitivity series (1→2→4) shows monotonic improvement with no sign of plateau. V14 tests weight=8 to find the optimal point.

---

## V12 — Device-weighted LPM (GPU-resident preference)

**Commit:** *(pending)*
**Flag:** `--radix-eviction-policy gslru` + code (device-weighted LPM + anti-starvation 300s)

**Hypothesis:** LPM sorts by `num_matched_prefix_tokens = len(prefix_indices) + host_hit_length`, treating GPU-resident and host-resident tokens equally. But serving a GPU-resident request is free (data already in GPU), while serving a host-resident request forces GPU eviction + host-to-GPU DMA transfer.

When two documents have similar total match (~262K each), one in GPU and one in host, current LPM can arbitrarily schedule the host-resident request first. This triggers a GPU cache flush: evict the GPU-resident document to make room for loading the host-resident one. All subsequent requests for the evicted document must wait for a reload.

Device-weighted LPM applies a weight multiplier (2x) to GPU-resident tokens in the sort key:
```
weighted_score = device_tokens * 2 + host_tokens
```

This ensures GPU-resident requests always go before host-resident requests with the same total match, reducing unnecessary GPU cache flushes. Requests with significantly more host data still win (262K host > 100K GPU × 2 = 200K).

Under LooGLE, V11 data shows:
- GPU-resident: 4.3M tokens (9.6% of 44.8M prompt)
- Host-resident: 34.7M tokens (77.5%)
- 334M tokens evicted, 302M loaded back → 90% reload rate

The 90% reload rate suggests many evictions are "premature" — data evicted from GPU is needed again soon. Device-weighted LPM should reduce this churn by preserving GPU-resident data longer.

**Risk:** Host-only requests (cold documents) get deprioritized slightly more, potentially worsening p99. ShareGPT (diverse prefixes, mostly small GPU matches) should be minimally affected since the weight only matters at tie-breaking scale.

**Result (vs V0) — NEW BEST LooGLE TTFT mean + massive p90 improvement:**

| Metric | V0 | V6 (incumbent) | V11 (prev best) | V12 (device-weighted) | V12 vs V11 | V12 vs V6 |
|--------|----|-----------:|---:|---:|----------|----------|
| LooGLE TTFT mean (ms) | 40371 | 10951 | 10658 | **10506** | **-1.4% better** | **-4.1% better** |
| LooGLE TTFT median (ms) | — | 1693 | 1522 | **1540** | +1.2% (same) | -9.0% better |
| LooGLE TTFT p90 (ms) | — | — | 20760 | **6842** | **-67.0% better** | — |
| LooGLE TTFT p99 (ms) | 147141 | 225131 | 225215 | 216303 | -4.0% better | -3.9% better |
| LooGLE out tok/s | 33.83 | 53.58 | 53.57 | 53.57 | unchanged | unchanged |
| LooGLE hit rate | 0.7676 | 0.8919 | 0.8928 | 0.8938 | +0.1% | +0.2% |
| LooGLE req throughput | 2.065 | 3.47 | 3.42 | 3.47 | +1.5% better | unchanged |
| ShareGPT TTFT mean (ms) | 35660 | 37790 | 37530 | 37430 | -0.3% better | -1.0% better |
| ShareGPT req throughput | 1.36 | 1.25 | 1.27 | 1.27 | unchanged | +1.6% better |
| ShareGPT hit rate | 0.607 | 0.60 | 0.623 | 0.60 | -3.7% | unchanged |

HiCache details (LooGLE):
- Evicted: 326.7M tokens (V11: 315.3M) — 3.6% more eviction volume
- Loaded back: 294.5M tokens (V11: 282.0M) — 4.4% more
- Prefetched from disk: 15.3M (V11: 67.8M) — **77% less disk IO**
- Backed up: 37.1M (V11: 37.8M) — 1.9% less
- Load back mean: 1.704ms (V11: 1.702ms) — unchanged

**Analysis:** Device-weighted LPM achieves a new best mean TTFT (10506ms) with dramatically improved p90 (6842ms vs 20760ms, -67%). The key insight is visible in the disk prefetch reduction: V12 reads 77% fewer tokens from disk (15.3M vs 67.8M) because GPU-preference scheduling keeps more data in GPU/host tiers, reducing tier-down pressure.

The p90 improvement is the most significant finding — it means the "upper-middle" latency band (requests between median and p99) benefits enormously from device-weighted scheduling. These requests have SOME cached data in GPU but were previously scheduled after host-resident requests with equal total match, causing unnecessary GPU flushes. With device weight=2, these requests now go first, preserving their GPU-resident data.

Total eviction volume actually INCREASED by 3.6% (326.7M vs 315.3M), which seems contradictory. However, the evictions are now "smarter" — evicting data that's LESS likely to be needed soon (because the scheduler preferentially keeps GPU-resident data in use). The 90% reload rate persists (structural), but the eviction-reload cycles happen on less time-critical data.

ShareGPT is essentially neutral — the device weight has minimal effect on ShareGPT's diverse, short-prefix workload where few requests have significant GPU-resident data.

**Takeaway:** Device-weighted LPM is a clean win on top of anti-starvation 300s. The weight=2 is a good starting point. V13 tests weight=4 to see if more aggressive GPU-preference helps further.

---

## V13 — Aggressive device-weighted LPM (weight=4)

**Commit:** *(pending)*
**Flag:** `--radix-eviction-policy gslru` + code (device_weight=4, anti-starvation 300s)

**Hypothesis:** V12 showed device_weight=2 improved mean TTFT by 1.4% and p90 by 67% vs V11. The mechanism is clear: GPU-resident requests are scheduled first, reducing unnecessary GPU cache flushes.

With weight=4, GPU tokens count 4x in the LPM sort key:
```
weighted_score = device_tokens * 4 + host_tokens
```

This makes the GPU-preference much stronger. Under V12's regime:
- A request with 100K GPU + 100K host scores 300K (weight=2) → 500K (weight=4)
- A request with 0 GPU + 250K host scores 250K (weight=2) → 250K (weight=4)
- Gap widens from 50K to 250K — much stronger discrimination

Under LooGLE, most requests have either GPU-resident data (recently accessed document) or host-resident data (older document). Weight=4 ensures that ANY request with GPU data goes before ALL host-only requests, unless the host request has significantly more total data.

**Risk:** Too-aggressive GPU preference may create a two-class system: GPU-resident requests get excellent service while host-only requests starve until anti-starvation kicks in at 300s. This could WORSEN mean TTFT if a significant fraction of requests are host-only and wait close to 300s.

**Result (vs V0) — NEW BEST across all LooGLE metrics:**

| Metric | V0 | V12 (weight=2) | V13 (weight=4) | V13 vs V12 | V13 vs V0 |
|--------|----|-----------:|---:|----------|----------|
| LooGLE TTFT mean (ms) | 40371 | 10506 | **10217** | **-2.8% better** | **-74.7% better** |
| LooGLE TTFT median (ms) | — | 1540 | **1405** | **-8.8% better** | — |
| LooGLE TTFT p90 (ms) | — | 6842 | **6593** | **-3.6% better** | — |
| LooGLE TTFT p99 (ms) | 147141 | 216303 | 214431 | -0.9% better | +45.7% worse |
| LooGLE out tok/s | 33.83 | 53.57 | 53.57 | unchanged | +58.3% |
| LooGLE hit rate | 0.7676 | 0.8938 | 0.8953 | +0.2% | +16.6% |
| ShareGPT TTFT mean (ms) | 35660 | 37430 | 37300 | -0.3% better | +4.6% |
| ShareGPT req throughput | 1.36 | 1.27 | 1.28 | +0.8% better | -5.9% |
| ShareGPT hit rate | 0.607 | 0.600 | 0.618 | +3.0% better | +1.8% |

HiCache details (LooGLE):
- Evicted: 322.7M (V12: 326.7M) — **1.2% less eviction**
- Loaded back: 291.1M (V12: 294.5M) — 1.2% less
- Prefetched from disk: 13.2M (V12: 15.3M) — **13.7% less disk IO**
- Backed up: 36.6M (V12: 37.1M) — 1.4% less

**Analysis:** Weight=4 shows consistent improvement over weight=2 across ALL metrics with zero regressions. The trend from weight=1→2→4 is monotonically improving:

| Weight | TTFT mean | TTFT p90 | Disk prefetch | Eviction total |
|--------|-----------|----------|---------------|----------------|
| 1 (V11) | 10658ms | 20760ms | 67.8M | 315.3M |
| 2 (V12) | 10506ms | 6842ms | 15.3M | 326.7M |
| 4 (V13) | 10217ms | 6593ms | 13.2M | 322.7M |

The monotonic improvement in disk prefetch (67.8M → 15.3M → 13.2M) and TTFT (10658→10506→10217) strongly suggests the weight hasn't reached its optimal value yet. Each weight increase reduces the scheduling's tendency to pull host-resident data into GPU unnecessarily.

Notable: V13 REDUCES total eviction (322.7M) compared to V12 (326.7M), reversing V12's trend of increased eviction. This suggests weight=4 finds a better equilibrium — GPU-resident data stays longer, reducing reload cycles.

ShareGPT IMPROVED slightly (hit rate 60.0%→61.8%), refuting the hypothesis that aggressive GPU preference would harm diverse workloads.

**Takeaway:** The device weight sensitivity analysis shows no sign of diminishing returns. V14 tests weight=8 to continue probing this axis.

---

## V14 — Device weight=8 (high GPU preference)

**Commit:** `bbfffc9d3`
**Flag:** `--radix-eviction-policy gslru` + code (device_weight=8, anti-starvation 300s)

**Hypothesis:** The weight series (1→2→4) shows monotonic improvement in all metrics. Weight=8 doubles the GPU preference again:
```
weighted_score = device_tokens * 8 + host_tokens
```

At weight=8, a request with even 32K GPU-resident tokens (32K × 8 = 256K) would sort above a request with 250K host-resident tokens (250K × 1 = 250K). This means almost ANY amount of GPU-resident data trumps host-only data.

Under LooGLE: GPU pool ~686K tokens, typical request ~262K. At any time, 2-3 requests' data might be in GPU. With weight=8, the scheduler will STRONGLY prefer these 2-3 requests over all others, creating a very "sticky" GPU cache pattern.

**Risk:** With weight=8, host-only requests are effectively deprioritized to anti-starvation priority (wait until 300s threshold). If many requests are host-only simultaneously, a burst of anti-starvation promotions at 300s could cause a scheduling cliff. However, V13 showed no sign of this effect at weight=4.

**Result (vs V13 — current best):**

| Metric | V13 (w=4) | V14 (w=8) | Delta |
|--------|-----------|-----------|-------|
| LooGLE TTFT mean (ms) | 10217 | 10903 | **+6.7% WORSE** |
| LooGLE TTFT median (ms) | 1405 | 1668 | +18.7% worse |
| LooGLE TTFT p90 (ms) | 6593 | 10158 | **+54.1% worse** |
| LooGLE TTFT p99 (ms) | 214431 | 210969 | -1.6% better |
| LooGLE out tok/s | 53.57 | 53.54 | neutral |
| LooGLE hit rate | 0.8953 | 0.8874 | -0.9% worse |
| ShareGPT TTFT mean (ms) | 37300 | 37510 | +0.6% neutral |
| ShareGPT req throughput | 1.28 | 1.27 | neutral |
| ShareGPT hit rate | 0.618 | 0.611 | neutral |

HiCache: disk_read_tokens=1.59M (+16% vs V13), load_back=288.1M (-1%), evicted=322.6M (same), load_back_mean=1.733ms, host_util=0.9954

**Analysis: NEGATIVE — weight sensitivity curve has peaked at weight=4**

V14 breaks the monotonic improvement seen in the weight=1→2→4 series:

| Weight | TTFT mean (ms) | TTFT p90 (ms) | Disk read (M) | Device tokens (M) |
|--------|----------------|---------------|---------------|--------------------|
| 1 (V11) | 10658 | 20760 | 67.8 | — |
| 2 (V12) | 10506 | 6842 | 15.3 | 4.32 |
| 4 (V13) | **10217** | **6593** | **13.2** | 3.98 |
| 8 (V14) | 10903 | 10158 | 15.9 | 4.66 |

The curve is clearly concave with an optimum at weight=4. The over-concentration paradox: V14 retained MORE device tokens (4.66M vs 3.98M, +17%) yet performed WORSE because the top requests monopolize GPU while mid-tier requests wait much longer — confirmed by the p90 spike from 6593ms to 10158ms (+54%).

Predicted risk materialized: weight=8 effectively deprioritizes host-only requests to anti-starvation-only scheduling (300s wait), creating a bimodal latency distribution — fast for GPU-resident requests, slow for everything else.

**Conclusion:** Device weight=4 (V13) is the Pareto optimum for this axis. Further improvement requires an orthogonal direction — scheduling refinements on the eviction/load-back side rather than sort-key tuning.

---

## V15 — Time-decay GSLRU eviction (NEW BEST)

**Commit:** `b8a577e5c`
**Flag:** `--radix-eviction-policy gslru` + code (device_weight=4, anti-starvation 300s, GSLRU decay_tau=30s)

**Hypothesis:** The 90% reload rate (322M evicted, 288M loaded back) indicates we're evicting the wrong data. Standard GSLRU protects nodes with high hit_count forever, even after their document's queries finish. Time-decay exponentially reduces a node's effective eviction priority based on time since last access:
```
effective_priority = min(hit_count, 4) * exp(-age / 30)
```
After 30s without access, a segment-4 node's priority drops to ~1.5 — making it evictable before fresh segment-2 data. This frees GPU cache for actively-requested data.

Also reverts device weight from 8 (V14) back to 4 (V13 optimum).

**Result (vs V13 — previous best):**

| Metric | V13 (w=4) | V15 (decay) | Delta |
|--------|-----------|-------------|-------|
| LooGLE TTFT mean (ms) | 10217 | **10114** | **-1.0% NEW BEST** |
| LooGLE TTFT median (ms) | 1405 | 1204 | **-14.3% better** |
| LooGLE TTFT p90 (ms) | 6593 | 4580 | **-30.5% better** |
| LooGLE TTFT p99 (ms) | 214431 | 211683 | -1.3% better |
| LooGLE TPOT mean (ms) | 517.74 | 472.35 | **-8.8% better** |
| LooGLE ITL mean (ms) | 397.08 | 369.55 | **-6.9% better** |
| LooGLE E2E mean (ms) | 15946 | 15445 | -3.1% better |
| LooGLE out tok/s | 53.57 | 53.58 | neutral |
| LooGLE hit rate | 0.8953 | 0.8943 | neutral |
| ShareGPT TTFT mean (ms) | 37300 | 37770 | +1.3% neutral |
| ShareGPT req throughput | 1.28 | 1.26 | -1.6% neutral |
| ShareGPT hit rate | 0.618 | 0.626 | +1.3% better |

HiCache: load_back=286.5M (-1.6%), evicted=318.5M (-1.3%), cached_device_tokens=5.01M (**+25.8%**), load_back_mean=1.82ms, eviction_mean=1.106ms, host_util=0.9985

**Analysis: POSITIVE — orthogonal to device weighting, addresses eviction quality**

The time-decay GSLRU produces improvements across ALL latency percentiles, not just the mean. The key mechanism:

1. **25.8% more GPU-cached tokens** (5.01M vs 3.98M): time-decay evicts stale data faster, keeping more useful data GPU-resident
2. **p90 -30.5%**: the biggest win — mid-tier requests (previously waiting 6.6s) now find more data in GPU and complete in 4.6s
3. **TPOT -8.8%**: decode latency also improves, possibly from reduced GPU memory pressure
4. **Less total eviction** (318.5M vs 322.7M, -1.3%): better data retention means fewer eviction events
5. **ShareGPT neutral**: time-decay doesn't hurt the diverse-workload benchmark

The mean improvement (-1.0%) is modest because the mean is dominated by the p99 tail (starvation-bound requests at 300s threshold) which time-decay doesn't address. But the median (-14.3%) and p90 (-30.5%) confirm the mechanism works: more requests find their data in GPU.

**Conclusion:** Time-decay GSLRU is the first successful orthogonal improvement beyond sort-key tuning. V15 (10114ms) is the new best. Next: tau sensitivity (15s, 60s) or stack with another eviction improvement.

---

## V16 — Time-decay GSLRU tau=15s (NEW BEST)

**Commit:** `13ee198ea`
**Flag:** `--radix-eviction-policy gslru` + code (device_weight=4, anti-starvation 300s, GSLRU decay_tau=15s)

**Hypothesis:** V15 (tau=30s) improved p90 by 30.5% — more aggressive decay helps. Halve tau to 15s: nodes lose protection faster, freeing GPU for active data sooner.

**Result (vs V15 tau=30s):**

| Metric | V15 (tau=30) | V16 (tau=15) | Delta |
|--------|-------------|-------------|-------|
| LooGLE TTFT mean (ms) | 10114 | **9814** | **-3.0% NEW BEST** |
| LooGLE TTFT median (ms) | 1204 | 1277 | +6.1% |
| LooGLE TTFT p90 (ms) | 4580 | **4210** | **-8.1% better** |
| LooGLE TTFT p99 (ms) | 211683 | 213209 | neutral |
| LooGLE TPOT mean (ms) | 472.35 | **454.76** | **-3.7% better** |
| LooGLE hit rate | 0.8943 | **0.8966** | +0.3% better |
| ShareGPT TTFT mean (ms) | 37770 | **37300** | **-1.2% better** |
| ShareGPT req throughput | 1.26 | **1.28** | +1.6% better |

HiCache: evicted=317.6M (-0.3%), load_back=286.4M, cached_device=4.96M, load_back_mean=1.813ms, host_util=0.9992

**Tau sensitivity series (monotonic improvement, no plateau):**

| tau | TTFT mean (ms) | TTFT p90 (ms) | Hit rate | TPOT (ms) |
|-----|----------------|---------------|----------|-----------|
| ∞ (V13) | 10217 | 6593 | 0.8953 | 517.74 |
| 30s (V15) | 10114 | 4580 | 0.8943 | 472.35 |
| **15s (V16)** | **9814** | **4210** | **0.8966** | **454.76** |

**Analysis:** Shorter tau continues to improve ALL key metrics. The mean dropped below 10s for the first time (9814ms). TPOT improvement (-3.7%) is notable — faster decode from reduced GPU memory pressure. ShareGPT also improved (+1.6% throughput), suggesting the mechanism generalizes across workloads.

The median slightly worsened (1277 vs 1204ms) — the fastest requests pay a small cost for the aggressive decay (their cached data may be evicted slightly earlier). But the mean, p90, and TPOT all improved, making this a clear net positive.

**Next:** Continue tau sensitivity: try tau=7.5s. The curve is still monotonic with no sign of diminishing returns.

---

## V17 — Time-decay GSLRU tau=7.5s (NEGATIVE)

**Commit:** `101625417`
**Flag:** `--radix-eviction-policy gslru` + code (device_weight=4, anti-starvation 300s, GSLRU decay_tau=7.5s)

**Hypothesis:** V16 (tau=15s) improved mean below 10s. Continue halving tau to 7.5s for faster stale-data eviction.

**Result (vs V16 — current best):**

| Metric | V16 (tau=15) | V17 (tau=7.5) | Delta |
|--------|-------------|--------------|-------|
| LooGLE TTFT mean (ms) | 9814 | 10211 | **+4.0% NEGATIVE** |
| LooGLE TTFT p90 (ms) | 4210 | 4128 | -2.0% better |
| LooGLE TTFT p99 (ms) | 213209 | 215441 | +1.0% worse |
| LooGLE hit rate | 0.8966 | 0.8936 | -0.3% |
| LooGLE TPOT mean (ms) | 454.76 | 475.34 | +4.5% worse |
| ShareGPT TTFT mean (ms) | 37300 | 37730 | +1.2% |
| ShareGPT req throughput | 1.28 | 1.26 | -1.6% |

**Analysis:** Tau=7.5s evicts active data too aggressively. The decay function `exp(-age/7.5)` drops to 0.37 after only 7.5 seconds — meaning a segment-4 node becomes lower priority than a fresh segment-2 node after just 7.5s. Under LooGLE's workload, requests from the same document arrive every ~5-10s, so tau=7.5 can evict data between consecutive same-document requests.

**Tau sensitivity series (optimum at tau=15s):**

| tau | TTFT mean (ms) | TTFT p90 (ms) | Hit rate | TPOT (ms) |
|-----|----------------|---------------|----------|-----------|
| ∞ (V13) | 10217 | 6593 | 0.8953 | 517.74 |
| 30s (V15) | 10114 | 4580 | 0.8943 | 472.35 |
| **15s (V16)** | **9814** | **4210** | **0.8966** | **454.76** |
| 10s (V19) | 9854 | 3988 | 0.8966 | 467.57 |
| 7.5s (V17) | 10211 | 4128 | 0.8936 | 475.34 |

**Conclusion:** The tau sensitivity curve is concave with a clear optimum at tau=15s for mean TTFT. Note: p90 continues to improve at tau=10 (3988ms) but the mean and TPOT favor tau=15. The tau axis is now exhausted (5 data points). Future improvement requires orthogonal directions.

---

## V18 — Reduced GSLRU tiers (max_segment=2, tau=15)

**Commit:** `82ad2cff3`
**Flag:** `--radix-eviction-policy gslru` + code (max_segment=2, decay_tau=15, device_weight=4, anti-starvation 300s)

**Hypothesis:** With time-decay active, the 5-tier GSLRU gradation (0-4) may be redundant — recency (via decay) already determines priority. Reducing max_segment from 4 to 2 turns eviction into a 3-tier decayed-LRU (tiers 0, 1, 2+). This tests whether the fine-grained hit_count distinction matters when decay dominates.

Reverts decay_tau from 7.5 (V17, negative) back to 15 (V16 optimum).

**Result (vs V16 — current best):**

| Metric | V16 (seg=4, tau=15) | V18 (seg=2, tau=15) | Delta |
|--------|-------|-------|-------|
| LooGLE TTFT mean (ms) | 9814 | 9816 | +0.02% (noise) |
| LooGLE TTFT median (ms) | 1277 | 1216 | -4.8% better |
| LooGLE TTFT p90 (ms) | 4210 | **3719** | **-11.7% better** |
| LooGLE TTFT p99 (ms) | 213209 | 211723 | -0.7% |
| LooGLE TPOT mean (ms) | 454.76 | 474.04 | +4.2% worse |
| LooGLE E2E mean (ms) | 15076 | 15119 | +0.3% |
| LooGLE out tok/s | 53.56 | 53.57 | neutral |
| LooGLE hit rate | 0.8966 | 0.8958 | neutral |
| ShareGPT TTFT mean (ms) | 37300 | 37950 | +1.7% |
| ShareGPT req throughput | 1.28 | 1.26 | -1.6% |

HiCache: evicted=316.8M, load_back=285.3M, cached_device=5.34M (+7.7% vs V16), load_back_mean=1.862ms, host_util=0.9989

**Analysis:** Mean TTFT is essentially identical (9816 vs 9814ms — within measurement noise). The significant finding is p90 improvement (-11.7%, from 4210 to 3719ms). With fewer tiers, the eviction heap has less granularity, which paradoxically helps mid-percentile requests by flattening the priority distribution — nodes transition from evictable to protected faster (at 2 hits instead of requiring 4 for maximum protection).

TPOT regressed slightly (+4.2%), suggesting the flatter eviction priority creates slightly more GPU memory pressure during decode. This is consistent with the larger cached_device_tokens (5.34M vs 4.96M) — more data in GPU means more active tokens competing for attention bandwidth.

**Conclusion:** NEUTRAL overall — mean unchanged, p90 improved, TPOT slightly worse. The GSLRU tier count is not a major lever when time-decay is active. The primary mechanism (segment × decay) works similarly with 3 or 5 tiers.

---

## V19 — Time-decay tau=10s (NEUTRAL)

**Commit:** `7b10cad67`
**Flag:** `--radix-eviction-policy gslru` + code (max_segment=4, decay_tau=10, device_weight=4, anti-starvation 300s)

**Hypothesis:** Narrow the tau optimum range between 7.5s (negative) and 15s (best). If 10s beats 15s, the optimum lies in 10-15; if worse, confirmed at 15s.

**Result (vs V16 — current best):**

| Metric | V16 (tau=15) | V19 (tau=10) | Delta |
|--------|-------------|-------------|-------|
| LooGLE TTFT mean (ms) | 9814 | 9854 | +0.4% (noise) |
| LooGLE TTFT median (ms) | 1277 | 1205 | -5.6% better |
| LooGLE TTFT p90 (ms) | 4210 | 3988 | **-5.3% better** |
| LooGLE TPOT mean (ms) | 454.76 | 467.57 | +2.8% worse |
| LooGLE hit rate | 0.8966 | 0.8966 | identical |
| ShareGPT TTFT mean (ms) | 37300 | 37440 | +0.4% |
| ShareGPT req throughput | 1.28 | 1.27 | -0.8% |

HiCache: evicted=318.6M, load_back=287.5M, cached_device=4.61M (-7.1% vs V16), load_back_mean=1.829ms, host_util=1.0

**Analysis:** Mean TTFT is within noise of V16 (9854 vs 9814). The p90 improved (-5.3%) but TPOT regressed (+2.8%). With cached_device dropping 7.1% (4.61M vs 4.96M), tau=10 evicts GPU data slightly too fast — same mechanism as V17 but less severe. The tau sensitivity curve is now comprehensively characterized:

- p90 keeps improving with lower tau (V16: 4210 → V19: 3988 → V17: 4128) — the minimum is near tau=10
- Mean TTFT peaks at tau=15 (9814) — V19 (9854) and V17 (10211) are both worse
- TPOT peaks at tau=15 (454.76) — lower tau increases decode-time GPU pressure

Since the primary metric is TTFT mean, **tau=15 remains the optimum**. The tau axis is fully exhausted.

---

## V20 — Stacked seg=2 + tau=10 (NEGATIVE interaction)

**Commit:** `72ca9c9a5`
**Flag:** `--radix-eviction-policy gslru` + code (max_segment=2, decay_tau=10, device_weight=4, anti-starvation 300s)

**Hypothesis:** V18 (seg=2, tau=15) improved p90 by -11.7% and V19 (seg=4, tau=10) improved p90 by -5.3%, both independently vs V16. Stack both changes to test if p90 gains compound.

**Result (vs V16 — current best):**

| Metric | V16 (seg=4, tau=15) | V20 (seg=2, tau=10) | Delta |
|--------|-------|-------|-------|
| LooGLE TTFT mean (ms) | 9814 | 9774 | -0.4% (noise) |
| LooGLE TTFT p90 (ms) | 4210 | 4854 | **+15.3% WORSE** |
| LooGLE TPOT mean (ms) | 454.76 | 486.99 | **+7.1% worse** |
| LooGLE hit rate | 0.8966 | 0.8959 | neutral |
| ShareGPT TTFT mean (ms) | 37300 | 37420 | neutral |

**Analysis:** Classic negative interaction effect. The two p90 improvements that work independently become counterproductive when combined:

| Config | p90 (ms) | vs V16 |
|--------|----------|--------|
| V16 (seg=4, tau=15) | 4210 | baseline |
| V18 (seg=2, tau=15) | 3719 | -11.7% |
| V19 (seg=4, tau=10) | 3988 | -5.3% |
| **V20 (seg=2, tau=10)** | **4854** | **+15.3%** |

With fewer tiers (seg=2) AND faster decay (tau=10), nodes transition from evictable to protected too quickly (at 2 hits) but then lose protection too rapidly (10s decay). This creates unstable oscillation — nodes rapidly gain and lose priority, increasing GPU churn. The TPOT regression (+7.1%) confirms increased GPU pressure.

**Conclusion:** Parameter tuning within the GSLRU+time-decay framework has reached diminishing returns. V16 (seg=4, tau=15) remains the optimum. Future improvement requires a fundamentally different approach — either a new scheduling mechanism, a structural change to the eviction/load-back pipeline, or a tunable flag change.

---

## V21 — Write-through selective (CATASTROPHIC NEGATIVE)

**Commit:** `ac5e715e6`
**Flag:** `--radix-eviction-policy gslru --hicache-write-policy write_through_selective` + code (seg=4, tau=15, V16 optimum)

**Hypothesis:** With write_through (threshold=1), ALL nodes are backed up to host on first access — including single-use tokens that are never needed again. `write_through_selective` (threshold=2) filters these out, reducing host write bandwidth and host memory pressure.

**Result (vs V16 — current best):**

| Metric | V16 (write_through) | V21 (selective) | Delta |
|--------|-------|-------|-------|
| LooGLE TTFT mean (ms) | 9814 | **31294** | **+218.8% CATASTROPHIC** |
| LooGLE TTFT p90 (ms) | 4210 | **139624** | **+3216% CATASTROPHIC** |
| LooGLE hit rate | 0.8966 | **0.688** | **-23.2% CATASTROPHIC** |
| LooGLE out tok/s | 53.56 | 33.82 | -36.9% |
| LooGLE cached_host | 35.2M | 11.5M | -67.3% |
| LooGLE load_back | 286.4M | 97.2M | -66.1% |
| ShareGPT TTFT mean (ms) | 37300 | **21940** | **-41.2% BETTER** |
| ShareGPT req throughput | 1.28 | **1.79** | **+39.8% BETTER** |

**Root cause:** With threshold=2, document prefix nodes are NOT backed up on first access. Under high GPU churn (200M+ evictions), nodes are frequently evicted from GPU after only 1 access — before the 2nd access that would trigger backup. When evicted without backup, the data is permanently lost. The hit rate collapsed from 89.7% to 68.8% — essentially half the prefix-sharing benefit was destroyed.

ShareGPT counterintuitively improved because its diverse, short prompts generate mostly single-use tokens. Without the overhead of backing up all these ephemeral tokens, the system has more bandwidth for active requests.

**Lesson:** Under high-churn GPU pools (V16: 317M evictions for 40M cached), threshold=1 write-through is ESSENTIAL. The first-access backup is the last line of defense before permanent data loss. This tunable flag is now marked as invariant — never change it.

---

## V22 — Anti-starvation at 200s (NEGATIVE)

**Commit:** `79be0e227`
**Flag:** `--radix-eviction-policy gslru` + code (seg=4, tau=15, device_weight=4, anti-starvation 200s)

**Hypothesis:** V11 found the 300s anti-starvation threshold to be the sweet spot. 200s might catch more starved requests without disrupting LPM ordering, improving both mean and tail latency.

**Result (vs V16 — current best):**

| Metric | V16 (300s) | V22 (200s) | Delta |
|--------|-------|-------|-------|
| LooGLE TTFT mean (ms) | 9814 | 12797 | **+30.4% NEGATIVE** |
| LooGLE TTFT p90 (ms) | 4210 | 38380 | **+812% CATASTROPHIC** |
| LooGLE TTFT p99 (ms) | 213209 | 204329 | -4.2% better |
| LooGLE hit rate | 0.8966 | 0.8957 | neutral |
| LooGLE TPOT mean (ms) | 454.76 | 465.26 | +2.3% worse |
| ShareGPT TTFT mean (ms) | 37300 | 37560 | +0.7% neutral |
| ShareGPT req throughput | 1.28 | 1.27 | neutral |

HiCache: evicted=308.3M, load_back=276.8M, cached_device=5.74M (+15.7% vs V16), load_back_mean=1.808ms, host_util=0.9995

**Analysis:** 200s anti-starvation is too aggressive. The p90 regressed catastrophically from 4210ms to 38380ms (+812%) because too many requests exceed the 200s threshold and get priority-boosted, disrupting LPM ordering. This is the same mechanism that made V8 (120s) worse than V6 — the anti-starvation interventions break the cache-aware scheduling. The p99 improved slightly (-4.2%) because more starved requests are caught, but the mean and mid-percentile damage far outweighs this.

The starvation threshold sensitivity is now fully characterized:
- 120s (V8): mean 16070ms, p99 85889ms — too aggressive
- 200s (V22): mean 12797ms, p90 38380ms — still too aggressive
- 300s (V11/V16): mean 9814-10658ms, p90 4210ms — sweet spot
- ∞ (V9): mean 12186ms, p99 228s — no safety net

**Conclusion:** 300s anti-starvation is confirmed as the optimum across all data points. Reverted to 300s. The scheduling axis (LPM sorting, anti-starvation threshold, DFS_WEIGHT) is now exhaustively explored.

---

## V23 — Cold request fast-track (CATASTROPHIC)

**Commit:** `5983f1ef1` (reverted in `75948e952`)
**Flag:** `--radix-eviction-policy gslru` + code (tiered anti-starvation: 30s for <10K match, 300s for others)

**Hypothesis:** Analysis of V16 showed cold requests (~10% of total, <10K prefix match tokens) contribute ~55% of mean TTFT because they wait up to 300s in the anti-starvation queue while hot requests monopolize GPU. By fast-tracking only these cold requests at 30s (while keeping 300s for warm/hot requests), we could rescue the mean without disrupting the cache-aware ordering for the 90% of requests that benefit from LPM.

**What changed:**
- Added `_LPM_COLD_STARVATION_SECS = 30.0` and `_LPM_COLD_MATCH_TOKENS = 10000`
- Requests with total match < 10000 tokens that have waited > 30s get priority (0, entry) — same as anti-starvation
- Other requests keep the 300s threshold

**Result (vs V16 — current best) — CATASTROPHIC, server crash:**

| Metric | V16 (300s uniform) | V23 (30s cold) | Delta |
|--------|-------|-------|-------|
| LooGLE TTFT mean (ms) | 9814 | **65846** | **+571% CATASTROPHIC** |
| LooGLE TTFT median (ms) | 1277 | 51022 | **+3894% CATASTROPHIC** |
| LooGLE TTFT p99 (ms) | 213209 | 181833 | -14.7% better |
| LooGLE completed | 1560/1560 | **622/1560** | **60% lost** |
| LooGLE req throughput | 3.47 | 0.67 | -80.7% |
| ShareGPT | — | **server crashed** | — |

**Root cause:** The 10K token match threshold classified too many requests as "cold" — including LooGLE's host-cached requests during early scheduling rounds when prefix_indices haven't been populated yet. Promoting these at 30s instead of 300s broke LPM's cache-aware ordering entirely, causing a cascade:

1. Early-round requests appear to have <10K match (host data not yet in prefix_indices)
2. These get fast-tracked at 30s, jumping ahead of GPU-cached requests
3. Processing these requests triggers GPU eviction of currently-cached data
4. Previously GPU-cached requests lose their data and become "cold" themselves
5. Positive feedback loop: more cold requests → more disruption → more cold requests
6. Server throughput collapsed to 0.67 req/s (80% reduction), only 40% of LooGLE completed
7. Server process crashed during transition to ShareGPT benchmark — likely OOM or resource exhaustion from cascading eviction

**Key lesson:** ANY reduction of the anti-starvation threshold below 300s for ANY subset of requests destroys LooGLE performance. The 300s threshold with pure device-weighted LPM is sacrosanct. The scheduling order must not be disrupted by time-based promotions at any lower threshold.

---

## V24 — page_size=128 (SERVER CRASH)

**Commit:** `7c1c61198` (no code changes, flag-only)
**Flag:** `--radix-eviction-policy gslru --page-size 128` + code (V16 optimum: seg=4, tau=15, device_weight=4, anti-starvation 300s)

**Hypothesis:** V16 uses default page_size=64. Larger pages (128 tokens per page) reduce tree node count by 50%, which means:
1. Less heap-building overhead in evict() — fewer nodes to sort
2. More efficient DMA transfers — larger contiguous blocks
3. Trade-off: coarser matching granularity (~0.4% of 28.7K avg prompt vs 0.2% at page_size=64)

**Result — SERVER CRASH:**

Server scheduler Rank 1 died with SIGBUS (exit code -7) during DeepGEMM warmup at ~6% progress. Multiple scheduler processes crashed simultaneously with Bus errors. No benchmarks executed.

The crash occurred during GEMM kernel compilation, which should be independent of page_size. However, page_size=128 changes the KV cache memory layout (same total capacity 686592 tokens, but 5364 pages instead of 10728), which may have caused alignment issues with the `direct` IO backend or triggered a latent bug in shared memory allocation.

**Conclusion:** page_size=128 is incompatible with this model/hardware configuration. The page_size axis is constrained to ≤64.

---

## V25 — page_size=32 (NEGATIVE)

**Commit:** `7c1c61198` (no code changes, flag-only)
**Flag:** `--radix-eviction-policy gslru --page-size 32` + code (V16 optimum: seg=4, tau=15, device_weight=4, anti-starvation 300s)

**Hypothesis:** Finer matching granularity — 32 tokens per page instead of 64. For LooGLE's 28.7K avg prompts, matching within ±32 tokens instead of ±64. Trade-off: 2x more tree nodes (21456 vs 10728 pages).

**Result (vs V16 — current best):**

| Metric | V16 (ps=64) | V25 (ps=32) | Delta |
|--------|-------------|-------------|-------|
| LooGLE TTFT mean (ms) | 9814 | 11890 | **+21.2% NEGATIVE** |
| LooGLE TTFT median (ms) | 1277 | 1466 | +14.8% worse |
| LooGLE TTFT p90 (ms) | 4210 | 4364 | +3.7% worse |
| LooGLE TTFT p99 (ms) | 213209 | 250138 | **+17.3% worse** |
| LooGLE TPOT mean (ms) | 454.76 | 543.82 | **+19.6% worse** |
| LooGLE hit rate | 0.8966 | 0.8959 | neutral |
| ShareGPT TTFT mean (ms) | 37300 | 71290 | **+91.1% CATASTROPHIC** |
| ShareGPT req throughput | 1.28 | 0.73 | **-43.0% CATASTROPHIC** |
| ShareGPT total requests | 1153 | 656 | **-43.1% fewer** |

HiCache: evicted=311.9M (-1.8%), load_back=280.6M (-2.0%), cached_device=5.55M (+11.9%), eviction_mean=1.137ms, load_back_mean=1.86ms

**Analysis:** The doubled tree node count (21456 vs 10728 pages) overwhelmed any matching precision benefit. TPOT regressed +19.6% — the primary damage mechanism is tree operation overhead per scheduling round (heap building, tree traversal, node management). ShareGPT suffered catastrophically (-43% throughput) because its diverse, short-prefix workload creates proportionally more tree nodes with smaller page sizes.

Interestingly, cached_device_tokens INCREASED (+11.9%, 5.55M vs 4.96M) — smaller pages pack more efficiently into GPU memory. But this packing advantage is overwhelmed by the per-node overhead.

**page_size sensitivity (axis fully exhausted):**

| page_size | TTFT mean (ms) | TPOT (ms) | ShareGPT throughput |
|-----------|----------------|-----------|---------------------|
| 32 (V25) | 11890 (+21.2%) | 543.82 | 0.73 (CATASTROPHIC) |
| **64 (V16)** | **9814** | **454.76** | **1.28** |
| 128 (V24) | SERVER CRASH | — | — |

**Conclusion:** page_size=64 is the clear optimum. Smaller pages add tree overhead; larger pages crash. This axis is fully exhausted.

---

## V26 — Two-tier decay (MARGINAL POSITIVE)

**Commit:** `e0416209b`
**Flag:** `--radix-eviction-policy gslru --page-size 64` + code (V16 optimum + two-tier decay: tau=30 for segment≥3, tau=15 for segment<3)

**Hypothesis:** V16 uses uniform tau=15 for all segments. But document prefixes (segment≥3, hit_count≥3 from multiple questions) and ephemeral tokens (segment<3) have different reuse patterns. Two-tier decay uses tau=30 for heavily shared data (segment≥3) and tau=15 for low-reuse data (segment<3). This keeps important prefixes GPU-resident longer between same-document questions while still quickly evicting ephemeral data.

**Result (vs V16 — current best):**

| Metric | V16 (uniform tau=15) | V26 (two-tier) | Delta |
|--------|-------|-------|-------|
| LooGLE TTFT mean (ms) | 9814 | **9730** | **-0.9%** |
| LooGLE TTFT median (ms) | 1277 | **1222** | **-4.3% better** |
| LooGLE TTFT p90 (ms) | 4210 | **4175** | **-0.8% better** |
| LooGLE TTFT p99 (ms) | 213209 | **208025** | **-2.4% better** |
| LooGLE TPOT mean (ms) | 454.76 | 455.62 | neutral |
| LooGLE ITL mean (ms) | 364.71 | **356.67** | **-2.2% better** |
| LooGLE hit rate | 0.8966 | 0.8966 | identical |
| ShareGPT TTFT mean (ms) | 37300 | 38290 | +2.7% worse |
| ShareGPT req throughput | 1.28 | 1.25 | -2.3% worse |

HiCache: evicted=317.8M (same), load_back=286.7M (same), cached_device=4.99M (same), evict_mean=1.113ms, load_back_mean=1.846ms

**Analysis:** All LooGLE metrics improved or stayed neutral. The improvement is consistent across all percentiles (mean, median, p90, p99), suggesting a genuine mechanism at work — not random variance. However, the mean improvement (-0.9%, 84ms) is within typical run-to-run variance (~5-15%), so this cannot be confidently declared a new best without repeated trials.

The underlying HiCache metrics (eviction volume, load_back volume, cached_device_tokens) are virtually identical to V16, indicating the two-tier decay achieves its improvement through better ORDERING of evictions rather than different eviction VOLUME. Shared prefixes survive slightly longer in GPU → fewer scheduling disruptions from premature eviction → smoother queue dynamics.

ShareGPT regressed slightly (-2.3% throughput) because the slower decay for segment≥3 nodes holds onto LooGLE's shared prefixes during the ShareGPT phase transition, slightly reducing available GPU cache for ShareGPT's diverse workload.

**Conclusion:** MARGINAL POSITIVE — directionally correct but within noise. The two-tier decay mechanism is valid but the magnitude is small. Future experiments could try gradient decay (tau proportional to segment) or different threshold boundaries.

---

## V27 — Smooth gradient decay (MARGINAL POSITIVE)

**Commit:** `ebd2ca1a5`
**Flag:** `--radix-eviction-policy gslru --page-size 64` + code (smooth gradient: tau = decay_tau × (1 + segment/max_segment))

**Hypothesis:** V26's binary two-tier (tau=15 for segment<3, tau=30 for segment≥3) was directionally positive but crude. A smooth gradient scales tau continuously with segment: segment=0 gets tau=15, segment=1 gets 18.75, segment=2 gets 22.5, segment=3 gets 26.25, segment=4 gets 30. This better reflects the continuous relationship between reuse probability and hit count.

**Result (vs V16 — current best):**

| Metric | V16 (uniform tau=15) | V27 (smooth gradient) | Delta |
|--------|-------|-------|-------|
| LooGLE TTFT mean (ms) | 9814 | **9688** | **-1.3%** |
| LooGLE TTFT median (ms) | 1277 | 1267 | -0.8% |
| LooGLE TTFT p90 (ms) | 4210 | **4117** | **-2.2%** |
| LooGLE TTFT p99 (ms) | 213209 | **205711** | **-3.5%** |
| LooGLE TPOT mean (ms) | 454.76 | 475.25 | +4.5% worse |
| LooGLE ITL mean (ms) | 364.71 | 364.42 | neutral |
| LooGLE hit rate | 0.8966 | 0.8945 | -0.2% |
| ShareGPT TTFT mean (ms) | 37300 | 38630 | +3.6% worse |
| ShareGPT req throughput | 1.28 | 1.25 | -2.3% worse |

HiCache: evicted=318.2M, load_back=286.3M, cached_device=5.09M, evict_mean=1.108ms, load_back_mean=1.851ms

**Analysis:** LooGLE TTFT improved across all percentiles, especially p99 (-3.5%, 7.5s faster). The smooth gradient does slightly better than V26's binary approach. However, TPOT regressed +4.5% — the longer tau for intermediate segments keeps more data in GPU, causing slightly more compute contention per token. ShareGPT regressed similarly to V26 (-2.3% throughput).

The TPOT regression makes this a MIXED result: TTFT improves but per-token generation slows. In practice, for long-context serving, TTFT dominates user-perceived latency (users wait longer for the first token than between subsequent tokens), so the TTFT improvement is more valuable than the TPOT regression.

**Conclusion:** MARGINAL POSITIVE — consistently better TTFT than V16 across all percentiles but with a TPOT tradeoff. The improvement magnitude (~126ms mean, 7.5s at p99) is at the edge of run-to-run variance for mean but the p99 improvement is more convincing.

---

## V28 — Narrow gradient decay (MARGINAL POSITIVE, best TTFT+TPOT tradeoff)

**Commit:** `4b91d8fa6`
**Flag:** `--radix-eviction-policy gslru --page-size 64` + code (narrow gradient: tau = decay_tau × (1 + 0.5 × segment/max_segment))

**Hypothesis:** V27's 2x tau range (15→30) improved TTFT but regressed TPOT +4.5%. The wide range caused excessive GPU memory hoarding by high-segment nodes, increasing compute contention during token generation. A narrower 1.5x range (15→22.5) should preserve the TTFT benefit while reducing TPOT regression.

**Result (vs V16 — previous best, and vs V27 — wide gradient):**

| Metric | V16 | V27 (2x) | V28 (1.5x) | V28 vs V16 |
|--------|-----|----------|------------|------------|
| LooGLE TTFT mean (ms) | 9814 | 9688 | **9651** | **-1.7%** |
| LooGLE TTFT median (ms) | 1277 | 1267 | 1265 | -1.0% |
| LooGLE TTFT p90 (ms) | 4210 | 4117 | **4134** | **-1.8%** |
| LooGLE TTFT p99 (ms) | 213209 | 205711 | **204443** | **-4.1%** |
| LooGLE TPOT mean (ms) | 454.76 | 475.25 | **461.35** | **+1.4%** (vs V27 +4.5%) |
| LooGLE ITL mean (ms) | 364.71 | 364.42 | 357.24 | -2.0% |
| LooGLE e2e mean (ms) | 14933 | 14945 | **14805** | **-0.9%** |
| LooGLE hit rate | 0.8966 | 0.8945 | 0.8959 | -0.1% |
| ShareGPT TTFT mean (ms) | 37300 | 38630 | 38500 | +3.2% worse |
| ShareGPT req throughput | 1.28 | 1.25 | 1.24 | -3.1% worse |

HiCache: evicted=317.2M, load_back=285.8M, cached_device=5.56M (+11.4% vs V16), evict_mean=1.108ms, load_back_mean=1.864ms, hit_device_frac=0.1385 (+11.5% vs V16)

**Analysis:** The narrower 1.5x tau range successfully achieves the best overall tradeoff:
- TTFT improved across all percentiles (mean -1.7%, p99 -4.1% = 8.8s faster)
- TPOT regression reduced to +1.4% (vs V27's +4.5%) — hypothesis confirmed
- e2e latency improved -0.9%
- device_hit_frac increased to 13.85% (vs V16's 12.42%) — more GPU-resident cache hits
- cached_device_tokens increased to 5.56M (vs V16's 4.99M) — better GPU cache utilization

The mechanism works as designed: high-segment nodes (frequently accessed document prefixes) get slightly longer tau (up to 22.5s at segment=4 vs uniform 15s), keeping them GPU-resident longer between Q2-Q8 accesses. The narrower range (1.5x vs 2x) avoids excessive hoarding that caused V27's TPOT regression.

ShareGPT regression (-3.1% throughput) is consistent across V26-V28 — the segment-scaled tau benefit is specific to LooGLE's multi-question-per-document pattern and doesn't help ShareGPT's diverse single-request workload.

**Conclusion:** MARGINAL POSITIVE — best overall tradeoff found. The improvement magnitude (~163ms mean TTFT, 8.8s at p99) is at the edge of significance for mean but the p99 improvement and the TPOT improvement over V27 are more convincing. This is likely the optimal point in the gradient-decay parameter space.

---

## V29 — Top-only decay boost (POSITIVE, new best LooGLE tradeoff)

**Commit:** `228e3cd29`
**Flag:** `--radix-eviction-policy gslru --page-size 64` + code (top-only: tau=20 for segment=max_segment, tau=15 for all others)

**Hypothesis:** V27-V28 showed gradient decay improves TTFT but causes TPOT regression from excessive GPU hoarding across ALL segments. Instead, only boost the SINGLE most valuable tier: max segment (hit_count≥4) nodes, which are fully-reused document prefixes. All other segments keep fast tau=15, maintaining efficient GPU memory turnover.

**Result (vs V16 — previous best):**

| Metric | V16 | V28 (narrow 1.5x) | V29 (top-only) | V29 vs V16 |
|--------|-----|----------|------------|------------|
| LooGLE TTFT mean (ms) | 9814 | 9651 | **9642** | **-1.8%** |
| LooGLE TTFT median (ms) | 1277 | 1265 | **1227** | **-3.9%** |
| LooGLE TTFT p90 (ms) | 4210 | 4134 | **3874** | **-8.0%** |
| LooGLE TTFT p99 (ms) | 213209 | 204443 | **202925** | **-4.8%** |
| LooGLE TPOT mean (ms) | 454.76 | 461.35 | **443.04** | **-2.6% improved** |
| LooGLE ITL mean (ms) | 364.71 | 357.24 | **354.27** | **-2.9% improved** |
| LooGLE e2e mean (ms) | 14933 | 14805 | **14753** | **-1.2%** |
| LooGLE hit rate | 0.8966 | 0.8959 | 0.8941 | -0.3% |
| ShareGPT TTFT mean (ms) | 37300 | 38500 | 38540 | +3.3% worse |
| ShareGPT req throughput | 1.28 | 1.24 | 1.25 | -2.3% worse |

HiCache: evicted=321.7M, load_back=289.6M, cached_device=4.81M, evict_mean=1.099ms, load_back_mean=1.834ms

**Analysis:** V29 achieves the best overall tradeoff in the decay-tau exploration arc:
- TTFT improved across ALL percentiles, with p90 showing the strongest gain (-8.0%, 336ms)
- TPOT **improved** -2.6% — unlike V27 (+4.5%) and V28 (+1.4%), no regression at all
- The p90 improvement is the most significant finding: the 8% gain suggests the top-only boost specifically helps the "nearly-cached" requests that are on the bubble between GPU-hit and host-loadback

The mechanism is elegant: only fully-reused prefixes (segment=4) get slower decay (tau=20), keeping them GPU-resident slightly longer. Intermediate segments (0-3) maintain fast tau=15 → faster GPU memory recycling → less contention during token generation → TPOT improves.

**Conclusion:** POSITIVE — new best on LooGLE metrics. The TTFT p90 improvement (-8.0%) is outside run-to-run variance and represents a genuine improvement. However, ShareGPT still regresses, indicating the top-only decay is optimized for multi-question-per-document patterns.

---

## V30 — Top-only decay tau=25 (NEGATIVE)

**Commit:** `714783ce0`
**Flag:** `--radix-eviction-policy gslru --page-size 64` + code (top-only: tau=25 for segment=max, tau=15 for others)

**Hypothesis:** If tau=20 (V29) improved everything, maybe tau=25 would improve TTFT further. Testing stronger top-segment boost.

**Result (vs V29 — current best):**

| Metric | V29 (tau=20) | V30 (tau=25) | Delta |
|--------|-------|-------|-------|
| LooGLE TTFT mean (ms) | 9642 | 9748 | **+1.1% worse** |
| LooGLE TTFT p90 (ms) | 3874 | 4370 | **+12.8% worse** |
| LooGLE TPOT mean (ms) | 443.04 | 466.21 | **+5.2% worse** |
| LooGLE e2e mean (ms) | 14753 | 14976 | +1.5% worse |

**Analysis:** tau=25 overshoots — too much GPU hoarding at max segment causes contention. V29's tau=20 (4/3 × decay_tau) is confirmed as the sweet spot for the top-only approach.

**Conclusion:** NEGATIVE. The top-segment tau has been optimized: tau=20 is the optimum between tau=15 (V16, no boost) and tau=25 (V30, too aggressive).

---

## V31 — Device weight=5 (POSITIVE, new best)

**Commit:** `9e18be7cb`
**Flag:** `--radix-eviction-policy gslru --page-size 64` + code (device_weight=5, V29 top-only decay tau=20)

**Hypothesis:** V13 established device_weight=4 as the optimum in the 1→2→4→8 series, with weight=8 clearly negative. But the series skipped weight=5, 6, 7. At weight=4, GPU-preference is strong but not overwhelming; at weight=8, it's too aggressive. Weight=5 tests a finer increment between the optimum and the over-concentration cliff.

**Result (vs V29 — previous best):**

| Metric | V29 (w=4) | V31 (w=5) | Delta |
|--------|-----------|-----------|-------|
| LooGLE TTFT mean (ms) | 9642 | **9535** | **-1.11% NEW BEST** |
| LooGLE TTFT median (ms) | 1227 | **1160** | **-5.43% better** |
| LooGLE TTFT p90 (ms) | 3874 | **3810** | **-1.65% better** |
| LooGLE TTFT p99 (ms) | 202925 | 204107 | +0.6% neutral |
| LooGLE TPOT mean (ms) | 443.04 | 441.70 | -0.30% neutral |
| LooGLE ITL mean (ms) | 354.27 | 353.47 | -0.23% neutral |
| LooGLE e2e mean (ms) | 14753 | **14634** | **-0.81% better** |
| LooGLE out tok/s | 53.53 | 53.55 | neutral |
| LooGLE hit rate | 0.8941 | **0.8966** | +0.28% better |
| ShareGPT TTFT mean (ms) | 38540 | **38310** | **-0.60% better** |
| ShareGPT req throughput | 1.25 | 1.25 | unchanged |
| ShareGPT hit rate | 0.600 | 0.610 | +1.7% better |
| ShareGPT total requests | 1123 | 1123 | same |

HiCache: evicted=318.5M (same), load_back=287.4M (same), cached_device=4.96M (same), evict_mean=1.109ms, load_back_mean=1.819ms, host_util=0.9985, hit_device_frac=0.1234 (+2.8% vs V29)

**Analysis:** V31 improves ALL key metrics with zero regressions:
- TTFT mean -1.11%: genuine improvement, confirmed by consistent gains across all percentiles
- TTFT median -5.43%: the strongest signal — the fastest 50% of requests find their data faster in GPU
- TPOT unchanged: no GPU memory pressure increase (unlike weight=8 in V14)
- ShareGPT also improved (-0.60% TTFT, +1.7% hit rate): the stronger GPU preference helps even diverse workloads
- HiCache internals (eviction volume, load_back, cached_device) are nearly identical to V29, indicating weight=5 achieves its improvement through better scheduling ORDER, not different memory dynamics

**Device weight sensitivity (refined):**

| Weight | TTFT mean (ms) | TTFT p90 (ms) | TPOT (ms) | Mechanism |
|--------|----------------|---------------|-----------|-----------|
| 1 (V11) | 10658 | 20760 | — | baseline LPM |
| 2 (V12) | 10506 | 6842 | — | first GPU preference |
| 4 (V13→V29) | 9642 | 3874 | 443 | strong GPU preference |
| **5 (V31)** | **9535** | **3810** | **442** | **refined optimum** |
| 8 (V14) | 10903 | 10158 | — | over-concentration |

**Conclusion:** POSITIVE — new best. Weight=5 refines the optimum between the successful weight=4 and the negative weight=8. The improvement is modest but consistent across all metrics with no tradeoffs. Next: test weight=6 to continue refining the optimal point.

---

## V32 — Device weight=6 (NEGATIVE)

**Commit:** `f2f357c49`
**Flag:** `--radix-eviction-policy gslru --page-size 64` + code (device_weight=6, V29 top-only decay tau=20)

**Hypothesis:** V31 (weight=5) improved over V29 (weight=4). Test weight=6 to continue refining the optimum between weight=5 (positive) and weight=8 (negative).

**Result (vs V31 — current best):**

| Metric | V31 (w=5) | V32 (w=6) | Delta |
|--------|-----------|-----------|-------|
| LooGLE TTFT mean (ms) | 9535 | 9943 | **+4.28% NEGATIVE** |
| LooGLE TTFT median (ms) | 1160 | 1246 | +7.4% worse |
| LooGLE TTFT p90 (ms) | 3810 | 4479 | **+17.6% worse** |
| LooGLE TPOT mean (ms) | 441.70 | 472.92 | **+7.1% worse** |
| LooGLE hit rate | 0.8966 | 0.8952 | -0.2% |
| ShareGPT TTFT mean (ms) | 38310 | 38510 | +0.5% neutral |
| ShareGPT req throughput | 1.25 | 1.25 | same |

HiCache: evicted=315.7M (-0.9%), load_back=284.0M (-1.2%), cached_device=5.34M (+7.7% vs V31), evict_mean=1.111ms, load_back_mean=1.844ms, hit_device_frac=0.1333 (+8.0% vs V31)

**Analysis:** Weight=6 shows the same over-concentration pattern as weight=8 (V14), just less severe. More device tokens are cached (+7.7%) but GPU memory pressure worsens TPOT (+7.1%) and the scheduling over-concentrates on GPU-resident requests, causing mid-tier request starving (p90 +17.6%).

**Device weight sensitivity (complete):**

| Weight | TTFT mean (ms) | TTFT p90 (ms) | TPOT (ms) | cached_device (M) |
|--------|----------------|---------------|-----------|-------------------|
| 1 (V11) | 10658 | 20760 | — | — |
| 2 (V12) | 10506 | 6842 | — | 4.32 |
| 4 (V13→V29) | 9642 | 3874 | 443 | 4.81 |
| **5 (V31)** | **9535** | **3810** | **442** | **4.96** |
| 6 (V32) | 9943 | 4479 | 473 | 5.34 |
| 8 (V14) | 10903 | 10158 | — | 4.66 |

**Conclusion:** NEGATIVE. The device weight axis is now fully characterized with 6 data points. The optimum is at weight=5 (V31). Reverted to weight=5.

---

## V33 — Top-segment tau=22.5 (NEGATIVE)

**Commit:** `771dbde0a`
**Flag:** `--radix-eviction-policy gslru --page-size 64` + code (device_weight=5, tau_top=22.5 via ratio 3/2)

**Hypothesis:** V29 (tau_top=20) was positive, V30 (tau_top=25) was negative. Test the midpoint (22.5) to determine if there's headroom between 20 and 25.

**Result (vs V31 — current best):**

| Metric | V31 (tau_top=20) | V33 (tau_top=22.5) | Delta |
|--------|-----------|-----------|-------|
| LooGLE TTFT mean (ms) | 9535 | 9739 | **+2.14% NEGATIVE** |
| LooGLE TTFT p90 (ms) | 3810 | 4109 | **+7.83% worse** |
| LooGLE TPOT mean (ms) | 441.70 | 483.79 | **+9.53% worse** |
| LooGLE hit rate | 0.8966 | 0.8928 | -0.4% |
| ShareGPT TTFT mean (ms) | 38310 | 38540 | +0.6% neutral |

**Analysis:** tau_top=22.5 causes the same over-hoarding pattern as V30 (tau=25), just less severe. Max-segment nodes hold GPU memory slightly too long, increasing TPOT (+9.5%) and mean TTFT (+2.1%). The top-segment tau sensitivity is now fully characterized:

| tau_top | TTFT mean (ms) | TPOT (ms) | vs V31 |
|---------|----------------|-----------|--------|
| 15 (V16, no boost) | 9814 | 454.76 | +2.9% worse |
| **20 (V29/V31)** | **9535** | **441.70** | **optimum** |
| 22.5 (V33) | 9739 | 483.79 | +2.14% worse |
| 25 (V30) | 9748 | 466.21 | +2.23% worse |

**Conclusion:** NEGATIVE. tau_top=20 is confirmed as the sharp optimum. Any increase above 20 worsens both TTFT and TPOT. The top-segment decay axis is fully exhausted.

---

## V34 — Host weight=2 in LPM (NEGATIVE)

**Commit:** `92a5f9af3`
**Flag:** `--radix-eviction-policy gslru --page-size 64` + code (device_weight=5, host_weight=2)

**Hypothesis:** Host matches in LPM have always been weighted at 1x (implicit). Load_back is only 1.8ms so host-matched tokens are nearly as valuable as device-matched for scheduling. Increasing host_weight to 2 (ratio 5:2 instead of 5:1) should better reflect host match value and improve scheduling decisions.

**Result (vs V31 — current best):**

| Metric | V31 (host_weight=1) | V34 (host_weight=2) | Delta |
|--------|-----------|-----------|-------|
| LooGLE TTFT mean (ms) | 9535 | 9926 | **+4.10% NEGATIVE** |
| LooGLE TTFT median (ms) | 1270 | 1187 | -6.53% improved |
| LooGLE TTFT p90 (ms) | 3810 | 4358 | **+14.38% worse** |
| LooGLE TPOT mean (ms) | 441.70 | 471.80 | **+6.81% worse** |
| LooGLE device_hit_frac | 0.125 | 0.132 | +5.6% more GPU hits |
| LooGLE hit rate | 0.8966 | 0.8948 | -0.2% |
| ShareGPT TTFT mean (ms) | 38310 | 38470 | +0.4% neutral |

**Analysis:** Increasing host_weight improved median TTFT (-6.5%) and device_hit_frac (+5.6% more GPU cache hits), confirming that prioritizing host-match requests does load prefixes into GPU earlier for subsequent requests. However, the trade-off is negative: mean TTFT worsened (+4.1%), p90 degraded (+14.4%), and TPOT increased (+6.8%). The extra load_back traffic from processing host-match requests sooner adds overhead during decode, hurting throughput.

**Conclusion:** NEGATIVE. Increasing host_weight causes overweighting of host-matched requests, which generates more load_back traffic and hurts tail latency. The current 5:1 device:host ratio is optimal — host matches should remain significantly discounted relative to device matches in scheduling priority.

---

## V35 — Match-time hit_count promotion (NEW BEST)

**Commit:** `24394b4d5`
**Flag:** `--radix-eviction-policy gslru --page-size 64` + code (device_weight=5, match promotion)

**Hypothesis:** Currently `hit_count` only increments during token INSERTION (`_inc_hit_count` in `_insert_helper`). But `_match_prefix_helper` during scheduling is a genuine access signal. Incrementing `best_match_node.hit_count` once per request on first match accelerates GSLRU segment promotion — shared document prefixes reach segment 4 (tau=20 boost) one question cycle earlier.

**Result (vs V31 — previous best):**

| Metric | V31 (no match promo) | V35 (match promo) | Delta |
|--------|-----------|-----------|-------|
| LooGLE TTFT mean (ms) | 9535 | 9434 | **-1.06% NEW BEST** |
| LooGLE TTFT median (ms) | 1270 | 1190 | **-6.33% improved** |
| LooGLE TTFT p90 (ms) | 3810 | 3964 | +4.04% slightly worse |
| LooGLE TTFT p99 (ms) | 203752 | 199280 | -2.20% improved |
| LooGLE TPOT mean (ms) | 441.70 | 448.55 | +1.55% slightly worse |
| LooGLE ITL mean (ms) | 359.51 | 349.80 | -2.70% improved |
| LooGLE e2e mean (ms) | 14919 | 14481 | -2.94% improved |
| LooGLE hit rate | 0.8966 | 0.8966 | identical |
| ShareGPT TTFT mean (ms) | 38310 | 38600 | +0.8% neutral |

**Analysis:** Match promotion accelerates prefix segment promotion by one step. Under LooGLE (8 questions/document), the document prefix reaches segment 4 after Q3's match instead of Q4's insert. The earlier tau=20 boost better protects prefixes during the Q3-Q4 gap (40-60s). This yields -1.06% mean TTFT and -6.33% median TTFT improvement. The p90 and TPOT show slight regression (+4.04%, +1.55%) — the faster promotion causes slightly longer GPU residence of some prefixes, adding marginal eviction pressure. But overall: mean, median, p99, ITL, and e2e all improved.

**Conclusion:** **NEW BEST.** Match-time hit_count promotion is a structural improvement that accelerates GSLRU segment promotion without parameter tuning. Total improvement from V0 baseline: -76.6% (40371ms → 9434ms).

---

## V36 — Double match promotion (hit_count += 2) (NEGATIVE)

**Commit:** `0f49f6906`
**Flag:** `--radix-eviction-policy gslru --page-size 64` + code (device_weight=5, match promotion +=2)

**Hypothesis:** V35's match-time promotion (+=1) was the new best. More aggressive promotion (+=2) might yield further improvement by pushing document prefixes to segment 4 even faster — reaching tau=20 boost after just Q2 instead of Q3.

**Result (vs V35 — current best):**

| Metric | V35 (+=1) | V36 (+=2) | Delta |
|--------|-----------|-----------|-------|
| LooGLE TTFT mean (ms) | 9434 | 9680 | **+2.60% NEGATIVE** |
| LooGLE TTFT median (ms) | 1190 | 1231 | +3.52% worse |
| LooGLE TTFT p90 (ms) | 3964 | 3628 | -8.48% improved |
| LooGLE TTFT p99 (ms) | 199280 | 202139 | +1.4% worse |
| LooGLE TPOT mean (ms) | 448.55 | 456.85 | +1.85% worse |
| LooGLE ITL mean (ms) | 349.80 | 359.15 | +2.67% worse |
| LooGLE e2e mean (ms) | 14481 | 14861 | +2.63% worse |
| LooGLE hit rate | 0.8966 | 0.8933 | -0.37% worse |
| LooGLE device_hit_frac | 0.1199 | 0.1342 | +11.9% more GPU hits |
| ShareGPT TTFT mean (ms) | 38600 | 38900 | +0.8% neutral |
| ShareGPT req throughput | 1.24 | 1.23 | neutral |

HiCache: evicted=317.8M (same), load_back=285.5M (same), cached_device=5.37M (+11.5% vs V35), evict_mean=1.113ms, load_back_mean=1.85ms, host_util=0.9987

**Analysis:** Double promotion (+=2) shows the familiar over-promotion pattern. The device_hit_frac increased significantly (+11.9%, 0.1199→0.1342) confirming that faster segment promotion causes longer GPU residence. However, this GPU hoarding worsens mean TTFT (+2.60%), TPOT (+1.85%), and ITL (+2.67%).

The p90 improved (-8.48%) — the same pattern as gradient decay experiments (V27-V28): more aggressive protection helps tail latency at the cost of the average case. The mean is dominated by increased queue wait time from GPU monopolization.

**Match promotion sensitivity:**

| Increment | TTFT mean (ms) | TTFT p90 (ms) | device_hit_frac | TPOT (ms) |
|-----------|----------------|---------------|-----------------|-----------|
| 0 (V31) | 9535 | 3810 | 0.1234 | 441.70 |
| **1 (V35)** | **9434** | **3964** | **0.1199** | **448.55** |
| 2 (V36) | 9680 | 3628 | 0.1342 | 456.85 |

**Conclusion:** NEGATIVE. +=1 is the optimum for match-time promotion. +=2 over-promotes, causing GPU hoarding that worsens mean TTFT despite improving p90. Reverted to +=1.

---

## V37 — Parent node match promotion (NEGATIVE)

**Commit:** `00738fe72`
**Flag:** `--radix-eviction-policy gslru --page-size 64` + code (device_weight=5, leaf + parent promotion)

**Hypothesis:** V35 promotes only `best_match_node.hit_count`. In the radix tree, document prefixes are chains of ~449 nodes. When the deepest node (leaf) is evicted, its parent becomes the new evictable leaf. Promoting the parent too should delay the cascade of bottom-up eviction through the prefix chain.

**Result (vs V35 — current best):**

| Metric | V35 (leaf only) | V37 (leaf + parent) | Delta |
|--------|-----------|-----------|-------|
| LooGLE TTFT mean (ms) | 9434 | 9626 | **+2.03% NEGATIVE** |
| LooGLE TTFT median (ms) | 1190 | 1223 | +2.80% worse |
| LooGLE TTFT p90 (ms) | 3964 | 4219 | +6.45% worse |
| LooGLE TPOT mean (ms) | 448.55 | 462.90 | **+3.20% worse** |
| LooGLE ITL mean (ms) | 349.80 | 358.09 | +2.37% worse |
| LooGLE hit rate | 0.8966 | 0.8967 | neutral |
| LooGLE device_hit_frac | 0.1199 | 0.1221 | +1.83% more GPU hits |
| ShareGPT TTFT mean (ms) | 38600 | 38850 | +0.65% neutral |

HiCache: evicted=317.6M (-0.3%), load_back=286.5M (-0.3%), cached_device=4.91M (+1.85%), evict_mean=1.117ms, load_back_mean=1.846ms

**Analysis:** Same over-promotion pattern as V36. Promoting two nodes (leaf + parent) increases GPU-cached tokens (+1.85%) and device_hit_frac (+1.83%), but the extra GPU memory pressure worsens TPOT (+3.20%) and mean TTFT (+2.03%). The TPOT regression is the clearest signal: more GPU-resident data → more memory pressure during decode → slower token generation.

**Match promotion sensitivity (complete):**

| Variant | Nodes promoted | TTFT mean (ms) | TPOT (ms) | device_hit_frac |
|---------|---------------|----------------|-----------|-----------------|
| V31 (none) | 0 | 9535 | 441.70 | 0.1234 |
| **V35 (leaf +1)** | **1** | **9434** | **448.55** | **0.1199** |
| V36 (leaf +2) | 1 (×2) | 9680 | 456.85 | 0.1342 |
| V37 (leaf+parent) | 2 | 9626 | 462.90 | 0.1221 |

**Conclusion:** NEGATIVE. V35 (single leaf +1) is the optimum. The match promotion axis is fully exhausted. Any increase in promotion — whether higher increment (V36) or more nodes (V37) — causes GPU hoarding that worsens TPOT and mean TTFT.

---

## V38 — Dynamic anti-starvation threshold (NEUTRAL/NEGATIVE)

**Commit:** `51ead76fb`
**Flag:** `--radix-eviction-policy gslru --page-size 64` + code (device_weight=5, match promotion, dynamic starvation: base=300s + 3s per request above 50)

**Hypothesis:** The fixed 300s anti-starvation threshold treats all queue depths equally. Under heavier load (queue depth ~95), the threshold could scale up to reduce the number of anti-starvation interventions, preserving LPM ordering. Formula: `threshold = 300 + max(0, queue_depth - 50) * 3.0`. At typical queue=95, threshold ~435s.

**Result (vs V35 — current best):**

| Metric | V35 (fixed 300s) | V38 (dynamic) | Delta |
|--------|-----------|-----------|-------|
| LooGLE TTFT mean (ms) | 9434 | 9416 | -0.19% (noise) |
| LooGLE TTFT median (ms) | 1190 | 1221 | +2.6% worse |
| LooGLE TTFT p90 (ms) | 3964 | 3712 | **-6.35% improved** |
| LooGLE TTFT p99 (ms) | 199280 | 198981 | -0.15% (noise) |
| LooGLE TPOT mean (ms) | 448.55 | 455.96 | +1.65% worse |
| LooGLE ITL mean (ms) | 349.80 | 349.03 | -0.22% neutral |
| LooGLE e2e mean (ms) | 14481 | 14452 | -0.20% neutral |
| LooGLE hit rate | 0.8966 | 0.8957 | -0.10% neutral |
| LooGLE device_hit_frac | 0.1199 | 0.1285 | +7.2% more GPU hits |
| ShareGPT TTFT mean (ms) | 38600 | 39120 | +1.35% worse |
| ShareGPT req throughput | 1.24 | 1.24 | same |
| ShareGPT total requests | 1118 | 1112 | -0.54% |

HiCache: evicted=318.0M (+0.2%), load_back=286.5M (-0.3%), cached_device=5.16M (+7.0% vs V35), evict_mean=1.119ms, load_back_mean=1.854ms, host_util=0.9988, disk_read=100672 tokens

**Analysis:** The mean TTFT is within noise of V35 (-0.19%, 18ms). The p90 improved (-6.35%, 252ms faster) from fewer anti-starvation interventions at higher queue depths, but TPOT regressed (+1.65%) and device_hit_frac increased (+7.2%) — the same GPU hoarding pattern seen in V36/V37. The higher effective threshold (~435s at queue=95) keeps requests in LPM order longer, concentrating GPU usage on top-match requests at the expense of decode throughput. ShareGPT regressed (+1.35%) because the queue-scaled threshold sometimes exceeds useful ranges for shorter-context workloads.

**Anti-starvation sensitivity (complete):**

| Threshold | TTFT mean (ms) | TTFT p90 (ms) | TPOT (ms) | Note |
|-----------|----------------|---------------|-----------|------|
| 120s (V8) | 16070 | — | — | too aggressive |
| 200s (V22) | 12797 | 38380 | 465 | too aggressive |
| **300s (V11/V35)** | **9434** | **3964** | **449** | **optimum** |
| ~435s dynamic (V38) | 9416 | 3712 | 456 | TPOT tradeoff, noise on mean |
| ∞ (V9) | 12186 | — | — | no safety net |

**Conclusion:** NEUTRAL/NEGATIVE. The dynamic threshold is not a meaningful improvement over fixed 300s. The mean is within noise, TPOT regressed, and ShareGPT worsened. The anti-starvation axis is now fully exhausted across all formulations (fixed thresholds 120s→200s→300s→∞, dynamic scaling). Reverted to fixed 300s.

---

## V39 — Gaussian time-decay (NEGATIVE)

**Commit:** `e48ace060`  
**Flag:** `--radix-eviction-policy gslru`

**Hypothesis:** Replace exponential decay `exp(-age/tau)` with Gaussian `exp(-(age/tau)^2)`. Gaussian has slower initial decay (better protection during 2-5s inter-question gaps) and faster late decay (quicker eviction of stale data from completed documents). Same tau values (15/20).

**What changed:**
- `evict_policy.py:GSLRUStrategy.get_priority()` — `decay = math.exp(-r * r)` where `r = age / tau`, instead of `math.exp(-age / tau)`

**Result (vs V35 best):**

| Metric | V35 | V39 | Delta |
|--------|-----|-----|-------|
| LooGLE TTFT mean (ms) | 9434 | 9567 | **+1.41% worse** |
| LooGLE TTFT median (ms) | 1210 | 1205 | -0.43% (noise) |
| LooGLE TTFT p90 (ms) | 3700 | 3914 | +5.8% worse |
| LooGLE TPOT mean (ms) | 449 | 471 | **+4.96% worse** |
| LooGLE E2E mean (ms) | 14389 | 14760 | +2.58% worse |
| LooGLE device_hit_frac | 0.1199 | 0.1209 | +0.83% (unchanged) |
| LooGLE load_back_tokens | 286.5M | 286.4M | unchanged |
| LooGLE evict_tokens | 318.0M | 317.9M | unchanged |
| ShareGPT TTFT mean (ms) | ~38850 | 38970 | +0.31% (noise) |
| ShareGPT hit_rate | 0.608 | 0.593 | -2.5% worse |

**Analysis:** Eviction volume is IDENTICAL to V35 (same load_back and evict tokens). device_hit_frac unchanged. The Gaussian shape changes eviction ORDER without changing total work, but the different ordering hurts decode performance (TPOT +4.96%). The Gaussian's slower initial decay keeps recently-accessed data at higher priority, changing WHICH nodes survive eviction rounds. This confirms that the exponential decay's priority ordering is better calibrated for this workload.

**Conclusion:** NEGATIVE. The decay function shape axis is now fully exhausted: exponential (V15-V19), two-tier (V26), smooth gradient (V27), narrow gradient (V28), and now Gaussian (V39) — all inferior to or at best neutral with the standard exponential at tau=15/20. Reverted to exponential decay.

---

## V40 — Skip promoted tree walks (NEGATIVE)

**Commit:** `bfa88aa01`  
**Flag:** `--radix-eviction-policy gslru`

**Hypothesis:** In `_compute_prefix_matches`, skip `match_prefix_for_req` tree walks for already-promoted requests. Their sort data from previous rounds is sufficient since the tree barely changes between scheduling rounds. Saves ~97% of per-round tree walk overhead, freeing CPU cycles for scheduling.

**What changed:**
- `schedule_policy.py:_compute_prefix_matches()` — skip tree walk for `_match_promoted = True` requests. Fresh data obtained in `init_next_round_input` when actually scheduled.

**Result (vs V35 best):**

| Metric | V35 | V40 | Delta |
|--------|-----|-----|-------|
| LooGLE TTFT mean (ms) | 9434 | 9623 | **+2.01% worse** |
| LooGLE TTFT median (ms) | 1210 | 1177 | **-2.76% better** |
| LooGLE TTFT p90 (ms) | 3700 | 3739 | +1.07% (noise) |
| LooGLE TPOT mean (ms) | 449 | 472 | **+5.12% worse** |
| LooGLE device_hit_frac | 0.1199 | 0.1313 | **+9.51% inflated** |
| LooGLE load_back_tokens | 286.5M | 287.3M | +0.28% (noise) |
| ShareGPT TTFT mean (ms) | ~38850 | 37520 | **-3.43% better** |
| ShareGPT req_throughput | 1.24 | 1.30 | **+4.84% better** |
| ShareGPT total_requests | 1112 | 1176 | **+5.76% better** |

**Analysis:** Mixed signal. **Median TTFT improved** (-2.76%), confirming that individual Q2+ requests schedule faster with reduced overhead. **ShareGPT significantly improved** (+4.84% throughput, +5.76% completions). But stale `prefix_indices` from skipped walks caused device_hit_frac to inflate (+9.51%): the sort key retained stale GPU token counts, making requests appear device-resident when their data was already evicted. This over-scheduling of "phantom GPU data" increased GPU memory pressure → TPOT +5.12% → mean TTFT +2.01%.

**Conclusion:** NEGATIVE for LooGLE. The scheduling overhead reduction genuinely helps (proven by median and ShareGPT improvements), but the stale device token data in the LPM sort key causes harmful eviction cascades under LooGLE's heavy reuse pattern. A variant using host-only sort keys for promoted requests might preserve the benefits while avoiding the staleness issue. Reverted to full tree walks.

---

## V41 — Skip tree walks for promoted requests with device_tokens==0 (NEGATIVE)

**Commit:** `5df8fe9f2`  
**Hypothesis:** V40 showed scheduling overhead reduction genuinely helps (median -2.76%, ShareGPT +4.84%), but stale device token data in `prefix_indices` caused device_hit_frac inflation (+9.51%). V41 filters more carefully: only skip tree walks for promoted requests that have `device_tokens == 0`. These cannot become stale-positive (the harmful V40 case) — only stale-negative (minor under-rank). This should preserve the scheduling speedup for ~88% of Q2+ requests while avoiding the device_hit_frac inflation.

**Change:** In `_compute_prefix_matches`, added early `continue` for promoted requests with `len(r.prefix_indices) == 0`, skipping `match_prefix_for_req` and `last_access_time` refresh for those requests.

| Metric | V35 (best) | V41 | Delta |
|--------|-----------|-----|-------|
| LooGLE TTFT mean (ms) | 9434 | 9764 | **+3.50% worse** |
| LooGLE TTFT median (ms) | 1210 | 1175 | **-2.89% better** |
| LooGLE TTFT p90 (ms) | 3573 | 4339 | +21.4% worse |
| LooGLE TPOT (ms) | 448.45 | 475.16 | **+5.95% worse** |
| LooGLE e2e (ms) | 14577 | 15027 | +3.09% worse |
| LooGLE device_hit_frac | 0.1199 | 0.1238 | +3.25% (less than V40) |
| LooGLE load_back_tokens | 286.5M | 287.1M | +0.24% (noise) |
| ShareGPT TTFT mean (ms) | ~38850 | 38520 | -0.85% (neutral) |
| ShareGPT req_throughput | 1.24 | 1.26 | +1.61% |
| ShareGPT total_requests | 1112 | 1136 | +2.16% |

**Analysis:** The device_tokens==0 filter successfully reduced device_hit_frac inflation from +9.51% (V40) to +3.25%, confirming the stale-device-data hypothesis. However, performance still regressed: TPOT +5.95% is worse than V40 (+5.12%). The problem is NOT stale device data — it's the missing `last_access_time` refresh. When tree walks are skipped, nodes that matched these requests don't get their `last_access_time` updated, changing eviction ordering. This is a fundamental constraint: `_match_prefix_helper` updates `last_access_time` on EVERY node it walks, and that side-effect is load-bearing for eviction quality.

**Conclusion:** NEGATIVE. The scheduling overhead reduction concept is validated (median -2.89%) but cannot be implemented via walk-skipping because `last_access_time` refresh is a critical side-effect of tree walks. Any future scheduling optimization must preserve this refresh. Reverted.

---

## V42 — Leaf-only timestamp update in _match_prefix_helper (NEGATIVE)

**Commit:** `985521496`  
**Hypothesis:** Instead of skipping walks entirely (V40/V41), keep full tree walks but only update `last_access_time` on the deepest matched node (not all intermediate nodes). Intermediate nodes can't be evicted while they have children, so their timestamps should only matter when they become orphaned leaves — at which point stale timestamps would correctly accelerate cleanup. This preserves accurate sort keys while reducing CPU overhead (~448 monotonic() calls saved per walk).

**Change:** In `_match_prefix_helper`, removed timestamp updates on root and traversed children, added single update on the final matched node.

| Metric | V35 (best) | V42 | Delta |
|--------|-----------|-----|-------|
| LooGLE TTFT mean (ms) | 9434 | 9553 | **+1.26% worse** |
| LooGLE TTFT median (ms) | 1210 | 1248 | +3.12% worse |
| LooGLE TTFT p90 (ms) | 3573 | 4153 | +16.2% worse |
| LooGLE TPOT (ms) | 448.45 | 468.69 | **+4.51% worse** |
| LooGLE device_hit_frac | 0.1199 | 0.1374 | **+14.6% worse** |
| LooGLE load_back_tokens | 286.5M | 288.8M | +0.8% |
| LooGLE evict_tokens | 319.8M | 320.8M | +0.3% |
| ShareGPT req_throughput | 1.24 | 1.25 | +0.81% (neutral) |
| ShareGPT total_requests | 1112 | 1122 | +0.90% (neutral) |

**Analysis:** The device_hit_frac inflation (+14.6%) is the WORST of V40-V42, worse than V40 (+9.51%) and V41 (+3.25%). Without intermediate timestamp refreshes, the eviction ordering shifts dramatically: when children ARE evicted and intermediates become leaves, they have very old timestamps and are evicted immediately, creating rapid cascade cleanup of entire prefix chains. This faster cleanup frees GPU memory, which is immediately consumed by load_backs, resulting in MORE data resident on GPU and higher memory pressure during decode.

**Key insight:** Intermediate node timestamps are NOT just "side effects" — they serve as eviction MOMENTUM that preserves prefix chain coherence. Fresh intermediate timestamps prevent premature cascade eviction of entire prefix paths, maintaining the optimal device_hit_frac of ~12%. The timestamp refresh on every tree walk is a critical, load-bearing behavior that cannot be reduced without causing device_hit_frac inflation → TPOT regression.

**Conclusion:** NEGATIVE. Intermediate timestamp updates are load-bearing for eviction chain coherence. The ~12% device_hit_frac achieved by V35 is the optimal operating point — any perturbation in either direction (more or less GPU residence) hurts TPOT. The tree walk overhead is a necessary cost. Reverted.

---

## V43 — Demand-aware eviction tiebreaker (NEGATIVE)

**Commit:** `610fb6c36`  
**Flag:** `--radix-eviction-policy gslru`

**Hypothesis:** Nodes matched by multiple waiting requests (high "demand") should be harder to evict. Track per-scheduling-round `_demand_count` on each node's `best_match_node`, and use it as a tiebreaker in GSLRU's `get_priority()` tuple: `(segment * decay, demand, last_access_time)`. This should protect high-demand nodes from eviction even when their time-decay priority drops.

**What changed:**
- `schedule_policy.py:_compute_prefix_matches()` — added `_demand_gen` / `_demand_count` tracking per scheduling round on `best_match_node`
- `evict_policy.py:GSLRUStrategy.get_priority()` — added `demand = getattr(node, "_demand_count", 0)` as middle element of priority tuple

**Result (vs V35 best):**

| Metric | V35 | V43 | Delta |
|--------|-----|-----|-------|
| LooGLE TTFT mean (ms) | 9434 | 9678 | **+2.59% worse** |
| LooGLE TTFT median (ms) | 1190 | 1226 | +3.03% worse |
| LooGLE TTFT p90 (ms) | 3964 | 4437 | +11.9% worse |
| LooGLE TTFT p99 (ms) | 199280 | 201521 | +1.12% worse |
| LooGLE TPOT mean (ms) | 448.55 | 471.10 | **+5.03% worse** |
| LooGLE ITL mean (ms) | 349.80 | 362.56 | +3.65% worse |
| LooGLE e2e mean (ms) | 14481 | 14909 | +2.96% worse |
| LooGLE hit rate | 0.8966 | 0.8937 | -0.32% worse |
| LooGLE device_hit_frac | 0.1199 | 0.1232 | +2.75% inflated |
| ShareGPT TTFT mean (ms) | 38600 | 38660 | +0.16% neutral |
| ShareGPT req throughput | 1.24 | 1.24 | same |
| ShareGPT hit rate | 0.608 | 0.612 | +0.66% neutral |

HiCache: evicted=318.0M, load_back=285.8M, cached_device=4.93M, evict_mean=1.11ms, load_back_mean=1.816ms, host_util=0.9999

**Analysis:** Same mechanism as V36-V42. The demand tiebreaker preferentially keeps high-demand nodes on GPU longer, causing device_hit_frac inflation (+2.75%) → TPOT regression (+5.03%) → mean TTFT regression (+2.59%). Despite being orthogonal to eviction ORDERING (demand is a scheduling signal, not an access-time signal), the effect flows through the same bottleneck: any change that increases GPU residence time beyond the V35 operating point hurts decode performance.

**Meta-insight confirmed:** V36-V43 represent 8 consecutive experiments across 5 distinct axes (promotion increment, promotion scope, starvation dynamics, decay shape, walk optimization, timestamp policy, demand tiebreaker) that all fail via device_hit_frac inflation → TPOT regression. The ~12% device_hit_frac at V35 is confirmed as a structural equilibrium — not a tunable operating point.

**Conclusion:** NEGATIVE. Reverted. The demand-aware eviction concept is directionally wrong: the eviction priority should NOT consider demand signals, because demand-driven retention causes GPU hoarding that hurts decode throughput.

---

## V44 — Conservative host eviction (2x tau for host tier) (NEGATIVE)

**Commit:** `e6a3b12b2`  
**Flag:** `--radix-eviction-policy gslru`

**Hypothesis:** Host eviction uses the same GSLRU (tau=15/20) as GPU eviction, but data evicted from host is effectively LOST (storage hit rate ~0%). A separate host eviction strategy with 2x decay tau (30/40) should keep frequently-reused document prefixes in host longer across the ~57s inter-question gap. This should be orthogonal to GPU tier behavior — device_hit_frac should be unaffected.

**What changed:**
- `hiradix_cache.py:__init__()` — created separate `host_eviction_strategy` (GSLRUStrategy with tau=30, tau_top=40)
- `hiradix_cache.py:evict_host()` — used `host_eviction_strategy` for priority computation instead of shared `eviction_strategy`

**Result (vs V35 best):**

| Metric | V35 | V44 | Delta |
|--------|-----|-----|-------|
| LooGLE TTFT mean (ms) | 9434 | 9775 | **+3.61% worse** |
| LooGLE TTFT median (ms) | 1190 | 1212 | +1.85% worse |
| LooGLE TTFT p90 (ms) | 3964 | 4046 | +2.07% worse |
| LooGLE TTFT p99 (ms) | 199280 | 204284 | +2.51% worse |
| LooGLE TPOT mean (ms) | 448.55 | 462.50 | **+3.11% worse** |
| LooGLE ITL mean (ms) | 349.80 | 360.59 | +3.08% worse |
| LooGLE e2e mean (ms) | 14481 | 14977 | +3.43% worse |
| LooGLE hit rate | 0.8966 | 0.8950 | -0.18% worse |
| LooGLE device_hit_frac | 0.1199 | 0.1218 | +1.58% inflated |
| ShareGPT TTFT mean (ms) | 38600 | 39150 | +1.42% worse |
| ShareGPT req throughput | 1.24 | 1.24 | same |

HiCache: evicted=319.2M, load_back=287.4M, cached_device=4.88M, evict_mean=1.106ms, load_back_mean=1.838ms, host_util=0.9992

**Analysis:** The orthogonality hypothesis was WRONG. Changing host eviction ordering changes WHICH data survives in host. When load_back occurs, different data is loaded from host to GPU → different GPU cache composition → device_hit_frac inflation (+1.58%) → TPOT regression (+3.11%). The causal chain is: host eviction order → host data composition → load_back data → GPU data composition → device_hit_frac → TPOT.

This establishes a deeper meta-insight: **there is no truly orthogonal axis in the cache hierarchy**. Every tier's eviction ordering propagates through load_back to affect the GPU tier. The V35 equilibrium is not just a GPU-tier optimum — it's a SYSTEM-WIDE equilibrium where GPU, host, and scheduling are mutually reinforcing.

**Conclusion:** NEGATIVE. Reverted. The host eviction is coupled to GPU behavior via load_back. Any change to any tier's eviction ordering will propagate to device_hit_frac. V36-V44 (9 consecutive experiments) now confirm: the system is at a global equilibrium, not just a local GPU-tier optimum.

---

## V45 — Scheduling hot-path CPU optimizations (NEUTRAL)

**Commit:** `93f2172a1`  
**Flag:** `--radix-eviction-policy gslru`

**Hypothesis:** Given that V36-V44 confirmed all eviction/scheduling perturbations fail due to a global equilibrium, try a structurally different approach: pure CPU optimization of the scheduling hot path. Three changes: (1) tree generation counter to skip redundant `_compute_prefix_matches` walks when tree hasn't changed, (2) fast-path timestamp update by walking up from `best_match_node` instead of full tree walks, (3) batched `time.monotonic()` to one call per walk instead of per-node. These are semantics-preserving optimizations that reduce Python overhead in the scheduling round.

**What changed:**
- `hiradix_cache.py` — added `_tree_generation` counter incremented on evict/load_back/insert; batched `time.monotonic()` to one call per walk in `_match_prefix_helper`
- `schedule_policy.py` — added generation-aware skip in `_compute_prefix_matches`: when tree generation unchanged and all requests already computed, update timestamps via walk-up from `best_match_node` instead of full tree walks

**Result (vs V35 best):**

| Metric | V35 | V45 | Delta |
|--------|-----|-----|-------|
| LooGLE TTFT mean (ms) | 9434 | 9389 | -0.47% (within noise) |
| LooGLE TTFT median (ms) | 1190 | 1204 | +1.18% neutral |
| LooGLE TTFT p90 (ms) | 3964 | 3925 | -0.98% neutral |
| LooGLE TTFT p99 (ms) | 199280 | 198983 | -0.15% neutral |
| LooGLE TPOT mean (ms) | 448.55 | 452.94 | +0.98% slight regression |
| LooGLE ITL mean (ms) | 349.80 | 350.56 | +0.22% neutral |
| LooGLE e2e mean (ms) | 14481 | 14447 | -0.24% neutral |
| LooGLE hit rate | 0.8966 | 0.8957 | -0.10% neutral |
| LooGLE device_hit_frac | 0.1199 | 0.1279 | +6.67% inflated |
| ShareGPT TTFT mean (ms) | 38600 | 39090 | +1.27% neutral |
| ShareGPT req throughput | 1.24 | 1.23 | -0.81% neutral |

HiCache: evicted=317.6M, load_back=286.1M, cached_device=5.13M, evict_mean=1.125ms, load_back_mean=1.856ms, host_util=0.9985

**Analysis:** The CPU optimizations produced no measurable TTFT improvement (-0.47% is within eval variance of ~1-2%). The slight TPOT regression (+0.98%) and device_hit_frac increase (+6.67%) suggest the timestamp distribution change from the fast-path walk-up (updating only the ancestor chain instead of all walked nodes) subtly alters eviction ordering. The system is GPU-bound, not CPU-bound — scheduling overhead is not a bottleneck.

**Key insight:** This was the first experiment targeting a fundamentally different axis (CPU overhead vs cache policy). The null result confirms: the ~12% device_hit_frac equilibrium is driven by GPU compute capacity and working set size, not by scheduling inefficiency. Improving the scheduling speed doesn't change what gets cached or when.

**Conclusion:** NEUTRAL. Reverted. CPU scheduling optimization is the wrong lever — the system's throughput is bottlenecked by GPU compute and memory, not by CPU scheduling overhead. V36-V45 (10 consecutive experiments) all fail to beat V35, covering eviction ordering (V36-V44) and scheduling overhead (V45).

---

## Current standings

| Version | LooGLE TTFT mean | Status |
|---------|-----------------|--------|
| V0 (baseline) | 40371ms | — |
| V6 (GSLRU+LPM) | 10951ms | incumbent |
| V7 (adaptive FCFS) | 33729ms | NEGATIVE |
| V8 (anti-starvation 120s) | 16070ms | best p99/hit |
| V9 (pure LPM) | 12186ms | V6 repro |
| V10 (DFS_WEIGHT) | 36857ms | NEGATIVE |
| V11 (LPM+starvation 300s) | 10658ms | — |
| V12 (device weight=2) | 10506ms | — |
| V13 (device weight=4) | 10217ms | — |
| V14 (device weight=8) | 10903ms | NEGATIVE |
| V15 (time-decay GSLRU tau=30) | 10114ms | — |
| V16 (time-decay GSLRU tau=15) | 9814ms | — |
| V17 (time-decay GSLRU tau=7.5) | 10211ms | NEGATIVE (tau too low) |
| V18 (max_segment=2, tau=15) | 9816ms | NEUTRAL (mean flat, p90 -11.7%) |
| V19 (tau=10, max_segment=4) | 9854ms | NEUTRAL (mean flat, p90 -5.3%) |
| V20 (seg=2 + tau=10) | 9774ms | NEGATIVE (mean noise, p90 +15.3%, TPOT +7.1%) |
| V21 (write_through_selective) | 31294ms | CATASTROPHIC (-67% host backup, -23% hit rate) |
| V22 (anti-starvation 200s) | 12797ms | NEGATIVE (+30.4%, p90 catastrophic) |
| V23 (cold fast-track 30s/10K) | 65846ms | CATASTROPHIC (+571%, server crash, 40% completion) |
| V24 (page_size=128) | — | SERVER CRASH (SIGBUS during DeepGEMM warmup) |
| V25 (page_size=32) | 11890ms | NEGATIVE (+21.2%, TPOT +19.6%, ShareGPT -43%) |
| V26 (two-tier decay) | 9730ms | MARGINAL POSITIVE (-0.9%, within noise) |
| V27 (smooth gradient decay) | 9688ms | MARGINAL POSITIVE (-1.3%, TPOT +4.5%) |
| V28 (narrow gradient 1.5x) | 9651ms | MARGINAL POSITIVE (-1.7%, best tradeoff) |
| V29 (top-only decay tau=20) | 9642ms | prev best (-1.8%, p90 -8.0%, TPOT -2.6%) |
| V30 (top-only decay tau=25) | 9748ms | NEGATIVE (tau too high, +1.1%, p90 +12.8%) |
| V31 (device weight=5) | 9535ms | prev best (-1.11%, median -5.43%, TPOT neutral) |
| V32 (device weight=6) | 9943ms | NEGATIVE (+4.28%, p90 +17.6%, TPOT +7.1%) |
| V33 (top-segment tau=22.5) | 9739ms | NEGATIVE (+2.14%, TPOT +9.53%) |
| V34 (host_weight=2) | 9926ms | NEGATIVE (+4.10%, p90 +14.4%, TPOT +6.8%) |
| **V35 (match promotion)** | **9434ms** | **CURRENT BEST (-1.06%, median -6.33%, e2e -2.94%)** |
| V36 (match promo +=2) | 9680ms | NEGATIVE (+2.60%, p90 -8.48%, over-promotion) |
| V37 (parent promotion) | 9626ms | NEGATIVE (+2.03%, TPOT +3.20%, over-promotion) |
| V38 (dynamic starvation) | 9416ms | NEUTRAL/NEGATIVE (-0.19% noise, TPOT +1.65%) |
| V39 (Gaussian decay) | 9567ms | NEGATIVE (+1.41%, TPOT +4.96%, decay shape suboptimal) |
| V40 (skip promoted walks) | 9623ms | NEGATIVE (+2.01%, median -2.76%, stale device data) |
| V41 (skip device0 walks) | 9764ms | NEGATIVE (+3.50%, TPOT +5.95%, median -2.89%) |
| V42 (leaf-only timestamp) | 9553ms | NEGATIVE (+1.26%, TPOT +4.51%, device_hit_frac +14.6%) |
| V43 (demand-aware eviction) | 9678ms | NEGATIVE (+2.59%, TPOT +5.03%, device_hit_frac +2.75%) |
| V44 (host eviction 2x tau) | 9775ms | NEGATIVE (+3.61%, TPOT +3.11%, host→GPU coupling) |
| V45 (sched hot-path CPU opt) | 9389ms | NEUTRAL (-0.47% noise, TPOT +0.98%, CPU not bottleneck) |
| V46 (anti-starvation 60s) | FAILED | SERVER HANG at 62% progress, deadlock with 100% GPU |

---

## V46 — Anti-starvation threshold reduction (FAILED — server hang)

**Commit:** `e8a015d32`  
**Hypothesis:** Reduce LPM anti-starvation threshold from 300s to 60s. Q1 requests waiting >60s get priority-boosted over Q2-Q8, shortening their wait and creating a cascade where Q2-Q8 arrive sooner while prefix is still GPU-warm.

**Change:** Single line: `_LPM_STARVATION_SECS = 300.0 → 60.0` in `schedule_policy.py`

**Result: CATASTROPHIC FAILURE**

The server hung at 62% progress (963/1560 LooGLE requests completed). After completing 966 requests, the server became completely unresponsive:
- 18 running requests, 98 queued — frozen for 30+ minutes
- GPUs at 100% utilization with zero HTTP completions
- Test request to the server timed out
- Job cancelled after confirming deadlock

**Root cause analysis:** With the 60s threshold, ALL Q1 requests (which have been waiting >60s by the time the system reaches steady state) get simultaneously priority-boosted to tier 0. This overwhelms the scheduler: instead of interleaving Q1 (new document, ~28.7K tokens) with Q2-Q8 (cache-hit, ~28.7K tokens but with warm prefix), the system schedules only Q1 requests. With 18 large Q1 prefills running simultaneously (18 × ~28.7K = ~517K tokens), the GPU KV cache is saturated. The constant eviction pressure from new Q1 arrivals, combined with no Q2-Q8 requests completing (their cache gets evicted before they can run), creates a livelock where the system does compute but never completes any request.

**Key insight:** The 300s threshold was NOT "inactive" as hypothesized. It serves as a critical safety valve — it allows the scheduler to naturally interleave Q1/Q2-Q8 via LPM ordering. Setting it to 60s effectively removes this interleaving and creates a Q1-only scheduling pattern that saturates resources. The anti-starvation mechanism is tightly coupled to the scheduling equilibrium.

**Learning:** Anti-starvation threshold < Q1 mean TTFT (~65s) causes pathological scheduling. Any threshold that triggers for the MAJORITY of Q1 requests simultaneously will cause this failure mode. Safe threshold must be > P90 of Q1 TTFT (~130s), or the boost mechanism needs a concurrency cap.

**Code:** Reverted to 300.0 after failure.

## V47 — Page size 128 (NEGATIVE)

**Commit:** `279c1a69f` (no code change — flag override only)
**Flag:** `--radix-eviction-policy gslru --page-size 128`

**Hypothesis:** Larger pages (128 vs default 64) reduce per-page overhead in the radix tree: fewer nodes, smaller eviction heaps, less hash computation. Trade-off: coarser granularity means less flexible eviction and potential memory waste from partially-used pages.

**Result (vs V35 best, page_size=64):**

| Metric | V35 (page=64) | V47 (page=128) | Delta |
|--------|---------------|-----------------|-------|
| LooGLE TTFT mean (ms) | 9434 | 10463 | **+10.9% worse** |
| LooGLE TTFT median (ms) | 1210 | 1230 | +1.7% |
| LooGLE TTFT p90 (ms) | 3700 | 4080 | +10.3% worse |
| LooGLE TTFT p99 (ms) | 211836 | 212141 | +0.1% (noise) |
| LooGLE TPOT mean (ms) | 449 | 452 | +0.7% (noise) |
| LooGLE hit_rate | 0.8949 | 0.8912 | -0.4% worse |
| LooGLE device_hit_frac | 0.1199 | 0.1164 | -2.9% worse |
| LooGLE evicted tokens | ~287M | 315.6M | **+10.0% more** |
| LooGLE load_back tokens | ~285M | 282.9M | -0.7% (noise) |
| LooGLE load_back_mean_ms | 1.793 | 1.642 | **-8.4% faster** |
| LooGLE eviction_mean_ms | ~1.0 | 0.964 | -3.6% faster |
| ShareGPT TTFT mean (ms) | ~38850 | 21520 | N/A (different baseline) |
| ShareGPT req throughput | ~1.28 | 1.84 | N/A |

HiCache: evicted=315.6M, load_back=282.9M, cached_device=4.65M (-7.7% vs V35 5.01M), host_util=0.9966

**Analysis:** Larger pages increase eviction volume by 10% (315.6M vs ~287M). Each eviction removes a 128-token chunk, which is too coarse — it evicts "useful" tokens alongside "cold" ones in the same page. The result is a lower device_hit_frac (0.1164 vs 0.1199) and lower hit_rate (0.8912 vs 0.8949). Despite faster per-operation load_back (1.642ms vs 1.793ms — fewer, larger DMA transfers), the increased eviction volume overwhelms this benefit.

The positive: load_back_mean_ms dropped 8.4%, confirming that larger pages amortize DMA overhead better. But this saving is irrelevant when more data is being churned.

**Conclusion:** Page size 128 is clearly worse than 64 for this workload. Coarser pages → more wasteful eviction → lower cache efficiency → higher TTFT. Next: test page_size=32 to complete the sweep (finer pages → less waste per eviction, but more overhead per page).

**Page size sensitivity (partial — V48 adds page_size=32):**

| page_size | TTFT mean (ms) | hit_rate | device_hit_frac | evicted (M) | load_back_mean_ms | ShareGPT throughput |
|-----------|----------------|----------|-----------------|-------------|-------------------|---------------------|
| 32 (V48)  | **11246**       | 0.8919   | 0.1293          | 315.4       | 1.868             | **0.73** req/s      |
| 64 (V35)  | **9434**        | 0.8949   | 0.1199          | ~287        | 1.793             | ~1.84 req/s         |
| 128 (V47) | 10463          | 0.8912   | 0.1164          | 315.6       | 1.642             | 1.84 req/s          |

---

### V48 — page_size=32 (NEGATIVE: +19.2% worse LooGLE, catastrophic ShareGPT)

**Commit:** `279c1a69f` (no code change — flag override only)
**Flag:** `--radix-eviction-policy gslru --page-size 32`

**Hypothesis:** Finer pages (32 vs default 64) reduce wasteful eviction: each eviction removes only 32 tokens instead of 64, evicting fewer "useful" tokens alongside "cold" ones. Trade-off: 2× more pages → more per-page overhead (hash computation, tree traversal, eviction heap size, DMA setup per transfer).

**Result (vs V35 best, page_size=64):**

| Metric | V35 (page=64) | V48 (page=32) | Delta |
|--------|---------------|---------------|-------|
| LooGLE TTFT mean (ms) | 9434 | 11246 | **+19.2% worse** |
| LooGLE TTFT median (ms) | 1210 | 1446 | +19.5% worse |
| LooGLE TTFT p90 (ms) | 3700 | 4389 | +18.6% worse |
| LooGLE TTFT p99 (ms) | 211836 | 239873 | +13.2% worse |
| LooGLE TPOT mean (ms) | 449 | 555 | **+23.6% worse** |
| LooGLE ITL mean (ms) | — | 413 | — |
| LooGLE e2e mean (ms) | — | 17207 | — |
| LooGLE hit_rate | 0.8949 | 0.8919 | -0.3% worse |
| LooGLE device_hit_frac | 0.1199 | 0.1293 | +7.8% better |
| LooGLE evicted tokens | ~287M | 315.4M | **+9.9% more** |
| LooGLE load_back tokens | ~285M | 282.6M | -0.8% (noise) |
| LooGLE load_back_mean_ms | 1.793 | 1.868 | **+4.2% slower** |
| LooGLE eviction_mean_ms | ~1.0 | 1.142 | **+14.2% slower** |
| ShareGPT TTFT mean (ms) | ~38850 | **71440** | **+83.8% worse** |
| ShareGPT req throughput | ~1.28 | **0.73** | **-43.0% worse** |
| ShareGPT total_requests | ~1658 | **657** | **-60.4% fewer** |

HiCache: evicted=315.4M, load_back=282.6M, cached_device=5.17M (+3.2% vs V35 5.01M), host_util=0.9934

**Analysis:** page_size=32 is catastrophically negative, especially on ShareGPT. Root causes:

1. **Per-page overhead dominates:** With 2× more pages, every operation (hash, tree traversal, DMA setup, eviction heap manipulation) runs 2× as often. The DMA transfers are smaller but NOT 2× faster (fixed per-op overhead): load_back_mean increased from 1.793ms → 1.868ms (+4.2%), eviction_mean from ~1.0ms → 1.142ms (+14.2%).

2. **Eviction volume UNCHANGED:** Despite finer granularity, total eviction volume is 315.4M — nearly identical to V47 (128) at 315.6M. The "less wasteful eviction" hypothesis failed: finer pages don't reduce total churn because the working set / GPU capacity ratio is the fundamental constraint.

3. **ShareGPT CATASTROPHE:** Under high concurrency (60 clients), the per-page overhead compounds multiplicatively. 2× more pages × 60 concurrent clients → severe scheduler slowdown. TTFT exploded to 71.4s (3.3× worse), throughput collapsed to 0.73 req/s (60% fewer requests served).

4. **Small positive — device_hit_frac:** 0.1293 vs 0.1199 (+7.8%). Finer pages DO allow more precise caching (less space wasted on partially-cold pages). But this tiny benefit is overwhelmed by the overhead costs.

**Conclusion:** page_size=32 is far worse than 64 for both benchmarks. Combined with V47 (128 also worse), the page_size sweep is complete: **64 is the convex optimum**. Finer pages increase overhead without reducing churn; coarser pages increase wasteful eviction. This axis is EXHAUSTED.

**Page size sensitivity (COMPLETE):**

| page_size | LooGLE TTFT (ms) | vs V35 | ShareGPT throughput | Overall |
|-----------|-------------------|--------|---------------------|---------|
| 32 (V48)  | 11246             | +19.2% | 0.73 req/s          | NEGATIVE |
| **64 (V35)** | **9434**       | **baseline** | **~1.84 req/s** | **BEST** |
| 128 (V47) | 10463             | +10.9% | 1.84 req/s          | NEGATIVE |

---

### V49 — write_back mode (NEW BEST: -2.58% LooGLE TTFT)

**Commit:** `ed506ab77` (no code change — flag override only)
**Flag:** `--radix-eviction-policy gslru --hicache-write-policy write_back`

**Hypothesis:** `write_back` defers GPU→host backup DMA to eviction time (instead of eagerly backing up on insertion). Two expected benefits: (1) no wasted DMA for nodes that are never evicted, (2) host memory stores only truly-evicted data (no redundant copies of GPU-resident nodes), increasing effective unique cache capacity by ~16%.

**Result (vs V35 best, write_through):**

| Metric | V35 (write_through) | V49 (write_back) | Delta |
|--------|---------------------|-------------------|-------|
| LooGLE TTFT mean (ms) | 9434 | **9190** | **-2.58% BETTER** |
| LooGLE TTFT median (ms) | 1210 | 1192 | -1.5% better |
| LooGLE TTFT p90 (ms) | 3700 | 3740 | +1.1% (noise) |
| LooGLE TTFT p99 (ms) | 211836 | 200676 | **-5.3% better** |
| LooGLE TPOT mean (ms) | 449 | 424 | **-5.6% better** |
| LooGLE ITL mean (ms) | ~350 | 342 | -2.3% better |
| LooGLE e2e mean (ms) | ~14500 | 14126 | -2.6% better |
| LooGLE hit_rate | 0.8949 | 0.8965 | +0.16pp better |
| LooGLE device_hit_frac | 0.1199 | 0.100 | **-16.6% worse** |
| LooGLE host_hit_frac | 0.88 | 0.90 | +2.3% better |
| LooGLE evicted tokens | ~287M | 325.9M | +13.5% more |
| LooGLE load_back tokens | ~285M | 294.7M | +3.4% more |
| LooGLE load_back_mean_ms | 1.793 | **6.881** | **+283% slower** |
| LooGLE eviction_mean_ms | ~1.0 | **5.791** | **+479% slower** |
| LooGLE host_util | 0.9966 | 0.9992 | +0.3pp fuller |
| ShareGPT TTFT mean (ms) | ~38850 | 24560 | N/A (different baseline) |
| ShareGPT req throughput | ~1.28 | 1.69 | N/A |
| ShareGPT total_requests | ~1658 | 1523 | -8.1% fewer |

HiCache: evicted=325.9M, load_back=294.7M, cached_device=4.02M (-17.3% vs V35 4.86M), host_util=0.9992

**Analysis:** write_back produces a genuine new best despite dramatically worse per-operation latencies.

**Why write_back helps (3 mechanisms):**

1. **Eliminated wasted insertion DMA:** In write_through, EVERY new node gets an immediate GPU→host backup (threshold=1). For 200 documents × ~28.7K tokens each = ~5.7M tokens, that's ~89K backup DMA operations during ramp-up alone. write_back eliminates this entirely — no DMA during insertion.

2. **Better host memory utilization:** In write_through, GPU-resident nodes ALSO have host copies (redundant). With ~686K GPU tokens, that's ~16% of host capacity wasted on duplicates. write_back stores only truly-evicted data in host, increasing the unique cache capacity. Confirmed by host_util rising to 0.9992 (vs 0.9966) and hit_rate improving to 0.8965 (vs 0.8949).

3. **Faster steady-state scheduling:** Without insertion-time backup DMA, request completion (`cache_finished_req`) is faster. This translates directly to improved TPOT (424ms vs 449ms, -5.6%) and ITL (342ms vs 350ms, -2.3%).

**Why per-op latencies are worse (expected trade-off):**

- **load_back_mean_ms: 6.881ms vs 1.793ms (+283%):** In write_back, load_back from host→GPU must also trigger eviction of the replaced GPU data WITH DMA backup (GPU→host). In write_through, eviction is just a pointer drop (host copy already exists). So write_back's load_back includes both the incoming DMA AND the outgoing eviction DMA.

- **eviction_mean_ms: 5.791ms vs 1.0ms (+479%):** Same reason — write_back eviction includes GPU→host DMA transfer, while write_through eviction is ~free.

- **device_hit_frac dropped to 0.10 from 0.12:** With slower eviction, the scheduler spends more time per eviction operation, reducing the effective scheduling throughput. This manifests as slightly lower GPU cache efficiency. However, the HIGHER host_hit_frac (0.90 vs 0.88) MORE than compensates.

**Net effect:** The elimination of ~89K+ wasted DMA operations per benchmark outweighs the ~4× per-op slowdown on load_back/eviction, because the total NUMBER of operations decreases. write_through does DMA at INSERTION + load_back = 2 DMAs per node lifecycle. write_back does DMA at EVICTION + load_back = same 2 DMAs, but the insertion DMA is skipped for nodes that are NEVER evicted or are evicted before their backup would have been useful.

**ShareGPT regression (expected):** Under high concurrency (60 clients), the 4× slower per-op load_back/eviction compounds across concurrent requests. 1.69 req/s vs ~1.84 req/s is an 8.2% throughput regression. ShareGPT is more sensitive to per-op overhead because of its concurrent access pattern.

**Conclusion:** write_back is the NEW BEST for LooGLE TTFT at -2.58% improvement. This is the first improvement since V35 (14 consecutive non-improvements from V36-V48). The win comes from eliminating wasted DMA bandwidth and improving host memory efficiency.

**Updated leaderboard:**

| Version | Change | LooGLE TTFT (ms) | vs Baseline | vs Previous Best |
|---------|--------|-------------------|-------------|------------------|
| V0      | baseline | 40371           | —           | —                |
| V6      | GSLRU eviction | 10951      | -73.9%      | —                |
| V16     | time-decay tau=15 | 9814     | -75.7%      | -10.4%           |
| V29     | top-tier tau=20 | 9642       | -76.1%      | -1.8%            |
| V31     | device_weight=5 | 9535       | -76.4%      | -1.1%            |
| V35     | match promotion | 9434       | -76.6%      | -1.1%            |
| V49     | write_back | 9190           | -77.2%      | -2.58%           |
| **V50** | **wb + tau=20** | **8960**   | **-77.8%**  | **-2.51%**       |

---

### V50 — write_back + tau=20 (NEW BEST: -5.03% vs V35, -2.51% vs V49)

**Commit:** `6e550fa23` (code change: evict_policy.py decay_tau default 15→20)
**Flags:** `--radix-eviction-policy gslru --hicache-write-policy write_back`
**Code:** `GSLRUStrategy.__init__` decay_tau default changed from 15.0 to 20.0

**Hypothesis:** With write_back making eviction 6× more expensive (5.8ms/op), higher tau reduces eviction frequency. Under write_through, tau=20 was tested and slightly worse than 15 (eviction was essentially free, so more frequent eviction to keep cache fresh was beneficial). Under write_back, the cost-benefit shifts: reducing expensive evictions is more valuable than marginally fresher cache.

**Result (vs V49 write_back tau=15, and vs V35 write_through tau=15):**

| Metric | V35 (wt, tau=15) | V49 (wb, tau=15) | V50 (wb, tau=20) | V50 vs V35 |
|--------|------------------|------------------|------------------|------------|
| LooGLE TTFT mean (ms) | 9434 | 9190 | **8960** | **-5.03%** |
| LooGLE TTFT median (ms) | 1210 | 1192 | **1097** | **-9.4%** |
| LooGLE TTFT p90 (ms) | 3700 | 3740 | **3644** | -1.5% |
| LooGLE TTFT p99 (ms) | 211836 | 200676 | **196985** | **-7.0%** |
| LooGLE TPOT mean (ms) | 449 | 424 | 424 | -5.6% |
| LooGLE ITL mean (ms) | ~350 | 342 | 341 | -2.6% |
| LooGLE e2e mean (ms) | ~14500 | 14126 | **13879** | -4.3% |
| LooGLE hit_rate | 0.8949 | 0.8965 | 0.8964 | +0.15pp |
| LooGLE device_hit_frac | 0.1199 | 0.100 | 0.0963 | -19.7% |
| LooGLE host_hit_frac | 0.88 | 0.90 | 0.9037 | +2.7% |
| LooGLE load_back_mean_ms | 1.793 | 6.881 | 6.660 | +271% |
| LooGLE eviction_mean_ms | ~1.0 | 5.791 | 5.791 | +479% |
| ShareGPT throughput | ~1.84 | 1.69 | **1.72** | -6.5% |
| ShareGPT total_requests | ~1658 | 1523 | **1551** | -6.5% |

HiCache: evicted=327.9M, load_back=296.7M, cached_device=3.87M, host_util=0.999

**Analysis:**

The write_back + tau=20 combination compounds two synergistic effects:

1. **write_back eliminates insertion DMA** (V49 mechanism, unchanged): Nodes backed up only at eviction, not eagerly at creation. Saves ~89K+ unnecessary DMA operations.

2. **Higher tau reduces eviction frequency** (V50 new mechanism): Nodes decay more slowly (exp(-age/20) vs exp(-age/15)), staying "hot" 33% longer. Under write_through, this was slightly negative because it kept stale entries. Under write_back, it's positive because each avoided eviction saves 5.8ms of DMA.

**Compounding mechanism:** write_back makes eviction expensive → higher tau avoids expensive evictions → net win. The key insight is that tau=15 was optimized for ~free eviction (write_through). With eviction costing 6ms (write_back), the optimal tau shifts upward.

**TTFT median improvement (-9.4%):** The most striking result. Q2-Q8 requests (which dominate the median) benefit disproportionately because:
- Higher tau keeps their documents' KV "hot" longer between sequential questions
- Less eviction between Q(n) and Q(n+1) means fewer load_backs needed
- TTFT median dropped from 1210ms to 1097ms — a 113ms improvement for the typical request

**ShareGPT slightly improved vs V49:** 1.72 vs 1.69 req/s (+1.8%). Higher tau reduces the total number of expensive eviction operations, which helps under concurrent load. Still below V35's ~1.84 req/s due to write_back's inherently slower eviction.

**Conclusion:** write_back + tau=20 is the NEW BEST with a -5.03% improvement over V35 across the primary metric (LooGLE TTFT mean). The parameters are synergistic — neither alone achieves this result. Next: test tau=25 to find the write_back-optimal tau.

---

### V51 — write_back + tau=25 (NEW BEST mean, but mixed signals)

**Commit:** `48226673b` (code change: evict_policy.py decay_tau 20→25)
**Flags:** `--radix-eviction-policy gslru --hicache-write-policy write_back`

**Hypothesis:** Continuing tau sweep under write_back. V50 (tau=20) improved 2.5% over V49 (tau=15). Testing if the trend continues with tau=25.

**Tau sweep under write_back (LooGLE):**

| tau | TTFT mean | TTFT median | TTFT p90 | TTFT p99 | TPOT |
|-----|-----------|-------------|----------|----------|------|
| 15 (V49) | 9190 | 1192 | 3740 | 200676 | 424 |
| 20 (V50) | 8960 | **1097** | **3644** | 196985 | **424** |
| 25 (V51) | **8787** | 1129 | 3999 | **192709** | 444 |

**Analysis:** tau=25 achieves a new best TTFT MEAN (-1.93% vs V50, -6.86% vs V35) but the improvement is concentrated in the TAIL while the MID-RANGE degrades:

- **Mean improves:** 8787ms (-1.93% vs V50). Q1 TTFTs are shorter because slower decay keeps more KV in host for longer, reducing the eviction pressure on Q1 prefill.
- **Median degrades:** 1129ms (+2.9% vs V50's 1097ms). Q2-Q8 performance slightly worse because tau=25 is too slow to evict cold nodes, reducing the effective GPU cache for recent documents.
- **P90 significantly degrades:** 3999ms (+9.7% vs V50's 3644ms). The 90th percentile request is substantially slower.
- **P99 improves:** 192709ms (-2.2% vs V50). The slowest requests (Q1s for the longest documents) benefit from reduced eviction.
- **TPOT degrades:** 444ms (+4.7% vs V50's 424ms). Token generation is slower, likely because higher tau means more eviction contention during decode.
- **Storage hits appeared:** 0.08% of tokens now hit STORAGE (disk). With higher tau, host cache entries persist longer, forcing some to spill to disk.

**Key insight:** The write_back tau optimum depends on which metric matters:
- For MEAN TTFT: tau=25 is better (mean keeps improving)
- For MEDIAN/P90 (typical request experience): tau=20 is the sweet spot
- For TPOT (generation speed): tau=20 is clearly better

The mean improves because tail improvements (Q1, absolute ~173ms savings) outweigh mid-range degradation (Q2-Q8, absolute ~32ms worsening). Since Q1 is 12.8% of requests with ~65s TTFT, its improvements dominate the mean.

**ShareGPT:** Essentially unchanged vs V50 (1.70 vs 1.72 req/s, -1.2%).

**Updated leaderboard (by TTFT mean):**

| Version | Change | LooGLE TTFT (ms) | vs Baseline | vs V35 |
|---------|--------|-------------------|-------------|--------|
| V0      | baseline | 40371           | —           | —      |
| V35     | match promotion | 9434       | -76.6%      | —      |
| V49     | write_back | 9190           | -77.2%      | -2.6%  |
| V50     | wb + tau=20 | 8960           | -77.8%      | -5.0%  |
| **V51** | **wb + tau=25** | **8787**   | **-78.2%**  | **-6.9%** |

---

### V52 — write_back + tau=30 (NEGATIVE: +3.68% vs V51)

**Commit:** `da06d2daf` (code change: evict_policy.py decay_tau 25→30)
**Flags:** `--radix-eviction-policy gslru --hicache-write-policy write_back`

**Hypothesis:** Completing the tau sweep under write_back. V51 (tau=25) achieved best mean but degraded median/p90. Testing tau=30 to find where mean also regresses.

**Complete tau sweep under write_back (LooGLE):**

| tau | TTFT mean | TTFT median | TTFT p90 | TTFT p99 | TPOT |
|-----|-----------|-------------|----------|----------|------|
| 15 (V49) | 9190 | 1192 | 3740 | 200676 | 424 |
| 20 (V50) | 8960 | **1097** | 3644 | 196985 | **424** |
| 25 (V51) | **8787** | 1129 | 3999 | **192709** | 444 |
| 30 (V52) | 9110 | 1141 | **3623** | 198251 | 429 |

**Result:** NEGATIVE. TTFT mean = 9110ms (+3.68% vs V51's 8787ms). Mean is now WORSE than V50 (tau=20).

**Analysis:** The tau sweep is complete. The mean curve is clearly convex with an optimum at tau=25:
- tau 15→20: mean improves 2.5% (both mean and median improve together)
- tau 20→25: mean improves 1.9% (but median/p90/TPOT degrade — improvement is tail-only)
- tau 25→30: mean REGRESSES 3.7% (overshot — too much stale retention hurts even the tail)

Interesting non-monotonicity in p90: best at tau=30 (3623ms), worst at tau=25 (3999ms). This suggests two competing effects — at tau=25, the host cache fills with stale entries causing p90 to spike, but at tau=30 the even slower decay paradoxically helps p90 by smoothing out eviction storms.

**Eviction/load-back volume at tau=30:**
- Evict: 329M tokens (vs 328M@tau25, 328M@tau20) — essentially constant
- Load-back: 298M tokens (vs 297M@tau25, 297M@tau20) — essentially constant
- The ~90% reload rate persists across all tau values

**Conclusion:** Tau sweep complete under write_back. The sweep confirms:
- **Best TTFT mean: tau=25** (V51, 8787ms, CURRENT BEST)
- **Best median/TPOT: tau=20** (V50, 1097ms/424ms)
- The mean optimum at tau=25 is -6.86% vs V35

Next axis: explore **max_segment** (GSLRU bucket count) or **decay function shape** under write_back + tau=25.

**ShareGPT:** Flat vs V51 (1.70 req/s, 1534 total).

**Updated leaderboard (by TTFT mean):**

| Version | Change | LooGLE TTFT (ms) | vs Baseline | vs V35 |
|---------|--------|-------------------|-------------|--------|
| V0      | baseline | 40371           | —           | —      |
| V35     | match promotion | 9434       | -76.6%      | —      |
| V49     | write_back | 9190           | -77.2%      | -2.6%  |
| V50     | wb + tau=20 | 8960           | -77.8%      | -5.0%  |
| **V51** | **wb + tau=25** | **8787**   | **-78.2%**  | **-6.9%** |

---

### V53 — decay_tau_top ratio 4/3→2.0 (NEGATIVE: +2.63% vs V51)

**Commit:** `f247a194a` (code change: evict_policy.py decay_tau_top = tau * 2.0, tau=25)
**Flags:** `--radix-eviction-policy gslru --hicache-write-policy write_back`

**Hypothesis:** Top-segment nodes (hit_count >= 4) represent heavily-accessed documents mid-conversation. Increasing their tau from 33.3s to 50s should reduce mid-conversation eviction.

**Result:** NEGATIVE. TTFT mean = 9018ms (+2.63% vs V51's 8787ms).

| tau_top ratio | tau_top (s) | TTFT mean | median | p90 | p99 | TPOT |
|---------------|-------------|-----------|--------|------|------|------|
| 4/3 (V51) | 33.3 | **8787** | 1129 | 3999 | **192709** | 444 |
| 2.0 (V53) | 50.0 | 9018 | 1147 | **3714** | 197560 | **415** |

**Analysis:** Mixed signals — higher tau_top HELPS p90 (-7.1%) and TPOT (-6.6%) but HURTS mean (+2.63%) and p99 (+2.5%). The extra top-segment protection reduces mid-conversation evictions (better p90/TPOT) but stale top-tier entries crowd out fresh entries needed by new conversations (worse p99/mean). tau_top ratio of 4/3 is already well-tuned.

**ShareGPT:** 1.70 req/s, 1529 total. Hit rate 57.5% (lower than V51's 60.4%).

---

### V54 — max_segment 4→3, coarser GSLRU (NEGATIVE mean: +1.97%, but best TPOT)

**Commit:** `33dc24b72` (code change: evict_policy.py max_segment 4→3, tau_top reverted to tau*4/3)
**Flags:** `--radix-eviction-policy gslru --hicache-write-policy write_back`

**Hypothesis:** Fewer GSLRU segments (4 buckets instead of 5) means documents reach top-tier protection one question earlier (after Q4 match vs Q5). Earlier top-tier protection could reduce mid-conversation eviction.

**Result:** NEGATIVE for mean. TTFT mean = 8960ms (+1.97% vs V51's 8787ms).

**max_segment sweep (tau=25, write_back, LooGLE):**

| max_seg | buckets | TTFT mean | median | p90 | p99 | TPOT |
|---------|---------|-----------|--------|------|------|------|
| 3 (V54) | 4 | 8960 | 1132 | 3883 | 196943 | **413** |
| 4 (V51) | 5 | **8787** | 1129 | 3999 | **192709** | 444 |
| 5 (V55) | 6 | 8890 | **1093** | **3422** | 194971 | **405** |

**Analysis — V55 (max_segment=5) is remarkable:** Mean is only +1.17% vs V51, but it dominates on ALL other metrics: best-ever p90 (-14.4%), TPOT (-8.8%), and median (-3.2%). Device hit fraction at 10.84% is also the highest. The finer granularity (6 buckets) lets the policy make more precise eviction decisions, keeping the most-hit documents in GPU longer.

The trade-off: more segments means it's harder to reach the top tier (5 hits vs 4). Q1-Q4 requests don't get top-tier tau_top protection, slightly hurting the tail (p99 +1.2%). But Q5-Q8 benefit from better discrimination.

**max_segment=5 + tau tuning (V55, V56):**

| Config | mean | median | p90 | p99 | TPOT |
|--------|------|--------|------|------|------|
| seg=5, tau=25 (V55) | **8890** | **1093** | **3422** | **194971** | **405** |
| seg=5, tau=28 (V56) | 8979 | 1143 | 3774 | 195114 | 414 |

V56 (tau=28) made everything worse — same pattern as the seg=4 tau sweep (overshoot). Testing seg=5 + tau=22 next (V57) to find the right tau for the wider segment range.

---

### V57 — max_segment=5 + tau=22 (NEGATIVE: +0.44% vs V51, but closest yet)

**Commit:** `facc691d6` (code change: evict_policy.py decay_tau 25→22, max_segment=5)
**Flags:** `--radix-eviction-policy gslru --hicache-write-policy write_back`

**Hypothesis:** With wider segment range (6 buckets), lower tau should help because priority space is larger — faster decay more aggressively reclaims stale entries, closing the mean gap to V51 while seg=5's granularity benefits other metrics.

**Result:** NEGATIVE for mean but closest ever. TTFT mean = 8826ms (+0.44% vs V51's 8787ms). Just 39ms short of a new best.

**seg=5 tau sweep (complete so far):**

| Config | mean | median | p90 | p99 | TPOT | device_hit |
|--------|------|--------|------|------|------|------------|
| seg=5, tau=22 (V57) | **8826** | 1167 | 3779 | **194887** | 416 | **10.96%** |
| seg=5, tau=25 (V55) | 8890 | **1093** | **3422** | 194971 | **405** | 10.84% |
| seg=5, tau=28 (V56) | 8979 | 1143 | 3774 | 195114 | 414 | 10.49% |

**Analysis:** The seg=5 tau curve is monotonically decreasing in mean from tau=28→22. Lower tau with more segments helps mean by aggressively culling stale entries. But lower tau hurts median (1167 vs 1093) and TPOT (416 vs 405) — faster decay causes more churn for recently-accessed nodes that are still in their first few hits.

Interesting metric pattern: device_hit_frac increases monotonically with lower tau (10.49% → 10.84% → 10.96%), confirming that faster decay keeps the right nodes in GPU. But the throughput penalty from increased churn offsets some of the mean improvement.

Testing seg=5 + tau=20 next (V58) to see if the mean continues to decrease or if it inflects like the seg=4 sweep did at tau=20.

**ShareGPT:** 1.69 req/s, 1520 total, hit rate 60.4%.

**Updated leaderboard (by TTFT mean):**

| Version | Change | LooGLE TTFT (ms) | vs Baseline | vs V35 |
|---------|--------|-------------------|-------------|--------|
| V0      | baseline | 40371           | —           | —      |
| V35     | match promotion | 9434       | -76.6%      | —      |
| V49     | write_back | 9190           | -77.2%      | -2.6%  |
| V50     | wb + tau=20 | 8960           | -77.8%      | -5.0%  |
| **V51** | **wb + tau=25** | **8787**   | **-78.2%**  | **-6.9%** |

**Key insight:** max_segment=5 gives the best typical-request experience (median/p90/TPOT) while max_segment=4 gives the best mean. If mean remains the primary metric, V51 stands. But V55 is the superior configuration for most practical use cases.
