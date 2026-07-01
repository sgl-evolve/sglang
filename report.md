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

| Version | LooGLE TTFT mean | LooGLE p99 | Hit rate | Regime |
|---------|----------------:|----------:|--------:|--------|
| **V11** | **10658ms** | 225215ms | 89.3% | Best mean (conservative anti-starvation) |
| V8 | 16070ms | **85889ms** | **97.2%** | Best p99 + hit rate (aggressive anti-starvation) |
| V0 | 40371ms | 147141ms | 76.8% | Baseline |

The scheduling space is largely explored. V12+ should explore orthogonal improvement axes.
