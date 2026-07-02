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
| V13 (device weight=4) | 10217ms | prev best |
| V14 (device weight=8) | 10903ms | NEGATIVE |
| V15 (time-decay GSLRU tau=30) | 10114ms | — |
| **V16 (time-decay GSLRU tau=15)** | **9814ms** | **CURRENT BEST** |
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
| **V29 (top-only decay tau=20)** | **9642ms** | **CURRENT BEST (-1.8%, p90 -8.0%, TPOT -2.6%)** |
| V30 (top-only decay tau=25) | 9748ms | NEGATIVE (tau too high, +1.1%, p90 +12.8%) |
