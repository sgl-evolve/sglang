# onyx-7q2 — sglang KV-cache / HiCache evolution report

Researcher: **onyx-7q2** · branch `evolve/onyx-7q2` · W&B run `sgl-evolve/onyx-7q2`
Bar to beat: **v0_official mean TTFT 87615 ms** (the better of the two given baselines; v0_tuned is
108824 ms — worse despite identical resolved_args, so I treat the official number as the honest bar).

## Baselines (reference points, not re-run)
| version | tag | mean TTFT (ms) | median TTFT | p90 TTFT | out tok/s | hit | L3 hit |
|---|---|---|---|---|---|---|---|
| v0_official | baseline | 87615 | 1224 | 243466 | 146.9 | 0.816 | 0.254 |
| v0_tuned | baseline | 108824 | 1438 | 295413 | 119.3 | 0.821 | 0.259 |

**Baseline diagnosis.** median TTFT ≈ 1.2 s but mean ≈ 87.6 s and p90 ≈ 243 s → the mean is dominated
by a tail of the ~25% of requests that must be read back from the SSD/L3 tier. `host_util ≈ 1.0` and
`evict_tokens 574M` ≫ working set → heavy host↔disk thrash.

**Root cause (from code map).** L3 disk reads are fully serialized: a single controller
`prefetch_io_aux_thread` feeds `HiCacheFile.batch_get`, which is a serial Python loop of
`open()+readinto()+close()` per 64-token page. A long prefix = hundreds of sequential blocking
syscalls on one thread, shared across all concurrent requests. Under `wait_complete` each L3 request
blocks on its whole serial read (→ p90 243 s) while holding host budget + `protect_host` locks that
throttle everyone. Prior art (Strata, arXiv 2508.18572): schedulers are "loading-bound, not
compute-bound"; fixes are load-aware scheduling + GPU-assisted IO. Our model is hybrid-Mamba →
`direct` io-backend only, so Strata's GPU-assisted-IO kernel is unavailable; the IO parallelism and
scheduling levers remain.

---

## v1-parallel-reads — parallelize L3 disk reads  [mechanism]
**Hypothesis.** The L3 tail is bounded by serial single-thread disk reads. File-read syscalls release
the GIL, so reading a batch's pages through a thread pool gives real NVMe parallelism, draining the
prefetch queue ~N× faster and collapsing the wait_complete tail — losslessly (same bytes, just read
concurrently; order preserved).

**Change (pure Python, no recompile).**
- `mem_cache/hicache_storage.py`: `HiCacheFile` gets a persistent `ThreadPoolExecutor`; `batch_get`
  reads pages via `pool.map` (ordered → results[i] still maps to keys[i]; distinct target buffers;
  evictor is internally locked → race-free). Serial fallback when threads≤1 or batch≤1.
- `environ.py`: new `SGLANG_HICACHE_FILE_READ_THREADS` (EnvInt, default 16).

**Screens (free, no model load).**
- Correctness: content + order preserved vs serial, incl. shuffled-key test → lossless at read layer. ✅
- IO microbench on real `/mnt/localssd` (page cache dropped so reads hit NVMe): serial 1025 MB/s →
  **6126 MB/s @ 16 threads (~6×)**; 8≈5.8×, 32≈5.6×, 64≈4.9× (over-parallelizes). Default 16. ✅

**Result vs baseline.** _(full eval pending)_
**Lossless check.** _(pending — bench output vs no-cache)_
**Takeaway.** _(pending)_
