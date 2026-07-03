# quartz-7m3 — sglang KV-cache evolution report

Researcher: **quartz-7m3** · branch `evolve/quartz-7m3` · W&B run `sgl-evolve/quartz-7m3`
Bar to beat: **v0_tuned** (mean TTFT 108824 ms). Reference/context: v0_official (87615 ms).
Headline metric: **mean TTFT** (lower better), lossless gate: outputs match no-cache.

## Environment / regime notes
- The two supervised baselines share **identical** `resolved_args` yet differ **24%** in mean TTFT
  (v0_official 87.6 s vs v0_tuned 108.8 s). Median TTFT is only ~1.2–1.4 s while p99 is ~270–322 s:
  the regime is **heavily overloaded and tail-dominated**, so mean TTFT is noisy. Wins must be large
  and robust, or confirmed across repeats.
- 3-tier state (golden run): cache hit 0.816 → device 31% / host 43% / **storage(SSD) 25%**;
  host_util ≈ 1.0 (host tier saturated); ~20.7 M tokens read from disk; load-back (host→GPU) ≈ 1.1 ms.

## v0_official — stock default  [config, reference]
Commit a334877e5. mean TTFT **87615 ms**, out_tok/s 146.9, hit 0.816. (Not re-run; logged from baseline.json.)

## v0_tuned — best stock config  [config, reference, THE BAR]
Commit a334877e5. mean TTFT **108824 ms**, out_tok/s 119.3, hit 0.821. (Not re-run; from baseline_tuned.json.)

---

## v1 — parallel L3 (disk) file I/O in HiCacheFile  [mechanism]
**Hypothesis.** The HiCache **file** storage backend (the L3/disk tier) reads and writes KV pages
**one at a time on a single prefetch thread per rank** (`HiCacheFile.batch_get`/`batch_set` →
sequential `open()+readinto()`; NVMe queue depth 1). Under `wait_complete` the scheduler blocks a
request's prefill admission until its disk-resident prefix is prefetched L3→host
(`scheduler.py:_get_new_batch_prefill_raw` → `check_prefetch_progress`), so a low-throughput L3 read
path directly inflates the TTFT tail. Raising the SSD queue depth should shrink the prefetch stall,
losslessly (identical bytes).

**What changed (files / mechanism).**
- `mem_cache/hicache_storage.py`: `HiCacheFile` gets a `ThreadPoolExecutor`; `batch_get`, `batch_set`,
  and `_batch_io_v2` issue concurrent page transfers (blocking file I/O releases the GIL). Reads land
  in distinct per-page buffers (`get_dummy_flat_data_page()` allocates fresh each call) and the LRU
  evictor is fully lock-guarded → race-free, byte-identical to serial. Added `close()`.
- `environ.py`: new `SGLANG_HICACHE_FILE_BACKEND_IO_THREADS` (`EnvInt`, default **4**).

**Calibration (free fast-screen, microbench on a3nodeset0-2 `/mnt/localssd`).** 8 concurrent
processes (= 8 TP ranks), real per-rank page size ~732 KB, cold reads (`posix_fadvise(DONTNEED)`):
aggregate 8×QD1 = **4.44 GB/s** → 8×QD4 = **5.96 GB/s** (peak, **+34%**) → 8×QD8 5.83 → 8×QD16 5.59
(oversubscribed). So 8 ranks at QD1 already reach ~aggregate QD8; per-rank parallelism adds a real but
**modest ~34%** L3-read ceiling, optimal per-rank threads = **4**. Set default accordingly.

**Result vs baseline.** _(full protocol eval pending)_

**Lossless check.** _(pending — outputs vs no-cache; bytes are identical by construction.)_

**Takeaway.** _(pending)_
