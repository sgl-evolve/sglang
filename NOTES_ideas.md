# quill-7m3 — idea backlog (prior art → lossless tail-TTFT levers)

Baseline bottleneck: mean TTFT ~88s vs median ~1.2s → long tail from requests
stalling on L3 (SSD) prefetch under `wait_complete`; host tier 100% full;
L3 = ~25% of hits. L3 I/O path was single-threaded serial (fixed in v1).

## Done
- **v1 (mechanism, commit 2dda8247a): parallel L3 file-backend I/O** (ThreadPool in
  HiCacheFile.batch_get/set/_batch_io_v2). Micro-bench: write 2.9x, read 1.6x @16w.
  = Mooncake/HiCache "parallel batch reads". EVAL RUNNING (job 18128).

## Ranked backlog (from Strata 2508.18572, Mooncake 2407.00079, HiCache blog, LMCache)
1. **Prefetch policy `timeout` / `best_effort`** (CONFIG, low effort, VERY HIGH). Caps
   worst-case L3 wait; admit with partial hit + recompute tail (lossless). Screen both.
   Knobs: prefetch_timeout_base / per_ki_token / max (hicache_storage.py PrefetchTimeoutConfig).
   These are `config` points — map the tail bottleneck cheaply; not novelty.
2. **Strata balanced-batch / bubble-filling** (MECHANISM, scheduler, HIGH). Defer
   "loading-bound" requests (io_latency > compute), fill bubbles with decode. Also
   "bundled hit": batch requests sharing a cache miss to load prefix once. Novel.
   Maps to scheduler.get_next_batch_to_run admission loop (scheduler.py:2861+).
3. **Read-priority / separate I/O pools + prefetch ordering** (MECHANISM, refine v1).
   Backup writes currently share v1's pool → can starve critical prefetch reads.
   Give reads their own pool + higher concurrency; process prefetch_buffer in
   scheduler-priority order (near-front-of-waiting-queue first) instead of FIFO.
   Clean, lossless, complements v1. cache_controller prefetch_io_aux_func / backup_thread.
4. **Layer-wise KV transfer overlap** (MECHANISM, model_runner, MED-HIGH). Start
   prefill with partial KV; wait layer N load, launch N+1, async store after N.
   load path already per-layer (start_loading loops layers); prefetch read is the gap.
5. **GPU-assisted I/O layout xform** (Strata §4.2, CUDA kernel, HIGH but heavy). thousands
   of threads, 128B chunks, ~50GB/s, <5% prefill hit. Needs JIT/sgl-kernel work.
6. **Transient/deferred radix nodes** (MECHANISM, radix, MED). Avoid duplicate prefill of
   same in-flight prefix; defer overlapping queued reqs. Depends on workload locality.
7. write_through_selective / write_back (CONFIG): cut L3 write volume → free SSD BW for reads.

## Notes
- Model is hybrid-Mamba → host pool page_first_direct + io-backend direct only.
- File backend uses generic v1 path (_generic_page_get→batch_get), NOT v2/zero-copy.
- host_util≈1.0 → every prefetch triggers host eviction (evict_host) before alloc;
  possible serialized contention worth profiling.

## OPERATIONAL GOTCHA (learned the hard way)
After model load (~11 min) + "Using hybrid linear attention backend / GDN kernel
dispatcher: TritonGDNKernel", there is a LONG SILENT CPU-BOUND phase (~10-20 min):
GPU util 0%, server.log frozen, /health=000, but cicc/cc1plus/ptxas/nvcc run at
~100% CPU = Triton/CUDA kernel JIT compilation during warmup. THIS IS NOT A HANG.
Do NOT kill it (I wasted node 0-2 doing so). eval.sh waits up to 45 min for /health.
Only suspect a real hang if the compiler procs are ALSO absent/idle. Verify via:
  srun --jobid=<hold> --overlap -w <node> bash -c 'nvidia-smi ...; ps -eo pcpu,comm --sort=-pcpu|head'
