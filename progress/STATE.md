# kv-onyx-mf76 — running STATE (resume anchor)

**Researcher:** kv-onyx-mf76 · node **slurm2-a3nodeset0-3** · holder slurm job **17732** (exclusive, 7d)
**Branch:** evolve/kv-onyx-mf76 · **W&B:** https://wandb.ai/acjy777-dartmouth/autoevolve-sglang/runs/kv-onyx-mf76
**Workspace:** /home/junyanch_google_com/autoresearch/programs/sglang/research-kv-onyx-mf76
Holder job id also at /home/junyanch_google_com/.kv-onyx-mf76.hold_jid

## How to run anything on the node
`srun --jobid=$(cat ~/.kv-onyx-mf76.hold_jid) --overlap -N1 bash -c '...'`
Full eval: copy run_eval.sh to /dev/shm and exec (avoids NFS stale-handle), e.g.
```
srun --jobid=$JID --overlap -N1 bash -c 'mkdir -p /dev/shm/kvonyx; cp $WORK/progress/harness/run_eval.sh /dev/shm/kvonyx/r.sh; exec bash /dev/shm/kvonyx/r.sh <version>'
```
PHASE env: AB (both, default), A (LooGLE only ~16min), B (ShareGPT only ~15min) for fast screening.
**Discipline:** do NOT edit sglang sources OR git-commit while a full/AB or B run is in flight
(ShareGPT phase re-imports sglang; git touch causes NFS stale-handle). Edits/commits only between runs.

## Environment fixes that make the protocol run (all in run_eval.sh; engine code UNMODIFIED for v0)
Driver is CUDA 12.8 but the locked venv is cu13-contaminated; fixes:
- Restored cu12 builds: nvidia-{nccl,cudnn,cusparselt}-cu12 (cu13 had clobbered them at shared paths).
- LD_LIBRARY_PATH: torch+cu12 libs first, nvidia/cu13/lib LAST (deep_gemm import needs .so.13).
- SGLANG_ENABLE_JIT_DEEPGEMM=0 (deep_gemm cu13 kernels can't run on 12.8 driver).
- FLASHINFER_USE_CUDA_NORM=1 (flashinfer norm cute/DSL is cu13; force cu12 CUDA norm).
- TRITON_PTXAS_PATH=/usr/local/cuda-12.8/bin/ptxas + fresh node-local TRITON_CACHE_DIR
  (shared ~/.triton had cu12.9/cu13 cubins -> GDN l2norm "device kernel image is invalid").
- --hicache-mem-layout page_first_direct (hybrid Qwen3.5 MambaPoolHost requires it; page_first errors).
- --hicache-size 64 PER RANK: hybrid = 2 host pools (attn+mamba), 2x64x8 = 1024 GB = 1 TB total budget
  (literal 1024 => 16 TB; 128 => 2 TB OOM).
- SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/mnt/localssd/<name> (file backend IGNORES extra_config
  file_path; defaulted to /tmp/hicache on the 194 GB root fs -> ENOSPC). Disk cap via extra_config
  max_size=2000G + min_free_space=300G (default is uncapped -> fills shared disk).
- Client: `uv pip install av`; loogle converter (new HF schema->old input/qa_pairs, grouped by doc);
  bench_serving data.get("usage") (sglang streaming chunks lack usage except final).

## v0 baseline numbers (preliminary, single-run; official AB run b2cspguf5 in flight)
- LooGLE (bench_serving rate10): mean_ttft ~42.9 s, median 34.4 s, p99 154 s, out_tput 24.8 tok/s,
  req_tput 1.62/s, completed 1545/1560.
- ShareGPT (bench_mix 60c): mean_ttft ~35.4 s, p90 39.5 s, median 36.4 s, throughput 1.34 req/s,
  cache_hit 0.545.

## Profile (the real bottlenecks)
- LooGLE: QUEUE-BOUND (queue ~111, running ~20); host L2 ~100% full; load_back ~23M tok (L2->L1,
  dominant); disk backup ~5M + prefetch ~0.9M (L3 active); device hit ~0 (thrashing: prefix evicted
  between turns and reloaded).
- ShareGPT: NOT queue-bound; host L2 ~10%; disk ~unused; load_back/backup ~0.4M.
=> Headline (mean TTFT) is dominated by LooGLE queueing/throughput. Disk L3 is on LooGLE's path
   (backup gates host turnover; prefetch gates load-from-disk) via single serial threads.

## Loop state
- [done] v0 baseline measured (official AB run finishing) -> log W&B v0 -> email baseline.
- [next] v1 = parallelize file-L3 HiCacheFile.batch_get + batch_set with ThreadPoolExecutor
  (config file_io_threads via extra_config; default serial=baseline). Lossless: disjoint files/host
  indices, thread-safe LRUFileEvictor, unique tmp names. Screen: mbench_file_io.py (cu) + PHASE=A
  quick LooGLE. Enable in run_eval via FILE_IO_THREADS=8.
  v1 patch (apply to python/sglang/srt/mem_cache/hicache_storage.py after v0):
    * import: from concurrent.futures import ThreadPoolExecutor
    * __init__: self._io_pool = ThreadPoolExecutor(int(extra_config.get("file_io_threads",0))) if >1 else None
    * batch_get/batch_set: if pool and len(keys)>1 -> submit per-key self.get/self.set, gather in order.
- [later] if v1 weak (disk < load_back): cache-aware queue ordering / anti-thrash admission;
  larger chunked-prefill; write_back vs write_through; prefetch-in-scheduled-order.
