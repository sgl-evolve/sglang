# kv-onyx-mf76 — SGLang HiCache KV-cache evolution report

**Researcher:** kv-onyx-mf76 (independent replicate) · **Node:** slurm2-a3nodeset0-3 (8×H100-80GB,
`--exclusive`) · **Branch:** `evolve/kv-onyx-mf76` (base commit `72812db13`) ·
**W&B:** https://wandb.ai/acjy777-dartmouth/autoevolve-sglang/runs/kv-onyx-mf76

Goal: beat the unmodified SGLang-HiCache baseline on the **fixed protocol**, **losslessly**, and
keep improving — headline metric **mean TTFT** (lower better), reporting all benchmark metrics per
dataset. Reference bar: Strata (2508.18572) + the HiCache blog.

## Fixed protocol (contract — never changed)
- Model `Qwen/Qwen3.5-397B-A17B-FP8`, TP=8; 3 tiers: GPU HBM (`--mem-fraction-static 0.85`) +
  **1 TB host total** + disk on `/mnt/localssd` (`--hicache-storage-backend file`).
  - Host budget note: `--hicache-size` is **per TP rank** in this build (`sync_fixed_hicache_size`
    syncs only across PP, not TP). The protocol's literal `1024` on TP=8 would request 8×1024 GB =
    **8 TB** — 8× over the stated 1 TB host budget and an instant OOM-kill on this 1.86 TB node
    (all 8 ranks race past the RAM check, then allocate concurrently → SIGKILL). To honor the
    **1 TB total** host ceiling I use `--hicache-size 128` (×8 = 1024 GB = 1 TB total). Applied
    identically to baseline and every version, so deltas are fair and the ceiling is respected
    (using 1024/rank would *exceed* the budget 8×, which the contract forbids).
- LooGLE via `bench_serving` (rate λ=10, `--max-concurrency 128`, 200 prompts, multiturn).
- ShareGPT via `bench_mix.py` (num_clients=60, inter-round think-time 60 s, duration 600 s).
- Full server reset (stop → free GPU+host KV → wipe disk tier) **between** the two benchmarks.
- Harness: `progress/harness/{run_eval.sh, parse_results.py, log_wandb.py}` (committed).
  Outputs/logs in `runs/` (git-ignored).

### Environment notes (reproducibility)
- Run on the node via the persistent exclusive holder slurm job; `run_eval.sh` sets the env.
- `LD_LIBRARY_PATH` must list torch + cu12 nvidia libs first and `nvidia/cu13/lib` **last**
  (deep_gemm needs cu13 `.so.13`; its `.so.12/.10` builds must not shadow torch's CUDA-12 libs).
- One-time venv repair (the locked env mixed cu12+cu13 nvidia wheels; cu13 builds clobbered cu12
  at shared paths, and the node driver is 570.195 = CUDA 12.8, which cannot run CUDA-13 kernels):
  - `nvidia-{nccl,cudnn,cusparselt}-cu13` (each 1:1 path-clobbered the cu12 build) →
    `cudaErrorInsufficientDriver` in NCCL init / cuda-graph capture. Fixed by force-reinstalling
    the cu12 builds (`nvidia-nccl-cu12==2.28.9`, `nvidia-cudnn-cu12==9.19.0.56`,
    `nvidia-cusparselt-cu12`). Verified by an 8-GPU NCCL all-reduce + a torch matmul/cudnn-conv/
    CUDA-graph-capture/flashinfer stack test on the node.
  - **deep_gemm** ships only a cu13 build; it imports (cu13 libs kept LAST on `LD_LIBRARY_PATH`)
    but its kernels need a CUDA-13 driver, so it is disabled at runtime via
    `SGLANG_ENABLE_JIT_DEEPGEMM=0`. The model therefore uses the cu12 FP8 MoE path
    (cutlass/flashinfer) instead of deep_gemm. This is applied identically to the baseline and
    every version, so all comparisons remain fair; it only shifts absolute numbers, not deltas.
- bench_mix server launch is done by `run_eval.sh` with the **identical** flags + a health gate
  (the stock `bench_mix.sh` races a no-wait client against a 10–25 min model load); the client
  driver (`bench_mix.py`) and config (60 clients / 60 s think-time / 600 s) are unchanged.

---

## Version log (one point per full-eval; W&B = evolution curve)

### v0 — unmodified baseline  (W&B version 0, commit 8fe479f18; engine = base 72812db13)
- **Hypothesis:** establish the honest reference.
- **Change:** none to the **engine**. Only environment/harness adaptations needed to run on this
  node (all in `run_eval.sh`, applied identically to every version — see STATE.md "Environment
  fixes"): cu12 lib restore (nccl/cudnn/cusparselt); deep_gemm off + flashinfer cu12 norm + cu12.8
  triton ptxas/fresh cache (cu13 JIT can't run on the 12.8 driver); page_first_direct layout
  (hybrid Mamba); hicache-size 64/rank = 1 TB total across the 2 host pools; file backend storage
  dir env -> /mnt/localssd + 2 TB disk cap; `av`; LooGLE schema converter; bench_serving usage fix.
- **Result (mean TTFT headline, lower better):**
  - **LooGLE:** mean **47.0 s**, median 10.8 s, p99 130.2 s; TPOT mean 780 ms; out 22.7 tok/s;
    req 1.49/s; e2e mean 56.7 s; completed 1545/1560; input 43.7M tok.
  - **ShareGPT:** mean **40.2 s**, p90 44.0 s, median 41.6 s; throughput 1.23 req/s; cache_hit 0.524.
- **Profile:** LooGLE QUEUE-BOUND (queue ~111 / running ~20), host L2 ~100% full, load_back ~23M tok
  (dominant), disk L3 active (backup ~5M + prefetch ~0.9M, single serial threads), device-reuse
  thrashing. ShareGPT light tiering (L2 ~10%, disk ~unused). (loogle `hit_rate` gauge logged 0.0 is
  an instantaneous-snapshot artifact; cache_metrics show real reuse: device 435K + host 181K cached.)
- **Lossless check:** baseline defines the reference outputs (lossless fingerprint captured going
  forward via `lossless_fp.py`).
- **Takeaway:** mean TTFT is dominated by LooGLE queueing/throughput under long contexts. Disk L3 is
  on LooGLE's critical path via single serial threads -> v1 parallelizes it (lossless); bigger
  levers (scheduling/anti-thrash, load_back) follow.

_(Subsequent versions appended here: hypothesis · change (files/mechanism) · result vs baseline ·
lossless check · takeaway.)_
