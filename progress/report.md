# kv-onyx-mf76 — SGLang HiCache KV-cache evolution report

**Researcher:** kv-onyx-mf76 (independent replicate) · **Node:** slurm2-a3nodeset0-3 (8×H100-80GB,
`--exclusive`) · **Branch:** `evolve/kv-onyx-mf76` (base commit `72812db13`) ·
**W&B:** https://wandb.ai/acjy777-dartmouth/autoevolve-sglang/runs/kv-onyx-mf76

Goal: beat the unmodified SGLang-HiCache baseline on the **fixed protocol**, **losslessly**, and
keep improving — headline metric **mean TTFT** (lower better), reporting all benchmark metrics per
dataset. Reference bar: Strata (2508.18572) + the HiCache blog.

## Fixed protocol (contract — never changed)
- Model `Qwen/Qwen3.5-397B-A17B-FP8`, TP=8; 3 tiers: GPU HBM (`--mem-fraction-static 0.85`) +
  1 TB host (`--hicache-size 1024`) + disk on `/mnt/localssd` (`--hicache-storage-backend file`).
- LooGLE via `bench_serving` (rate λ=10, `--max-concurrency 128`, 200 prompts, multiturn).
- ShareGPT via `bench_mix.py` (num_clients=60, inter-round think-time 60 s, duration 600 s).
- Full server reset (stop → free GPU+host KV → wipe disk tier) **between** the two benchmarks.
- Harness: `progress/harness/{run_eval.sh, parse_results.py, log_wandb.py}` (committed).
  Outputs/logs in `runs/` (git-ignored).

### Environment notes (reproducibility)
- Run on the node via the persistent exclusive holder slurm job; `run_eval.sh` sets the env.
- `LD_LIBRARY_PATH` must list torch + cu12 nvidia libs first and `nvidia/cu13/lib` **last**
  (deep_gemm needs cu13 `.so.13`; its `.so.12/.10` builds must not shadow torch's CUDA-12 libs).
- One-time venv repair: the locked env installed both `nvidia-nccl-cu12` and `nvidia-nccl-cu13`
  (2.28.9) to the *same* `nvidia/nccl/lib/libnccl.so.2`; the cu13 build clobbered it → CUDA-13
  NCCL on a 12.8 driver → `ncclCommInitRank` "driver insufficient". Fixed by force-reinstalling
  `nvidia-nccl-cu12==2.28.9`. Verified by an 8-GPU NCCL all-reduce.
- bench_mix server launch is done by `run_eval.sh` with the **identical** flags + a health gate
  (the stock `bench_mix.sh` races a no-wait client against a 10–25 min model load); the client
  driver (`bench_mix.py`) and config (60 clients / 60 s think-time / 600 s) are unchanged.

---

## Version log (one point per full-eval; W&B = evolution curve)

### v0 — unmodified baseline  *(running)*
- **Hypothesis:** establish the honest reference.
- **Change:** none (pristine fork @ 72812db13, protocol flags exactly as specified).
- **Result:** _pending — will fill mean/median/p99 TTFT, TPOT, ITL, throughput, hit rate for
  LooGLE + ShareGPT, and log as W&B version 0._
- **Lossless check:** baseline defines the reference outputs.
- **Takeaway:** _pending._

_(Subsequent versions appended here: hypothesis · change (files/mechanism) · result vs baseline ·
lossless check · takeaway.)_
