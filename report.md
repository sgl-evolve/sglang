# kv-heron-eb9 — sglang KV-cache evolution report

Researcher: **kv-heron-eb9** · branch `evolve/kv-heron-eb9` · W&B run `kv-heron-eb9` in `sgl-evolve`
Bar to beat: **v0 official** mean TTFT = 87615 ms (the stronger of the two references).

## Fixed protocol (contract — never changed)
Qwen3.5-122B-A10B-FP8 (hybrid-Mamba MoE), TP=8, ctx 262144, mem-frac 0.85, HiCache 3-tier
(GPU HBM + 768 GB host [`--hicache-size 96`] + 1.8 TB disk `file` backend on /mnt/localssd),
io-backend `direct` + layout `page_first_direct` (forced by MambaPoolHost), page_size 64.
Load: real-text Mooncake 1:1:1 mix (1553 convs, ~19 M tok), loogle loader, λ=3.5,
max-concurrency 128, num-prompts 1553. Headline: mean TTFT (lower better). Lossless gate.

## Reference points (logged, not re-run)
| ver | tag | ttft_mean ms | ttft_p50 | ttft_p90 | ttft_p99 | out_tok/s | hit | L3 frac | host_util |
|-----|-----|-------------|----------|----------|----------|-----------|-----|---------|-----------|
| v0_official | baseline | 87615 | 1225 | 243466 | 270799 | 146.9 | 0.816 | 0.254 | 1.00 |
| v0_tuned    | baseline | 108824 | 1438 | 295413 | 322801 | 119.3 | 0.821 | 0.259 | 1.00 |

Note: the "tuned" reference is *worse* than official on TTFT & throughput despite identical
resolved_args — implies meaningful run-to-run variance and/or an unrecorded config change.
I treat **official (87.6 s)** as the honest bar.

## Regime analysis (from the baseline metrics + code study)
- **TTFT-wait-dominated:** median TTFT 1.2 s but p90 243 s; e2e_mean 108 s ≈ TTFT 87 s + ~21 s decode.
  With max-concurrency 128 (asyncio semaphore → ≤128 requests ever in-server), TTFT is the time a
  request waits for its prefill to be scheduled+run behind the other ≤128 concurrent requests.
  → Reducing mean TTFT requires **higher prefill/decode goodput**, not just reordering.
- **Not disk-BW-bound (screened, job 18117):** single-thread read of ~732 KB pages on /mnt/localssd
  already hits **4.9 GB/s**; thread-pool parallelism only ~1.3×. Reading all 20.7 M disk tokens costs
  ~48 s spread over ~105 min. So the naive per-page serial file backend is NOT the bottleneck. (This
  screened out a "parallelize disk IO" idea before spending a full eval.)
- **Transfers already overlapped:** H→D load-back runs on a separate `load_stream` with a layer-wise
  `LayerDoneCounter` (cache_controller.start_loading), and this works for the `direct` backend too.
- **LPM always active:** the `LPM→FCFS` fallback triggers only at waiting_queue > 128, which the
  semaphore prevents — so cache-aware scheduling is on. (Screened out a "restore LPM under overload" idea.)
- **Host tier saturated / heavy tier movement:** host_util 1.00; of 81.5 M cached prompt tokens,
  device 31%, host 43%, disk 25%. Per-run movement: evict 574 M, load-back 444 M, prefetch 166 M,
  offload 145 M (summed over 8 ranks).

## Prior art (Strata arXiv 2508.18572 + HiCache blog) — what's already done vs headroom
- HiCache's **layer-wise transfer/compute overlap is a `kernel`-backend feature** (GPU-assisted IO).
  This model is forced onto `direct` (MambaPoolHost only supports page_first_direct), so we may miss
  the fastest transfer path. Protocol explicitly flags "make kernel/page_first work for Mamba" as a
  legit direction.
- **Strata's central finding: serving is LOADING-BOUND, not compute-bound.** On LooGLE with SGLang CPU
  offloading, **"74% of prefill time is blocked on KV transfers"** (host→device load-back). Even with
  optimized IO (~75% PCIe) up to 24% of prefill stays stalled on load. Our workload *includes LooGLE*
  and uses HiCache offloading → we are likely H→D-loading-bound.
- Strata ablations: **IO efficiency = the biggest lever (+76–95% throughput)**; scheduling +1.8×.
  - GPU-assisted IO: one CUDA kernel, 1000s of threads, 128B granularity, free layout transforms,
    ~50 GB/s CPU→GPU confined to ≤2 SM blocks (<5% prefill / 10% decode degradation).
  - Cache-aware scheduling: (a) **defer** on delay-hit via transient HiRadix nodes; (b) **balanced
    batches** — skip requests whose load/compute ratio > ~100, backfill later; (c) **bundle hits** —
    batch a compute-heavy with a load-heavy request so PCIe-load overlaps HBM-compute; (d) **bubble
    filling** — run a DECODE batch during a long context load (decode saturates HBM BW, load saturates
    PCIe → overlap with little contention).
- Two distinct load paths (don't conflate): **disk→host prefetch** (governed by prefetch policy;
  what v1-besteffort tests) vs **host→device load-back** (layer-wise-overlapped on load_stream; where
  the 74% stall lives). The dominant stall is likely H→D load-back, which best_effort does NOT touch.

## Leading hypotheses (to test)
0. **Loading-bound on H→D load-back** (Strata's 74%): prefill waits on host→device transfer of long
   reused prefixes (load ≫ compute per layer, so layer-wise overlap can't hide it). If confirmed by
   batch dynamics (prefill batches with huge #cached-token, small #new-token, GPU idle during load),
   the mechanism targets H→D: faster transfer, or Strata-style stall-hiding/bundling, or device retention.
1. **Prefetch-wait starves the running batch.** Under the baseline `wait_complete`, `get_new_prefill_batch`
   skips (`continue`) any request whose storage prefetch isn't fully done; with ≤128 concurrent, if a
   chunk are prefetch-pending, the running batch shrinks → lower throughput → higher TTFT.
   → **v1-besteffort** (config): `--hicache-storage-prefetch-policy best_effort`. Diagnostic + possible win.
2. If (1) helps: build a **partial-hit prefill overlap** mechanism (admit resident prefix immediately,
   stream disk KV in the background) — best_effort's batch-fill without its recompute cost (Strata-style).
3. If (1) is neutral: pivot to **throughput/memory** mechanisms (fit more running KV / reduce per-step overhead).

## Environment blocker (RESOLVED) — flashinfer allreduce-fusion workspace deadlock
On a **fresh workspace / cold cache**, every server launch deadlocked during warmup: rank 0 stalls
right after "GDN kernel dispatcher" (all threads `hrtimer_nanosleep`, 0% GPU, VmLck=16KB — a
wait-for-collective, not I/O), while other TP ranks reach the FlashInfer allreduce-fusion **trtllm
workspace** NCCL setup and time out after 600 s ("Disabling flashinfer allreduce fusion permanently"),
never recovering. Reproduced on 3 nodes (0-2, 1-2, 1-1) and independent of `--disable-cuda-graph` /
`--disable-custom-all-reduce`. venv matches the lockfile exactly.
**Fix:** launch with **`--enforce-disable-flashinfer-allreduce-fusion`** → server reaches "ready" in
~5 min post-load. Since `enable_flashinfer_allreduce_fusion` is already `False` (default), this only
skips a *failing* init and does **not** change the compute path → comparable to a healthy golden run
(to be validated by v1-basefix ≈ golden 87.6 s). **I pass this flag on every eval.** Also confirmed:
first-run kernel compile is slow (~15-30 min) but the per-workspace `$WORK/.cache` (on shared NFS) now
warm, so subsequent evals are fast.

## Versions
- **v1-basefix** (config): baseline eval.sh config + `--enforce-disable-flashinfer-allreduce-fusion`.
  Purpose: anchor my comparable reference (validate vs golden 87.6 s) + capture batch dynamics. [running on ondem-3]

### Batch-dynamics diagnosis (from v1-basefix server.log, live) — the real bottleneck
- **Prefill:decode batches ≈ 1165:16** → overwhelmingly **prefill-bound**; decode starved (⇒ low out_tok/s).
- **Device KV pool only ~34% used** (`full token usage 0.33-0.42`), yet the baseline serves **43% of hits
  from host + 25% from disk**. So HiCache keeps device "lean" (device=active KV, host=cached prefixes)
  and **66% of device capacity sits idle** while reused prefixes pay H→D load-back (and disk reads).
- ~120 running req (near max-concurrency 128), small queue (7-11). Not memory-capacity-bound.
- Most prefill batches are all-new 6144-token chunks; cache-hit batches load big #cached-token bursts.
- **Leading mechanism (data-grounded, lossless): retain hot cached prefixes on the under-used device
  tier** (use the free 66%) → convert host/disk hits into device hits → fewer H→D load-backs & disk
  reads → faster prefill → lower TTFT. Candidate #2: prefill/decode scheduling balance.
