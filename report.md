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

## Versions
(none logged yet beyond the two references; v1-besteffort eval queued on the shared pool)
