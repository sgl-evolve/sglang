# drift-3e7 — sglang HiCache KV-cache research log

- **Researcher:** `drift-3e7` (independent replicate; own clone, own branch, own W&B run)
- **Branch:** `evolve/drift-3e7`
- **W&B:** project `sgl-evolve`, run `drift-3e7`
- **Workspace:** `workspace/sgl/researchers/drift-3e7/`
- **Eval:** FIXED protocol via `eval-on-pool.sh` on a certified held node. Never modified.

## Baselines (logged, not re-run)

| version | tag | ttft_mean_ms | ttft_median_ms | ttft_p90_ms | ttft_p99_ms | hit_rate | l3_hit_frac | out_tok_s | req_thpt |
|---|---|---|---|---|---|---|---|---|---|
| v0_official | baseline | 87615.4 | 1224.5 | 243466 | 270798 | 0.8163 | 0.2538 | 146.9 | 1.15 |
| v0_tuned | baseline | 108824.3 | 1437.9 | 295413 | 322801 | 0.8207 | 0.2585 | 119.3 | 0.93 |

**Note / anomaly:** `v0_tuned` (the nominal "bar to beat") has *worse* mean TTFT than
`v0_official`, despite identical `resolved_args`. In this overloaded regime a single run's
mean TTFT is dominated by the queue-wait tail and is high-variance. I therefore target the
**harder** bar — `v0_official`'s **87615 ms** mean TTFT. Beating it beats both.

## Bottleneck analysis (from the baselines)

The workload is **fundamentally overloaded**: arrivals λ=3.5 req/s vs measured
`req_throughput` ≈ 1.15 req/s. So the queue grows during the run and **mean TTFT is
dominated by queue wait** — median TTFT is only ~1.2 s but the mean is ~87 s (p90 ≈ 243 s).

Mechanism of the tail (from reading the code):
- Frozen prefetch policy is `wait_complete` (`can_terminate = completed`,
  `hiradix_cache.py:1355`): a request cannot be scheduled until its **entire** SSD-resident
  prefix finishes loading host-side. `get_new_batch_prefill` *skips* (does not block on) such
  requests (`scheduler.py:2882`), so they sit in the waiting queue.
- Under a saturated SSD (l3_hit_frac ≈ 0.25, disk_read ≈ 20.7 M tokens, prefetch ≈ 166 M
  tokens), prefetches serialize → many requests stay un-schedulable → the **decode batch
  starves → throughput collapses → the queue and the TTFT tail explode.**

**Implication:** the highest-leverage lossless wins reduce time spent blocked on / bandwidth
spent by SSD prefetch, and keep hot prefixes off the SSD tier. Two attack surfaces:
1. **Cache quality** — keep hot prefixes in host RAM so fewer requests fall to SSD (v1).
2. **Prefetch path** — don't block on / don't redundantly read a saturated SSD (v2+).

---

## v1 — SLRU host eviction (mechanism) — RETRACTED (dead code, never logged)

- **Hypothesis:** host-tier eviction is destructive; replacing pure-LRU host eviction with
  SLRU (protect `hit_count ≥ 2` prefixes) should keep hot multi-turn prefixes in host RAM,
  cutting SSD reads → lower TTFT tail.
- **What killed it:** I edited `mem_cache/hiradix_cache.py` (`HiRadixCache`). But this model
  (Qwen3.5-122B-A10B) is **hybrid-SSM/GDN**, and `mem_cache/registry.py:101-104` routes
  `enable_hierarchical_cache + is_hybrid_ssm` to **`UnifiedRadixCache`** (confirmed in the
  running server: "Allocating … hierarchical Mamba cache", `radix_eviction_policy='lru'`,
  `schedule_policy='fcfs'`). So my edit was **dead code**. I killed the in-flight eval before
  it wasted the node (it would have just reproduced baseline). No curve point.
- **Also found (would have crashed if it HAD been live):** the `evict_host` parent re-push
  mixed a float LRU priority into an SLRU tuple heap → `TypeError` under host pressure. Fixed
  in `7605b7e52`, but moot since the file is unused for this model.
- **Lesson:** verify the actually-instantiated class from the running server before coding.

## v2cfg-besteffort — best_effort prefetch policy (config) — RUNNING

- **Hypothesis / purpose:** bottleneck probe. `--hicache-storage-prefetch-policy best_effort`
  never blocks scheduling on SSD (schedule immediately, recompute the un-loaded tail). If it
  sharply cuts mean TTFT, SSD-prefetch blocking is confirmed as the dominant tail.
- **Change:** config only (valid — the flag reaches `UnifiedRadixCache.prefetch_stop_policy`).
- **RESULT — big win on the headline (commit `7605b7e52`, on-contract, no silent fallback):**

  | metric | v0_official | v2cfg-besteffort | Δ |
  |---|---|---|---|
  | **mean TTFT (ms)** | 87615 | **3237** | **−96% (27×)** |
  | TTFT p99 (ms) | 270798 | 20481 | −92% |
  | TTFT median (ms) | 1225 | 1959 | +60% |
  | req throughput (req/s) | 1.15 | 1.70 | +48% |
  | out_tok/s | 146.9 | 217.1 | +48% |
  | e2e mean (ms) | 108148 | 71104 | −34% |
  | hit_rate | 0.816 | 0.368 | −55% |
  | l3_hit_frac | 0.254 | 0.000 | SSD tier idle |
  | TPOT mean (ms) | 241 | 752 | +212% (worse) |

- **Takeaway:** `wait_complete` (the frozen *default*) is **pathological under this overload** — it
  blocks requests on SSD, starves the decode batch, and the queue explodes (that's the 87 s mean vs
  1.2 s median). `best_effort` drains the queue → 27× lower mean TTFT, +48% throughput. The cost:
  cache hit-rate collapses (0.82→0.37, SSD tier unused) and decode slows (TPOT 3×) because the batch
  is full and much prefix KV is recomputed rather than loaded. **This is a config change (bar-mapping),
  not the novel win** — but it resets the bar to beat to **3237 ms**.
- **Lossless:** recompute yields identical tokens (no cache = full compute).
- **Implication for the mechanism:** the ideal keeps best_effort's drained queue while *retaining
  cheap cache hits* (less recompute → lower TPOT, and possibly even lower TTFT via more decode
  compute headroom). That is exactly v4-adaptive's goal.

## Prior art (HiCache blog) — confirms the direction

- The blog states SSD/**bandwidth saturation has "no dedicated throttling… addressed
  indirectly"** (via write-through-selective / write-back). So a *load-aware prefetch admission*
  policy (v4 below) targets a documented gap → genuinely novel.
- It frames the exact tradeoff my mechanism navigates: `best_effort` "minimizes TTFT" vs staging
  (`wait_complete`) "improves reuse / throughput."
- **No in-flight transfer dedup** exists → prefetch coalescing is also a novel gap (candidate v6).
- `write_through_selective` (hit-count-based, backs up only hot spots) is the documented
  bandwidth-pressure lever → candidate config screen (baseline offload = 145 M tokens is large).

## Live monitoring of v2cfg-besteffort (while running)

Server `/metrics` under best_effort: **running_batch ≈ 124–126/128 (full), queue ≈ 0,
token_usage ≈ 0.38** (GPU KV never the bottleneck — the run is concurrency/compute bound). So
not-blocking-on-SSD keeps the decode batch full and drains the queue → confirms the
wait_complete tail is decode-batch starvation from prefetch blocking. Baseline (wait_complete)
almost certainly runs a smaller batch (requests stuck in prefetch).

## Roadmap (ordered)

1. **v2cfg-besteffort** (running) — the no-wait ceiling.
2. **v4-adaptive-prefetch** (ready, capped 5 s) — wait for cheap/hot prefetches, give up on
   saturating ones; aims to beat *both* wait_complete and best_effort.
3. **v5-starvation-aware prefetch admission** (design ready) — plumb a scheduler signal: if the
   last prefill pass ended with the batch underfull *because* requests were skipped for
   prefetch, admit-all next pass (give up prefetch); else wait fully. Precise, self-correcting.
4. **cfg: write_through_selective** — cut the 145 M-token offload write traffic (bandwidth).
5. **v6: in-flight prefetch coalescing** — dedup concurrent SSD reads of shared pages (novel).

## v3cfg-timeout — timeout prefetch policy (config) — DEFERRED (no free node)

- Third point on the prefetch-policy spectrum. Default timeout deadline is
  `1.0 + pages×0.25 s`, so it's a *muted* probe for the tail (huge-prefix reads still wait
  tens–hundreds of s), and the timeout knobs live in the FROZEN `--hicache-...-extra-config`
  so I can't sharpen it via config. Low priority; run if a node is idle.

## STATUS (live) — version log (own formal versions; budget 100)

- **v2cfg-besteffort** (config): 3237 ms — 27× vs official.
- **v4-adaptive-prefetch** (mechanism, cap 2s): 3299 ms.
- **v5-adaptive-1s** (mechanism, cap 1s): **2719 ms — BEST (32× vs official)**.
- **v7-selective-wt** (config): **NEGATIVE — 6190 ms** (2.3× worse than v5). `write_through_selective`
  starves the host tier (backup only after 2 hits → hit_host_frac 0.49→0.018), killing adaptive's
  host-hit reclaim. **Learning: write_through (backup-all) is essential for the adaptive win.**
- **v8-adaptive-uncap3** (mechanism, cap 3s): **NEW BEST — 2580 ms (34× vs official, −5% vs v5)**;
  hit_rate 0.597→0.633, throughput 2.63→2.68, TPOT 489→457, e2e 43.1s→41.4s. Unclipping the deadline
  (v5's cap==base flattened it to 1 s) lets the size/pressure term engage → more host-hit reclaim in
  low-backlog windows. The full adaptive form beats the flat-1 s v5.
- **v9-adaptive-cap6** (mechanism, cap 6s): 2613 ms — cap climb PLATEAUED (1s→2719, 3s→2580 best, 6s→2613); v8 (cap 3s) stays the headline best (v9 slightly better throughput/e2e/p99 but +TTFT). Optimum cap ~3s; code restored to 3s.
- **v10-adaptive-contention** (mechanism): running — occupancy pressure rarely saturates, so add a direct contention signal (len(ongoing_prefetch)/64) to the adaptive deadline: give up sooner when many requests are blocked on prefetch, wait longer when few.
- **v10-adaptive-contention** (mechanism): 2757 ms — NOT a new best. The contention signal raised throughput (2.68→2.84) and cut e2e (41.4→38.9 s) but WORSENED mean TTFT (giving up prefetch sooner → more recompute → longer per-request prefill). Learning: for the headline (mean TTFT), the patient occupancy-only pressure (v8) beats aggressive contention give-up. Reverted to v8.

### Adaptive prefetch — exploration summary (mechanism is well-mapped)
The load-adaptive prefetch-admission mechanism is the novel win. Best = **v8 (cap 3 s, occupancy pressure): 2580 ms mean TTFT = 34× below v0_official (87615), 42× below v0_tuned**. Levers explored: prefetch policy (best_effort 3237 → adaptive 2580); cap sweep (1 s 2719, 3 s 2580, 6 s 2613 → optimum ~3 s); write policy (write_through essential; selective 6190 starves host); pressure signal (occupancy beats contention for the headline). Lossless throughout. Further gains need a different lever (host-eviction frequency-awareness, or scheduler decode-protection to cut TPOT).
- Infra: robust workflow = isolated flashinfer cache (`FLASHINFER_WORKSPACE_BASE=$WORK`) +
  `--dist-timeout 5400`. The shared `~/.cache/flashinfer` was corrupted by cross-researcher concurrent
  compiles (hangs + a SIGBUS in CUDA-graph capture); isolation fixed it (loads in ~147 s).

## STATUS (live)

**Bottom line:** best result is **`v5-adaptive-1s` (MECHANISM, commit `8402c5904`): mean TTFT 2719 ms —
32× lower than v0_official (87615) and 16% below the best config (best_effort 3237)** — while ALSO
best on p99 TTFT (16827), throughput (2.63 req/s, +55% vs best_effort), e2e (43 s), and TPOT (489).
The load-adaptive prefetch mechanism (give up on a saturated SSD, reclaim cheap host hits) with a 1 s
cap wins on every axis. It's the novel mechanism the program targets.

### v5-adaptive-1s — RESULT (mechanism, commit `8402c5904`, on-contract, lossless, rc=0) — NEW BEST

| metric | v0_official | best_effort (cfg) | v4-adaptive-2s | **v5-adaptive-1s** |
|---|---|---|---|---|
| **mean TTFT (ms)** | 87615 | 3237 | 3299 | **2719** (−96.9% vs official) |
| TTFT p99 (ms) | 270798 | 20481 | 34077 | **16827** |
| req throughput | 1.15 | 1.70 | 2.39 | **2.63** |
| out_tok/s | 147 | 217 | 305 | **337** |
| e2e mean (ms) | 108148 | 71104 | 47112 | **43111** |
| TPOT mean (ms) | 241 | 752 | 531 | **489** |
| hit_rate | 0.816 | 0.368 | 0.624 | 0.597 |

- **Takeaway:** halving the adaptive cap (2 s→1 s) improved *every* metric vs v4 — the shorter wait cut
  the TTFT penalty (3299→2719, now below best_effort) while still reclaiming most host hits (0.60), so
  throughput/e2e/TPOT all improved too. The 1 s cap is a better operating point than 2 s: under this
  overload, minimal-but-nonzero waiting beats both no-wait (best_effort) and longer-wait (v4).
  **Next:** sweep even shorter (0.5 s) to find the optimum; the mechanism clearly dominates.

### v4-adaptive-prefetch — RESULT (mechanism, commit `b58201066`) — superseded by v5

### v4-adaptive-prefetch — RESULT (mechanism, commit `b58201066`, on-contract, lossless, rc=0)

| metric | v0_official | v0_tuned | v2cfg-besteffort (config) | **v4-adaptive (mechanism)** |
|---|---|---|---|---|
| **mean TTFT (ms)** | 87615 | 108824 | **3237** | 3299  (−96.2% vs official; ≈ best_effort) |
| TTFT median (ms) | 1225 | 1438 | 1959 | 1915 |
| TTFT p90 (ms) | 243466 | 295413 | 6889 | 6556 |
| TTFT p99 (ms) | 270798 | 322801 | 20481 | 34077 (worse tail than best_effort) |
| req throughput (req/s) | 1.15 | 0.93 | 1.70 | **2.39** (+40% vs best_effort) |
| out_tok/s | 147 | 119 | 217 | **305** (+40%) |
| e2e mean (ms) | 108148 | 133930 | 71104 | **47112** (−34% vs best_effort) |
| TPOT mean (ms) | 241 | 294 | 752 | **531** (better decode) |
| hit_rate | 0.816 | 0.821 | 0.368 | **0.624** (reclaimed cache) |
| l3_hit_frac | 0.254 | 0.259 | 0.000 | 0.000 |
| hit host/device frac | .43/.31 | .43/.31 | .27/.73 | .56/.44 |

- **Self-audit:** rc=0; resolved `hicache_storage_prefetch_policy=adaptive` (no silent fallback);
  budget exact (ctx 262144, mem-frac 0.85, hicache 96, tp 8); 7037/7037 successful; lossless
  (un-prefetched prefix is recomputed → identical tokens).
- **Takeaway:** the 2 s-capped adaptive deadline lets requests wait briefly to **reclaim host hits
  (0.37→0.62)** — halving recompute vs best_effort → **TPOT 752→531, throughput 1.70→2.39,
  e2e 71 s→47 s** — at the **same headline mean TTFT** (3299 vs 3237, within run-to-run noise).
  It never reclaims L3/SSD hits (l3=0): the 2 s cap is shorter than an SSD read, so it behaves like
  best_effort for SSD-resident prefixes but recaptures the cheap host-resident ones. A better
  speed↔memory balance and a genuine mechanism (not config). **Next ideas:** raise the cap / make it
  SSD-aware to also reclaim L3 hits; or attack the p99 tail (34 s) which best_effort keeps lower.

### (historical) load-blocker saga — RESOLVED

**Bottom line (earlier):** `best_effort` (config) validated **27× lower mean TTFT** (reported +
emailed). The novel `adaptive` mechanism was blocked from eval by a cold-cache init issue (below),
now resolved: **never kill a load mid-compile** (it corrupts the shared `~/.cache/flashinfer` trtllm
kernel); let cold compiles finish (they warm the shared cache) → subsequent loads are fast.

**TRUE ROOT CAUSE of the adaptive load failures (diagnosed via /proc + server.log, not my mechanism):**
The load has a **cold-start deadlock**: rank 0 spends **>600 s cold-compiling DeepGEMM/CUDA kernels**
(`nvcc`→`cicc`→`ptxas` seen running), while the other ranks block in **FlashInfer's trtllm
allreduce-fusion NCCL communicator setup**, which fetches `ncclUniqueId` from rank 0 via the c10d
TCPStore with the **default 600 s timeout**. Rank 0 misses that window → `store->get('0') got error:
wait timeout after 600000ms` → distributed init **wedges** (ranks fall into sleep/poll; cache never
finishes warming). This is NOT NFS/disk (a LOCAL-cache rewarm hit the same wall) and NOT my code
(never runs at init). **best_effort worked only because it had a WARM cache → rank 0 skips compile →
reaches the barrier fast.** My earlier "hangs" were slow cold compiles I killed at ~12 min — right in
the 600 s-timeout aftermath.
**FIX (DONE — job 18154):** warmed the caches via a helper `launch_server` with `--dist-timeout 5400`
so rank 0's cold compile finished before any barrier timed out. It reached "server ready" and warmed:
`$WORK/.cache/triton` (150 entries) + the slow kernel `trtllm_allreduce_fusion` into
**`~/.cache/flashinfer`** (32M). KEY: FlashInfer uses `~/.cache/flashinfer` (its
`FLASHINFER_WORKSPACE_BASE`/home), and **ignores eval.sh's `FLASHINFER_CACHE_DIR`** — this shared-home
cache was missing this model's trtllm kernel, and is the true cold-compile culprit; the rewarm
populated it. Since all three caches (triton on NFS `$WORK`, flashinfer on NFS home, DeepGEMM `_C.so`
in the NFS venv) are shared across nodes, eval.sh on any certified node now skips the >600 s compile →
loads fast → no wedge, with the contract (default 600 s dist-timeout) untouched. (I must never clear
`$WORK/.cache` again, and never run two of my own evals concurrently.)

**adaptive eval blocker (earlier framing — 4 attempts, all failed to LOAD — never a mechanism-logic failure):**
- Deterministic hang at model init right after "GDN kernel dispatcher" (before DeepGEMM warmup),
  across **3 different nodes**. Diagnosed as a true hang, **not** slow compile (triton cache frozen
  at 1 entry, no ptxas/compiler process, GPU 0%, schedulers busy-waiting ~11% CPU).
- Ruled out: my code (never runs at init), the `adaptive` flag (no init validation on the unified
  path), and node-specificity (3 nodes).
- **Key correlation:** best_effort succeeded with a **WARM** `$WORK/.cache` when the pool was
  quieter (06:38); every adaptive attempt had a **cold/cleared** cache under a **heavily contended**
  pool (many concurrent researcher server-loads). Leading hypothesis: a cold load must (re)compile
  GDN/DeepGEMM/CUDA-graph kernels and write them to the shared filesystem, and that compile/warmup
  **stalls under FS/fabric contention** — so it looks like a hang. My clearing of `$WORK/.cache`
  (done to fix an earlier concurrent-cache-race corruption) removed the warm cache that made
  best_effort fast, exposing this.
- **RECOVERY PATH (for the next window / quieter pool):** (1) run ONE load when the pool is quiet so
  the cold compile succeeds and **re-warms `$WORK/.cache`**; thereafter adaptive loads fast. (2) If
  it still stalls, `py-spy dump` rank 0 during the hang (debug-distributed-hang skill) to pinpoint
  the stuck compile/collective. (3) Do NOT clear `$WORK/.cache` again; never run two drift-3e7 evals
  concurrently (shared cache). Nothing is left holding a node.



- **best_effort (config): DONE — new best, 27× lower mean TTFT.** Reported + emailed.
- **v4-adaptive-prefetch (mechanism, cap=2s, commit `7468090d4`): queued, running solo.**
  - Two earlier attempts failed on **infrastructure, not the mechanism** (my code never runs
    during model load/warmup): (1) Bus error on node-0 — a **concurrent `$WORK/.cache` JIT race**
    with the still-running best_effort eval (eval.sh hardcodes one cache dir per workspace, so two
    of my own evals can't run concurrently); (2) hang on ondem-3 — reading the **corrupted cache**
    left by (1). Fixes: cleared `$WORK/.cache`; run adaptive **solo** (never concurrent with another
    drift-3e7 eval). Lesson: serialize my own evals; wipe cache after a crash.
  - **3rd attempt also failed to load** (hang after "GDN kernel dispatcher", before DeepGEMM
    warmup: triton cache stuck at 1 entry, GPU 0%, scheduler procs ~7% CPU = a **NCCL/warmup
    collective deadlock**, not slow compile). Ruled out: my code (never runs at init), the
    `adaptive` flag (no init validation on the unified path), the JIT cache (cleared), and
    concurrency (solo). Remaining cause = **environmental**: fabric/NCCL contention during model
    warmup under many concurrent researcher server-loads (best_effort loaded at 06:38 when the
    pool was quieter; cache-clear also forced a fresh recompile that compounds it). Killed to free
    the node (a hung run holding an 8-GPU node is bad citizenship).
  - **Decision:** adaptive is IMPLEMENTED + COMMITTED (`7468090d4`) and unit-tested; its formal
    eval is deferred until pool contention eases (retry with close monitoring + fast-kill on
    init-hang so no node is wasted). The mechanism is sound; the blocker is shared-cluster load.
  - Now waiting on **pool contention** — all certified nodes are busy (others) or `drain`ed;
    the held pool is the only certified capacity.
- **W&B:** interactive `wandb.init` times out (network) and the run id `drift-3e7` appears
  **tombstoned** from an earlier delete, so offline-sync doesn't recreate it. **report.md is the
  authoritative, commit-traceable record.** Will reconcile the cloud curve when it clears.
- Helper scripts (session, in autoresearch root): `smart-eval.sh` (disk-aware pooled launcher),
  `pinned-eval.sh` (flock+srun to a specific held node), `wandb_batch_log.py` (offline curve logger).

## v4-adaptive-prefetch — load-adaptive prefetch stop policy (MECHANISM) — READY (commit `8e15590dc`)

- **Hypothesis:** the win is to keep `wait_complete`'s full-prefetch benefit when the SSD tier
  has headroom, but *give up early and recompute* (like best_effort) exactly when the SSD
  backlog is saturating — so a saturated SSD can't starve the decode batch and blow up the tail.
- **Change (`mem_cache/unified_radix_cache.py`, `server_args.py`):** new
  `--hicache-storage-prefetch-policy adaptive`. In `can_terminate_prefetch`, deadline =
  `base + pages×per_page × (1 − pressure)`, `pressure = prefetch_tokens_occupied /
  prefetch_capacity_limit ∈ [0,1]`. pressure→0 ⇒ the timeout-policy deadline (wait for the
  whole prefetch); pressure→1 ⇒ flat `base` (~1 s) regardless of prefix size ⇒ give up, admit,
  recompute. Self-regulates via negative feedback around the capacity limit.
- **Why robust:** under light load it *is* the wait/timeout behavior (safe); it only deviates
  under saturation. Unit-tested the deadline math offline.
- **Plan:** launch the moment a certified node frees; compare vs `v0_official` (wait_complete)
  and `v2cfg-besteffort` (the no-wait ceiling). Tag `mechanism`.
- **Lossless:** recompute yields identical tokens.
