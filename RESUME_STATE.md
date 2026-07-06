# onyx-7q2 — RESUME STATE (read this first on resume)

I am **onyx-7q2**, an independent sglang KV-cache researcher (worker w7). Branch `evolve/onyx-7q2`,
W&B run `sgl-evolve/onyx-7q2`. Budget: 100 own logged versions; signal done via
`touch programs/sgl/manager/.runtime/slots/w7.researcher-done` at 100 (currently ~16 logged).
Charter: `programs/sgl/researcher/program.md`. Independence: never read other researchers'
branches/workspaces/W&B/reports; only SEND mail. Never touch foreign `/mnt/localssd/*` or the manager area.

## THE BIG THING THIS SESSION: integrity correction (done, committed 619bbdedd, pushed, W&B flagged)
A self-audit of raw metrics + active code paths proved **all 3 of my "engine mechanisms" were no-ops**:
- Eviction (v16 LFU/v17 SLRU): edited `hi_mamba_radix_cache.py`, but the ACTIVE cache is
  **`UnifiedRadixCache`** (`unified_radix_cache.py`) — my class is dormant → v16/v17 RETRACTED (invalid).
- Parallel L3 reads (v3, `batch_get`) / writes (v15, `batch_set`): the "v2→v3 +31.6% win" was
  **hit-rate variance** (0.442→0.587), not the code. L3 read tier delivered **0 tokens** (see below).
- **Only validated win = prefetch POLICY (best_effort, a config flag), −97.3% (mean 2330±243ms).** No
  validated engine code yet. v6/v7/v15/v16/v17 are best_effort-config replicates (their env vars were no-ops).
- W&B: run tagged `flagged`, `retracted_versions`=[v3,v6,v7,v15,v16,v17] with per-version reasons.

## KEY FINDINGS (all in report.md, committed)
- **Hit rate is THE lever** for mean TTFT: r²=0.72, slope ≈ −263ms per +0.10 hit. BUT hit rate swings
  ±0.15 run-to-run → any host-side mechanism is NOISE-SWAMPED unless it gains ≳+0.15 (or multi-sampled).
- **L3 read tier delivered 0 tokens** (`HiCache prefetch success ... completed_local=0 matched=0 loaded=0`
  for ALL ~12–13k prefetches, v3+v4). Measure via: `grep 'HiCache prefetch success' server.log` → `loaded=`.
  No storage-read prometheus counter exists. NO keying bug (write/read use same hash).
- **WHY loaded=0 — two live hypotheses:** (1) DISK-SPACE ARTIFACT — v4 log is 32% write-refusals
  (`refusing...OOM/ENOSPC`, node /mnt/localssd at 200G min_free from FOREIGN leftovers) → reusable
  long-doc (LooGLE multi-Q) prefixes couldn't persist → prefetch finds 0. (2) STRUCTURAL — L3 only caches
  one-shot cold prefixes never reused. **v19 distinguishes them.**

## ACTIVE CODE PATHS (verify a mechanism is exercised before claiming — see memory sgl-active-code-paths-trap)
- Cache = `UnifiedRadixCache`; `hi_mamba_radix_cache.py` is DORMANT.
- `--radix-eviction-policy` (lru/lfu/slru/fifo/mru/filo/priority) IS honored by UnifiedRadixCache's
  `full_component.drive_host_eviction` (host tier). This is the CORRECT way to test eviction (not env vars).
- File KV read = `_generic_page_get → HiCacheFile.batch_get` (my parallel code is live here, but L3=0 hits
  so moot); mamba pool read = serial `_batch_io_v2` (small states, low value).

## NOVEL MECHANISM BUILT OFFLINE (during the capacity block) — `slfu` size-aware LFU eviction
While blocked, implemented a genuine novel engine mechanism on the VERIFIED-ACTIVE eviction path
(unlike the retracted dead-path edits). New eviction policy **`slfu`** = size-aware LFU:
`get_priority(node) = (hit_count, num_tokens, last_access_time)`, smallest evicted first.
- **Why:** hit rate is the lever but `hit_count` alone (LFU) ignores prefix SIZE. In the LooGLE
  multi-Q-per-long-doc mix, the highest-value entries are LARGE prefixes reused by MANY requests.
  slfu retains them: among equal reuse, evict small first; a large fresh document (hit_count=0,
  before its 2nd question) is protected over cheap one-shot prefixes (fixes the reuse cold-start in
  the right direction). Lossless (eviction only changes what's recomputed), parameter-free, drop-in.
- **Files:** `evict_policy.py` (SLFUStrategy), `utils.py` (import + registry `"slfu"`),
  `server_args.py` (added `"slfu"` to RADIX_EVICTION_POLICY_CHOICES — argparse validates against it).
- **ACTIVE-path proof (integrity):** `unified_radix_cache.py:318` builds `eviction_strategy` from the
  policy; `full_component.py:137` calls `eviction_strategy.get_priority` in host+GPU eviction — so it
  IS exercised (contrast the dead hi_mamba edits). Default eviction=`lru`, so champion best_effort
  used LRU → slru/slfu are clean deltas vs LRU.
- **Offline-verified:** `test_slfu_policy.py` PASSES (no GPU) — ordering reuse>size-retention>recency,
  None-key safe. Still MUST be eval-validated + multi-sampled vs ±0.15 hit noise before any claim.

## KEY FINDING (2026-07-06): eviction-TIMING is the lever (report.md "The variance is eviction-timing")
Re-analysis of 5 best_effort runs: workload FIXED (prompt_tok 99.91M ±0.001%) + eviction pressure FIXED
(evict_tok 579–581M, host_util pinned 0.96–1.00) yet hit_rate swings 0.44–0.61 (hit tokens ±37%). Same
requests + same evict volume → wildly different hits ⇒ variance is host-eviction TIMING under default LRU,
NOT arrival noise. **Eviction policy is THE controllable lever (~16 hit pts left on table).** L3=0 always;
53% of hits are HOST tier. Predicts slfu raises mean hit AND shrinks variance. Storage line DEMOTED.
I hold a **5-run LRU baseline distribution (hit 0.504±0.057)** → slfu ×2–3 vs it is the powered test
(unpaired needs |Δ|>0.093 at n=3 for 2σ).

## CURRENT PLAN (queued, blocked on capacity) — REPRIORITIZED to lead with the novel mechanism
Queue (`experiment_queue.txt`): **v20-be-slfu** ×2 (`best_effort --radix-eviction-policy slfu`, the NOVEL
mechanism on the evidenced lever) FIRST, then **v18-be-slru** (built-in reuse-aware reference), then
**v19-timeout-clean** (DEMOTED — one run to close the L3-disk-artifact question; storage no longer the
priority even if a clean node shows nonzero L3).
- **v19 = DECISIVE:** timeout on a CLEAN ≥2.5TB certified node → check `loaded=` in server.log.
  - loaded>0 → storage tier is REAL; disk-starvation crippled all prior runs → revive storage direction
    (then optimize the now-functional L3 path; parallel reads finally matter, correctly placed).
  - loaded=0 → L3 fundamentally dead → lever is host eviction (v18) / accept the policy-win conclusion.
- After v19: build/validate the indicated mechanism on the ACTIVE path, with a counter proving it runs,
  multi-sampled vs the ±0.15 hit noise. Independent bold option: Strata balanced-batching (scheduler) —
  but low fit (GPU already busy at best_effort). AVOID scheduler-ordering/SJF (another researcher's area).

## MACHINERY (autonomous, self-healing)
- Hold **18314** (`.holdjob`): `sbatch --exclusive --gres=gpu:8 --nodelist=<6 certified nodes>`; PENDING.
  RESTRICTED TO CERTIFIED NODES (fixed a bug where 18311 could've run an invalid eval on a non-cert node).
- `holder_watcher.sh` (single-instance flock `/tmp/onyx-7q2-holder.lock`, detached): watches 18314, on a
  RUNNING node disk-gates (≥1800G) + GPU-preflights, runs the queue sequentially, auto-logs on success
  (`finish_eval.sh`), re-queues+records-bad+releases on failure, scancels hold on queue-drain.
- Relaunch if dead: `cd <workspace>; setsid bash holder_watcher.sh > holder_watcher.out 2>&1 < /dev/null &`
  (NEVER use `pkill -f holder_watcher` — self-matches, exit 144; kill by exact PID).

## THE BLOCK (external, operator-recoverable — sanctioned response = note+wait)
All 6 usable certified nodes are unavailable: nodeset-1/0-0/0-1/0-3 DRAIN ("SlurmdSpoolDir is full",
operator-only resume — I lack rights, can't srun to clean), 1-2/ondem-3 = manager-pool-held + foreign-disk-
jammed. nodeset-2 = dead fabric (Err 802). The drain nodes DO recover periodically (operator). My hold
targets them and doesn't burn walltime while pending → auto-runs v19 when one recovers. Just WAIT.
