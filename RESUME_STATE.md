# kv-lynx-4d2 — RESUME STATE (read first on resume)

## ‼️ CURRENT (2026-07-06 ~09:50) — HOLD at honest completion (accessible lossless levers exhausted)
- **Active cache = `UnifiedRadixCache`** (registry.py: hybrid-SSM + hierarchical → `_create_unified_radix_cache`).
  `HiMambaRadixCache` is DORMANT — old v8/v9 there NEVER ran; NOT logged. Always prove a mechanism active via
  the `'kv-lynx-4d2 UnifiedRadixCache ACTIVE'` server.log line before logging. See [[sgl-active-code-paths-trap]].
- **This session logged 3 real active-path versions, ALL NEUTRAL** (vs stock re-baseline 1696 ms / hit_rate 0.55):
  - v10-umamba-clock (CLOCK Mamba eviction, MAXSKIP=8/THR=2) → 1762 ms, hit_rate 0.5525. `mechanism`.
  - v11-sched-lpm (`--schedule-policy lpm`) → 1715 ms, hit_rate 0.536, device-hit +13%. `config`.
  - v12-clock-thr1 (CLOCK aggressive, MAXSKIP=32/THR=1) → 1753 ms, hit_rate 0.5494. `mechanism`. Eviction axis CLOSED.
  Mechanism code committed `64a07eddc` (env `KVLYNX_MAMBA_CLOCK_MAXSKIP`/`_THR`, 0=stock, offline-verified,
  lossless). v9 = stock re-baseline (reproduces v4 within noise), recorded in report.md, NOT a W&B point.
- **CONCLUSION: lossless optimum reached across all 4 accessible axes** — storage/query (WON, v2 scandir),
  capacity (WON+MAXED, v3/v4 mamba-ratio 1.5), eviction-order (NEUTRAL, v10/v12), scheduling (NEUTRAL, v11).
  Residual ~1.7 s / 56× wall = contract-imposed hit-rate ceiling (frozen `hicache-size 96`, host tier full) +
  lossless bar. Prefill-coalescing idea REJECTED — sglang already shares in-flight prefixes via chunked-prefill
  incremental radix commit (`cache_unfinished_req(chunked=True)` inserts prefilled-so-far tokens).
- **BUBBLE/OVERLAP axis explored + CLOSED (2026-07-06 ~10:50):** v13-prefetch-besteffort
  (`--hicache-storage-prefetch-policy best_effort`) = NEUTRAL (1685 ms vs 1696; prefetch-wait is not a bubble;
  storage tier ~irrelevant), LOGGED `config`. v14-tbo (`--enable-two-batch-overlap`) = INCOMPATIBLE — crashes at
  startup (`moe_a2a_backend cannot be 'none'`; TBO is multi-node EP-only, we're single-node TP8), NOT run, no W&B
  point. sglang overlap scheduler already ON. `load_back_mean` 0.94 ms → transfer isn't a bubble.
- **COMPREHENSIVE CONCLUSION: v4 (56×) is the lossless optimum across EVERY accessible axis** — cache policy
  (eviction/scheduling/admission/disk), capacity (frozen), transfer/prefetch, compute-overlap (TBO incompat),
  algorithmic (GPU-bound). TTFT stable within ~5% across ALL policy levers → engine is compute+reuse bound and
  already well-overlapped. hit_rate 0.55 = intrinsic reuse rate (frozen 9.6M-token cache vs 100M stream).
- **PREFILL/DECODE scheduling axis also tested + CLOSED (v15, 2026-07-06 ~11:50):** `--enable-mixed-chunk` →
  mean TTFT **5328 ms = 3.1× WORSE** (prefill shares slots with decode → first-token delayed), BUT throughput
  3.52 (↑, >λ) and hit_rate 0.582 (↑). Classic prefill↔decode tradeoff, wrong way for TTFT. LOGGED `config`
  (kept negative). NOTE: my independent result is a TRADEOFF, NOT the hit_rate collapse a prior context implied —
  good that I tested it myself. Stock is already TTFT-optimal on this axis (prefill-dedicated, delayer off,
  concurrency-capped 128); the throughput/hit_rate headroom is inseparable from the TTFT cost.
- **ADMISSION axis tested + CLOSED (v16, 2026-07-06 ~13:45):** `--hicache-write-policy write_through_selective`
  (host-backup only ≥2-hit prefixes) → mean TTFT 2042 (+20%), **hit_rate 0.378 (−31% from 0.55)**. HURTS — the
  workload has substantial hit-once-then-reused-later traffic that selective backup drops. **Stock write_through
  (threshold=1) is the admission OPTIMUM** (more selective misses second-uses; eager would pollute). LOGGED
  `config`. KEY: this proves hit_rate IS sensitive (can drop to 0.38) → the flat-0.55 under eviction reordering is
  the genuine REUSE CEILING, not an insensitive metric. Prior-art study (HiCache blog) done; remaining untested
  levers (io-backend=kernel, timeout-prefetch) are high-confidence-neutral (transfer 0.94ms + storage 1% = proven
  non-bottleneck) — not worth slots on the heavily-contended pool (v16 waited 55 min for a node).
- **STANCE = HOLD (thoroughly justified — even the bold mixed_chunk lever tested).** Every axis characterized;
  stock is TTFT-optimal everywhere; v4 (~53–56×) is the lossless optimum. Only untried = full Strata
  balanced-batching CODE build (low-EV: throughput headroom only buys itself via TTFT cost; high stall/wedge risk).
  Keep loop alive + monitor; released the node (no positive-EV experiment). Act only on a genuinely novel
  positive-EV low-risk lossless idea or a material change. Budget ~11/100 (CEILING, not target — do NOT touch
  `w8.researcher-done`).
- **RULE: run evals SERIALLY** (parallel = cold-`~/.cache/flashinfer` JIT race; v8 crashed this way, [[sgl-flashinfer-jit-race]]).
  Eval mechanism: `setsid nohup bash run_eval.sh <ver> --mamba-full-memory-ratio 1.5 --enforce-disable-flashinfer-allreduce-fusion > runs/<ver>_launch.log 2>&1 </dev/null &`

Last updated: 2026-07-04 ~19:35. Researcher: **kv-lynx-4d2**, branch `evolve/kv-lynx-4d2`, W&B run `kv-lynx-4d2` (project `sgl-evolve`).

## Where things stand
- **6 own versions logged** (honest curve): v2 scandir-fix (`mechanism`, 36×) → v3/v4 Mamba-KV rebalance
  (`config`; **v4 = best, mean TTFT 1558 ms, 56.2× vs baseline 87615 ms**) → v5 ratio-1.6 (reverted) →
  v6 page32 (reverted). Full analysis in `report.md`. Budget: 100 own versions (ceiling); nowhere near.
- **v7 write-selective**: predicted ~neutral (disk tier irrelevant, 1.14% hits); dropped, NOT run/logged.
- **v8 = NOVEL mechanism, PENDING EVAL**: frequency-aware (aged-LFU) eviction for the hybrid Mamba radix
  cache. Code committed `97d6058ec` in `python/sglang/srt/mem_cache/hi_mamba_radix_cache.py`
  (`_EVICT_FREQ_ALPHA`, `_evict_key`, applied to evict()/evict_host() heap keys). Lossless (victim order
  only; α=0 == stock LRU). Toggled by env **`KVLYNX_EVICT_FREQ_ALPHA`** (v8 uses **50**).

## ⚠️⚠️ INFRA BROKEN — DAYS-LONG OPERATOR ISSUE (confirmed 2026-07-04 22:52)
Drain reasons (`sinfo -N -o "%N %t %E"`): `-1`,`0-1`,`0-3` = **"SlurmdSpoolDir is full"** drained by
**root since 2026-06-30/07-01 (4+ days)**; `0-0` = "Epilog error". These are node-local OS-disk fills,
**operator-only fix** (`scontrol resume` → "Invalid user id"; can't srun to clean). ondem-3 + 1-2 are the
only non-drained certified nodes and are (a) disk-bad/cluttered AND (b) continuously held by other
researchers' `sgl-hold-*` jobs. **The certified eval pool will NOT auto-recover** — needs an operator to
clean SlurmdSpoolDir + `scontrol resume` the drained nodes. Until then NO eval can run for ANY researcher.
If you (future session) still see this after the pool is fixed, just let v8 (jid 18303) run.

## ⚠️ INFRA BROKEN (2026-07-04 21:35) — entire certified pool unusable for eval
No certified node currently has the required ≥1.8 TB free `/mnt/localssd`:
- `-1`, `0-1`, `0-3` = **drained** (operator-only resume); `0-0` = **drained + bad-GPU**;
- `1-2` = **bad-disk** (846 G); `ondem-3` = **idle but disk cluttered** (1106 G of other users'
  abandoned June data — shagarw/rqiang/root dirs — which I CANNOT delete: skill says only rm your own).
v8 (jid **18303**) is queued `--exclude`-ing all bad nodes → pends **ReqNodeNotAvail** until an operator
resumes a drained node (or cleans ondem-3). It auto-runs on recovery. **RECHECK on resume:** if
`ondem-3` disk clears to ≥1.8 TB, resubmit without excluding it; if a drained node resumes, v8 runs itself.

## PENDING ACTION — log v8 when its eval finishes
- v8 eval is a self-contained slurm job: **`squeue -n eval-v8-kvlynx`** (jid saved in `.v8_sbatch_jid`).
  It runs `eval.sh kv-lynx-4d2 v8-freq-evict50 --mamba-full-memory-ratio 1.5
  --enforce-disable-flashinfer-allreduce-fusion` with `KVLYNX_EVICT_FREQ_ALPHA=50` on an --exclusive
  certified node (excludes bad 0-0 + non-cert). Severe cluster contention → may not run for many hours.
- **When `runs/v8-freq-evict50/summary.json` appears:**
  1. VERIFY the mechanism was active: `grep _EVICT_FREQ_ALPHA runs/v8-freq-evict50/server.log` →
     must show `=50.0`. If it shows 0.0 (env didn't propagate) the run is stock LRU — DISCARD, re-run.
  2. Self-audit: mix.txt `Successful requests` ≥ 6685/7037, resolved_args ctx 262144/mem 0.85/hicache 96/tp 8.
  3. Log: `set -a; . /home/junyanch_google_com/autoresearch/.env; set +a;
     uv run --with wandb --python 3.11 python3 <report-sop>/scripts/log_wandb.py kv-lynx-4d2
     runs/v8-freq-evict50/summary.json v8-freq-evict50 97d6058ec mechanism`  (uv+wandb; W&B needs LOGIN node).
  4. Update `report.md`; email only if it's a NEW BEST (< v4's 1558 ms above the ~24% noise floor).

## v9 ALREADY BUILT (commit 407bc9376) — ready to run
v9 = frequency-aware **Mamba** eviction (CLOCK second-chance in `evict_mamba`), targets the BINDING
Mamba-capacity tier. Lossless (victim choice only), offline-verified (5 cases: stock-identical when off,
protects hot, always frees requested, terminates). Toggle **`KVLYNX_MAMBA_FREQ_MAXSKIP`** (0=stock LRU;
try 8) + `KVLYNX_MAMBA_FREQ_THR` (default 2). Separate from v8's toggle. To run:
`KVLYNX_MAMBA_FREQ_MAXSKIP=8 bash <EVAL/hold_run> v9-mamba-freq --mamba-full-memory-ratio 1.5 --enforce-disable-flashinfer-allreduce-fusion`.

### Run order when a node frees (eval slots are RARE — broken infra):
1. **v8** first (host-freq α=50) — safest (heap re-key), guaranteed data even if neutral. [queued jid 18303]
2. **v9** next (mamba-freq MAXSKIP=8) — higher-leverage (binding tier), offline-verified.
3. **v10** (`--schedule-policy lpm`) — cache-aware scheduling (Strata pillar); SAFE config (no code), HIGH-EV
   (prioritize cache-hit reqs → lower mean TTFT in this prefill-bound regime); different axis. See design_v3.md.
   If only 1 slot ever: consider running v10 first (safest + highest-EV) — but v8/v9 are the NOVEL contributions.
- If either beats v4 (1558ms) above ~24% noise → sweep its param, log best, email.
- If both within noise → eviction policy isn't a TTFT lever at fixed capacity; v4 stands as near-optimal;
  pivot to a different axis (scheduler cache-aware admission) or conclude honestly.
- VERIFY activation in server.log: `grep 'kv-lynx-4d2 eviction' runs/<ver>/server.log` shows the α / MAXSKIP.

## Infra reality (why evals are slow) — see memory sgl-eval-infra-ops
- Cluster severely degraded: **3 of 6 certified nodes DRAINED/dead** (-1, 0-1, 0-3; unrecoverable w/o admin),
  ondem-3 + 1-2 busy, **0-0 is bad-GPU** (OOMs at init despite showing idle — SKIP it, always).
- Manager held pool stale/degraded → per charter use a self-lock/queued certified sbatch. **Do NOT leave a
  `sleep infinity` hold queued** if the session may end — it squats a node 12h. One-shot self-contained
  eval jobs (like v8's) are safe.
- Launcher tools: `run_eval.sh` (held-pool --overlap poller, has WORK-bug fix + contamination-retry),
  `hold_run.sh` (runs evals into a session hold). Both good-citizen; `scancel` any hold when idle.
