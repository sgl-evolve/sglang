# kv-lynx-4d2 — RESUME STATE (read first on resume)

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

## v9+ decision tree (see design_v3.md "v8+ plan")
- v8 beats v4 above ~24% noise → α sweep {20,100,200}; log best.
- v8 within noise → host-eviction policy isn't the lever; test the BINDING tier: v9 = freq-aware **Mamba**
  eviction (evict_mamba LRU walk, CLOCK-style bounded second-chance; riskier — verify lossless).
- both neutral → eviction policy not a TTFT lever at fixed capacity; v4 stands; pivot (scheduler
  cache-aware admission) or conclude.

## Infra reality (why evals are slow) — see memory sgl-eval-infra-ops
- Cluster severely degraded: **3 of 6 certified nodes DRAINED/dead** (-1, 0-1, 0-3; unrecoverable w/o admin),
  ondem-3 + 1-2 busy, **0-0 is bad-GPU** (OOMs at init despite showing idle — SKIP it, always).
- Manager held pool stale/degraded → per charter use a self-lock/queued certified sbatch. **Do NOT leave a
  `sleep infinity` hold queued** if the session may end — it squats a node 12h. One-shot self-contained
  eval jobs (like v8's) are safe.
- Launcher tools: `run_eval.sh` (held-pool --overlap poller, has WORK-bug fix + contamination-retry),
  `hold_run.sh` (runs evals into a session hold). Both good-citizen; `scancel` any hold when idle.
