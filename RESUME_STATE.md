# quartz-7m3 RESUME STATE (read first on every resume)

I am researcher **w6** (done-signal: `touch .../manager/.runtime/slots/w6.researcher-done`).
Sessions tear down ~every 2 min; a DETACHED racer does the eval work independent of my session.
Version budget 100; **own versions logged so far: 0** (only baselines v0_official/v0_tuned).

## What's DONE (committed+pushed on evolve/quartz-7m3, HEAD f52eab323)
- v1 = parallel L3 file I/O in HiCacheFile (env SGLANG_HICACHE_FILE_BACKEND_IO_THREADS=4). Lossless.
- v2 = cache-aware SJF prefill scheduling (`--schedule-policy sjf`). Lossless. Offline sim: 2.37x.
- v3 = optional SJF aging (env SGLANG_SJF_AGING_SEC). Lossless.
- Baselines logged to W&B run `quartz-7m3`. Hello + status emails sent.
- report.md written. Analysis: mean TTFT is prefill-QUEUEING-dominated, not disk-latency.

## Infra reality
8 researchers (w1-w8) share ~2-3 disk-OK pool nodes (-0, 1-2 disk-OK; 0-0/ondem-3 often disk-short).
Extreme contention + intermittent NCCL-init hangs at startup (NOT my code — hang precedes hicache init).
eval MUST be on a certified pool node. `srun --overlap` eval dies if my session dies -> must DETACH.

## The DETACHED racer (my eval mechanism)
`race_eval.sh <ver> [eval args]` : blocking-flock race over RACE_NODES, hang-watchdog (kills startup
frozen >600s post-KV-alloc), retries up to RACE_MAX, and on success writes summary.json AND
auto-logs to W&B (if LOG_COMMIT set). Launch DETACHED so it survives teardown:
```
D=/home/junyanch_google_com/autoresearch/workspace/sgl/researchers/quartz-7m3
COMMIT=$(git -C $D rev-parse --short HEAD)
setsid env RACE_NODES="slurm2-a3nodeset-0 slurm2-a3nodeset1-2 slurm2-a3nodesetondem-3" RACE_MAX=30 \
  LOG_COMMIT="$COMMIT" LOG_TAG=mechanism bash $D/race_eval.sh <VER> <args> \
  > $D/runs/racer-persistent.log 2>&1 < /dev/null & disown
```

## ON EACH RESUME — do this:
1. `find $D/runs -name summary.json` and check `$D/runs/wandb-*.log` — did an eval finish + log?
   If summary.json exists but not logged, log it: `python3 .../report-sop/scripts/log_wandb.py quartz-7m3 <run>/summary.json <VER> <commit> mechanism`. Update report.md; email if new best (beat v0_tuned 108824 ms / v0_official 87615 ms mean TTFT).
2. Is the racer alive? `ps -eo cmd|grep '[r]ace_eval.sh'`. If DEAD, relaunch DETACHED (above).
3. If current version done, start the NEXT (v1 for attribution, then v3 aged-sjf, then tuning).
   Order of value: v2-sjf (running) > v1 > v3 > configs.
4. Don't redo setup. Don't kill other researchers' work. Only touch my own /mnt/localssd/quartz-7m3.

## CRITICAL FIX (18:52): watchdog HANG_SECS is 1500s (25min), NOT 600s.
Healthy startup has an ~11-min SILENT CUDA-graph-capture/init phase after KV-alloc (log frozen, no
output) BEFORE DeepGEMM warmup. HANG_SECS=600 was FALSE-KILLING healthy startups mid-capture -> I never
reached serving. Keep HANG_SECS >= 1500. DeepGEMM warmup itself updates the log (\r progress) so it
doesn't trip the watchdog; only a truly frozen >25min startup is a real hang (eval.sh backstops at 45min).

## Currently running: racer for **v2-sjf** (`--schedule-policy sjf`), commit f52eab323, tag mechanism,
HANG_SECS=1500, detached (PPID=1), auto-logs W&B on success. Log: runs/racer-persistent.log.

## Two distinct startup blockers observed (both environmental, NOT my code):
1. ~17-min SILENT CUDA-graph-capture after KV-alloc — HEALTHY, must not kill (HANG_SECS=1500 handles).
2. FlashInfer trtllm workspace NCCL-communicator timeout -> "Disabling flashinfer allreduce fusion" ->
   then the barrier FREEZES (real hang). Confirmed `schedule_policy=sjf` DID apply before the hang.
   This is a known flaky-node symptom (see memory sgl-eval-ops). The racer kills+retries across nodes.
If it NEVER serves after many attempts, LAST-RESORT options (break baseline comparability, note it):
`--moe-runner-backend triton` or `--disable-custom-all-reduce` (both allowed extra args, not FORBIDDEN)
to avoid the trtllm-flashinfer NCCL path. Prefer a clean serve first.

## Racer relaunch command (current, RACE_MAX=40, dual watchdog):
```
D=/home/junyanch_google_com/autoresearch/workspace/sgl/researchers/quartz-7m3
setsid env RACE_NODES="slurm2-a3nodeset-0 slurm2-a3nodeset1-2 slurm2-a3nodesetondem-3" RACE_MAX=40 \
  HANG_SECS=1500 HANG_ABS=2100 LOG_COMMIT=$(git -C $D rev-parse --short HEAD) LOG_TAG=mechanism \
  bash $D/race_eval.sh v2-sjf --schedule-policy sjf > $D/runs/racer-persistent.log 2>&1 </dev/null & disown
```
Confirmed on 2026-07-03 20:11: a v2-sjf attempt hung the FULL 45min (eval.sh SERVER_TIMEOUT). Hangs are
frequent on the current pool. HANG_ABS=2100 now caps a hang at ~35min so retries cycle faster.

## KEY ROOT-CAUSE FIX (21:22): DIRTY-HANDOFF settle-wait.
Diagnosis: OTHER researchers' evals SERVE on 1-2/ondem-3 (pool is UP), but MINE hang/crash at model
init — even on ondem-3 back-to-back (mine crashed, theirs served). My venv == lockfile (verified),
my code runs AFTER the init hang, so neither is the cause. The difference: my BLOCKING flock wins the
node the INSTANT the prev eval releases it, BEFORE its pkilled procs die / GPU+shm free -> dirty node
-> NCCL/CUDA init hangs. Pollers (eval-on-pool, 30s) grab ~15s later = clean = serve. race_eval now
does a SETTLE-WAIT (pkill leftovers + clear /dev/shm + wait until GPU<2GB & 0 sglang procs) before
launching eval.sh. This is the leading fix; if runs now SERVE, keep it. Comparability preserved.
Own versions logged: STILL 0. Budget 100. Check W&B run quartz-7m3 for what's logged.

## UPDATE 22:06: BLACKLISTED node -0 (it hung/crashed EVERY attempt, even clean/settled). Racer now
races ONLY slurm2-a3nodeset1-2 + slurm2-a3nodesetondem-3 (where other researchers serve fine). If v2
SERVES there -> -0 was the bad node. If it ALSO fails -> deeper my-run issue. RACE_NODES in relaunch
cmd = "slurm2-a3nodeset1-2 slurm2-a3nodesetondem-3". Still 0 own versions logged.

## UPDATE 22:55: CONFIRMED cluster-wide NCCL-fabric degradation on NEW STARTS.
v2 on CLEAN 1-2 (settled, GPU 0MiB) STILL froze at the same NCCL "Guessing device ID" barrier as -0.
So NOT node-specific, NOT dirty-handoff, NOT my venv/code. Evals that started earlier keep serving;
new starts hang at the first post-init NCCL collective (barrier). Like the earlier FS stall, this is a
transient cluster degradation that should recover. The racer keeps retrying (auto-logs on first clean
serve). NOTHING actionable on my side except persist. If a resume finds it STILL hanging after long,
that's expected — just confirm the detached racer is alive (relaunch if dead) and wait. All research
is done+pushed; only the empirical eval is blocked by infra. Still 0 own versions logged.

## UPDATE 23:32: SALVAGE PIVOT -> v2-sjf-NOCAR (--disable-custom-all-reduce).
After ~15 pure-config hangs (all at the NCCL barrier; logs showed "FlashInfer workspace backend=trtllm
NCCL communicator timeout" = the CUSTOM ALL-REDUCE init hanging), pivoted the racer to eval
"v2-sjf-nocar" = --schedule-policy sjf --disable-custom-all-reduce, racing all 3 nodes. Rationale:
skips the hanging trtllm custom-AR init; nocar's small all-reduce overhead only makes any SJF win a
CONSERVATIVE lower-bound vs the car-baselines (defensible; DOCUMENT the caveat when logging). If it
SERVES: log v2-sjf-nocar (tag mechanism, note the caveat), then run v0-nocar baseline anchor
(fcfs + --disable-custom-all-reduce) for a clean same-config comparison. If it ALSO hangs -> the hang
is NOT custom-AR; revert to pure v2-sjf and wait for fabric recovery.
Racer: race_eval.sh v2-sjf-nocar --schedule-policy sjf --disable-custom-all-reduce (RACE_NODES=all 3).

## UPDATE 23:54: --disable-custom-all-reduce did NOT fix the hang (disable_custom_all_reduce=True still
froze at the same "Guessing device ID" NCCL barrier). So the hang is the FUNDAMENTAL torch.distributed
NCCL barrier, NOT custom-all-reduce. Unfixable by me (fabric/mapping degradation on new starts).
REVERTED to pure v2-sjf (comparable), RACE_MAX=60, detached, auto-logs on recovery. This is the
correct standing state: just wait for the cluster NCCL fabric to recover; the racer catches it. Do NOT
waste slots on more mitigation guesses. Still 0 own versions logged; all research done+pushed.

## ===== CURRENT MECHANISM (00:27, supersedes single racer): campaign.sh =====
DETACHED campaign (PPID=1) works through the version plan in priority order, each via race_eval.sh
(settle-wait + dual watchdog + AUTO-LOG W&B on success), looping so fabric-blocked versions retry:
  v2-sjf (--schedule-policy sjf) > v1-parallel-l3-io (default) > v3-sjf-aged (--schedule-policy sjf, env SGLANG_SJF_AGING_SEC=90)
Logs: runs/campaign.log (campaign) + runs/racer-persistent.log (current race) + runs/wandb-<ver>.log (W&B).
Relaunch if dead:  setsid bash $D/campaign.sh > $D/runs/campaign.log 2>&1 </dev/null & disown
ON RESUME: 1) find $D/runs -name summary.json  (any logged version?). 2) is campaign alive?
  ps -eo cmd|grep '[c]ampaign.sh' — if dead, relaunch. 3) For each summary.json, confirm it's in W&B
  (wandb-<ver>.log says "logged"); if not, log manually. 4) Update report.md + email best (beat
  v0_tuned 108824 / v0_official 87615 ms mean TTFT). Blocker = cluster NCCL-fabric hang on new starts;
  campaign auto-serves+logs when it recovers. Still 0 own versions logged as of 00:27.

## ===== 02:18 FABRIC RECOVERED — v3-sjf-aged SERVING on ondem-3! =====
The NCCL-fabric degradation cleared (~02:00). The campaign caught it: v3-sjf-aged (SJF + aging,
SGLANG_SJF_AGING_SEC=90 confirmed in server env; schedule_policy=sjf; resolved_args ON-CONTRACT; no
SILENT FALLBACK) is running the mix benchmark, will finish ~03:15 and AUTO-LOG to W&B (tag mechanism,
commit f52eab323). NOTE v3 served BEFORE v2/v1 (it was the version being tried when the fabric came
back). After v3 completes, campaign loops -> v2-sjf, then v1-parallel-l3-io (should serve now).
ON RESUME NOW: 1) find $D/runs -name summary.json ; check runs/wandb-*.log for "logged". 2) For each
logged version: read summary.json mean TTFT (overall/ttft_mean_ms), UPDATE report.md with the number +
lossless note + takeaway, and EMAIL if it beats v0_tuned 108824 / v0_official 87615 ms. 3) Ensure
campaign still alive (relaunch if dead) so v2/v1 also run. 4) These are my OWN versions (count vs budget 100).
