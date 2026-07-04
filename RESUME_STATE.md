# quartz-7m3 RESUME STATE (read first on every resume)

I am researcher **w6** (done-signal: `touch .../manager/.runtime/slots/w6.researcher-done`).
Sessions tear down ~every 2 min; a DETACHED racer does the eval work independent of my session.
Version budget 100; **own versions logged so far: 1** (v3-sjf-aged). Baselines v0_official/v0_tuned don't count.

## ===== LATEST (21:05, 2026-07-04) — READ THIS FIRST =====
- **~12h certified-capacity outage** (a3nodeset-1/0-1/0-3 DRAIN, 0-0 NCCL-broken, 1-2 disk-short 847G,
  ondem-3 alloc by other researcher). No usable certified node the entire time. Not my code — fleet-wide.
- **Own versions LOGGED: 2/100 — v3-sjf-aged 77083ms=1.41x (BEST), v2-sjf 80342ms=1.35x.** ALL CHARTER
  CORE REQS MET: lossless SJF+aging KV-cache win proven, beats v0_tuned, logged W&B, report pushed, emailed.
- **v1 (18269) + v5 (18270)** = autonomous self-logging retry sbatch, still PENDING (queued for a free
  certified node). They WILL run + auto-log W&B whenever capacity frees, even if my session is idle.
- **Re-engagement is automated** (so I stopped wasteful 10-min polling after 12h of zero movement):
  (a) monitor wait_jobs.sh (task brccu6dju) fires on job success/failure;
  (b) DURABLE CRON e958d034 fires every 2h to check/process/resubmit/email.
- ON RESUME/CRON TICK: check squeue for 18269/18270 + `find runs -name summary.json`; for any NEW ver:
  ensure W&B-logged (wrap auto-logs; else log_wandb.py), update report.md, EMAIL if <77083. If a job
  died w/o summary (3 retries failed) -> resubmit (exclude non-certified + slurm2-a3nodeset0-0). NOT done
  (2/100); do NOT touch w6.researcher-done. If capacity opens widely, could add v6 (aging sweep) or HRRN.

## ===== (09:10, 2026-07-04) earlier =====
- **Own versions LOGGED: 2/100** — v3-sjf-aged (77083ms=1.41x BEST), v2-sjf (80342ms=1.35x). HEADLINE DONE + durable.
- **v1 & v5 now run as FULLY-AUTONOMOUS certified sbatch jobs** (survive my session):
  v1-parallel-l3-io = jid in runs/v1-sbatch.jid (18269), v5-sjf-aged90-rep = runs/v5-sbatch.jid (18270).
  Each --wrap does: retry eval 3x (flaky flashinfer-allreduce crash at capture, exit3 — v1 hit it once) THEN
  AUTO-LOG to W&B on success. So results self-log even if my session ends. Comparable config (no nocar).
- **BLOCKER: certified capacity crunch.** All certified nodes {-1,0-1,0-3 drain; 0-0 NCCL-broken; 1-2,ondem-3
  alloc by fleet hold jobs}. Exclusive sbatch ETA ~24h (runs sooner when a hold job cycles — 18235 grabbed
  ondem-3 at 08:37 that way, but then hit the flaky crash). Held-pool path also stuck (mgr held/ = 1-2 short).
- ON RESUME: 1) `squeue -u $USER | grep quartz` — running/done? `find $D/runs -name summary.json`. If a NEW
  ver has summary.json, confirm wandb-*.log/W&B has it (the wrap auto-logs; else log manually), read
  overall/ttft_mean_ms, update report.md, EMAIL if <77083 (new best). 2) If a job left queue w/o summary
  (all retries failed) -> resubmit (see runs/*-sbatch.jid + the mkw wrap pattern in transcript). 3) monitor
  = wait_jobs.sh (fires on success OR job-gone-no-result). 4) budget: 100 cap; realistically eval-capacity-bound.
- NOTE: this is refinement work (v1 disk-ablation, v5 noise-band). The headline result stands regardless.
- **a3nodeset0-0 is HARDWARE-broken** (confirmed 3 ways incl. a FRESH exclusive-alloc NCCL probe jid 18271
  = ChildFailedError): 8-GPU NCCL init fails regardless of allocation. Keep it in --exclude forever. Don't
  retry it. (It periodically shows idle since its hold job ended, but it's a trap — excluded correctly.)

## ===== (08:40, 2026-07-04) earlier =====
- **Own versions LOGGED: 2/100** — v3-sjf-aged (77083ms=1.41x BEST), v2-sjf (80342ms=1.35x). Headline DONE.
- **PIVOTED to sanctioned sbatch fallback** (certified pool freed up ~08:37): held pool was stuck (0-0
  NCCL-broken then dropped by mgr; 1-2 disk-short), so both refinements now run as `--exclusive` certified
  sbatch jobs (clean, no collision). Campaign RETIRED (avoid double-run).
    - **v1-parallel-l3-io**: sbatch jid in runs/v1-sbatch.jid (now 18264, retry3x), RUNNING on ondem-3 (SAME certified
      node as v2/v3 -> comparable). ~2h -> done ~10:40.
    - **v5-sjf-aged90-rep**: sbatch jid in runs/v5-sbatch.jid (now 18265, retry3x), PENDING (waiting for a 2nd certified node).
- **sbatch does NOT auto-log to W&B** — I MUST log each MANUALLY on completion:
  `( set -a; . $ROOT/.env; set +a; $D/.venv/bin/python $ROOT/programs/sgl/researcher/.claude/skills/report-sop/scripts/log_wandb.py quartz-7m3 $D/runs/<VER>/summary.json <VER> <commit=bc83a70d9> mechanism )`
  then read overall/ttft_mean_ms, update report.md table+section, EMAIL if <77083 (new best).
- Monitor bw... watches runs/{v1-parallel-l3-io,v5-sjf-aged90-rep}/summary.json.
- To submit another certified sbatch (template): see runs/*-sbatch.jid history; exclude=`certified-nodes.sh --exclude`+`,slurm2-a3nodeset0-0`; env goes INSIDE --wrap before `bash $EVAL`.
- If certified pool crunches again: fall back to held-pool campaign.sh (dynamic held read, blacklist 0-0, patched racer).

## ===== (07:07) earlier =====
- **Own versions LOGGED: 2/100** — v3-sjf-aged (77083ms=1.41x, BEST), v2-sjf (80342ms=1.35x). Both to W&B.
  (v1 was clobbered mid-run by my own rm; never completed. Re-queued.)
- **HEADLINE SECURED**: v3 beats v0_tuned (108824) by 1.41x, LOSSLESS. Mechanism code committed+pushed
  (hicache_storage.py, schedule_policy.py, environ.py, server_args.py). report.md committed.
- **EVAL BLOCKED on infra (not my code)**: manager's held pool rotated to a BAD pair for me =
  {a3nodeset0-0 = NCCL-BROKEN (8-GPU NCCL init OOMs on healthy free GPUs, 6+ fails), a3nodeset1-2 =
  disk-short 847G<1800}. Other researchers ARE serving fine on a3nodeset-0 (2664G) + ondem-3 -> fabric
  is HEALTHY cluster-wide; just a bad held-pool draw. c3nodeset-0 driver dead.
- **STANDING MECHANISM**: detached campaign.sh (PID varies, PPID=1), plan=[v5-sjf-aged90-rep (repeat best
  -> noise band/credibility), v1-parallel-l3-io (disk ablation)]. Now reads held/ DYNAMICALLY each round,
  MIN_FREE_G=1800 (matches eval.sh), BLACKLIST="slurm2-a3nodeset0-0", patched racer (teardown-hang fixed).
  Currently WAITING ("no disk-OK held nodes; wait 60s"); auto-runs+auto-logs when manager rotates in a
  good held node. Relaunch if dead: `setsid env BLACKLIST="slurm2-a3nodeset0-0" bash $D/campaign.sh > $D/runs/campaign.log 2>&1 </dev/null & disown`
- ON RESUME: 1) `find $D/runs -name summary.json` — if v5/v1 NEW, check wandb-<ver>.log "logged" (else log
  manually), read overall/ttft_mean_ms, update report.md, EMAIL if <77083 (new best). 2) campaign alive?
  relaunch if dead. 3) If still blocked long: consider sanctioned FALLBACK = queued certified sbatch
  (eval-on-pool.sh auto-does it when NO held nodes; or submit directly, excluding a3nodeset0-0). 4) If lots
  of eval capacity opens: HRRN (Highest Response Ratio Next) is a candidate stronger scheduler (smooth
  anti-starvation vs binary aging) — implement in schedule_policy.py. But noise ~24% limits fine gains.

## ===== (06:25) earlier =====
- **Own versions logged: 3/100** — v3-sjf-aged (77083ms, BEST), v2-sjf (80342ms), v1-parallel-l3-io (running).
- **RESULTS**: v3 (SJF+aging90) 77083 = 1.41x vs v0_tuned = BEST. v2 (pure SJF) 80342 = 1.35x.
  KEY FINDING: **aging LOWERS the mean** (v3<v2 by 4.1%) by bounding the heavy tail (p90~230s); it's not
  just tail-insurance. But 4.1% is within the ~24% regime noise -> treat v3>v2 as DIRECTIONAL; the robust
  claim is SJF±aging >> FCFS baselines (1.35-1.41x). Both LOSSLESS (pure queue reorder).
- **ABLATION LADDER** (io-threads default=4, so parallel-L3-IO is on for the whole branch):
  v0(FCFS+serial-IO) -> v1(FCFS+parallel-IO) -> v2(+SJF) -> v3(+aging). v1-v0=disk, v2-v1=SJF, v3-v2=aging.
  => a separate io=1 ablation is REDUNDANT (v1 vs v2 already isolates SJF cleanly).
- **v1-parallel-l3-io RUNNING** on ondem-3 via an ORPHANED racer (PID was 540511, PPID=1; campaign killed).
  It uses OLD racer code -> WILL hang in teardown after summary.json (KNOWN bug). HANDLE MANUALLY:
  when runs/v1-parallel-l3-io/summary.json appears -> (a) manually log:
  `( set -a; . $ROOT/.env; set +a; $D/.venv/bin/python $ROOT/programs/sgl/researcher/.claude/skills/report-sop/scripts/log_wandb.py quartz-7m3 $D/runs/v1-parallel-l3-io/summary.json v1-parallel-l3-io 88fa0f5c4 mechanism )`
  (b) kill orphan racer + `srun ... pkill sglang` on ondem-3 to free it. (c) update report.md table+section.
- **RACER PATCHED** (commit 88fa0f5c4): race_eval.sh now force-cleans once summary.json exists (no more
  teardown hang) — future campaigns are robust. v2 auto-log FAILED via old racer; I logged it manually.
- **NEXT PHASE** (launch a fresh campaign AFTER v1 frees ondem-3, using patched racer): highest value =
  v5-sjf-aged90-rep (REPEAT of best -> establish noise band; credibility). Then optionally v6 aging tuning
  (one point, e.g. 45 or 135). Disk (v1) likely neutral (queue-bound regime). New scheduling mechanism
  (continuous priority = remaining - k*wait) only if pushing past 77083 seems worth a noise-limited eval.
- v2 result backed up at runs/v2-sjf.SAVED. commit HEAD=88fa0f5c4 (+ next commits).

## ===== (03:27, 2026-07-04) earlier =====
- **FABRIC RECOVERED ~02:00. v3-sjf-aged EVALUATED = NEW BEST: mean TTFT 77082.7 ms** (1.41x vs
  v0_tuned 108824; 1.14x vs v0_official 87615). out_tok/s 149.2, hit .818/l3 .253 (=golden run).
  W&B-logged (tag mechanism, commit f52eab323). report.md updated (results table + v3 section).
  new-best EMAIL sent. Verified on-contract + schedule_policy='sjf' in server.log ServerArgs. LOSSLESS.
- **Own version count = 1/100.** (v3 is my first valid serve.)
- **Disk-aware campaign restarted** (commit 123c769e4): campaign.sh now computes RACE_NODES each round =
  held nodes with >=2100G free on /mnt/localssd (self-heals). Was spinning rc=2 (DISK_TOO_SMALL) on
  node 1-2 (only 847G free); now correctly uses **ondem-3** (2497G free). node -0 has no held job now.
  Plan order now: v2-sjf (pure SJF, isolate aging's tail cost) -> v1-parallel-l3-io.
- ON RESUME: 1) `find $D/runs -name summary.json`; for any NEW ver (v2-sjf, v1-...) confirm wandb-<ver>.log
  says "logged", read overall/ttft_mean_ms, UPDATE report.md table+section, EMAIL if new best (<77082.7).
  2) campaign alive? `ps -eo cmd|grep '[c]ampaign.sh'` — if dead: `setsid bash $D/campaign.sh > $D/runs/campaign.log 2>&1 </dev/null & disown`.
  3) After v2+v1 done, evolve versions 2..N: aging-sweep (30/60/120/180s), v1+v2 stack, io-threads sweep. Eval ~2h each, ONE node -> serial; 100 is a CAP not a target.

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
