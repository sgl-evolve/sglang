#!/usr/bin/env bash
# sweep_driver.sh [--dry] — autonomous design-space sweep for onyx-7q2 (budget 100).
# Keeps <=MAXINFLIGHT run_eval_robust wrappers alive, launching queued experiments as slots free,
# and autologs any run whose summary.json appears (idempotent via .logged marker). Safe across
# teardowns: on restart it detects running wrappers (pgrep) + done runs (summary.json) and continues.
# NEVER double-launches (checks wrapper alive + summary + a .launched marker).
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/researchers/onyx-7q2
DRY="${1:-}"
MAXINFLIGHT=2
PORTBASE=30740

# Experiment table: LABEL | ENV (space-sep, or -) | FLAGS | TAG
# Base is always: --enforce-disable-flashinfer-allreduce-fusion + timeout policy (unless flags override).
# best_effort is the winning base policy (v4). Sweep best_effort x {read-threads, page-size, write-policy}.
# Reordered by expected upside vs v4 (IO bench: 16 threads is the aggregate NVMe optimum, so
# thread-count experiments won't beat v4 -> deprioritized to the end). Frontier levers first.
EXPERIMENTS=(
  "v13-grace|SGLANG_HICACHE_PREFETCH_TIMEOUT_BASE=0.3 SGLANG_HICACHE_PREFETCH_TIMEOUT_PER_KI=0.03 SGLANG_HICACHE_PREFETCH_TIMEOUT_MAX=2|--hicache-storage-prefetch-policy timeout|config"
  "v11-be-selective|-|--hicache-storage-prefetch-policy best_effort --hicache-write-policy write_through_selective|config"
  "v12-be-writeback|-|--hicache-storage-prefetch-policy best_effort --hicache-write-policy write_back|config"
  "v9-be-page128|-|--hicache-storage-prefetch-policy best_effort --page-size 128|config"
  "v10-be-page32|-|--hicache-storage-prefetch-policy best_effort --page-size 32|config"
  "v6-be-thr32|SGLANG_HICACHE_FILE_READ_THREADS=32|--hicache-storage-prefetch-policy best_effort|config"
  "v7-be-thr8|SGLANG_HICACHE_FILE_READ_THREADS=8|--hicache-storage-prefetch-policy best_effort|config"
  "v8-be-thr64|SGLANG_HICACHE_FILE_READ_THREADS=64|--hicache-storage-prefetch-policy best_effort|config"
)

# runs that already exist outside the queue but should be autologged if they finish (v4,v5)
EXTRA_AUTOLOG=("v4-parallel-besteffort:config")

# count DISTINCT experiment labels in flight (avoid counting wrapper subshells twice)
running_count(){ pgrep -af "run_eval_robust.sh v" 2>/dev/null | grep -oE "run_eval_robust.sh v[0-9][a-z0-9-]*" | sort -u | wc -l | tr -d ' '; }

autolog_if_done(){  # $1=label $2=tag
  local lbl="$1" tag="$2"
  [ -f "runs/$lbl/summary.json" ] || return 1
  [ -f "runs/$lbl/.logged" ] && return 0
  echo "[driver] autolog $lbl ($tag)"
  [ "$DRY" = "--dry" ] && { echo "  DRY: finish_eval.sh $lbl $tag"; return 0; }
  bash finish_eval.sh "$lbl" "$tag" >> "logs_finish_$lbl.txt" 2>&1 && touch "runs/$lbl/.logged"
}

idx=0
while :; do
  # 1) autolog any completed extra runs + queued runs
  for e in "${EXTRA_AUTOLOG[@]}"; do autolog_if_done "${e%%:*}" "${e##*:}"; done
  for exp in "${EXPERIMENTS[@]}"; do IFS='|' read -r lbl env flags tag <<<"$exp"; autolog_if_done "$lbl" "$tag"; done

  # 2) launch next not-started experiment if a slot is free
  rc=$(running_count)
  if [ "$rc" -lt "$MAXINFLIGHT" ]; then
    for i in "${!EXPERIMENTS[@]}"; do
      IFS='|' read -r lbl env flags tag <<<"${EXPERIMENTS[$i]}"
      # skip if done (summary) or wrapper already alive; else (re)launch. No double-launch
      # because only ONE driver runs and pgrep guards; relaunches a died-without-completing eval.
      [ -f "runs/$lbl/summary.json" ] && continue
      pgrep -f "run_eval_robust.sh $lbl " >/dev/null 2>&1 && continue
      echo "[driver] launch $lbl  env=[$env] flags=[$flags]"
      if [ "$DRY" = "--dry" ]; then echo "  DRY: would launch"; touch ".launched_$lbl"; break; fi
      ( unset SGLANG_HICACHE_FILE_READ_THREADS SGLANG_HICACHE_PREFETCH_TIMEOUT_BASE SGLANG_HICACHE_PREFETCH_TIMEOUT_PER_KI SGLANG_HICACHE_PREFETCH_TIMEOUT_MAX
        [ "$env" != "-" ] && export $env
        export PORT=$((PORTBASE + i))
        nohup bash run_eval_robust.sh "$lbl" --enforce-disable-flashinfer-allreduce-fusion $flags > "eval-$lbl.log" 2>&1 &
      )
      touch ".launched_$lbl"
      break   # one launch per cycle
    done
  fi

  # 3) exit when all queued done+logged
  alldone=1
  for exp in "${EXPERIMENTS[@]}"; do IFS='|' read -r lbl _ _ _ <<<"$exp"; [ -f "runs/$lbl/.logged" ] || alldone=0; done
  [ "$alldone" = 1 ] && { echo "[driver] all queued experiments done"; break; }
  [ "$DRY" = "--dry" ] && { echo "[driver] dry-run one pass done"; break; }
  sleep 60
done
