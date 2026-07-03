#!/usr/bin/env bash
# launch_eval.sh <version> [extra eval args...]
# Self-healing, race-winning eval launcher for the manager-held certified pool.
# FLOCK-FIRST: on each poll it tries to grab each held node's lock immediately
# (winning the instant it frees), then checks free /mnt/localssd; if >=MIN_GB it
# runs the FIXED evaluator, else releases and tries the next. On failure (no
# summary.json -> crash/timeout, e.g. flaky fabric) it BLACKLISTS that node.
# Succeeds only when runs/<ver>/summary.json is produced. Run backgrounded.
set -uo pipefail
NAME=quill-7m3
ROOT=/home/junyanch_google_com/autoresearch
RT=$ROOT/programs/sgl/manager/.runtime
EVAL=$ROOT/programs/sgl/researcher/.claude/skills/evaluation-sop/scripts/eval.sh
WORK=$ROOT/workspace/sgl/researchers/$NAME
VER="${1:?usage: launch_eval.sh <version> [extra args...]}"; shift || true
SUMMARY="$WORK/runs/$VER/summary.json"
MIN_GB=1850
POLL=5
EXCLUDE_NODES="${EXCLUDE_NODES:-}"   # fusion-disable fix makes all nodes usable (their SIGBUS was the fusion, not fabric)
BLACKLIST=""

held(){ for f in "$RT"/held/*; do [ -e "$f" ] && basename "$f"; done; }
skip(){ for e in $EXCLUDE_NODES $BLACKLIST; do [ "$1" = "$e" ] && return 0; done; return 1; }
freegb(){ timeout 22 srun --jobid="$1" --overlap -N1 -w "$2" \
    bash -c 'df -BG /mnt/localssd 2>/dev/null|tail -1|awk "{gsub(/G/,\"\",\$4);print \$4}"' 2>/dev/null; }

attempt=0
while :; do
  attempt=$((attempt+1))
  for node in $(held); do
    skip "$node" && continue
    jid=$(cat "$RT/held/$node" 2>/dev/null) || continue
    squeue -h -j "$jid" >/dev/null 2>&1 || continue
    exec 200>"$RT/locks/$node.lock"
    # Blocking wait in the kernel lock queue (win the race the instant it frees).
    if ! flock -w 8 200; then exec 200>&-; continue; fi
    gb=$(freegb "$jid" "$node"); gb=${gb:-0}
    if ! [[ "$gb" =~ ^[0-9]+$ ]] || (( gb < MIN_GB )); then
      echo "[launch #$attempt] locked $node but ${gb}G < ${MIN_GB}G, release @ $(date +%H:%M:%S)"
      flock -u 200; exec 200>&-; continue
    fi
    echo "[launch #$attempt] EVAL $VER on $node (${gb}G, hold $jid) @ $(date +%H:%M:%S)"
    rm -f "$SUMMARY" 2>/dev/null
    srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" "$NAME" "$VER" "$@"
    rc=$?
    flock -u 200; exec 200>&-
    if [ -f "$SUMMARY" ]; then
      echo "[launch] SUCCESS $VER on $node (rc=$rc) @ $(date +%H:%M:%S)"; exit 0
    fi
    echo "[launch] FAIL $VER on $node (rc=$rc, no summary) -> blacklist @ $(date +%H:%M:%S)"
    BLACKLIST="$BLACKLIST $node"
  done
  sleep "$POLL"
done
