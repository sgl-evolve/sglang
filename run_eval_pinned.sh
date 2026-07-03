#!/usr/bin/env bash
# Disk-aware, lock-respecting eval launcher over the manager's held certified pool.
# flock-FIRST (grab the node the instant it frees), THEN disk-check; skip held nodes with < 1.8T free.
# On a transient load failure (hang/timeout during warmup), RETRY on the same held node up to
# MAXTRY times WHILE HOLDING THE FLOCK (so we don't lose the node to contention); if still failing,
# release and move to another node. Exits 0 only on a clean eval (exit 0) or a real MIX failure.
# Usage: run_eval_pinned.sh <name> <version> [eval args...]
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a; export HF_TOKEN="$HF_API_KEY"
RT=/home/junyanch_google_com/autoresearch/programs/sgl/manager/.runtime
EVAL=/home/junyanch_google_com/autoresearch/programs/sgl/researcher/.claude/skills/evaluation-sop/scripts/eval.sh
NAME="${1:?name}"; VER="${2:?version}"; shift 2
MINGB=1800; MAXTRY=2
while :; do
  any=0
  for f in "$RT"/held/*; do
    [ -e "$f" ] || continue
    node=$(basename "$f"); jid=$(cat "$f" 2>/dev/null) || continue
    squeue -h -j "$jid" >/dev/null 2>&1 || continue
    any=1
    exec 200>"$RT/locks/$node.lock"
    if flock -n 200; then
      freeG=$(timeout 12 srun --jobid="$jid" --overlap -N1 -w "$node" -t 0:01:00 \
                df -BG --output=avail /mnt/localssd 2>/dev/null | tail -1 | tr -dc '0-9')
      if [ -n "$freeG" ] && [ "$freeG" -ge "$MINGB" ]; then
        try=1
        while [ "$try" -le "$MAXTRY" ]; do
          echo "[pinned] eval $VER on $node (job $jid) free=${freeG}G, attempt $try/$MAXTRY"
          srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" "$NAME" "$VER" "$@"
          rc=$?
          # rc 0 = success; rc 8 = MIX_BENCH_FAILED (real, don't retry); else = load/timeout (retry)
          if [ "$rc" -eq 0 ] || [ "$rc" -eq 8 ]; then
            flock -u 200; exec 200>&-; echo "[pinned] eval $VER exit $rc (final)"; exit $rc
          fi
          echo "[pinned] eval $VER attempt $try failed (rc=$rc); retrying on same node"
          try=$((try+1)); sleep 5
        done
        echo "[pinned] $node: $MAXTRY load failures; releasing, trying another node"
        flock -u 200; exec 200>&-
      else
        echo "[pinned] $node: free=${freeG:-?}G < ${MINGB}G, release+skip"
        flock -u 200; exec 200>&-
      fi
    else
      exec 200>&-
    fi
  done
  [ "$any" -eq 0 ] && { echo "[pinned] no live held nodes"; exit 3; }
  sleep 15
done
