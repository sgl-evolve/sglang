#!/usr/bin/env bash
# Disk-aware, lock-respecting eval launcher over the manager's held certified pool.
# Same semantics as eval-on-pool.sh but skips held nodes whose /mnt/localssd has < 1.8T free
# (the pool currently contains nodes with leftover data that fail eval.sh's disk gate).
# Usage: run_eval_pinned.sh <name> <version> [eval args...]
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a; export HF_TOKEN="$HF_API_KEY"
RT=/home/junyanch_google_com/autoresearch/programs/sgl/manager/.runtime
EVAL=/home/junyanch_google_com/autoresearch/programs/sgl/researcher/.claude/skills/evaluation-sop/scripts/eval.sh
NAME="${1:?name}"; VER="${2:?version}"; shift 2
MINGB=1800
while :; do
  any=0
  for f in "$RT"/held/*; do
    [ -e "$f" ] || continue
    node=$(basename "$f"); jid=$(cat "$f" 2>/dev/null) || continue
    squeue -h -j "$jid" >/dev/null 2>&1 || continue          # hold job dead -> skip
    any=1
    freeG=$(srun --jobid="$jid" --overlap -N1 -w "$node" -t 0:01:00 \
              df -BG --output=avail /mnt/localssd 2>/dev/null | tail -1 | tr -dc '0-9')
    [ -z "$freeG" ] && { echo "[pinned] $node: cannot stat disk, skip"; continue; }
    if [ "$freeG" -lt "$MINGB" ]; then echo "[pinned] $node: ${freeG}G < ${MINGB}G, skip"; continue; fi
    exec 200>"$RT/locks/$node.lock"
    if flock -n 200; then
      echo "[pinned] running eval $VER on held node $node (job $jid), free=${freeG}G"
      srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" "$NAME" "$VER" "$@"
      rc=$?; flock -u 200; exec 200>&-
      echo "[pinned] eval $VER exit code: $rc"; exit $rc
    fi
    exec 200>&-                                              # locked by another researcher
  done
  [ "$any" -eq 0 ] && { echo "[pinned] no live held nodes"; exit 3; }
  echo "[pinned] all good held nodes busy — waiting 30s..."; sleep 30
done
