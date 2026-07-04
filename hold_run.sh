#!/usr/bin/env bash
# hold_run.sh <version> [eval policy args...]
# Runs ONE eval into MY session hold (jid in .hold_jid) once it is RUNNING, via
# `srun --jobid=<hold> --overlap`. The hold is an --exclusive certified node I own for
# the session (charter: "no functional held pool -> lock ONE certified node"), so there
# is no neighbor and no flock race. Env vars (e.g. KVLYNX_EVICT_FREQ_ALPHA) propagate via
# --export=ALL. Blocks until the hold lands + eval finishes -> run in background.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; export HF_TOKEN="${HF_API_KEY:-}"; }
WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL=$ROOT/programs/sgl/researcher/.claude/skills/evaluation-sop/scripts/eval.sh
NAME=kv-lynx-4d2
VER="${1:?usage: hold_run.sh <version> [args...]}"; shift || true
holdjid=$(cat "$WORK/.hold_jid")

# 1) wait for the hold to be RUNNING and learn its node
while :; do
  st=$(squeue -h -j "$holdjid" -o "%T" 2>/dev/null)
  [ -z "$st" ] && { echo "[hold_run] hold $holdjid gone (not in queue) — aborting $VER"; exit 9; }
  if [ "$st" = "RUNNING" ]; then
    node=$(squeue -h -j "$holdjid" -o "%N" 2>/dev/null)
    [ -n "$node" ] && break
  fi
  echo "[hold_run] hold $holdjid $st (waiting for node) $(date +%H:%M:%S)"; sleep 30
done
echo "[hold_run] hold landed on $node — running $VER $(date +%H:%M:%S)"

# 2) health gate (should be clean on an exclusive node); wait briefly for disk if needed
for _t in 1 2 3 4 5 6; do
  read free_kb gpu_max < <(srun --jobid="$holdjid" --overlap -N1 -w "$node" bash -c \
    "echo \$(df --output=avail /mnt/localssd | tail -1) \$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" 2>/dev/null)
  free_kb=${free_kb:-0}; gpu_max=${gpu_max:-999999}
  echo "[hold_run] $node disk=$((free_kb/1024/1024))G gpu_max=${gpu_max}MiB"
  [ "$free_kb" -ge 1932735283 ] && [ "$gpu_max" -lt 10000 ] && break
  sleep 20
done

# 3) run the eval into my held node (whole node, no --overlap contention worries)
srun --export=ALL --jobid="$holdjid" --overlap -N1 -w "$node" --gres=gpu:8 \
  bash "$EVAL" "$NAME" "$VER" "$@"
rc=$?
echo "[hold_run] $VER eval exited rc=$rc $(date +%H:%M:%S)"
exit $rc
