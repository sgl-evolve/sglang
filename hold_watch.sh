#!/usr/bin/env bash
# hold_watch.sh — wait for my exclusive hold job to land, then run queued evals into it via
# srun --overlap (guaranteed, race-free path). Keeps the hold for back-to-back hill-climbing.
# Coordinates with the micro-race launcher via a flag file so we never double-run the same version.
# Usage: hold_watch.sh <holdjid>
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a; export HF_TOKEN="$HF_API_KEY"
EVAL="$ROOT/programs/sgl/researcher/.claude/skills/evaluation-sop/scripts/eval.sh"
WS="$ROOT/workspace/sgl/researchers/kv-heron-e29"
NAME=kv-heron-e29
JID="${1:?holdjid}"

echo "[holdwatch] waiting for hold $JID to land $(date '+%H:%M:%S')"
while :; do
  st=$(squeue -h -j "$JID" -o '%T' 2>/dev/null)
  [ -z "$st" ] && { echo "[holdwatch] hold $JID gone (cancelled/ended); exiting"; exit 2; }
  [ "$st" = "RUNNING" ] && break
  sleep 20
done
NODE=$(squeue -h -j "$JID" -o '%N' 2>/dev/null)
echo "[holdwatch] hold $JID LANDED on $NODE $(date '+%H:%M:%S')"
echo "$NODE" > "$WS/.holdnode"

# Run each queued version into the held node. Queue file: one "VER<TAB>args..." per line.
QUEUE="$WS/.holdqueue"
run_ver(){
  local ver="$1"; shift
  local mutex="$WS/.mutex-$ver"
  if [ -f "$WS/runs/$ver/mix.txt" ] && grep -aqE "Benchmark duration" "$WS/runs/$ver/mix.txt" 2>/dev/null; then
    echo "[holdwatch] $ver already has a result; skipping"; return 0
  fi
  # atomic mutex shared with run_eval_pinned.sh (micro-race) so we never double-run into a run dir
  if ! mkdir "$mutex" 2>/dev/null; then echo "[holdwatch] $ver already running elsewhere (mutex); skipping"; return 0; fi
  echo "[holdwatch] RUN $ver on $NODE $(date '+%H:%M:%S'): $*"
  local try
  for try in 1 2 3; do
    srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 bash "$EVAL" "$NAME" "$ver" "$@"
    rc=$?
    if [ "$rc" -eq 0 ] || [ "$rc" -eq 8 ]; then echo "[holdwatch] $ver exit $rc (final)"; break; fi
    echo "[holdwatch] $ver attempt $try rc=$rc; retry"; mv "$WS/runs/$ver" "$WS/runs/$ver-failed-$try" 2>/dev/null || true; sleep 5
  done
  rmdir "$mutex" 2>/dev/null
}

while IFS=$'\t' read -r ver rest; do
  [ -z "$ver" ] && continue
  # shellcheck disable=SC2086
  run_ver "$ver" $rest
done < "$QUEUE"
# Good citizenship: the queue is drained (v26/v27 are the last planned bets — headline design space
# is exhausted), so RELEASE the exclusive 8-GPU node back to the oversubscribed pool immediately
# instead of squatting it idle. If more versions are ever queued later, re-submit a fresh hold.
echo "[holdwatch] queue drained $(date '+%H:%M:%S'); releasing hold $JID back to the pool"
scancel "$JID" 2>/dev/null || true
