#!/usr/bin/env bash
# Fast node acquisition + eval. Polls flock every 2 seconds instead of 30.
# Usage: nohup bash fast_acquire.sh <name> <version> > fast_acquire.log 2>&1 &
set -uo pipefail
ROOT="${AUTORESEARCH_ROOT:-/home/junyanch_google_com/autoresearch}"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
SGL_HOME="/home/junyanch_google_com/autoresearch/programs/sgl/v0.25_ablations/sgl_mech"
EVAL="$SGL_HOME/researcher/.claude/skills/evaluation-sop/scripts/eval.sh"
POOL="${SGL_POOL_DIR:-$ROOT/workspace/sgl/v0.25_ablations/_pool}"

NAME="${1:?usage: fast_acquire.sh <name> <version>}"
VER="${2:?usage: fast_acquire.sh <name> <version>}"; shift 2

echo "[fast] $(date) starting acquisition for $NAME $VER (2s poll)"

while :; do
  for node_file in "$POOL/held"/*; do
    [ -f "$node_file" ] || continue
    node=$(basename "$node_file")
    jid=$(cat "$node_file" 2>/dev/null) || continue
    squeue -h -j "$jid" >/dev/null 2>&1 || continue

    exec 200>"$POOL/locks/$node.lock"
    if flock -n 200; then
      echo "[fast] $(date) ACQUIRED $node (job $jid) — running eval"
      srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" "$NAME" "$VER" "$@"
      rc=$?
      flock -u 200; exec 200>&-
      echo "[fast] $(date) EVAL COMPLETE rc=$rc"
      exit $rc
    fi
    exec 200>&-
  done
  sleep 2
done
