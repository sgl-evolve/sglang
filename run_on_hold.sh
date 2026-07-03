#!/usr/bin/env bash
# run_on_hold.sh — run the frozen eval.sh into MY self-locked exclusive certified node
# (holder job id in .holdjob). Exclusive => no foreign server, no collision, true --exclusive
# isolation (protocol-faithful). Unique PORT for extra safety.
#   nohup bash run_on_hold.sh <version> [extra eval args...] > eval-<version>.log 2>&1 &
set -uo pipefail
NAME=onyx-7q2
ROOT=/home/junyanch_google_com/autoresearch
EVAL=$ROOT/programs/sgl/researcher/.claude/skills/evaluation-sop/scripts/eval.sh
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; export HF_TOKEN="$HF_API_KEY"; }
export PORT="${PORT:-30729}"
VER="${1:?usage: run_on_hold.sh <version> [extra args...]}"; shift || true
cd "$ROOT/workspace/sgl/researchers/$NAME"
JID=$(cat .holdjob)
# wait until the holder is RUNNING and learn its node
while :; do
  st=$(squeue -h -j "$JID" -o "%t" 2>/dev/null)
  [ "$st" = "R" ] && break
  [ -z "$st" ] && { echo "[run_on_hold] holder job $JID gone"; exit 1; }
  echo "[run_on_hold] holder $JID state=$st, waiting..."; sleep 30
done
NODE=$(squeue -h -j "$JID" -o "%N" 2>/dev/null)
echo "[run_on_hold] eval $VER on my exclusive node $NODE (job $JID); port=$PORT extra=[$*]"
srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 --export=ALL,PORT="$PORT" bash "$EVAL" "$NAME" "$VER" "$@"
echo "[run_on_hold] eval $VER exit rc=$?"
