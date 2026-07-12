#!/usr/bin/env bash
# Primary same-node pipeline on held node 0-3 (hold job 19375): stock_c -> v1_c -> pc_c.
# nohup'd so it survives session boundaries; flock so no collision. All 3 same-node = clean A/B.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
EVAL=$ROOT/programs/sgl/v0.31/research/researcher/.claude/skills/evaluation-sop/scripts/eval.sh
POOL=$ROOT/workspace/sgl/v0.31/_pool
NODE=slurm2-a3nodeset1-2
JID=19374
LOGDIR=$ROOT/workspace/sgl/v0.31/research/researchers/valiant
run(){ # <version> <extra env assignments...>
  local ver="$1"; shift
  echo "[pipe03] $(date -u +%H:%M:%S) START $ver ($*)"
  env "$@" srun --jobid=$JID --overlap -N1 -w $NODE --gres=gpu:8 \
      bash "$EVAL" valiant "$ver" > "$LOGDIR/eval-$ver.log" 2>&1
  echo "[pipe03] $(date -u +%H:%M:%S) END $ver rc=$?"
}
exec 200>"$POOL/locks/$NODE.lock"
flock 200 || { echo "[pipe03] could not lock $NODE"; exit 1; }
echo "[pipe03] locked $NODE, starting pipeline $(date -u +%H:%M:%S)"
run stock_c VALIANT_PIN_ENABLE=0
run v1_c   VALIANT_PIN_ENABLE=1 VALIANT_PC_ENABLE=0
run pc_c   VALIANT_PIN_ENABLE=1 VALIANT_PC_ENABLE=1
flock -u 200
echo "[pipe03] pipeline DONE $(date -u +%H:%M:%S)"
