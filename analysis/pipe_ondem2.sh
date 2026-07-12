#!/usr/bin/env bash
# Fast pc hit-gain read on idle held node ondem-2 (job 19376): pc_e (post-completion) only.
# hit-rate is node-independent, so pc_e hit vs stock hit (cross-node) is a valid early read.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
EVAL=$ROOT/programs/sgl/v0.31/research/researcher/.claude/skills/evaluation-sop/scripts/eval.sh
POOL=$ROOT/workspace/sgl/v0.31/_pool
NODE=slurm2-a3nodesetondem-2; JID=19376
LOGDIR=$ROOT/workspace/sgl/v0.31/research/researchers/valiant
exec 200>"$POOL/locks/$NODE.lock"; flock 200 || { echo "lock fail"; exit 1; }
echo "[pipeE] locked $NODE $(date -u +%H:%M:%S)"
echo "[pipeE] START pc_e $(date -u +%H:%M:%S)"
env VALIANT_PIN_ENABLE=1 VALIANT_PC_ENABLE=1 srun --jobid=$JID --overlap -N1 -w $NODE --gres=gpu:8 \
    bash "$EVAL" valiant pc_e > "$LOGDIR/eval-pc_e.log" 2>&1
echo "[pipeE] END pc_e rc=$? $(date -u +%H:%M:%S)"
flock -u 200
