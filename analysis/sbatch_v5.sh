#!/usr/bin/env bash
#SBATCH --job-name=valiant-v5
#SBATCH --nodes=1
#SBATCH --nodelist=slurm2-a3nodesetondem-2
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=07:00:00
#SBATCH --output=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/valiant/sbatch-v5-%j.log
# 4th same-node magnitude datapoint on freed certified node ondem-2 (clean exclusive grab, no held-pool).
# Serial (avoids flashinfer JIT race): stock_v5 (PIN off) -> v1_v5 (PIN on). Same-node A/B controls node variance.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
EVAL=$ROOT/programs/sgl/v0.31/research/researcher/.claude/skills/evaluation-sop/scripts/eval.sh
LOG=$ROOT/workspace/sgl/v0.31/research/researchers/valiant
echo "[v5] $(date -u +%H:%M:%S) START on $(hostname)"
env VALIANT_PIN_ENABLE=0 bash "$EVAL" valiant stock_v5 > "$LOG/eval-stock_v5.log" 2>&1
echo "[v5] $(date -u +%H:%M:%S) stock_v5 rc=$? ; starting v1_v5"
env VALIANT_PIN_ENABLE=1 bash "$EVAL" valiant v1_v5   > "$LOG/eval-v1_v5.log" 2>&1
echo "[v5] $(date -u +%H:%M:%S) v1_v5 rc=$? ; DONE"
