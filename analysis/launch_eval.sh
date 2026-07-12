#!/usr/bin/env bash
# valiant ablation launcher. Env-var-configured so multiple configs share ONE working tree
# (sbatch snapshots env at submit, so queued configs keep their own env). Usage:
#   VALIANT_PIN_ENABLE=1 VALIANT_PIN_FRACTION=0.5 bash analysis/launch_eval.sh <version-label>
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
CELL=$ROOT/programs/sgl/v0.31/research/researcher
VER="${1:?usage: launch_eval.sh <version-label>}"
export AUTORESEARCH_ROOT="$ROOT"
# pass-through env (defaults match code)
export VALIANT_PIN_ENABLE="${VALIANT_PIN_ENABLE:-1}"
export VALIANT_PIN_FRACTION="${VALIANT_PIN_FRACTION:-0.5}"
LOG=$ROOT/workspace/sgl/v0.31/research/researchers/valiant/eval-$VER.log
echo "[launch] $VER  PIN_ENABLE=$VALIANT_PIN_ENABLE FRACTION=$VALIANT_PIN_FRACTION"
cd "$CELL"
nohup bash .claude/skills/submit-gpu-job/scripts/eval-on-pool.sh valiant "$VER" > "$LOG" 2>&1 &
sleep 8; cat "$LOG"
