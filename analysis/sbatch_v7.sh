#!/usr/bin/env bash
#SBATCH --job-name=valiant-v7fix
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/valiant/sbatch-v7-%j.log
# GPU stress-VALIDATION of the eviction crash fix (both-set discard in _remove_leaf_from_parent).
# STRESS: VALIANT_PIN_FRACTION=1.0 forces high pin counts (crash needed ~26 pins at λ7 pre-fix).
# PASS = full λ-sweep completes with NO `assert v==node` crash AND pin counts reach the crash regime (>=20).
# Submit fairly over the 4 VERIFIED nodes: sbatch --nodes=1 --nodelist=<4 verified> analysis/sbatch_v7.sh
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
EVAL=$ROOT/programs/sgl/v0.31/research/researcher/.claude/skills/evaluation-sop/scripts/eval.sh
LOG=$ROOT/workspace/sgl/v0.31/research/researchers/valiant
echo "[v7] $(date -u +%H:%M:%S) START on $(hostname) — fix stress-validation, PIN_FRACTION=1.0"
env VALIANT_PIN_ENABLE=1 VALIANT_PIN_FRACTION=1.0 bash "$EVAL" valiant v1_fixstress > "$LOG/eval-v1_fixstress.log" 2>&1
rc=$?
echo "[v7] $(date -u +%H:%M:%S) v1_fixstress rc=$rc"
CR=$(grep -c "assert v == node" "$LOG/runs/v1_fixstress/server.log" 2>/dev/null || echo 0)
MP=$(grep -oE "pinned_reqs=[0-9]+" "$LOG/runs/v1_fixstress/server.log" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
echo "[v7] RESULT: assert-crashes=$CR  max_pins=$MP  (PASS if crashes=0 and max_pins>=20)"
echo "[v7] $(date -u +%H:%M:%S) DONE"
