#!/usr/bin/env bash
#SBATCH --job-name=valiant-v6
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=07:00:00
#SBATCH --output=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/valiant/sbatch-v6-%j.log
# Re-attempt of the 4th magnitude datapoint. FIX vs v5: wait for host DRAM to FREE between the two runs
# (v5 failed: eval.sh's DRAM guard [needs >=1300G] tripped 'DRAM_TOO_LOW 327G' because stock's 768GB host KV
# cache hadn't freed in the ~4s trap window before v1's check). Here we poll MemAvailable until it recovers.
# Submit with an explicit free certified node: sbatch --nodelist=<node> analysis/sbatch_v6.sh
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
EVAL=$ROOT/programs/sgl/v0.31/research/researcher/.claude/skills/evaluation-sop/scripts/eval.sh
LOG=$ROOT/workspace/sgl/v0.31/research/researchers/valiant
wait_dram(){ # block until MemAvailable > 1450 GiB (host KV cache freed) or 10 min timeout
  for i in $(seq 1 60); do
    local g=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
    echo "[v6] $(date -u +%H:%M:%S) DRAM avail=${g}G"; [ "${g:-0}" -gt 1450 ] && return 0; sleep 10
  done; return 0; }
echo "[v6] $(date -u +%H:%M:%S) START on $(hostname)"
wait_dram
env VALIANT_PIN_ENABLE=0 bash "$EVAL" valiant stock_v6 > "$LOG/eval-stock_v6.log" 2>&1; echo "[v6] stock_v6 rc=$?"
echo "[v6] $(date -u +%H:%M:%S) settling DRAM before v1_v6"; sleep 30; wait_dram
env VALIANT_PIN_ENABLE=1 bash "$EVAL" valiant v1_v6   > "$LOG/eval-v1_v6.log"   2>&1; echo "[v6] v1_v6 rc=$?"
echo "[v6] $(date -u +%H:%M:%S) DONE"
