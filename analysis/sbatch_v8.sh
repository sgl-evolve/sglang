#!/usr/bin/env bash
#SBATCH --job-name=valiant-v8
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=07:00:00
#SBATCH --output=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/valiant/sbatch-v8-%j.log
# Clean same-node A/B at PIN_FRACTION=1.0 to (a) CONFIRM the budget-recovered gain (v7 showed ondem-2
# +1.8pp@λ3 at fraction 1.0 vs +0.1pp at 0.5 — gain is budget-gated), and (b) RE-STRESS the crash fix
# (v1 at max pins on whatever node). DRAM-settle wait between runs (v5 lesson).
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
EVAL=$ROOT/programs/sgl/v0.31/research/researcher/.claude/skills/evaluation-sop/scripts/eval.sh
LOG=$ROOT/workspace/sgl/v0.31/research/researchers/valiant
wait_dram(){ for i in $(seq 1 60); do local g=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo); echo "[v8] $(date -u +%H:%M:%S) DRAM=${g}G"; [ "${g:-0}" -gt 1450 ] && return 0; sleep 10; done; }
echo "[v8] $(date -u +%H:%M:%S) START on $(hostname)"
wait_dram
env VALIANT_PIN_ENABLE=0 bash "$EVAL" valiant stock_v8 > "$LOG/eval-stock_v8.log" 2>&1; echo "[v8] stock_v8 rc=$?"
sleep 30; wait_dram
env VALIANT_PIN_ENABLE=1 VALIANT_PIN_FRACTION=1.0 bash "$EVAL" valiant v1f1_v8 > "$LOG/eval-v1f1_v8.log" 2>&1; echo "[v8] v1f1_v8 rc=$?"
CR=$(grep -c "assert v == node" "$LOG/runs/v1f1_v8/server.log" 2>/dev/null || echo 0)
MP=$(grep -oE "pinned_reqs=[0-9]+" "$LOG/runs/v1f1_v8/server.log" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
echo "[v8] RESULT: v1 crashes=$CR max_pins=$MP ; DONE $(date -u +%H:%M:%S)"
