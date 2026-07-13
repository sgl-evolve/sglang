#!/usr/bin/env bash
#SBATCH --job-name=valiant-v9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=06:30:00
#SBATCH --output=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/valiant/sbatch-v9-%j.log
# NOISE-FLOOR HARDENING (addresses the #1 PC objection: n=3 underpowered).
# Two back-to-back STOCK full sweeps, SAME node, to (a) tighten the stock hit-rate noise floor at one node
# (n=3 -> n=5), and (b) add two more fixed-config p99 points that reinforce Table 3's claim that the
# +-127% tail swing is pure run-to-run variance (not mechanism). DRAM-settle wait between runs (v5 lesson).
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
EVAL=$ROOT/programs/sgl/v0.31/research/researcher/.claude/skills/evaluation-sop/scripts/eval.sh
LOG=$ROOT/workspace/sgl/v0.31/research/researchers/valiant
wait_dram(){ for i in $(seq 1 60); do local g=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo); echo "[v9] $(date -u +%H:%M:%S) DRAM=${g}G"; [ "${g:-0}" -gt 1450 ] && return 0; sleep 10; done; }
echo "[v9] $(date -u +%H:%M:%S) START on $(hostname)"
wait_dram
env VALIANT_PIN_ENABLE=0 bash "$EVAL" valiant stock_v9a > "$LOG/eval-stock_v9a.log" 2>&1; echo "[v9] stock_v9a rc=$?"
sleep 30; wait_dram
env VALIANT_PIN_ENABLE=0 bash "$EVAL" valiant stock_v9b > "$LOG/eval-stock_v9b.log" 2>&1; echo "[v9] stock_v9b rc=$?"
HA=$(awk -F, '$2==3{print $8}' "$LOG/runs/stock_v9a/curve.csv" 2>/dev/null)
HB=$(awk -F, '$2==3{print $8}' "$LOG/runs/stock_v9b/curve.csv" 2>/dev/null)
echo "[v9] RESULT: stock hit@3 v9a=$HA v9b=$HB (vs prior ondem-2 0.6721/0.6728/0.6813) ; DONE $(date -u +%H:%M:%S)"
