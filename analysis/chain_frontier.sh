#!/usr/bin/env bash
# Paper 6 FRONTIER TEST: giant-accel FULL sweep (v13_accelfull, GIANT_ACCEL=1, rates 3/5/7/10) on a FRESH held node.
# Tests: (1) does λ=5 now PASS the SLO (goodput frontier 3→5)? (2) does C@λ10 RISE vs stock 4.14 (accel raises the
# capacity ceiling / P2 K-lever) or stay flat (accel only cuts λ=3 contention)? Logs W&B; releases node.
# Requires /tmp/turing_node.txt + /tmp/turing_hold_jid.txt set to a FRESH acquired node.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing
L=/tmp/turing_chain_frontier.log
NODE=$(cat /tmp/turing_node.txt); JID=$(cat /tmp/turing_hold_jid.txt)
SOP=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher/.claude/skills
CELL=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher
source .venv/bin/activate 2>/dev/null; set -a; . /home/junyanch_google_com/autoresearch/.env 2>/dev/null; set +a
echo "[frontier] START node=$NODE jid=$JID $(date -u +%H:%M:%S)" > "$L"
for i in $(seq 1 30); do
  m=$(timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f "sglang.launch_server|bench_serving" 2>/dev/null; sleep 1; awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo' 2>/dev/null | tail -1)
  echo "[frontier] MemAvail=${m}G" >> "$L"; [ "${m:-0}" -ge 1300 ] 2>/dev/null && break; sleep 15
done
VER=v13_accelfull
echo "[frontier] launch $VER (GIANT_ACCEL=1 full sweep) $(date -u +%H:%M:%S)" >> "$L"
nohup srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 \
  bash -c "cd $CELL; export SGLANG_TURING_GIANT_ACCEL=1 SGLANG_TURING_ACCEL_THETA=0.85 SGLANG_TURING_ACCEL_FACTOR=2.0; bash .claude/skills/evaluation-sop/scripts/eval.sh turing $VER" \
  > /tmp/turing_$VER.log 2>&1 &
for i in $(seq 1 260); do [ -f runs/$VER/summary.json ] && break; sleep 60; done
if [ -f runs/$VER/summary.json ]; then
  echo "[frontier] $VER done; log W&B $(date -u +%H:%M:%S)" >> "$L"
  python3 "$SOP/report-sop/scripts/log_wandb.py" turing runs/$VER/summary.json "$VER" 8bf42d159 mechanism >> "$L" 2>&1
else echo "[frontier] $VER NO summary $(date -u +%H:%M:%S)" >> "$L"; fi
echo "[frontier] release node $(date -u +%H:%M:%S)" >> "$L"; scancel "$JID"
echo "[frontier] DONE $(date -u +%H:%M:%S)" >> "$L"
