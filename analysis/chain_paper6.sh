#!/usr/bin/env bash
# Paper 6 test: giant acceleration (v10_accel, SGLANG_TURING_GIANT_ACCEL=1) on the held node.
# Compares to v9_stock2 (same-node stock control from the Paper 5 chain). Full sweep; logs W&B; releases node.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing
L=/tmp/turing_chain_paper6.log
NODE=$(cat /tmp/turing_node.txt); JID=$(cat /tmp/turing_hold_jid.txt)
SOP=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher/.claude/skills
CELL=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher
source .venv/bin/activate 2>/dev/null; set -a; . /home/junyanch_google_com/autoresearch/.env 2>/dev/null; set +a
echo "[p6] START node=$NODE jid=$JID $(date -u +%H:%M:%S)" > "$L"

# free DRAM (kill any lingering server; probe flaky -> 40x then proceed, eval.sh gates DRAM anyway)
for i in $(seq 1 40); do
  m=$(timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f sglang.launch_server 2>/dev/null; sleep 1; awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo' 2>/dev/null | tail -1)
  echo "[p6] MemAvail=${m}G" >> "$L"; [ "${m:-0}" -ge 1300 ] 2>/dev/null && break; sleep 15
done

VER=v10_accel
echo "[p6] launch $VER (GIANT_ACCEL=1 theta=0.85 factor=2.0) $(date -u +%H:%M:%S)" >> "$L"
nohup srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 \
  bash -c "cd $CELL; export SGLANG_TURING_GIANT_ACCEL=1 SGLANG_TURING_ACCEL_THETA=0.85 SGLANG_TURING_ACCEL_FACTOR=2.0; bash .claude/skills/evaluation-sop/scripts/eval.sh turing $VER" \
  > /tmp/turing_$VER.log 2>&1 &
echo "[p6] launched $VER pid $! $(date -u +%H:%M:%S)" >> "$L"
for i in $(seq 1 240); do [ -f runs/$VER/summary.json ] && break; sleep 60; done
if [ -f runs/$VER/summary.json ]; then
  echo "[p6] $VER done; log W&B $(date -u +%H:%M:%S)" >> "$L"
  python3 "$SOP/report-sop/scripts/log_wandb.py" turing runs/$VER/summary.json "$VER" 8bf42d159 mechanism >> "$L" 2>&1
else
  echo "[p6] $VER NO summary after wait $(date -u +%H:%M:%S)" >> "$L"
fi
echo "[p6] release node $(date -u +%H:%M:%S)" >> "$L"; scancel "$JID"
echo "[p6] DONE $(date -u +%H:%M:%S)" >> "$L"
