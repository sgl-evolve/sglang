#!/usr/bin/env bash
# Paper 6 FIRMING: replicate the giant-accel WIN against the coin-flip. Same-node (19833) back-to-back:
# v10b_accel (accel #2) then v11_stock3 (stock #3). Both λ=3 land before the node's 07:24 timeout.
# Each logs W&B. Goal: accel n>=2 all-PASS (<8s) vs stock coin-flip → Fisher-style SLO-pass evidence.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing
L=/tmp/turing_chain_rep.log
NODE=$(cat /tmp/turing_node.txt); JID=$(cat /tmp/turing_hold_jid.txt)
SOP=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher/.claude/skills
CELL=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher
source .venv/bin/activate 2>/dev/null; set -a; . /home/junyanch_google_com/autoresearch/.env 2>/dev/null; set +a
echo "[rep] START node=$NODE jid=$JID $(date -u +%H:%M:%S)" > "$L"

free_dram(){
  for i in $(seq 1 40); do
    m=$(timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f sglang.launch_server 2>/dev/null; sleep 1; awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo' 2>/dev/null | tail -1)
    echo "[rep] MemAvail=${m}G" >> "$L"; [ "${m:-0}" -ge 1300 ] 2>/dev/null && break; sleep 15
  done
}
run(){ # $1=ver $2=env $3=commit
  echo "[rep] launch $1 ($2) $(date -u +%H:%M:%S)" >> "$L"
  nohup srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 \
    bash -c "cd $CELL; $2 bash .claude/skills/evaluation-sop/scripts/eval.sh turing $1" > /tmp/turing_$1.log 2>&1 &
  for i in $(seq 1 240); do [ -f runs/$1/summary.json ] && break; sleep 60; done
  if [ -f runs/$1/summary.json ]; then
    echo "[rep] $1 done; log $(date -u +%H:%M:%S)" >> "$L"
    python3 "$SOP/report-sop/scripts/log_wandb.py" turing runs/$1/summary.json "$1" "$3" mechanism >> "$L" 2>&1
  else echo "[rep] $1 NO summary $(date -u +%H:%M:%S)" >> "$L"; fi
}

free_dram
run v10b_accel "export SGLANG_TURING_GIANT_ACCEL=1 SGLANG_TURING_ACCEL_THETA=0.85 SGLANG_TURING_ACCEL_FACTOR=2.0;" 8bf42d159
free_dram
run v11_stock3 "" 8bf42d159
echo "[rep] release node $(date -u +%H:%M:%S)" >> "$L"; scancel "$JID"
echo "[rep] DONE $(date -u +%H:%M:%S)" >> "$L"
