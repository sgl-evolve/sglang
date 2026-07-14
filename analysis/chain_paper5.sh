#!/usr/bin/env bash
# Paper 5 same-node A/B on the held node: (1) fair-share chunk interleaving (v8_fair,
# SGLANG_TURING_FAIR_PREFILL=1) then (2) fresh same-node stock control (v9_stock2), each full sweep,
# each logged to W&B. Fair FIRST (de-risk vs 9h timeout: if stock is cut, fall back to existing v1_stock).
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing
L=/tmp/turing_chain_paper5.log
NODE=$(cat /tmp/turing_node.txt); JID=$(cat /tmp/turing_hold_jid.txt)
SOP=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher/.claude/skills
CELL=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher
source .venv/bin/activate 2>/dev/null; set -a; . /home/junyanch_google_com/autoresearch/.env 2>/dev/null; set +a
echo "[p5] START node=$NODE jid=$JID $(date -u +%H:%M:%S)" > "$L"

run_eval(){  # $1=version  $2=extra-env-exports
  local VER="$1" ENVX="$2"
  echo "[p5] launch $VER ($ENVX) $(date -u +%H:%M:%S)" >> "$L"
  nohup srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 \
    bash -c "cd $CELL; $ENVX bash .claude/skills/evaluation-sop/scripts/eval.sh turing $VER" \
    > /tmp/turing_$VER.log 2>&1 &
  echo "[p5] launched $VER pid $! $(date -u +%H:%M:%S)" >> "$L"
  # wait for summary.json (up to ~4h)
  for i in $(seq 1 240); do [ -f runs/$VER/summary.json ] && break; sleep 60; done
  if [ -f runs/$VER/summary.json ]; then
    echo "[p5] $VER done; log W&B $(date -u +%H:%M:%S)" >> "$L"
    python3 "$SOP/report-sop/scripts/log_wandb.py" turing runs/$VER/summary.json "$VER" 48ad6612b mechanism >> "$L" 2>&1
  else
    echo "[p5] $VER NO summary after wait — abort chain $(date -u +%H:%M:%S)" >> "$L"; return 1
  fi
}

free_dram(){
  for i in $(seq 1 40); do
    m=$(timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f sglang.launch_server 2>/dev/null; sleep 1; awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo' 2>/dev/null | tail -1)
    echo "[p5] MemAvail=${m}G" >> "$L"; [ "${m:-0}" -ge 1300 ] 2>/dev/null && break; sleep 15
  done
}

# (1) fair-share chunk interleaving
run_eval v8_fair "export SGLANG_TURING_FAIR_PREFILL=1 SGLANG_TURING_FAIR_FRAC=0.5;" || { scancel "$JID"; exit 1; }
free_dram
# (2) fresh same-node stock control
run_eval v9_stock2 "" || { scancel "$JID"; exit 1; }

echo "[p5] BOTH done; release node $(date -u +%H:%M:%S)" >> "$L"
scancel "$JID"
echo "[p5] DONE $(date -u +%H:%M:%S)" >> "$L"
