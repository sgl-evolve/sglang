#!/usr/bin/env bash
# Paper 6 DOSE-RESPONSE: giant-accel FACTOR sweep. f=2 is v13_accelfull (goodput@SLO 3->5, C~5.2).
# This runs f=3 (v14_accelf3) then f=1.5 (v15_accelf15), each a FULL rate sweep, on ONE held node.
# Q1: does a BIGGER boost (f=3) push the frontier PAST 5 (does lam=7 pass)? => goodput 3->7.
# Q2: is f=1.5 enough (frontier monotonic in f) or does f=2 already saturate (diminishing returns)?
# Requires /tmp/turing_node.txt + /tmp/turing_hold_jid.txt set to a FRESH acquired node. Logs W&B; releases node.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing
L=/tmp/turing_chain_factor.log
NODE=$(cat /tmp/turing_node.txt); JID=$(cat /tmp/turing_hold_jid.txt)
SOP=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher/.claude/skills
CELL=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher
source .venv/bin/activate 2>/dev/null; set -a; . /home/junyanch_google_com/autoresearch/.env 2>/dev/null; set +a
echo "[factor] START node=$NODE jid=$JID $(date -u +%H:%M:%S)" > "$L"

run_one () {  # $1=version  $2=factor
  local VER="$1" F="$2"
  for i in $(seq 1 30); do
    m=$(timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f "sglang.launch_server|bench_serving" 2>/dev/null; sleep 1; awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo' 2>/dev/null | tail -1)
    echo "[factor] $VER MemAvail=${m}G" >> "$L"; [ "${m:-0}" -ge 1300 ] 2>/dev/null && break; sleep 15
  done
  echo "[factor] launch $VER (GIANT_ACCEL=1 FACTOR=$F full sweep) $(date -u +%H:%M:%S)" >> "$L"
  nohup srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 \
    bash -c "cd $CELL; export SGLANG_TURING_GIANT_ACCEL=1 SGLANG_TURING_ACCEL_THETA=0.85 SGLANG_TURING_ACCEL_FACTOR=$F; bash .claude/skills/evaluation-sop/scripts/eval.sh turing $VER" \
    > /tmp/turing_$VER.log 2>&1 &
  for i in $(seq 1 260); do [ -f runs/$VER/summary.json ] && break; sleep 60; done
  if [ -f runs/$VER/summary.json ]; then
    echo "[factor] $VER done; log W&B $(date -u +%H:%M:%S)" >> "$L"
    python3 "$SOP/report-sop/scripts/log_wandb.py" turing runs/$VER/summary.json "$VER" 8bf42d159 mechanism >> "$L" 2>&1
  else echo "[factor] $VER NO summary $(date -u +%H:%M:%S)" >> "$L"; fi
}

run_one v14_accelf3  3.0
run_one v15_accelf15 1.5
echo "[factor] release node $(date -u +%H:%M:%S)" >> "$L"; scancel "$JID"
echo "[factor] DONE $(date -u +%H:%M:%S)" >> "$L"
