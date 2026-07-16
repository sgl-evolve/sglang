#!/usr/bin/env bash
# Firm the flagship's PRIMARY DETERMINISTIC HEADLINE (+36% capacity, C 4.14->5.65): stock baseline is currently n=1
# (v1_stock lam=10=4.14) and accel-f2 is n=2 (v13 5.65 / v16 5.79). Run FULL sweeps to lam=10 to push stock->n>=2 and
# accel-f2->n=3, strengthening both legs of +36% for top-venue rigor. Capacity = saturated (lam=10) achieved throughput.
# stock = default write_through (no accel); accel = GIANT_ACCEL=1 f=2. Full sweep (wait for summary.json). DRAM
# gate>=1700 + 90s settle; bracket pkill. W&B-logs; releases node. Requires /tmp/turing_node.txt + hold_jid.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing
L=/tmp/turing_chain_capfirm.log
NODE=$(cat /tmp/turing_node.txt); JID=$(cat /tmp/turing_hold_jid.txt)
SOP=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher/.claude/skills
CELL=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher
source .venv/bin/activate 2>/dev/null; set -a; . /home/junyanch_google_com/autoresearch/.env 2>/dev/null; set +a
START=$(date +%s); DEADLINE=$((6*3600))   # stop launching new sweeps after +6h (each ~3h; last finishes < 9h hold)
ACC="export SGLANG_TURING_GIANT_ACCEL=1 SGLANG_TURING_ACCEL_THETA=0.85 SGLANG_TURING_ACCEL_FACTOR=2.0;"
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
echo "[capfirm] START node=$NODE jid=$JID $(date -u +%H:%M:%S) commit=$COMMIT" > "$L"
free_dram(){
  for i in $(seq 1 40); do
    m=$(timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f "[s]glang.launch_server|[b]ench_serving" 2>/dev/null; sleep 1; awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo' 2>/dev/null | tail -1)
    echo "[capfirm] MemAvail=${m}G" >> "$L"; [ "${m:-0}" -ge 1700 ] 2>/dev/null && break; sleep 15
  done
  sleep 90
}
run_sweep(){ # $1=ver  $2=env-prefix ("" for stock, $ACC for accel)
  echo "[capfirm] launch $1 $(date -u +%H:%M:%S)" >> "$L"
  nohup srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 \
    bash -c "cd $CELL; $2 bash .claude/skills/evaluation-sop/scripts/eval.sh turing $1" > /tmp/turing_$1.log 2>&1 &
  for i in $(seq 1 200); do [ -f runs/$1/summary.json ] && break; sleep 60; done
  if [ -f runs/$1/summary.json ]; then
    cap=$(python3 -c "import json;j=json.load(open('runs/$1/summary.json'));c={r['rate']:r.get('req_throughput') for r in j['curve'] if isinstance(r,dict)};print(c.get('10','?'))" 2>/dev/null)
    echo "[capfirm] $1 DONE lam10_capacity=$cap $(date -u +%H:%M:%S)" >> "$L"
    python3 "$SOP/report-sop/scripts/log_wandb.py" turing runs/$1/summary.json "$1" "$COMMIT" mechanism >> "$L" 2>&1
  else echo "[capfirm] $1 NO summary $(date -u +%H:%M:%S)" >> "$L"; fi
  timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f "[s]glang.launch_server|[b]ench_serving"' >/dev/null 2>&1 || true
}
maybe(){ local now; now=$(( $(date +%s) - START ))
  if [ "$now" -ge "$DEADLINE" ]; then echo "[capfirm] past deadline -> stop" >> "$L"; return 1; fi
  [ -f runs/$1/summary.json ] && { echo "[capfirm] $1 done -> skip" >> "$L"; return 0; }
  free_dram; run_sweep "$1" "$2"
}
maybe v23_stockcap1 ""
maybe v23_accelcap1 "$ACC"
maybe v23_stockcap2 ""
echo "[capfirm] release node $(date -u +%H:%M:%S)" >> "$L"; scancel "$JID"
echo "[capfirm] DONE $(date -u +%H:%M:%S)" >> "$L"
