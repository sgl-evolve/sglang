#!/usr/bin/env bash
# P7 candidate: server-side CONCURRENCY-CAP characterization (untested axis; distinct from P1's L2-backup admission).
# --max-running-requests N caps the running batch. Hypothesis A (basin-control POSITIVE): capping below the runaway
# threshold prevents the metastable bad basin => reliable goodput@SLO. Hypothesis B (bounded NEGATIVE): capping just
# trades decode-interference for admission queue-wait (TTFT includes queue-wait) => neutral/worse. Full rate sweep each
# (goodput@SLO curve). STOCK-based (no accel) to isolate the cap. Same-node A/B: cap64 vs stock control vs cap96.
# On-contract: --max-running-requests NOT forbidden, appended as EXTRA (server-side; MAXC=256 is the CLIENT cap, frozen).
# DRAM-race-safe (gate>=1600 + 60s settle), bracket-trick pkill. W&B-logs full summary; releases node.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing
L=/tmp/turing_chain_cap.log
NODE=$(cat /tmp/turing_node.txt); JID=$(cat /tmp/turing_hold_jid.txt)
SOP=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher/.claude/skills
CELL=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher
source .venv/bin/activate 2>/dev/null; set -a; . /home/junyanch_google_com/autoresearch/.env 2>/dev/null; set +a
START=$(date +%s); DEADLINE=$((6*3600))   # stop LAUNCHING new sweeps after +6h (each full sweep ~3h; last finishes < 9h hold)
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
echo "[cap] START node=$NODE jid=$JID $(date -u +%H:%M:%S) commit=$COMMIT deadline=+6h" > "$L"

free_dram(){
  for i in $(seq 1 40); do
    m=$(timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f "[s]glang.launch_server|[b]ench_serving" 2>/dev/null; sleep 1; awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo' 2>/dev/null | tail -1)
    echo "[cap] MemAvail=${m}G" >> "$L"; [ "${m:-0}" -ge 1600 ] 2>/dev/null && break; sleep 15
  done
  sleep 60   # DRAM-race fix: let the 768GB pool fully reclaim before the next server allocates
}
run_sweep(){ # $1=ver  $2=extra-launch-args (e.g. "--max-running-requests 64" or "")
  echo "[cap] launch $1 (extra='$2') $(date -u +%H:%M:%S)" >> "$L"
  nohup srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 \
    bash -c "cd $CELL; bash .claude/skills/evaluation-sop/scripts/eval.sh turing $1 $2" > /tmp/turing_$1.log 2>&1 &
  for i in $(seq 1 210); do [ -f runs/$1/summary.json ] && break; sleep 60; done
  if [ -f runs/$1/summary.json ]; then
    gp=$(python3 -c "import json;print(json.load(open('runs/$1/summary.json'))['panel'].get('overall/goodput_reqs_at_SLO','?'))" 2>/dev/null)
    echo "[cap] $1 DONE goodput@SLO=$gp $(date -u +%H:%M:%S)" >> "$L"
    python3 "$SOP/report-sop/scripts/log_wandb.py" turing runs/$1/summary.json "$1" "$COMMIT" mechanism >> "$L" 2>&1
  else echo "[cap] $1 NO summary $(date -u +%H:%M:%S)" >> "$L"; fi
}
maybe(){ local now; now=$(( $(date +%s) - START ))
  if [ "$now" -ge "$DEADLINE" ]; then echo "[cap] past deadline -> stop" >> "$L"; return 1; fi
  [ -f runs/$1/summary.json ] && { echo "[cap] $1 already done -> skip" >> "$L"; return 0; }
  free_dram; run_sweep "$1" "$2"
}

maybe v20_cap64  "--max-running-requests 64"
maybe v20_stock0 ""
maybe v20_cap96  "--max-running-requests 96"
echo "[cap] release node $(date -u +%H:%M:%S)" >> "$L"; scancel "$JID"
echo "[cap] DONE $(date -u +%H:%M:%S)" >> "$L"
