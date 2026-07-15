#!/usr/bin/env bash
# P6 DEPTH: trigger-signal ablation for giant acceleration. Which pressure signal best gates the boost —
# device-KV occupancy (occ, the current mechanism), waiting-queue depth (queue, P5's HOL-blocking signal),
# or either? Same-node lam=3-only A/B on node2 (19994): n=2 each of occ/queue/either, alternated for pairing.
# Deterministic tpot/concurrency is the coin-flip-ROBUST comparison (does queue-trigger match/beat occ?);
# SLO-pass is the coin-flip-sensitive headline. All accel (GIANT_ACCEL=1, FACTOR=2.0). Efficient (kill after
# bench_r3, ~55min/run). Wall-time-guarded under the 9h hold. Uses node2 temp files (NOT the firming node).
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing
L=/tmp/turing_chain_trig.log
NODE=$(cat /tmp/turing_node2.txt); JID=$(cat /tmp/turing_hold_jid2.txt)
SOP=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher/.claude/skills
CELL=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher
source .venv/bin/activate 2>/dev/null; set -a; . /home/junyanch_google_com/autoresearch/.env 2>/dev/null; set +a
START=$(date +%s); DEADLINE=$((6*3600 + 1800))
BASE="export SGLANG_TURING_GIANT_ACCEL=1 SGLANG_TURING_ACCEL_THETA=0.85 SGLANG_TURING_ACCEL_FACTOR=2.0"
COMMIT=54aa8eb75
echo "[trig] START node=$NODE jid=$JID $(date -u +%H:%M:%S) deadline=+6.5h" > "$L"

free_dram(){
  for i in $(seq 1 30); do
    m=$(timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f "[s]glang.launch_server|[b]ench_serving" 2>/dev/null; sleep 1; awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo' 2>/dev/null | tail -1)
    echo "[trig] MemAvail=${m}G" >> "$L"; [ "${m:-0}" -ge 1300 ] 2>/dev/null && break; sleep 15
  done
}
kill_srv(){ timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f "[s]glang.launch_server|[b]ench_serving" 2>/dev/null; echo k' >/dev/null 2>&1 || true; }
caplog(){
  python3 - "$1" "$2" >> "$L" 2>&1 <<'PY'
import json,sys,csv
ver,commit=sys.argv[1],sys.argv[2]
b=json.load(open(f'runs/{ver}/bench_r3.json'))
hit=0.0
try:
  for r in csv.DictReader(open(f'runs/{ver}/curve.csv')):
    if r['rate']=='3' and r.get('hit_rate'): hit=float(r['hit_rate'])
except Exception: pass
p99=b['p99_ttft_ms']
d={'version':ver,'commit':commit,'note':'Paper6 trigger-signal ablation lam=3-only','panel':{
 'overall/goodput_reqs_at_SLO':(3.0 if p99<=8000 else 0),'overall/peak_req_s':b['request_throughput'],
 'overall/hit_rate':hit,'overall/slo_ms':8000.0,'curve/ttft_p99@3':p99,
 'curve/tpot_ms@3':b['mean_tpot_ms'],'curve/concurrency@3':b['concurrency']},
 'curve':[{'label':'sweep','rate':'3','req_throughput':str(b['request_throughput']),'out_tok_s':str(b['output_throughput']),'ttft_p50_ms':str(b['median_ttft_ms']),'ttft_p99_ms':str(p99),'e2e_p99_ms':'','hit_rate':str(hit)}]}
json.dump(d,open(f'runs/{ver}/summary.json','w'),indent=1)
print(f'[trig] {ver} lam=3 p99={p99:.0f} PASS={p99<=8000} tpot={b["mean_tpot_ms"]:.0f} conc={b["concurrency"]:.0f} completed={b["completed"]}')
PY
  python3 "$SOP/report-sop/scripts/log_wandb.py" turing runs/$1/summary.json "$1" "$2" mechanism >> "$L" 2>&1
}
run3(){ # $1=ver $2=extra-env
  rm -f runs/$1/bench_r3.json 2>/dev/null
  echo "[trig] launch $1 ($2) $(date -u +%H:%M:%S)" >> "$L"
  nohup srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 \
    bash -c "cd $CELL; $BASE $2; bash .claude/skills/evaluation-sop/scripts/eval.sh turing $1" > /tmp/turing_$1.log 2>&1 &
  for i in $(seq 1 95); do [ -f runs/$1/bench_r3.json ] && break; sleep 60; done
  sleep 25
  if [ -f runs/$1/bench_r3.json ]; then echo "[trig] $1 lam=3 landed $(date -u +%H:%M:%S)" >> "$L"; caplog "$1" "$COMMIT"; else echo "[trig] $1 NO bench_r3 $(date -u +%H:%M:%S)" >> "$L"; fi
  kill_srv; free_dram
}
maybe(){ # $1=ver $2=extra-env : wall-time + skip-if-done guard
  local now; now=$(( $(date +%s) - START ))
  if [ "$now" -ge "$DEADLINE" ]; then echo "[trig] past deadline (${now}s) -> stop" >> "$L"; return 1; fi
  [ -f runs/$1/summary.json ] && { echo "[trig] $1 already done -> skip" >> "$L"; return 0; }
  run3 "$1" "$2"
}

free_dram
maybe v19_occ1    "SGLANG_TURING_ACCEL_SIGNAL=occ"
maybe v19_queue1  "SGLANG_TURING_ACCEL_SIGNAL=queue SGLANG_TURING_ACCEL_QTHRESH=2"
maybe v19_either1 "SGLANG_TURING_ACCEL_SIGNAL=either SGLANG_TURING_ACCEL_QTHRESH=2"
maybe v19_occ2    "SGLANG_TURING_ACCEL_SIGNAL=occ"
maybe v19_queue2  "SGLANG_TURING_ACCEL_SIGNAL=queue SGLANG_TURING_ACCEL_QTHRESH=2"
maybe v19_either2 "SGLANG_TURING_ACCEL_SIGNAL=either SGLANG_TURING_ACCEL_QTHRESH=2"
echo "[trig] release node $(date -u +%H:%M:%S)" >> "$L"; scancel "$JID"
echo "[trig] DONE $(date -u +%H:%M:%S)" >> "$L"
