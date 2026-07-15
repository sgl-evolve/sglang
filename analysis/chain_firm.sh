#!/usr/bin/env bash
# Paper 6 FIRMING (efficient lam=3-only): the headline reliable win is accel 4/4 vs stock 1/5 (Fisher SLO-pass
# p=0.0397 = weakest leg; MWU tpot/conc already p<0.02). ADD same-node paired accel/stock lam=3 replicates
# under NEW names (v18_f*) to push Fisher p<0.01. Each replicate = fresh server (independent cold-start coin-flip
# draw) -> lam=3 benchmark; we wait only for bench_r3.json (~55min), capture+log, KILL server (skip lam5/7/10),
# free DRAM, next. Alternate accel/stock (accel first). Wall-time guarded so nothing outlives the 9h hold.
# NB: these are lam=3-ONLY firming replicates (goodput panel reflects the lam=3 point only) -- do NOT cite as
# full-sweep goodput; they feed the lam=3 SLO-pass Fisher table + the coin-flip-robust tpot/conc distributions.
# Requires /tmp/turing_node.txt + /tmp/turing_hold_jid.txt (a held node).
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing
L=/tmp/turing_chain_firm.log
NODE=$(cat /tmp/turing_node.txt); JID=$(cat /tmp/turing_hold_jid.txt)
SOP=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher/.claude/skills
CELL=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher
source .venv/bin/activate 2>/dev/null; set -a; . /home/junyanch_google_com/autoresearch/.env 2>/dev/null; set +a
START=$(date +%s); DEADLINE=$((6*3600 + 1800))   # stop LAUNCHING after +6.5h (leaves margin under the 9h hold)
ACCEL_ENV="export SGLANG_TURING_GIANT_ACCEL=1 SGLANG_TURING_ACCEL_THETA=0.85 SGLANG_TURING_ACCEL_FACTOR=2.0;"
echo "[firm] START node=$NODE jid=$JID $(date -u +%H:%M:%S) deadline=+6.5h" > "$L"

kill_srv(){ timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f "[s]glang.launch_server|[b]ench_serving" 2>/dev/null; echo k' >/dev/null 2>&1 || true; }
free_dram(){
  for i in $(seq 1 30); do
    m=$(timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f "[s]glang.launch_server|[b]ench_serving" 2>/dev/null; sleep 1; awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo' 2>/dev/null | tail -1)
    echo "[firm] MemAvail=${m}G" >> "$L"; [ "${m:-0}" -ge 1600 ] 2>/dev/null && break; sleep 15
  done
  sleep 60   # DRAM-race fix: wait for the 768GB pool to FULLY reclaim (gate fires mid-climb otherwise -> OOM)
}
caplog(){ # $1=ver $2=commit — build a lam=3 summary from bench_r3.json + curve, log W&B
  python3 - "$1" "$2" >> "$L" 2>&1 <<'PY'
import json,sys,csv,os
ver,commit=sys.argv[1],sys.argv[2]
b=json.load(open(f'runs/{ver}/bench_r3.json'))
hit=0.0
try:
  for r in csv.DictReader(open(f'runs/{ver}/curve.csv')):
    if r['rate']=='3' and r.get('hit_rate'): hit=float(r['hit_rate'])
except Exception: pass
p99=b['p99_ttft_ms']
d={'version':ver,'commit':commit,'note':'Paper6 firming lam=3-only replicate','panel':{
 'overall/goodput_reqs_at_SLO':(3.0 if p99<=8000 else 0),'overall/peak_req_s':b['request_throughput'],
 'overall/hit_rate':hit,'overall/slo_ms':8000.0,'curve/ttft_p99@3':p99,
 'curve/tpot_ms@3':b['mean_tpot_ms'],'curve/concurrency@3':b['concurrency']},
 'curve':[{'label':'sweep','rate':'3','req_throughput':str(b['request_throughput']),'out_tok_s':str(b['output_throughput']),'ttft_p50_ms':str(b['median_ttft_ms']),'ttft_p99_ms':str(p99),'e2e_p99_ms':'','hit_rate':str(hit)}]}
json.dump(d,open(f'runs/{ver}/summary.json','w'),indent=1)
print(f'[firm] {ver} lam=3 p99={p99:.0f} PASS={p99<=8000} tpot={b["mean_tpot_ms"]:.0f} conc={b["concurrency"]:.0f} completed={b["completed"]}')
PY
  python3 "$SOP/report-sop/scripts/log_wandb.py" turing runs/$1/summary.json "$1" "$2" mechanism >> "$L" 2>&1
}
run3(){ # $1=ver $2=env $3=commit
  rm -f runs/$1/bench_r3.json 2>/dev/null
  echo "[firm] launch $1 $(date -u +%H:%M:%S)" >> "$L"
  nohup srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 \
    bash -c "cd $CELL; $2 bash .claude/skills/evaluation-sop/scripts/eval.sh turing $1" > /tmp/turing_$1.log 2>&1 &
  for i in $(seq 1 95); do [ -f runs/$1/bench_r3.json ] && break; sleep 60; done
  sleep 25
  if [ -f runs/$1/bench_r3.json ]; then echo "[firm] $1 lam=3 landed $(date -u +%H:%M:%S)" >> "$L"; caplog "$1" "$3"; else echo "[firm] $1 NO bench_r3 $(date -u +%H:%M:%S)" >> "$L"; fi
  kill_srv; free_dram
}
maybe(){ # guard each replicate by wall-time + skip-if-done
  local now; now=$(( $(date +%s) - START ))
  if [ "$now" -ge "$DEADLINE" ]; then echo "[firm] past deadline (${now}s) -> stop" >> "$L"; return 1; fi
  [ -f runs/$1/summary.json ] && { echo "[firm] $1 already done -> skip" >> "$L"; return 0; }
  run3 "$1" "$2" "$3"
}

free_dram
maybe v18_faccel1 "$ACCEL_ENV" 8bf42d159
maybe v18_fstock1 ""          8bf42d159
maybe v18_faccel2 "$ACCEL_ENV" 8bf42d159
maybe v18_fstock2 ""          8bf42d159
maybe v18_faccel3 "$ACCEL_ENV" 8bf42d159
maybe v18_fstock3 ""          8bf42d159
echo "[firm] release node $(date -u +%H:%M:%S)" >> "$L"; scancel "$JID"
echo "[firm] DONE $(date -u +%H:%M:%S)" >> "$L"
