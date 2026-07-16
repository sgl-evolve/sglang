#!/usr/bin/env bash
# P7 concurrency-cap axis MAP (lam=3-only, efficient). cap64 already decisive (lam=3 p99=94.4s, catastrophic
# admission-starvation). Map the threshold: stock control + cap128/cap96/cap192 at lam=3. Expect p99 to RISE as the
# cap tightens below the natural lam=3 concurrency (74-166); a non-binding cap192 (>166) should ~= stock (sanity).
# Establishes the bounded negative: crude running-batch capping trades decode-interference for admission queue-wait.
# STOCK-based (no accel) to isolate the cap. lam=3-only (kill after bench_r3, ~55min). DRAM-race-safe, bracket pkill.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing
L=/tmp/turing_chain_cap2.log
NODE=$(cat /tmp/turing_node.txt); JID=$(cat /tmp/turing_hold_jid.txt)
SOP=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher/.claude/skills
CELL=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher
source .venv/bin/activate 2>/dev/null; set -a; . /home/junyanch_google_com/autoresearch/.env 2>/dev/null; set +a
START=$(date +%s); DEADLINE=$((5*3600))
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
echo "[cap2] START node=$NODE jid=$JID $(date -u +%H:%M:%S) commit=$COMMIT" > "$L"
free_dram(){
  for i in $(seq 1 40); do
    m=$(timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f "[s]glang.launch_server|[b]ench_serving" 2>/dev/null; sleep 1; awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo' 2>/dev/null | tail -1)
    echo "[cap2] MemAvail=${m}G" >> "$L"; [ "${m:-0}" -ge 1600 ] 2>/dev/null && break; sleep 15
  done
  sleep 60
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
d={'version':ver,'commit':commit,'note':'P7 concurrency-cap axis lam=3-only','panel':{
 'overall/goodput_reqs_at_SLO':(3.0 if p99<=8000 else 0),'overall/peak_req_s':b['request_throughput'],
 'overall/hit_rate':hit,'overall/slo_ms':8000.0,'curve/ttft_p99@3':p99,
 'curve/tpot_ms@3':b['mean_tpot_ms'],'curve/concurrency@3':b['concurrency']},
 'curve':[{'label':'sweep','rate':'3','req_throughput':str(b['request_throughput']),'out_tok_s':str(b['output_throughput']),'ttft_p50_ms':str(b['median_ttft_ms']),'ttft_p99_ms':str(p99),'e2e_p99_ms':'','hit_rate':str(hit)}]}
json.dump(d,open(f'runs/{ver}/summary.json','w'),indent=1)
print(f'[cap2] {ver} lam=3 p99={p99:.0f} PASS={p99<=8000} tpot={b["mean_tpot_ms"]:.0f} conc={b["concurrency"]:.0f} completed={b["completed"]}')
PY
  python3 "$SOP/report-sop/scripts/log_wandb.py" turing runs/$1/summary.json "$1" "$2" mechanism >> "$L" 2>&1
}
run3(){ # $1=ver $2=extra
  rm -f runs/$1/bench_r3.json 2>/dev/null
  echo "[cap2] launch $1 (extra='$2') $(date -u +%H:%M:%S)" >> "$L"
  nohup srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 \
    bash -c "cd $CELL; bash .claude/skills/evaluation-sop/scripts/eval.sh turing $1 $2" > /tmp/turing_$1.log 2>&1 &
  for i in $(seq 1 95); do [ -f runs/$1/bench_r3.json ] && break; sleep 60; done
  sleep 25
  if [ -f runs/$1/bench_r3.json ]; then echo "[cap2] $1 landed $(date -u +%H:%M:%S)" >> "$L"; caplog "$1" "$COMMIT"; else echo "[cap2] $1 NO bench_r3 $(date -u +%H:%M:%S)" >> "$L"; fi
  kill_srv
}
maybe(){ local now; now=$(( $(date +%s) - START ))
  if [ "$now" -ge "$DEADLINE" ]; then echo "[cap2] past deadline -> stop" >> "$L"; return 1; fi
  [ -f runs/$1/summary.json ] && { echo "[cap2] $1 done -> skip" >> "$L"; return 0; }
  free_dram; run3 "$1" "$2"
}
maybe v20_stock0 ""
maybe v20_cap128 "--max-running-requests 128"
maybe v20_cap96  "--max-running-requests 96"
maybe v20_cap192 "--max-running-requests 192"
echo "[cap2] release node $(date -u +%H:%M:%S)" >> "$L"; scancel "$JID"
echo "[cap2] DONE $(date -u +%H:%M:%S)" >> "$L"
