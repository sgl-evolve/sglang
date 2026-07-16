#!/usr/bin/env bash
# Firm the TWO-REGIME claim's weakest leg: co-design (accel+write_back, C=6.33, margin C-lam=1.33 at lam=5) lam=5
# reliability is currently n=1 (v17 FAILED 15.6s). Run n=3 more independent cold-start draws to characterize the
# margin-1.33 reliability: if >=2/3 pass -> co-design meaningfully improves lam=5 goodput (upgrade); if 0-1/3 ->
# confirms knee coin-flip even at margin 1.33 (firms two-regime). Each replicate ALSO firms co-design lam=3 (also n=1).
# accel (mine, env) + write_back (config, --hicache-write-policy, not forbidden, last-wins over stock write_through).
# Sweep up to lam=5 then kill (skip lam7/10, already characterized 5.86/6.33). DRAM gate>=1700 + 90s settle (safer than
# the 1600 that raced on full-pool back-to-back). W&B-logs; releases node. Requires /tmp/turing_node.txt + hold_jid.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing
L=/tmp/turing_chain_cd5.log
NODE=$(cat /tmp/turing_node.txt); JID=$(cat /tmp/turing_hold_jid.txt)
SOP=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher/.claude/skills
CELL=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher
source .venv/bin/activate 2>/dev/null; set -a; . /home/junyanch_google_com/autoresearch/.env 2>/dev/null; set +a
START=$(date +%s); DEADLINE=$((6*3600 + 1800))
ACC="export SGLANG_TURING_GIANT_ACCEL=1 SGLANG_TURING_ACCEL_THETA=0.85 SGLANG_TURING_ACCEL_FACTOR=2.0;"
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
echo "[cd5] START node=$NODE jid=$JID $(date -u +%H:%M:%S) commit=$COMMIT" > "$L"
free_dram(){
  for i in $(seq 1 40); do
    m=$(timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f "[s]glang.launch_server|[b]ench_serving" 2>/dev/null; sleep 1; awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo' 2>/dev/null | tail -1)
    echo "[cd5] MemAvail=${m}G" >> "$L"; [ "${m:-0}" -ge 1700 ] 2>/dev/null && break; sleep 15
  done
  sleep 90
}
kill_srv(){ timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f "[s]glang.launch_server|[b]ench_serving" 2>/dev/null; echo k' >/dev/null 2>&1 || true; }
caplog(){ # $1=ver : build summary from bench_r3 + bench_r5, log W&B
  python3 - "$1" "$COMMIT" >> "$L" 2>&1 <<'PY'
import json,sys
ver,commit=sys.argv[1],sys.argv[2]
b3=json.load(open(f'runs/{ver}/bench_r3.json')); b5=json.load(open(f'runs/{ver}/bench_r5.json'))
p3,p5=b3['p99_ttft_ms'],b5['p99_ttft_ms']
gp=5.0 if p5<=8000 else (3.0 if p3<=8000 else 0)
d={'version':ver,'commit':commit,'note':'co-design accel+write_back lam<=5 firming','panel':{
 'overall/goodput_reqs_at_SLO':gp,'overall/peak_req_s':b5['request_throughput'],'overall/slo_ms':8000.0,
 'curve/ttft_p99@3':p3,'curve/ttft_p99@5':p5,'curve/tpot_ms@5':b5['mean_tpot_ms'],'curve/concurrency@5':b5['concurrency']},
 'curve':[{'label':'sweep','rate':'3','ttft_p99_ms':str(p3),'req_throughput':str(b3['request_throughput'])},
          {'label':'sweep','rate':'5','ttft_p99_ms':str(p5),'req_throughput':str(b5['request_throughput'])}]}
json.dump(d,open(f'runs/{ver}/summary.json','w'),indent=1)
print(f'[cd5] {ver} lam3 p99={p3:.0f} PASS={p3<=8000} | lam5 p99={p5:.0f} PASS={p5<=8000} | goodput@SLO={gp}')
PY
  python3 "$SOP/report-sop/scripts/log_wandb.py" turing runs/$1/summary.json "$1" "$COMMIT" mechanism >> "$L" 2>&1
}
run_cd(){ # $1=ver
  rm -f runs/$1/bench_r3.json runs/$1/bench_r5.json 2>/dev/null
  echo "[cd5] launch $1 (accel+write_back) $(date -u +%H:%M:%S)" >> "$L"
  nohup srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 \
    bash -c "cd $CELL; $ACC bash .claude/skills/evaluation-sop/scripts/eval.sh turing $1 --hicache-write-policy write_back" > /tmp/turing_$1.log 2>&1 &
  for i in $(seq 1 150); do [ -f runs/$1/bench_r5.json ] && break; sleep 60; done
  sleep 25
  if [ -f runs/$1/bench_r5.json ]; then echo "[cd5] $1 lam5 landed $(date -u +%H:%M:%S)" >> "$L"; caplog "$1"; else echo "[cd5] $1 NO bench_r5 $(date -u +%H:%M:%S)" >> "$L"; fi
  kill_srv
}
maybe(){ local now; now=$(( $(date +%s) - START ))
  if [ "$now" -ge "$DEADLINE" ]; then echo "[cd5] past deadline -> stop" >> "$L"; return 1; fi
  [ -f runs/$1/summary.json ] && { echo "[cd5] $1 done -> skip" >> "$L"; return 0; }
  free_dram; run_cd "$1"
}
maybe v21_cdwb1
maybe v21_cdwb2
maybe v21_cdwb3
echo "[cd5] release node $(date -u +%H:%M:%S)" >> "$L"; scancel "$JID"
echo "[cd5] DONE $(date -u +%H:%M:%S)" >> "$L"
