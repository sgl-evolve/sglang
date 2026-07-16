#!/usr/bin/env bash
# Resolve the non-monotonicity question: firm ACCEL-ALONE (write_through, C=5.65, margin 0.65) lam=5 to n=5 (v13 PASS +
# v16 FAIL + 3 fresh) for a clean comparison vs co-design (accel+write_back, C=6.33, margin 1.33, firmed 0/4). Hypothesis:
# does write_back HURT the lam=5 knee (accel-alone passes MORE despite LOWER margin)? If accel-alone >> co-design => novel
# counterintuitive finding (write_back degrades metastable knee). If ~equal (both ~0) => noise, both unreliable (clean).
# accel-alone = GIANT_ACCEL=1 + stock write_through (NO --hicache-write-policy). Sweep to lam=5 then kill. DRAM gate>=1700
# +90s settle. W&B-logs; releases node. Requires /tmp/turing_node.txt + hold_jid.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing
L=/tmp/turing_chain_acc5.log
NODE=$(cat /tmp/turing_node.txt); JID=$(cat /tmp/turing_hold_jid.txt)
SOP=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher/.claude/skills
CELL=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher
source .venv/bin/activate 2>/dev/null; set -a; . /home/junyanch_google_com/autoresearch/.env 2>/dev/null; set +a
START=$(date +%s); DEADLINE=$((6*3600 + 1800))
ACC="export SGLANG_TURING_GIANT_ACCEL=1 SGLANG_TURING_ACCEL_THETA=0.85 SGLANG_TURING_ACCEL_FACTOR=2.0;"
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
echo "[acc5] START node=$NODE jid=$JID $(date -u +%H:%M:%S) commit=$COMMIT" > "$L"
free_dram(){
  for i in $(seq 1 40); do
    m=$(timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f "[s]glang.launch_server|[b]ench_serving" 2>/dev/null; sleep 1; awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo' 2>/dev/null | tail -1)
    echo "[acc5] MemAvail=${m}G" >> "$L"; [ "${m:-0}" -ge 1700 ] 2>/dev/null && break; sleep 15
  done
  sleep 90
}
kill_srv(){ timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f "[s]glang.launch_server|[b]ench_serving" 2>/dev/null; echo k' >/dev/null 2>&1 || true; }
caplog(){
  python3 - "$1" "$COMMIT" >> "$L" 2>&1 <<'PY'
import json,sys
ver,commit=sys.argv[1],sys.argv[2]
b3=json.load(open(f'runs/{ver}/bench_r3.json')); b5=json.load(open(f'runs/{ver}/bench_r5.json'))
p3,p5=b3['p99_ttft_ms'],b5['p99_ttft_ms']
gp=5.0 if p5<=8000 else (3.0 if p3<=8000 else 0)
d={'version':ver,'commit':commit,'note':'accel-alone lam<=5 firming (non-monotonicity resolve)','panel':{
 'overall/goodput_reqs_at_SLO':gp,'overall/peak_req_s':b5['request_throughput'],'overall/slo_ms':8000.0,
 'curve/ttft_p99@3':p3,'curve/ttft_p99@5':p5,'curve/tpot_ms@5':b5['mean_tpot_ms'],'curve/concurrency@5':b5['concurrency']},
 'curve':[{'label':'sweep','rate':'3','ttft_p99_ms':str(p3),'req_throughput':str(b3['request_throughput'])},
          {'label':'sweep','rate':'5','ttft_p99_ms':str(p5),'req_throughput':str(b5['request_throughput'])}]}
json.dump(d,open(f'runs/{ver}/summary.json','w'),indent=1)
print(f'[acc5] {ver} lam3 p99={p3:.0f} PASS={p3<=8000} | lam5 p99={p5:.0f} PASS={p5<=8000} | goodput={gp}')
PY
  python3 "$SOP/report-sop/scripts/log_wandb.py" turing runs/$1/summary.json "$1" "$COMMIT" mechanism >> "$L" 2>&1
}
run_acc(){ # $1=ver
  rm -f runs/$1/bench_r3.json runs/$1/bench_r5.json 2>/dev/null
  echo "[acc5] launch $1 (accel-alone write_through) $(date -u +%H:%M:%S)" >> "$L"
  nohup srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 \
    bash -c "cd $CELL; $ACC bash .claude/skills/evaluation-sop/scripts/eval.sh turing $1" > /tmp/turing_$1.log 2>&1 &
  for i in $(seq 1 150); do [ -f runs/$1/bench_r5.json ] && break; sleep 60; done
  sleep 25
  if [ -f runs/$1/bench_r5.json ]; then echo "[acc5] $1 lam5 landed $(date -u +%H:%M:%S)" >> "$L"; caplog "$1"; else echo "[acc5] $1 NO bench_r5 $(date -u +%H:%M:%S)" >> "$L"; fi
  kill_srv
}
maybe(){ local now; now=$(( $(date +%s) - START ))
  if [ "$now" -ge "$DEADLINE" ]; then echo "[acc5] past deadline -> stop" >> "$L"; return 1; fi
  [ -f runs/$1/summary.json ] && { echo "[acc5] $1 done -> skip" >> "$L"; return 0; }
  free_dram; run_acc "$1"
}
maybe v22_acc5_1
maybe v22_acc5_2
maybe v22_acc5_3
echo "[acc5] release node $(date -u +%H:%M:%S)" >> "$L"; scancel "$JID"
echo "[acc5] DONE $(date -u +%H:%M:%S)" >> "$L"
