#!/usr/bin/env bash
# FOLLOW-ON: complete the capstone Fig2 with a CLEAN SAME-COMMIT co-design leg (remove the earlier-commit asterisk).
# The current Fig2 is stock 4.83 (current) / accel 5.90 (current) / co-design 6.33 (EARLIER commit, flagged *). This
# runs accel+write_back on the CURRENT commit (same as stockcap1/accelcap1) so all three bars are same-commit.
# Waits for the capfirm chain to finish + release node 20243, then acquires a FRESH node and runs the co-design sweep.
# co-design = accel (mine, env) + write_back (config, --hicache-write-policy, not forbidden, last-wins over write_through).
# Full sweep (lam=3,5,7,10) → lam=10 = saturated co-design capacity for Fig2 (+ bonus co-design curve / lam=5 knee draw).
# DRAM gate>=1700 + 90s settle; bracket pkill; W&B-logs; releases node.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing
L=/tmp/turing_chain_cdcur.log
SOP=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher/.claude/skills
CELL=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher
source .venv/bin/activate 2>/dev/null; set -a; . /home/junyanch_google_com/autoresearch/.env 2>/dev/null; set +a
ACC="export SGLANG_TURING_GIANT_ACCEL=1 SGLANG_TURING_ACCEL_THETA=0.85 SGLANG_TURING_ACCEL_FACTOR=2.0;"
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
echo "[cdcur] START $(date -u +%H:%M:%S) commit=$COMMIT" > "$L"

# 1) wait for the capfirm chain to finish + its node hold to be gone (so we don't collide on the same node)
OLDJID=$(cat /tmp/turing_hold_jid.txt 2>/dev/null || echo "")
for i in $(seq 1 360); do   # up to 6h
  chain_alive=0; pgrep -f chain_capfirm >/dev/null && chain_alive=1
  sc2_done=0; [ -f runs/v23_stockcap2/summary.json ] && sc2_done=1
  hold_gone=1; [ -n "$OLDJID" ] && squeue -j "$OLDJID" -h 2>/dev/null | grep -q . && hold_gone=0
  echo "[cdcur] wait chain_alive=$chain_alive sc2_done=$sc2_done hold_gone=$hold_gone $(date -u +%H:%M:%S)" >> "$L"
  [ "$chain_alive" = 0 ] && [ "$sc2_done" = 1 ] && [ "$hold_gone" = 1 ] && break
  sleep 60
done

# 2) acquire a fresh node — PATIENT but good-neighbor (only grabs an IDLE certified node, never preempts).
#    Retry every 5 min for ~2h; co-design is optional polish, so don't hog scarce nodes indefinitely.
for a in $(seq 1 24); do
  bash analysis/acquire_node.sh 5 >> "$L" 2>&1
  NODE=$(cat /tmp/turing_node.txt 2>/dev/null); JID=$(cat /tmp/turing_hold_jid.txt 2>/dev/null)
  # confirm running AND it is a fresh hold (not the old 20243)
  if [ -n "$JID" ] && [ "$JID" != "20243" ] && squeue -j "$JID" -h -t R 2>/dev/null | grep -q .; then
    echo "[cdcur] node $NODE (jid $JID) RUNNING $(date -u +%H:%M:%S)" >> "$L"; break
  fi
  echo "[cdcur] acquire attempt $a/24: no idle certified node yet, wait 300s" >> "$L"; sleep 300
done
NODE=$(cat /tmp/turing_node.txt 2>/dev/null); JID=$(cat /tmp/turing_hold_jid.txt 2>/dev/null)
if [ -z "$JID" ] || ! squeue -j "$JID" -h -t R 2>/dev/null | grep -q .; then
  echo "[cdcur] FAILED to acquire a running node; abort (co-design polish is optional)" >> "$L"; exit 1
fi

free_dram(){
  for i in $(seq 1 40); do
    m=$(timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f "[s]glang.launch_server|[b]ench_serving" 2>/dev/null; sleep 1; awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo' 2>/dev/null | tail -1)
    echo "[cdcur] MemAvail=${m}G" >> "$L"; [ "${m:-0}" -ge 1700 ] 2>/dev/null && break; sleep 15
  done
  sleep 90
}
VER=v24_cdcur
free_dram
echo "[cdcur] launch $VER (accel+write_back, current commit $COMMIT) $(date -u +%H:%M:%S)" >> "$L"
nohup srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 \
  bash -c "cd $CELL; $ACC bash .claude/skills/evaluation-sop/scripts/eval.sh turing $VER --hicache-write-policy write_back" > /tmp/turing_$VER.log 2>&1 &
for i in $(seq 1 220); do [ -f runs/$VER/summary.json ] && break; sleep 60; done
if [ -f runs/$VER/summary.json ]; then
  python3 - "$VER" >> "$L" 2>&1 <<'PY'
import json,sys
v=sys.argv[1]; j=json.load(open(f'runs/{v}/summary.json'))
c={r['rate']:r.get('req_throughput') for r in j['curve'] if isinstance(r,dict)}
print(f"[cdcur] {v} DONE curve={c} lam10_capacity={c.get('10','?')}")
PY
  python3 "$SOP/report-sop/scripts/log_wandb.py" turing runs/$VER/summary.json "$VER" "$COMMIT" mechanism >> "$L" 2>&1
else echo "[cdcur] $VER NO summary $(date -u +%H:%M:%S)" >> "$L"; fi
timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f "[s]glang.launch_server|[b]ench_serving"' >/dev/null 2>&1 || true
echo "[cdcur] release node $(date -u +%H:%M:%S)" >> "$L"; scancel "$JID"
echo "[cdcur] DONE $(date -u +%H:%M:%S)" >> "$L"
