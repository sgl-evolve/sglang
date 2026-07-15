#!/usr/bin/env bash
# CO-DESIGN test (motivated by v16: accel lam=5 is coin-flip because margin C-lam=0.65 is small).
# Stack the two orthogonal capacity levers of P2's C=K/(1-h): accel (K-lever, mine) + write_back
# (h-lever: hit 0.65->0.74 => C via 1/(1-h); known config, output-lossless per P1). Combined C should
# be higher => larger lam=5 margin => potentially RELIABLE goodput@SLO=5. On-contract: write-policy is
# NOT forbidden and eval.sh appends EXTRA launch args (write_back overrides write_through, last-wins).
# v17_accel_wb = GIANT_ACCEL=1 + --hicache-write-policy write_back full sweep. W&B-logs; releases node.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing
L=/tmp/turing_chain_accelwb.log
NODE=$(cat /tmp/turing_node.txt); JID=$(cat /tmp/turing_hold_jid.txt)
SOP=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher/.claude/skills
CELL=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher
source .venv/bin/activate 2>/dev/null; set -a; . /home/junyanch_google_com/autoresearch/.env 2>/dev/null; set +a
echo "[accelwb] START node=$NODE jid=$JID $(date -u +%H:%M:%S)" > "$L"
for i in $(seq 1 30); do
  m=$(timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f "[s]glang.launch_server|[b]ench_serving" 2>/dev/null; sleep 1; awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo' 2>/dev/null | tail -1)
  echo "[accelwb] MemAvail=${m}G" >> "$L"; [ "${m:-0}" -ge 1300 ] 2>/dev/null && break; sleep 15
done
VER=v17_accel_wb
echo "[accelwb] launch $VER (GIANT_ACCEL=1 + write_back full sweep) $(date -u +%H:%M:%S)" >> "$L"
nohup srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 \
  bash -c "cd $CELL; export SGLANG_TURING_GIANT_ACCEL=1 SGLANG_TURING_ACCEL_THETA=0.85 SGLANG_TURING_ACCEL_FACTOR=2.0; bash .claude/skills/evaluation-sop/scripts/eval.sh turing $VER --hicache-write-policy write_back" \
  > /tmp/turing_$VER.log 2>&1 &
for i in $(seq 1 260); do [ -f runs/$VER/summary.json ] && break; sleep 60; done
if [ -f runs/$VER/summary.json ]; then
  echo "[accelwb] $VER done; log W&B $(date -u +%H:%M:%S)" >> "$L"
  python3 "$SOP/report-sop/scripts/log_wandb.py" turing runs/$VER/summary.json "$VER" 8bf42d159 mechanism >> "$L" 2>&1
else echo "[accelwb] $VER NO summary $(date -u +%H:%M:%S)" >> "$L"; fi
echo "[accelwb] release node $(date -u +%H:%M:%S)" >> "$L"; scancel "$JID"
echo "[accelwb] DONE $(date -u +%H:%M:%S)" >> "$L"
