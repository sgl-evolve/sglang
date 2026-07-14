#!/usr/bin/env bash
# Replaces chain_flat's tail: after v6_flat completes, log it, then run v7_decfloor (Paper-4
# occupancy-feedback damping, SGLANG_TURING_DECODE_FLOOR=1) on the SAME warm held node, then
# release. v6_flat itself is a separate nohup'd srun (unaffected by swapping this finalizer).
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing
L=/tmp/turing_chain_v7.log
NODE=$(cat /tmp/turing_node.txt 2>/dev/null || echo slurm2-a3nodeset0-3)
JID=$(cat /tmp/turing_hold_jid.txt 2>/dev/null || echo 19793)
SOP=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher/.claude/skills
source .venv/bin/activate 2>/dev/null; set -a; . /home/junyanch_google_com/autoresearch/.env 2>/dev/null; set +a

echo "[v7] wait v6_flat summary $(date -u +%H:%M:%S)" > "$L"
for i in $(seq 1 300); do [ -f runs/v6_flat_sweep/summary.json ] && break; sleep 60; done
echo "[v7] log v6_flat $(date -u +%H:%M:%S)" >> "$L"
python3 "$SOP/report-sop/scripts/log_wandb.py" turing runs/v6_flat_sweep/summary.json v6_flat_sweep ce01c1c79 mechanism >> "$L" 2>&1

# free DRAM: kill v6 server, wait for the 768GB L2 pool to drain (direct probe)
for i in $(seq 1 40); do
  m=$(timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f sglang.launch_server 2>/dev/null; sleep 1; awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo' 2>/dev/null | tail -1)
  echo "[v7] MemAvail=${m}G" >> "$L"
  [ "${m:-0}" -ge 1300 ] 2>/dev/null && break
  sleep 15
done

echo "[v7] launch v7_decfloor (DECODE_FLOOR=1 theta_hi=0.90 gain=0.5) $(date -u +%H:%M:%S)" >> "$L"
nohup srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 \
  bash -c 'cd /home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher; export SGLANG_TURING_DECODE_FLOOR=1 SGLANG_TURING_THETA_HI=0.90 SGLANG_TURING_GAIN=0.5; bash .claude/skills/evaluation-sop/scripts/eval.sh turing v7_decfloor' \
  > /tmp/turing_v7_decfloor.log 2>&1 &
echo "[v7] launched pid $! $(date -u +%H:%M:%S)" >> "$L"

for i in $(seq 1 300); do [ -f runs/v7_decfloor/summary.json ] && break; sleep 60; done
echo "[v7] log v7_decfloor $(date -u +%H:%M:%S)" >> "$L"
python3 "$SOP/report-sop/scripts/log_wandb.py" turing runs/v7_decfloor/summary.json v7_decfloor 239608a1e mechanism >> "$L" 2>&1
scancel "$JID" >> "$L" 2>&1
echo "[v7] done + released node $(date -u +%H:%M:%S)" >> "$L"
