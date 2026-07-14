#!/usr/bin/env bash
# When v3_wb finishes: log it to W&B, free DRAM, launch write_back replicate v3b_wb.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing
NODE=$(cat /tmp/turing_node.txt 2>/dev/null || echo slurm2-a3nodeset0-3)
JID=$(cat /tmp/turing_hold_jid.txt 2>/dev/null || echo 19787)
L=/tmp/turing_chain_v3b.log
echo "[chain] waiting for v3_wb summary $(date -u +%H:%M:%S)" > "$L"
for i in $(seq 1 260); do [ -f runs/v3_wb/summary.json ] && break; sleep 60; done
echo "[chain] v3_wb done $(date -u +%H:%M:%S); logging" >> "$L"
source .venv/bin/activate 2>/dev/null; set -a; . /home/junyanch_google_com/autoresearch/.env 2>/dev/null; set +a
python3 /home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher/.claude/skills/report-sop/scripts/log_wandb.py \
  turing runs/v3_wb/summary.json v3_wb ce01c1c79 config >> "$L" 2>&1
# free DRAM: kill server, wait for MemAvailable>=1300G
echo "[chain] freeing DRAM" >> "$L"
for i in $(seq 1 30); do
  m=$(timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f sglang.launch_server 2>/dev/null; sleep 1; awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo' 2>/dev/null | tail -1)
  echo "[chain] MemAvail=${m}G" >> "$L"
  [ "${m:-0}" -ge 1300 ] 2>/dev/null && break
  sleep 20
done
echo "[chain] launching v3b_wb (write_back replicate) $(date -u +%H:%M:%S)" >> "$L"
nohup srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 \
  bash -c 'cd /home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher; export SGLANG_TURING_ADMIT=writeback; bash .claude/skills/evaluation-sop/scripts/eval.sh turing v3b_wb' \
  > /tmp/turing_v3b_wb.log 2>&1 &
echo "[chain] v3b_wb launched pid $! $(date -u +%H:%M:%S)" >> "$L"
