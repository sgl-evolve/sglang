#!/usr/bin/env bash
# When v4_lpm finishes: log it to W&B, free DRAM, launch v5_size (size-conditioned
# admission, SIZE_TOK=16384 = gate only giant docs, admit small/medium eagerly).
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing
L=/tmp/turing_chain_size.log
NODE=$(cat /tmp/turing_node.txt 2>/dev/null || echo slurm2-a3nodeset0-3)
JID=$(cat /tmp/turing_hold_jid.txt 2>/dev/null || echo 19793)
echo "[chain] wait v4_lpm summary $(date -u +%H:%M:%S)" > "$L"
for i in $(seq 1 300); do [ -f runs/v4_lpm/summary.json ] && break; sleep 60; done
echo "[chain] v4_lpm done; logging $(date -u +%H:%M:%S)" >> "$L"
source .venv/bin/activate 2>/dev/null; set -a; . /home/junyanch_google_com/autoresearch/.env 2>/dev/null; set +a
python3 /home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher/.claude/skills/report-sop/scripts/log_wandb.py \
  turing runs/v4_lpm/summary.json v4_lpm ce01c1c79 config >> "$L" 2>&1
# free DRAM (direct probe; kill any lingering server)
for i in $(seq 1 40); do
  m=$(timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f sglang.launch_server 2>/dev/null; sleep 1; awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo' 2>/dev/null | tail -1)
  echo "[chain] MemAvail=${m}G" >> "$L"
  [ "${m:-0}" -ge 1300 ] 2>/dev/null && break
  sleep 15
done
echo "[chain] launch v5_size (SIZE_TOK=16384) $(date -u +%H:%M:%S)" >> "$L"
nohup srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 \
  bash -c 'cd /home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher; export SGLANG_TURING_ADMIT=size SGLANG_TURING_GATE_HITS=2 SGLANG_TURING_SIZE_TOK=16384; bash .claude/skills/evaluation-sop/scripts/eval.sh turing v5_size' \
  > /tmp/turing_v5_size.log 2>&1 &
echo "[chain] v5_size launched pid $! $(date -u +%H:%M:%S)" >> "$L"
