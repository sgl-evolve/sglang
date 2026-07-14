#!/usr/bin/env bash
# Chain after v5_size: log v5_size, then run v6_flat_sweep (flat admission, FULL sweep)
# on the SAME held node = the LOW-HIT capacity anchor for the C=K/(1-h) law (Paper 2)
# + crater replicate at higher rates (Paper 1). Release node only when v6 done.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing
L=/tmp/turing_chain_flat.log
NODE=$(cat /tmp/turing_node.txt 2>/dev/null || echo slurm2-a3nodeset0-3)
JID=$(cat /tmp/turing_hold_jid.txt 2>/dev/null || echo 19793)
SOP=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher/.claude/skills
echo "[chain_flat] wait v5_size summary $(date -u +%H:%M:%S)" > "$L"
for i in $(seq 1 300); do [ -f runs/v5_size/summary.json ] && break; sleep 60; done
source .venv/bin/activate 2>/dev/null; set -a; . /home/junyanch_google_com/autoresearch/.env 2>/dev/null; set +a
echo "[chain_flat] log v5_size $(date -u +%H:%M:%S)" >> "$L"
python3 "$SOP/report-sop/scripts/log_wandb.py" turing runs/v5_size/summary.json v5_size ce01c1c79 mechanism >> "$L" 2>&1

# free DRAM on the held node (kill lingering server; wait for L2 pool to drain)
for i in $(seq 1 40); do
  m=$(timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f sglang.launch_server 2>/dev/null; sleep 1; awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo' 2>/dev/null | tail -1)
  echo "[chain_flat] MemAvail=${m}G" >> "$L"
  [ "${m:-0}" -ge 1300 ] 2>/dev/null && break
  sleep 15
done

echo "[chain_flat] launch v6_flat_sweep (ADMIT=flat, full sweep) $(date -u +%H:%M:%S)" >> "$L"
nohup srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 \
  bash -c 'cd /home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher; export SGLANG_TURING_ADMIT=flat SGLANG_TURING_GATE_HITS=2; bash .claude/skills/evaluation-sop/scripts/eval.sh turing v6_flat_sweep' \
  > /tmp/turing_v6_flat.log 2>&1 &
echo "[chain_flat] v6 launched pid $! $(date -u +%H:%M:%S)" >> "$L"

# wait for v6 to finish, log it, release node
for i in $(seq 1 300); do [ -f runs/v6_flat_sweep/summary.json ] && break; sleep 60; done
echo "[chain_flat] log v6_flat_sweep $(date -u +%H:%M:%S)" >> "$L"
python3 "$SOP/report-sop/scripts/log_wandb.py" turing runs/v6_flat_sweep/summary.json v6_flat_sweep ce01c1c79 mechanism >> "$L" 2>&1
scancel "$JID" >> "$L" 2>&1
echo "[chain_flat] done + released node $(date -u +%H:%M:%S)" >> "$L"
