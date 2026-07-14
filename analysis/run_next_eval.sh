#!/usr/bin/env bash
# Launch a turing eval into the held node via srun, with a DRAM-wait guard (the
# 768GB L2 host pool of a just-finished server takes time to free; eval.sh aborts
# if MemAvailable < 1300G). Reads node/jid from /tmp files (set by acquire_node.sh).
# Usage: run_next_eval.sh <version> [ADMIT] [GATE_HITS] [SIZE_TOK]
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher
NODE=$(cat /tmp/turing_node.txt 2>/dev/null || echo slurm2-a3nodeset0-3)
JID=$(cat /tmp/turing_hold_jid.txt 2>/dev/null || echo 19733)
VER="${1:?usage: run_next_eval.sh <version> [ADMIT] [GATE_HITS] [SIZE_TOK]}"
ADMIT="${2:-off}"; GATE="${3:-2}"; SIZE="${4:-4096}"
LOG="/tmp/turing_${VER}.log"

# DRAM-wait: ensure the node has >=1300G free before launching (avoid DRAM_TOO_LOW race)
echo "[run_next_eval] $VER on $NODE (job $JID): waiting for DRAM>=1300G ..."
for i in $(seq 1 30); do
  m=$(timeout 40 srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'pkill -9 -f sglang.launch_server 2>/dev/null; awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo' 2>/dev/null | tail -1)
  echo "  MemAvail=${m}G"
  [ "${m:-0}" -ge 1300 ] 2>/dev/null && break
  sleep 20
done

echo "[run_next_eval] version=$VER ADMIT=$ADMIT GATE_HITS=$GATE SIZE_TOK=$SIZE node=$NODE job=$JID"
nohup srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 \
  bash -c "export SGLANG_TURING_ADMIT=$ADMIT SGLANG_TURING_GATE_HITS=$GATE SGLANG_TURING_SIZE_TOK=$SIZE; \
           bash .claude/skills/evaluation-sop/scripts/eval.sh turing $VER" \
  > "$LOG" 2>&1 &
echo "[run_next_eval] launched pid $! ; log $LOG"
