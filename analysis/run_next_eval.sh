#!/usr/bin/env bash
# Launch the next turing eval into the held node (job 19733, node slurm2-a3nodeset0-3)
# via srun. Passes the SGLANG_TURING_* env into the server. Serial (one at a time).
# Usage: run_next_eval.sh <version> [ADMIT] [GATE_HITS] [SIZE_TOK]
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher
NODE=slurm2-a3nodeset0-3
JID=$(cat /tmp/turing_hold_jid.txt 2>/dev/null || echo 19733)
VER="${1:?usage: run_next_eval.sh <version> [ADMIT] [GATE_HITS] [SIZE_TOK]}"
ADMIT="${2:-off}"; GATE="${3:-2}"; SIZE="${4:-4096}"
LOG="/tmp/turing_${VER}.log"
echo "[run_next_eval] version=$VER ADMIT=$ADMIT GATE_HITS=$GATE SIZE_TOK=$SIZE node=$NODE job=$JID"
nohup srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 \
  bash -c "export SGLANG_TURING_ADMIT=$ADMIT SGLANG_TURING_GATE_HITS=$GATE SGLANG_TURING_SIZE_TOK=$SIZE; \
           bash .claude/skills/evaluation-sop/scripts/eval.sh turing $VER" \
  > "$LOG" 2>&1 &
echo "[run_next_eval] launched pid $! ; log $LOG"
