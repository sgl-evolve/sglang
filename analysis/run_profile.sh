#!/usr/bin/env bash
# Launch the Paper 7 saturation profiler on the held profiler node (job in /tmp/turing_prof_jid.txt),
# then release the node. Runs _prof_inner.sh on the node via srun --overlap. Detach-friendly.
set -uo pipefail
WORK=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing
NODE=$(cat /tmp/turing_prof_node.txt); JID=$(cat /tmp/turing_prof_jid.txt)
L=/tmp/turing_run_profile.log
echo "[run_profile] START node=$NODE jid=$JID $(date -u +%H:%M:%S)" > "$L"
srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 bash "$WORK/analysis/_prof_inner.sh" >> "$L" 2>&1
echo "[run_profile] inner done; release node $(date -u +%H:%M:%S)" >> "$L"
scancel "$JID"
echo "[run_profile] DONE $(date -u +%H:%M:%S)" >> "$L"
