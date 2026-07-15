#!/usr/bin/env bash
# Autonomous serial launcher for the P6 dose-response (chain_factor.sh).
# Waits for the in-flight chain_frontier (v13 on node 19916) to FINISH + release its node,
# then acquires a FRESH certified node and runs chain_factor (f=3, f=1.5). Serial => no
# concurrent turing evals (avoids flashinfer JIT race). Safe to background.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing
L=/tmp/turing_factor_launcher.log
CELL=/home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher
echo "[launcher] START $(date -u +%H:%M:%S) waiting for chain_frontier to finish" > "$L"
# 1) wait until frontier chain is done (v13 summary present OR its log says DONE) AND node 19916 released
for i in $(seq 1 240); do   # up to 4h
  done_flag=0
  grep -q "\[frontier\] DONE" /tmp/turing_chain_frontier.log 2>/dev/null && done_flag=1
  # also treat node-19916-gone as released
  if ! squeue -j 19916 -h -t R >/dev/null 2>&1 || [ -z "$(squeue -j 19916 -h -t R 2>/dev/null)" ]; then
    [ "$done_flag" = "1" ] && { echo "[launcher] frontier DONE + node released $(date -u +%H:%M:%S)" >> "$L"; break; }
  fi
  sleep 60
done
# 2) acquire a fresh node (writes /tmp/turing_node.txt + turing_hold_jid.txt)
echo "[launcher] acquiring fresh node $(date -u +%H:%M:%S)" >> "$L"
for a in $(seq 1 20); do
  bash analysis/acquire_node.sh 9 >> "$L" 2>&1
  jid=$(cat /tmp/turing_hold_jid.txt 2>/dev/null)
  if [ -n "$jid" ] && [ -n "$(squeue -j "$jid" -h -t R 2>/dev/null)" ]; then
    echo "[launcher] acquired node=$(cat /tmp/turing_node.txt) jid=$jid $(date -u +%H:%M:%S)" >> "$L"; break
  fi
  echo "[launcher] no running node yet (attempt $a); wait 300s" >> "$L"; sleep 300
done
# 3) run the dose-response chain
echo "[launcher] launching chain_factor $(date -u +%H:%M:%S)" >> "$L"
bash analysis/chain_factor.sh >> "$L" 2>&1
echo "[launcher] DONE $(date -u +%H:%M:%S)" >> "$L"
