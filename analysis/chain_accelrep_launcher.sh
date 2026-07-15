#!/usr/bin/env bash
# Wait for the f=1.5 dose chain (chain_factor) to finish + free node 19940, then acquire a fresh
# certified node and run the accel(f=2) frontier replicate (chain_accelrep). Serial => no
# concurrent turing evals. Safe to background.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/turing
L=/tmp/turing_accelrep_launcher.log
echo "[arl] START $(date -u +%H:%M:%S) waiting for f=1.5 chain (chain_factor) to finish" > "$L"
for i in $(seq 1 300); do   # up to 5h
  # chain_factor.sh prints "[factor] DONE" at the end and scancels node 19940
  if grep -q "\[factor\] DONE" /tmp/turing_chain_factor.log 2>/dev/null; then
    echo "[arl] chain_factor DONE $(date -u +%H:%M:%S)" >> "$L"; break
  fi
  # also proceed if v15 summary exists (sweep done) even if scancel lagging
  [ -f runs/v15_accelf15/summary.json ] && { echo "[arl] v15 summary present $(date -u +%H:%M:%S)" >> "$L"; break; }
  sleep 60
done
sleep 30
echo "[arl] acquiring fresh node $(date -u +%H:%M:%S)" >> "$L"
for a in $(seq 1 20); do
  bash analysis/acquire_node.sh 6 >> "$L" 2>&1
  jid=$(cat /tmp/turing_hold_jid.txt 2>/dev/null)
  if [ -n "$jid" ] && [ -n "$(squeue -j "$jid" -h -t R 2>/dev/null)" ]; then
    echo "[arl] acquired node=$(cat /tmp/turing_node.txt) jid=$jid $(date -u +%H:%M:%S)" >> "$L"; break
  fi
  echo "[arl] no running node yet (attempt $a); wait 300s" >> "$L"; sleep 300
done
echo "[arl] launching chain_accelrep $(date -u +%H:%M:%S)" >> "$L"
bash analysis/chain_accelrep.sh >> "$L" 2>&1
echo "[arl] DONE $(date -u +%H:%M:%S)" >> "$L"
