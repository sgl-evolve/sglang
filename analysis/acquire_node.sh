#!/usr/bin/env bash
# Acquire (hold) a free CERTIFIED node for a back-to-back same-node eval session.
# Writes NODE + JID to /tmp/turing_node.txt and /tmp/turing_hold_jid.txt.
# Usage: acquire_node.sh [hours]   (default 9h — fits ~2-3 full sweeps)
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/programs/sgl/v0.31/research/researcher
HRS="${1:-9}"
mapfile -t CERT < <(bash .claude/skills/submit-gpu-job/scripts/certified-nodes.sh 2>/dev/null)
echo "certified: ${CERT[*]}"
for NODE in "${CERT[@]}"; do
  st=$(sinfo -n "$NODE" -h -o "%t" 2>/dev/null | head -1)
  echo "  $NODE state=$st"
  if [[ "$st" == "idle" ]]; then
    jid=$(sbatch --parsable -p a3 -N1 --exclusive --gres=gpu:8 -w "$NODE" -t "${HRS}:00:00" \
          -J hold-turing --wrap "sleep infinity" 2>/dev/null)
    [[ -n "$jid" ]] || { echo "  sbatch failed on $NODE"; continue; }
    sleep 4
    if squeue -j "$jid" -h -t R >/dev/null 2>&1 && [[ -n "$(squeue -j "$jid" -h -t R 2>/dev/null)" ]]; then
      echo "$NODE" > /tmp/turing_node.txt; echo "$jid" > /tmp/turing_hold_jid.txt
      echo "ACQUIRED $NODE (job $jid, ${HRS}h)"; exit 0
    fi
    echo "  $jid not running yet; leaving queued, trying next";
    echo "$NODE" > /tmp/turing_node.txt; echo "$jid" > /tmp/turing_hold_jid.txt
    echo "HELD (queued) $NODE (job $jid)"; exit 0
  fi
done
echo "No idle certified node; falling back to eval-on-pool (manager held pool) at eval time." >&2
exit 1
