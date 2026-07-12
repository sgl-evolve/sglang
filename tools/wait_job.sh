#!/usr/bin/env bash
# Wait until slurm job $1 starts running (PD->R) or leaves the queue. Exits promptly then.
JID="$1"; CAP="${2:-54000}"; t=0
while [ $t -lt $CAP ]; do
  st=$(squeue -j "$JID" -h -o "%t" 2>/dev/null)
  if [ -z "$st" ]; then echo "JOB $JID no longer in queue (finished or cancelled) at t=${t}s"; exit 0; fi
  if [ "$st" = "R" ]; then echo "JOB $JID is RUNNING at t=${t}s"; exit 0; fi
  sleep 60; t=$((t+60))
done
echo "wait cap ${CAP}s reached; job $JID still $st"
