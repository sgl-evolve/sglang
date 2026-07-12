#!/usr/bin/env bash
# Exit when eval job $1's summary.json appears (success) or the job finishes/vanishes. 
JID="$1"; SUMM="$2"; CAP="${3:-72000}"; t=0; ran=0
while [ $t -lt $CAP ]; do
  if [ -f "$SUMM" ]; then echo "SUMMARY READY: $SUMM (t=${t}s)"; exit 0; fi
  st=$(squeue -j "$JID" -h -o "%t" 2>/dev/null)
  [ "$st" = "R" ] && ran=1
  if [ -z "$st" ]; then
    sleep 20
    if [ -f "$SUMM" ]; then echo "SUMMARY READY (post-exit): $SUMM (t=${t}s)"; exit 0; fi
    echo "JOB $JID left queue at t=${t}s; ran=$ran; no summary yet (check for failure)"; exit 0
  fi
  sleep 120; t=$((t+120))
done
echo "cap ${CAP}s reached; job $JID still around"
