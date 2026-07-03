#!/usr/bin/env bash
# launch_seq.sh — grab ONE good certified node and run a SEQUENCE of versions back-to-back
# on that single hold (maximizes value per scarce node). Each version: checkout its code into
# python/, run the FIXED evaluator with the fusion-disable fix + its extra args, and spawn a
# post_eval watcher (auto audit/log/email). Blocking-flock + disk-aware + self-healing.
#
# SEQ entries: "version|gitref|tag|extra eval args"   (gitref's python/ is checked out first)
set -uo pipefail
NAME=quill-7m3
ROOT=/home/junyanch_google_com/autoresearch
RT=$ROOT/programs/sgl/manager/.runtime
EVAL=$ROOT/programs/sgl/researcher/.claude/skills/evaluation-sop/scripts/eval.sh
WORK=$ROOT/workspace/sgl/researchers/$NAME
POST=$WORK/post_eval.sh
FUSION="--enforce-disable-flashinfer-allreduce-fusion"
V1REF=2dda8247a
MIN_GB=1850; POLL=5; BLACKLIST=""
SEQ=(
  "v1b-besteffort|$V1REF|config|--hicache-storage-prefetch-policy best_effort"
  "v2-read-priority|3a62f45c2|mechanism|"
  "v3-aux-threads|a338bff98|mechanism|"
)
held(){ for f in "$RT"/held/*; do [ -e "$f" ] && basename "$f"; done; }
skip(){ for e in $BLACKLIST; do [ "$1" = "$e" ] && return 0; done; return 1; }
freegb(){ timeout 22 srun --jobid="$1" --overlap -N1 -w "$2" bash -c 'df -BG /mnt/localssd 2>/dev/null|tail -1|awk "{gsub(/G/,\"\",\$4);print \$4}"' 2>/dev/null; }

run_seq_on(){  # $1=node $2=jid
  local node="$1" jid="$2" spec ver ref tag args
  for spec in "${SEQ[@]}"; do
    IFS='|' read -r ver ref tag args <<< "$spec"
    echo "[seq] === $ver (ref=$ref tag=$tag args='$args') on $node @ $(date +%H:%M:%S) ==="
    git -C "$WORK" checkout "$ref" -- python/ 2>&1 | tail -1 || { echo "[seq] checkout $ref FAILED, skip $ver"; continue; }
    rm -f "$WORK/runs/$ver/summary.json" 2>/dev/null
    nohup bash "$POST" "$ver" "$ref" "$tag" > "$WORK/post_${ver}.nohup" 2>&1 & disown
    srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" "$NAME" "$ver" $FUSION $args
    if [ -f "$WORK/runs/$ver/summary.json" ]; then echo "[seq] $ver OK @ $(date +%H:%M:%S)"; else echo "[seq] $ver NO SUMMARY @ $(date +%H:%M:%S)"; fi
  done
  git -C "$WORK" checkout "$V1REF" -- python/ 2>&1 | tail -1  # restore v1 code
}

while :; do
  for node in $(held); do
    skip "$node" && continue
    jid=$(cat "$RT/held/$node" 2>/dev/null) || continue
    squeue -h -j "$jid" >/dev/null 2>&1 || continue
    exec 200>"$RT/locks/$node.lock"
    if ! flock -w 8 200; then exec 200>&-; continue; fi
    gb=$(freegb "$jid" "$node"); gb=${gb:-0}
    if ! [[ "$gb" =~ ^[0-9]+$ ]] || (( gb < MIN_GB )); then flock -u 200; exec 200>&-; continue; fi
    echo "[seq] GOT $node (${gb}G) @ $(date +%H:%M:%S)"
    run_seq_on "$node" "$jid"
    flock -u 200; exec 200>&-
    echo "[seq] sequence DONE, released $node @ $(date +%H:%M:%S)"; exit 0
  done
  sleep "$POLL"
done
