#!/usr/bin/env bash
# run_batch2.sh — autonomous batch runner covering BOTH recovery modes:
#   (A) manager HELD pool node: srun --overlap into the existing hold.
#   (B) idle CERTIFIED node NOT in the held pool: self-hold (sbatch sleep infinity, -t 3h),
#       run the batch via srun, then ALWAYS release (scancel).
# Blacklists nodes that produce nothing (NCCL fail / disk-full / crash). post_eval watchers are
# spawned ONCE up front (idempotent capture -> no dup logs/emails). Pass SEQSPEC (newline: ver|ref|tag|args).
set -uo pipefail
NAME=quill-7m3
ROOT=/home/junyanch_google_com/autoresearch
RT=$ROOT/programs/sgl/manager/.runtime
SK=$ROOT/programs/sgl/researcher/.claude/skills
EVAL=$SK/evaluation-sop/scripts/eval.sh
CERT=$SK/submit-gpu-job/scripts/certified-nodes.sh
WORK=$ROOT/workspace/sgl/researchers/$NAME
POST="${POST:-$WORK/post_eval.sh}"
FUSION="--enforce-disable-flashinfer-allreduce-fusion"
V1REF=2dda8247a
MIN_GB=1850; POLL=10; FLOCK_W="${FLOCK_W:-200}"; BLACKLIST="${BLACKLIST:-slurm2-a3nodeset0-2 slurm2-a3nodeset0-0}"
SEQSPEC="${SEQSPEC:?set SEQSPEC}"

held(){ for f in "$RT"/held/*; do [ -e "$f" ] && basename "$f"; done; }
certified(){ bash "$CERT" 2>/dev/null; }
skip(){ for e in $BLACKLIST; do [ "$1" = "$e" ] && return 0; done; return 1; }
freegb(){ timeout 25 srun --jobid="$1" --overlap -N1 -w "$2" bash -c 'df -BG /mnt/localssd 2>/dev/null|tail -1|awk "{gsub(/G/,\"\",\$4);print \$4}"' 2>/dev/null; }

remaining="$SEQSPEC"; ok_any=0; newrem=""

# run all still-pending versions on (jid,node) via srun --overlap. sets globals ok_any, newrem.
run_on(){
  local jid="$1" node="$2"; newrem=""; ok_any=0
  local spec ver ref tag args
  while IFS= read -r spec; do
    [ -z "$spec" ] && continue
    IFS='|' read -r ver ref tag args <<< "$spec"
    [ -f "$WORK/runs/$ver/summary.json" ] && continue
    echo "[batch2] === $ver ref=$ref on $node @ $(date +%H:%M:%S) ==="
    git -C "$WORK" checkout "$ref" -- python/ 2>&1 | tail -1 || true
    srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" "$NAME" "$ver" $FUSION $args < /dev/null
    if [ -f "$WORK/runs/$ver/summary.json" ]; then echo "[batch2] $ver OK @ $(date +%H:%M:%S)"; ok_any=1
    else echo "[batch2] $ver NO SUMMARY @ $(date +%H:%M:%S)"; newrem+="$spec"$'\n'; fi
  done <<< "$remaining"
  git -C "$WORK" checkout "$V1REF" -- python/ 2>&1 | tail -1
}

done_check(){ [ -z "${remaining//[$'\n\t ']/}" ] && { echo "[batch2] ALL DONE @ $(date +%H:%M:%S)"; exit 0; }; }

# spawn one post_eval watcher per version (idempotent)
while IFS= read -r spec; do
  [ -z "$spec" ] && continue
  IFS='|' read -r ver ref tag args <<< "$spec"
  rm -f "$WORK/runs/$ver/summary.json" 2>/dev/null
  nohup bash "$POST" "$ver" "$ref" "$tag" > "$WORK/post_${ver}.nohup" 2>&1 & disown
done <<< "$SEQSPEC"

echo "[batch2] START @ $(date +%H:%M:%S)  blacklist='$BLACKLIST'"
while :; do
  # --- mode A: manager HELD pool ---
  for node in $(held); do
    skip "$node" && continue
    jid=$(cat "$RT/held/$node" 2>/dev/null) || continue
    squeue -h -j "$jid" >/dev/null 2>&1 || continue
    exec 200>"$RT/locks/$node.lock"
    flock -w "$FLOCK_W" 200 || { exec 200>&-; continue; }
    gb=$(freegb "$jid" "$node"); gb=${gb:-0}
    if ! [[ "$gb" =~ ^[0-9]+$ ]] || (( gb < MIN_GB )); then flock -u 200; exec 200>&-; continue; fi
    echo "[batch2] HELD GOT $node (${gb}G) @ $(date +%H:%M:%S)"
    run_on "$jid" "$node"
    flock -u 200; exec 200>&-
    remaining="$newrem"; done_check
    [ "$ok_any" = 0 ] && { echo "[batch2] $node nothing -> blacklist"; BLACKLIST="$BLACKLIST $node"; }
  done
  # --- mode B: idle CERTIFIED node not held/blacklisted -> self-hold, run, release ---
  for node in $(certified); do
    skip "$node" && continue
    [ -e "$RT/held/$node" ] && continue                          # mode A owns held nodes
    [ "$(sinfo -n "$node" -h -o '%T' 2>/dev/null)" = "idle" ] || continue
    echo "[batch2] self-holding idle certified $node @ $(date +%H:%M:%S)"
    hjid=$(sbatch --parsable -p a3 -N1 --exclusive --gres=gpu:8 -w "$node" -t 3:00:00 -J hold-quill --wrap "sleep infinity" 2>/dev/null) || continue
    ok=0; for i in $(seq 1 40); do [ "$(squeue -h -j "$hjid" -o '%T' 2>/dev/null)" = "RUNNING" ] && { ok=1; break; }; sleep 5; done
    [ "$ok" = 1 ] || { scancel "$hjid" 2>/dev/null; continue; }
    gb=$(freegb "$hjid" "$node"); gb=${gb:-0}
    if ! [[ "$gb" =~ ^[0-9]+$ ]] || (( gb < MIN_GB )); then echo "[batch2] $node disk ${gb}G<$MIN_GB -> blacklist"; BLACKLIST="$BLACKLIST $node"; scancel "$hjid" 2>/dev/null; continue; fi
    echo "[batch2] SELF-HELD $node (${gb}G) @ $(date +%H:%M:%S)"
    run_on "$hjid" "$node"
    scancel "$hjid" 2>/dev/null
    remaining="$newrem"; done_check
    [ "$ok_any" = 0 ] && { echo "[batch2] $node nothing -> blacklist"; BLACKLIST="$BLACKLIST $node"; }
  done
  sleep "$POLL"
done
