#!/usr/bin/env bash
# Run a SEQUENCE of versions back-to-back on my OWN held node (0-2, job in .myhold_jobid).
# Each: checkout its code into python/, spawn a login-side post_eval watcher (auto
# audit/log/email), and srun the FIXED evaluator (with the fusion-disable fix) into the hold.
# Keeps the hold up after (add more versions to SEQ and re-run, or scancel when done).
set -uo pipefail
NAME=quill-7m3
ROOT=/home/junyanch_google_com/autoresearch
EVAL=$ROOT/programs/sgl/researcher/.claude/skills/evaluation-sop/scripts/eval.sh
WORK=$ROOT/workspace/sgl/researchers/$NAME
POST=$WORK/post_eval.sh
FUSION="--enforce-disable-flashinfer-allreduce-fusion"
V1REF=2dda8247a
JID=$(cat "$WORK/.myhold_jobid")
NODE=slurm2-a3nodeset0-2
# version|gitref|tag|extra eval args   (pass SEQ via env SEQSPEC newline-separated to override)
DEFAULT_SEQ="be-writeback|$V1REF|config|--hicache-storage-prefetch-policy best_effort --hicache-write-policy write_back
v4-tunedto|5014a841a|mechanism|--hicache-storage-prefetch-policy timeout
be-wtsel|$V1REF|config|--hicache-storage-prefetch-policy best_effort --hicache-write-policy write_through_selective
v1d-timeout|$V1REF|config|--hicache-storage-prefetch-policy timeout"
SEQSPEC="${SEQSPEC:-$DEFAULT_SEQ}"

while IFS= read -r spec; do
  [ -z "$spec" ] && continue
  IFS='|' read -r ver ref tag args <<< "$spec"
  echo "[myhold] === $ver (ref=$ref tag=$tag args='$args') @ $(date +%H:%M:%S) ==="
  git -C "$WORK" checkout "$ref" -- python/ 2>&1 | tail -1 || { echo "[myhold] checkout $ref FAILED, skip $ver"; continue; }
  rm -f "$WORK/runs/$ver/summary.json" 2>/dev/null
  nohup bash "$POST" "$ver" "$ref" "$tag" > "$WORK/post_${ver}.nohup" 2>&1 & disown
  srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 bash "$EVAL" "$NAME" "$ver" $FUSION $args < /dev/null
  if [ -f "$WORK/runs/$ver/summary.json" ]; then echo "[myhold] $ver OK @ $(date +%H:%M:%S)"; else echo "[myhold] $ver NO SUMMARY @ $(date +%H:%M:%S)"; fi
done <<< "$SEQSPEC"
git -C "$WORK" checkout "$V1REF" -- python/ 2>&1 | tail -1
echo "[myhold] SEQUENCE DONE @ $(date +%H:%M:%S). Hold $JID still up."
