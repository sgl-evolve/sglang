#!/usr/bin/env bash
# Batch holder-watcher: wait for my --exclusive certified hold to land, then run a SEQUENCE of
# evals back-to-back on it (max value from a scarce, hard-won node), then RELEASE the hold.
# Each eval: wipe my own L3 first (cold cache + so a prior version's on-disk format can't poison
# the next — v12 writes zlib-compressed pages, v13 reads with compression OFF), disk/dram-gate,
# run frozen eval.sh. Evals are independent: one failing does not stop the rest.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a
SGL_HOME="${SGL_HOME:-$ROOT/programs/sgl}"
EVAL="$SGL_HOME/researcher/.claude/skills/evaluation-sop/scripts/eval.sh"
NAME=kv-heron-eb9
HOLDJID="$1"

echo "[batch] waiting for hold $HOLDJID to land…"
while :; do
  st=$(squeue -h -j "$HOLDJID" -o '%T' 2>/dev/null)
  [ "$st" = RUNNING ] && break
  if [ -z "$st" ]; then echo "[batch] hold $HOLDJID gone before landing — exit"; exit 1; fi
  sleep 15
done
node=$(squeue -h -j "$HOLDJID" -o '%N' 2>/dev/null | tr -d ' ')
echo "[batch] hold $HOLDJID landed on $node"

probe(){ timeout 60 srun --jobid="$HOLDJID" --overlap -N1 -w "$node" bash -c \
  'd=$(df --output=avail -BG /mnt/localssd | tail -1 | tr -dc 0-9); m=$(awk "/MemAvailable/{print int(\$2/1024/1024)}" /proc/meminfo); echo $d $m' 2>/dev/null; }

runeval(){  # $@ = version + eval.sh args ; env (compression) set by caller before the call
  local ver="$1"
  echo "[batch] --- prep $ver: wiping my L3 /mnt/localssd/$NAME ---"
  srun --jobid="$HOLDJID" --overlap -N1 -w "$node" bash -c "rm -rf /mnt/localssd/$NAME/* 2>/dev/null; true" 2>/dev/null
  read -r free memg < <(probe)
  echo "[batch] $ver: disk=${free:-?}G ram=${memg:-?}G  (COMPRESS=${SGLANG_HICACHE_FILE_COMPRESS:-0})"
  if [ -n "${free:-}" ] && [ "${free:-0}" -ge 1800 ] && [ -n "${memg:-}" ] && [ "${memg:-0}" -ge 1300 ]; then
    echo "[batch] === running $ver on $node ==="
    srun --jobid="$HOLDJID" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" "$NAME" "$@"
    echo "[batch] $ver rc=$?"
  else
    echo "[batch] $ver SKIP: disk/dram gate failed (need >=1800/>=1300)"
  fi
}

# --- v12: transparent zlib disk-tier compression (mechanism), best_effort + concurrent-IO ---
export SGLANG_HICACHE_FILE_COMPRESS=1 SGLANG_HICACHE_FILE_COMPRESS_LEVEL=1
runeval v12-be-zlibL1 --enforce-disable-flashinfer-allreduce-fusion --hicache-storage-prefetch-policy best_effort
unset SGLANG_HICACHE_FILE_COMPRESS SGLANG_HICACHE_FILE_COMPRESS_LEVEL

# --- v13: reallocate frozen GPU budget mamba->device KV (device retention, +43% KV, config) ---
runeval v13-be-mamba700 --enforce-disable-flashinfer-allreduce-fusion --hicache-storage-prefetch-policy best_effort --max-mamba-cache-size 700

scancel "$HOLDJID" && echo "[batch] released hold $HOLDJID"
echo "[batch] DONE"
