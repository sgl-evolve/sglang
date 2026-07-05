#!/usr/bin/env bash
# Round-2 batch on the v6 base (best_effort + concurrent-IO). Runs a PRIORITY-ORDERED list of
# evals back-to-back on one hard-won certified node, then releases the hold. Scarce eval slots
# go to the highest-EV experiment first:
#   v16 = --schedule-policy spf : NEW aged shortest-prefill-first scheduler (mechanism). SJF is
#         optimal for mean flow time = the eval headline (mean TTFT); cuts the long-context
#         prefill tail. Lossless (reorder-only, starvation-bounded).
#   v14 = --schedule-conservativeness 0.5 : more aggressive prefill admission (config).
#   v15 = --enable-mixed-chunk : mix prefill+decode in a batch for better overlap (config).
# All compression OFF, single-lever vs v6. Wipe L3 between; each eval independent.
# A node-level failure (rc6 NCCL/GPU wedge, rc9 disk/dram gate) aborts the rest, records the node
# bad, and releases the hold so the re-queue can --exclude it.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a
SGL_HOME="${SGL_HOME:-$ROOT/programs/sgl}"
EVAL="$SGL_HOME/researcher/.claude/skills/evaluation-sop/scripts/eval.sh"
NAME=kv-heron-eb9
BAD=/home/junyanch_google_com/autoresearch/workspace/sgl/researchers/$NAME/.bad_nodes
HOLDJID="$1"

echo "[batch2] waiting for hold $HOLDJID to land…"
while :; do
  st=$(squeue -h -j "$HOLDJID" -o '%T' 2>/dev/null)
  [ "$st" = RUNNING ] && break
  if [ -z "$st" ]; then echo "[batch2] hold $HOLDJID gone before landing — exit"; exit 1; fi
  sleep 15
done
node=$(squeue -h -j "$HOLDJID" -o '%N' 2>/dev/null | tr -d ' ')
echo "[batch2] hold $HOLDJID landed on $node"

# GPU-health gate: a node freed from the manager pool can carry foreign leftover GPU state
# (NCCL init then OOMs). If GPU is occupied, record bad + release + exit so re-queue --excludes it.
maxmem=$(timeout 60 srun --jobid="$HOLDJID" --overlap -N1 -w "$node" bash -c \
  'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1' 2>/dev/null | tr -dc 0-9)
echo "[batch2] $node max GPU mem used = ${maxmem:-?} MiB"
if [ -z "${maxmem:-}" ] || [ "${maxmem:-99999}" -gt 2000 ]; then
  echo "[batch2] WEDGED node $node (GPU occupied) — recording + releasing"
  echo "$node" >> "$BAD"; scancel "$HOLDJID"; echo "[batch2] released $HOLDJID (wedged)"; exit 2
fi

probe(){ timeout 60 srun --jobid="$HOLDJID" --overlap -N1 -w "$node" bash -c \
  'd=$(df --output=avail -BG /mnt/localssd | tail -1 | tr -dc 0-9); m=$(awk "/MemAvailable/{print int(\$2/1024/1024)}" /proc/meminfo); echo $d $m' 2>/dev/null; }

runeval(){  # $@ = version + eval.sh args ; returns eval rc (6 = NCCL preflight, 9 = disk/dram gate)
  local ver="$1"
  echo "[batch2] --- prep $ver: wiping my L3 /mnt/localssd/$NAME ---"
  srun --jobid="$HOLDJID" --overlap -N1 -w "$node" bash -c "rm -rf /mnt/localssd/$NAME/* 2>/dev/null; true" 2>/dev/null
  read -r free memg < <(probe)
  echo "[batch2] $ver: disk=${free:-?}G ram=${memg:-?}G"
  if [ -n "${free:-}" ] && [ "${free:-0}" -ge 1800 ] && [ -n "${memg:-}" ] && [ "${memg:-0}" -ge 1300 ]; then
    echo "[batch2] === running $ver on $node ==="
    srun --jobid="$HOLDJID" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" "$NAME" "$@"
    local rc=$?; echo "[batch2] $ver rc=$rc"; return $rc
  else
    echo "[batch2] $ver SKIP: disk/dram gate failed"; return 9
  fi
}

# Priority-ordered list: (version, eval-args...). Highest-EV first.
run_one(){  # $@ = version + args ; on node-level fail (rc6/rc9) record bad node, release, exit 2
  runeval "$@"; local rc=$?
  if [ "$rc" = 6 ] || [ "$rc" = 9 ]; then
    echo "$node" >> "$BAD"
    echo "[batch2] node-level FAIL on $node (rc$rc) — recorded bad, aborting rest"
    scancel "$HOLDJID"; echo "[batch2] released $HOLDJID (bad node)"; exit 2
  fi
}

run_one v16-be-spf   --enforce-disable-flashinfer-allreduce-fusion --hicache-storage-prefetch-policy best_effort --schedule-policy spf
run_one v14-be-cons0.5 --enforce-disable-flashinfer-allreduce-fusion --hicache-storage-prefetch-policy best_effort --schedule-conservativeness 0.5
run_one v15-be-mixchunk --enforce-disable-flashinfer-allreduce-fusion --hicache-storage-prefetch-policy best_effort --enable-mixed-chunk

scancel "$HOLDJID" && echo "[batch2] released hold $HOLDJID"
echo "[batch2] DONE"
