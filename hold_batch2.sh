#!/usr/bin/env bash
# Round-2 batch: prefill-scheduling config probes on the v6 base (best_effort + concurrent-IO).
# v14: --schedule-conservativeness 0.5 (more aggressive prefill admission -> less queue-wait).
# v15: --enable-mixed-chunk (mix prefill+decode in a batch -> better overlap).
# Both compression OFF, single-lever vs v6. Wipe L3 between; each eval independent; release hold after.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a
SGL_HOME="${SGL_HOME:-$ROOT/programs/sgl}"
EVAL="$SGL_HOME/researcher/.claude/skills/evaluation-sop/scripts/eval.sh"
NAME=kv-heron-eb9
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
# (NCCL init then OOMs). Check max GPU mem used; if occupied, record the bad node + release + exit
# so the re-queue can --exclude it (avoids wasting eval preflights on a wedged node).
maxmem=$(timeout 60 srun --jobid="$HOLDJID" --overlap -N1 -w "$node" bash -c \
  'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1' 2>/dev/null | tr -dc 0-9)
echo "[batch2] $node max GPU mem used = ${maxmem:-?} MiB"
if [ -z "${maxmem:-}" ] || [ "${maxmem:-99999}" -gt 2000 ]; then
  echo "[batch2] WEDGED node $node (GPU occupied) — recording + releasing, re-queue must --exclude it"
  echo "$node" >> /home/junyanch_google_com/autoresearch/workspace/sgl/researchers/kv-heron-eb9/.bad_nodes
  scancel "$HOLDJID"; echo "[batch2] released $HOLDJID (wedged)"; exit 2
fi

probe(){ timeout 60 srun --jobid="$HOLDJID" --overlap -N1 -w "$node" bash -c \
  'd=$(df --output=avail -BG /mnt/localssd | tail -1 | tr -dc 0-9); m=$(awk "/MemAvailable/{print int(\$2/1024/1024)}" /proc/meminfo); echo $d $m' 2>/dev/null; }

runeval(){  # $@ = version + eval.sh args ; returns eval rc (6 = NCCL/GPU preflight fail = bad node)
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

# A wedged GPU (dead context) shows 0 MiB used but OOMs at NCCL init -> eval.sh returns rc=6.
# On the first NCCL fail, record the node as bad + release so the re-queue --excludes it (one
# wasted preflight, not two). Only run v15 if v14's node proved healthy.
runeval v14-be-cons0.5 --enforce-disable-flashinfer-allreduce-fusion --hicache-storage-prefetch-policy best_effort --schedule-conservativeness 0.5
rc14=$?
if [ "$rc14" = 6 ]; then
  echo "$node" >> /home/junyanch_google_com/autoresearch/workspace/sgl/researchers/kv-heron-eb9/.bad_nodes
  echo "[batch2] NCCL/GPU preflight FAIL on $node (rc6) — recorded bad, skipping rest"
  scancel "$HOLDJID"; echo "[batch2] released $HOLDJID (bad node)"; exit 2
fi

runeval v15-be-mixchunk --enforce-disable-flashinfer-allreduce-fusion --hicache-storage-prefetch-policy best_effort --enable-mixed-chunk

scancel "$HOLDJID" && echo "[batch2] released hold $HOLDJID"
echo "[batch2] DONE"
