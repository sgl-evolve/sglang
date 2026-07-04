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

probe(){ timeout 60 srun --jobid="$HOLDJID" --overlap -N1 -w "$node" bash -c \
  'd=$(df --output=avail -BG /mnt/localssd | tail -1 | tr -dc 0-9); m=$(awk "/MemAvailable/{print int(\$2/1024/1024)}" /proc/meminfo); echo $d $m' 2>/dev/null; }

runeval(){  # $@ = version + eval.sh args
  local ver="$1"
  echo "[batch2] --- prep $ver: wiping my L3 /mnt/localssd/$NAME ---"
  srun --jobid="$HOLDJID" --overlap -N1 -w "$node" bash -c "rm -rf /mnt/localssd/$NAME/* 2>/dev/null; true" 2>/dev/null
  read -r free memg < <(probe)
  echo "[batch2] $ver: disk=${free:-?}G ram=${memg:-?}G"
  if [ -n "${free:-}" ] && [ "${free:-0}" -ge 1800 ] && [ -n "${memg:-}" ] && [ "${memg:-0}" -ge 1300 ]; then
    echo "[batch2] === running $ver on $node ==="
    srun --jobid="$HOLDJID" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" "$NAME" "$@"
    echo "[batch2] $ver rc=$?"
  else
    echo "[batch2] $ver SKIP: disk/dram gate failed"
  fi
}

runeval v14-be-cons0.5 --enforce-disable-flashinfer-allreduce-fusion --hicache-storage-prefetch-policy best_effort --schedule-conservativeness 0.5
runeval v15-be-mixchunk --enforce-disable-flashinfer-allreduce-fusion --hicache-storage-prefetch-policy best_effort --enable-mixed-chunk

scancel "$HOLDJID" && echo "[batch2] released hold $HOLDJID"
echo "[batch2] DONE"
