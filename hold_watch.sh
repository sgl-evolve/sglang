#!/usr/bin/env bash
# Holder-watcher: wait for my own --exclusive certified hold job to land, disk/dram-gate it,
# run the frozen eval.sh into it, then RELEASE the hold (good citizen — never squat).
# This is the guaranteed race-free path (skill submit-gpu-job: self-lock a certified node for
# back-to-back evals) — superior to the unwinnable pool flock micro-race under a manager.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a
SGL_HOME="${SGL_HOME:-$ROOT/programs/sgl}"
EVAL="$SGL_HOME/researcher/.claude/skills/evaluation-sop/scripts/eval.sh"
NAME=kv-heron-eb9
HOLDJID="$1"; VER="$2"; shift 2
RELEASE_AFTER="${RELEASE_AFTER:-1}"   # 1 = scancel hold after eval; 0 = keep for back-to-back

echo "[watch] waiting for hold $HOLDJID to land (VER=$VER)…"
while :; do
  st=$(squeue -h -j "$HOLDJID" -o '%T' 2>/dev/null)
  [ "$st" = RUNNING ] && break
  if [ -z "$st" ]; then echo "[watch] hold $HOLDJID gone before landing — exit"; exit 1; fi
  sleep 15
done
node=$(squeue -h -j "$HOLDJID" -o '%N' 2>/dev/null | tr -d ' ')
echo "[watch] hold $HOLDJID landed on $node — probing disk/dram"
read -r free memg < <(timeout 60 srun --jobid="$HOLDJID" --overlap -N1 -w "$node" bash -c \
  "d=\$(df --output=avail -BG /mnt/localssd | tail -1 | tr -dc 0-9); m=\$(awk '/MemAvailable/{print int(\$2/1024/1024)}' /proc/meminfo); echo \$d \$m" 2>/dev/null)
echo "[watch] $node: ${free:-?}G disk / ${memg:-?}G RAM avail"

# If disk short, clean only my own L3 dir once, then re-probe.
if [ -z "${free:-}" ] || [ "${free:-0}" -lt 1800 ]; then
  echo "[watch] disk<1800G — cleaning my own /mnt/localssd/$NAME and re-probing"
  srun --jobid="$HOLDJID" --overlap -N1 -w "$node" bash -c "rm -rf /mnt/localssd/$NAME/* 2>/dev/null; true" 2>/dev/null
  read -r free memg < <(timeout 60 srun --jobid="$HOLDJID" --overlap -N1 -w "$node" bash -c \
    "d=\$(df --output=avail -BG /mnt/localssd | tail -1 | tr -dc 0-9); m=\$(awk '/MemAvailable/{print int(\$2/1024/1024)}' /proc/meminfo); echo \$d \$m" 2>/dev/null)
  echo "[watch] after clean: ${free:-?}G disk / ${memg:-?}G RAM"
fi

if [ -n "${free:-}" ] && [ "${free:-0}" -ge 1800 ] && [ -n "${memg:-}" ] && [ "${memg:-0}" -ge 1300 ]; then
  echo "[watch] running eval $VER on $node (hold $HOLDJID)"
  srun --jobid="$HOLDJID" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" "$NAME" "$VER" "$@"
  echo "[watch] eval rc=$?"
else
  echo "[watch] $node failed disk/dram gate (need >=1800/>=1300) — NOT running (would be non-comparable)"
fi

if [ "$RELEASE_AFTER" = 1 ]; then
  scancel "$HOLDJID" && echo "[watch] released hold $HOLDJID"
fi
