#!/usr/bin/env bash
# Run the fixed evaluator on MY self-held certified node via srun --overlap, retrying on a
# transient load hang/timeout (rc not in {0,8}). Usage: run_on_mynode.sh <name> <version> [args...]
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a; export HF_TOKEN="$HF_API_KEY"
EVAL="$ROOT/programs/sgl/researcher/.claude/skills/evaluation-sop/scripts/eval.sh"
JID="${HOLD_JID:-18136}"; NODE="${HOLD_NODE:-slurm2-a3nodeset0-2}"
NAME="${1:?name}"; VER="${2:?version}"; shift 2
MAXTRY="${MAXTRY:-3}"
for try in $(seq 1 "$MAXTRY"); do
  echo "[mynode] eval $VER attempt $try/$MAXTRY on $NODE $(date '+%H:%M:%S')"
  srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 bash "$EVAL" "$NAME" "$VER" "$@"
  rc=$?
  if [ "$rc" -eq 0 ] || [ "$rc" -eq 8 ]; then
    echo "[mynode] eval $VER exit $rc (final) $(date '+%H:%M:%S')"; exit $rc
  fi
  echo "[mynode] eval $VER attempt $try failed rc=$rc; retrying $(date '+%H:%M:%S')"
  # fresh run dir between attempts so monitoring isn't confused by a stuck server.log
  mv "$ROOT/workspace/sgl/researchers/$NAME/runs/$VER" \
     "$ROOT/workspace/sgl/researchers/$NAME/runs/$VER-failed-$try" 2>/dev/null || true
  sleep 5
done
echo "[mynode] gave up after $MAXTRY attempts"; exit 1
