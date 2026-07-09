#!/usr/bin/env bash
# eval-on-pool.sh — run the FIXED evaluator on a CERTIFIED node.
# Prefers the manager's held eval pool (guaranteed capacity, no queue): grabs a free held node behind a
# flock (so two researchers never collide) and srun's the eval into it. If NO nodes are held, falls back
# to a queued certified sbatch. Blocks for the whole eval, so run it in the BACKGROUND and tail the log:
#   nohup bash .../eval-on-pool.sh <name> <version> [eval args...] > eval.log 2>&1 &
set -uo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AUTORESEARCH_ROOT:-/home/junyanch_google_com/autoresearch}"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
# Hardcoded paths since this script lives in workspace/tools, not the skill tree
SGL_HOME="/home/junyanch_google_com/autoresearch/programs/sgl/v0.25_ablations/sgl_free"
RESEARCHER="$SGL_HOME/researcher"
EVAL="$RESEARCHER/.claude/skills/evaluation-sop/scripts/eval.sh"
RT="${SGL_RUNTIME:-$SGL_HOME/manager/.runtime}"
POOL="${SGL_POOL_DIR:-$ROOT/workspace/sgl/v0.25_ablations/_pool}"
PART="${SGL_PARTITION:-a3}"

NAME="${1:?usage: eval-on-pool.sh <name> <version> [eval args...]}"
VER="${2:?usage: eval-on-pool.sh <name> <version> [eval args...]}"; shift 2
[ -x "$EVAL" ] || [ -f "$EVAL" ] || { echo "eval-on-pool: evaluator not found at $EVAL" >&2; exit 1; }

held_nodes(){ [ -n "${RT:-}" ] && compgen -G "$POOL/held/*" >/dev/null 2>&1 && for f in "$POOL/held"/*; do basename "$f"; done; }

run_fallback(){
  local excl; excl=$(bash "$SELF/certified-nodes.sh" --exclude 2>/dev/null || true)
  local ex=(); [[ -n "$excl" ]] && ex=(--exclude="$excl")
  local log="${RT:-/tmp}/logs/eval-$NAME-$VER-%j.out"; mkdir -p "$(dirname "$log")" 2>/dev/null || log="/tmp/eval-$NAME-$VER-%j.out"
  local extra=""; [[ $# -gt 0 ]] && extra=" $(printf '%q ' "$@")"
  echo "[eval-on-pool] no held pool — sbatch onto a free CERTIFIED node (queued)"
  local jid
  jid=$(sbatch --parsable -p "$PART" -N1 --exclusive --gres=gpu:8 "${ex[@]}" -t 6:00:00 \
        -J "eval-$NAME-$VER" -o "$log" \
        --wrap "bash $(printf '%q' "$EVAL") $(printf '%q' "$NAME") $(printf '%q' "$VER")$extra") \
    || { echo "[eval-on-pool] sbatch failed" >&2; return 1; }
  echo "[eval-on-pool] submitted job $jid  (log: ${log/\%j/$jid})"
  echo "[eval-on-pool] watch: squeue -j $jid ; tail -f ${log/\%j/$jid}"
}

# No held pool at all -> queued certified sbatch.
if [[ -z "$(held_nodes)" ]]; then run_fallback "$@"; exit $?; fi

# Held pool exists: acquire a free live held node (blocking retry), srun the eval into it.
while :; do
  any_alive=0
  for node in $(held_nodes); do
    jid=$(cat "$POOL/held/$node" 2>/dev/null) || continue
    squeue -h -j "$jid" >/dev/null 2>&1 || continue          # hold job dead -> skip (manager re-heals)
    any_alive=1
    exec 200>"$POOL/locks/$node.lock"
    if flock -n 200; then               # acquired this node's lock (separate from eval's exit code)
      echo "[eval-on-pool] running eval on held node $node (hold job $jid)"
      srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" "$NAME" "$VER" "$@"
      rc=$?
      flock -u 200; exec 200>&-
      exit $rc                          # real eval exit code (never confused with 'busy')
    fi
    exec 200>&-                         # couldn't lock -> another researcher has it, try next node
  done
  if [[ $any_alive -eq 0 ]]; then
    echo "[eval-on-pool] no live held nodes — falling back to queued certified sbatch"
    run_fallback "$@"; exit $?
  fi
  echo "[eval-on-pool] all held nodes busy — waiting for one to free..."; sleep 5
done
