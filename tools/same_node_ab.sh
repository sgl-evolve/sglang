#!/usr/bin/env bash
# same_node_ab.sh — definitive A/B: run baseline (fcfs) then exclusive on the SAME held node, back-to-back,
# holding the flock across BOTH evals, so the TTFT delta has NO node-to-node variance (the p99 confound).
# Both at the fixed lambda=3 protocol. Autonomous: retries nodes until one runs a clean baseline, then
# immediately runs exclusive on it. Run in background.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
POOL="$ROOT/workspace/sgl/v0.25_ablations/_pool"
SGL_HOME="$ROOT/programs/sgl/v0.25_ablations/sgl_free"
EVAL="$SGL_HOME/researcher/.claude/skills/evaluation-sop/scripts/eval.sh"
WS="$ROOT/workspace/sgl/v0.25_ablations/sgl_free/researchers/sgl_free"
SUF="${SUFFIX:-}"; BASE_VER="v_ab${SUF}_baseline"; EXC_VER="v_ab${SUF}_exclusive"
held_nodes(){ for f in "$POOL"/held/*; do [ -e "$f" ] && basename "$f"; done; }
echo "[ab] same-node A/B: $BASE_VER (fcfs) then $EXC_VER (exclusive) on ONE node"
for try in $(seq 1 200); do
  [ -f "$WS/runs/$EXC_VER/summary.json" ] && { echo "[ab] already done"; break; }
  for node in $(held_nodes); do
    jid=$(cat "$POOL/held/$node" 2>/dev/null) || continue
    squeue -h -j "$jid" >/dev/null 2>&1 || continue
    exec 200>"$POOL/locks/$node.lock"
    if flock -n 200; then
      echo "[ab] try $try got $node (job $jid) $(date -u +%H:%M:%S)"
      # --- eval 1: baseline (fcfs, stock) ---
      unset SGLANG_HICACHE_EXCLUSIVE
      srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" sgl_free "$BASE_VER"
      if [ ! -f "$WS/runs/$BASE_VER/summary.json" ]; then
        echo "[ab] baseline failed on $node (likely DRAM); release + try another"
        flock -u 200; exec 200>&-; break
      fi
      # --- eval 2: exclusive (self-contained, SAME node, flock still held) ---
      export SGLANG_HICACHE_EXCLUSIVE=1
      srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" sgl_free "$EXC_VER"
      flock -u 200; exec 200>&-
      [ -f "$WS/runs/$EXC_VER/summary.json" ] && { echo "[ab] SUCCESS both on $node"; break 2; }
      echo "[ab] exclusive leg failed on $node; will retry"; break
    fi
    exec 200>&-
  done
  sleep 45
done
echo "[ab] done. results:"
for v in "$BASE_VER" "$EXC_VER"; do
  python3 - "$WS/runs/$v/summary.json" "$v" <<'PY' 2>/dev/null || echo "  $v: (no summary)"
import json,sys
p=json.load(open(sys.argv[1]))['panel']
print(f"  {sys.argv[2]}: hit={p['overall/hit_rate']} p99={p['overall/ttft_p99_ms']} mean={p['overall/ttft_mean_ms']} req_s={p['overall/req_throughput']}")
PY
done
