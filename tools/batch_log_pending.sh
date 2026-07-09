#!/usr/bin/env bash
# Batch-check and log pending eval results. Run from workspace root.
# Usage: bash tools/batch_log_pending.sh
set -euo pipefail

WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"

set -a; source /home/junyanch_google_com/autoresearch/.env; set +a
export HF_TOKEN="$HF_API_KEY"
export SGL_HOME=/home/junyanch_google_com/autoresearch/programs/sgl/v0.25_ablations/sgl_free/researcher
export SGL_WORKSPACE=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/sgl_free
export SGLFREE_RUN_ID=sgl_free-v025

COMMIT="$(git rev-parse --short HEAD)"
LOGGED=0

VERSIONS=(
  "v_exclusive_rep3:excl_replicate"
  "v_random_excl:excl_alt_evict"
  "v_size_lru_excl:excl_alt_evict"
  "v_fifo_excl:excl_alt_evict"
  "v_mru_excl:excl_alt_evict"
  "v_filo_excl:excl_alt_evict"
  "v_gdsf_excl:excl_alt_evict"
  "v_2q_excl:excl_alt_evict"
  "v_sjf_excl:excl_sched"
  "v_discard128_excl:excl_discard"
  "v_discard512_excl:excl_discard"
  "v_random_wb:wb_alt_evict"
  "v_random_base:base_alt_evict"
)

for entry in "${VERSIONS[@]}"; do
  name="${entry%%:*}"
  kind="${entry##*:}"
  summary="runs/${name}/summary.json"
  if [ -f "$summary" ]; then
    echo "[batch_log] Found $summary — logging to W&B..."
    .venv/bin/python tools/log_wandb_sglfree.py "$name" "$summary" "$name" "$COMMIT" "$kind" || true
    LOGGED=$((LOGGED + 1))
  fi
done

echo "[batch_log] Done. Logged $LOGGED / ${#VERSIONS[@]} versions."
