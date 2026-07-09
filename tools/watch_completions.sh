#!/usr/bin/env bash
# Continuously watch for completed evals and log them to W&B.
# Run in background: nohup bash tools/watch_completions.sh &

set -euo pipefail
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"

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

LOGFILE="$WS/tools/watch_completions.log"
LOGGED_FILE="$WS/tools/.logged_versions"
touch "$LOGGED_FILE"

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%SZ')] $*" | tee -a "$LOGFILE"; }

log "Starting completion watcher (${#VERSIONS[@]} versions)"

while true; do
  all_done=true
  for entry in "${VERSIONS[@]}"; do
    ver="${entry%%:*}"
    kind="${entry##*:}"
    
    # Skip already logged
    grep -qx "$ver" "$LOGGED_FILE" 2>/dev/null && continue
    
    all_done=false
    summary="$WS/runs/$ver/summary.json"
    if [ -f "$summary" ]; then
      log "FOUND: $ver ($kind) — logging to W&B"
      
      # Load secrets
      set -a; source /home/junyanch_google_com/autoresearch/.env; set +a
      export HF_TOKEN="$HF_API_KEY"
      export SGL_HOME="/home/junyanch_google_com/autoresearch/programs/sgl/v0.25_ablations/sgl_free/researcher"
      export SGL_WORKSPACE="$WS"
      export SGLFREE_RUN_ID=sgl_free-v025
      
      # Extract key metrics for log
      hit=$(.venv/bin/python -c "import json; d=json.load(open('$summary')); print(d.get('mix',{}).get('hit_rate','?'))")
      p99=$(.venv/bin/python -c "import json; d=json.load(open('$summary')); print(d.get('mix',{}).get('ttft_p99_ms','?'))")
      log "  hit=$hit p99=$p99"
      
      # Log to W&B
      commit=$(cd "$SGL_HOME" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
      .venv/bin/python tools/log_wandb_sglfree.py "$ver" "$summary" "$ver" "$commit" "$kind" 2>&1 | tail -3 | while read -r line; do log "  wandb: $line"; done
      
      echo "$ver" >> "$LOGGED_FILE"
      log "LOGGED: $ver"
    fi
  done
  
  if $all_done; then
    log "ALL VERSIONS LOGGED — watcher exiting"
    break
  fi
  
  sleep 120
done
