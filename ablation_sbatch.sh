#!/usr/bin/env bash
# Direct sbatch: submit all remaining ablation experiments as one job on a certified node.
# Runs sequentially on the same node — no flock racing needed.
# Usage: bash ablation_sbatch.sh (submits and exits; monitor via squeue + log)
set -uo pipefail
ROOT="${AUTORESEARCH_ROOT:-/home/junyanch_google_com/autoresearch}"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
export HF_TOKEN="${HF_API_KEY:-}"
SGL_HOME="/home/junyanch_google_com/autoresearch/programs/sgl/v0.25_ablations/sgl_mech"
EVAL="$SGL_HOME/researcher/.claude/skills/evaluation-sop/scripts/eval.sh"
NAME="sgl_mech"
WS="/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/sgl_mech/researchers/sgl_mech"

# Certified nodes to exclude non-certified
EXCL=$(bash "$SGL_HOME/researcher/.claude/skills/submit-gpu-job/scripts/certified-nodes.sh" --exclude 2>/dev/null || true)
EXCL_FLAG=()
[ -n "$EXCL" ] && EXCL_FLAG=(--exclude="$EXCL")

LOGDIR="$WS/logs"
mkdir -p "$LOGDIR"

# Create the wrapper script that will run inside sbatch
WRAPPER="$WS/ablation_wrapper.sh"
cat > "$WRAPPER" << 'WRAPPER_EOF'
#!/usr/bin/env bash
set -uo pipefail
ROOT="${AUTORESEARCH_ROOT:-/home/junyanch_google_com/autoresearch}"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
export HF_TOKEN="${HF_API_KEY:-}"
SGL_HOME="/home/junyanch_google_com/autoresearch/programs/sgl/v0.25_ablations/sgl_mech"
EVAL="$SGL_HOME/researcher/.claude/skills/evaluation-sop/scripts/eval.sh"
NAME="sgl_mech"
WS="/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/sgl_mech/researchers/sgl_mech"

clean_env() {
    unset SGLANG_EVICTION_POLICY_OVERRIDE SGLANG_ENABLE_COST_AWARE_EVICTION \
          SGLANG_COST_AWARE_EVICT_THRESHOLD SGLANG_SCHEDULE_OVERRIDE \
          SGLANG_WRITE_ADMISSION_MIN_COST SGLANG_CONTINUOUS_COST_ALPHA \
          SGLANG_COST_FREQ_THRESHOLD SGLANG_FREQ_DECAY_RATE \
          SGLANG_SIZE_AWARE_THRESHOLD SGLANG_LOADBACK_MIN_COST \
          SGLANG_LOADBACK_VALUE_GATE SGLANG_WRITE_THROUGH_THRESHOLD \
          SGLANG_FREQ_BOOST_WEIGHT SGLANG_COST_AWARE_REUSE_MIN \
          SGLANG_COST_AWARE_COST_MODE SGLANG_COST_AWARE_EVICT_THRESHOLD2 2>/dev/null || true
}

run_one() {
    local ver="$1"; shift
    local sf="$WS/runs/$ver/summary.json"
    if [ -f "$sf" ]; then
        echo "========== $(date) ========== SKIP: $ver (has summary.json) =========="
        return 0
    fi
    echo "========== $(date) ========== START: $ver =========="
    env | grep -E "SGLANG_(EVICTION|ENABLE_COST|SCHEDULE|WRITE_ADMISSION|LOADBACK|WRITE_THROUGH|CONTINUOUS_COST|FREQ_BOOST|COST_AWARE|SIZE_AWARE|FREQ_DECAY)" || echo "(no overrides)"
    bash "$EVAL" "$NAME" "$ver" "$@"
    local rc=$?
    echo "========== $(date) ========== DONE: $ver rc=$rc =========="
    return $rc
}

echo "========== $(date) ========== ABLATION BATCH START on $(hostname) =========="

# v10-lfu
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=0; export SGLANG_EVICTION_POLICY_OVERRIDE=lfu
run_one v10-lfu || true

# v11-slru
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=0; export SGLANG_EVICTION_POLICY_OVERRIDE=slru
run_one v11-slru || true

# v12-costfreq
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=0; export SGLANG_EVICTION_POLICY_OVERRIDE=cost_freq; export SGLANG_COST_FREQ_THRESHOLD=2048
run_one v12-costfreq || true

# v13-contcost
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=0; export SGLANG_EVICTION_POLICY_OVERRIDE=continuous_cost; export SGLANG_CONTINUOUS_COST_ALPHA=1.0
run_one v13-contcost || true

# v14-writeadmit
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=1; export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048; export SGLANG_WRITE_ADMISSION_MIN_COST=2048
run_one v14-writeadmit || true

# v15-sjf
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=1; export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048; export SGLANG_SCHEDULE_OVERRIDE=sjf
run_one v15-sjf || true

# v16-warmfirst
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=1; export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048; export SGLANG_SCHEDULE_OVERRIDE=warmfirst
run_one v16-warmfirst || true

# v17-freqdecay
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=0; export SGLANG_EVICTION_POLICY_OVERRIDE=freq_decay; export SGLANG_FREQ_DECAY_RATE=0.999
run_one v17-freqdecay || true

# v18-sizelru
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=0; export SGLANG_EVICTION_POLICY_OVERRIDE=size_aware_lru; export SGLANG_SIZE_AWARE_THRESHOLD=2048
run_one v18-sizelru || true

# v19-loadback
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=1; export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048; export SGLANG_LOADBACK_MIN_COST=512
run_one v19-loadback || true

# v0-ctl3
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=0
run_one v0-ctl3 || true

# v21-valuegate
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=1; export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048; export SGLANG_LOADBACK_VALUE_GATE=1
run_one v21-valuegate || true

# v22-wt2
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=1; export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048; export SGLANG_WRITE_THROUGH_THRESHOLD=2
run_one v22-wt2 || true

# v23-wt3
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=1; export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048; export SGLANG_WRITE_THROUGH_THRESHOLD=3
run_one v23-wt3 || true

# v24-lru-valuegate
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=0; export SGLANG_LOADBACK_VALUE_GATE=1
run_one v24-lru-valuegate || true

# v25-fullstack
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=1; export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048; export SGLANG_WRITE_THROUGH_THRESHOLD=2; export SGLANG_LOADBACK_VALUE_GATE=1
run_one v25-fullstack || true

# v26-lru-wt2
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=0; export SGLANG_WRITE_THROUGH_THRESHOLD=2
run_one v26-lru-wt2 || true

# v27-backupcost
clean_env; export SGLANG_EVICTION_POLICY_OVERRIDE=backup_aware_cost; export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
run_one v27-backupcost || true

# v28-freqcost
clean_env; export SGLANG_EVICTION_POLICY_OVERRIDE=recency_boosted_cost; export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048; export SGLANG_FREQ_BOOST_WEIGHT=5.0
run_one v28-freqcost || true

# v29-contcost-a50
clean_env; export SGLANG_EVICTION_POLICY_OVERRIDE=continuous_cost; export SGLANG_CONTINUOUS_COST_ALPHA=50.0
run_one v29-contcost-a50 || true

# v30-contcost-a200
clean_env; export SGLANG_EVICTION_POLICY_OVERRIDE=continuous_cost; export SGLANG_CONTINUOUS_COST_ALPHA=200.0
run_one v30-contcost-a200 || true

# v31-reuse1
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=1; export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048; export SGLANG_COST_AWARE_REUSE_MIN=1
run_one v31-reuse1 || true

# v32-t4096
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=1; export SGLANG_COST_AWARE_EVICT_THRESHOLD=4096
run_one v32-t4096 || true

# v33-backupcost-wt2
clean_env; export SGLANG_EVICTION_POLICY_OVERRIDE=backup_aware_cost; export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048; export SGLANG_WRITE_THROUGH_THRESHOLD=2
run_one v33-backupcost-wt2 || true

# v34-3tier
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=1; export SGLANG_COST_AWARE_EVICT_THRESHOLD=1024; export SGLANG_COST_AWARE_EVICT_THRESHOLD2=4096
run_one v34-3tier || true

# v35-freqcost-w20
clean_env; export SGLANG_EVICTION_POLICY_OVERRIDE=recency_boosted_cost; export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048; export SGLANG_FREQ_BOOST_WEIGHT=20.0
run_one v35-freqcost-w20 || true

# v0-ctl4
clean_env; export SGLANG_ENABLE_COST_AWARE_EVICTION=0
run_one v0-ctl4 || true

clean_env
echo "========== $(date) ========== ALL EXPERIMENTS COMPLETE =========="
WRAPPER_EOF
chmod +x "$WRAPPER"

# Submit the batch job — 2-day time limit, exclusive certified node
JOB_LOG="$LOGDIR/ablation-batch-%j.out"
JID=$(sbatch --parsable \
    -p a3 \
    -N1 --exclusive --gres=gpu:8 \
    "${EXCL_FLAG[@]}" \
    -t 2-00:00:00 \
    -J "sgl_mech-ablation-batch" \
    -o "$JOB_LOG" \
    --wrap "bash $WRAPPER" \
    2>&1) || { echo "sbatch failed: $JID" >&2; exit 1; }

echo "Submitted job $JID"
echo "Log: ${JOB_LOG/\%j/$JID}"
echo "Monitor: squeue -j $JID; tail -f ${JOB_LOG/\%j/$JID}"
